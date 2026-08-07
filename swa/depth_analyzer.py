# Authorship: ALifer Ramos & Antigravity (AI Co-Architect)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from threading import Thread
import time
import os


@dataclass
class VisionObstacle:
    """Detected obstacle from vision analysis."""
    direction: str  # 'front', 'left', 'right', etc.
    confidence: float  # 0.0 to 1.0
    depth: float  # estimated distance
    plane_type: str  # 'vertical' (wall), 'horizontal' (floor/ceiling), etc.


@dataclass
class RuntimeEvent:
    """Event for runtime broadcasting."""
    type: str
    payload: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SpatialSemanticAnalyzer(nn.Module):
    """
    Spatial Semantic Analyzer for SWA Phase 4: Semantic Spatial Reasoning.
    Converts vision_tower semantic embeddings into obstacle likelihood estimation.
    Implements closed-loop vision -> event -> bias fusion for multimodal constrained decoding.
    """

    def __init__(self, vision_hidden_size: int = 1024, spatial_resolution: Tuple[int, int] = (224, 224),
                 device: str = "cpu", dtype: torch.dtype = torch.float32):
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.spatial_resolution = spatial_resolution
        self.device = device
        self.dtype = dtype

        # Obstacle likelihood head: estimates semantic obstacle probability from center patch
        self.obstacle_likelihood_head = nn.Sequential(
            nn.Linear(vision_hidden_size, vision_hidden_size // 2),
            nn.GELU(),
            nn.Linear(vision_hidden_size // 2, 1),
            nn.Sigmoid()  # Output in [0, 1] range - semantic obstacle likelihood
        )

        # Vertical plane detector: focuses on wall-like structures
        self.vertical_plane_detector = nn.Sequential(
            nn.Conv2d(vision_hidden_size, vision_hidden_size // 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(vision_hidden_size // 4, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial proximity tensor reconstruction
        self.spatial_reconstructor = nn.Sequential(
            nn.Linear(vision_hidden_size, vision_hidden_size // 2),
            nn.ReLU(),
            nn.Linear(vision_hidden_size // 2, spatial_resolution[0] * spatial_resolution[1]),
        )

        # Confidence threshold for obstacle detection (semantic obstacle likelihood)
        self.collision_threshold = 0.85  # Higher threshold to avoid false positives

        # Move to device
        self.to(device, dtype)

    def forward(self, vision_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process vision embeddings to extract semantic obstacle likelihood and spatial information.

        Args:
            vision_embeddings: Raw semantic embeddings from vision_tower (batch, seq_len, hidden_size)

        Returns:
            Tuple of (obstacle_likelihood_front, vertical_semantic_planes, spatial_navigability)
        """
        batch_size = vision_embeddings.shape[0]

        # Extract center patch for frontal obstacle likelihood estimation
        # Assuming vision_embeddings are patch embeddings in sequence
        center_idx = vision_embeddings.shape[1] // 2  # Middle patch
        center_patch = vision_embeddings[:, center_idx, :]  # (batch, hidden_size)

        # Front obstacle likelihood estimation: L_front = σ(W_obstacle · v_vision^center + b)
        # This represents semantic likelihood of frontal blocking/navigation obstruction
        obstacle_likelihood_front = self.obstacle_likelihood_head(center_patch)  # (batch, 1)

        # Vertical semantic plane detection across all patches
        # Detects semantic patterns indicative of wall-like/blocking structures
        h, w = self.spatial_resolution
        if vision_embeddings.shape[1] == h * w:  # Assuming patches match spatial resolution
            spatial_grid = vision_embeddings.view(batch_size, h, w, -1).permute(0, 3, 1, 2)  # (batch, hidden, h, w)
            vertical_semantic_planes = self.vertical_plane_detector(spatial_grid)  # (batch, 1, h, w)
            vertical_semantic_planes = vertical_semantic_planes.view(batch_size, -1)  # Flatten
        else:
            # Fallback: treat as sequence and apply temporal convolution
            vertical_semantic_planes = self.vertical_plane_detector(
                vision_embeddings.permute(0, 2, 1).unsqueeze(-1)
            ).squeeze(-1).permute(0, 2, 1).mean(dim=-1)  # (batch, seq_len)

        # Spatial navigability tensor reconstruction
        # Estimates semantic navigability across spatial regions
        spatial_navigability = self.spatial_reconstructor(vision_embeddings.mean(dim=1))  # (batch, h*w)
        spatial_navigability = spatial_navigability.view(batch_size, h, w)  # Reshape to spatial grid

        return obstacle_likelihood_front, vertical_semantic_planes, spatial_navigability

    def detect_obstacles(self, obstacle_likelihood_front: torch.Tensor, vertical_semantic_planes: torch.Tensor,
                        spatial_navigability: torch.Tensor) -> list[VisionObstacle]:
        """
        Analyze semantic obstacle likelihood and spatial navigability to detect navigation constraints.

        Args:
            obstacle_likelihood_front: Frontal obstacle likelihood estimation (batch, 1)
            vertical_semantic_planes: Vertical semantic plane confidence (batch, num_patches)
            spatial_navigability: Spatial navigability tensor (batch, h, w)

        Returns:
            List of detected VisionObstacle objects
        """
        obstacles = []
        batch_size = obstacle_likelihood_front.shape[0]

        for batch_idx in range(batch_size):
            # Check frontal obstacle likelihood
            front_confidence = obstacle_likelihood_front[batch_idx].item()
            if front_confidence > self.collision_threshold:
                obstacles.append(VisionObstacle(
                    direction='front',
                    confidence=front_confidence,
                    depth=1.0 - front_confidence,  # Semantic navigability score (inverse of obstacle likelihood)
                    plane_type='vertical'
                ))

            # Check for obstacles in other directions based on spatial navigability
            # Divide spatial grid into directional regions
            h, w = spatial_navigability.shape[1:]
            left_region = spatial_navigability[batch_idx, :, :w//3]
            right_region = spatial_navigability[batch_idx, :, -w//3:]
            front_region = spatial_navigability[batch_idx, h//3:2*h//3, w//3:2*w//3]  # Center front

            # Calculate directional obstacle likelihoods based on semantic patterns
            # Use sigmoid to convert navigability scores to obstacle likelihoods
            left_confidence = 1.0 / (1.0 + torch.exp(left_region.mean()))  # Sigmoid-based obstacle likelihood
            right_confidence = 1.0 / (1.0 + torch.exp(right_region.mean()))
            front_center_confidence = 1.0 / (1.0 + torch.exp(front_region.mean()))

            # Check left obstacle
            if left_confidence > self.collision_threshold:
                obstacles.append(VisionObstacle(
                    direction='left',
                    confidence=left_confidence,
                    depth=1.0 - left_confidence,
                    plane_type='vertical'
                ))

            # Check right obstacle
            if right_confidence > self.collision_threshold:
                obstacles.append(VisionObstacle(
                    direction='right',
                    confidence=right_confidence,
                    depth=1.0 - right_confidence,
                    plane_type='vertical'
                ))

        return obstacles

    def analyze_frame_async(self, vision_embeddings: torch.Tensor, callback: callable = None) -> None:
        """
        Analyze vision frame asynchronously to avoid blocking token streaming.

        Args:
            vision_embeddings: Raw vision embeddings
            callback: Function to call with detected obstacles
        """
        def _analyze():
            with torch.no_grad():
                obstacle_likelihood_front, vertical_semantic_planes, spatial_navigability = self.forward(vision_embeddings)
                obstacles = self.detect_obstacles(obstacle_likelihood_front, vertical_semantic_planes, spatial_navigability)

                if callback:
                    callback(obstacles)

        # Run analysis in background thread
        analysis_thread = Thread(target=_analyze, daemon=True)
        analysis_thread.start()

    def fuse_multimodal_bias(self, manual_bias: Dict[str, float],
                           vision_obstacles: list[VisionObstacle]) -> Dict[str, float]:
        """
        Fuse manual keyboard bias with visual obstacle detection.
        Vision has vectorial priority over manual control.

        Args:
            manual_bias: Bias from keyboard input (direction -> strength)
            vision_obstacles: Detected obstacles from vision

        Returns:
            Fused bias dictionary
        """
        fused_bias = manual_bias.copy()

        # Apply vision priority: if vision detects obstacle, override manual direction
        for obstacle in vision_obstacles:
            if obstacle.confidence > self.collision_threshold:
                # Penalize movement in obstacle direction
                if obstacle.direction in fused_bias:
                    fused_bias[obstacle.direction] = max(0.0, fused_bias[obstacle.direction] - obstacle.confidence)

                # Boost safe fallback directions
                if obstacle.direction == 'front':
                    # If front blocked, boost backward and turning
                    fused_bias['backward'] = fused_bias.get('backward', 0.0) + 0.5
                    fused_bias['left'] = fused_bias.get('left', 0.0) + 0.3
                    fused_bias['right'] = fused_bias.get('right', 0.0) + 0.3
                elif obstacle.direction == 'left':
                    fused_bias['right'] = fused_bias.get('right', 0.0) + 0.4
                elif obstacle.direction == 'right':
                    fused_bias['left'] = fused_bias.get('left', 0.0) + 0.4

        return fused_bias

    def generate_runtime_events(self, vision_obstacles: list[VisionObstacle]) -> list[RuntimeEvent]:
        """
        Generate runtime events from vision obstacles.

        Args:
            vision_obstacles: Detected obstacles

        Returns:
            List of RuntimeEvent objects
        """
        events = []

        for obstacle in vision_obstacles:
            if obstacle.confidence > self.collision_threshold:
                event = RuntimeEvent(
                    type='vision_obstacle',
                    payload={
                        'direction': obstacle.direction,
                        'confidence': obstacle.confidence,
                        'depth': obstacle.depth,
                        'plane_type': obstacle.plane_type
                    }
                )
                events.append(event)

        return events


def create_spatial_semantic_analyzer_from_gemma_config(gemma_config: Any, device: str = "cpu") -> SpatialSemanticAnalyzer:
    """
    Create SpatialSemanticAnalyzer instance configured for Gemma 3 vision tower.

    Args:
        gemma_config: Gemma model configuration object
        device: Target device

    Returns:
        Configured SpatialSemanticAnalyzer
    """
    # Extract vision configuration from Gemma config
    vision_config = getattr(gemma_config, 'vision_config', None)
    if vision_config:
        hidden_size = getattr(vision_config, 'hidden_size', 1024)
        image_size = getattr(vision_config, 'image_size', 896)
        patch_size = getattr(vision_config, 'patch_size', 14)
        # Number of patches per side: image_size / patch_size = 64 for 896/14
        # SWA navigability grid is the 16×16 logical grid (after mm_tokens_per_image=256 pooling)
        # 256 tokens = 16×16 grid → spatial_resolution for SpatialSemanticAnalyzer
        mm_tokens = getattr(gemma_config, 'mm_tokens_per_image', 256)
        nav_grid_side = int(mm_tokens ** 0.5)  # sqrt(256) = 16
        spatial_resolution = (nav_grid_side, nav_grid_side)
    else:
        # Default values for Gemma 3
        hidden_size = 1152
        spatial_resolution = (16, 16)

    dtype = torch.float32  # Match Gemma's dtype

    return SpatialSemanticAnalyzer(
        vision_hidden_size=hidden_size,
        spatial_resolution=spatial_resolution,
        device=device,
        dtype=dtype
    )


# Image buffer monitoring functionality
class ImageBufferMonitor:
    """Monitor input_frames directory for new images to process."""

    def __init__(self, frames_dir: str = "input_frames", analyzer: SpatialSemanticAnalyzer = None, gemma_gen = None):
        self.frames_dir = frames_dir
        self.analyzer = analyzer
        self.gemma_gen = gemma_gen
        self.processed_frames = set()
        self.running = False

        # Create frames directory if it doesn't exist
        os.makedirs(frames_dir, exist_ok=True)

    def start_monitoring(self, callback: callable = None):
        """Start monitoring the frames directory asynchronously."""
        self.running = True

        def _monitor():
            while self.running:
                try:
                    # Check for new image files
                    for filename in os.listdir(self.frames_dir):
                        if filename.endswith(('.jpg', '.jpeg', '.png')) and filename not in self.processed_frames:
                            filepath = os.path.join(self.frames_dir, filename)
                            self._process_image_file(filepath, callback)
                            self.processed_frames.add(filename)

                    time.sleep(0.1)  # Poll every 100ms

                except Exception as e:
                    print(f"Image monitoring error: {e}")
                    time.sleep(1.0)

        monitor_thread = Thread(target=_monitor, daemon=True)
        monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring."""
        self.running = False

    def _process_image_file(self, filepath: str, callback: callable):
        """Process a single image file, extracting real vision embeddings if possible."""
        print(f"Processing image: {filepath}")

        # Check if we have the real GemmaGenerator and components available
        if (self.gemma_gen is not None 
                and getattr(self.gemma_gen, "processor", None) is not None 
                and getattr(self.gemma_gen, "vision_tower", None) is not None):
            try:
                from PIL import Image
                
                # 1. Load real image using PIL
                image = Image.open(filepath).convert("RGB")
                
                # 2. Preprocess the image through AutoProcessor
                # Use image_processor directly — bypasses Gemma3Processor boi_token requirement
                inputs = self.gemma_gen.processor.image_processor(images=image, return_tensors="pt")
                pixel_values = inputs["pixel_values"]
                
                # Ensure device and dtype alignment
                device = self.gemma_gen.device
                dtype = self.gemma_gen.dtype
                pixel_values = pixel_values.to(device=device, dtype=dtype)
                
                # 3. Extract the actual vision embeddings using vision_tower
                with torch.no_grad():
                    vision_outputs = self.gemma_gen.vision_tower(pixel_values)
                    
                    if hasattr(vision_outputs, "last_hidden_state"):
                        _raw = vision_outputs.last_hidden_state
                    else:
                        _raw = vision_outputs[0]
                    
                    # Pool 4096 patches → 256 tokens for SpatialSemanticAnalyzer.
                    # SiglipVisionModel with 896×896 @ patch_size=14 produces 64×64=4096 patches.
                    # The multi_modal_projector compresses to 256, but here we use adaptive
                    # average pooling to remain independent of the LM projector.
                    if _raw.shape[1] != 256:
                        import torch.nn.functional as F
                        _perm = _raw.permute(0, 2, 1)              # (1, 1152, N)
                        _perm = F.adaptive_avg_pool1d(_perm, 256)  # (1, 1152, 256)
                        vision_embeddings = _perm.permute(0, 2, 1) # (1, 256, 1152)
                    else:
                        vision_embeddings = _raw
                
                # Detach and clone to prevent accumulation of dynamic graph gradients
                vision_embeddings = vision_embeddings.detach().clone().float()
                
            except Exception as e:
                print(f"⚠️ Error extracting real vision embeddings from {filepath}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to simulated embeddings on error
                dev = self.analyzer.device if self.analyzer else "cpu"
                dt = self.analyzer.dtype if self.analyzer else torch.float32
                hidden_size = self.analyzer.vision_hidden_size if self.analyzer else 1024
                vision_embeddings = torch.randn(1, 256, hidden_size, device=dev, dtype=dt)
        else:
            # Fallback/Simulation mode (RAM fallback/no model loaded)
            time.sleep(0.05)  # Simulate processing delay
            # Determine appropriate device and dtype from analyzer
            dev = self.analyzer.device if self.analyzer else "cpu"
            dt = self.analyzer.dtype if self.analyzer else torch.float32
            hidden_size = self.analyzer.vision_hidden_size if self.analyzer else 1024
            vision_embeddings = torch.randn(1, 256, hidden_size, device=dev, dtype=dt)

        # Run analysis
        if self.analyzer:
            self.analyzer.analyze_frame_async(vision_embeddings, callback)


if __name__ == "__main__":
    # Test the SpatialSemanticAnalyzer
    analyzer = SpatialSemanticAnalyzer(vision_hidden_size=1024, device="cpu")

    # Simulate vision embeddings (semantic visual features)
    vision_embeds = torch.randn(1, 256, 1024)  # Batch of 1, 256 patches, 1024 hidden size

    # Test forward pass
    obstacle_likelihood_front, vertical_semantic_planes, spatial_navigability = analyzer(vision_embeds)
    print(f"Obstacle likelihood front shape: {obstacle_likelihood_front.shape}")
    print(f"Vertical semantic planes shape: {vertical_semantic_planes.shape}")
    print(f"Spatial navigability shape: {spatial_navigability.shape}")

    # Test obstacle detection
    obstacles = analyzer.detect_obstacles(obstacle_likelihood_front, vertical_semantic_planes, spatial_navigability)
    print(f"Detected obstacles: {len(obstacles)}")

    # Test event generation
    events = analyzer.generate_runtime_events(obstacles)
    print(f"Generated events: {len(events)}")

    print("SpatialSemanticAnalyzer test completed successfully")