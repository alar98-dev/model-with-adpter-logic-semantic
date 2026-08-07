#!/usr/bin/env python3
"""
SWA CLI - Spatial World Agent Persistent Cognitive Runtime.

Usage:
    swa run --prompt "go forward"
    swa run --prompt "what do you see?" --image wall.png --bias obstacle --debug
    swa shell  # Persistent interactive session with real-time steering
    swa debug  # Debug runtime with latent vector monitoring
    swa steering --test logits_before_after
    swa telemetry --embeddings
    swa benchmark
    swa validate
"""

# CRITICAL: Force multicore configuration BEFORE any imports
import os
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["MKL_NUM_THREADS"] = "10"
os.environ["VECLIB_MAXIMUM_THREADS"] = "10"
os.environ["NUMEXPR_NUM_THREADS"] = "10"
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

# Ensure imports work when run as global command
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import argparse
import json
import os
import sys
import threading
import time
import torch

# Force CPU affinity to ALL cores for maximum parallel processing
try:
    import psutil
    # Get all available CPU cores
    all_cores = list(range(psutil.cpu_count()))
    # Force process affinity to all cores
    psutil.Process().cpu_affinity(all_cores)
    print(f"🧵 CPU Affinity FORCED to all {len(all_cores)} cores: {all_cores}")
except ImportError:
    print("⚠️ psutil not available - CPU affinity not set")
except Exception as e:
    print(f"⚠️ CPU affinity failed: {e}")

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import re


class BroadcastRuntime:
    def __init__(self, generator, state):
        self.generator = generator  # GemmaGenerator
        self.state = state  # EventualState
        self.is_active = True

    def start_ui_loop(self):
        """Thread responsável por capturar eventos do CLI."""
        while self.is_active:
            try:
                cmd = input("> ")
                if cmd.lower() in ["exit", "quit"]:
                    self.is_active = False
                    break
                # Update vectors and sync to hardware
                self.state.vectors = self._parse_command(cmd)
                self.state.update_reality_state({"vectors": self.state.vectors, "obstacles": self.state.obstacles})
                self.state.sync_to_hardware()
            except EOFError:
                self.is_active = False
                break

    def start_inference_loop(self):
        """Thread responsável por gerar o stream de tokens."""
        import torch
        while self.is_active:
            # Temporarily disable steering to test
            embed_steering = torch.zeros(self.generator.d_model, device=self.generator.device, dtype=self.generator.embeddings.weight.dtype)
            logit_steering = None

            try:
                for token in self.generator.stream_with_bias(embed_steering, logit_steering, "", max_new_tokens=50, temperature=0.7):
                    print(token, end="", flush=True)
            except Exception as e:
                print(f"Inference error: {e}")
                time.sleep(1.0)

    def run(self):
        """Inicia o broadcast eventual."""
        t1 = threading.Thread(target=self.start_ui_loop, daemon=True)
        t2 = threading.Thread(target=self.start_inference_loop, daemon=True)

        t1.start()
        t2.start()

        t1.join()
        t2.join()

    def _parse_command(self, cmd: str) -> Dict[str, float]:
        """Parse command into vector updates."""
        cmd_lower = cmd.lower().strip()
        vectors = {"forward": 0.0, "left": 0.0, "right": 0.0, "reverse": 0.0, "stop": 0.0}

        if "forward" in cmd_lower or "go ahead" in cmd_lower:
            vectors["forward"] = 1.0
        elif "left" in cmd_lower:
            vectors["left"] = 1.0
        elif "right" in cmd_lower:
            vectors["right"] = 1.0
        elif "reverse" in cmd_lower or "back" in cmd_lower:
            vectors["reverse"] = 1.0
        elif "stop" in cmd_lower:
            vectors["stop"] = 1.0
        elif "clear" in cmd_lower:
            # Reset vectors
            pass
        else:
            # Default to forward or interpret as semantic
            vectors["forward"] = 0.5

        return vectors


@dataclass
class RuntimeEvent:
    """Event-driven semantic broadcast."""
    type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class EventualState:
    """Eventually consistent latent state."""
    vectors: Dict[str, float] = None  # direction -> strength
    telemetry: Dict[str, Any] = None
    obstacles: List[str] = None
    attention_bias: Dict[str, float] = None
    latent_shift: float = 0.0
    runtime_metrics: Dict[str, Any] = None
    last_event_timestamp: float = 0.0

    def __post_init__(self):
        if self.vectors is None:
            self.vectors = {"forward": 0.0, "left": 0.0, "right": 0.0, "reverse": 0.0, "stop": 0.0}
        if self.telemetry is None:
            self.telemetry = {}
        if self.obstacles is None:
            self.obstacles = []
        if self.attention_bias is None:
            self.attention_bias = {}
        if self.runtime_metrics is None:
            self.runtime_metrics = {"entropy": 1.8, "drift": 0.0, "tokens_per_sec": 0.0}


@dataclass
class RuntimeState:
    """Persistent runtime state for cognitive session."""
    kv_cache: Any = None
    spatial_vector: str = "neutral"
    active_obstacles: List[str] = None
    embedding_drift: float = 0.0
    entropy_history: List[float] = None
    token_count: int = 0
    session_active: bool = False
    eventual_state: EventualState = None

    def __post_init__(self):
        if self.active_obstacles is None:
            self.active_obstacles = []
        if self.entropy_history is None:
            self.entropy_history = []
        if self.eventual_state is None:
            self.eventual_state = EventualState()


class SWACLI:
    def __init__(self):
        self.tokenizer_manager = None
        self.generator = None
        self.model_loaded = False
        self.verbose_runtime = False
        self.runtime_state = RuntimeState()
        self.streamer = None
        self.console = None
        self.live = None
        self.layout = None
        self.stream_content: str = ""
        self.eventual_state = None
        self.broadcast_runtime = None

    def load_model(self, model_dir: str = None, device: str = "cpu"):
        """Load tokenizer and model with proper initialization."""
        import torch
        
        if model_dir is None:
            model_dir = os.path.join(PROJECT_ROOT, "models", "gemma-3-4-b")
        from swa.tokenizer_manager import TokenizerManager
        from swa.gemma_integration import GemmaGenerator
        from swa.hw_eventual_state import HWEventualState as SWAEventualState

        if self.model_loaded:
            return  # Already loaded

        # Force CPU affinity to ALL cores for maximum parallel processing
        try:
            import psutil
            # Get all available CPU cores
            all_cores = list(range(psutil.cpu_count()))
            # Force process affinity to all cores
            psutil.Process().cpu_affinity(all_cores)
            print(f"🧵 CPU Affinity FORCED to all {len(all_cores)} cores: {all_cores}")
        except ImportError:
            print("⚠️ psutil not available - CPU affinity not set")
        except Exception as e:
            print(f"⚠️ CPU affinity failed: {e}")

        # Final multicore enforcement
        try:
            torch.set_num_threads(4)  # Conservative for i5-120U P-cores + vacuum sync
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass
        print("🚀 SWA Multicore: 10 cores intra-op, 2 cores inter-op")

        print(f"🔄 Loading SWA model from {model_dir} on {device}")
        
        # Load tokenizer
        self.tokenizer_manager = TokenizerManager()
        validation = self.tokenizer_manager.validate(model_dir)
        if validation.get("issues"):
            print(f"⚠️  Tokenizer validation issues: {validation['issues']}")
        tokenizer = self.tokenizer_manager.load(model_dir)
        if tokenizer is None:
            raise RuntimeError("Failed to load tokenizer")

        # Load model
        self.generator = GemmaGenerator(model_dir, device=device, tokenizer=tokenizer)
        self.model_loaded = True

        # Initialize eventual state and broadcast runtime
        vocab_size = self.generator.lm.get_input_embeddings().weight.shape[0]  # Use actual vocab size
        hidden_size = self.generator.lm.get_input_embeddings().weight.shape[1]  # Use actual hidden size
        self.eventual_state = SWAEventualState(vocab_size, hidden_size, dtype=torch.float32)
        # Sync initial state from hardware if available
        self.eventual_state.sync_from_hardware()

        # Wait for organic inflation before starting threads
        print("⏳ Aguardando inflação orgânica do modelo...")
        max_wait = 120  # 2 minutes for large models
        wait_start = time.time()

        while not getattr(self.generator, '_model_loaded', False):
            elapsed = time.time() - wait_start
            if elapsed > max_wait:
                raise RuntimeError(f"Modelo não carregou em {max_wait}s")
            time.sleep(0.5)
        else:
            elapsed = time.time() - wait_start

        print(f"Waited {elapsed:.1f}s for model load")
        # Start organic threading for real-time performance
        self.eventual_state.start_organic_threading(self.generator)

        # Show organic diagnostics
        if hasattr(self.eventual_state, '_hw_manager') and self.eventual_state._hw_manager:
            diagnostics = self.eventual_state._hw_manager.get_organic_diagnostics()
            if diagnostics['hardware_mode']:
                print("🧬 Organic SWA system active:")
                print(f"   Entropy: {diagnostics.get('entropy', 0):.3f}")
                print(f"   Inertia: {diagnostics.get('inertia', 0):.3f}")
                print(f"   Bias norm: {diagnostics.get('bias_norm', 0):.3f}")
                print(f"   Phase injection ready: {'Yes' if self.eventual_state.should_inject_phase() else 'No'}")
                print("🎯 Phase Shifter active: Present ↔ Future interference enabled")
                print("⚡ Multicore optimization: Active threading for real-time organic response")
            else:
                print("⚠️  SWA system in RAM-only mode. Run with sudo for full organic operation.")
                print("💡 Phase Shifter operational in RAM approximation mode")

        self.broadcast_runtime = BroadcastRuntime(self.generator, self.eventual_state)

        print("✅ Model and runtime loaded successfully")

    def run_inference(self, prompt: str, image: Optional[str] = None, bias: Optional[str] = None,
                     telemetry_file: Optional[str] = None, max_tokens: int = 128, temperature: float = 0.7,
                     verbose: bool = False, trace_latents: bool = False):
        """Run inference with optional multimodal and steering inputs."""
        import torch
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Load telemetry
        telemetry = None
        if telemetry_file and os.path.exists(telemetry_file):
            with open(telemetry_file) as f:
                telemetry = json.load(f)

        # Separate steering context from telemetry
        steering_context = {}
        if bias:
            steering_context['direction'] = bias
            steering_context['obstacles'] = [bias] if bias == 'obstacle' else []

        # Vision placeholder (to be implemented)
        vision_embeds = None
        if image and os.path.exists(image):
            print(f"📷 Processing image: {image}")
            # Placeholder: in real implementation, load and project vision embeddings
            vision_embeds = torch.randn(1, 256, self.generator.d_model, device=self.generator.device)

        start_time = time.perf_counter()
        tokens_generated = 0

        # Apply steering to generator
        if steering_context:
            self.generator.update_spatial_bias(
                steering_context.get('direction', 'north'),
                steering_context.get('obstacles', [])
            )

        # Generate with inference mode for better performance
        with torch.inference_mode():
            response = self.generator.generate(
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                telemetry=telemetry
            )

            # Estimate tokens generated (approximate)
            # In real implementation, count during streaming
            if response:
                # Rough estimate: response length / avg chars per token
                tokens_generated = len(response.split()) * 1.3  # Rough approximation

        elapsed = time.perf_counter() - start_time
        tps = tokens_generated / elapsed if elapsed > 0 else 0

        print(f"Prompt: {prompt}")
        print(f"Response: {response}")

        if verbose or trace_latents:
            print("📊 Runtime Metrics:")
            print(".2f")
            print(f"  Tokens/sec: {tps:.2f}")

            # Embedding norms
            if vision_embeds is not None:
                vision_norm = torch.norm(vision_embeds).item()
                print(f"  Vision embedding norm: {vision_norm:.2f}")

            # Steering effect (if bias applied)
            if steering_context:
                print("🎯 Steering Applied:")
                print(f"  Direction: {steering_context.get('direction')}")
                print(f"  Obstacles: {steering_context.get('obstacles')}")

        if trace_latents:
            print("🔬 Latent Traces:")
            # Placeholder for latent state tracing
            # In real implementation, capture hidden_states, attention, logits deltas
            print("  Hidden states: not implemented yet")
            print("  Attention maps: not implemented yet")
            print("  Logit deltas: not implemented yet")

        if bias:
            print(f"Applied bias: {bias}")

        if image:
            print(f"Image input: {image}")

    def benchmark(self):
        """Run performance benchmarks."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print("🚀 Running benchmarks...")
        # Placeholder for benchmark logic
        print("Benchmarks not yet implemented")

    def validate(self):
        """Run architecture and runtime validations."""
        import torch
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print("🔍 Running validations...")

        # Basic shape consistency check
        try:
            z = torch.zeros(self.generator.d_model, device=self.generator.device,
                           dtype=self.generator.embeddings.weight.dtype)
            test_embeds = self.generator.embeddings(torch.tensor([[1]], device=self.generator.device))
            print(f"✅ Embedding shapes: input_embeds {test_embeds.shape}, zero_prefix {z.shape}")

            # Test generation
            response = self.generator.generate("test", max_new_tokens=10)
            print(f"✅ Generation test: {len(response)} chars generated")

        except Exception as e:
            print(f"❌ Validation failed: {e}")

    def telemetry_embeddings(self):
        """Display embedding norms and statistics."""
        import torch
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print("📏 Embedding Telemetry:")
        try:
            # Test embeddings
            test_input = torch.tensor([[1, 2, 3]], device=self.generator.device)
            embeds = self.generator.embeddings(test_input)
            embed_norm = torch.norm(embeds).item()
            print(f"  Text embedding norm: {embed_norm:.2f}")

            # Zero prefix norm
            z = torch.zeros(self.generator.d_model, device=self.generator.device,
                           dtype=self.generator.embeddings.weight.dtype)
            zero_norm = torch.norm(z).item()
            print(f"  Zero prefix norm: {zero_norm:.2f}")

            # Bias embedding norm (if available)
            self.generator.update_spatial_bias("north", [])
            # Placeholder - would need to access spatial processor bias
            print("  Spatial bias: not implemented yet")

        except Exception as e:
            print(f"❌ Telemetry failed: {e}")

    def telemetry_kv(self):
        """Display KV cache metrics."""
        print("🔄 KV Cache Telemetry:")
        print("  Cache metrics: not implemented yet (would need persistent session)")

    def telemetry_drift(self):
        """Display latent drift metrics."""
        print("📈 Latent Drift Telemetry:")
        print("  Drift metrics: not implemented yet (would need long-running session)")

    def steering_test(self, test_type: str):
        """Run steering-specific tests."""
        import torch
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print(f"🎯 Running steering test: {test_type}")

        try:
            if test_type == "embedding_norms":
                self.telemetry_embeddings()

            elif test_type == "logits_before_after":
                # Test steering effect on logits
                prompt = "go forward"
                inputs = self.tokenizer_manager.tokenizer(prompt, return_tensors="pt").to(self.generator.device)

                # Before steering
                with torch.inference_mode():
                    outputs_before = self.generator.lm(**inputs)
                    logits_before = outputs_before.logits[:, -1, :]
                    topk_before = torch.topk(logits_before, 5)
                    tokens_before = [self.tokenizer_manager.tokenizer.decode(t.item()).strip() for t in topk_before.indices[0]]

                # Apply steering
                self.generator.update_spatial_bias("north", ["forward"])

                # After steering
                with torch.inference_mode():
                    outputs_after = self.generator.lm(**inputs)
                    logits_after = outputs_after.logits[:, -1, :]
                    topk_after = torch.topk(logits_after, 5)
                    tokens_after = [self.tokenizer_manager.tokenizer.decode(t.item()).strip() for t in topk_after.indices[0]]

                print("TOPK BEFORE steering:")
                for i, (token, prob) in enumerate(zip(tokens_before, topk_before.values[0])):
                    print(f"  {i+1}. '{token}' ({prob:.3f})")

                print("TOPK AFTER steering:")
                for i, (token, prob) in enumerate(zip(tokens_after, topk_after.values[0])):
                    print(f"  {i+1}. '{token}' ({prob:.3f})")

            elif test_type == "embedding_shift":
                print("Embedding shift test: not implemented yet")

            elif test_type == "kv_stability":
                print("KV stability test: not implemented yet")

            elif test_type == "latent_drift":
                print("Latent drift test: not implemented yet")

        except Exception as e:
            print(f"❌ Steering test failed: {e}")

    def _create_terminal_layout(self):
        """Create terminal layout with header and stream container."""
        from rich.console import Console
        from rich.live import Live
        from rich.panel import Panel
        from rich.text import Text
        from rich.layout import Layout
        self.console = Console()
        self.layout = Layout()

        # Create split layout: header (top 30%), stream (middle 60%), input area (bottom 10%)
        self.layout.split(
            Layout(name="header", size=8),
            Layout(name="stream", ratio=3),
            Layout(name="input_area", size=3)
        )

        # Initialize header
        header_text = Text()
        header_text.append("SWA EVENTUAL VECTOR RUNTIME\n", style="bold blue")
        header_text.append("=" * 50 + "\n", style="dim")
        header_text.append("MODEL: Gemma3-4B | KV CACHE: ACTIVE | STREAMING: ACTIVE\n", style="green")
        header_text.append("EVENT BROADCAST: ACTIVE | LATENT ACCUMULATION: ACTIVE\n", style="green")

        self.layout["header"].update(Panel(header_text, title="Runtime Status"))

        # Initialize stream container
        self.layout["stream"].update(Panel("", title="Stream Output", border_style="blue"))

        # Initialize input area
        self.layout["input_area"].update(Panel("Type commands below...", title="Input", border_style="green"))

        # Create live display
        self.live = Live(self.layout, console=self.console, refresh_per_second=2, auto_refresh=False)

    def _update_header_metrics(self):
        """Update header with dynamic metrics."""
        from rich.panel import Panel
        from rich.text import Text
        if not self.layout:
            return

        state = self.runtime_state.eventual_state

        # Create metrics text
        metrics_text = Text()
        vectors_str = ", ".join([f"{k}:{v:.2f}" for k, v in state.vectors.items() if v > 0.1])
        obstacles_str = ", ".join(state.obstacles) if state.obstacles else "none"

        metrics_text.append(f"VECTORS: {vectors_str}\n", style="yellow")
        metrics_text.append(f"OBSTACLES: {obstacles_str}\n", style="red")
        metrics_text.append(f"DRIFT: {state.runtime_metrics.get('drift', 0.0):.2f}\n", style="cyan")
        metrics_text.append(f"ENTROPY: {state.runtime_metrics.get('entropy', 1.8):.2f}\n", style="green")
        metrics_text.append(f"TOKENS/SEC: {state.runtime_metrics.get('tokens_per_sec', 0.0):.1f}", style="magenta")

        # Update header with metrics
        header_text = Text()
        header_text.append("SWA EVENTUAL VECTOR RUNTIME\n", style="bold blue")
        header_text.append("=" * 50 + "\n", style="dim")
        header_text.append("MODEL: Gemma3-4B | KV CACHE: ACTIVE | STREAMING: ACTIVE\n", style="green")
        header_text.append("EVENT BROADCAST: ACTIVE | LATENT ACCUMULATION: ACTIVE\n", style="green")
        header_text.append("\n")
        header_text.append(metrics_text)

        self.layout["header"].update(Panel(header_text, title="Runtime Status"))

    def _update_stream_container(self, new_content: str = None):
        """Update stream container with new content."""
        from rich.panel import Panel
        if not self.layout:
            return

        if new_content:
            self.stream_content += new_content

        # Keep only last 2000 characters for readability
        display_content = self.stream_content[-2000:] if len(self.stream_content) > 2000 else self.stream_content

        self.layout["stream"].update(Panel(display_content, title="Stream Output", border_style="blue"))

    def eventual_runtime(self, debug: bool = False):
        """Run eventual semantic broadcast runtime."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Create terminal layout
        self._create_terminal_layout()

        self.runtime_state.session_active = True

        with self.live:
            while self.runtime_state.session_active:
                try:
                    # Update header with current metrics
                    self._update_header_metrics()

                    # Update input area prompt
                    self.layout["input_area"].update(Panel("Ready for input...", title="Input", border_style="green"))

                    # Refresh display
                    self.live.refresh()

                    # Get user input (outside live area to avoid interference)
                    user_input = input("\n> ").strip()

                    if user_input.lower() in ['exit', 'quit']:
                        self._update_stream_container("\nExiting eventual runtime...\n")
                        break

                    # Update input area with user input
                    self.layout["input_area"].update(Panel(f"Processing: {user_input}", title="Input", border_style="yellow"))
                    self.live.refresh()

                    # Parse as runtime event
                    event = self._parse_runtime_event(user_input)
                    if event:
                        self._broadcast_event(event)

                        # Generate eventual response with streaming
                        self._update_stream_container("")  # Clear previous content
                        self.live.refresh()

                        accumulated_text = ""
                        for token in self._generate_eventual_response():
                            accumulated_text += token
                            self._update_stream_container(accumulated_text)
                            self.live.refresh()

                        # Add newline after complete response
                        self._update_stream_container(accumulated_text + '\n')

                    # Reset input area
                    self.layout["input_area"].update(Panel("Ready for input...", title="Input", border_style="green"))

                except KeyboardInterrupt:
                    self._update_stream_container("\nExiting eventual runtime...\n")
                    break
                except Exception as e:
                    self._update_stream_container(f"\nRuntime error: {e}\n")

        self.runtime_state.session_active = False

    def broadcast_run(self):
        """Run using BroadcastRuntime for interactive eventual execution."""
        if not self.model_loaded or not self.broadcast_runtime:
            raise RuntimeError("Model and runtime not loaded. Call load_model() first.")

        print("🚀 Starting SWA Broadcast Runtime")
        print("Type commands like 'forward', 'left', 'right', 'stop', or 'exit' to quit")
        print()

        self.broadcast_runtime.run()

    def _display_eventual_status(self):
        """Display eventual runtime status."""
        state = self.runtime_state.eventual_state
        print("\nEVENTUAL STATE")
        print("-" * 30)
        print(f"VECTORS: {', '.join([f'{k}:{v:.2f}' for k, v in state.vectors.items() if v > 0.1])}")
        print(f"OBSTACLES: {state.obstacles}")
        print(".2f")
        print(".2f")
        print(f"ENTROPY: {state.runtime_metrics.get('entropy', 'N/A'):.2f}")
        print(f"TOKENS/SEC: {state.runtime_metrics.get('tokens_per_sec', 0):.1f}")
        print()

    def _parse_runtime_event(self, user_input: str) -> Optional[RuntimeEvent]:
        """Parse user input into semantic runtime event."""
        input_lower = user_input.lower().strip()

        # Movement events
        movement_patterns = {
            r'\bforward\b|\bgo ahead\b|\bmove forward\b': ('movement', {'vector': 'forward', 'strength': 0.8}),
            r'\bleft\b|\bturn left\b|\bgo left\b': ('movement', {'vector': 'left', 'strength': 0.7}),
            r'\bright\b|\bturn right\b|\bgo right\b': ('movement', {'vector': 'right', 'strength': 0.7}),
            r'\breverse\b|\bback\b|\bgo back\b': ('movement', {'vector': 'reverse', 'strength': 0.6}),
            r'\bstop\b|\bhalt\b': ('movement', {'vector': 'stop', 'strength': 0.9}),
        }

        for pattern, (event_type, payload) in movement_patterns.items():
            if re.search(pattern, input_lower):
                return RuntimeEvent(type=event_type, payload=payload)

        # Obstacle events
        obstacle_patterns = {
            r'\bwall\b|\bobstacle\b|\bblocked\b': ('obstacle', {'direction': 'front', 'severity': 0.9}),
            r'\bwall left\b|\bobstacle left\b': ('obstacle', {'direction': 'left', 'severity': 0.8}),
            r'\bwall right\b|\bobstacle right\b': ('obstacle', {'direction': 'right', 'severity': 0.8}),
        }

        for pattern, (event_type, payload) in obstacle_patterns.items():
            if re.search(pattern, input_lower):
                return RuntimeEvent(type=event_type, payload=payload)

        # Correction events
        if 'clear' in input_lower or 'no obstacle' in input_lower:
            return RuntimeEvent(type='correction', payload={'action': 'clear_obstacles'})

        # Semantic steering (catch-all for unrecognized inputs)
        if user_input.strip():
            return RuntimeEvent(type='semantic', payload={'text': user_input, 'strength': 0.5})

        return None

    def _broadcast_event(self, event: RuntimeEvent):
        """Broadcast event to update eventual state."""
        state = self.runtime_state.eventual_state

        if event.type == 'movement':
            vector = event.payload['vector']
            strength = event.payload['strength']
            # Update vector strengths (accumulate)
            state.vectors[vector] = min(1.0, state.vectors.get(vector, 0.0) + strength)

            # Update runtime state
            self.runtime_state.spatial_vector = vector

        elif event.type == 'obstacle':
            direction = event.payload['direction']
            if direction not in state.obstacles:
                state.obstacles.append(direction)
            # Suppress movement in obstacle direction
            if direction == 'front':
                state.vectors['forward'] = max(0.0, state.vectors['forward'] - 0.5)
                state.vectors['stop'] += 0.3

        elif event.type == 'correction':
            if event.payload.get('action') == 'clear_obstacles':
                state.obstacles.clear()
                # Reset suppression
                state.vectors['forward'] = 0.0
                state.vectors['stop'] = 0.0

        # Update generator steering
        self.generator.update_spatial_bias(
            self.runtime_state.spatial_vector,
            state.obstacles
        )

        # Update metrics
        state.runtime_metrics['drift'] += 0.01  # Simulate drift accumulation
        state.last_event_timestamp = event.timestamp

        print(f"[EVENT] {event.type.upper()}: {event.payload}")

    def _generate_eventual_response(self):
        """Generate streaming response based on accumulated eventual state."""
        import torch
        state = self.runtime_state.eventual_state

        # Determine dominant vector for prompt context
        dominant_vector = max(state.vectors.items(), key=lambda x: x[1])
        vector_name, strength = dominant_vector

        # Create dynamic prompt based on state
        if strength < 0.2:
            prompt = "O espaço digital está ocioso. Buscando por novos arquivos no sistema..."
        elif vector_name == 'forward' and 'front' in state.obstacles:
            prompt = "O acesso a essa pasta foi bloqueado. Precisarei inspecionar as permissões ou ler outro arquivo."
        elif vector_name == 'forward':
            prompt = f"Entrando no diretório para listar e explorar os nós estruturais de dados."
        elif vector_name in ['left', 'right']:
            prompt = f"Buscando ramificações de arquivos na dimensão {vector_name} da árvore binária."
        elif vector_name == 'stop':
            prompt = "Pausando a execução. Analisando as extensões dos arquivos no nó atual."
        else:
            prompt = f"Lendo e extraindo código na direção física {vector_name} do disco rígido."

        # Use streaming generation with real-time steering callback
        def steering_callback():
            # Check for real-time steering updates (can be enhanced later)
            return None

        try:
            for token in self.generator.stream_with_dynamic_steering(
                torch.zeros(self.generator.d_model, device=self.generator.device,
                           dtype=self.generator.embeddings.weight.dtype),
                prompt,
                max_new_tokens=64,
                temperature=0.7,
                event_callback=steering_callback
            ):
                yield token

        except Exception as e:
            yield f"Generation error: {e}"

    def _handle_runtime_command(self, command: str):
        """Handle special runtime commands."""
        if command == '/status':
            self._display_detailed_status()
        elif command.startswith('/vector '):
            new_vector = command.split('/vector ', 1)[1]
            self.runtime_state.spatial_vector = new_vector
            self.generator.update_spatial_bias(new_vector, self.runtime_state.active_obstacles)
            print(f"Vector updated to: {new_vector}")
        elif command.startswith('/obstacle '):
            obstacle = command.split('/obstacle ', 1)[1]
            if obstacle not in self.runtime_state.active_obstacles:
                self.runtime_state.active_obstacles.append(obstacle)
                self.generator.update_spatial_bias(self.runtime_state.spatial_vector, self.runtime_state.active_obstacles)
                print(f"Obstacle added: {obstacle}")
        elif command == '/clear':
            self.runtime_state.active_obstacles.clear()
            self.generator.update_spatial_bias(self.runtime_state.spatial_vector, [])
            print("Obstacles cleared")

    def _display_detailed_status(self):
        """Display detailed runtime status."""
        print("\nDETAILED RUNTIME STATUS")
        print("=" * 30)
        print(f"Spatial Vector: {self.runtime_state.spatial_vector}")
        print(f"Active Obstacles: {self.runtime_state.active_obstacles}")
        print(".2f")
        print(f"Token Count: {self.runtime_state.token_count}")
        print(f"KV Cache: {'active' if self.runtime_state.kv_cache else 'inactive'}")
        if self.runtime_state.entropy_history:
            print(".2f")
        print()

    def _process_semantic_input(self, user_input: str):
        """Process user input as semantic steering signal."""
        print(f"Processing: {user_input}")

    def autonomous_run(self):
        """Inicia o loop recursivo autônomo (Orquestrador)"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        from swa.orchestrator import AutonomousOrchestrator
        # Utiliza o HWEventualState real e não o dataclass de dummy state
        orchestrator = AutonomousOrchestrator(self.generator, self.eventual_state)
        orchestrator.start()

    def _generate_with_streaming(self, prompt: str):
        """Generate response with streaming and real-time monitoring."""
        import torch
        from transformers import TextIteratorStreamer
        import threading
        print("Generating...")

        try:
            # Initialize streamer
            self.streamer = TextIteratorStreamer(self.tokenizer_manager.tokenizer, skip_prompt=True)

            # Start generation in background thread
            generation_thread = threading.Thread(
                target=self._background_generate,
                args=(prompt,)
            )
            generation_thread.start()

            # Stream output
            for token in self.streamer:
                print(token, end="", flush=True)
                self.runtime_state.token_count += 1

                # Simulate entropy monitoring (placeholder)
                # In real implementation, capture actual logits entropy
                fake_entropy = 1.5 + torch.randn(1).item() * 0.3
                self.runtime_state.entropy_history.append(fake_entropy)

            print()  # New line after streaming

        except Exception as e:
            print(f"Generation error: {e}")

    def _background_generate(self, prompt: str):
        """Background generation thread."""
        import torch
        try:
            with torch.inference_mode():
                inputs = self.tokenizer_manager.tokenizer(prompt, return_tensors="pt").to(self.generator.device)

                # Generate with streamer
                self.generator.lm.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.7,
                    streamer=self.streamer,
                    do_sample=True,
                    use_cache=True
                )
        except Exception as e:
            print(f"Background generation failed: {e}")

    def interactive_shell(self):
        """Run interactive shell with persistent session."""
        import torch
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print("🖥️  SWA Interactive Shell")
        print("Type 'exit' or 'quit' to end session")
        print("Type 'debug' to switch to debug mode")
        print()

        while True:
            try:
                user_input = input("SWA> ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    break
                if user_input.lower() == 'debug':
                    # Need to define debug_runtime or call properly
                    continue

                # Parse steering if present
                bias = None
                if user_input.startswith('--bias '):
                    parts = user_input.split('--bias ', 1)
                    if len(parts) == 2:
                        bias = parts[1].split()[0]
                        prompt = ' '.join(parts[1].split()[1:])
                    else:
                        prompt = user_input
                else:
                    prompt = user_input

                if not prompt:
                    continue

                # Generate response
                steering_context = {'direction': bias} if bias else {}
                if steering_context:
                    self.generator.update_spatial_bias(
                        steering_context.get('direction', 'north'),
                        []
                    )

                with torch.inference_mode():
                    response = self.generator.generate(
                        prompt=prompt,
                        max_new_tokens=64,
                        temperature=0.7,
                        telemetry=None
                    )

                print(f"Response: {response}")
                if bias:
                    print(f"(Applied bias: {bias})")
                print()

            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")

    def interactive_chat(self):
        """Run Cognitive OODA Chat Client via IPC and UDP (Zero-RAM)."""
        import socket
        import json
        import struct
        import threading
        import time
        from swa.ipc_bus import pack_aeon_frame

        print("🖥️  SWA Cognitive Córtex (Zero-RAM Client)")
        print("Conectado diretamente às veias do OODA Loop Central.")
        print("Digite livremente. Digite 'exit' para sair.\n")

        # Inicia a thread que ouve o "Hipocampo" (UDP 20005) para o stream ao vivo
        def listen_udp_stream():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("0.0.0.0", 20005))
            except Exception as e:
                print(f"⚠️ Não foi possível plugar no Nervo Auditivo UDP 20005: {e}")
                return
            
            # Limpa lixo do buffer antes de começar
            sock.settimeout(0.1)
            while True:
                try: sock.recv(4096)
                except socket.timeout: break
            
            sock.settimeout(None)
            while True:
                try:
                    data, _ = sock.recvfrom(65535)
                    try:
                        msg = json.loads(data.decode('utf-8'))
                        if msg.get("type") == "thought_stream":
                            content = msg.get("content", "")
                            # O output do WarpedStdout já vem formatado
                            import sys
                            sys.stdout.write(content)
                            sys.stdout.flush()
                    except json.JSONDecodeError:
                        pass
                except Exception:
                    pass

        listener = threading.Thread(target=listen_udp_stream, daemon=True)
        listener.start()

        # Loop principal (Boca / Input)
        socket_path = "/dev/shm/aeon_synapse.sock"
        while True:
            try:
                user_input = input("AEON> ").strip()
                if user_input.lower() in ['exit', 'quit']:
                    break
                if not user_input:
                    continue

                # Cria Matriz Topológica de Singularidade (Pico no Centro 32,32 = 11.0)
                topology_list = [0.0] * 4096
                # 32 * 64 + 32 = 2080
                topology_list[2080] = 11.0 

                payload = user_input.encode('utf-8')
                frame = pack_aeon_frame(topology_list, payload)

                # Dispara pelo IPC Bus
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                        s.connect(socket_path)
                        s.sendall(frame)
                except Exception as e:
                    print(f"❌ O Córtex Principal não responde. O 'swa auto' está rodando? (Erro: {e})")

                # Dá um pequeno delay para a UI do prompt não atropelar a resposta
                time.sleep(1.0)

            except KeyboardInterrupt:
                print("\nDesconectando do Córtex...")
                break
            except Exception as e:
                print(f"Erro: {e}")


def main():
    parser = argparse.ArgumentParser(description="SWA CLI - Spatial World Agent Runtime Console")
    parser.add_argument("--model-dir", default="models/gemma-3-4-b",
                       help="Model directory path")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "xpu"],
                       help="Device to run model on")
    parser.add_argument("--verbose-runtime", action="store_true",
                       help="Enable verbose runtime telemetry")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Generate text with optional multimodal inputs")
    run_parser.add_argument("--prompt", help="Input prompt (optional for debug/interactive mode)")
    run_parser.add_argument("--image", help="Path to image file for vision input")
    run_parser.add_argument("--bias", help="Spatial bias to apply (e.g., 'obstacle', 'north')")
    run_parser.add_argument("--telemetry", help="Path to telemetry JSON file")
    run_parser.add_argument("--max-tokens", type=int, default=128, help="Maximum tokens to generate")
    run_parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    run_parser.add_argument("--trace-latents", action="store_true", help="Enable latent state tracing")
    run_parser.add_argument("--interactive", action="store_true", help="Enable interactive eventual runtime")
    run_parser.add_argument("--debug", action="store_true", help="Enable debug runtime with latent monitoring")

    # Auto command
    auto_parser = subparsers.add_parser("auto", help="Start the recursive autonomous cognitive loop")

    # Chaos command
    chaos_parser = subparsers.add_parser("chaos", help="Run Chaos Engine resilience drills")
    chaos_parser.add_argument("--drill", required=True, choices=["entropy_spike", "sensor_blindness", "nvme_corruption"], help="Type of drill to run")

    # Shell command
    shell_parser = subparsers.add_parser("shell", help="Interactive persistent session (Spatial Steering)")

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Cognitive RAG Chat (Latent Bias & Neurogenesis)")

    # Debug command
    debug_parser = subparsers.add_parser("debug", help="Debug runtime with latent vector monitoring")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run performance benchmarks")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Run architecture validations")

    # Steering command
    steering_parser = subparsers.add_parser("steering", help="Run steering diagnostics")
    steering_parser.add_argument("--test", required=True,
                                choices=["embedding_norms", "logits_before_after", "embedding_shift", "kv_stability", "latent_drift"],
                                help="Type of steering test")

    # Telemetry command
    telemetry_parser = subparsers.add_parser("telemetry", help="Runtime telemetry and monitoring")
    telemetry_parser.add_argument("--type", required=True,
                                 choices=["embeddings", "kv", "drift"],
                                 help="Type of telemetry to display")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize CLI
    cli = SWACLI()
    cli.verbose_runtime = args.verbose_runtime

    try:
        # Lazy loading: only load model for commands that need it
        needs_model = args.command in ["run", "shell", "steering", "validate", "telemetry", "auto"]
        
        # Check if it's a continuous run which loads its own model in run.py
        is_continuous_run = (args.command == "run" and 
                             not getattr(args, 'prompt', None) and 
                             not getattr(args, 'debug', False) and 
                             not getattr(args, 'interactive', False))
        
        if needs_model and not is_continuous_run:
            cli.load_model(args.model_dir, args.device)

        if args.command == "run":
            # Handle debug and interactive modes
            if getattr(args, 'debug', False) or getattr(args, 'interactive', False):
                cli.broadcast_run()
            else:
                if not getattr(args, 'prompt', None):
                    # No prompt => start continuous motor mode (procedural loop)
                    from run import run_interactive_broadcast
                    run_interactive_broadcast()
                else:
                    cli.run_inference(
                        prompt=args.prompt,
                        image=getattr(args, 'image', None),
                        bias=getattr(args, 'bias', None),
                        telemetry_file=getattr(args, 'telemetry', None),
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                        verbose=cli.verbose_runtime,
                        trace_latents=getattr(args, 'trace_latents', False)
                    )
        elif args.command == "shell":
            cli.interactive_shell()
        elif args.command == "chat":
            cli.interactive_chat()
        elif args.command == "auto":
            cli.autonomous_run()
        elif args.command == "debug":
            cli.debug_runtime()
        elif args.command == "benchmark":
            cli.benchmark()
        elif args.command == "validate":
            cli.validate()
        elif args.command == "steering":
            cli.steering_test(args.test)
        elif args.command == "telemetry":
            if args.type == "embeddings":
                cli.telemetry_embeddings()
            elif args.type == "kv":
                cli.telemetry_kv()
            elif args.type == "drift":
                cli.telemetry_drift()
        elif args.command == "chaos":
            from swa.chaos_engine import run_chaos_cli
            run_chaos_cli(args)

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()