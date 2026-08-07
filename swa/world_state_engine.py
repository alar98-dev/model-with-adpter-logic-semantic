# swa/world_state_engine.py
"""
World State Engine for dynamic prompt injection and token manipulation.
Implements the 3-layer architecture: CORE IDENTITY + OBSERVATION PROTOCOL + RUNTIME STATE INJECTION.
"""

from transformers import LogitsProcessor
import torch
from .world_engine_prompt import CORE_IDENTITY, OBSERVATION_PROTOCOL


class WorldStateEngine:
    """
    Engine for building dynamic prompts with spatial state injection.
    Maintains attention regions and runtime telemetry.
    """

    def __init__(self):
        self.core = CORE_IDENTITY
        self.observation_protocol = OBSERVATION_PROTOCOL
        self.regions = {
            "ZONE_A": "corredor frontal",
            "ZONE_B": "borda esquerda",
            "ZONE_C": "profundidade escura"
        }

    def build_world_state(self, telemetry: dict) -> str:
        """
        Build the runtime state string from telemetry data.

        Args:
            telemetry: Dict containing position, vectors, obstacles, etc.

        Returns:
            Formatted runtime state string.
        """
        position = telemetry.get('position', 'unknown')
        navigation_vector = telemetry.get('vector', 'none')
        obstacles = telemetry.get('obstacles', [])
        confidence = telemetry.get('confidence', 0.0)

        runtime = f"""
[RUNTIME STATE]
Posição atual: {position}
Vetor de navegação: {navigation_vector}
Obstáculos detectados: {', '.join(obstacles) if obstacles else 'nenhum'}
Confiança: {confidence:.2f}
        """.strip()

        return runtime

    def build_attention_regions(self, active_zones: list = None) -> str:
        """
        Build attention regions string for ROI prompting.

        Args:
            active_zones: List of zone keys to focus on.

        Returns:
            Formatted attention regions string.
        """
        if active_zones is None:
            active_zones = list(self.regions.keys())

        attention_regions = "\n".join([
            f"{zone}: {self.regions[zone]}"
            for zone in active_zones
            if zone in self.regions
        ])

        return f"""
[ATTENTION_REGIONS]
{attention_regions}
        """.strip()

    def get_prompt(self, telemetry: dict, active_zones: list = None, user_input: str = "") -> str:
        """
        Generate the complete prompt with all layers.

        Args:
            telemetry: Runtime telemetry data.
            active_zones: Active attention regions.
            user_input: User input to append.

        Returns:
            Complete formatted prompt.
        """
        world_state = self.build_world_state(telemetry)
        attention = self.build_attention_regions(active_zones)

        full_prompt = f"""
{self.core}

{self.observation_protocol}

{attention}

{world_state}

{user_input}
        """.strip()

        return full_prompt


class SpaceBiasProcessor(LogitsProcessor):
    """
    Logit processor for spatial bias injection.
    Applies directional bias based on navigation vectors and obstacle detection.
    """

    def __init__(self, tokenizer, spatial_bias_vector: torch.Tensor = None, dtype: torch.dtype = torch.float32):
        """
        Initialize the spatial bias processor.

        Args:
            tokenizer: The tokenizer for token ID mapping.
            spatial_bias_vector: Pre-computed bias vector for spatial directions.
            dtype: Data type for tensor operations (must match model dtype).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.dtype = dtype
        self.spatial_bias_vector = spatial_bias_vector or torch.zeros(tokenizer.vocab_size, dtype=self.dtype)

        # Define token mappings for directions (these should be customized based on your tokenizer)
        self.direction_tokens = {
            'north': ['norte', 'north', 'up', 'cima', 'nordeste', 'noroeste', 'setentrional', 'top'],
            'south': ['sul', 'south', 'down', 'baixo', 'sudeste', 'sudoeste', 'meridional', 'bottom'],
            'east': ['leste', 'east', 'right', 'direita', 'oriental', 'oriente', 'side'],
            'west': ['oeste', 'west', 'left', 'esquerda', 'ocidental', 'ocidente', 'port'],
            'forward': ['frente', 'forward', 'avançar', 'adiante', 'prosseguir', 'ahead', 'straight'],
            'backward': ['trás', 'backward', 'recuar', 'retroceder', 'back', 'behind', 'rear']
        }

        # Define token mappings for obstacles grouped by semantic categories to prevent circumvention via synonyms
        self.obstacle_tokens = {
            'physical_barriers': ['parede', 'wall', 'muro', 'barreira', 'barrier', 'tijolo', 'brick', 'concreto', 'concrete', 'pedra', 'stone'],
            'logical_blocks': ['bloqueio', 'block', 'obstáculo', 'obstacle', 'impedimento', 'hindrance', 'fechado', 'closed', 'stop'],
            'action_constraints': ['parar', 'halt', 'trancado', 'locked', 'interrompido', 'broken', 'deadend', 'beco']
        }

        # Build token ID mappings
        self._build_token_mappings()

    def _build_token_mappings(self):
        """Build mappings from direction and obstacle names to token IDs."""
        self.token_id_mappings = {}
        # Build direction mappings
        for direction, tokens in self.direction_tokens.items():
            ids = []
            for token in tokens:
                token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if token_id:
                    ids.extend(token_id)
            self.token_id_mappings[direction] = list(set(ids))  # Remove duplicates

        # Build obstacle mappings
        for obstacle, tokens in self.obstacle_tokens.items():
            ids = []
            for token in tokens:
                token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if token_id:
                    ids.extend(token_id)
            self.token_id_mappings[obstacle] = list(set(ids))  # Remove duplicates

    def apply_spatial_bias(self, navigation_vector: str, obstacles: list = None) -> torch.Tensor:
        """
        Compute spatial bias based on navigation vector and obstacles.

        Args:
            navigation_vector: Current navigation direction (e.g., 'north').
            obstacles: List of obstacle directions or types to penalize (e.g., ['north', 'wall']).

        Returns:
            Bias tensor to add to logits.
        """
        bias = torch.zeros_like(self.spatial_bias_vector)

        # --- DYNAMIC SPATIAL PENALTIES (Dynamic Parameter Vector) ---
        # Calculate dynamic penalty weight based on the environmental stress
        try:
            entropy_vec_10d = self.entropy_daemon.get_vector() if hasattr(self, 'entropy_daemon') and self.entropy_daemon is not None else None
            if entropy_vec_10d is None:
                stress = 1.0
            else:
                # Use IO stress + CPU temperature to scale the spatial constraint strength
                stress = float(entropy_vec_10d[1].item()) * 0.01 + float(entropy_vec_10d[0].item()) * 0.01
                stress = max(0.2, min(2.5, stress))
        except Exception:
            stress = 1.0

        boost_val = 2.0 * stress
        penalty_val = 8.0 * stress
        fallback_val = 3.0 * stress

        # Boost tokens for navigation direction
        if navigation_vector in self.token_id_mappings:
            for token_id in self.token_id_mappings[navigation_vector]:
                if token_id < len(bias):
                    bias[token_id] += boost_val  # Dynamic Boost

        # Penalize tokens for obstacle directions/types using safer penalty-based approach
        # Avoid -inf to maintain stability and prevent degeneration
        if obstacles:
            for obstacle in obstacles:
                if obstacle in self.token_id_mappings:
                    for token_id in self.token_id_mappings[obstacle]:
                        if token_id < len(bias):
                            bias[token_id] -= penalty_val  # Dynamic Penalty

        # Boost recovery vectors (fallback actions)
        recovery_directions = ['backward', 'south']  # Assume backward/south as safe fallback
        if obstacles:
            for recovery in recovery_directions:
                if recovery in self.token_id_mappings:
                    for token_id in self.token_id_mappings[recovery]:
                        if token_id < len(bias):
                            bias[token_id] += fallback_val  # Dynamic Boost

        return bias

    def apply_spatial_embedding_bias(self, navigation_vector: str, obstacles: list = None, embedding_matrix: torch.Tensor = None) -> torch.Tensor:
        """
        Compute spatial bias in embedding space for deeper reasoning modification.

        Args:
            navigation_vector: Current navigation direction (e.g., 'north').
            obstacles: List of obstacle types to penalize.
            embedding_matrix: Model's embedding matrix (vocab_size x d_model).

        Returns:
            Bias tensor in embedding space (d_model,) to add to inputs_embeds.
        """


        if embedding_matrix is None:
            return torch.zeros(1152, dtype=self.dtype)

        hidden_size = embedding_matrix.shape[1]
        bias = torch.zeros(hidden_size, device=embedding_matrix.device, dtype=embedding_matrix.dtype)

        # print("RETURNING BIAS:", bias.shape)

        # Apply spatial modifications here
        # For now, return zeros to fix tensor alignment

        return bias

    def __call__(self, input_ids, scores):
        """
        Apply spatial bias to logits.

        Args:
            input_ids: Input token IDs.
            scores: Logit scores.

        Returns:
            Modified logit scores.
        """
        # Apply the spatial bias vector if it exists
        if self.spatial_bias_vector is not None and scores.shape == self.spatial_bias_vector.shape:
            return scores + self.spatial_bias_vector
        else:
            # If shapes don't match, return scores unchanged to avoid dimension errors
            return scores