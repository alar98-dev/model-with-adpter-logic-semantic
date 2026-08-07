from __future__ import annotations

import re
import torch
from typing import Optional
from threading import Thread, Event

from .gemma_integration import GemmaGenerator
from .eventual_state import EventualState


class BroadcastRuntime:
    """Thalamus-like runtime: filters input, updates state, generates multimodal responses."""

    def __init__(self, gemma_gen: GemmaGenerator, tokenizer, eventual_state: EventualState, device: str):
        self.gemma_gen = gemma_gen
        self.tokenizer = tokenizer
        self.eventual_state = eventual_state
        self.device = device
        self.running = False
        self.stop_event = Event()
        self.vision_analyzer = None

    def start(self):
        """Start the runtime."""
        self.running = True
        self.stop_event.clear()
        print("BroadcastRuntime started")

    def stop(self):
        """Stop the runtime."""
        self.running = False
        self.stop_event.set()
        print("BroadcastRuntime stopped")

    def set_vision_analyzer(self, analyzer):
        """Set optional vision analyzer."""
        self.vision_analyzer = analyzer

    def inject_user_command(self, command: str):
        """Parse and inject user command into eventual state, then generate response."""
        self._parse_command(command)
        spatial_bias = self.eventual_state.get_embedding_steering()
        logit_bias = self.eventual_state.get_logit_steering()
        if spatial_bias is None:
            spatial_bias = torch.zeros(self.gemma_gen.d_model, device=self.device, dtype=self.gemma_gen.dtype)
        if logit_bias is None:
            logit_bias = torch.zeros(self.gemma_gen.vocab_size, device=self.device, dtype=self.gemma_gen.dtype)

        # Generate multimodal response
        response = ""
        for token in self.gemma_gen.stream_with_multimodal_bias(spatial_bias, logit_bias, command):
            print(token, end="", flush=True)
            response += token
        print()  # New line after response

    def _parse_command(self, command: str):
        """Parse natural language commands to update eventual state."""
        cmd = command.lower().strip()

        # Movement commands
        if "move forward" in cmd or "avançar" in cmd:
            self.eventual_state.vectors = {"forward": 1.0}
        elif "move backward" in cmd or "recuar" in cmd:
            self.eventual_state.vectors = {"backward": 1.0}
        elif "turn left" in cmd or "virar esquerda" in cmd:
            self.eventual_state.vectors = {"left": 1.0}
        elif "turn right" in cmd or "virar direita" in cmd:
            self.eventual_state.vectors = {"right": 1.0}
        elif "stop" in cmd or "parar" in cmd:
            self.eventual_state.vectors = {"forward": 0.0, "backward": 0.0, "left": 0.0, "right": 0.0}

        # Obstacle commands
        obstacle_match = re.search(r'obstacle (\w+) (\d*\.?\d+)', cmd)
        if obstacle_match:
            direction, confidence = obstacle_match.groups()
            confidence = float(confidence)
            self.eventual_state.obstacles = [{"direction": direction, "confidence": confidence}]

        # Clear obstacles
        if "clear obstacles" in cmd or "limpar obstáculos" in cmd:
            self.eventual_state.obstacles = []

        # Default: treat as prompt
        else:
            # For general prompts, no state change
            pass