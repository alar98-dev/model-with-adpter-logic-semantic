import os
import json
import time

from cortex.obelisk.dreamer.world_model import WorldModel
from cortex.obelisk.dreamer.simulator import Simulator
from cortex.obelisk.dreamer.planner import Planner

PLAN_FILE = "/var/run/aeon/dreamer.plan"
EPISODE_FILE = "/var/lib/obelisk/dreamer_episodes.jsonl"


class Dreamer:
    def __init__(self):
        self.wm = WorldModel()
        self.wm.load_history()
        self.sim = Simulator(self.wm)
        self.planner = Planner(self.wm)
        self._last_plan = None
        self._episodes = self._load_episodes()

    def decide(self, tensor: list[float], immune_label: str = "nominal", immune_action: str | None = None) -> dict:
        plan = self.planner.plan(tensor, immune_label, immune_action, episodes=self._episodes)
        self._last_plan = plan
        self._persist_plan(plan)
        return plan

    def observe_transition(self, tensor_before: list, action: str | None, tensor_after: list):
        self.wm.observe(tensor_before, action, tensor_after)
        if self._last_plan:
            episode = {
                "tensor_before": tensor_before,
                "action": action,
                "tensor_after": tensor_after,
                "plan": {k: v for k, v in self._last_plan.items() if k != "simulations"},
                "timestamp": time.time(),
            }
            self._episodes.append(episode)
            self._persist_episode(episode)

    def inhibit(self) -> bool:
        if self._last_plan:
            return self._last_plan.get("inhibit", False)
        return False

    def current_plan(self) -> dict | None:
        return self._last_plan

    def load_current_plan(self) -> dict | None:
        try:
            if os.path.exists(PLAN_FILE):
                with open(PLAN_FILE) as f:
                    return json.load(f)
        except Exception:
            pass
        return None

    def _persist_plan(self, plan: dict):
        try:
            os.makedirs(os.path.dirname(PLAN_FILE), exist_ok=True)
            with open(PLAN_FILE, "w") as f:
                json.dump(plan, f, indent=2)
        except Exception:
            pass

    def _load_episodes(self) -> list:
        episodes = []
        if os.path.exists(EPISODE_FILE):
            with open(EPISODE_FILE) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        episodes.append(json.loads(line))
        return episodes[-100:]

    def _persist_episode(self, episode: dict):
        try:
            os.makedirs(os.path.dirname(EPISODE_FILE), exist_ok=True)
            with open(EPISODE_FILE, "a") as f:
                f.write(json.dumps(episode) + "\n")
        except Exception:
            pass
