from dataclasses import dataclass, asdict, field
import time

@dataclass
class SWAEvent:
    sender: str
    intent: str
    data: dict
    timestamp: float = field(default_factory=time.time)

    def to_json(self):
        return asdict(self)
