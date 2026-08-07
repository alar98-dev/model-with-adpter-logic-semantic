import inspect
from trl import SFTConfig, SFTTrainer

print("SFTConfig params:", list(inspect.signature(SFTConfig).parameters.keys()))
print("SFTTrainer params:", list(inspect.signature(SFTTrainer).parameters.keys()))
