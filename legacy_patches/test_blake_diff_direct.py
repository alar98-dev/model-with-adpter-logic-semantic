import torch
import numpy as np
from swa.blake_tools import auto_blake_diff
estado = np.random.randn(1152).tolist()
delta = auto_blake_diff('rooms', estado)
print('Delta type:', type(delta))
if delta:
    print('Delta length:', len(delta))
    try:
        t = torch.tensor(delta, dtype=torch.bfloat16)
        print('Tensor created successfully:', t.shape)
    except Exception as e:
        print('Tensor error:', e)
