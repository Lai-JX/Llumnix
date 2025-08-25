import torch
import time
# cache_device = torch.device(f"cuda:0")
# test_tensor = torch.tensor([1.0]*10000, device=cache_device)
# time.sleep(90)

import cupy
with cupy.cuda.Device(0):  # 设备切换通过上下文管理器实现
    test_array = cupy.ones((10000,), dtype=cupy.float32)
time.sleep(90)