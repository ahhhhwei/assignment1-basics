import numpy as np
import torch

def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length # 最大允许起点
    start_indices = np.random.randint(0, max_start, size=batch_size) # 在[0, max_start) 中随机选 batch_size 个起点
    x = np.stack([dataset[i : i + context_length] for i in start_indices]) # 构造输入 x     np.stack() 是把多个一维数组堆成二维数组
    y = np.stack([dataset[i + 1 : i + context_length + 1] for i in start_indices]) # 构造目标 y
    # 转成 PyTorch Tensor
    x = torch.tensor(x, dtype=torch.long, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, y