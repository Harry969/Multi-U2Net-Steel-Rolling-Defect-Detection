import torch
import torch.nn as nn
from model.u2net import U2NETP

# 假设你有一个模型，例如 Unet
model = U2NETP(n_channels=3, n_classes=4)

# 计算模型的总参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params}")