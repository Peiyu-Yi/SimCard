#!/usr/bin/env python
# encoding: utf-8
"""
@author: yipeiyu
@time: 2023/12/29 12:16
@desc:
"""
import torch
import torch.nn as nn
from torchsummary import summary

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 32 * 32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

model = SimpleModel()

summary(model, (3, 32, 32))  # 输入张量的形状为(3, 32, 32)

def calculate_model_memory(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_MB = total_size_bytes / (1024**2)

    return total_params, total_size_MB

# 计算模型参数数量和内存大小
total_params, total_size_MB = calculate_model_memory(model)

print(f"Total parameters: {total_params}")
print(f"Total model size: {total_size_MB:.2f} MB")
