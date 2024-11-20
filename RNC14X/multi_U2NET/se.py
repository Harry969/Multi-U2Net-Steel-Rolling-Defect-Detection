import torch
import torch.nn as nn
import math

class se_block(nn.Module):
    def __init__(self, channel, ratio=2):
        super(se_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Sequential(
                nn.Linear(channel, channel // ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channel // ratio, channel, bias=False),
                nn.Sigmoid())
        self.fc_2 = nn.Sequential(
                nn.Conv2d(channel, channel // ratio, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // ratio, channel, 1, bias=False),
                nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc_1(y).view(b, c, 1, 1)
        return x * y


# model = se_block(512)
# print(model)
# inputs = torch.randn([2, 512, 26, 26])
# outputs = model(inputs)


model = se_block(3)
print(model)
inputs = torch.randn([8, 3, 488, 488])
outputs = model(inputs)

