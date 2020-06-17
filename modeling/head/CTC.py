# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 10:57
# @Author  : zhoujun
from torch import nn


class CTC(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, n_class)

    def forward(self, x):
        return self.fc(x)
