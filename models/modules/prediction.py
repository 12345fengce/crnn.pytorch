#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/19 下午2:56
'''

from torch import nn


class CTC(nn.Module):
    def __init__(self, in_channels, n_class, **kwargs):
        super().__init__()
        self.fc = nn.Linear(in_channels, n_class)

    def forward(self, x):
        return self.fc(x)
