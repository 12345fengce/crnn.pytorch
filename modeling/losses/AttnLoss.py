# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 15:03
# @Author  : zhoujun

from torch import nn


class AttnLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, preds, batch_data):
        target = batch_data['targets'][:, 1:]  # without [GO] Symbol
        loss = self.func(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))
        return loss
