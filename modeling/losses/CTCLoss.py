# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 11:17
# @Author  : zhoujun
import torch
from torch import nn


class CTCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.func = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, preds, batch_data):
        cur_batch_size = batch_data['img'].shape[0]
        targets = batch_data['targets']
        targets_lengths = batch_data['targets_lengths']
        preds = preds.log_softmax(2)
        preds_lengths = torch.tensor([preds.size(1)] * cur_batch_size, dtype=torch.long)
        loss = self.func(preds.permute(1, 0, 2), targets, preds_lengths, targets_lengths)
        return loss
