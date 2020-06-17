# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 11:17
# @Author  : zhoujun
__all__ = ['build_loss']

from .CTCLoss import CTCLoss
from .AttnLoss import AttnLoss

support_loss = ['CTCLoss', 'AttnLoss']


def build_loss(loss_name, **kwargs):
    assert loss_name in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_name)(**kwargs)
    return criterion
