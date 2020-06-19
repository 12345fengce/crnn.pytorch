# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 11:17
# @Author  : zhoujun
import copy
from .CTCLoss import CTCLoss
from .AttnLoss import AttnLoss

__all__ = ['build_loss']
support_loss = ['CTCLoss', 'AttnLoss']


def build_loss(config):
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_type)(**copy_config)
    return criterion
