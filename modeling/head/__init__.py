# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 10:57
# @Author  : zhoujun

from .CTC import CTC
from .Attn import Attn

__all__ = ['build_head']
support_head = ['CTC', 'Attn']


def build_head(head_name, **kwargs):
    assert head_name in support_head, f'all support head is {support_head}'
    head = eval(head_name)(**kwargs)
    return head
