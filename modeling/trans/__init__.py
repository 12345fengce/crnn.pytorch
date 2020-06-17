# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 10:59
# @Author  : zhoujun

from .TPS import TPS

__all__ = ['build_trans']
support_trans = ['TPS', 'None']


def build_trans(trans_name, **kwargs):
    assert trans_name in support_trans, f'all support head is {support_trans}'
    if trans_name == 'None':
        return None
    head = eval(trans_name)(**kwargs)
    return head
