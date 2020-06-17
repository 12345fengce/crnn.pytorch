# -*- coding: utf-8 -*-
# @Time    : 2020/6/17 10:49
# @Author  : zhoujun

from .sequence_modeling import RNNDecoder, CNNDecoder, Reshape

__all__ = ['build_neck']

support_neck = ['RNNDecoder', 'CNNDecoder', 'Reshape']


def build_neck(neck_name, **kwargs):
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck
