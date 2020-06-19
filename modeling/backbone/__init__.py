# -*- coding: utf-8 -*-
# @Time    : 2020/4/25 12:06
# @Author  : zhoujun

from .feature_extraction import CNN_lite, VGG, ResNet, DenseNet
from .resnet import ResNet_FeatureExtractor, ResNet_MT
from .resnet_torch import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2
from .MobileNetV3 import MobileNetV3

__all__ = ['build_backbone']
support_backbone = ['CNN_lite', 'VGG', 'ResNet', 'DenseNet',
                    'ResNet_FeatureExtractor', 'ResNet_MT',
                    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
                    'MobileNetV3']


def build_backbone(backbone_name, **kwargs):
    assert backbone_name in support_backbone, f'all support backbone is {support_backbone}'
    backbone = eval(backbone_name)(**kwargs)
    return backbone
