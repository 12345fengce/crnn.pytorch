# -*- coding: utf-8 -*-
# @Time    : 2019/7/11 10:05
# @Author  : zhoujun

from .seg import UNet, ResNetFPN
from .transformation import TPS_SpatialTransformerNetwork as TPS
from .feature_extraction import *
from .sequence_modeling import RNNDecoder, CNNDecoder
from .prediction import CTC, Attention

model_dict = {
    'transformation': {
        "TPS": TPS,
        'UNet': UNet,
        'ResNetFPN': ResNetFPN,
    },
    'feature_extraction': {
        'VGG': VGG,
        'ResNet': ResNet,
        'ResNet_MT': ResNet_MT,
        'ResNet_FeatureExtractor': ResNet_FeatureExtractor,
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'resnext50_32x4d': resnext50_32x4d,
        'resnext101_32x8d': resnext101_32x8d,
        'wide_resnet50_2': wide_resnet50_2,
        'wide_resnet101_2': wide_resnet101_2,
    },
    'sequence_model': {
        'RNN': RNNDecoder,
        'CNN': CNNDecoder,
    },
    'prediction': {
        'CTC': CTC,
        'Attn': Attention
    }
}
