#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/17 上午11:02
'''
import torch
from torch import nn
import torch.nn.functional as F
from models.modules.basic import ConvBnRelu

from models.modules.seg.resnet import *


class FPN(nn.Module):
    def __init__(self, backbone_out_channels, inner_channels=256):
        """
        :param backbone_out_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        inplace = True
        self.conv_out_channels = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(backbone_out_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(backbone_out_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(backbone_out_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(backbone_out_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out_channels, self.conv_out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out_channels),
            nn.ReLU(inplace=inplace)
        )
        self.out_conv =  nn.Conv2d(in_channels=self.conv_out_channels, out_channels=1, kernel_size=1)

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.reduce_conv_c5(c5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)
        x = self.out_conv(x)
        return x

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1)


class ResNetFPN(nn.Module):
    def __init__(self, backbone, pretrained, **kwargs):
        """
        PANnet
        :param model_config: 模型配置
        """
        super().__init__()
        self.k = kwargs.get('k', 1)
        backbone_dict = {
            'resnet18': {'models': resnet18, 'out': [64, 128, 256, 512]},
            'deformable_resnet18': {'models': deformable_resnet18, 'out': [64, 128, 256, 512]},
            'resnet34': {'models': resnet34, 'out': [64, 128, 256, 512]},
            'resnet50': {'models': resnet50, 'out': [256, 512, 1024, 2048]},
            'deformable_resnet50': {'models': deformable_resnet50, 'out': [256, 512, 1024, 2048]},
            'resnet101': {'models': resnet101, 'out': [256, 512, 1024, 2048]},
            'resnet152': {'models': resnet152, 'out': [256, 512, 1024, 2048]},
        }
        assert backbone in backbone_dict, 'backbone must in: {}'.format(backbone_dict)
        backbone_model, backbone_out = backbone_dict[backbone]['models'], backbone_dict[backbone]['out']
        self.backbone = backbone_model(pretrained=pretrained)
        self.segmentation_head = FPN(backbone_out)
        self.out_channels = 1

    def forward(self, x):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        y = self.segmentation_head(backbone_out)
        y = torch.sigmoid(y * self.k)
        y = F.interpolate(y, size=(H, W))
        return y
