# -*- coding: utf-8 -*-
# @Time    : 2019/12/20 11:14
# @Author  : zhoujun
import math
import torch
from torch import nn

__all__ = ['BasicConv', 'BasicBlockV2', 'DWConv', 'DWBlock', 'SELayer', 'GhostModule', 'GhostBottleneck']


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', use_bn=True,
                 use_relu=True, inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else None
        self.relu = nn.ReLU(inplace=inplace) if use_relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=3, downsample=True,use_se=False, **kwargs):
        super(BasicBlockV2, self).__init__()
        self.se = SELayer(out_channels) if use_se else None
        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.relu1 = nn.ReLU()
        self.conv = nn.Sequential(
            BasicConv(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
            BasicConv(out_channels, out_channels, kernel_size, 1, kernel_size // 2, bias=False,use_bn=False,use_relu=False),
        )
        if downsample:
            self.downsample = BasicConv(in_channels, out_channels, 1, stride, bias=False, use_bn=False, use_relu=False)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv(x)
        if self.se != None:
            x = self.se(x)
        return x + residual


class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=False):
        super().__init__()
        self.use_bn = use_bn
        self.DWConv = BasicConv(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, use_bn=use_bn)
        self.conv1x1 = BasicConv(in_channels, out_channels, 1, 1, 0, use_bn=use_bn)

    def forward(self, x):
        x = self.DWConv(x)
        x = self.conv1x1(x)
        return x


class DWBlock(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, in_channels, out_channels, expand_size, kernel_size, stride, use_se=False):
        super().__init__()
        self.stride = stride
        self.se = SELayer(out_channels) if use_se else None
        # pw
        self.conv1 = BasicConv(in_channels, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        # dw
        self.conv2 = BasicConv(expand_size, expand_size, kernel_size, stride, padding=kernel_size // 2, groups=expand_size)
        # pw
        self.conv3 = BasicConv(expand_size, out_channels, kernel_size=1, stride=1, padding=0, bias=False, use_relu=False)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_channels != out_channels:
            self.shortcut = BasicConv(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, use_relu=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel), )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        y = torch.clamp(y, 0, 1)
        return x * y


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, **kwargs):
        super().__init__()
        self.oup = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = BasicConv(in_channels, init_channels, kernel_size, stride, kernel_size // 2, use_relu=relu)

        self.cheap_operation = BasicConv(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False, use_relu=relu)

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expand_size, kernel_size, stride, use_se=False):
        super().__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(in_channels, expand_size, kernel_size=1, relu=True),
            # dw
            BasicConv(expand_size, expand_size, kernel_size, stride, kernel_size // 2, use_relu=False) if stride == 2 else nn.Sequential(),
            # Squeeze-and-Excite
            SELayer(expand_size) if use_se else nn.Sequential(),
            # pw-linear
            GhostModule(expand_size, out_channels, kernel_size=1, relu=False),
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                BasicConv(in_channels, in_channels, 3, stride, kernel_size // 2, use_relu=True),
                BasicConv(in_channels, out_channels, 1, 1, 0, bias=False, use_relu=False)
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)
