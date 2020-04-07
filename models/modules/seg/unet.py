#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/17 下午1:51
'''
import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBlock(in_channels, out_channels, 3),
            ConvBlock(out_channels, out_channels, 3)
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, shrink=True, **kwargs):
        super().__init__()
        self.conv3_0 = ConvBlock(in_channels, out_channels, 3)
        if shrink:
            self.conv3_1 = ConvBlock(out_channels, int(out_channels / 2), 3)
        else:
            self.conv3_1 = ConvBlock(out_channels, out_channels, 3)

    def forward(self, x, s):
        x = F.interpolate(x, scale_factor=2)

        x = torch.cat([s, x], dim=1)
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.k = kwargs.get('k', 1)
        self.stage_channels = [32, 64, 128, 256, 512]
        self.d0 = DownBlock(in_channels, self.stage_channels[0])

        self.d1 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True), DownBlock(self.stage_channels[0], self.stage_channels[1]))

        self.d2 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True), DownBlock(self.stage_channels[1], self.stage_channels[2]))

        self.d3 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True), DownBlock(self.stage_channels[2], self.stage_channels[3]))

        self.d4 = nn.Sequential(nn.MaxPool2d(2, 2, ceil_mode=True), DownBlock(self.stage_channels[3], self.stage_channels[4]))

        self.u3 = UpBlock(self.stage_channels[3] + self.stage_channels[4], self.stage_channels[3], shrink=True)
        self.u2 = UpBlock(self.stage_channels[3], self.stage_channels[2], shrink=True)
        self.u1 = UpBlock(self.stage_channels[2], self.stage_channels[1], shrink=True)
        self.u0 = UpBlock(self.stage_channels[1], self.stage_channels[0], shrink=False)

        self.conv = nn.Conv2d(self.stage_channels[0], 1, 1, bias=False)
        self.out_channels = 1

    def forward(self, x):
        x0 = self.d0(x)
        x1 = self.d1(x0)
        x2 = self.d2(x1)
        x3 = self.d3(x2)
        x4 = self.d4(x3)

        y3 = self.u3(x4, x3)
        y2 = self.u2(y3, x2)
        y1 = self.u1(y2, x1)
        y0 = self.u0(y1, x0)
        out = self.conv(y0)
        out = torch.sigmoid(out * self.k)
        return out


if __name__ == '__main__':

    x = torch.zeros(1, 3, 32, 320)
    net = UNet(3)
    y = net(x)
    print(y.shape)
