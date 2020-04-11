# -*- coding: utf-8 -*-
# @Time    : 2020/4/11 10:29
# @Author  : zhoujun
from torch import nn
from models.modules.basic import *


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.se = CBAM(out_channels) if use_cbam else None
        self.conv1 = BasicConv(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.conv2 = BasicConv(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.se != None:
            out = self.se(out)
        out += residual
        out = self.relu(out)

        return out


class ReaNet(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.out_channels = out_channels

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = BasicConv(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False, use_bn=True)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class ResNet_FeatureExtractor(ReaNet):
    """ FeatureExtractor of https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/modules/feature_extraction.py """

    def __init__(self, in_channels, out_channels=512):
        super().__init__(out_channels)
        layers = [1, 2, 5, 3]
        block = BasicBlock
        output_channel_block = [int(out_channels / 4), int(out_channels / 2), out_channels, out_channels]

        self.inplanes = int(out_channels / 8)
        self.conv0 = nn.Sequential(
            BasicConv(in_channels, int(out_channels / 16), kernel_size=3, stride=1, padding=1, bias=False, use_bn=True, use_relu=True),
            BasicConv(int(out_channels / 16), self.inplanes, kernel_size=3, stride=1, padding=1, bias=False, use_bn=True, use_relu=True)
        )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, output_channel_block[0], layers[0])
        self.conv1 = BasicConv(output_channel_block[0], output_channel_block[0], kernel_size=3, stride=1, padding=1, bias=False, use_bn=True,
                               use_relu=True)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer2 = self._make_layer(block, output_channel_block[1], layers[1], stride=1)
        self.conv2 = BasicConv(output_channel_block[1], output_channel_block[1], kernel_size=3, stride=1, padding=1, bias=False, use_bn=True,
                               use_relu=True)

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=(2, 1), padding=(0, 1))
        self.layer3 = self._make_layer(block, output_channel_block[2], layers[2], stride=1)
        self.conv3 = BasicConv(output_channel_block[2], output_channel_block[2], kernel_size=3, stride=1, padding=1, bias=False, use_bn=True,
                               use_relu=True)

        self.layer4 = self._make_layer(block, output_channel_block[3], layers[3], stride=1)
        self.conv4 = nn.Sequential(
            BasicConv(output_channel_block[3], output_channel_block[3], kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False, use_bn=True,
                      use_relu=True),
            BasicConv(output_channel_block[3], output_channel_block[3], kernel_size=2, stride=1, padding=0, bias=False, use_bn=True,
                      use_relu=True)
        )

    def forward(self, x):
        x = self.conv0(x)

        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.conv1(x)

        x = self.maxpool2(x)
        x = self.layer2(x)
        x = self.conv2(x)

        x = self.maxpool3(x)
        x = self.layer3(x)
        x = self.conv3(x)

        x = self.layer4(x)
        x = self.conv4(x)
        return x


class ResNet_MT(ReaNet):
    """ resnet of ReADS arxiv.org/pdf/2004.02070.pdf"""

    def __init__(self, in_channels, out_channels=512):
        super().__init__()
        layers = [3, 4, 6, 3]
        block = BasicBlock
        output_channel_block = [int(out_channels / 16), int(out_channels / 8), int(out_channels / 4), int(out_channels / 2), out_channels]

        self.inplanes = output_channel_block[0]
        self.conv0 = BasicConv(in_channels, output_channel_block[0], kernel_size=3, stride=1, padding=1, bias=False, use_bn=True, use_relu=True)

        self.layer1 = self._make_layer(block, output_channel_block[1], layers[0])
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.layer2 = self._make_layer(block, output_channel_block[2], layers[1], stride=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.layer3 = self._make_layer(block, output_channel_block[3], layers[2], stride=1)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))

        self.layer4 = self._make_layer(block, output_channel_block[4], layers[3], stride=1)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.conv4 = BasicConv(output_channel_block[4], out_channels, kernel_size=2, bias=False,
                               use_bn=True, use_relu=True)

    def forward(self, x):
        x = self.conv0(x)

        x = self.layer1(x)
        x = self.maxpool1(x)

        x = self.layer2(x)
        x = self.maxpool2(x)

        x = self.layer3(x)
        x = self.maxpool3(x)

        x = self.layer4(x)
        x = self.maxpool4(x)
        x = self.conv4(x)
        return x


if __name__ == '__main__':
    import torch
    net = ResNet_FeatureExtractor(3, 512)
    x = torch.rand((1, 3, 32, 100))
    y = net(x)
    print(y.shape)
