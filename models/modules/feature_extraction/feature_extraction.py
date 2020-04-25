from torch import nn
from models.modules.basic import *
from torchvision.models.densenet import _DenseBlock


class CNN_lite(nn.Module):
    # a vgg like net
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        channels = [24, 128, 256, 256, 512, 512, 512]
        self.out_channels = channels[-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=5, stride=2, padding=2),
            nn.ReLU(True),
            DWConv(channels[0], channels[1], kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            DWConv(channels[1], channels[2], kernel_size=3, stride=1, padding=1, use_bn=True),
            DWConv(channels[2], channels[3], kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            DWConv(channels[3], channels[4], kernel_size=3, stride=1, padding=1, use_bn=True),
            DWConv(channels[4], channels[5], kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            DWConv(channels[5], channels[6], kernel_size=2, stride=1, padding=0, use_bn=True),
        )

    def forward(self, x):
        conv = self.cnn(x)
        return conv


class VGG(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        conv_type = kwargs.get('conv_type', 'BasicConv')
        assert conv_type in ['BasicConv', 'DWConv', 'GhostModule']
        basic_conv = globals()[conv_type]
        channels = [64, 128, 256, 256, 512, 512, 512]
        self.features = nn.Sequential(
            # conv layer
            BasicConv(in_channels=in_channels, out_channels=channels[0], kernel_size=3, padding=1, use_bn=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # second conv layer
            basic_conv(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1, use_bn=False),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # third conv layer
            basic_conv(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1, use_bn=False),

            # fourth conv layer
            basic_conv(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1, use_bn=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # fifth conv layer
            basic_conv(in_channels=channels[3], out_channels=channels[4], kernel_size=3, padding=1, bias=False),

            # sixth conv layer
            basic_conv(in_channels=channels[4], out_channels=channels[5], kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            # seren conv layer
            BasicConv(in_channels=channels[5], out_channels=channels[6], kernel_size=2, use_bn=False, use_relu=True),
        )
        self.out_channels = channels[-1]

    def forward(self, x):
        return self.features(x)


class ResNet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        conv_type = kwargs.get('conv_type', 'BasicBlockV2')
        assert conv_type in ['BasicBlockV2', 'DWBlock', 'GhostBottleneck']

        BasicBlock = globals()[conv_type]

        channels = [64, 64, 64, 128, 128, 256, 256, 512, 512, 512]
        expand_size = [64, 64, 128, 128, 256]
        self.out_channels = channels[-1]

        self.features = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=channels[0], kernel_size=3, padding=1, bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=2, stride=2, bias=False),

            BasicBlock(in_channels=channels[0], out_channels=channels[2], expand_size=expand_size[0], kernel_size=3,
                       stride=1),
            BasicBlock(in_channels=channels[2], out_channels=channels[3], expand_size=expand_size[1], kernel_size=3,
                       stride=1),
            nn.Dropout(0.2),

            BasicBlock(in_channels=channels[3], out_channels=channels[4], expand_size=expand_size[2], kernel_size=3,
                       stride=2, use_se=True),
            BasicBlock(in_channels=channels[4], out_channels=channels[5], expand_size=expand_size[3], kernel_size=3,
                       stride=1, use_se=True),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=channels[5], out_channels=channels[6], kernel_size=2, stride=(2, 1), padding=(0, 1),
                      bias=False),

            BasicBlock(in_channels=channels[6], out_channels=channels[7], expand_size=expand_size[4], kernel_size=3,
                       stride=1, use_se=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=(2,1)),
            BasicConv(in_channels=channels[7], out_channels=channels[8], kernel_size=3, padding=0, bias=False),
            BasicConv(in_channels=channels[8], out_channels=channels[9], kernel_size=2, padding=(0, 1), bias=False),
        )

    def forward(self, x):
        return self.features(x)


class ResNet_MT(nn.Module):
    """
    美团论文ReADS里的resnet
    """

    def __init__(self, in_channels, **kwargs):
        super().__init__()
        conv_type = kwargs.get('conv_type', 'BasicBlockV2')
        assert conv_type in ['BasicBlockV2', 'DWBlock', 'GhostBottleneck']

        BasicBlock = globals()[conv_type]

        channels = [32, 64, 64, 128, 128, 256, 256, 512, 512, 512]
        expand_size = [64, 64, 128, 128, 256]
        self.out_channels = channels[-1]

        self.features = nn.Sequential(
            BasicConv(in_channels=in_channels, out_channels=channels[0], kernel_size=3, padding=1, bias=False),

            BasicBlock(in_channels=channels[0], out_channels=channels[2], expand_size=expand_size[0], kernel_size=3,
                       stride=1, use_cbam=True, downsample=False),
            BasicBlock(in_channels=channels[2], out_channels=channels[3], expand_size=expand_size[1], kernel_size=3,
                       stride=1, use_cbam=True, downsample=False),
            nn.Dropout(0.2),

            BasicBlock(in_channels=channels[3], out_channels=channels[4], expand_size=expand_size[2], kernel_size=3,
                       stride=2, use_cbam=True, downsample=False),
            BasicBlock(in_channels=channels[4], out_channels=channels[5], expand_size=expand_size[3], kernel_size=3,
                       stride=1, use_cbam=True, downsample=False),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=channels[5], out_channels=channels[6], kernel_size=2, stride=(2, 1), padding=(0, 1),
                      bias=False),

            BasicBlock(in_channels=channels[6], out_channels=channels[7], expand_size=expand_size[4], kernel_size=3,
                       stride=1, use_cbam=True, downsample=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=(2,1)),
            BasicConv(in_channels=channels[7], out_channels=channels[8], kernel_size=3, padding=0, bias=False),
            BasicConv(in_channels=channels[8], out_channels=channels[9], kernel_size=2, padding=(0, 1), bias=False),
        )

    def forward(self, x):
        return self.features(x)


def _make_transition(in_channels, out_channels, pool_stride, pool_pad, dropout):
    out = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    )
    if dropout:
        out.add_module('dropout', nn.Dropout(dropout))
    out.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=pool_stride, padding=pool_pad))
    return out


class DenseNet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2, stride=2, bias=False),
            _DenseBlock(8, 64, 4, 8, 0),
            _make_transition(128, 128, 2, 0, 0.2),

            _DenseBlock(8, 128, 4, 8, 0),
            _make_transition(192, 128, (2, 1), 0, 0.2),

            _DenseBlock(8, 128, 4, 8, 0),

            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.out_channels = 768

    def forward(self, x):
        x = self.features(x)
        B, C, H, W = x.shape
        x = x.reshape((B, C * H, 1, W))
        return x


if __name__ == '__main__':
    import torch

    device = torch.device('cpu')
    net = VGG(3).to(device)
    a = torch.randn(2, 3, 32, 320).to(device)
    import time

    tic = time.time()
    for i in range(1):
        b = net(a)[0]
    print(b.shape)
    print((time.time() - tic) / 1)
