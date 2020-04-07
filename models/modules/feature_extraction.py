from torch import nn
from models.modules.basic import ConvBnRelu
from torchvision.models.densenet import _DenseBlock

class CNN_lite(nn.Module):

    def __init__(self, in_channels):
        """
        是否加入lstm特征层
        """
        super().__init__()

        ks = [5, 3, 3, 3, 3, 3, 2]
        ps = [2, 1, 1, 1, 1, 1, 0]
        # ss = [1, 1, 1, 1, 1, 1, 1]
        ss = [2, 1, 1, 1, 1, 1, 1]
        nm = [24, 128, 256, 256, 512, 512, 512]
        # nm = [32, 64, 128, 128, 256, 256, 256]
        # exp_ratio = [2,2,2,2,1,1,2]
        cnn = nn.Sequential()
        self.out_channels = nm[-1]
        def convRelu(i, batchNormalization=False):
            nIn = in_channels if i == 0 else nm[i - 1]
            nOut = nm[i]
            # exp  = exp_ratio[i]
            # exp_num = exp * nIn
            if i == 0:
                cnn.add_module('conv_{0}'.format(i),
                               nn.Conv2d(nIn , nOut , ks[i], ss[i], ps[i]))
                cnn.add_module('relu_{0}'.format(i), nn.ReLU(True))
            else:

                cnn.add_module('conv{0}'.format(i),
                               nn.Conv2d( nIn,  nIn, ks[i], ss[i], ps[i],groups=nIn))
                if batchNormalization:
                    cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nIn))
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

                cnn.add_module('convproject{0}'.format(i),
                               nn.Conv2d(nIn, nOut, 1, 1, 0))
                if batchNormalization:
                    cnn.add_module('batchnormproject{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))




        convRelu(0)
        # cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)

        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16

        # cnn.add_module('pooling{0}'.format(2),
        #                nn.MaxPool2d((2, 2))) # 256x4x16

        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16

        # cnn.add_module('pooling{0}'.format(3),
        #                nn.MaxPool2d((2, 2))) # 256x4x16

        convRelu(6, True)  # 512x1x16

        self.cnn = cnn

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        return conv

class VGG(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            # conv layer
            ConvBnRelu(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # second conv layer
            ConvBnRelu(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            # third conv layer
            ConvBnRelu(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),

            # fourth conv layer
            ConvBnRelu(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # fifth conv layer
            ConvBnRelu(in_channels=256, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),

            # sixth conv layer
            ConvBnRelu(in_channels=512, out_channels=512, kernel_size=(3, 3), padding=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),

            # seren conv layer
            ConvBnRelu(in_channels=512, out_channels=512, kernel_size=(2, 2)),
        )
        self.out_channels = 512

    def forward(self, x):
        return self.features(x)


class BasicBlockV2(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=False):
        super(BasicBlockV2, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)
        self.relu1 = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        )
        if downsample:
            self.downsample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
        else:
            self.downsample = None

    def forward(self, x):
        residual = x
        x = self.bn1(x)
        x = self.relu1(x)
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv(x)

        return x + residual


class ResNet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.features = nn.Sequential(
            ConvBnRelu(in_channels=in_channels, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, bias=False),

            BasicBlockV2(in_channels=64, out_channels=64, stride=1, downsample=True),
            BasicBlockV2(in_channels=64, out_channels=128, stride=1, downsample=True),
            nn.Dropout(0.2),

            BasicBlockV2(in_channels=128, out_channels=128, stride=2, downsample=True),
            BasicBlockV2(in_channels=128, out_channels=256, stride=1, downsample=True),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=(2, 1), padding=(0, 1), bias=False),

            BasicBlockV2(in_channels=256, out_channels=512, stride=1, downsample=True),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            ConvBnRelu(in_channels=512, out_channels=1024, kernel_size=3, padding=0, bias=False),
            ConvBnRelu(in_channels=1024, out_channels=2048, kernel_size=2, padding=(0, 1), bias=False),
        )
        self.out_channels = 2048

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

    x = torch.zeros(1, 3, 32, 320)
    net = VGG(3)
    y = net(x)
    print(y.shape)
