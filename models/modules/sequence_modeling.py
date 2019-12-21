from torch import nn

from models.modules.basic import ConvBnRelu


class BidirectionalGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nOut):
        super(BidirectionalGRU, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, nOut)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.fc(x)  # [T * b, nOut]
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True, batch_first=True)

    def forward(self, x):
        x, _ = self.rnn(x)
        return x


class RNNDecoder(nn.Module):
    def __init__(self, in_channels, hidden_size=256, num_layers=1):
        super(RNNDecoder, self).__init__()
        self.lstm = nn.Sequential(
            BidirectionalLSTM(in_channels, hidden_size, num_layers),
            BidirectionalLSTM(hidden_size * 2, hidden_size, num_layers)
        )
        self.out_channels = hidden_size * 2

    def forward(self, x):
        x = x.squeeze(axis=2)
        x = x.permute((0, 2, 1))  # (NTC)(batch, width, channel)s
        x = self.lstm(x)
        return x


class CNNDecoder(nn.Module):
    def __init__(self, in_channels, hidden_size=256):
        super().__init__()
        self.cnn_decoder = nn.Sequential(
            ConvBnRelu(in_channels=in_channels, out_channels=hidden_size, kernel_size=3, padding=1, stride=(2, 1), bias=False),
            ConvBnRelu(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1, stride=(2, 1), bias=False),
            ConvBnRelu(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1, stride=(2, 1), bias=False),
            ConvBnRelu(in_channels=hidden_size, out_channels=hidden_size, kernel_size=3, padding=1, stride=(2, 1), bias=False)
        )
        self.out_channels = hidden_size

    def forward(self, x):
        x = self.cnn_decoder(x)
        x = x.squeeze(dim=2)
        x = x.permute(0, 2, 1)
        return x
