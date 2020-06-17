import copy
from addict import Dict
import torch
from torch import nn

from modeling.trans import build_trans
from modeling.backbone import build_backbone
from modeling.neck import build_neck
from modeling.head import build_head


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        model_config = copy.deepcopy(config)
        model_config = Dict(model_config)

        trans_type = model_config.trans.pop('type')
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type')
        self.head_type = model_config.head.pop('type')
        if self.head_type == 'Attention':
            assert neck_type == 'RNNDecoder'
        self.trans = build_trans(trans_type, **model_config.backbone)
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(self.head_type, in_channels=self.neck.out_channels, **model_config.head)

        self.name = f'RecModel_{trans_type}_{backbone_type}_{neck_type}_{self.head_type}'
        self.batch_max_length = -1
        self.init()

    def get_batch_max_length(self, x):
        # 特征提取阶段
        if self.trans is not None:
            x = self.trans(x)
        x = self.backbone(x)
        self.batch_max_length = x.shape[-1]
        return self.batch_max_length

    def init(self):
        import torch.nn.init as init
        # weight initialization
        for name, param in self.named_parameters():
            if 'localization_fc2' in name:
                print(f'Skip {name} as it is already initialized')
                continue
            try:
                if 'bias' in name:
                    init.constant_(param, 0.0)
                elif 'weight' in name:
                    init.kaiming_normal_(param)
            except Exception as e:  # for batchnorm.
                if 'weight' in name:
                    param.data.fill_(1)
                continue

    def forward(self, x, text=None):
        if self.trans is not None:
            x = self.trans(x)
        y = self.backbone(x)
        y = self.neck(y)
        # 预测阶段
        if self.head_type == 'CTC':
            y = self.head(y)
        elif self.head_type == 'Attention':
            y = self.head(y, text, self.batch_max_length)
        else:
            raise NotImplementedError
        return y, x


if __name__ == '__main__':
    import os
    import anyconfig
    from utils import parse_config, load, get_parameter_number

    config = anyconfig.load(open("config/imagedataset_None_VGG_RNN_CTC.yaml", 'rb'))
    if 'base' in config:
        config = parse_config(config)
    if os.path.isfile(config['dataset']['alphabet']):
        config['dataset']['alphabet'] = load(config['dataset']['alphabet'])

    device = torch.device('cpu')
    config['arch']['backbone']['in_channels'] = 3
    config['arch']['head']['n_class'] = 95
    net = Model(config['arch']).to(device)
    print(net.name, len(config['dataset']['alphabet']))
    a = torch.randn(2, 3, 32, 320).to(device)

    import time

    text_for_pred = torch.LongTensor(2, 25 + 1).fill_(0)
    tic = time.time()
    for i in range(1):
        b = net(a, text_for_pred)[0]
    print(b.shape)
    print((time.time() - tic) / 1)
    print(get_parameter_number(net))
