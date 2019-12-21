import torch
from torch import nn
from models.modules import *


def init_modules(config, module_name, canbe_none=True, **kwargs):
    if module_name not in config:
        return None, None
    module_config = config[module_name]
    module_type = module_config['type']
    if len(module_type) == 0:
        return None, None
    if 'args' not in module_config or module_config['args'] is None:
        module_args = {}
    else:
        module_args = module_config['args']
    module_args.update(**kwargs)
    if canbe_none:
        try:
            module = eval(module_type)(**module_args)
        except:
            module = None
    else:
        module = eval(module_type)(**module_args)
    return module, module_type


class Model(nn.Module):
    def __init__(self, in_channels, n_class, config):
        super(Model, self).__init__()

        # 二值分割网络
        self.binarization, self.binarization_type = init_modules(config, 'binarization', canbe_none=True, in_channels=in_channels)

        # 特征提取模型设置
        if self.binarization is not None:
            in_channels = self.binarization.out_channels
        self.feature_extraction, self.feature_extraction_type = init_modules(config, 'feature_extraction', canbe_none=False, in_channels=in_channels)
        in_channels = self.feature_extraction.out_channels
        # 序列模型
        self.sequence_model, self.sequence_model_type = init_modules(config, 'sequence_model', canbe_none=True, in_channels=in_channels)

        # 预测设置
        if self.sequence_model is not None:
            in_channels = self.sequence_model.out_channels
        self.prediction, self.prediction_type = init_modules(config, 'prediction', canbe_none=False, in_channels=in_channels, n_class=n_class)

        self.model_name = '{}_{}_{}_{}'.format(self.binarization_type, self.feature_extraction_type, self.sequence_model_type, self.prediction_type)
        self.batch_max_length = -1

    def get_batch_max_length(self, x):
        # 特征提取阶段
        if self.binarization is not None:
            x = self.binarization(x)
        visual_feature = self.feature_extraction(x)
        self.batch_max_length = visual_feature.shape[-1]
        return self.batch_max_length

    def forward(self, x):
        if self.binarization is not None:
            x = self.binarization(x)
        # 特征提取阶段
        visual_feature = self.feature_extraction(x)
        # 序列建模阶段
        if self.sequence_model is not None:
            contextual_feature = self.sequence_model(visual_feature)
        else:
            contextual_feature = visual_feature.squeeze(axis=2).permute((0, 2, 1))
        # 预测阶段
        if self.prediction_type == 'CTC':
            prediction = self.prediction(contextual_feature)
        else:
            raise NotImplementedError
        return prediction, x


if __name__ == '__main__':
    import os
    import anyconfig
    import numpy as np
    from utils import parse_config

    config = anyconfig.load(open("config/icdar2015_win.yaml", 'rb'))
    if 'base' in config:
        config = parse_config(config)
    if os.path.isfile(config['dataset']['alphabet']):
        config['dataset']['alphabet'] = str(np.load(config['dataset']['alphabet']))
    net = Model(3, len(config['dataset']['alphabet']), config['arch']['args'])
    print(net.model_name, len(config['dataset']['alphabet']))
    a = torch.zeros(2, 3, 32, 320)
    print(net.get_batch_max_length(a))
    b, _ = net(a)
    print(b.shape)
