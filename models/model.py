import torch
from torch import nn
from models.modules import model_dict


def init_modules(config, stage, **kwargs):
    if stage not in model_dict:
        return None, None
    stage_model_dict = model_dict[stage]
    module_config = config[stage]
    module_type = module_config['type']
    if len(module_type) == 0 or module_type == 'None':
        return None, None

    assert module_type in stage_model_dict, f'{stage} must in {stage_model_dict.keys()} or empty str or "None"'
    if 'args' not in module_config or module_config['args'] is None:
        module_args = {}
    else:
        module_args = module_config['args']
    module_args.update(**kwargs)
    module = stage_model_dict[module_type](**module_args)
    return module, module_type


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, a=-0.07, b=0.07)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class Model(nn.Module):
    def __init__(self, in_channels, n_class, config):
        super(Model, self).__init__()

        # 图像变换阶段
        self.transformation, self.transformation_type = init_modules(config, 'transformation', in_channels=in_channels)

        # 特征提取模型设置
        if self.transformation is not None:
            in_channels = self.transformation.out_channels
        self.feature_extraction, self.feature_extraction_type = init_modules(config, 'feature_extraction', in_channels=in_channels)
        in_channels = self.feature_extraction.out_channels
        # 序列模型
        self.sequence_model, self.sequence_model_type = init_modules(config, 'sequence_model', in_channels=in_channels)

        # 预测设置
        if self.sequence_model is not None:
            in_channels = self.sequence_model.out_channels
        arg = {}
        if config['prediction']['type'] == 'Attn':
            assert self.sequence_model_type == 'RNN', 'attention predict must be used with RNN sequence_model'
            arg = {'hidden_size': config['sequence_model']['args']['hidden_size']}
        self.prediction, self.prediction_type = init_modules(config, 'prediction', in_channels=in_channels, n_class=n_class, **arg)

        self.model_name = '{}_{}_{}_{}'.format(self.transformation_type, self.feature_extraction_type, self.sequence_model_type, self.prediction_type)
        self.batch_max_length = -1
        self.apply(weights_init)

    def get_batch_max_length(self, x):
        # 特征提取阶段
        if self.transformation is not None:
            x = self.transformation(x)
        visual_feature = self.feature_extraction(x)
        self.batch_max_length = visual_feature.shape[-1]
        return self.batch_max_length

    def forward(self, x, text=None):
        if self.transformation is not None:
            x = self.transformation(x)
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
        elif self.prediction_type == 'Attn':
            prediction = self.prediction(contextual_feature, text,self.batch_max_length)
        else:
            raise NotImplementedError
        return prediction, x


if __name__ == '__main__':
    import os
    import anyconfig
    from utils import parse_config, load, get_parameter_number

    config = anyconfig.load(open("config/imagedataset_TPS_VGG_RNN_Attn.yaml", 'rb'))
    if 'base' in config:
        config = parse_config(config)
    if os.path.isfile(config['dataset']['alphabet']):
        config['dataset']['alphabet'] = load(config['dataset']['alphabet'])

    device = torch.device('cpu')
    net = Model(3, 95, config['arch']['args']).to(device)
    print(net.model_name, len(config['dataset']['alphabet']))
    a = torch.randn(2, 3, 32, 320).to(device)

    import time

    text_for_pred = torch.LongTensor(2, 25 + 1).fill_(0)
    tic = time.time()
    for i in range(1):
        b = net(a, text_for_pred)[0]
    print(b.shape)
    print((time.time() - tic) / 1)
    print(get_parameter_number(net))
    torch.save(net.state_dict(), '1.pth')
