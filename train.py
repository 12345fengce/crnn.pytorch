# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun
import os


def main(config):
    import torch
    from torch.nn import CTCLoss

    from models import get_model
    from data_loader import get_dataloader
    from trainer import Trainer
    from utils import CTCLabelConverter, AttnLabelConverter, load

    if os.path.isfile(config['dataset']['alphabet']):
        config['dataset']['alphabet'] = ''.join(load(config['dataset']['alphabet']))

    prediction_type = config['arch']['args']['prediction']['type']

    # loss 设置
    if prediction_type == 'CTC':
        criterion = CTCLoss(blank=0, zero_infinity=True)
        converter = CTCLabelConverter(config['dataset']['alphabet'])
    elif prediction_type == 'Attn':
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        converter = AttnLabelConverter(config['dataset']['alphabet'])
    else:
        raise NotImplementedError
    config['dataset']['alphabet'] = converter.character
    img_channel = 3 if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
    model = get_model(img_channel, len(config['dataset']['alphabet']), config['arch']['args'])

    img_h, img_w = 32, 100
    for process in config['dataset']['train']['dataset']['args']['pre_processes']:
        if process['type'] == "Resize":
            img_h = process['args']['img_h']
            img_w = process['args']['img_w']
            break
    sample_input = torch.zeros((2, img_channel, img_h, img_w))
    num_label = model.get_batch_max_length(sample_input)
    train_loader = get_dataloader(config['dataset']['train'], num_label)
    assert train_loader is not None
    if 'validate' in config['dataset'] and config['dataset']['validate']['dataset']['args']['data_path'][0] is not None:
        validate_loader = get_dataloader(config['dataset']['validate'], num_label)
    else:
        validate_loader = None

    trainer = Trainer(config=config, model=model, criterion=criterion, train_loader=train_loader, validate_loader=validate_loader, sample_input=sample_input,
                      converter=converter)
    trainer.train()


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='crnn.pytorch')
    parser.add_argument('--config_file', default='config/imagedataset_None_VGG_RNN_CTC.yaml', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import sys
    import anyconfig

    project = 'crnn.pytorch'  # 工作项目根目录
    sys.path.append(os.getcwd().split(project)[0] + project)

    from utils import parse_config

    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in config['trainer']['gpus']])
    main(config)
