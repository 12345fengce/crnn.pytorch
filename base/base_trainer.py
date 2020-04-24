# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun

import os
import shutil
import pathlib
from pprint import pformat
import traceback
import torch
from utils import setup_logger, save


class BaseTrainer:
    def __init__(self, config, model, criterion, sample_input):
        config['trainer']['output_dir'] = os.path.join(str(pathlib.Path(os.path.abspath(__name__)).parent),
                                                       config['trainer']['output_dir'])
        config['name'] = config['name'] + '_' + model.model_name
        self.save_dir = os.path.join(config['trainer']['output_dir'], config['name'])
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        self.alphabet = config['dataset']['alphabet']

        if config['trainer']['resume_checkpoint'] == '' and config['trainer']['finetune_checkpoint'] == '':
            shutil.rmtree(self.save_dir, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # 保存本次实验的alphabet 到模型保存的地方
        save(self.alphabet, os.path.join(self.save_dir, 'dict.txt'))
        self.global_step = 0
        self.start_epoch = 0
        self.config = config

        self.model = model
        self.criterion = criterion
        # logger
        self.use_tensorboard = self.config['trainer']['tensorboard']
        self.epochs = self.config['trainer']['epochs']
        self.display_interval = self.config['trainer']['display_interval']

        self.logger = setup_logger(os.path.join(self.save_dir, 'train.log'))
        self.logger.info(pformat(self.config))
        self.logger.info(self.model)

        # device set
        torch.manual_seed(self.config['trainer']['seed'])  # 为CPU设置随机种子
        if len(self.config['trainer']['gpus']) > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.logger.info('train with pytorch {} gpu {}'.format(torch.__version__, self.config['trainer']['gpus']))
            self.gpus = {i: item for i, item in enumerate(self.config['trainer']['gpus'])}
            self.device = torch.device("cuda:0")
        else:
            self.with_cuda = False
            self.logger.info('train with pytorch {} and cpu'.format(torch.__version__))
            self.device = torch.device("cpu")

        self.metrics = {'val_acc': 0, 'train_loss': float('inf'), 'best_model': '', 'best_model_epoch': 0}

        self.optimizer = self._initialize('optimizer', torch.optim, model.parameters())

        # resume or finetune
        if self.config['trainer']['resume_checkpoint'] != '':
            self._laod_checkpoint(self.config['trainer']['resume_checkpoint'], resume=True)
        elif self.config['trainer']['finetune_checkpoint'] != '':
            self._laod_checkpoint(self.config['trainer']['finetune_checkpoint'], resume=False)
        self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)

        self.model.to(self.device)
        self.batch_max_length = self.model.batch_max_length
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(model)

        if self.use_tensorboard:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'  # 只显示 Error
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir)
            try:
                # add graph
                self.writer.add_graph(self.model, sample_input.to(self.device))
                torch.cuda.empty_cache()
            except:
                import traceback
                self.logger.error(traceback.format_exc())
                self.logger.warn('add graph to tensorboard failed')

    def train(self):
        """
        Full training logic
        """
        try:
            for epoch in range(self.start_epoch + 1, self.epochs + 1):
                self.epoch_result = self._train_epoch(epoch)
                self.scheduler.step()
                self._on_epoch_finish()
        except:
            self.logger.error(traceback.format_exc())
        if self.use_tensorboard:
            self.writer.close()
        self._on_train_finish()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _eval(self):
        """
        eval logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _on_epoch_finish(self):
        raise NotImplementedError

    def _on_train_finish(self):
        raise NotImplementedError

    def _save_checkpoint(self, epoch, file_name, save_best=False):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth.tar'
        """
        state_dict = self.model.module.state_dict() if torch.cuda.device_count() > 1 else self.model.state_dict()
        state = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': state_dict,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics
        }
        filename = os.path.join(self.checkpoint_dir, file_name)
        torch.save(state, filename)
        if save_best:
            shutil.copy(filename, os.path.join(self.checkpoint_dir, 'model_best.pth'))
            self.logger.info("Saving current best: {}".format(file_name))
        else:
            self.logger.info("Saving checkpoint: {}".format(filename))

    def _laod_checkpoint(self, checkpoint_path, resume):
        """
        Resume from saved checkpoints
        :param checkpoint_path: Checkpoint path to be resumed
        """
        self.logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'], strict=resume)
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch']
            self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
            # self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.logger.info("resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))
        else:
            self.logger.info("finetune from checkpoint {}".format(checkpoint_path))

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)
