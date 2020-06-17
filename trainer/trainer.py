# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:20
# @Author  : zhoujun
from addict import Dict
import shutil
import time
import Levenshtein
from tqdm import tqdm
import torch

from base import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, sample_input, converter, validate_loader=None):
        super().__init__(config, model, criterion, sample_input)
        self.train_loader = train_loader
        self.train_loader_len = len(train_loader)
        self.validate_loader = validate_loader
        self.converter = converter
        if self.validate_loader is not None:
            self.logger.info(f'train dataset has {self.train_loader.dataset_len} samples,{len(train_loader)} in dataloader,'
                             f'validate dataset has {self.validate_loader.dataset_len} samples,{len(self.validate_loader)} in dataloader')
        else:
            self.logger.info(f'train dataset has {len(self.train_loader.dataset)} samples,{len(self.train_loader)} in dataloader')

        self.run_time_dict = {}

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        self.run_time_dict['batch_start'] = time.time()
        self.run_time_dict['epoch'] = epoch
        self.run_time_dict['n_correct'] = 0
        self.run_time_dict['train_num'] = 0
        self.run_time_dict['train_loss'] = 0
        self.run_time_dict['norm_edit_dis'] = 0

        for i, batch in enumerate(self.train_loader):
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            self.run_time_dict['lr'] = self.optimizer.param_groups[0]['lr']
            self.run_time_dict['iter'] = i
            batch = self._before_step(batch)
            batch_out = self._run_step(batch)
            self._after_step(batch_out)
        epoch_time = time.time() - epoch_start
        self.run_time_dict['train_acc'] = self.run_time_dict['n_correct'] / self.run_time_dict['train_num']
        self.logger.info(f"[{self.run_time_dict['epoch']}/{self.epochs}], "
                         f"train_acc: {self.run_time_dict['train_acc']:.4f}, "
                         f"train_loss: {self.run_time_dict['train_loss'] / self.train_loader_len:.4f}, "
                         f"time: {epoch_time:.4f}, "
                         f"lr: {self.run_time_dict['lr']}")

    def _before_step(self, batch):
        targets, targets_lengths = self.converter.encode(batch['label'], self.batch_max_length)
        batch['img'] = batch['img'].to(self.device)
        batch['targets'] = targets.to(self.device)
        batch['targets_lengths'] = targets_lengths
        return batch

    def _run_step(self, batch):
        # forward
        cur_batch_size = batch['img'].shape[0]
        targets = batch['targets']
        if self.model.head_type == 'CTC':
            preds = self.model(batch['img'])[0]
            loss = self.criterion(preds, batch)
        elif self.model.head_type == 'Attention':
            preds = self.model(batch['img'], targets[:, :-1])[0]
            loss = self.criterion(preds, batch)
        else:
            raise NotImplementedError
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
        self.optimizer.step()
        batch_dict = self.accuracy_batch(preds, batch['label'])
        batch_dict['loss'] = loss.item()
        batch_dict['batch_size'] = cur_batch_size
        return batch_dict

    def _after_step(self, batch_out):
        # loss 和 acc 记录到日志
        self.run_time_dict['train_num'] += batch_out['batch_size']
        self.run_time_dict['train_loss'] += batch_out['loss']
        self.run_time_dict['n_correct'] += batch_out['n_correct']
        self.run_time_dict['norm_edit_dis'] += batch_out['norm_edit_dis']

        acc = batch_out['n_correct'] / batch_out['batch_size']
        norm_edit_dis = 1 - batch_out['norm_edit_dis'] / batch_out['batch_size']
        if self.use_tensorboard:
            # write tensorboard
            self.writer.add_scalar('TRAIN/loss', batch_out['loss'], self.global_step)
            self.writer.add_scalar('TRAIN/acc', acc, self.global_step)
            self.writer.add_scalar('TRAIN/norm_edit_dis', norm_edit_dis, self.global_step)
            self.writer.add_scalar('TRAIN/lr', self.run_time_dict['lr'], self.global_step)
            self.writer.add_text('Train/pred_gt', ' || '.join(batch_out['show_str'][:10]), self.global_step)

        if self.global_step % self.display_interval == 0:
            batch_time = time.time() - self.run_time_dict['batch_start']
            speed = self.display_interval * batch_out['batch_size'] / batch_time
            self.logger.info(f"[{self.run_time_dict['epoch']}/{self.epochs}], "
                             f"[{self.run_time_dict['iter'] + 1}/{self.train_loader_len}], global_step: {self.global_step}, "
                             f"Speed: {speed:.1f} samples/sec, loss:{batch_out['loss']:.4f}, "
                             f"acc:{acc:.4f}, norm_edit_dis:{norm_edit_dis:.4f} lr:{self.run_time_dict['lr']}, "
                             f"time:{batch_time:.2f}")
            self.run_time_dict['batch_start'] = time.time()

    def _eval(self, max_step=None, dest='test model'):
        self.model.eval()
        n_correct = 0
        norm_edit_dis = 0
        for i, (images, labels) in enumerate(tqdm(self.validate_loader, desc=dest)):
            if max_step is not None and i >= max_step:
                break
            images = images.to(self.device)
            with torch.no_grad():
                preds = self.model(images)[0]
            batch_dict = self.accuracy_batch(preds, labels)
            n_correct += batch_dict['n_correct']
            norm_edit_dis += batch_dict['norm_edit_dis']
        return {'n_correct': n_correct, 'norm_edit_dis': norm_edit_dis}

    def _on_epoch_finish(self):
        net_save_path = f'{self.checkpoint_dir}/model_latest.pth'
        self._save_checkpoint(self.run_time_dict['epoch'], net_save_path)

        if self.validate_loader is not None:
            epoch_eval_dict = self._eval()
            val_acc = epoch_eval_dict['n_correct'] / self.validate_loader.dataset_len
            norm_edit_dis = 1 - epoch_eval_dict['norm_edit_dis'] / self.validate_loader.dataset_len

            if self.use_tensorboard:
                self.writer.add_scalar('EVAL/acc', val_acc, self.global_step)
                self.writer.add_scalar('EVAL/edit_distance', norm_edit_dis, self.global_step)

            self.logger.info(f"[{self.run_time_dict['epoch']}/{self.epochs}], val_acc: {val_acc:.6f}, "
                             f"norm_edit_dis: {norm_edit_dis}")

            if val_acc >= self.metrics['val_acc']:
                self.metrics['val_acc'] = val_acc
                self.metrics['train_loss'] = self.run_time_dict['train_loss']
                self.metrics['train_acc'] = self.run_time_dict['train_acc']
                self.metrics['best_acc_epoch'] = self.run_time_dict['epoch']
                best_save_path = f'{self.checkpoint_dir}/model_bect_acc.pth'
                shutil.copy(net_save_path, best_save_path)
                self.logger.info(f"Saving current best acc : {best_save_path}")
            if norm_edit_dis >= self.metrics['norm_edit_dis']:
                self.metrics['norm_edit_dis'] = norm_edit_dis
                self.metrics['train_loss'] = self.run_time_dict['train_loss']
                self.metrics['train_acc'] = self.run_time_dict['train_acc']
                self.metrics['best_ned_epoch'] = self.run_time_dict['epoch']
                best_save_path = f'{self.checkpoint_dir}/model_bect_ned.pth'
                shutil.copy(net_save_path, best_save_path)
                self.logger.info(f"Saving current best norm_edit_dis : {best_save_path}")
        else:
            if self.run_time_dict['train_acc'] > self.metrics['train_acc']:
                self.metrics['train_loss'] = self.run_time_dict['train_loss']
                self.metrics['train_acc'] = self.run_time_dict['train_acc']
                self.metrics['best_model_epoch'] = self.run_time_dict['epoch']
                best_save_path = f'{self.checkpoint_dir}/model_bect_loss.pth'
                shutil.copy(net_save_path, best_save_path)
                self.logger.info(f"Saving current best loss : {best_save_path}")
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {}, '.format(k, v)
        self.logger.info(best_str)

    def accuracy_batch(self, predictions, labels):
        n_correct = 0
        norm_edit_dis = 0.0
        predictions = predictions.softmax(dim=2).detach().cpu().numpy()
        preds_str = self.converter.decode(predictions)
        show_str = []
        for (pred, pred_conf), target in zip(preds_str, labels):
            norm_edit_dis += Levenshtein.distance(pred, target) / max(len(pred), len(target))
            show_str.append(f'{pred} -> {target}')
            if pred == target:
                n_correct += 1
        return {'n_correct': n_correct, 'norm_edit_dis': norm_edit_dis, 'show_str': show_str}

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger.info(f'{k}:{v}')
        self.logger.info('finish train')
