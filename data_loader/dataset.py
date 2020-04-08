# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:18
# @Author  : zhoujun
import sys
import six
import lmdb
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import punctuation_mend, get_datalist
from base import BaseDataSet


class ImageDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, num_label, ignore_chinese_punctuation, remove_blank, pre_processes, transform=None, **kwargs):
        """
        数据集初始化
        :param data_txt: 存储着图片路径和对于label的文件
        :param data_shape: 图片的大小(h,w)
        :param img_channel: 图片通道数
        :param num_label: 最大字符个数,应该和网络最终输出的序列宽度一样
        :param alphabet: 字母表
        """
        super().__init__(data_path, img_mode, num_label, ignore_chinese_punctuation, remove_blank, pre_processes, transform, **kwargs)

    def load_data(self, data_path: str) -> list:
        return get_datalist(data_path)

    def get_sample(self, index):
        img_path, label = self.data_list[index]
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.remove_blank:
            label = label.replace(' ', '')
        if self.ignore_chinese_punctuation:
            label = punctuation_mend(label)
        return img, label


class LmdbDataset(BaseDataSet):
    def __init__(self, data_path: str, img_mode, num_label, ignore_chinese_punctuation, remove_blank, pre_processes, transform=None, **kwargs):
        super().__init__(data_path, img_mode, num_label, ignore_chinese_punctuation, remove_blank, pre_processes, transform, **kwargs)

    def get_sample(self, index):
        index = self.data_list[index]
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            if self.img_mode == 'RGB':
                img = Image.open(buf).convert('RGB')  # for color image
            elif self.img_mode == "GRAY":
                img = Image.open(buf).convert('L')
            else:
                raise NotImplementedError
            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            if self.remove_blank:
                label = label.replace(' ', '')
            if self.ignore_chinese_punctuation:
                label = punctuation_mend(label)
            img = np.array(img)
        return img, label

    def load_data(self, data_path: str) -> list:
        self.env = lmdb.open(data_path, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (data_path))
            sys.exit(0)

        filtered_index_list = []
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
            for index in range(self.nSamples):
                index += 1  # lmdb starts with 1
                label_key = 'label-%09d'.encode() % index
                label = txn.get(label_key).decode('utf-8')
                if len(label) > self.num_label:
                    # print(f'The length of the label is longer than max_length: length
                    # {len(label)}, {label} in dataset {self.root}')
                    continue

                # By default, images containing characters which are not in opt.character are filtered.
                # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                filtered_index_list.append(index)
        return filtered_index_list


class Batch_Balanced_Dataset(object):
    def __init__(self, dataset_list: list, ratio_list: list, loader_args: dict):
        """
        对datasetlist里的dataset按照ratio_list里对应的比例组合，似的每个batch里的数据按按照比例采样的
        :param dataset_list: 数据集列表
        :param ratio_list: 比例列表
        :param loader_args: dataloader的配置
        """
        assert sum(ratio_list) == 1 and len(dataset_list) == len(ratio_list)

        self.dataset_len = 0
        self.data_loader_list = []
        self.dataloader_iter_list = []
        all_batch_size = loader_args.pop('batch_size')
        for _dataset, batch_ratio_d in zip(dataset_list, ratio_list):
            _batch_size = max(round(all_batch_size * float(batch_ratio_d)), 1)
            _data_loader = DataLoader(dataset=_dataset, batch_size=_batch_size, drop_last=True, **loader_args)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))
            self.dataset_len += len(_dataset)

    def __iter__(self):
        return self

    def __len__(self):
        return min([len(x) for x in self.data_loader_list])

    def __next__(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = next(data_loader_iter)
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = next(self.dataloader_iter_list[i])
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)
        return balanced_batch_images, balanced_batch_texts


if __name__ == '__main__':
    import os
    from tqdm import tqdm
    import anyconfig
    from torchvision import transforms
    from utils import parse_config

    train_transfroms = transforms.Compose([
        transforms.ColorJitter(brightness=0.5),
        transforms.ToTensor()
    ])
    config = anyconfig.load(open("config/icdar2015_win.yaml", 'rb'))
    if 'base' in config:
        config = parse_config(config)
    if os.path.isfile(config['dataset']['alphabet']):
        config['dataset']['alphabet'] = str(np.load(config['dataset']['alphabet']))

    dataset_args = config['dataset']['validate']['dataset']['args']
    dataset_args['num_label'] = 80
    dataset_args['alphabet'] = config['dataset']['alphabet']
    dataset = ImageDataset(transform=train_transfroms,**dataset_args)
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
    for i, (images, labels) in enumerate(tqdm(data_loader)):
        pass
        print(images.shape)
        print(labels)
        img = images[0].numpy().transpose((1, 2, 0))
        from matplotlib import pyplot as plt

        plt.imshow(img)
        plt.show()
