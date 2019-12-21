# -*- coding: utf-8 -*-
# @Time    : 2019/11/6 15:08
# @Author  : zhoujun
from torch.utils.data import Dataset
from data_loader.modules import *


class BaseDataSet(Dataset):
    def __init__(self, data_path: str, img_mode,num_label, ignore_chinese_punctuation, remove_blank, pre_processes, transform=None, **kwargs):
        """
        :param ignore_chinese_punctuation: 是否转换全角为半角
        """
        assert img_mode in ['RGB', 'BRG', 'GRAY']
        self.data_list = self.load_data(data_path)
        self.img_mode = img_mode
        self.num_label = num_label
        self.transform = transform
        self.remove_blank = remove_blank
        self.ignore_chinese_punctuation = ignore_chinese_punctuation
        self._init_pre_processes(pre_processes)

    def _init_pre_processes(self, pre_processes):
        self.aug = []
        if pre_processes is not None:
            for aug in pre_processes:
                if 'args' not in aug:
                    args = {}
                else:
                    args = aug['args']
                if isinstance(args, dict):
                    cls = eval(aug['type'])(**args)
                else:
                    cls = eval(aug['type'])(args)
                self.aug.append(cls)

    def load_data(self, data_path: str) -> list:
        """
        把数据加载为一个list：
        :params data_path: 存储数据的文件夹或者文件
        return a list ,包含img_path和label
        """
        raise NotImplementedError

    def apply_pre_processes(self, data):
        for aug in self.aug:
            data = aug(data)
        return data

    def get_sample(self, index):
        raise NotImplementedError

    def __getitem__(self, index):
        img, label = self.get_sample(index)
        img = self.apply_pre_processes(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data_list)
