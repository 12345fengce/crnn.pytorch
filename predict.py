# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 22:21
# @Author  : zhoujun
import os
import cv2
import numpy as np
import torch

from data_loader import get_transforms


def decode(preds, alphabet, raw=False):
    if len(preds.shape) > 2:
        preds_idx = preds.argmax(axis=2)
        preds_prob = preds.max(axis=2)
    else:
        preds_idx = preds
        preds_prob = np.ones_like(preds)
    result_list = []
    for word, prob in zip(preds_idx, preds_prob):
        if raw:
            result_list.append((''.join([alphabet[int(i)] for i in word]), prob))
        else:
            result = []
            conf = []
            try:

                for i, index in enumerate(word):
                    if i < len(word) - 1 and word[i] == word[i + 1]:  # Hack to decode label as well
                        continue
                    if index == 0:
                        continue
                    else:
                        result.append(alphabet[int(index)])
                        conf.append(prob[i])
            except:
                a = 1
            result_list.append((''.join(result), conf))
    return result_list


class PytorchNet:
    def __init__(self, model_path, gpu_id=None):
        """
        初始化模型
        :param model_path: 模型地址
        :param gpu_id: 在哪一块gpu上运行
        """
        checkpoint = torch.load(model_path)
        print('load {} epoch params'.format(checkpoint['epoch']))
        config = checkpoint['config']
        alphabet = config['dataset']['alphabet']
        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)
        self.transform = get_transforms(self.transform)

        self.gpu_id = gpu_id
        img_h, img_w = 32, 100
        for process in config['dataset']['train']['dataset']['args']['pre_processes']:
            if process['type'] == "Resize":
                img_h = process['args']['img_h']
                img_w = process['args']['img_w']
                break
        self.img_w = img_w
        self.img_h = img_h
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.alphabet = alphabet
        img_channel = 3 if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
        self.net = get_model(img_channel, len(self.alphabet), config['arch']['args'])
        self.net.load_state_dict(checkpoint['state_dict'])
        # self.net = torch.jit.load('crnn_lite_gpu.pt')
        self.net.to(self.device)

    def predict(self, img_path, model_save_path=None):
        """
        对传入的图像进行预测，支持图像地址和numpy数组
        :param img_path: 图像地址
        :return:
        """
        assert os.path.exists(img_path), 'file is not exists'
        img = self.pre_processing(img_path)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze(dim=0)

        tensor = tensor.to(self.device)
        preds, tensor_img = self.net(tensor)

        preds = preds.softmax(dim=2).detach().cpu().numpy()
        # result = decode(preds, self.alphabet, raw=True)
        # print(result)
        result = decode(preds, self.alphabet)
        if model_save_path is not None:
            # 输出用于部署的模型
            save(self.net, tensor, model_save_path)
        return result, tensor_img

    def pre_processing(self, img_path):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
        :param img_path: 图片地址
        :return:
        """
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        ratio_h = float(self.img_h) / h
        new_w = int(w * ratio_h)

        if new_w < self.img_w:
            img = cv2.resize(img, (new_w, self.img_h))
            step = np.zeros((self.img_h, self.img_w - new_w, img.shape[-1]), dtype=img.dtype)
            img = np.column_stack((img, step))
        else:
            img = cv2.resize(img, (self.img_w, self.img_h))
        return img


def save(net, input, save_path):
    # 在gpu导出的模型只能在gpu使用，cpu导出的只能在cpu使用
    net.eval()
    traced_script_module = torch.jit.trace(net, input)
    traced_script_module.save(save_path)


if __name__ == '__main__':
    from models import get_model
    import time
    from matplotlib import pyplot as plt
    from matplotlib.font_manager import FontProperties

    font = FontProperties(fname=r"msyh.ttc", size=14)

    img_path = '0.jpg'
    model_path = 'output/crnn_lmdb_DWBlock_None_ResNet_RNN_CTC/checkpoint/model_best.pth'

    crnn_net = PytorchNet(model_path=model_path, gpu_id=0)
    start = time.time()
    for i in range(100):
        result, img = crnn_net.predict(img_path,'vgg.pt')
        break
    print((time.time() - start) *1000/ 100)

    label = result[0][0]
    plt.title(label, fontproperties=font)
    plt.imshow(img.detach().cpu().numpy().squeeze().transpose((1, 2, 0)), cmap='gray')
    plt.show()
