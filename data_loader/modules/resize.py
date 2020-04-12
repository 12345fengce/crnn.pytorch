#!/usr/bin/env python
# encoding: utf-8
'''
@author: zhoujun
@time: 2019/12/19 下午3:23
'''
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

class Resize:
    def __init__(self, img_h, img_w, pad=True, random_crop=False, **kwargs):
        self.img_h = img_h
        self.img_w = img_w
        self.pad = pad
        self.random_crop = random_crop

    def __call__(self, img: np.ndarray):
        """
        对图片进行处理，先按照高度进行resize，resize之后如果宽度不足指定宽度，就补黑色像素，否则就强行缩放到指定宽度
        :param img_path: 图片地址
        :return:
        """
        img_h = self.img_h
        img_w = self.img_w
        augment = self.random_crop and np.random.rand() > 0.5
        if augment:
            img_h += 20
            img_w += 20
        h, w = img.shape[:2]
        ratio_h = self.img_h / h
        new_w = int(w * ratio_h)
        if new_w < img_w and self.pad:
            img = cv2.resize(img, (new_w, img_h))
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
            step = np.zeros((img_h, img_w - new_w, img.shape[-1]), dtype=img.dtype)
            img = np.column_stack((img, step))
        else:
            img = cv2.resize(img, (img_w, img_h))
            if len(img.shape) == 2:
                img = np.expand_dims(img, 2)
        if img.shape[-1] == 1:
            img = img[:, :, 0]
        if augment:
            img = transforms.RandomCrop((self.img_h, self.img_w))(Image.fromarray(img))
            img = np.array(img)
        return img


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    r = Resize(32, 320,random_crop=True)
    im = cv2.imread('0.jpg', 1)
    plt.imshow(im)
    plt.show()
    resize_img = r(im)
    plt.imshow(resize_img)
    plt.show()
