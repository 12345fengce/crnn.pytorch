# -*- coding: utf-8 -*-
# @Time    : 2020/4/27 11:28
# @Author  : zhoujun
import numpy as np
from .augment import distort, stretch, perspective

__all__ = ['RandomAug']


class RandomAug:
    def __init__(self):
        pass

    def __call__(self, img):
        if np.random.randn() > 0.3:
            img = distort(img, 3)
        elif np.random.randn() > 0.6:
            img = stretch(img, 3)
        else:
            img = perspective(img)
        return img


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import cv2
    r = RandomAug()
    im = cv2.imread(r'D:\code\crnn.pytorch\0.jpg')
    plt.imshow(im)
    resize_img = r(im)
    plt.figure()
    plt.imshow(resize_img)
    plt.show()
