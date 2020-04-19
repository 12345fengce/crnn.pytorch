Convolutional Recurrent Neural Network
======================================

This software implements the Convolutional Recurrent Neural Network (CRNN) in pytorch.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)


## Requirements
* pytorch 1.3+
* torchvision 0.4+

## Data Preparation
Prepare a text in the following format
```
/path/to/img/img.jpg label
...
```

# Performance

data link [baiduyun]( https://pan.baidu.com/s/1w7KssjsOHbBTLtjaltLJ0w) code: 9p2m, the dataset is generate by  <https://github.com/Belval/TextRecognitionDataGenerator>  
dataset contains 10w images for train and 1w images for test:1w  
for all arch ,we stop training after 30 epochs  
environment: cuda9.2 torch1.4 torchvision0.5

| arch                    | model size(m)   | gpu mem(m) | speed(ms,avg of 100 inference)   | acc |
| ----------------------- | ------ | -------- | ------ | ------ |
| CNN_lite_LSTM_CTC | 6.25 | 2731     | 6.91ms | 0.8866 |
| VGG(BasicConv)_LSTM_CTC(w320) | 25.45 | 2409     | 4.02ms | 0.9874 |
| VGG(BasicConv)_LSTM_CTC(w160) | 25.45 | 2409     | 4.02ms | 0.9908 |
| VGG(BasicConv)_LSTM_CTC(w160_no_imagenet_mean_std) | 25.45 | 2409     | 4.02ms | 0.9927 |
| VGG(BasicConv)_LSTM_CTC(w160.sub_(0.5).div_(0.5)) | 25.45 | 2409     | 4.02ms | 0.9927 |
| VGG(BasicConv)_LSTM_CTC(w160 origin crnn rnn) | 25.45 | 2409     | 4.02ms | 0.9922 |
| VGG(DWconv)_LSTM_CTC(w160_no_imagenet_mean_std) | 25.45 | 2409     | 4.01ms | 0.9725 |
| VGG(GhostModule)_LSTM_CTC(w160_no_imagenet_mean_std) | 25.45 | 2329     | 5.46ms | 0.9878 |
| VGG(DWConv)_LSTM_CTC | 24.45 | 3985     | 6.47ms | 0.893 |
| VGG(GhostModule)_LSTM_CTC | 9.23 | 4289     | 8.13ms | 0.04 |
| ResNet(BasicBlockV2)_LSTM_CTC | 37.21 | 5515     | 8.6ms | 0.9608|
| ResNet(DWBlock_no_se)_LSTM_CTC | 19.22 | 5533     | 12ms | 0.9566|
| ResNet(DWBlock_se)_LSTM_CTC | 19.90 |   5729   | 10ms | 0.9559 |
| ResNet(GhostBottleneck_se)_LSTM_CTC | 23.10 | 6291     | 13ms | 0.97|


## Train

1. config the `dataset['train']['dataset']['data_path']`,`dataset['validate']['dataset']['data_path']` in [config.yaml](config/icdar2015.yaml)
2. generate alphabet
  use fellow script to generate `alphabet.py` in the some folder with `train.py` 
```sh
python3 utils/get_keys.py
```
2. use following script to run
```sh
python3 train.py --config_path config.yaml
```

## Predict 
[predict.py](predict.py) is used to inference on single image

1. config `model_path`, `img_path` in [predict.py](predict.py)
2. use following script to predict
```sh
python3 predict.py
```