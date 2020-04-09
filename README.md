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
for all arch ,we stop training after 10 epochs
environment: cuda9.2 torch1.4 torchvision0.5
| arch                    | model size(m)   | gpu mem(m) | speed(ms,avg of 100 inference)   | acc |
| ----------------------- | ------ | -------- | ------ | ------ |
| CNN_lite_LSTM_CTC | 6.25 | 2731     | 6.91ms | 0.8866 |
| VGG(BasicConv)_LSTM_CTC | 25.45 | 3989     | 6.63ms | 0.9531 |
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
[predict.py](src/scripts/predict.py) is used to inference on single image

1. config `model_path`, `img_path` in [predict.py](src/scripts/predict.py)
2. use following script to predict
```sh
python3 predict.py
```