# Supreme: Fine-grained Radio map Reconstruction via Spatial-Temporal Fusion Network
In this paper, we propose a fine-grained radio map reconstruction framework, called Supreme, based on
crowd-sourced data in an image super-resolution manner. 

## Paper
Kehan Li, Jiming Chen, Baosheng Yu, Chao Li, Zhangchong Shen, Shibo He. "Supreme: Fine-grained Radio map Reconstruction via Spatial-Temporal Fusion Network.", Submitted to IPSN 2020.

## Requirements
Supreme uses the following dependencies:
* [Pytorch 0.4.3](https://pytorch.org/get-started/locally/) and its dependencies
* Numpy, Scipy and Pandas
* CUDA 10.0 or latest version. And **cuDNN** is highly recommended

## Model Training
Main arguments:
- n_epochs：number of epochs of training
- batch_size: training batch size
- lr: learning rate
- n_residuals: number of residual blocks
- base_channels: number of feature maps
- img__width: radio map width
- img_height: radio map height
- depth: number of historical radio maps
- channels: number of radio map channels
- sample_interval: interval of validation
- zoom: upscale factor of radio maps
- ext_flag: whether to use external factor

Examples on model training:
* Training Supreme with default setting:
```
python train.py --ext_flag
```

* Training Supreme with given setting (with external factor):
```
python train.py --n_residuals=16 --base_channels=64 --depth=6 --ext_flag
```

* Training Supreme without external factor 
```
python train.py --n_residuals=16 --base_channels=64 --depth=6
```

## Model Test
To test trained model, following code can be used:
```
python test.py --n_residuals=16 --base_channels=64 --depth=6
```

## Dataset
The data is divided into train/validation/test set with a ratio 4:1:1, more details can be found in our paper. 

