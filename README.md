# Circular Kernel

We provide code and our dataset for the Paper:

**Integrating Large Circular Kernels into CNNs through Neural Architecture Search**\
Kun He, Chao Li, Yixiao Yang, Gao Huang, John E. Hopcroft\

## Introduction

The square convolution kernel has been regarded as the standard and core unit of Convolutional Neural Networks (CNNs) since the first recognized CNN of *LeNet* proposed in 1989, and especially after *AlexNet* won the ILSVRC (ImageNet Large Scale Visual Recognition Competition) in 2012. Since then, various variants of convolution kernels have been proposed, including  separable convolution, dilated convolution, deformable convolution, \etc. 
Inspired by the fact that the retinal ganglion cells in the biological visual system have approximately concentric receptive fields (RFs), we propose the concept of circular kernels for the convolution operation. A $K \times K$ circular kernel is defined as a kernel that evenly samples K<sup>2</sup> pixels on the concentric circles to form a circular receptive field. 

## Getting Started

## Environment Setup

To set up the enviroment you can easily run the following command:
```buildoutcfg
conda create -n CK python=3.6
conda activate CK
pip install -r requirements.txt
```

## Data Preparation 
You need to first download the [ImageNet-2012](http://www.image-net.org/) to the folder `./data/imagenet` and move the validation set to the subfolder `./data/imagenet/val`. To move the validation set, you cloud use the following script: <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>

The directory structure is the standard layout as following.
```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

## Run

#### Search on CIFAR10

```python
python train_search.py --operation PRIMITIVES_circle_square \\
```
#### Search on ImageNet

Data preparation: 10% and 2.5% images need to be random sampled prior from earch class of trainingset as train and val, respectively. The sampled data is save into `./imagenet_search`.
Note that not to use torch.utils.data.sampler.SubsetRandomSampler for data sampling as imagenet is too large.

```python
python train_search_imagenet.py 
		--operation PRIMITIVES_circle_square \\
		--tmp_data_dir /path/to/your/sampled/data \\
```

#### Evaluation on CIFAR10:

```python
python train.py \\
       --auxiliary \\
       --cutout \\
       --arch  PC_DARTS_Circle_cifar 
```

#### Evaluation on ImageNet (mobile setting):

```python
python train_imagenet.py \\
       --tmp_data_dir /path/to/your/data \\
       --auxiliary \\
       --arch PC_DARTS_Circle_image
```
## Pretrained models
Coming soon!

## Reference