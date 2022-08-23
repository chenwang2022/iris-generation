# Iris Generation
This is the implementation of "Generating Intra- and Inter-Class Iris Images by Identity Contrast" (accepted by IJCB 2022). Iris recognition is one of the most accurate and reliable biometric technologies. However, due to the high collection costs and privacy of the iris, it is difficult to build a large-scale iris image database for training iris recognition models. To augment limited iris data, this paper proposes an iris image generation algorithm which can produce numerous iris images with realistic composition and clear texture. The experimental results have verified that a large-scale synthetic dataset is beneficial to improving the performance of iris recognition. Follows are the intra- and inter-class iris image synthesis results of the proposed approach.

<img src="https://user-images.githubusercontent.com/111242515/184875536-04cbc109-1120-4e43-a5ba-2792811bbe5d.PNG" width="700px">

## Requirements
This implementation is based on Pytorch. Our environment is:
* Python 3.8
* CUDA 11.3
* CuDNN 8.2
* Pytorch 1.11

or:
* Python 3.6
* CUDA 10.1
* CuDNN 7.6
* Pytorch 1.9

## Usage
You can train model with single GPU
```
CUDA_VISIBLE_DEVICES=0 python train.py DATA_PATH
```
or multiple GPUs
```
python -m torch.distributed.launch --nproc_per_node=N_GPU train.py DATA_PATH
```

## Synthetic Dataset
We have picked high-quality generated images and built a synthetic dataset, which consists of 187,717 images of 10,000 classes at 256×256 resolution. There are 10-20 images are available for each class. The classes in the synthetic dataset are completely created by our model and do not exist in the real world. You can use, redistribute, and adapt it for non-commercial purposes, as long as you (a) give appropriate credit by citing our paper, (b) indicate any changes that you've made. The download link of the synthetic dataset is https://pan.baidu.com/s/1XFou9qgXxlm5-v5S1u4K2Q?pwd=avgv

## License
Our GANs are based on the StyleGAN2 framework: https://github.com/rosinality/stylegan2-pytorch
