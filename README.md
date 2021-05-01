# Semantic Segmentation on PyTorch
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]

This repository is created for mid-term project of DATA620004, Fudan University. The main part of the code is forked from https://github.com/Tramac/awesome-semantic-segmentation-pytorch, which implements a various of advanced deep learning models for the sematic segmentation task in pytorch. For our mid-term project, we train the BiSeNet model on cityscapes supervised by different loss functions and try to explore the influence brought by them. For the simplicity of reproduce, we will introduce how to install the package, how to train and evaluate the model as follows.

<p align="center"><img width="100%" src="docs/weimar_000091_000019_gtFine_color.png" /></p>

## Installation
```
# semantic-segmentation-pytorch dependencies
pip install ninja tqdm

# follow PyTorch installation in https://pytorch.org/get-started/locally/
conda install pytorch torchvision -c pytorch

# install PyTorch Segmentation
git clone https://github.com/Tramac/awesome-semantic-segmentation-pytorch.git

# the following will install the lib with symbolic links, so that you can modify
# the files if you want and won't need to re-build it
cd awesome-semantic-segmentation-pytorch/core/nn
python setup.py build develop
```

## Usage
### Train
-----------------
- **Use CrossEntropy**
```
# for example, use single GPU and set batchsize as 8:
python train.py --model bisenet --backbone resnet18 --dataset citys --lr 0.01 --epochs 80 --batch-size 8
```

- **Use Auxiliary Loss**
```
# for example, use single GPU and set batchsize as 8:
python train.py --model bisenet --backbone resnet18 --dataset citys --lr 0.01 --epochs 80 --batch-size 8 --aux
```

- **Use Ohem CrossEntropy**

```
# for example, use multi GPUs and set batchzie as 32:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py --model bisenet --backbone resnet18 --dataset citys --lr 0.025 --epochs 100 --use-ohem True
```

### Evaluation
-----------------
- **Single GPU evaluating**
```
# for example, evaluate bisenet_resnet18_citys with single GPU:
python eval.py --model bisenet --backbone resnet18 --dataset citys
```
- **Multi-GPU evaluating**
```
# for example, evaluate bisenet_resnet18_citys_aux with 4 GPUs:
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS eval.py --model bisenet --backbone resnet18 --dataset citys --aux
```


### Demo
```
cd ./scripts
python demo.py --model fcn32s_vgg16_voc --input-pic ./datasets/test.jpg
```

```
.{SEG_ROOT}
├── scripts
│   ├── demo.py
│   ├── eval.py
│   └── train.py
```

### Visiualization
```
cd ./runs
tensorboard --logdir tensorboardx
```

## Support

#### Model

- [FCN](https://arxiv.org/abs/1411.4038)
- [ENet](https://arxiv.org/pdf/1606.02147)
- [PSPNet](https://arxiv.org/pdf/1612.01105)
- [ICNet](https://arxiv.org/pdf/1704.08545)
- [DeepLabv3](https://arxiv.org/abs/1706.05587)
- [DeepLabv3+](https://arxiv.org/pdf/1802.02611)
- [DenseASPP](http://openaccess.thecvf.com/content_cvpr_2018/papers/Yang_DenseASPP_for_Semantic_CVPR_2018_paper.pdf)
- [EncNet](https://arxiv.org/abs/1803.08904v1)
- [BiSeNet](https://arxiv.org/abs/1808.00897)
- [PSANet](https://hszhao.github.io/papers/eccv18_psanet.pdf)
- [DANet](https://arxiv.org/pdf/1809.02983)
- [OCNet](https://arxiv.org/pdf/1809.00916)
- [CGNet](https://arxiv.org/pdf/1811.08201)
- [ESPNetv2](https://arxiv.org/abs/1811.11431)
- [CCNet](https://arxiv.org/pdf/1811.11721)
- [DUNet(DUpsampling)](https://arxiv.org/abs/1903.02120)
- [FastFCN(JPU)](https://arxiv.org/abs/1903.11816)
- [LEDNet](https://arxiv.org/abs/1905.02423)
- [Fast-SCNN](https://github.com/Tramac/Fast-SCNN-pytorch)
- [LightSeg](https://github.com/Tramac/Lightweight-Segmentation)
- [DFANet](https://arxiv.org/abs/1904.02216)

[DETAILS](https://github.com/Tramac/awesome-semantic-segmentation-pytorch/blob/master/docs/DETAILS.md) for model & backbone.
```
.{SEG_ROOT}
├── core
│   ├── models
│   │   ├── bisenet.py
│   │   ├── danet.py
│   │   ├── deeplabv3.py
│   │   ├── deeplabv3+.py
│   │   ├── denseaspp.py
│   │   ├── dunet.py
│   │   ├── encnet.py
│   │   ├── fcn.py
│   │   ├── pspnet.py
│   │   ├── icnet.py
│   │   ├── enet.py
│   │   ├── ocnet.py
│   │   ├── ccnet.py
│   │   ├── psanet.py
│   │   ├── cgnet.py
│   │   ├── espnet.py
│   │   ├── lednet.py
│   │   ├── dfanet.py
│   │   ├── ......
```

#### Dataset

You can run script to download dataset, such as:

```
cd ./core/data/downloader
python ade20k.py --download-dir ../datasets/ade
```

|                           Dataset                            | training set | validation set | testing set |
| :----------------------------------------------------------: | :----------: | :------------: | :---------: |
| [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) |     1464     |      1449      |      ✘      |
| [VOCAug](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) |    11355     |      2857      |      ✘      |
| [ADK20K](http://groups.csail.mit.edu/vision/datasets/ADE20K/) |    20210     |      2000      |      ✘      |
| [Cityscapes](https://www.cityscapes-dataset.com/downloads/)  |     2975     |      500       |      ✘      |
| [COCO](http://cocodataset.org/#download)           |              |                |             |
| [SBU-shadow](http://www3.cs.stonybrook.edu/~cvl/content/datasets/shadow_db/SBU-shadow.zip) |     4085     |      638       |      ✘      |
| [LIP(Look into Person)](http://sysu-hcp.net/lip/)       |    30462     |     10000      |    10000    |

```
.{SEG_ROOT}
├── core
│   ├── data
│   │   ├── dataloader
│   │   │   ├── ade.py
│   │   │   ├── cityscapes.py
│   │   │   ├── mscoco.py
│   │   │   ├── pascal_aug.py
│   │   │   ├── pascal_voc.py
│   │   │   ├── sbu_shadow.py
│   │   └── downloader
│   │       ├── ade20k.py
│   │       ├── cityscapes.py
│   │       ├── mscoco.py
│   │       ├── pascal_voc.py
│   │       └── sbu_shadow.py
```

## Result
- **PASCAL VOC 2012**

|Loss|Backbone|TrainSet|EvalSet|Learning rate|epochs|JPU|Mean IoU|pixAcc|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|CrossEntropy|resnet18|train|val|0.01|80|✘|63.555|93.459|
|CrossEntropy+aux|resnet18|train|val|0.01|80|✘|64.406|93.642|
|Ohem|resnet18|train|val|0.025|100|✘|63.130|93.113|
|Ohem+aux|resnet18|train|val|0.025|100|✘|63.724|93.044|



Note: `lr=1e-4, batch_size=4, epochs=80`.

## Overfitting Test
See [TEST](https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/tree/master/tests) for details.

```
.{SEG_ROOT}
├── tests
│   └── test_model.py
```

## References
- [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding)
- [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
- [gloun-cv](https://github.com/dmlc/gluon-cv)
- [imagenet](https://github.com/pytorch/examples/tree/master/imagenet)

<!--
[![python-image]][python-url]
[![pytorch-image]][pytorch-url]
[![lic-image]][lic-url]
-->

[python-image]: https://img.shields.io/badge/Python-2.x|3.x-ff69b4.svg
[python-url]: https://www.python.org/
[pytorch-image]: https://img.shields.io/badge/PyTorch-1.1-2BAF2B.svg
[pytorch-url]: https://pytorch.org/
[lic-image]: https://img.shields.io/badge/Apache-2.0-blue.svg
[lic-url]: https://github.com/Tramac/Awesome-semantic-segmentation-pytorch/blob/master/LICENSE
