![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
 
# CSG: Contrastive Syn-to-Real Generalization
 
<!-- ### [Project](https://) | [Paper](https://arxiv.org/abs/XXX) -->
[Paper](https://arxiv.org/abs/XXX)
 
Contrastive Syn-to-Real Generalization.<br>
[Wuyang Chen](https://chenwydj.github.io/), [Zhiding Yu](https://chrisding.github.io/), [Shalini De Mello](https://research.nvidia.com/person/shalini-gupta), [Sifei Liu](https://www.sifeiliu.net/), [Jose M. Alvarez](https://rsu.data61.csiro.au/people/jalvarez/), [Zhangyang Wang](https://www.atlaswang.com/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/).<br>
In ICLR 2021.

* Visda-17 to COCO
- [x] train resnet101 with CSG
- [x] evaluation
* GTA5 to Cityscapes
- [x] train deeplabv2 (resnet50/resnet101) with CSG
- [x] evaluation

## Usage

### Visda-17
* Download [Visda-17 Dataset](http://ai.bu.edu/visda-2017/#download)

#### Evaluation
* Download [pretrained ResNet101 on Visda17](https://drive.google.com/file/d/1VdbrwevsYy7I5S3Wo7-S3MwrZZjj09QS/view?usp=sharing)
* Put the checkpoint under `./CSG/pretrained/`
* Put the code below in `train.sh`
```bash
python train.py \
--epochs 30 \
--batch-size 32 \
--lr 1e-4 \
--rand_seed 0 \
--csg 0.1 \
--apool \
--augment \
--csg-stages 3.4 \ 
--factor 0.1 \
--resume pretrained/csg_res101_vista17_best.pth.tar \
--evaluate
```
* Run `CUDA_VISIBLE_DEVICES=0 bash train.sh`
  - Please update the GPU index via `CUDA_VISIBLE_DEVICES` based on your need.

#### Train with CSG
* Put the code below in `train.sh`
```bash
python train.py \
--epochs 30 \
--batch-size 32 \
--lr 1e-4 \
--rand_seed 0 \
--csg 0.1 \
--apool \
--augment \
--csg-stages 3.4 \ 
--factor 0.1 \
```
* Run `CUDA_VISIBLE_DEVICES=0 bash train.sh`
  - Please update the GPU index via `CUDA_VISIBLE_DEVICES` based on your need.


### GTA5 &rarr; Cityscapes
* Download [GTA5 dataset](https://download.visinf.tu-darmstadt.de/data/from_games/).
* Download the [leftImg8bit_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=3) and [gtFine_trainvaltest.zip](https://www.cityscapes-dataset.com/file-handling/?packageID=1) from the Cityscapes.
* Prepare the annotations by using the [createTrainIdLabelImgs.py](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py).
* Put the [file of image list](tools/datasets/cityscapes/) into where you save the dataset.
* **Remember to properly set the `C.dataset_path` in the `config_seg.py` to the path where datasets reside.**

#### Evaluation
* Download pretrained [DeepLabV2-ResNet50](https://drive.google.com/file/d/1E2CosTtGVgIe6BfLBV9vNmyj6l9aYUbk/view?usp=sharing) and [DeepLabV2-ResNet101](https://drive.google.com/file/d/17Pe86m4OCGMFLcxLl_V-1bcG5otOqdvb/view?usp=sharing) on GTA5
* Put the checkpoint under `./CSG/pretrained/`
* Put the code below in `train_seg.sh`
```bash
python train_seg.py \
--epochs 50 \
--switch-model deeplab50 \
--batch-size 6 \
--lr 1e-3 \ 
--num-class 19 \
--gpus 0 \
--factor 0.1 \
--csg 75 \
--apool \
--csg-stages 3.4 \
--chunks 8 \
--augment \
--evaluate \
--resume pretrained/csg_res101_segmentation_best.pth.tar \
```
* Change `--switch-model` (`deeplab50` or `deeplab101`) and `--resume` (path to pretrained checkpoints) accordingly.
* Run `CUDA_VISIBLE_DEVICES=0 bash train_seg.sh`
  - Please update the GPU index via `CUDA_VISIBLE_DEVICES` based on your need.

#### Train with CSG
* Put the code below in `train_seg.sh`
```bash
python train_seg.py \
--epochs 50 \
--switch-model deeplab50 \
--batch-size 6 \
--lr 1e-3 \ 
--num-class 19 \
--gpus 0 \
--factor 0.1 \
--csg 75 \
--apool \
--csg-stages 3.4 \
--chunks 8 \
--augment
```
* Change `--switch-model` (`deeplab50` or `deeplab101`) accordingly.
* Run `CUDA_VISIBLE_DEVICES=0 bash train_seg.sh`
  - Please update the GPU index via `CUDA_VISIBLE_DEVICES` based on your need.

 
## Citation
 
If you use this code for your research, please cite:
 
```BibTeX
@inproceedings{chen2020automated,
 author = {Chen, Wuyang and Yu, Zhiding and Shalini De Mello and Sifei Liu and Jose M. Alvarez and Wang, Zhangyang and Anandkumar, Anima},
 booktitle = {},
 pages = {},
 title = {Contrastive Syn-to-Real Generalization},
 year = {2021}
}
```
