# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

python train_seg.py \
--epochs 50 \
--switch-model deeplab101 \
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
# --resume pretrained/csg_res50_segmentation_best.pth.tar \
