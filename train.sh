# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under a NVIDIA Open Source Non-commercial license.)

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
# --resume pretrained/csg_res101_vista17_best.pth.tar \
# --evaluate
