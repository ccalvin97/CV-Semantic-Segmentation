#!/bin/bash

## Start for HRNet ##

python tools/train.py \
--cfg /home/GDDC-CV1/Desktop/CV-Semantic-Segmentation/HRNet-Semantic-Segmentation/experiments/seg_hrnet_w18_small_v1_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml

: << !
Please specify the configuration file.

For example, train the HRNet-W48 on Cityscapes with a batch size of 12 on 4 GPUs:

python tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
!