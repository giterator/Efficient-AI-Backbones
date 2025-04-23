#!/bin/bash

# Script to train Graphormer model on ImageNet with specific configurations

python train.py /scratch/graphormer_imagenet/imagenet \
  --model vig_ti_224_gelu \
  --sched cosine \
  --epochs 30 \
  --opt adamw \
  -j 8 \
  --warmup-lr 1e-6 \
  --mixup 0.8 \
  --cutmix 1.0 \
  --model-ema \
  --model-ema-decay 0.99996 \
  --aa rand-m9-mstd0.5-inc1 \
  --color-jitter 0.4 \
  --warmup-epochs 20 \
  --opt-eps 1e-8 \
  --repeated-aug \
  --remode pixel \
  --reprob 0.25 \
  --amp \
  --lr 2e-3 \
  --weight-decay 0.05 \
  --smoothing 0.1 \
  --drop 0 \
  --drop-path 0.1 \
  -b 1024 \
  --output /afs/ece.cmu.edu/usr/oanaveka/Private/11785/Graphormer/Graphormer/outputs/Graphormer \
  --pretrain_path /scratch/vig_ckpts/vig_ti_74.5.pth

