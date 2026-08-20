#!/bin/bash

set -e

EPOCHS=30
INPUT_SIZE=224
BATCH_SIZE=64

python experiments/train_tpu.py \
  --model shvit_s1_ratio_1_8 \
  --data-path /kaggle/working/cifar100 \
  --output-dir results_tpu/ratio_1_8 \
  --epochs $EPOCHS \
  --input-size $INPUT_SIZE \
  --batch-size $BATCH_SIZE

python experiments/train_tpu.py \
  --model shvit_s1_ratio_default \
  --data-path /kaggle/working/cifar100 \
  --output-dir results_tpu/ratio_default \
  --epochs $EPOCHS \
  --input-size $INPUT_SIZE \
  --batch-size $BATCH_SIZE

python experiments/train_tpu.py \
  --model shvit_s1_ratio_1_2 \
  --data-path /kaggle/working/cifar100 \
  --output-dir results_tpu/ratio_1_2 \
  --epochs $EPOCHS \
  --input-size $INPUT_SIZE \
  --batch-size $BATCH_SIZE

python experiments/train_tpu.py \
  --model shvit_s1_progressive \
  --data-path /kaggle/working/cifar100 \
  --output-dir results_tpu/progressive \
  --epochs $EPOCHS \
  --input-size $INPUT_SIZE \
  --batch-size $BATCH_SIZE