#!/bin/bash

# Working directory
WORKING_DIR=$HOME/projects

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=$WORKING_DIR/vae.tensorflow.slim/exp1


CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --train_dir=${TRAIN_DIR} \
    --batch_size=64 \
    --max_steps=100000 \
    --save_steps=1000 \
