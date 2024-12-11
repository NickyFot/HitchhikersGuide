#!/bin/bash

#conda activate ffvqa

# DDP settings
num_gpus=1

# set environment variables
#export CUDA_VISIBLE_DEVICES="0"

#export TORCH_DISTRIBUTED_DEBUG=INFO

experiment_config="experiments/FF_Blip.yaml"
experiment_name=$(date "+%Y-%m-%d")

port=$(shuf -i 1300-1399 -n 1)

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$port --nproc_per_node=$num_gpus --nnodes=1 --node_rank 0  main.py \
  --exp_name=$experiment_name \
  --config_file=$experiment_config
