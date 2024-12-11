#!/bin/bash
#SBATCH --array=1-9%5                      # 9 array jobs, 5 at a time
#SBATCH --nodes=1                          # number of nodes
#SBATCH --ntasks=4                         # number of tasks
#SBATCH --time=16:01:00                    # time (HH:MM:SS)

# environ
export PATH="~/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh

source activate ffvqa

#export TORCH_DISTRIBUTED_DEBUG=INFO

# DDP settings
num_gpus=4

echo $SLURM_ARRAY_TASK_ID
experiments=($(ls 'experiments/' | grep Clip))
experiments_config=${experiments[$SLURM_ARRAY_TASK_ID]}
echo $experiments_config
experiment_config="experiments/$experiments_config"
echo $experiment_config
experiment_name=$(date "+%Y-%m-%d")

port=$(shuf -i 1300-1399 -n 1)

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:$port --nproc_per_node=$num_gpus --nnodes=1 --node_rank 0  clip_eval.py \
  --exp_name=$experiment_name \
  --config_file=$experiment_config
