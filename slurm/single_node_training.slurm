#!/bin/bash

## partition name
##  --partition=learnlab,devlab
## number of nodes
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
## number of tasks per node
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=640GB

#SBATCH --time=72:00:00

#SBATCH --output=./logs/%x/%x-%j.out
#SBATCH --error=./logs/%x/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL

source /home/yashs/miniconda3/etc/profile.d/conda.sh
conda activate perception_models
cd /home/yashs/projects/perception_models
echo "conda activated and slurm script working"

export OMP_NUM_THREADS=1
export HF_DATASETS_OFFLINE=1
export NCCL_TIMEOUT=120
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export NCCL_TIMEOUT=120
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_PRELOAD=/usr/local/cuda-12.4/lib/libnccl.so 
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH

# Check if a config file was provided as an argument
if [ -z "$1" ]; then
    echo "Error: No config file provided."
    exit 1
fi

CONFIG_FILE=$1
echo "Using config file: $CONFIG_FILE"

torchrun --nproc-per-node 8 -m apps.plm.train config=$CONFIG_FILE
conda deactivate
echo "finished training