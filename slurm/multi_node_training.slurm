#!/bin/bash

## partition name
## --partition=learnlab,devlab
## number of nodes
#SBATCH --nodes=16 
#SBATCH --gpus-per-node=8
## number of tasks per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=640GB
#SBATCH --verbose

#SBATCH --time=72:00:00

#SBATCH --output=./logs/%x/%x-%j.out
#SBATCH --error=./logs/%x/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL

source /home/yashs/miniconda3/etc/profile.d/conda.sh
conda activate perception_models
cd /home/yashs/projects/perception_models
echo "conda activated and slurm script working"

echo "about to run on $SLURM_JOB_NUM_NODES nodes"
export OMP_NUM_THREADS=1
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500 
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
echo "Number of GPU nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks created by SLURM per GPU node: $SLURM_NTASKS_PER_NODE"

# Run torchrun with rendezvous configuration
srun torchrun \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc-per-node 8 \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    -m apps.plm.train config=$CONFIG_FILE

conda deactivate
echo "finished training"