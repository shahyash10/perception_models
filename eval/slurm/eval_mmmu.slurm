#!/bin/bash

## partition name
## --partition=learnlab,devlab
## number of nodes
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
## number of tasks per node
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=640GB
#SBATCH --account=memorization
#SBATCH --time=72:00:00

#SBATCH --output=./logs/%x/%x-%j.out
#SBATCH --error=./logs/%x/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL


echo "> eval_mmmu.slurm $@"
benchmark="mmmu"

################# Parse Arguments #################

# take command line args
# Default values
conv_mode="llama_3"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
        ckpt="$2"
        shift 2
        ;;
    --conv_mode)
        conv_mode="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done

# Check if the required argument ckpt is provided
if [[ -z "$ckpt" ]]; then
  echo "Error: --ckpt argument is required."
  exit 1
fi

# Print the values
echo "Benchmark: $benchmark"
echo "Checkpoint path: $ckpt"
echo "Conversation mode: $conv_mode"

# Initialize the Conda environment for zsh
source /home/yashs/miniconda3/etc/profile.d/conda.sh
conda activate perception_models
cd /home/yashs/projects/perception_models
echo "conda activated and slurm script working"

export OMP_NUM_THREADS=1
export TORCH_USE_CUDA_DSA=1
export LD_PRELOAD=/usr/local/cuda-12.4/lib/libnccl.so 
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
bash /home/yashs/projects/perception_models/eval/scripts/run_benchmark.sh --benchmark $benchmark --ckpt $ckpt --conv_mode $conv_mode

conda deactivate