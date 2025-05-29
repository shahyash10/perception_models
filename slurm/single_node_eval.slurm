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
export NCCL_TIMEOUT=120
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1
export LD_PRELOAD=/usr/local/cuda-12.4/lib/libnccl.so 
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH


CHECKPOINTS_PATH=/fsx-checkpoints/yashs/plm/plm_1b_cambrian_stage1_shuffled/checkpoints/0000008000/
# SELECTED_TASK="docvqa,chartqa,textvqa,infovqa,ai2d_no_mask,ok_vqa,vizwiz_vqa,mme,pope,ocrbench,coco_karpathy_val,nocaps,vqav2_val"
SELECTED_TASK="mmmu,docvqa,chartqa,ocrbench,realworldqa,mme,mmbench_en_dev,ok_vqa,vizwiz_vqa,vqav2_val"
OUTPUT_PATH=/fsx-checkpoints/yashs/plm/eval_results/7M_stage1_checkpoint/
echo "checkpoint path: $CHECKPOINTS_PATH"
echo "selected task: $SELECTED_TASK"
accelerate launch --num_processes=8 \
-m lmms_eval \
--model plm \
--model_args pretrained=$CHECKPOINTS_PATH \
--tasks $SELECTED_TASK \
--batch_size 1 \
--log_samples \
--log_samples_suffix plm \
--output_path $OUTPUT_PATH \
--verbosity DEBUG
conda deactivate
echo "finished evaluations"