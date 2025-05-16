#!/bin/bash

## partition name
## --partition=learnfair
## number of nodes
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
## number of tasks per node
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=640GB

#SBATCH --time=72:00:00

#SBATCH --output=./logs/%x/%x-%j.out
#SBATCH --error=./logs/%x/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL

source /private/home/yashs/miniconda3/etc/profile.d/conda.sh
conda activate perception_models_eval
cd /private/home/yashs/projects/perception_models/
echo "conda activated and slurm script working"

export OMP_NUM_THREADS=1
export NCCL_TIMEOUT=120
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_USE_CUDA_DSA=1
export CUDA_LAUNCH_BLOCKING=1


CHECKPOINTS_PATH=/checkpoint/yashs/logs/plm/plm_1b_stage2/checkpoints/0000004000
SELECTED_TASK="docvqa,chartqa,textvqa,infovqa,ai2d_no_mask,ok_vqa,vizwiz_vqa,mme,realworldqa,pope,mmmu,ocrbench,coco_karpathy_val,nocaps,vqav2_val"
OUTPUT_PATH=/private/home/yashs/projects/perception_models/eval_results/trial1/

head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

srun accelerate launch \
--num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
--num_machines $SLURM_NNODES \
--rdzv_backend c10d \
--main_process_ip $head_node_ip \
--main_process_port 29500 \
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
echo "finished training