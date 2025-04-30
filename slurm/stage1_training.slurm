#!/bin/bash

## partition name
#SBATCH --partition=learnlab,devlab
## number of nodes
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
## number of tasks per node
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=640GB

#SBATCH --time=72:00:00

#SBATCH --output=./logs/%x/eval-%x-%j.out
#SBATCH --error=./logs/%x/eval-%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL

source /private/home/yashs/miniconda3/etc/profile.d/conda.sh
conda activate perception_models
cd /private/home/yashs/projects/perception_models/
echo "conda activated and slurm script working"
torchrun --nproc-per-node 8 -m apps.plm.train config=apps/plm/configs/stage_1/plm_1b.yaml
conda deactivate
echo "finished training