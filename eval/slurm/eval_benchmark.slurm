#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10
#SBATCH --mem=640GB
#SBATCH --account=memorization
#SBATCH --time=72:00:00
#SBATCH --output=./logs/%x/%x-%j.out
#SBATCH --error=./logs/%x/%x-%j.err
#SBATCH --export=ALL
#SBATCH --mail-type=FAIL

echo "> eval_benchmark.slurm $@"
################# Parse Arguments #################

conv_mode="llama_3" 

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark) benchmark="$2"; shift 2 ;;
    --ckpt) ckpt="$2"; shift 2 ;;
    --name) name="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ -z "$ckpt" ]]; then
  echo "Error: --ckpt argument is required."
  exit 1
fi

if [[ -z "$name" ]]; then
  echo "Error: --name argument is required."
  exit 1
fi

echo "Benchmark: $benchmark"
echo "Checkpoint path: $ckpt"
echoo "Name: $name"
echo "Conversation mode: $conv_mode"

################# Setup Environment #################

source /home/yashs/miniconda3/etc/profile.d/conda.sh
conda activate perception_models
cd /home/yashs/projects/perception_models/eval
echo "Conda activated and slurm script working"

export OMP_NUM_THREADS=1
export TORCH_USE_CUDA_DSA=1
export LD_PRELOAD=/usr/local/cuda-12.4/lib/libnccl.so 
export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib/:$LD_LIBRARY_PATH
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1


benchmark_dir="/home/yashs/projects/perception_models/eval/eval/${benchmark}"
eval_file="${benchmark_dir}/${benchmark}_eval.py"
test_file="${benchmark_dir}/${benchmark}_test.py"

for file in "$benchmark_dir" "$eval_file" "$test_file"; do
  if [[ ! -e $file ]]; then
    echo "Error: $file does not exist."
    exit 1
  fi
done

cd $benchmark_dir
model_basename=$name

answers_dir="./answers"
if [[ ! -d "$answers_dir" ]]; then
  echo "Answers directory '$answers_dir' does not exist. Creating it."
  mkdir -p "$answers_dir"
fi

answers_file="$answers_dir/answers_${model_basename}.jsonl"
if [[ ! -f "$answers_file" ]]; then
  echo "Answers file '$answers_file' does not exist. Creating an empty file."
  touch "$answers_file"
fi

if [[ ! -d "$ckpt" ]]; then
  echo "Error: Checkpoint directory '$ckpt' does not exist. Please provide a valid local path."
  exit 1
fi

################# Handle Chunking #################

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}
echo "Number of GPUs: $CHUNKS"

################# Run Evaluation #################

current_date_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $current_date_time

for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" python $eval_file \
    --model_path "$ckpt" \
    --model_base "$name" \
    --num_chunks "$CHUNKS" \
    --chunk_idx "$IDX" \
    --answers_file "$answers_file" \
    --conv_mode "$conv_mode" &
done

wait

################# Combine Results #################

current_date_time=$(date "+%Y-%m-%d %H:%M:%S")
echo $current_date_time

if [[ -f "$answers_file" ]]; then
  mv "$answers_file" "${answers_file}.bak.$(date +%s)"
  echo "Moved existing answers file to ${answers_file}.bak.$(date +%s)"
fi


for IDX in $(seq 0 $((CHUNKS-1))); do
  idx_file="./answers/answers_${model_basename}_${IDX}.jsonl"
  if [[ -f "$idx_file" ]]; then
    cat "$idx_file" >> "$answers_file"
    rm "$idx_file"
  fi
done

################# Run Testing #####################

python $test_file --answers_file "$answers_file"

echo "Done evaluation and testing for $benchmark on model at $ckpt with conversation mode $conv_mode"
echo "Answers file: $(realpath "$answers_file")"

conda deactivate