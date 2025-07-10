#!/bin/bash

conv_mode="llama_3"
# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --ckpt)
        ckpt="$2"
        shift 2
        ;;
    --name)
        name="$2"
        shift 2
        ;;
    --help)
        echo "$helpmsg"
        exit 1
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
  esac
done

# Check if the required arguments benchmark and ckpt are provided
if [[ -z "$benchmark" ]]; then
  echo "Error: --benchmark is a required argument."
  exit 1
fi

if [[ -z "$ckpt" ]]; then
  echo "Error: --ckpt is a required argument."
  exit 1
fi

if [[ -z "$name" ]]; then
  echo "Error: --name is a required argument."
  exit 1
fi

# Check if consolidated.pth exists
if [[ ! -f "$ckpt/consolidated.pth" ]]; then
  echo "consolidated.pth not found in $ckpt. Running Python command."
  source /home/yashs/miniconda3/etc/profile.d/conda.sh
  conda activate perception_models
  python /home/yashs/projects/perception_models/apps/plm/consolidate.py --ckpt $ckpt
  conda deactivate
  echo "consolidated.pth created successfully."
fi

echo "ckpt: $ckpt"
ckpt_parent_dir=$(dirname "$ckpt")
ckpt_dir="$ckpt_parent_dir/$(basename "$ckpt")"
for folder in "$ckpt_parent_dir"/*; do
  if [[ -d "$folder" && "$folder" != "$ckpt_dir" ]]; then
    echo "Deleting folder: $folder"
    rm -rf "$folder"
  fi
done

if [ "$benchmark" = "all" ]; then
    benchmarks=(
        # vqav2
        gqa
        vizwiz
        scienceqa
        textvqa
        pope
        mme
        mmbench_en
        mmbench_cn
        seed
        # llava_w
        mmvet
        mmmu
        mathvista
        ai2d
        chartqa
        docvqa
        infovqa
        stvqa
        ocrbench
        mmstar
        realworldqa
        mmvp
        vstar
        synthdog
        # vision
        qbench
        blink
        # CV-Bench
        omni
        ade
        coco
    )
else
    if [ ! -d "eval/$benchmark" ]; then # check that the eval/$benchmark directory exists
        echo "Error: eval/$benchmark directory does not exist. Benchmark $benchmark may not be supported."
        exit 1
    fi
    benchmarks=($benchmark)
fi

for benchmark in "${benchmarks[@]}"; do
    script="slurm/eval_benchmark.slurm --benchmark $benchmark" 
    # fi
    job_name=${benchmark}-eval
    sbatch -J $job_name $script --ckpt $ckpt --name $name
done