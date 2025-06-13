#!/bin/bash

conv_mode="llama_3"
ckpt="/fsx-checkpoints/yashs/plm/plm_1b_cambrian7M_stage2_1024_baseline1/checkpoints/0000007000/"
# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark)
        benchmark="$2"
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
if [[  -z "$benchmark" ]]; then
  echo "Error: --benchmark is a required argument."
  exit 1
fi


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
    # script="slurm/eval_$benchmark.slurm"
    # # default to eval_benchmark.slurm if the benchmark specific script does not exist
    # if [ ! -f $script ]; then
    script="slurm/eval_benchmark.slurm --benchmark $benchmark" 
    # fi
    job_name=${benchmark}-eval
    sbatch -J $job_name $script --ckpt $ckpt
done