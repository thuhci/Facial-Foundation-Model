#!/bin/bash

# Multi-GPU training script using Gloo backend for Gaze360 finetuning
# Usage: ./run_multi_gpu_gloo.sh [num_gpus] [config_file]

# Default values
NUM_GPUS=${1:-4}
CONFIG_FILE=${2:-"configs/gaze360_finetune_gloo.yaml"}
OUTPUT_DIR="./output/gaze360_finetune_gloo_${NUM_GPUS}gpu_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Multi-GPU Training with Gloo Backend"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Backend: Gloo (CPU communication)"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE not found"
    exit 1
fi

# Set environment variables
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting multi-GPU training with Gloo backend..."
echo "Command: torchrun --nproc_per_node=$NUM_GPUS --master_port=29504 run_finetuning_with_yacs.py --config $CONFIG_FILE --output_dir $OUTPUT_DIR"
echo "=========================================="

# Run training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29504 \
    run_finetuning_with_yacs.py \
    --config "$CONFIG_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --seed 42

echo "=========================================="
echo "Training completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
