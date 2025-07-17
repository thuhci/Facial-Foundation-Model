#!/bin/bash

# 配置参数
server=165
pretrain_dataset='voxceleb2'
pretrain_server=170
finetune_dataset='gaze360'
input_size=160
lr=1e-3
epochs=50
batch_size=32
num_gpus=4  # 使用的GPU数量

# 输出目录
OUTPUT_DIR="./output/gaze360_finetune_distributed_lr${lr}_epoch${epochs}_bs${batch_size}_server${server}"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p $OUTPUT_DIR
fi

# 数据路径
DATA_PATH="saved/data/gaze360"
# 预训练模型路径
MODEL_PATH="saved/model/pretraining/voxceleb2/videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target_server170/checkpoint-49.pth"

# 启动分布式训练
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node ${num_gpus} --master_port 11223 \
    run_gaze360_finetuning.py \
    --model vit_base_dim512_no_depth_patch16_160 \
    --data_set Gaze360 \
    --data_path ${DATA_PATH} \
    --finetune ${MODEL_PATH} \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --epochs ${epochs} \
    --batch_size ${batch_size} \
    --input_size ${input_size} \
    --short_side_size ${input_size} \
    --lr ${lr} \
    --nb_classes 2 \
    --mixup 0 \
    --cutmix 0 \
    --num_sample 1 \
    --warmup_epochs 5 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --save_ckpt_freq 1 \
    --num_workers 8 \
    # --dist_eval \
    >${OUTPUT_DIR}/training.log 2>&1 &

echo "分布式训练已启动，使用 ${num_gpus} 张GPU"
echo "输出目录: ${OUTPUT_DIR}"
echo "日志文件: ${OUTPUT_DIR}/training.log"
echo "可以使用 tail -f ${OUTPUT_DIR}/training.log 查看训练日志"