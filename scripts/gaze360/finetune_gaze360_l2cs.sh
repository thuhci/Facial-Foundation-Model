python run_gaze360_finetuning.py \
    --model vit_base_dim512_no_depth_patch16_160 \
    --data_set Gaze360 \
    --data_path saved/data/gaze360 \
    --finetune saved/model/pretraining/voxceleb2/videomae_pretrain_base_dim512_local_global_attn_depth16_region_size2510_patch16_160_frame_16x4_tube_mask_ratio_0.9_e100_with_diff_target_server170/checkpoint-49.pth \
    --epochs 50 \
    --batch_siz 32 \
    --input_size 160 \
    --lr 1e-3 \
    --output_dir ./output/gazecapture_finetune_debug \
    --nb_classes 2 \
    --mixup 0 \
    --cutmix 0 \
    --num_sample 1 \
    --use_l2cs --num_bins 90 --alpha_reg 1.0 --bin_width 2\
    --warmup_epochs 2 \
    --world_size 3 \
    # --device cpu \
    # --enable_deepspeed