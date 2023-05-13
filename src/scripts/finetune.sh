#!/bin/bash

WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1234 finetune.py \
    --base_model 'minlik/chinese-llama-7b-merged' \
    --data_path '' \
    --output_dir './outputs/LawGPT' \
    --prompt_template_name 'law_template' \
    --micro_batch_size 16 \
    --batch_size 128 \
    --num_epochs 3 \
    --val_set_size 10000 \
    --lora_target_modules='[q_proj,k_proj,v_proj,o_proj]' \
    --lora_r 16 \
    --lora_alpha 32 \
    --learning_rate 3e-4 \
    --cutoff_len 512 \
    --resume_from_checkpoint './outputs/LawGPT' \