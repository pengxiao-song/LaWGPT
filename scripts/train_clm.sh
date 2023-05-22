#!/bin/bash

WORLD_SIZE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=1235 train_clm.py \
    --base_model './models/base_models/chinese_llama_7b' \
    --data_path './data/train_clm_data.json' \
    --output_dir './outputs/train-clm' \
    --batch_size 128 \
    --micro_batch_size 8 \
    --num_epochs 1 \
    --learning_rate 0.0003 \
    --cutoff_len 1024 \
    --val_set_size 0 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj, v_proj, k_proj, o_proj]' \
    --train_on_inputs True \
    --add_eos_token True \
    --group_by_length True \
    --resume_from_checkpoint './outputs/train-clm'