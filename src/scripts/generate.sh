#!/bin/bash
CUDA_VISIBLE_DEVICES=1 python src/generate.py \
    --load_8bit \
    --base_model 'minlik/chinese-llama-7b-merged' \
    --lora_weights 'entity303/lawgpt-lora-7b' \
    --prompt_template 'law_template' \
    --share_gradio
