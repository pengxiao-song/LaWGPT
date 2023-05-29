#!/bin/bash


# 使用huggingface上已经训练好的模型
python webui.py \
    --load_8bit False \
    --base_model 'minlik/chinese-alpaca-plus-7b-merged' \
    --lora_weights 'entity303/lawgpt-lora-7b-v2' \
    --prompt_template "law_template" \
    --server_name "0.0.0.0" \
    --share_gradio True \


# 使用自己finetune的lora, 把自己的模型放到对应目录即可
# python webui.py \
#     --load_8bit True \
#     --base_model 'minlik/chinese-alpaca-plus-7b-merged' \
#     --lora_weights './outputs/chinese-alpaca-plus-7b-law-e1' \
#     --prompt_template "alpaca" \
#     --server_name "0.0.0.0" \
#     --share_gradio True \