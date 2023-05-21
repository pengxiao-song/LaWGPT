
# LawGPT
python infer.py \
    --base_model 'minlik/chinese-alpaca-plus-7b-merged' \
    --lora_weights './outputs/chinese-alpaca-plus-7b-law-e1' \
    --instruct_dir './data/infer_law_data.json' \
    --prompt_template 'alpaca'


# Chinese-Alpaca-plus-7B
python infer.py \
    --base_model 'minlik/chinese-alpaca-plus-7b-merged' \
    --lora_weights '' \
    --instruct_dir './data/infer_law_data.json' \
    --prompt_template 'alpaca'

