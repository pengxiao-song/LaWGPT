
python infer.py \
    --load_8bit True \
    --base_model 'minlik/chinese-llama-7b-merged' \
    --lora_weights 'entity303/lawgpt-lora-7b' \
    --infer_data_path './resources/example_infer_data.json' \
    --prompt_template 'law_template'
