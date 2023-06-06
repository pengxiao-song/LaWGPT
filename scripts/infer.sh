
python infer.py \
    --load_8bit True \
    --base_model 'minlik/chinese-llama-7b-merged' \
    --lora_weights 'entity303/lawgpt-lora-7b' \
    --prompt_template 'law_template' \
    --infer_data_path './resources/example_infer_data.json'