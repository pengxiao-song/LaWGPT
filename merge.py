import os

import torch
import transformers
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer  # noqa: F402


import argparse
parser = argparse.ArgumentParser(description='Merge Base Model and Lora')
parser.add_argument('--base_model', type=str, default="minlik/chinese-llama-7b-merged", help='base model path')
parser.add_argument('--lora_model', type=str, default="entity303/legal-lora-7b", help='lora model path')
parser.add_argument('--output_dir', type=str, default="./models/base_models/llama-7b-legal-lora-merged", help='output model path')
args = parser.parse_args()

BASE_MODEL = args.base_model
LORA_MODEL = args.lora_model
OUTPUT_DIR = args.output_dir


assert (
    BASE_MODEL
), "Please specify a value for BASE_MODEL environment variable, e.g. `export BASE_MODEL=huggyllama/llama-7b`"  # noqa: E501


print(f"{'*'*20} Using base model: {BASE_MODEL} {'*'*20}")
print(f"{'*'*20} Using lora model: {LORA_MODEL} {'*'*20}")
print(f"{'*'*20} Saving to: {OUTPUT_DIR} {'*'*20}")

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

base_model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

first_weight = base_model.model.layers[0].self_attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    LORA_MODEL,
    device_map={"": "cpu"},
    torch_dtype=torch.float16,
)

lora_weight = lora_model.base_model.model.model.layers[
    0
].self_attn.q_proj.weight

assert torch.allclose(first_weight_old, first_weight)

# merge weights - new merging method from peft
lora_model = lora_model.merge_and_unload()

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()
deloreanized_sd = {
    k.replace("base_model.model.", ""): v
    for k, v in lora_model_sd.items()
    if "lora" not in k
}

LlamaForCausalLM.save_pretrained(
    base_model, OUTPUT_DIR, state_dict=deloreanized_sd, max_shard_size="2048MB"
)

LlamaTokenizer.save_pretrained(tokenizer, OUTPUT_DIR)