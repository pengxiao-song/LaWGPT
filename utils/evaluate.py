import math
import os
import sys

import fire
from tqdm import tqdm
import pandas as pd
import torch
import transformers
from peft import PeftModel
import datasets
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

device = "cuda"


def main(
    load_8bit: bool = True,
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = "./lora-alpaca",
    data_path: str = "./data",
    output_path: str = "./output",
    eval_rate: float = 0.1,
    batch_size: int = 32,
    # The prompt template to use, will default to alpaca.
    prompt_template: str = "alpaca",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (base_model), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate_one(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=2,
        max_new_tokens=128,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s, skip_special_tokens=True)
        return prompter.get_response(output)

    def evaluate_all():
        # data = datasets.load_dataset("json", data_files=data_path)
        # data = data["train"]
        # df = data.to_pandas()
        df = pd.read_json(data_path, orient='records')
        print(df.info())
        # 计算准确率
        correct = 0
        total = 0
        total_step = len(df)
        pbar = tqdm(total=total_step, unit='batch')
        error = []
        for i in range(total_step):
            instruction = df['instruction'].iloc[i]
            input = df['input'].iloc[i]
            label = df['output'].iloc[i]
            pred = evaluate_one(instruction=instruction, input=input)
            if pred == label:
                correct += 1
            else:
                error.append((label, pred))
            total += 1
            acc = correct / total
            # 更新进度条
            # Update the progress bar
            pbar.set_description(
                f"Testing: Sample [{total}/{total_step}] Acc: {acc :.4f}")
            pbar.update(1)

        for e in error:
            print(e)

    def evaluate_by_batch(
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=32
    ):
        df = pd.read_json(data_path, orient='records')
        # df = df.sample(frac=eval_rate).reset_index(drop=True)
        df['prompt'] = df.apply(lambda x: prompter.generate_prompt(
            x['instruction'], x['input']), axis=1)
        tokenizer.padding_side = "left"  # Allow batched inference

        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams
        )

        outputs = []
        total = 0
        total_step = math.ceil(len(df) / batch_size)
        pbar = tqdm(total=total_step, unit='batch')
        # 计算准确率
        with torch.no_grad():
            for i in range(total_step):
                batch = df.iloc[i*batch_size:(i+1)*batch_size]
                inputs = tokenizer(batch['prompt'].tolist(), return_tensors="pt", padding=True)[
                    'input_ids'].to(device)

                generation_outputs = model.generate(
                    input_ids=inputs,
                    generation_config=generation_config,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id
                )

                for g in generation_outputs:
                    decoded_item = tokenizer.decode(
                        g, skip_special_tokens=True)
                    try:
                        output = prompter.get_response(decoded_item)
                    except:
                        output = decoded_item
                    outputs.append(output)
                    total += 1

                # 更新进度条
                pbar.set_description(f"Testing: Sample [{total}/{len(df)}] ")
                pbar.update(1)
        df['pred'] = outputs
        df['pred'].to_csv(output_path, index=False)

    evaluate_by_batch()


if __name__ == "__main__":
    # fire.Fire(main)
    import yaml
    dataset_param = sys.argv[1]
    with open("./configs/evaluate_params.yaml", "r") as stream:
        # try:
        params = yaml.safe_load(stream)
        print('=' * 80)
        print(params[dataset_param])
        print('=' * 80)

    # fire.Fire(train)
    main(**params[dataset_param])
