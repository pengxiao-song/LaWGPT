import sys
import json

import fire
import torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"


def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "",
    infer_data_path: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
):
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    try:
        print(f"Using lora {lora_weights}")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    except:
        print("*"*50, "\n Attention! No Lora Weights \n", "*"*50)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
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
            # repetition_penalty=10.0,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        return prompter.get_response(output)

    def infer_from_file():
        with open(infer_data_path) as f:
            for line in f:
                data = json.loads(line)
                instruction = data["instruction"]
                output = data["output"]
                print('=' * 100)
                print(f"Base Model: {base_model}    Lora Weights: {lora_weights}")
                print("Instruction:\n", instruction)
                model_output = evaluate(instruction)
                print("Model Output:\n", model_output)
                print("Ground Truth:\n", output)
                print('=' * 100)

    try:
        infer_from_file()
    except:
        print("Read infer_data_path Failed! Now Interactive Mode: ")
        while True:
            print('=' * 100)
            instruction = input("请输入您的问题: ")
            print("LaWGPT:")
            print(evaluate(instruction))
            print('=' * 100)


if __name__ == "__main__":
    fire.Fire(main)
