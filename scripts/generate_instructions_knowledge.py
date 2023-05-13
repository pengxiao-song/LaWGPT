import argparse
import openai
import yaml
import random


def return_random_prompt():
    system_prompt = "你需要针对法条内容尽可能联想多样化的场景生成问答数据。我们将用于人工评估 ChatGPT 模型对指令的完成情况。要求:\n"

    # generate random tasks
    system_prompt += "1. 结合真实问题，表述多样化。\n"

    # other requirements
    system_prompt += "2. 如果遇到无法处理的指令（只靠文本无法回答），给出无法处理的回复。\n"
    system_prompt += "3. 除非特别要求，请使用中文，指令可以是命令句、疑问句、或其他合适的类型。\n"
    system_prompt += "4. <Reference>：违反本法规定，对妇女实施性骚扰的，由公安机关给予批评教育或者出具告诫书，并由所在单位依法给予处分。\n学校、用人单位违反本法规定，未采取必要措施预防和制止性骚扰，造成妇女权益受到侵害或者社会影响恶劣的，由上级机关或者主管部门责令改正；拒不改正或者情节严重的，依法对直接负责的主管人员和其他直接责任人员给予处分。\n"
    system_prompt += "5. <input>是结合法条内容联想到的真实场景下的问题。要求该场景下存在违法者和受害人\n"
    system_prompt += "6. <output>是结合法条内容对该问题的适当且真实的回应，不能只回复答应或拒绝请求。尽可能地指明违法行为可能遭受的惩罚，并向受害者提出维权建议。\n\n"
    system_prompt += "请给出满足条件的10条JSON格式数据：\n"

    return system_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_path', default='../config.yaml', type=str)
    parser.add_argument('--save_path', default='./output.json', type=str)
    args = parser.parse_args()

    with open(args.cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    openai.api_key = cfg['API_KEY']
    openai.api_base = cfg['API_BASE_URL']

    output_file = open(args.save_path, 'w')

    # number of data to generate (each prompt contains 20 JSON-formatted data)
    # TODO: 改成流式的，不然会中途断掉
    MAX_EPOCHS = 1
    for k in range(MAX_EPOCHS):
        response = openai.ChatCompletion.create(
            # here we use `gpt-3.5-turbo` model, while Stanford-Alpaca uses `text-davinci-003`
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": return_random_prompt()},
            ]
        )
        output_file.write(response["choices"][0]["message"]["content"] + '\n')
    output_file.close()
