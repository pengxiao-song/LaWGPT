import argparse
import openai
import yaml
import sys
import random


def return_random_prompt():
    system_prompt = "你需要针对输入尽可能给出多样化的任务指令和对应的回答。我们将用于人工评估ChatGPT模型对指令的完成情况。要求:\n"

    # generate random tasks
    task_list = ["开放式生成", "分类", "问答", "编辑", "摘要", "写作", "分析", "抽取"]
    system_prompt += "1. 表述多样化，结合真实问题；指令类型多样化，例如：" + "、".join(random.sample(task_list, 7)) + "等。\n"

    # other requirements
    system_prompt += "2. 如果遇到无法处理的指令（只靠文本无法回答），给出无法处理的回复。\n"
    system_prompt += "3. 除非特别要求，请使用中文，指令可以是命令句、疑问句、或其他合适的类型。\n"
    system_prompt += "4. <input>是：'第十三条 一切危害国家主权、领土完整和安全，分裂国家、颠覆人民民主专政的政权和推翻社会主义制度，破坏社会秩序和经济秩序，侵犯国有财产或者劳动群众集体所有的财产，侵犯公民私人所有的财产，侵犯公民的人身权利、民主权利和其他权利，以及其他危害社会的行为，依照法律应当受刑罚处罚的，都是犯罪，但是情节显著轻微危害不大的，不认为是犯罪。'"
    system_prompt += "5. <output>应该是对指令的适当且真实的回应，不能只回复答应或拒绝请求。如果需要额外信息才能回复时，请努力预测用户意图并尝试回复。<output>的内容应少于" + str(random.randint(128, 512)) + "字。\n\n"
    system_prompt += "请给出满足条件的20条JSON格式数据：\n"

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
