# LaWGPT：Large language model based on Chinese legal knowledge

<p align="center">
  <a href="assets/logo/lawgpt.jpeg">
    <img src="./assets/logo/lawgpt.jpeg" width="80%" >
  </a>
</p>

<p align="center">
    <a href="https://github.com/pengxiao-song/LaWGPT/wiki"><img src="https://img.shields.io/badge/docs-Wiki-brightgreen"></a>
    <a href=""><img src="https://img.shields.io/badge/version-beta1.0-blue"></a>
    <a href=""><img src="https://img.shields.io/github/last-commit/pengxiao-song/lawgpt"></a>
    <!-- <a href="https://www.lamda.nju.edu.cn/"><img src="https://img.shields.io/badge/support-NJU--LAMDA-9cf.svg"></a> -->
</p>

LaWGPT is a series of open source large language models based on Chinese legal knowledge.

This series of models expands the special vocabulary in the legal field on the basis of general Chinese base models (such as Chinese-LLaMA, ChatGLM, etc.), **large-scale Chinese legal corpus pre-training**, and strengthens the foundation of large models in the legal field semantic understanding. On this basis, **constructed the dialogue question-and-answer data set in the legal field and the Chinese judicial examination data set to fine-tune the instructions**, which improved the model's ability to understand and execute legal content.

Please refer to [Technical Report]() for details.

---

This project continues to develop, and the data sets and series of models in the legal field will be open sourced one after another, so stay tuned.

## renew

- 🪴 2023/05/15 (Awesome Chinese Legal Resources）](https://github.com/pengxiao-song/awesome-chinese-legal-resources(https://git)hub.com/pengxiao-song/LaWGPT/blob/main/resources/legal_vocab.txt)
- 🌟 2023/05/13: Public release
  <a href=""><img src="https://img.shields.io/badge/Model-Legal--Base--7B-blue"></a>
  <a href=""><img src="https://img.shields.io/badge/Model-LaWGPT--7B--beta1.0-yellow"></a>
  
  - **Legal-Base-7B**: legal base model, using 50w Chinese judgment document data for secondary pre-training
  
  - **LaWGPT-7B-beta1.0**: Legal dialogue model, constructing a 30w high-quality legal question-and-answer dataset based on Legal-Base-7B instruction fine-tuning
  
- 🌟 2023/04/12: internal testing
  <a href=""><img src="https://img.shields.io/badge/Model-Lawgpt--7B--alpha-yellow"></a>
  - **LaWGPT-7B-alpha**: On the basis of Chinese-LLaMA-7B, directly construct a 30w legal question answering data set instruction fine-tuning
  ## Quick start

1. Prepare the code and create the environment

   ```bash
   git clone git@github.com:pengxiao-song/LaWGPT.git
   cd LaWGPT
   conda activate lawgpt
   pip install -r requirements.txt
   ```

2. Combine model weights (optional)
**If you want to use the LaWGPT-7B-alpha model, you can skip this step and go directly to step 3.**

   If you want to use the LaWGPT-7B-beta1.0 model:

   Since neither [LLaMA](https://github.com/facebookresearch/llama) nor [Chinese-LLaMA](https://github.com/ymcui/Chinese-LLaMA-Alpaca) open source the model weights. According to the corresponding open source license, **This project can only release the LoRA weight**, and cannot release the complete model weight, please understand.
   
   This project gives [Merge Method](https://github.com/pengxiao-song/LaWGPT/wiki/%E6%A8%A1%E5%9E%8B%E5%90%88%E5%B9%B6) , please obtain the original copyright and reconstruct the model by yourself.


3. Start the example

   Start the local service:

   ```bash
   conda activate lawgpt
   cd LaWGPT
   sh src/scripts/generate.sh
   ```
   Access service:

   <p align="center">
      <img src="./assets/demo/demo.png" width="80%">
   </p>


## Project structure

```bash
LaWGPT
├── assets # project static resources
├── data # corpus and fine-tuning data
├── tools # Data cleaning and other tools
├── README.md
├── requirements.txt
└── src # source code
├── finetune.py
    ├── generate.py
    ├── models # Base model and Lora weight
    │ ├── base_models
    │ └── lora_weights
    ├── outputs
    ├── scripts # script file
    │ ├── finetune.sh # instruction fine-tuning
    │ └── generate.sh # service creation
    ├── templates
    └── utils
```


## Data construction

This project is based on the data sets of legal documents and judicial examination data published by the Chinese Judgment Documents Network. For details, please refer to [Chinese Legal Data Summary]()
This project is based on the data sets of legal documents and judicial examination data published by the Chinese Judgment Documents Network. For details, please refer to [Chinese Legal Data Summary]()

1. Primary data generation: According to [Stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca#data-generation-process) and [self-instruct](https://github.com/yizhongw/self- instruct) to generate dialogue question and answer data
2. Knowledge-guided data generation: Generate data based on Chinese legal structured knowledge through the Knowledge-based Self-Instruct method.
3. Introduce ChatGPT cleaning data to assist in the construction of high-quality data sets.

## Model training

The training process of LawGPT series models is divided into two stages:

1. The first stage: expanding the vocabulary in the legal field, pre-training Chinese-LLaMA on large-scale legal documents and code data
2. The second stage: Construct a dialogue question-and-answer dataset in the legal field, and fine-tune instructions based on the pre-trained model

### Secondary training process
1. Refer to `src/data/example_instruction_train.json` to construct a secondary training dataset
2. Run `src/scripts/train_lora.sh`

### Command fine-tuning steps

1. Refer to `src/data/example_instruction_tune.json` to construct instruction tuning dataset
2. Run `src/scripts/finetune.sh`

### Computing resources

8 Tesla V100-SXM2-32GB: the second training phase takes about 24h / epoch, and the fine-tuning phase takes about 12h / epoch

## Model Evaluation

### output example

<details><summary>Question: Please give a verdict. </summary>
![](assets/demo/example-05.jpeg)

</details>

<details><summary>Question: Please introduce the definition of gambling crime. </summary>

![](assets/demo/example-06.jpeg)

</details>

<details><summary>Question: How to calculate overtime wages? </summary>

![](assets/demo/example-04.jpeg)

</details>

<details><summary>Question: What is the legal interest of private lending protected by the state?</summary>

![](assets/demo/example-02.jpeg)

</details>

<details><summary>Question: Will I go to jail if I owe credit card money? </summary>

![](assets/demo/example-01.jpeg)

</details>

<details><summary>Question: Can you write a description of the robbery charge? </summary>

![](assets/demo/example-03.jpeg)

</details>

### Limitations

Due to the limitations of computing resources, data scale and other factors, LawGPT has many limitations at the current stage:

1. Limited data resources and small model capacity lead to relatively weak model memory and language ability. Therefore, incorrect results may be generated when faced with factual knowledge tasks.
2. This family of models only performs preliminary human intent alignment. Therefore, unpredictable harmful content and content that does not conform to human preferences and values ​​may be produced.
3. There are problems in self-awareness ability, and Chinese comprehension ability needs to be improved.

Please understand the above problems before use, so as to avoid misunderstanding and unnecessary trouble.


## Collaborators

The following cooperation (in alphabetical order): [@cainiao](https://github.com/herobrine19), [@njuyxw](https://github.com/njuyxw), [@pengxiao-song]( https://github.com/pengxiao-song)


## Disclaimer

Please strictly abide by the following agreement:

1. Any resources in this project are only for academic research use, and any commercial use is strictly prohibited**.
2. The output of the model is affected by various uncertain factors. This project cannot guarantee its accuracy at present. **It is strictly forbidden to be used in real legal scenarios**.
3. This project does not assume any legal responsibility, nor is it responsible for any losses that may arise from the use of related resources and output results.

## feedback

If you have problems, please submit them in GitHub Issue.

- Before submitting a question, it is recommended to check the FAQ and previous issues to see if it can solve your problem.
- Please discuss politely to build a harmonious community.

The collaborators promote the progress of the project in addition to scientific research. Due to limited manpower, it is difficult to give real-time feedback, which may cause inconvenience to you. Sorry for your understanding!


## Acknowledgments

This project is based on the following open source projects, and I would like to express my sincere thanks to the relevant projects and developers:

- Chinese-LLaMA-Alpaca: https://github.com/ymcui/Chinese-LLaMA-Alpaca
-LLaMA: https://github.com/facebookresearch/llama
- Alpaca: https://github.com/tatsu-lab/stanford_alpaca
- alpaca-lora: https://github.com/tloen/alpaca-lora
- ChatGLM-6B: https://github.com/THUDM/ChatGLM-6B

In addition, this project is based on open data resources, see [Awesome Chinese Legal Resources](https://github.com/pengxiao-song/awesome-chinese-legal-resources) for details, and thank you.


## quoting

If you find our work helpful to you, please consider citing this project.


