####### GPT-3 ZERO/FEW SHOT 

import openai
from openai import OpenAI
import json
from sklearn.metrics import precision_recall_fscore_support


client = OpenAI()
openai.api_key = ""  #insert your openai key

inp_list = []
with open("INPUT_PROMPT_SAMPLE.jsonl") as f:  #input is a list of dicts with prompt and completion keys
    for line in f:
        line = json.loads(line)
        inp_list.append(line)

ground_truth_list = []
for dic in inp_list:
    gt = dic['completion']
    gt = gt.strip()
    gt = gt.replace('\n', '')
    gt = gt.strip()
    ground_truth_list.append(gt)

prompt_list = []
for dic in inp_list:
    sent_here = dic['prompt']
    sent_here = "Decide in a single word if the drug response is sensitive or resistant.\n" + sent_here
    sent_here = sent_here + ' Drug response:\n'
    sent_here = sent_here + '\n\n###\n\n'
    prompt_list.append(sent_here)


pred_list = []
gt_list = []
for i in range(len(prompt_list)):
    sent = prompt_list[i]
    gt = ground_truth_list[i]
    output = client.completions.create(model='babbage-002',prompt=sent,max_tokens=2,logprobs=2) #insert openai base model name 
    output = dict(output)
    output_dict = dict(output['choices'][0])
    text = output_dict['text']
    text = text.split('#')
    text = list(filter(None, text))
    text = [tok.replace('\n', '') for tok in text]
    text = [tok.strip() for tok in text]
    text = list(filter(None, text))
    if text:
        if 'sensitive' in text:
            gt_list.append(gt)
            pred_list.append('sensitive')
        elif 'resistant' in text:
            gt_list.append(gt)
            pred_list.append('resistant')


p, r, f, _ = precision_recall_fscore_support(gt_list, pred_list, average='micro')

print('f1: ', f)
