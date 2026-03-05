import json
import os
from random import shuffle

dataset = json.load(open("/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/kvasir_doctor_exam/final_doctor_exam_62.json"))
new_dataset = []
options = {
    0: "option_A",
    1: "option_B",
    2: "option_C",
    3: "option_D"
}
question_id = 0
for data in dataset:
    new_data = {}
    new_data["dataset"] = f'kvasir'
    new_data["question_id"] = f'{question_id}'
    new_data["question_type"] = 'Disease Diagnosis'
    new_data["question"] = '这张图像最可能代表了下列哪种疾病？'
    new_data["gt_answer"] = f"{data['label_A']}"
    options_text = ''
    labels = data['x1_labels']
    shuffle(labels)
    for idx, label in enumerate(labels):
        new_data[options[idx]] =  label
        options_text += f'  {options[idx]}: {label}\n' 
    new_data['image'] = data['x0']
    new_data['image_paths'] = [data['x0']]
    formatted_text = f"""You are currently participating in a Visual Question Answering examination. 
Please strictly observe the following instructions.
According to the content of the image, answer the following single-choice question.
Image: <|Image_start|><image><|Image_end|>
{new_data["question"]}\n{options_text}
Note: Put the Corresponding Option for the final answer inside the '<answer></answer>' tag."""

    prompt_text = f"""You are currently participating in a Visual Question Answering examination. 
Please strictly observe the following instructions.
According to the content of the image, answer the following single-choice question.
{new_data["question"]}\n{options_text}
Note: Put the Corresponding Option for the final answer inside the '<answer></answer>' tag."""
    new_data['prompt_text'] = prompt_text
    new_data['formatted_text'] = formatted_text
    new_dataset.append(new_data)
    question_id += 1
    
with open("/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/kvasir_doctor_exam/new_eval_tsy_llm_62.json", "w") as f:
    json.dump(new_dataset, f, indent=4, ensure_ascii=False)
    