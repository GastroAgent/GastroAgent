import json
from modelscope import AutoModelForCausalLM, AutoTokenizer
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

model_name = "/mnt/inaisfs/data/home/tansy_criait/weights/Qwen2.5-32B-Instruct"
# data_path = '/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/answer/answer_Qwen2.5-VL-32B-Instruct_one_buwei.json'
# data_path = '/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_result/Qwen2.5-VL-32B-Instruct/Qwen2.5-VL-32B-Instruct.json'
data_path = '/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_result/Llava-Qwen2-7B-tune-med/Kvasir-en-True_VQA.json'

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)

new_dataset = []
files = open(data_path, 'r', encoding='utf-8')

dataset = json.load(files)
for data in dataset:
# for text in files:
    # data = json.loads(text)
    if "extracted_answer" in data:
        new_dataset.append(data)
        continue
    # text = data['text']
    text = data['generated_text']
    question = data['prompt_text'].split('\nQuestion')[-1].replace('<|im_end|>\n<|im_start|>assistant\n', '')

    ## prepare the model input
    prompt = f"Question: {question}\nResponse: {text}\n\nYour Task: What I need is the corresponding option for the result. However, the above reply provided the result itself. I require you to select the option from the question based on the reply result, and place the option within the <answer> and </answer> tags. If the above reply has already been placed within the tags, you can simply output it."
    # prompt = f"{text}\n\nExtract the final answer from the above response and place the final answer in the <answer> and </answer> tags. \nNote: Do not answer the original question or correct the answer, your task is just to extract the answer from the content into the specified format."
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=128
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("thinking content:", thinking_content)
    print("content:", content)
    data['extracted_answer'] = content
    new_dataset.append(data)

with open(data_path, 'w', encoding='utf-8') as f:
    json.dump(new_dataset, f, ensure_ascii=False, indent=4)

# with open(data_path, mode='w', encoding='utf-8') as ans_file:
#     for data in new_dataset:
#         ans_file.write(json.dumps(data, ensure_ascii=False) + "\n")