import json
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import pandas as pd
import torch
import gc

# dataset = json.load(open('/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/instrcut_dia_mulimg_format2.json'))
# answers_file = '/home/dalhxwlyjsuo/criait_tansy/project/Llava_Qwen2/data/answer/answer-HuatuoGPT-Vision-34B-mul.json'

dataset = json.load(open('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管/eval_all_llm.json', 'r'))
# answers_file = '/home/dalhxwlyjsuo/criait_tansy/jmf/GRPO_result/HuatuoGPT-Vision-34B.json'

questions = [question for question in dataset if not len(question['image_paths']) > 10]
# 准备一个列表来保存每一行数据
# results = []
# 考试成绩
# 多图的 测试 Demo。

# ### HuatuoGPT
# from HuatuoGPT.cli import HuatuoChatbot
# huatuogpt_vision_model_path = '/home/dalhxwlyjsuo/criait_tansy/weight/HuatuoGPT-Vision-34B'
# bot = HuatuoChatbot(huatuogpt_vision_model_path)
#
# print('Test HuatuoGPT-Vision-34B')
# for question in questions:
#     # text = question['text']
#     # image_paths = question['images']
#     text = question['formatted_text']
#     image_paths = question['image_paths']
#     output = bot.inference(text, image_paths)  # 获取模型输出
#     model_response = output[0]
#
#     # question_id = question['question_id']
#     # gt_answer = question.get('gpt4_answer', '')  # 如果没有 gpt4_answer 字段，设为空字符串
#
#     question['generated_text'] = model_response
#     results.append({
#         'generated_text': model_response,
#         'prompt_text': text,
#         "answer": question['answer'],
#         "gt_answer": question['gt_answer'],
#         "image_paths": question['image_paths'],
#         "question_id": question['question_id'],
#         "question_type": question['question_type'],
#     })
# json.dump(results, open(answers_file,'w', encoding='utf-8'), indent=4, ensure_ascii=False)
# print('Infer Done!')
#
#     # # 把这一条结果加入列表
#     # results.append({
#     #     '问题ID': question_id,
#     #     '问题文本': text,
#     #     '模型回答': model_response,
#     #     '标准答案': gt_answer,
#     #     '模型名称': 'HuatuoGPT-Vision-34B'
#     # })
#     # print(f"问题ID: {question_id}")
#     # print(model_response)
#     # print('=' * 100, '\n')
# del bot
# torch.cuda.empty_cache()
# gc.collect()
#
# ans_file = open(answers_file, "w")
# for result in results:
#     ans_file.write(json.dumps({
#                                "prompt": result['问题文本'],
#                                "text": result['模型回答'],
#                                "model_id": huatuogpt_vision_model_path,
#                                "gpt4_answer": result['标准答案'],
#                                "metadata": {}}, ensure_ascii=False) + "\n")
#     ans_file.flush()
# ans_file.close()

# 转换为 DataFrame
# results_df = pd.DataFrame(results)
# # 使用 pivot 将模型名称转为列，问题ID 作为索引
# wide_df = results_df.pivot(
#     index='问题ID',
#     columns='模型名称',
#     values='模型回答'
# )
# # 如果需要保留问题文本或标准答案，可以合并回来
# final_df = wide_df.reset_index()
# final_df['问题文本'] = results_df['问题文本'].values  # 注意：这里需要确保顺序一致
# final_df['标准答案'] = results_df['标准答案'].values
# # 写入 Excel 文件
# final_df.to_excel("/home/dalhxwlyjsuo/criait_tansy/jmf/Eval-Metric/other_LLM_test/HuatuoGPT-Vision-34B.xlsx", index=False)
# print("✅ 已成功写入 /home/dalhxwlyjsuo/criait_tansy/jmf/Eval-Metric/other_LLM_test/HuatuoGPT-Vision-34B.xlsx")
# print('Test HuatuoGPT-Vision-34B Done.')
#
# print('Test HuatuoGPT-Vision-7B')
# from HuatuoGPT.cli import HuatuoChatbot
# huatuogpt_vision_model_path = '/home/dalhxwlyjsuo/criait_tansy/weight/HuatuoGPT-Vision-7B'
# bot = HuatuoChatbot(huatuogpt_vision_model_path)
# print('Test HuatuoGPT-Vision-7B')
# for question in questions:
#     text = question['text']
#     image_paths = question['images']
#     model_response = bot.inference(text, image_paths)[0]
#
#     # 保存结果到字典
#     results.append({
#         '问题ID': question['question_id'],
#         '问题文本': text,
#         '模型名称': 'HuatuoGPT-Vision-7B',
#         '模型回答': model_response,
#         '标准答案': question.get('gpt4_answer', '')
#     })
#
# del bot
# torch.cuda.empty_cache()
# gc.collect()
# # 如果你有多个模型，可以重复上面的逻辑，只改模型路径和模型名称
#
# # 转换为 DataFrame
# results_df = pd.DataFrame(results)
#
# # 使用 pivot 将模型名称转为列，问题ID 作为索引
# wide_df = results_df.pivot(
#     index='问题ID',
#     columns='模型名称',
#     values='模型回答'
# )
#
# # 如果需要保留问题文本或标准答案，可以合并回来
# final_df = wide_df.reset_index()
# final_df['问题文本'] = results_df['问题文本'].values  # 注意：这里需要确保顺序一致
# final_df['标准答案'] = results_df['标准答案'].values
#
# # 写入 Excel 文件
# final_df.to_excel("/home/dalhxwlyjsuo/criait_tansy/jmf/Eval-Metric/other_LLM_test/HuatuoGPT-Vision-7B.xlsx", index=False)
#
# print("✅ 已成功写入 /home/dalhxwlyjsuo/criait_tansy/jmf/Eval-Metric/other_LLM_test/HuatuoGPT-Vision-7B.xlsx")
# print('Test HuatuoGPT-Vision-7B Done.')
#
# # print('Test RedFM')
# # print("Setup tokenizer")
# # from RadFM.Quick_demo.test import get_tokenizer, combine_and_preprocess
# # # Initialize tokenizer with special image tokens
# # text_tokenizer, image_padding_tokens = get_tokenizer('/home/dalhxwlyjsuo/criait_tansy/jmf/Eval-Metric/other_LLM_test/RadFM/Quick_demo/Language_files')
# # print("Finish loading tokenizer")
# #
# # print("Setup Model")
# # from RadFM.Quick_demo.Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
# # # Initialize the multimodal model
# # model = MultiLLaMAForCausalLM(
# #     lang_model_path='/home/dalhxwlyjsuo/criait_tansy/weight/RedFM',
# #     # Build up model based on LLaMa-13B config
# # )
# #
# # # # Load pretrained model weights
# # # ckpt = torch.load('/home/dalhxwlyjsuo/criait_tansy/weight/RedFM/pytorch_model.bin',
# # #                   map_location='cpu')  # Please download our checkpoint from huggingface and decompress the original zip file first
# # # model.load_state_dict(ckpt)
# # print("Finish loading model")
# #
# # # Move model to GPU and set to evaluation mode
# # # model = model.to(model.device)
# # model.eval()
# #
# # for question in questions:
# #     query = question['text']
# #     image_paths = question['images']
# #
# #     # Specify the image path and where to insert it in the question
# #     image = [
# #         {
# #             'img_path': image_path,
# #             'position': idx,  # Insert at the beginning of the question
# #         } for idx, image_path in enumerate(image_paths) # Can add arbitrary number of images
# #     ]
# #     print('image', image)
# #     # Combine text and images into model-ready format
# #     text, vision_x = combine_and_preprocess(query, image, image_padding_tokens)
# #     print('text', text)
# #     # Run inference without gradient computation
# #     with torch.no_grad():
# #         # Tokenize the combined text with image placeholders
# #         lang_x = text_tokenizer(
# #             text, max_length=2048, truncation=True, return_tensors="pt"
# #         )['input_ids'].to(model.lang_model.device)
# #
# #         # Move image tensor to GPU
# #         vision_x = vision_x.to(model.lang_model.device).to(torch.bfloat16)
# #         print(model)
# #         print('vision_x shape', vision_x.shape)
# #         # Generate text response
# #         generation = model.generate(lang_x, vision_x)
# #
# #         # Decode the generated token IDs to text
# #         generated_texts = text_tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
# #         print('Input: ', text)
# #         print('Output: ', generated_texts)
# #
# #     # 保存结果到字典
# #     results.append({
# #         '问题ID': question['question_id'],
# #         '问题文本': query,
# #         '模型名称': 'HuatuoGPT-Vision-7B',
# #         '模型回答': generated_texts,
# #         '标准答案': question.get('gpt4_answer', '')
# #     })
# #
# # # 转换为 DataFrame
# # results_df = pd.DataFrame(results)
# #
# # # 使用 pivot 将模型名称转为列，问题ID 作为索引
# # wide_df = results_df.pivot(
# #     index='问题ID',
# #     columns='模型名称',
# #     values='模型回答'
# # )
# #
# # # 如果需要保留问题文本或标准答案，可以合并回来
# # final_df = wide_df.reset_index()
# # final_df['标准答案'] = results_df['标准答案'].values
# # final_df['问题文本'] = results_df['问题文本'].values  # 注意：这里需要确保顺序一致
# #
# #
# # # 写入 Excel 文件
# # final_df.to_excel("/home/dalhxwlyjsuo/criait_tansy/jmf/Eval-Metric/other_LLM_test/RedFM.xlsx", index=False)
# #
# # print("✅ 已成功写入 /home/dalhxwlyjsuo/criait_tansy/jmf/Eval-Metric/other_LLM_test/RedFM.xlsx")
# # print('Test RedFM Done.')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
model_name = 'Qwen2.5-VL-32B-Instruct'
model_path = f'/mnt/inaisfs/data/home/tansy_criait/weights/{model_name}'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="bfloat16", device_map="auto",
    attn_implementation="flash_attention_2"
)
answers_file = f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/食管/{model_name}/食管.json'
os.makedirs(f'/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/食管/{model_name}', exist_ok=True)
processor = AutoProcessor.from_pretrained(model_path)
results = []

sys_prompt = (
        "You are a helpful Medical AI assistant and an authoritative expert in the field of the medicine. "
        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, and use natural language to assist users in various tasks. "
        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. ")

print(model_name)
from tqdm import tqdm
for idx, question in tqdm(enumerate(questions)):
    print(f"{idx}/{len(questions)}")
    text = question['prompt_text'].replace(' ASSISTANT:', "\nNote: Put the Corresponding Option for the final answer inside the '<answer></answer>' tag.")

    image_files = question['image_paths']
    images = [{
        "type": "image",
        "image": image_file,
    } for image_file in image_files]
    messages = [
        # {
        #     "role": "system",
        #     "content": [
        #         {"type": "text", "text": sys_prompt},
        #     ],
        # },
        {
            "role": "user",
            "content": images + [
                {"type": "text", "text": text},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # Inference: Generation of the output
    generated_ids = model.generate(**inputs,
                                   max_new_tokens=1024,
                                   do_sample=True,
                                   temperature=0.5,
                                   top_p=0.95,
                                   use_cache=True
                                   )
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    model_response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    if isinstance(model_response, list):
        model_response = model_response[0]

    # question_id = question['question_id']
    # gt_answer = question.get('gpt4_answer', '')  # 如果没有 gpt4_answer 字段，设为空字符串

    question['generated_text'] = model_response
    print(model_response)
    results.append({
        'generated_text': model_response,
        'prompt_text': text,
        "gt_answer": question['gt_answer'],
        "image_paths": question['image_paths'],
        "question_id": question['question_id'],
        "question_type": question['question_type'],
        "option_A": question['option_A'],
        "option_B": question['option_B'],
        "option_C": question['option_C'],
        "option_D": question['option_D'],
        "option_E": question['option_E'] if 'option_E' in question else None,
        "option_F": question['option_F'] if 'option_F' in question else None,
        "option_G": question['option_G'] if 'option_G' in question else None,
        "option_H": question['option_H'] if 'option_H' in question else None,
    })
json.dump(results, open(answers_file, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)
print('Infer Done!')


