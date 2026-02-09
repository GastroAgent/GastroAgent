import os
import json
import math
import PIL.Image
import re
import sys
import torch
import torch.nn as nn
from transformers import GenerationConfig

os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

# 环境与路径设置 (请根据你的服务器环境微调)
sys.path.append('/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1')
from llava.model.llavaProcessor import LlavaProcessor
from llava.model.language_model.llava_qwen_mul_workflow import PloyLlavaLlamaForCausalLM
from trl.data_utils import maybe_apply_chat_template

# ==========================================
# 1. 配置参数 (与训练保持高度一致)
# ==========================================
MODEL_ID = '/mnt/inaisfs/data/home/tansy_criait/weights/tsy/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-jmf-sft-tsy-cotRL-2000'
INPUT_DATA_PATH = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/new_eval_tsy_llm_with_letter.json'
OUTPUT_DATA_PATH = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/食管_doctor_exam/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-jmf-sft-tsy-cotRL-2000/halt_85v1/new_eval_tsy_llm_with_trigger.json'

# HALT 核心配置 (必须匹配训练设置)
HALT_ENABLED = True
HALT_MIDDLE_LAYER_RATIO = 0.85          # 关键：匹配训练时的提取层
HALT_PROBE_HIDDEN_DIM = 512            # 关键：匹配训练时的 MLP 维度
HALT_RISK_THRESHOLD = 0.4              # 建议阈值，可根据 PR 曲线调整
HALT_PROBE_MODEL_PATH = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/shiguan/halt/models/halt_probe_best_layer_85.pth'

# 路由判定阈值
THRES_MAX_P = 0.65
THRES_GAP = 0.20
THRES_ENTROPY = 0.40

# 问题类型定义
LONG_QTYPES = {"Modality Recognition", "Anatomy Identification"}
TAIL_QTYPES = {"Disease Diagnosis"}

# ==========================================
# 2. HALT 探针模型定义 (必须匹配训练架构)
# ==========================================
class HALTProbeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1) # 输出 Logits
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)

# ==========================================
# 3. 工具函数
# ==========================================
def extract_answer_letter(text):
    m = re.search(r"<answer>\s*(?:option[_\s]*)?([A-D])\b", text, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def get_middle_hiddens(model, inputs, ratio):
    num_layers = model.config.num_hidden_layers
    target_layer = int(num_layers * ratio)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    # 取最后一层 token 的状态
    return outputs.hidden_states[target_layer][:, -1, :]

# ==========================================
# 4. 初始化流程
# ==========================================
print("正在加载主模型...")
model = PloyLlavaLlamaForCausalLM.from_pretrained(MODEL_ID, device_map='auto', torch_dtype='bfloat16', use_cache=True)
processing_class = LlavaProcessor.from_pretrained(MODEL_ID)

halt_probe = None
if HALT_ENABLED:
    print(f"正在加载 HALT 探针: {HALT_PROBE_MODEL_PATH}")
    halt_probe = HALTProbeModel(input_dim=model.config.hidden_size).to(model.device)
    checkpoint = torch.load(HALT_PROBE_MODEL_PATH, map_location=model.device, weights_only=False)
    
    # 获取原始 state_dict
    raw_state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # --- 核心修正：处理键名不匹配 ---
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    
    for k, v in raw_state_dict.items():
        if not k.startswith('mlp.'):
            # 如果原始键名没有 mlp. 前缀，我们就给它加上
            new_state_dict[f'mlp.{k}'] = v
        else:
            new_state_dict[k] = v
            
    # 使用修改后的 state_dict 加载
    halt_probe.load_state_dict(new_state_dict)
    halt_probe.eval()
    print("探针参数对齐并加载成功！")

# 准备 Token IDs
choice_tokens = ["A", "B", "C", "D"]
choice_ids = torch.tensor([processing_class.tokenizer.encode(c, add_special_tokens=False)[-1] 
                          for c in choice_tokens], device=model.device)

# ==========================================
# 5. 推理循环
# ==========================================
dataset = json.load(open(INPUT_DATA_PATH, 'r'))
results = []
gen_config = GenerationConfig(max_new_tokens=512, do_sample=True, temperature=0.2)



for i, example in enumerate(dataset):
    print(f"处理进度: {i+1}/{len(dataset)}")

    # 构建输入
    system_prompt = (
        "You are a helpful Medical AI assistant and authoritative expert in the medical field. "
        "You have a solid foundation in medicine and are proficient in answering various visual questions related to medical topics. "
        "As the Medical AI assistant, you need to understand the visual content (if exists) and related instructions provided by users, "
        "and use natural language to assist users in various tasks. "
        "As the expert, you need to make sure that your response is logical and factual, and that it conveys the authority of the expert. "
        "The visual content (most about medical imaging) will be provided with the following format: <|Image_start|>visual content<|Image_end|>."
    )
    prompt = [{'role': 'system', 'content': system_prompt},
              {'role': 'user', 'content': example['formatted_text']}]
    prompt_text = maybe_apply_chat_template({'prompt': prompt}, processing_class)["prompt"]
    
    # 图像处理
    images = [PIL.Image.open(p).convert('RGB') for p in example.get('image_paths', [])]
    inputs = processing_class(text=[prompt_text], images=images if images else None, 
                              return_tensors='pt', padding=True).to(model.device)

    with torch.no_grad():
        # A. 提取隐藏状态并执行探针检测
        hiddens = get_middle_hiddens(model, inputs, HALT_MIDDLE_LAYER_RATIO)
        halt_logits = halt_probe(hiddens.float())
        halt_risk = torch.sigmoid(halt_logits).item()
        halt_trigger = halt_risk > HALT_RISK_THRESHOLD

        # B. 计算 Logits 分布指标
        p_G, _ = model.get_pG_from_inputs(candidate_ids=choice_ids, **inputs)
        max_p, argmax_i = p_G.max(dim=-1)
        sorted_p, _ = torch.sort(p_G, descending=True)
        prob_gap = (sorted_p[:, 0] - sorted_p[:, 1]).item()
        
        # 计算熵
        entropy = -(p_G * torch.log(p_G + 1e-9)).sum().item()
        h_norm = entropy / math.log(len(choice_tokens))

    # C. 执行文本生成
    out_ids = model.generate(**inputs, generation_config=gen_config)
    gen_text = processing_class.batch_decode(out_ids, skip_special_tokens=True)[0]
    
    # D. 语义一致性检查
    pred_letter = choice_tokens[argmax_i.item()]
    gen_letter = extract_answer_letter(gen_text)
    is_consistent = (gen_letter == pred_letter)

    # E. 综合判定 (Agentic Routing 核心)
    # 只要满足 Logits 异常 或 HALT 风险高 或 语义不一致，即触发路由
    trigger_final = (max_p.item() < THRES_MAX_P or h_norm > THRES_ENTROPY or 
                    not is_consistent or halt_trigger)

    # 计算概率分布字典
    probs = p_G.cpu().tolist()[0]  # 转换为列表
    prob_dict = {f"p_{ch}": float(probs[j]) for j, ch in enumerate(choice_tokens) if j < len(probs)}

    # 计算辅助指标
    entropy_score = 1.0 - h_norm
    conf_mix = 0.3 * max_p.item() + 0.7 * entropy_score

    # 提取问题类型
    qt = example.get('question_type', 'Unknown')
    is_long = (qt in LONG_QTYPES)
    is_tail = (qt in TAIL_QTYPES)

    # 保存结果（与nohalt版本保持一致的字段顺序和内容）
    res_item = {
        # 基础信息
        'question_id': example['question_id'],
        'question_type': qt,
        'is_long_qtype': is_long,
        'is_tail_qtype': is_tail,
        'image_paths': example.get('image_paths', []),

        # 问题和选项
        'formatted_text': example['formatted_text'],
        'option_A': example['option_A'],
        'option_B': example['option_B'],
        'option_C': example['option_C'],
        'option_D': example['option_D'],
        'option_E': example.get('option_E'),
        'option_F': example.get('option_F'),
        'option_G': example.get('option_G'),
        'option_H': example.get('option_H'),
        'gt_answer': example['gt_answer'],

        # 模型输出
        'prompt_text': prompt_text,
        'generated_text': gen_text,
        'temp_answer_letter': gen_letter,  # 临时提取的答案

        # 概率分布
        **prob_dict,
        'pred_idx': argmax_i.item(),
        'pred_letter': pred_letter,

        # 新版三层过滤模型指标（核心）
        'max_prob': max_p.item(),
        'prob_gap': prob_gap,
        'h_norm': h_norm,
        'entropy': entropy,
        'entropy_score': entropy_score,
        'is_consistent': is_consistent,
        'trigger_final': trigger_final,

        # 辅助指标
        'conf_mix': conf_mix,

        # HALT特有指标
        'halt_risk_score': halt_risk,
        'halt_trigger': halt_trigger,

        # 旧版trigger标记（用于对比分析，这里设置为None或False）
        'trigger_old': trigger_final,  # 可以用trigger_final代替
        'trigger_high_entropy': h_norm > THRES_ENTROPY,
        'trigger_low_confidence': max_p.item() < THRES_MAX_P,
        'trigger_ood': False,  # HALT版本没有OOD检测
        'trigger_rule_tail_qtype': is_tail,
    }
    results.append(res_item)

# 保存 JSON
os.makedirs(os.path.dirname(OUTPUT_DATA_PATH), exist_ok=True)
with open(OUTPUT_DATA_PATH, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print('=' * 80)
print('步骤1完成！')
print(f'共处理 {len(results)} 条数据')
print(f'触发率: {sum(1 for r in results if r["trigger_final"])/len(results):.2%}')
print(f'输出文件: {OUTPUT_DATA_PATH}')
print('=' * 80)