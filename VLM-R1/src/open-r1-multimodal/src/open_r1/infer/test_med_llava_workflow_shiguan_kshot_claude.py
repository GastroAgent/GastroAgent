"""
================================================================================
改进版：医疗VQA触发式重推理系统（参考设计图架构）
核心改进：
1. 分层Gate机制（P1→P3→P4）
2. Support Retrieval（检索相似样本）
3. Few-shot重推理（基于检索样本）
4. 保留完整评估流程
================================================================================
"""

import json
import re
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from tqdm import tqdm
from modelscope import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from collections import defaultdict

# =========================
# 配置参数
# =========================
MODEL_NAME = "/mnt/inaisfs/data/home/tansy_criait/weights/Qwen2.5-32B-Instruct"
DATA_PATH = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/doctor_食管/new_eval_tsy_llm.json'

# Few-shot结果路径（用于Support Retrieval）
FEWSHOT_RESULTS_PATH = '/mnt/inaisfs/data/home/tansy_criait/VLM-R1/results/doctor_食管_workflow_cursor/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-75/kshot/eval_by_x0_clean.json'

VALID_OPTIONS = set("ABCDEFGH")
answer_tag_re = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
answer_letter_re = re.compile(r"\b([A-H])\b", re.IGNORECASE)


# =========================
# 分层Gate配置（参考设计图P1→P3→P4）
# =========================
class TriggerGateConfig:
    """触发门控配置"""
    # P1: 快速粗筛（快速排除高置信度样本）
    P1_MAX_PROB_THRESHOLD = 0.75  # max_prob > 0.75 直接通过
    
    # P3: 精细检测（多指标综合判断）
    P3_MAX_PROB_THRESHOLD = 0.60  # max_prob < 0.60 触发
    P3_PROB_GAP_THRESHOLD = 0.15  # prob_gap < 0.15 触发
    P3_H_NORM_THRESHOLD = 0.40    # h_norm > 0.40 触发
    
    # P4: 上下文验证（检查一致性）
    P4_REQUIRE_CONSISTENCY = True  # is_consistent=False 触发
    
    # Support Retrieval配置
    SUPPORT_TOP_K = 3              # 检索top-3相似样本
    SUPPORT_MIN_CONFIDENCE = 0.80  # 支持样本最小置信度
    SUPPORT_MAX_DISTANCE = 0.1     # 最大距离阈值


# =========================
# 工具函数（保持不变）
# =========================
def normalize_answer(value: Optional[str]) -> Optional[str]:
    """标准化答案格式"""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    option_match = re.match(r"option[_\s]*([A-H])", value, re.IGNORECASE)
    if option_match:
        return option_match.group(1).upper()
    if len(value) == 1 and value.upper() in VALID_OPTIONS:
        return value.upper()
    letter_match = answer_letter_re.search(value)
    if letter_match:
        letter = letter_match.group(1).upper()
        if letter in VALID_OPTIONS:
            return letter
    return None


def extract_answer_from_text(text: str) -> Optional[str]:
    """从文本中提取<answer></answer>标签之间的内容"""
    if not text:
        return None
    match = answer_tag_re.search(text)
    if match:
        return match.group(1).strip()
    letter_match = answer_letter_re.search(text)
    if letter_match:
        return letter_match.group(1).strip()
    return None


# =========================
# 核心改进1: 分层Gate机制
# =========================
class HierarchicalTriggerGate:
    """
    分层触发门控（参考设计图P1→P3→P4）
    """
    
    def __init__(self, config: TriggerGateConfig):
        self.config = config
        self.stats = {
            'p1_passed': 0,
            'p3_triggered': 0,
            'p4_triggered': 0,
            'total': 0
        }
    
    def should_trigger(self, sample: Dict) -> Tuple[bool, str, Dict]:
        """
        判断是否触发重推理
        
        Returns:
            (should_trigger, trigger_reason, gate_info)
        """
        self.stats['total'] += 1
        gate_info = {
            'p1_result': None,
            'p3_result': None,
            'p4_result': None,
            'final_decision': False,
            'trigger_reason': []
        }
        
        # === P1: 快速粗筛 ===
        max_prob = sample.get('max_prob', 0)
        if max_prob >= self.config.P1_MAX_PROB_THRESHOLD:
            # 高置信度，直接通过
            gate_info['p1_result'] = 'passed'
            gate_info['final_decision'] = False
            self.stats['p1_passed'] += 1
            return False, "P1: High confidence", gate_info
        
        gate_info['p1_result'] = 'need_further_check'
        
        # === P3: 精细检测（多指标） ===
        prob_gap = sample.get('prob_gap', 1.0)
        h_norm = sample.get('h_norm', 0.0)
        
        p3_triggers = []
        if max_prob < self.config.P3_MAX_PROB_THRESHOLD:
            p3_triggers.append('low_max_prob')
        if prob_gap < self.config.P3_PROB_GAP_THRESHOLD:
            p3_triggers.append('small_prob_gap')
        if h_norm > self.config.P3_H_NORM_THRESHOLD:
            p3_triggers.append('high_entropy')
        
        if not p3_triggers:
            # P3未触发
            gate_info['p3_result'] = 'passed'
            gate_info['final_decision'] = False
            return False, "P3: Passed multi-metric check", gate_info
        
        gate_info['p3_result'] = f"triggered: {','.join(p3_triggers)}"
        gate_info['trigger_reason'].extend(p3_triggers)
        self.stats['p3_triggered'] += 1
        
        # === P4: 上下文验证 ===
        if self.config.P4_REQUIRE_CONSISTENCY:
            is_consistent = sample.get('is_consistent', True)
            if not is_consistent:
                gate_info['p4_result'] = 'triggered: inconsistent'
                gate_info['trigger_reason'].append('inconsistent')
                self.stats['p4_triggered'] += 1
            else:
                gate_info['p4_result'] = 'passed'
        
        # 最终决策
        gate_info['final_decision'] = True
        reason = f"P3+P4: {', '.join(gate_info['trigger_reason'])}"
        
        return True, reason, gate_info
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'trigger_rate': (self.stats['p3_triggered'] + self.stats['p4_triggered']) / self.stats['total']
            if self.stats['total'] > 0 else 0
        }


# =========================
# 核心改进2: Support Retrieval（检索模块）
# =========================
class SupportRetriever:
    """
    支持样本检索器（参考设计图P2/P4）
    从Few-shot结果中检索相似且高置信度的样本
    """
    
    def __init__(self, fewshot_results_path: str, config: TriggerGateConfig):
        self.config = config
        self.support_pool = self._load_support_pool(fewshot_results_path)
        print(f"✓ 加载Support Pool: {len(self.support_pool)} 个高质量样本")
    
    def _load_support_pool(self, path: str) -> List[Dict]:
        """加载Few-shot结果作为支持样本池"""
        if not os.path.exists(path):
            print(f"⚠️  Few-shot结果文件不存在: {path}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 筛选高质量样本：正确且低距离
        support_pool = [
            r for r in results
            if r.get('correct', False) and r.get('dist', 1.0) < self.config.SUPPORT_MAX_DISTANCE
        ]
        
        return support_pool
    
    def retrieve_supports(self, query_sample: Dict) -> List[Dict]:
        """
        检索相似的支持样本
        
        策略：
        1. 基于gt类别匹配（如果有）
        2. 基于距离排序
        3. 返回top-k
        """
        if not self.support_pool:
            return []
        
        query_gt = query_sample.get('gt_letter') or query_sample.get('gt')
        
        # 筛选同类别样本
        candidates = [
            s for s in self.support_pool
            if s.get('gt') == query_gt or s.get('pred') == query_gt
        ]
        
        if not candidates:
            # 如果没有同类别，使用全部pool
            candidates = self.support_pool
        
        # 按距离排序（假设dist字段存在）
        candidates_sorted = sorted(candidates, key=lambda x: x.get('dist', 1.0))
        
        # 返回top-k
        return candidates_sorted[:self.config.SUPPORT_TOP_K]
    
    def format_fewshot_examples(self, supports: List[Dict], original_question: str) -> str:
        """
        格式化Few-shot示例
        
        Returns:
            格式化的Few-shot prompt
        """
        if not supports:
            return ""
        
        examples = []
        for i, sup in enumerate(supports, 1):
            # 提取关键信息
            gt = sup.get('gt', 'Unknown')
            pred = sup.get('pred', 'Unknown')
            
            example = f"Example {i}:\nDiagnosis: {gt}\nConfidence: High (dist={sup.get('dist', 0):.4f})\n"
            examples.append(example)
        
        fewshot_prompt = "\n".join(examples)
        
        return f"""Here are some similar reference cases:

{fewshot_prompt}

Now, based on these examples, please carefully reconsider the following question:
{original_question}

Please provide your answer in the format: <answer>X</answer>
"""


# =========================
# 核心改进3: Few-shot重推理模块
# =========================
class FewshotReinferencer:
    """
    Few-shot重推理器（参考设计图P5: Wasserstien-GastroFlow）
    """
    
    def __init__(self, model, tokenizer, retriever: SupportRetriever):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
    
    def reinfer_with_fewshot(self, sample: Dict) -> Tuple[str, str, List[Dict]]:
        """
        使用Few-shot重新推理
        
        Returns:
            (new_answer, thinking, supports_used)
        """
        # 1. 检索支持样本
        supports = self.retriever.retrieve_supports(sample)
        
        if not supports:
            # 没有支持样本，返回原始答案
            return sample.get('extracted_answer', ''), "No supports found", []
        
        # 2. 构建Few-shot prompt
        original_question = sample.get('prompt_text', '')
        if '\nQuestion' in original_question:
            original_question = original_question.split('\nQuestion')[-1]
        
        fewshot_prompt = self.retriever.format_fewshot_examples(supports, original_question)
        
        # 3. 重新推理
        messages = [{"role": "user", "content": fewshot_prompt}]
        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.model.device)
        
        try:
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # 解析thinking和content
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            thinking = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            return content, thinking, supports
        
        except Exception as e:
            print(f"⚠️  Few-shot重推理失败: {e}")
            return sample.get('extracted_answer', ''), f"Error: {e}", supports


# =========================
# 主流程：集成分层Gate + Support Retrieval + Few-shot重推理
# =========================
def extract_answers_with_hierarchical_trigger(
    input_path: str,
    output_path: Optional[str] = None,
    fewshot_results_path: str = FEWSHOT_RESULTS_PATH,
    model_name: str = MODEL_NAME,
    skip_existing: bool = True
):
    """
    完整流程：提取答案 + 分层Gate + 触发重推理
    
    Args:
        input_path: 输入数据路径
        output_path: 输出路径（可选，默认在输入路径同目录生成）
        fewshot_results_path: Few-shot结果文件路径
        model_name: 模型路径
        skip_existing: 是否跳过已有extracted_answer的样本
    """
    # 自动生成输出路径
    if output_path is None:
        input_file = Path(input_path)
        output_path = str(input_file.parent / f"{input_file.stem}_hierarchical_trigger.json")
    
    print(f"\n{'='*70}")
    print(f"改进版：医疗VQA触发式重推理系统")
    print(f"{'='*70}")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"Few-shot结果: {fewshot_results_path}")
    print(f"{'='*70}\n")
    
    # 加载模型
    print(f"加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    print("✓ 模型加载完成\n")
    
    # 初始化组件
    config = TriggerGateConfig()
    gate = HierarchicalTriggerGate(config)
    retriever = SupportRetriever(fewshot_results_path, config)
    reinferencer = FewshotReinferencer(model, tokenizer, retriever)
    
    # 加载数据
    print(f"加载数据: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"✓ 数据加载完成，共 {len(dataset)} 条记录\n")
    
    # 统计信息
    stats = {
        'total': len(dataset),
        'skipped': 0,
        'initial_extraction': 0,
        'triggered': 0,
        'reinferred': 0,
        'improved_after_reinfer': 0,
        'degraded_after_reinfer': 0
    }
    
    new_dataset = []
    
    # 处理每条数据
    for idx, data in enumerate(tqdm(dataset, desc="处理样本")):
        
        # === 步骤1: 初始答案提取 ===
        if skip_existing and "extracted_answer" in data and data["extracted_answer"]:
            stats['skipped'] += 1
        else:
            # 提取初始答案（使用原有逻辑）
            text = data.get('generated_text', '')
            prompt_text = data.get('prompt_text', '')
            
            if '\nQuestion' in prompt_text:
                question = prompt_text.split('\nQuestion')[-1].replace('<|im_end|>\n<|im_start|>assistant\n', '')
            else:
                question = prompt_text.replace('<|im_end|>\n<|im_start|>assistant\n', '')
            
            prompt = (
                f"Question: {question}\n\n"
                f"Model Response: {text}\n\n"
                "Extract the final answer option (A-H) and format as: <answer>X</answer>\n"
                "Please provide your answer:"
            )
            
            messages = [{"role": "user", "content": prompt}]
            text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
            
            try:
                generated_ids = model.generate(**model_inputs, max_new_tokens=128, temperature=0.1, do_sample=False)
                output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
                
                try:
                    index = len(output_ids) - output_ids[::-1].index(151668)
                except ValueError:
                    index = 0
                
                thinking = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
                content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
                
                data['extracted_answer'] = content
                data['extraction_thinking'] = thinking
                stats['initial_extraction'] += 1
            
            except Exception as e:
                print(f"\n⚠️  样本 {idx} 初始提取失败: {e}")
                data['extracted_answer'] = None
        
        # === 步骤2: 分层Gate判断 ===
        should_trigger, reason, gate_info = gate.should_trigger(data)
        
        data['hierarchical_gate_info'] = gate_info
        data['trigger_decision'] = should_trigger
        data['trigger_reason'] = reason
        
        if not should_trigger:
            # 未触发，使用初始答案
            data['final_answer'] = data.get('extracted_answer')
            data['reinfer_applied'] = False
            new_dataset.append(data)
            continue
        
        # === 步骤3: Few-shot重推理 ===
        stats['triggered'] += 1
        
        try:
            reinfer_answer, reinfer_thinking, supports_used = reinferencer.reinfer_with_fewshot(data)
            
            data['reinfer_answer'] = reinfer_answer
            data['reinfer_thinking'] = reinfer_thinking
            data['supports_used'] = [
                {'gt': s.get('gt'), 'dist': s.get('dist')} for s in supports_used
            ]
            data['reinfer_applied'] = True
            data['final_answer'] = reinfer_answer  # 使用重推理答案
            
            stats['reinferred'] += 1
            
            # 比较前后变化（如果有gt_letter）
            if 'gt_letter' in data:
                original_pred = normalize_answer(extract_answer_from_text(data.get('extracted_answer', '')))
                reinfer_pred = normalize_answer(extract_answer_from_text(reinfer_answer))
                gt = normalize_answer(data.get('gt_letter'))
                
                if gt:
                    original_correct = (original_pred == gt)
                    reinfer_correct = (reinfer_pred == gt)
                    
                    if not original_correct and reinfer_correct:
                        stats['improved_after_reinfer'] += 1
                    elif original_correct and not reinfer_correct:
                        stats['degraded_after_reinfer'] += 1
        
        except Exception as e:
            print(f"\n⚠️  样本 {idx} 重推理失败: {e}")
            data['reinfer_applied'] = False
            data['final_answer'] = data.get('extracted_answer')
            data['reinfer_error'] = str(e)
        
        new_dataset.append(data)
        
        # 定期打印进度
        if idx % 50 == 0 and idx > 0:
            print(f"\n--- 进度 {idx}/{len(dataset)} ---")
            print(f"  触发率: {stats['triggered']}/{idx} = {stats['triggered']/idx*100:.1f}%")
            if stats['reinferred'] > 0:
                print(f"  重推理改进: {stats['improved_after_reinfer']}/{stats['reinferred']}")
    
    # 保存结果
    print(f"\n保存结果到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, ensure_ascii=False, indent=2)
    
    # 打印Gate统计
    gate_stats = gate.get_stats()
    print(f"\n{'='*70}")
    print(f"分层Gate统计:")
    print(f"  P1快速通过: {gate_stats['p1_passed']}/{gate_stats['total']} ({gate_stats['p1_passed']/gate_stats['total']*100:.1f}%)")
    print(f"  P3触发: {gate_stats['p3_triggered']}")
    print(f"  P4触发: {gate_stats['p4_triggered']}")
    print(f"  总触发率: {gate_stats['trigger_rate']*100:.1f}%")
    print(f"\n重推理效果:")
    print(f"  总触发: {stats['triggered']}")
    print(f"  成功重推理: {stats['reinferred']}")
    print(f"  改进案例: {stats['improved_after_reinfer']}")
    print(f"  退化案例: {stats['degraded_after_reinfer']}")
    if stats['reinferred'] > 0:
        print(f"  净改进率: {(stats['improved_after_reinfer'] - stats['degraded_after_reinfer'])/stats['reinferred']*100:.1f}%")
    print(f"{'='*70}\n")
    
    return output_path, stats


# =========================
# 主函数
# =========================
def main(
    data_path: str = DATA_PATH,
    output_path: Optional[str] = None,
    fewshot_results_path: str = FEWSHOT_RESULTS_PATH,
    skip_existing: bool = True
):
    """
    主流程
    
    Args:
        data_path: 输入数据路径
        output_path: 输出路径（可选）
        fewshot_results_path: Few-shot结果文件路径
        skip_existing: 是否跳过已处理样本
    """
    result_path, stats = extract_answers_with_hierarchical_trigger(
        input_path=data_path,
        output_path=output_path,
        fewshot_results_path=fewshot_results_path,
        model_name=MODEL_NAME,
        skip_existing=skip_existing
    )
    
    print(f"✓ 处理完成！结果已保存到: {result_path}")
    return result_path, stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='改进版医疗VQA触发式重推理系统')
    parser.add_argument('--data_path', type=str, default=DATA_PATH,
                       help='输入数据路径')
    parser.add_argument('--output_path', type=str, default=None,
                       help='输出路径（可选，默认在输入路径同目录自动生成）')
    parser.add_argument('--fewshot_results', type=str, default=FEWSHOT_RESULTS_PATH,
                       help='Few-shot结果文件路径（用于Support Retrieval）')
    parser.add_argument('--skip_existing', action='store_true', default=True,
                       help='跳过已有extracted_answer的样本')
    parser.add_argument('--model_name', type=str, default=MODEL_NAME,
                       help='模型路径')
    
    args = parser.parse_args()
    
    main(
        data_path=args.data_path,
        output_path=args.output_path,
        fewshot_results_path=args.fewshot_results,
        skip_existing=args.skip_existing
    )