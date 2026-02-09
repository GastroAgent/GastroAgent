# 医疗VQA推理Pipeline - 使用指南

## 📋 流程概览

```
原始数据 (new_eval_tsy_llm.json)
    ↓
【步骤1】模型推理 + Trigger策略分析
    ↓
new_eval_tsy_llm_with_trigger.json (包含 trigger_final, max_prob, prob_gap 等)
    ↓
【步骤2】LLM提取答案
    ↓
new_eval_tsy_llm_extracted.json (添加 extracted_answer)
    ↓
【步骤3】重新评估correct
    ↓
new_eval_tsy_llm_final.json (更新 correct)
    ↓
【步骤4】重新分析trigger性能
    ↓
new_eval_tsy_llm_trigger_report.json (性能分析报告)
```

## 🚀 快速开始

### 方法1: 执行完整Pipeline（推荐）

```bash
python run_full_pipeline.py
```

这个脚本会自动按顺序执行所有4个步骤。

### 方法2: 单独执行各个步骤

```bash
# 步骤1: 模型推理
python step1_model_inference.py

# 步骤2: LLM提取答案
python step2_extract_answers_with_llm.py

# 步骤3: 重新评估correct
python step3_reevaluate_correct.py

# 步骤4: 分析trigger性能
python step4_reanalyze_trigger_performance.py
```

## 📁 各步骤详细说明

### 步骤1: 模型推理 + Trigger策略分析

**脚本:** `step1_model_inference.py`

**功能:**
- 加载Llava-Qwen2多模态模型
- 对输入数据进行推理，生成答案
- 计算概率分布 (p_A, p_B, p_C, p_D)
- 计算trigger相关指标:
  - `max_prob`: 最大概率值
  - `prob_gap`: 第一名与第二名的概率差距
  - `h_norm`: 归一化熵
  - `is_consistent`: 语义一致性（生成答案与最大概率选项是否一致）
  - `trigger_final`: 最终触发决策（三层过滤模型）

**输入:** `new_eval_tsy_llm.json`

**输出:** `new_eval_tsy_llm_with_trigger.json`

**关键参数:**
```python
THRES_MAX_P = 0.65      # 最大概率阈值
THRES_GAP = 0.20        # 概率差距阈值
THRES_ENTROPY = 0.40    # 归一化熵阈值
```

### 步骤2: LLM提取答案

**脚本:** `step2_extract_answers_with_llm.py`

**功能:**
- 使用更强的LLM（Gemini/GPT-4/Claude）从生成文本中提取答案
- 处理各种格式和边缘情况
- 标准化答案为单个字母 (A/B/C/D)

**输入:** `new_eval_tsy_llm_with_trigger.json`

**输出:** `new_eval_tsy_llm_extracted.json`

**配置选项:**
```python
USE_LLM_API = True                  # 是否使用LLM API
LLM_API_TYPE = "gemini"             # API类型: gemini/openai/anthropic
LLM_MODEL = "gemini-1.5-flash"      # 模型名称
LLM_API_KEY = os.getenv("GEMINI_API_KEY")  # API密钥
```

**环境变量设置:**
```bash
export GEMINI_API_KEY="your-api-key-here"
# 或者
export OPENAI_API_KEY="your-api-key-here"
```

### 步骤3: 重新评估correct

**脚本:** `step3_reevaluate_correct.py`

**功能:**
- 将提取的答案与ground truth进行对比
- 计算`correct`字段 (0/1)
- 计算`p_chosen`（提取答案对应的概率）
- 统计整体准确率

**输入:** `new_eval_tsy_llm_extracted.json`

**输出:** `new_eval_tsy_llm_final.json`

### 步骤4: 重新分析Trigger性能

**脚本:** `step4_reanalyze_trigger_performance.py`

**功能:**
- 分析trigger策略的性能指标
- 按问题类型分组分析
- 对比不同trigger策略
- 分析概率指标分布
- 生成详细的性能报告和建议

**输入:** `new_eval_tsy_llm_final.json`

**输出:** `new_eval_tsy_llm_trigger_report.json`

**关键指标:**
- `trigger_rate`: 触发率
- `safety_accuracy`: 未触发样本的准确率（安全性指标）
- `precision`: 触发样本中的错误率
- `recall`: 所有错误中被捕获的比例
- `overall_accuracy`: 整体准确率

## ⚙️ 配置说明

### 修改数据路径

在每个脚本的开头修改以下参数:

```python
data_name = '胃'  # 数据集名称
input_data_path = f'/path/to/your/data/new_eval_tsy_llm.json'
output_data_path = f'/path/to/your/output/result.json'
```

### 修改模型路径

在 `step1_model_inference.py` 中:

```python
model_id = '/path/to/your/model/Llava-Qwen2-7B'
```

### 调整Trigger阈值

在 `step1_model_inference.py` 中:

```python
# 新版三层过滤策略
THRES_MAX_P = 0.65      # 提高该值会降低触发率，提高安全性
THRES_GAP = 0.20        # 提高该值会增加触发率，更严格
THRES_ENTROPY = 0.40    # 降低该值会增加触发率，更严格
```

**调整建议:**
- 如果未触发样本准确率低于95%，应该**降低** `THRES_MAX_P` 或**提高** `THRES_GAP`/`THRES_ENTROPY`
- 如果触发率过高（>50%），可以适当调整阈值使触发更保守
- 医疗诊断等高风险场景建议使用更严格的阈值

## 📊 输出文件说明

### 1. new_eval_tsy_llm_with_trigger.json

包含所有推理结果和trigger相关字段:

```json
{
    "question_id": "xxx",
    "generated_text": "模型生成的完整文本",
    "p_A": 0.45, "p_B": 0.30, "p_C": 0.15, "p_D": 0.10,
    "max_prob": 0.45,
    "prob_gap": 0.15,
    "h_norm": 0.35,
    "is_consistent": true,
    "trigger_final": false,
    ...
}
```

### 2. new_eval_tsy_llm_extracted.json

添加了LLM提取的答案:

```json
{
    ...(继承步骤1的所有字段),
    "extracted_answer": "A"
}
```

### 3. new_eval_tsy_llm_final.json

添加了correct评估结果:

```json
{
    ...(继承步骤2的所有字段),
    "gt_letter": "A",
    "extracted_answer_normalized": "A",
    "correct": 1,
    "p_chosen": 0.45
}
```

### 4. new_eval_tsy_llm_trigger_report.json

详细的性能分析报告:

```json
{
    "overall_performance": {
        "total_samples": 1000,
        "trigger_rate": 0.35,
        "safety_accuracy": 0.97,
        "recall": 0.78,
        ...
    },
    "by_question_type": {...},
    "strategy_comparison": {...},
    "recommendations": [...]
}
```

## 🔍 常见问题

### Q1: 如何不使用LLM API提取答案？

在 `step2_extract_answers_with_llm.py` 中设置:

```python
USE_LLM_API = False
```

这样会使用基于规则的方法提取答案。

### Q2: 如何处理API限流？

在步骤2中，代码已包含延迟:

```python
time.sleep(0.1)  # 避免API限流
```

如果仍然遇到限流，可以增加延迟时间。

### Q3: 如何只运行部分步骤？

直接运行对应的步骤脚本即可，但需要确保前置步骤的输出文件存在。

### Q4: 如何并行处理多个数据集？

修改 `data_name` 参数，然后运行多个进程:

```bash
# 终端1
python step1_model_inference.py  # data_name = '胃'

# 终端2
python step1_model_inference.py  # data_name = '食管'
```

## 📈 性能优化建议

1. **GPU内存优化**: 如果遇到OOM，可以减小 `batch_size`
2. **推理速度**: 已启用Flash Attention 2加速
3. **API成本**: 使用 `gemini-1.5-flash` 或 `gpt-4o-mini` 等经济模型
4. **并行处理**: 可以在多个GPU上并行处理不同数据集

## 🛠️ 依赖环境

```bash
pip install torch transformers peft pillow
pip install google-generativeai  # 如果使用Gemini
pip install openai  # 如果使用OpenAI
```

## 📞 联系方式

如有问题，请联系项目维护者。

---

**版本**: 1.0
**最后更新**: 2024
