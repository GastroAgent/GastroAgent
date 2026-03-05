# GastroAgent

<div align="center">

[![License](https://img.shields.io/badge/license-APACHE-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1+-orange.svg)](https://pytorch.org/)

**几何感知多模态AI解决胃肠道诊断中的长尾悖论**

*Geometry-aware multimodal AI resolves the long-tail paradox in gastrointestinal diagnostics*

[English](README_EN.md) | 简体中文

</div>

---

## 目录

- [简介](#简介)
- [系统要求](#系统要求)
- [安装指南](#安装指南)（约 10–15 分钟）
- [模型权重与演示数据](#模型权重与演示数据)
- [快速复现 Demo](#快速复现-demo)
  - [Demo A — Wasserstein-GastroFlow 推理 & 评估](#demo-a--wasserstein-gastroflow-推理--评估)
  - [Demo B — GastroAgent 端到端融合推理](#demo-b--gastroagent-端到端融合推理)
- [训练说明](#训练说明)
- [项目结构](#项目结构)
- [引用](#引用)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 简介

GastroAgent 是一个面向上消化道内窥镜检查的多模态人工智能诊断框架。核心目标是在真实临床场景中同时兼顾**常见病灶的稳定识别**与**长尾罕见病变的可靠覆盖**。

系统由三个核心模块和一个融合控制器组成：

| 模块 | 职责 |
|------|------|
| **GastroMLLM** | 多模态大语言模型，负责医学推理与报告生成 |
| **Flow-Match 生成器** | 学习从查询图像到参考分布的生成轨迹 |
| **Wasserstein-GastroFlow** | 沿生成轨迹计算最优传输代价，实现少样本几何匹配 |
| **熵感知自适应权重控制器** | 动态融合 GastroMLLM 与 Wasserstein-GastroFlow 的预测结果 |

在标准化 Kvasir 数据集上达到 **93.7%** 诊断准确率。自适应融合在保持常见疾病性能的同时显著提升长尾队列：食管不常见病变 **81.4%**、胃部 **84.8%**、十二指肠 **83.8%**。

<!-- 注意：请将 assets/figures/ 下的 PDF 文件转为 PNG 格式以便在 GitHub 上正常渲染 -->
<!-- ![GastroAgent workflow](assets/figures/overview-ill.png) -->

---

## 系统要求

### 硬件

| 项目 | 要求 |
|------|------|
| **GPU** | NVIDIA GPU，显存 ≥ 24 GB（推荐 A100 40 GB） |
| **内存** | ≥ 32 GB RAM |
| **存储** | ≥ 40 GB 可用空间（含模型权重与数据集） |

### 软件

| 项目 | 版本 |
|------|------|
| **操作系统** | Linux（推荐 Ubuntu 20.04 / 22.04） |
| **Python** | ≥ 3.11 |
| **PyTorch** | ≥ 2.5.1 |
| **CUDA** | ≥ 12.1 |

**已验证环境**：Ubuntu 22.04 / Python 3.11.8 / PyTorch 2.5.1 / CUDA 12.1

---

## 安装指南

> **预计耗时**：约 10–15 分钟（取决于网络速度）。

### 步骤 1：克隆仓库

```bash
git clone https://github.com/GastroAgent/GastroAgent.git
cd GastroAgent
```

### 步骤 2：创建虚拟环境

```bash
conda create -n GastroAgent python=3.11 -y
conda activate GastroAgent
```

### 步骤 3：安装 PyTorch（CUDA 12.1）

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

> 如使用其他 CUDA 版本，请参考 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择对应命令。

### 步骤 4：安装项目依赖

```bash
pip install -r requirements.txt
```

> **注意**：若 `requirements.txt` 中的 `flash-attn` 安装失败，可先注释该行，手动按 [Flash Attention 官方指南](https://github.com/Dao-AILab/flash-attention) 安装。

### 步骤 5：验证安装

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

预期输出：

```
PyTorch: 2.5.1
CUDA: True
```

---

## 模型权重与演示数据

所有模型权重与演示数据均托管在 Hugging Face：[https://huggingface.co/GastroAgent/GastroAgent](https://huggingface.co/GastroAgent/GastroAgent)

### 需要下载的文件

| 文件名 | 大小 | 用途 |
|--------|------|------|
| `LlavaQwen2-GRPO-Tricks-Total-CoT-6000.tar.gz` | 12.6 GB | GastroMLLM 模型权重（RL-GRPO 微调） |
| `kvasir-extra.zip` | 7.44 GB | Wasserstein-GastroFlow 权重（kvasir-two-label） |
| `kvasir.zip` | 7.42 GB | Wasserstein-GastroFlow 权重（kvasir-three-label） |
| `image_option_two_label/` | — | kvasir-two-label 的候选参考图像 |
| `image_options/` | — | kvasir-three-label 的候选参考图像 |

### 下载与解压

以下假设将权重统一存放在 `<WEIGHT_DIR>` 目录下（请替换为您的实际路径）：

```bash
# 1. 下载（以 huggingface-cli 为例，也可手动下载）
huggingface-cli download GastroAgent/GastroAgent --local-dir <WEIGHT_DIR>

# 2. 解压 GastroMLLM 权重
cd <WEIGHT_DIR>
tar -xzf LlavaQwen2-GRPO-Tricks-Total-CoT-6000.tar.gz

# 3. 解压 Wasserstein-GastroFlow 权重
unzip kvasir-extra.zip -d kvasir-extra
unzip kvasir.zip -d kvasir
```

### `kvasir-extra.zip` 解压后的文件说明

| 文件 | 说明 | 对应命令行参数 |
|------|------|----------------|
| `wass_model.pt` | Wasserstein 度量模型权重 | `--wass_model_path` |
| `otcfm_weights_step_50000.pt` | Flow-Match 轨迹生成器权重 | `--checkpoint` |
| `convnext2.pt` | 判停器权重 | `--sim_model_path` |

---

## 快速复现 Demo

> **预计运行时间**：每个 Demo 在单张 40 GB A100 GPU 上约 10–15 分钟。

下文使用以下路径占位符，请替换为您的实际路径：

| 占位符 | 含义 |
|--------|------|
| `<WEIGHT_DIR>` | 模型权重解压后的根目录 |
| `<PROJECT_ROOT>` | 项目根目录（即 `GastroAgent/`） |

---

### Demo A — Wasserstein-GastroFlow 推理 & 评估

**目标**：在 kvasir-two-label 数据集上运行 Wasserstein-GastroFlow，得到分类准确率。

#### 前置数据准备

1. 将 Hugging Face 上的 `image_option_two_label/` 目录下载到本地
2. 编辑 `demo_data/kvasir-two-label/final_eval/final_eval_flat.json`，确保：
   - `x0` 字段中的图像路径指向 `<PROJECT_ROOT>/demo_data/kvasir-two-label/final_eval/` 下的图像
   - `x1` 字段中的图像路径指向您下载的 `image_option_two_label/` 目录

#### A-1. 计算 Wasserstein 距离

```bash
conda activate GastroAgent

python wasserstein-gastroFlow/wass_flow_train_Kvasir/eval/cal_wass.py \
  --data_path ./demo_data/kvasir-two-label/final_eval/final_eval_flat.json \
  --checkpoint <WEIGHT_DIR>/kvasir-extra/otcfm_weights_step_50000.pt \
  --output_dir ./results/wass_kvasir_two_label \
  --wass_model_path <WEIGHT_DIR>/kvasir-extra/wass_model.pt \
  --sim_model_path <WEIGHT_DIR>/kvasir-extra/convnext2.pt
```

**预期输出**：`./results/wass_kvasir_two_label/result.json`

> 也可使用 Slurm 提交：`sbatch wasserstein-gastroFlow/wass_flow_train_Kvasir/eval/cal_wass.sh`（需先修改其中的路径）。

#### A-2. 计算分类准确率

```bash
python demo_code/wass_flow_match/cal_label_acc.py
```

> **注意**：运行前需编辑该脚本，将 `in_path` 修改为上一步生成的 `result.json` 路径。`dist_val` 字段包含不同距离策略下的计算结果。

**预期输出**：
- `eval_by_x0_clean.json` — 逐样本评估结果
- `summary_by_x0_clean.json` — 汇总统计

#### 完整复现 Paper 中 Kvasir 数据集结果

要获得论文报告的完整 Kvasir 准确率，还需对 **kvasir-three-label** 子集执行相同流程：

1. 使用 `kvasir.zip` 中的权重
2. 数据路径改为 `demo_data/kvasir-three-label/final_eval/final_eval_flat.json`
3. `x1` 参考图像使用 Hugging Face 上的 `image_options/` 目录
4. 将两个子集的结果合并，得到 Wasserstein-GastroFlow 在 Kvasir 完整集上的准确率

---

### Demo B — GastroAgent 端到端融合推理

**目标**：运行 GastroMLLM + Wasserstein-GastroFlow 融合流水线，得到 GastroAgent 的最终准确率。

#### B-1. 获取 GastroMLLM 预测概率

**运行前需修改**：编辑 `demo_code/kvasir_pipeline/step1_model_inference.py`：
- 将 `model_id` 改为 GastroMLLM 权重路径：`<WEIGHT_DIR>/LlavaQwen2-GRPO-Tricks-Total-CoT-6000`
- 将 `input_data_path` 改为：`<PROJECT_ROOT>/demo_data/kvasir-two-label/mllm/final_doctor_exam_43.json`
- 将 `output_data_path` 改为您期望的输出路径

```bash
python demo_code/kvasir_pipeline/run_full_pipeline.py
```

该脚本依次执行 4 个步骤：

| 步骤 | 脚本 | 输出文件 |
|------|------|----------|
| 1 | `step1_model_inference.py` | `new_eval_tsy_llm_with_trigger.json` |
| 2 | `step2_extract_answers_with_llm_latest.py` | `new_eval_tsy_llm_extracted.json` |
| 3 | `step3_reevaluate_correct.py` | `new_eval_tsy_llm_final.json` |
| 4 | `step4_reanalyze_trigger_performance.py` | `new_eval_tsy_llm_trigger_report.json` |

**最终输出**：`new_eval_tsy_llm_final.json`（包含 MLLM 的预测概率与分类结果）

#### B-2. 融合 MLLM 与 Flow 模型结果

```bash
python demo_code/kvasir_pipeline/fusion/run_fusion_only.py \
  --mllm-path <B-1步输出的 new_eval_tsy_llm_final.json 路径> \
  --flow-path <A-2步输出的 eval_by_x0_clean.json 路径> \
  --output-dir ./results/fusion_kvasir_two_label
```

可选超参数（使用默认值即可复现论文结果）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--alpha-min` | 0.3 | MLLM 不自信时的最小权重 |
| `--alpha-max` | 0.9 | MLLM 自信时的最大权重 |
| `--steepness` | 10.0 | sigmoid 陡峭度 |

**预期输出**：
- `fusion_results.json` — 逐样本融合结果
- `fusion_statistics.json` — 汇总准确率统计

#### 完整复现 Paper 中 GastroAgent 在 Kvasir 数据集上的结果

对 **kvasir-three-label** 子集重复以上步骤（MLLM 输入改为 `demo_data/kvasir-three-label/mllm/final_doctor_exam_62.json`），然后将两个子集结果合并。

---

## 训练说明

如需从头训练各模块，请参考以下说明。各模块可独立训练。

### 训练依赖关系

```
GastroMLLM (独立)        Flow-Match 生成器 (独立)
       \                       |
        \                      v
         \           Wasserstein-GastroFlow
          \                   /
           v                 v
         GastroAgent (加载权重 + 融合校准)
```

### 1. GastroMLLM

多模态大语言模型，用于医学推理与报告生成。

**SFT 训练**：

```bash
bash VLM-R1/src/open-r1-multimodal/run_scripts/sft_data_v1/stage3_generate_lora_run.sh
# 或 Slurm 提交:
sbatch VLM-R1/src/open-r1-multimodal/run_scripts/sft_data_v1/stage3_submmit.sh
```

**RL (GRPO) 微调**：

```bash
bash VLM-R1/src/open-r1-multimodal/run_scripts/rl/run_grpo_my_server_qwen_8gpu_nodes_run.sh
# 或 Slurm 提交:
sbatch VLM-R1/src/open-r1-multimodal/run_scripts/rl/run_grpo_my_server_qwen_8gpu_nodes_summit.sh
```

**产物**：GastroMLLM 模型权重

### 2. Flow-Match 生成器

学习生成轨迹，为 Wasserstein-GastroFlow 提供路径证据。**建议先于 Wasserstein-GastroFlow 训练完成。**

```bash
python train_flow_by_vae/train.py
```

**产物**：Flow-Match 生成器权重（如 `otcfm_weights_step_*.pt`）

### 3. Wasserstein-GastroFlow

基于最优传输代价的少样本学习模块，依赖 Flow-Match 生成器权重。

```bash
python wasserstein-gastroFlow/wass_flow_train_Kvasir/train/train_old_kvasir_Disease_extra.py \
  --train_json_glob <训练数据目录> \
  --test_json <测试数据JSON文件>
```

**产物**：Wasserstein 度量模型权重（`wass_model.pt`）

### 4. GastroAgent 整体融合

加载上述三个模块的权重，在验证集上校准融合超参数（`alpha-min`、`alpha-max`、`steepness`），或直接使用论文中的默认值。

---

## 项目结构

```
GastroAgent/
├── demo_code/                          # Demo 复现脚本
│   ├── kvasir_pipeline/                #   GastroAgent 端到端流水线
│   │   ├── run_full_pipeline.py        #     一键执行全部4步
│   │   ├── step1_model_inference.py    #     MLLM 推理
│   │   ├── step2_extract_answers_with_llm_latest.py
│   │   ├── step3_reevaluate_correct.py
│   │   ├── step4_reanalyze_trigger_performance.py
│   │   └── fusion/
│   │       ├── run_fusion_only.py      #     融合入口
│   │       └── fusion_pipeline.py      #     融合逻辑
│   └── wass_flow_match/                #   Wasserstein-GastroFlow 评估
│       ├── cal_wass.py                 #     计算 Wasserstein 距离
│       └── cal_label_acc.py            #     计算分类准确率
├── demo_data/                          # Demo 演示数据
│   ├── kvasir-two-label/
│   │   ├── final_eval/final_eval_flat.json
│   │   └── mllm/final_doctor_exam_43.json
│   └── kvasir-three-label/
│       ├── final_eval/final_eval_flat.json
│       └── mllm/final_doctor_exam_62.json
├── wasserstein-gastroFlow/             # Wasserstein-GastroFlow 训练 & 评估
│   ├── wass_flow_train_Kvasir/
│   ├── wass_flow_match_esophagus/
│   ├── wass_flow_match_stomach/
│   └── wass_flow_match_duodenum/
├── VLM-R1/                             # GastroMLLM 训练框架
├── train_flow_by_vae/                  # Flow-Match 生成器训练
├── discriminator/                      # 判停器训练
├── model_utils/                        # 模型辅助函数
├── my_models/                          # Flow-Match 模型定义
├── conditional_flow_matcher/           # 条件流匹配核心库
├── train_clip/                         # 医学视觉编码器
├── train_vae/                          # 潜在空间编-解码器
├── utils/                              # 通用辅助函数
├── assets/figures/                     # 文档图片
├── requirements.txt                    # Python 依赖
└── README.md
```

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{GastroAgent2026,
  title={Geometry-aware multimodal AI resolves the long-tail paradox in gastrointestinal diagnostics},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 致谢

- 感谢所有数据提供方和医学专家的支持
- 本项目基于 [PyTorch](https://pytorch.org/) 开发
- 感谢开源社区的贡献

---

## 许可证

本项目采用 [Apache-2.0 License](LICENSE) 开源协议。

**注意**：本项目仅供研究使用，不得用于临床诊断。任何医学决策应由专业医生做出。

---

## 联系方式

- **问题反馈**：[GitHub Issues](https://github.com/GastroAgent/GastroAgent/issues)
- **邮件**：shuyuetan0@gmail.com
