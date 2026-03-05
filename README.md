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

## 📋 目录

- [简介](#简介)
- [主要特性](#主要特性)
- [更新日志](#更新日志)
- [系统要求](#系统要求)
- [安装指南](#安装指南)
- [模型权重](#模型权重)
- [Demo](#demo)
  - [Demo 1：推理（快速开始）](#demo-1推理快速开始)
- [使用说明](#使用说明)
- [数据集](#数据集)
- [训练](#训练)
  - [GastroMLLM](#gastromllm)
  - [Flow-Match 生成器](#flow-match-生成器)
  - [Wasserstein-GastroFlow](#wasserstein-gastroflow)
  - [整体融合：GastroAgent](#整体融合gastroagent)
- [评估](#评估)
- [项目结构](#项目结构)
- [可视化](#可视化)
- [引用](#引用)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 📖 简介

GastroAgent 是一个面向上消化道内窥镜检查的多模态人工智能医学助手系统，旨在在真实临床场景中同时兼顾**常见病灶的稳定识别**与**长尾罕见病变的可靠覆盖**。

胃肠道（GI）内镜检查面临疾病谱系的长尾分布：许多具有重要临床意义的诊断相对少见。传统 AI 系统往往只对常见病灶优化，导致对不常见但关键的疾病缺乏覆盖，成为安全临床部署的主要障碍。

我们提出 **GastroAgent（本项目整体诊断框架）**：通过 **熵感知自适应权重控制器** 动态融合多模态大语言模型（**GastroMLLM**）的通用推理能力与少样本学习模块（**Wasserstein-GastroFlow**）的几何匹配结果。该模块以最优传输（optimal transport）为相似性度量：沿学习到的生成轨迹，将查询图像变换到带标签样本所需的传输代价作为证据，从而为不常见疾病提供可解释支撑。

在四个 GI 基准数据集上，该集成工作流整体优于度量学习与扩散基线，并在标准化 Kvasir 数据集上达到 **93.7%** 诊断准确率。自适应融合在保持常见疾病性能的同时显著提升长尾队列：食管不常见病变 **81.4%**、胃部 **84.8%**、十二指肠 **83.8%**。通过统一多模态报告生成与稳健少样本识别，GastroAgent 提供了更完整、贴合临床的广谱内镜智能方案。

- **整体工作流（流程图）**

![GastroAgent workflow ](/assets/figures/overview-ill.pdf)

## 系统要求

在安装前请确保您的环境满足以下要求。

### 硬件

| 项目 | 要求 |
|------|------|
| **CPU** | x86_64（推荐多核） |
| **内存** | 建议 ≥ 32 GB RAM |
| **GPU** | NVIDIA GPU，显存建议 ≥ 24 GB（如 A100、V100、RTX 3090/4090） |
| **存储** | 建议 ≥ 40 GB 可用空间（含模型与数据集） |

### 软件

| 项目 | 版本要求 |
|------|----------|
| **操作系统** | Linux（推荐 Ubuntu 20.04 / 22.04） |
| **Python** | ≥ 3.11 |
| **PyTorch** | ≥ 2.5.1 |
| **CUDA** | ≥ 12.1（GPU 训练与推理必需） |
| **cuDNN** | 与 CUDA 版本匹配 |

已测试环境：Ubuntu 22.04, Python 3.11.8, PyTorch 2.5.1, CUDA 12.1

### 可选依赖

- **Flash Attention**：加速注意力计算，需根据 CUDA 与 PyTorch 版本自行安装对应 wheel。
- 其余 Python 依赖见项目根目录 `requirements.txt`。

---

## 安装指南

按以下步骤完成环境搭建与依赖安装。
**安装预计耗时 (Typical install time on a standard desktop):** 约 10-15 分钟（取决于网络下载速度）。

### 1. 克隆仓库

```bash
git clone https://github.com/GastroAgent/GastroAgent.git
cd GastroAgent
```

### 2. 创建并激活虚拟环境

使用 Conda（推荐）：

```bash
conda create -n GastroAgent python=3.11
conda activate GastroAgent
```

或使用 venv：

```bash
python3.11 -m venv venv
source venv/bin/activate   # Linux/macOS
# 或 Windows: venv\Scripts\activate
```

### 3. 安装 PyTorch（含 CUDA）

请根据 [PyTorch 官网](https://pytorch.org/get-started/locally/) 选择与您 CUDA 版本匹配的安装命令，例如 CUDA 12.1：

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

**注意**：若 `requirements.txt` 中包含本地路径依赖（如 Flash Attention、自定义包），请根据您的环境修改对应行或先单独安装这些依赖后再执行上述命令。

### 5. 验证安装

在项目根目录下检查 PyTorch 与 CUDA 是否可用：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

若输出中 `CUDA available: True`，则 GPU 环境配置正确。

---

## 📦 模型权重

### 预训练模型

我们提供以下训练模型权重，包括GastroMLLM、Flow-Match 生成器和Wasserstein-GastroFlow：
[huggingface](https://huggingface.co/GastroAgent/GastroAgent) 

在对应文件中指定权重路径即可


我们在 Hugging Face 仓库中提供了以下模型权重和演示数据集供下载：

| 文件名 (File Name) | 大小 (Size) | 说明 (Description) |
| :--- | :--- | :--- |
| `LlavaQwen2-GRPO-Tricks-Total-CoT-6000.tar.gz` | 12.6 GB | GastroMLLM 预训练模型权重（经过强化学习 GRPO 微调） |
| `kvasir-extra.zip` | 7.44 GB | 针对kvasir-two-label数据集的Wasserstein-GastroFlow权重用于Demo 运行与复现） |
| `kvasir.zip` | 7.42 GB | 针对kvasir-three-label数据集的Wasserstein-GastroFlow权重（用于 Demo 运行与复现） |


#### `kvasir-extra.zip` 内容清单及用途说明：
解压 `kvasir-extra.zip` 后，您将获得执行 Wasserstein-GastroFlow 推理所需的完整权重与测试配置：

* **`wass_model.pt`**: Wasserstein度量的核心模型权重（对应Demo启动命令中的 `--wass_model_path`）。
* **`otcfm_weights_step_50000.pt`**: Flow-Match 轨迹生成器的预训练权重（对应Demo启动命令中的 `--checkpoint`）。
* **`convnext2.pt`**: 判停器权重（对应Demo启动命令中的 `--sim_model_path`）。



## 🎬 Demo

本节提供推理演示流程，训练相关说明请参见 [训练](#训练) 章节。

### 前置条件

- 已完成 [安装指南](#安装指南) 中的环境配置
- 已从 [模型权重](#模型权重) 下载预训练权重，并解压至下方命令中 `your_path` 对应路径

---

### 快速开始1：Wasserstein-GastroFlow在kvasir-two-label数据集上的演示

使用预训练权重对内镜图像进行批量推理（请在项目根目录下执行，并将`data_path`中json所对应的`x0`所对应的image路径为/mnt/inaisfs/data/home/tansy_criait/GasAgent-main/demo_data/kvasir-two-label/final_eval，`x1`所对应的image路径为https://huggingface.co/GastroAgent/GastroAgent/tree/main中的image_option_two_label ，`your_path` 替换为实际路径，）：


```bash
conda activate GastroAgent

python wasserstein-gastroFlow /wass_flow_train_Kvasir/eval/cal_wass.py \
  --data_path ./demo_data/kvasir-two-label/final_eval/final_eval_flat.json \  
  --checkpoint your_path/base-flow-match_vae/otcfm/otcfm_weights_step_50000.pt \
  --output_dir your_path/wass_flow_train_Kvasir/result \
  --wass_model_path your_path/best_flow_weights/wass_model.pt \
  --sim_model_path your_path/discriminator/latent_model_weight/convnext2.pt

或者直接使用
sbatch wasserstein-gastroFlow/wass_flow_train_Kvasir/eval/cal_wass.sh
```

推理完成后，结果将保存在 `--output_dir` 指定目录（默认生成 `result.json`）。

得到各图像path聚合后的距离，根据我们的策略，将动态距离最后转化为图像分类的指标，两张图像之间距离越小越相似作为判别标准，最终得出在数据集上的分类acc情况
python demo_code /wass_flow_match/cal_label_acc.py 

将`in_path`修改为实际执行后得到的result.json，不同path策略下的距离可参见`dist_val`字段
计算完成后结果将保存在`save_dir`指定路径（默认生成`eval_by_x0_clean.json`，`summary_by_x0_clean.json`）


注：若要完全复现paper中Wasserstein-GastroFlow在kvasir数据集上的acc，需要按上述操作将demo_data中的kvasir-three-label按照上述步骤进行推理(数据路径参见demo_data下的kvasir-three-label/final_eval/final_eval_flat.json，image路径同样更新为demo_data下的kvasir-three-label/final_eval以及https://huggingface.co/GastroAgent/GastroAgent/tree/main中的image_options)。最后进行合并后，得出Wasserstein-GastroFlow在kvasir subset上的acc。

预期运行时间：在单张 40G A100 GPU 上，该 Demo 流程预计耗时约 10-15 分钟。


### 快速开始2：GastroAgent在kvasir-two-label数据集上的演示
1、获得mllm后的预测概率文件
python demo_code /kvasir_pipeline/run_full_pipeline.py

需要在调用的step1_model_inference.py中修改`model_id`为https://huggingface.co/GastroAgent/GastroAgent/tree/main/下的LlavaQwen2-GRPO-Tricks-Total-CoT-6000.tar.gz
，`input_data_path`为demo_data下的/mnt/inaisfs/data/home/tansy_criait/GasAgent-main/demo_data/kvasir-two-label/mllm/final_doctor_exam_43.json

推理完成后，结果将保存在 `output_data_path` 指定目录（默认生成 `new_eval_tsy_llm_final.json`等）。

2、将MLLM和 Flow模型的预测结果进行动态加权融合，获得最终的acc文件
python demo_code /kvasir_pipeline/fusion/run_fusion_only.py

修改`mllm-path`为我们第1步执行后名为`new_eval_tsy_llm_final.json`的文件，`flow-path`修改为Wasserstein-GastroFlow模型经计算后的结果文件`eval_by_x0_clean.json`


推理完成后，结果将保存在 `output-dir` 指定目录（默认生成 `fusion_results.json`, `fusion_statistics.json`）。

注：若要完全复现paper中GastroAgent在kvasir数据集上的acc，需要按上述操作将demo_data中的kvasir-three-label按照上述步骤进行推理(数据路径参见demo_data下的kvasir-three-label/mllm/final_doctor_exam_62.json。最后进行合并后，得出GastroAgent在kvasir subset上的acc。

预期运行时间：在单张 40G A100 GPU 上，该 Demo 流程预计耗时约 10-15 分钟。


## 📘 训练说明

以下为使用 GastroAgent 进行（可选）训练的通用步骤说明。

### 1. 环境与数据准备

- 按照 [系统要求](#系统要求) 与 [安装指南](#安装指南) 完成环境搭建
- 准备待推理的内镜图像（支持常见格式如 `.jpg`、`.png`）；若需复现论文评估，请按 [数据集](#数据集) 说明准备对应数据集

### 2. 下载模型权重（可选）

- 从 [模型权重](#模型权重) 提供的链接（如 Hugging Face）下载预训练权重
- 在项目配置文件（如 `configs/inference_config.yaml` 或各模块对应配置）中指定权重路径

### 3. 模块介绍与训练方式

本项目的训练核心分为三部分（模块级训练）：

- **GastroMLLM**：多模态大语言模型（医学推理与报告生成）
- **Flow-Match 生成器**：学习生成轨迹/变换路径的生成模型
- **Wasserstein-GastroFlow**：基于最优传输代价的少样本学习模块（依赖 Flow-Match 的生成轨迹作为证据路径）

整体上，**GastroAgent（整体诊断框架）= GastroMLLM + Flow-Match 生成器 + Wasserstein-GastroFlow + 熵感知自适应权重控制器（融合）**。融合阶段通常不需要“从零训练一个新模型”，更多是**加载三模块权重，并在验证集上进行必要的校准（如阈值等）**。

以下按“模块职责 + 训练输入/输出 + 产物 + 训练命令”说明，便于按需复现。

#### 3.1 GastroMLLM

- **目标**：获得具备内镜场景理解与医学报告生成能力的多模态模型，用于“医学推理 + 文本生成”能力底座。
- **典型训练数据**：
  - 内镜图像/视频帧（或关键帧） + 结构化标注（病灶类型/部位/属性）
  - 文本侧监督（报告、结论、对话式指令数据等）
- **训练产物**：GastroMLLM 权重。
- **与其他模块关系**：与 Flow-Match / Wasserstein-GastroFlow **可独立训练**；在 GastroAgent 推理融合阶段被调用以生成解释与报告。

- **SFT 训练**：
  - 直接运行：`bash VLM-R1/src/open-r1-multimodal/run_scripts/sft_data_v1/stage3_generate_lora_run.sh`
  - Slurm 提交：`sbatch VLM-R1/src/open-r1-multimodal/run_scripts/sft_data_v1/stage3_submmit.sh`

- **RL 微调**：
  - 直接运行：`bash VLM-R1/src/open-r1-multimodal/run_scripts/rl/run_grpo_my_server_qwen_8gpu_nodes_run.sh`
  - Slurm 提交：`sbatch VLM-R1/src/open-r1-multimodal/run_scripts/rl/run_grpo_my_server_qwen_8gpu_nodes_summit.sh`

#### 3.2 Flow-Match 生成器

- **目标**：学习“从查询样本到参考分布”的生成轨迹/变换路径，为后续最优传输代价计算提供可解释的路径证据。
- **典型训练数据**：以内镜图像为主（可按器官/部位/病灶类别划分训练子域），用于学习稳定的生成轨迹。
- **训练产物**：Flow-Match 生成器权重（用于生成轨迹/中间态）。
- **与其他模块关系**：为 **Wasserstein-GastroFlow** 提供生成轨迹（或轨迹上的代价评估路径）；因此一般建议**先完成该模块训练**再训练/评估少样本学习模块。

训练命令：`python train_flow_by_vae/train.py`

#### 3.3 Wasserstein-GastroFlow

- **目标**：在少样本/长尾类别下，使用“沿生成轨迹的最优传输代价”作为相似性度量，实现稳健的几何匹配与可解释证据输出。
- **典型训练/构建方式**：
  - few-shot 支持集（带标签）与查询集（待识别）
  - 结合 Flow-Match 生成轨迹，计算查询到各候选类别支持集的传输代价/匹配分数
- **训练产物**：少样本学习模块参数（如有）与/或支持集特征索引、代价度量配置；以及用于评估的代价统计与可视化结果。
- **与其他模块关系**：**依赖 Flow-Match 生成器**；其输出作为 GastroAgent 融合控制器的重要“少样本学习证据”来源。

训练命令：
`python wasserstein-gastroFlow/wass_flow_train_Kvasir/train/train_old_kvasir_Disease_extra.py \
  --train_json_glob your_train_dir \
  --test_json your_test.json`

### 4. 整体融合：GastroAgent

- **目标**：把 **GastroMLLM（医学多模态模型）** 与 **Wasserstein-GastroFlow（少样本学习）** 的证据进行自适应融合，在常见类稳定性与长尾覆盖之间取得更优平衡。
- **典型流程**：
  - 加载 GastroMLLM 与 Wasserstein-GastroFlow（以及 Flow-Match 生成器）权重/配置
  - 在验证集上做**熵/置信度相关的融合超参数校准** 或者直接使用我们的超参数设置

### 结果图

- **医生数据集**

![Benchmark comparison](assets/figures/doctor-dataset.pdf)

- **少样本学习示例**

![Long-tail performance](assets/figures/kshot-case.pdf)

- **工作流结果**

![Evidence visualization](assets/figures/workflow-result-ill.pdf)

---

## 📁 项目结构

```
GastroAgent/
├── abnormal_dectect/           # 病灶区域检测
├── assets/                     # 文档图片等静态资源
│   └── figures/                # README 结果图/流程图
├── conditional_flow_matcher/   # 条件流匹配
├── dataset/                    # 数据处理脚本
│   ├── eval_data/              # 测试数据
│   ├── xxx                     # 其他测试脚本
├── discriminator/              # 判停器的训练相关脚本
├── GasAgenteent/               # Agent 触发脚本
│   ├── Agent_pipeline_result/  # 测试结果
├── model_utils/                # 模型的辅助函数
├── my_models/                  # Flow-Match 模型结构文件
├── train_clip/                 # 医学视觉编码器
├── train_flow_by_vae/          # 训练 Flow-Match 生成器
├── train_vae/                  # 潜在空间编-解码器
├── utils/                      # 辅助函数
├── VLM-R1/                     # MLLM 训练框架
├── requirements.txt            # 依赖列表
├── wass_flow_match_duodenum/   # Wasserstein-GastroFlow 
└── README.md                   # readme
```

## 📄 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{GastroAgent2026,
  title={GastroAgent: Geometry-aware multimodal AI resolves the long-tail paradox in gastrointestinal diagnostics},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 🙏 致谢

- 感谢所有数据提供方和医学专家的支持
- 本项目基于 [PyTorch](https://pytorch.org/) 和 [MMDetection](https://github.com/open-mmlab/mmdetection) 开发
- 感谢开源社区的贡献

---

## 📜 许可证

本项目采用 [Apache-2.0 License](LICENSE) 开源协议。

**注意**：本项目仅供研究使用，不得用于临床诊断。任何医学决策应由专业医生做出。

---

## 📮 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/GastroAgent/GastroAgent/issues)
- **邮件**: shuyuetan0@gmail.com
- **主页**: [项目主页](https://yourproject.github.io)

---

## 🌟 Star History

如果这个项目对您有帮助，请给我们一个 ⭐️！

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/GastroAgent&type=Date)](https://star-history.com/#yourusername/GastroAgent&Date)
