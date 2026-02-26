# GastroAgent

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1+-orange.svg)](https://pytorch.org/)

**面向上消化道内窥镜的多模态人工智能医学助手**

*A Multimodal AI Medical Assistant for Upper Gastrointestinal Endoscopy*

[English](README_EN.md) | 简体中文

</div>

---

## 📋 目录

- [简介](#简介)
- [主要特性](#主要特性)
- [更新日志](#更新日志)
- [安装](#安装)
- [模型权重](#模型权重)
- [快速开始](#快速开始)
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

我们提出 **GastroAgent（本项目整体诊断框架）**：通过 **熵感知自适应权重控制器** 动态融合多模态大语言模型（**GastroMLLM**）的通用推理能力与少样本专科模块（**Wasserstein-GastroFlow**）的几何匹配结果。专科模块以最优传输（optimal transport）为相似性度量：沿学习到的生成轨迹，将查询图像变换到带标签样本所需的传输代价作为证据，从而为不常见疾病提供可解释支撑。

在四个 GI 基准数据集上，该集成工作流整体优于度量学习与扩散基线，并在标准化 Kvasir 数据集上达到 **93.7%** 诊断准确率。自适应融合在保持常见疾病性能的同时显著提升长尾队列：食管不常见病变 **81.4%**、胃部 **84.8%**、十二指肠 **83.8%**。通过统一多模态报告生成与稳健少样本识别，GastroAgent 提供了更完整、贴合临床的广谱内镜智能方案。

## 🛠️ 安装

### 环境要求

- Python >= 3.11
- PyTorch >= 2.5.1
- CUDA >= 12.1 (推荐使用GPU)
- 其他依赖见 `requirements.txt`

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/GastroAgent/GastroAgent.git
cd GastroAgent
```

2. **创建虚拟环境**

```bash
conda create -n GastroAgent python=3.11
conda activate GastroAgent
```

3. **安装依赖**

```bash
pip install -r requirements.txt
```

4. **安装项目**

```bash
pip install -e .
```

---

## 📦 模型权重

### 预训练模型

我们提供以下训练模型权重：GastroAgent
[huggingface](https://huggingface.co/GastroAgent/GastroAgent) 

或在配置文件中指定权重路径即可

---

## 🚀 快速开始

### 推理示例


## 🔧 训练

本项目的**训练核心分为三部分**（模块级训练），分别对应：

- **GastroMLLM**：多模态大语言模型（通用推理与报告生成）
- **Flow-Match 生成器**：学习生成轨迹/变换路径的生成模型
- **Wasserstein-GastroFlow**：基于最优传输代价的少样本专科模块（依赖 Flow-Match 的生成轨迹作为证据路径）

其中 **GastroAgent（整体诊断框架）= 上述三者 + 熵感知自适应权重控制器（融合）**。融合本身通常不需要“从零训练一个新模型”，更多是**加载三模块权重并做验证集校准/阈值设定**（如需）。

> 说明：下面以“模块职责 + 训练输入/输出 + 产物”来区分三部分，便于读者按需训练与复现。具体命令行参数以仓库现有训练入口（如 `train.py`）与配置文件（如 `configs/train_config.yaml`）为准。

### GastroMLLM

- **目标**：获得具备内镜场景理解与医学报告生成能力的多模态模型，用于“通用推理 + 文本生成”能力底座。
- **典型训练数据**：
  - 内镜图像/视频帧（或关键帧） + 结构化标注（病灶类型/部位/属性）
  - 文本侧监督（报告、结论、对话式指令数据等）
- **训练产物**：GastroMLLM 权重（以及 tokenizer / 适配器权重等，如有）。
- **与其他模块关系**：与 Flow-Match / Wasserstein-GastroFlow **可独立训练**；在 GastroAgent 推理融合阶段被调用以生成解释与报告。

### Flow-Match 生成器

- **目标**：学习“从查询样本到参考分布”的生成轨迹/变换路径，为后续最优传输代价计算提供可解释的路径证据。
- **典型训练数据**：以内镜图像为主（可按器官/部位/病灶类别划分训练子域），用于学习稳定的生成轨迹。
- **训练产物**：Flow-Match 生成器权重（用于生成轨迹/中间态）。
- **与其他模块关系**：为 **Wasserstein-GastroFlow** 提供生成轨迹（或轨迹上的代价评估路径）；因此一般建议**先完成该模块训练**再训练/评估专科模块。

### Wasserstein-GastroFlow

- **目标**：在少样本/长尾类别下，使用“沿生成轨迹的最优传输代价”作为相似性度量，实现稳健的几何匹配与可解释证据输出。
- **典型训练/构建方式**：
  - few-shot 支持集（带标签）与查询集（待识别）
  - 结合 Flow-Match 生成轨迹，计算查询到各候选类别支持集的传输代价/匹配分数
- **训练产物**：专科模块参数（如有）与/或支持集特征索引、代价度量配置；以及用于评估的代价统计与可视化结果。
- **与其他模块关系**：**依赖 Flow-Match 生成器**；其输出作为 GastroAgent 融合控制器的重要“专科证据”来源。

### 整体融合：GastroAgent

- **目标**：把 **GastroMLLM（通用）** 与 **Wasserstein-GastroFlow（专科）** 的证据进行自适应融合，在常见类稳定性与长尾覆盖之间取得更优平衡。
- **典型流程**：
  - 加载 GastroMLLM 与 Wasserstein-GastroFlow（以及 Flow-Match 生成器）权重/配置
  - 在验证集上做**熵/置信度相关的融合超参数校准**（例如阈值、权重范围、温度等；若实现中包含）
  - 固化为推理配置（如 `configs/inference_config.yaml`）供后续批量推理/评估使用
  
> 如果你希望我把这里的“命令行示例/参数名”也补得非常具体（例如每个模块对应的脚本、参数、输出目录），我可以继续只基于 `README.md` 里已经出现的入口文件名，把示例写成**不依赖阅读其它文件**的形式，并用“占位符变量”保证不会误导。

## 📈 评估

---

## 📁 项目结构

```
GastroAgent/
├── configs/                  # 配置文件
│   ├── train_config.yaml
│   └── inference_config.yaml
├── data/                     # 数据目录
│   ├── train/
│   ├── val/
│   └── test/
├── GastroAgent/                 # 核心代码
│   ├── models/              # 模型定义
│   ├── datasets/            # 数据集加载
│   ├── utils/               # 工具函数
│   └── core/                # 核心算法
├── scripts/                  # 脚本文件
│   ├── inference.py
│   └── video_inference.py
├── tools/                    # 工具脚本
│   ├── prepare_data.py
│   └── generate_report.py
├── demo/                     # 演示代码
│   └── app.py
├── weights/                  # 模型权重
├── wass_flow_match_duodenum/ # Wasserstein流匹配模块
│   └── eval/
│       └── cal_wass.py
├── train.py                  # 训练脚本
├── eval.py                   # 评估脚本
├── requirements.txt          # 依赖列表
└── README.md                 # 本文件
```

## 📄 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{GastroAgent2024,
  title={GastroAgent: A Multimodal AI Medical Assistant for Upper Gastrointestinal Endoscopy},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

---

## 🙏 致谢

- 感谢所有数据提供方和医学专家的支持
- 本项目基于 [PyTorch](https://pytorch.org/) 和 [MMDetection](https://github.com/open-mmlab/mmdetection) 开发
- 感谢开源社区的贡献

---

## 📜 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

**注意**：本项目仅供研究使用，不得用于临床诊断。任何医学决策应由专业医生做出。

---

## 📮 联系方式

- **问题反馈**: [GitHub Issues](https://github.com/yourusername/GastroAgent/issues)
- **邮件**: your.email@example.com
- **主页**: [项目主页](https://yourproject.github.io)

---

## 🌟 Star History

如果这个项目对您有帮助，请给我们一个 ⭐️！

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/GastroAgent&type=Date)](https://star-history.com/#yourusername/GastroAgent&Date)
