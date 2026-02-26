# GastroAgent

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.12+-orange.svg)](https://pytorch.org/)

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
- [评估](#评估)
- [项目结构](#项目结构)
- [可视化](#可视化)
- [引用](#引用)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 📖 简介

GastroAgent 是一个面向上消化道内窥镜检查的多模态人工智能医学助手系统，旨在在真实临床场景中同时兼顾**常见病灶的稳定识别**与**长尾罕见病变的可靠覆盖**。

### 核心内容（3 部分）

#### Part I：背景与挑战（长尾覆盖难题）
胃肠道（GI）内镜检查面临疾病谱系的长尾分布：许多具有重要临床意义的诊断相对少见。传统 AI 系统往往只对常见病灶优化，导致对不常见但关键的疾病缺乏覆盖，成为安全临床部署的主要障碍。

#### Part II：方法概述（通用推理 + 专科精度的自适应融合）
我们提出 **GastroAgent（本项目整体诊断框架）**：通过 **熵感知自适应权重控制器** 动态融合多模态大语言模型（**GastroMLLM**）的通用推理能力与少样本专科模块（**Wasserstein-GastroFlow**）的几何匹配结果。专科模块以最优传输（optimal transport）为相似性度量：沿学习到的生成轨迹，将查询图像变换到带标签样本所需的传输代价作为证据，从而为不常见疾病提供可解释支撑。

#### Part III：结果与结论（跨数据集验证与长尾收益）
在四个 GI 基准数据集上，该集成工作流整体优于度量学习与扩散基线，并在标准化 Kvasir 数据集上达到 **93.7%** 诊断准确率。自适应融合在保持常见疾病性能的同时显著提升长尾队列：食管不常见病变 **81.4%**、胃部 **84.8%**、十二指肠 **83.8%**。通过统一多模态报告生成与稳健少样本识别，GastroAgent 提供了更完整、贴合临床的广谱内镜智能方案。

---

## ✨ 主要特性

- **多模态融合**：整合图像、视频和文本信息进行综合分析
- **高精度检测**：采用最新的目标检测和分割算法
- **实时处理**：优化的推理流程支持实时视频分析
- **可解释性**：提供可视化的注意力机制和决策依据
- **易于部署**：支持多种部署方式（CPU/GPU/分布式）
- **可扩展架构**：模块化设计便于功能扩展

---

## 🔄 更新日志

- **[2024-XX-XX]** v1.0.0 - 首次发布
  - 发布完整的训练和推理代码
  - 提供预训练模型权重
  - 支持多种数据集格式

---

## 🛠️ 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12.0
- CUDA >= 11.3 (推荐使用GPU)
- 其他依赖见 `requirements.txt`

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/yourusername/GastroAgent.git
cd GastroAgent
```

2. **创建虚拟环境**

```bash
conda create -n GastroAgent python=3.8
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

我们提供以下预训练模型权重：

| 模型名称 | 骨干网络 | 数据集 | mAP | 下载链接 |
|---------|---------|--------|-----|---------|
| GastroAgent-Base | ResNet-50 | Dataset-1 | 85.3% | [百度云](链接) / [Google Drive](链接) |
| GastroAgent-Large | ResNet-101 | Dataset-1 | 87.6% | [百度云](链接) / [Google Drive](链接) |
| GastroAgent-ViT | ViT-B/16 | Dataset-2 | 89.1% | [百度云](链接) / [Google Drive](链接) |

### 下载与配置

下载模型权重后，请将文件放置在以下目录：

```
GastroAgent/
├── weights/
│   ├── GastroAgent_base.pth
│   ├── GastroAgent_large.pth
│   └── GastroAgent_vit.pth
```

或在配置文件中指定权重路径：

```yaml
model:
  checkpoint: /path/to/your/weights.pth
```

---

## 🚀 快速开始

### 推理示例

#### 单张图像推理

```python
from GastroAgent import GastroAgent

# 加载模型
model = GastroAgent.from_pretrained('weights/GastroAgent_base.pth')

# 推理
result = model.predict('path/to/image.jpg')
print(result)
```

#### 批量推理

```bash
python scripts/inference.py \
    --model weights/GastroAgent_base.pth \
    --input data/test_images/ \
    --output results/ \
    --device cuda
```

#### 视频推理

```bash
python scripts/video_inference.py \
    --model weights/GastroAgent_base.pth \
    --video path/to/video.mp4 \
    --output output_video.mp4
```

### Demo演示

运行Web界面进行交互式演示：

```bash
python demo/app.py --port 7860
```

然后在浏览器中打开 `http://localhost:7860`

---

## 📊 数据集

### 支持的数据集

- **Dataset-1**: 上消化道内窥镜图像数据集
- **Dataset-2**: 多中心内窥镜视频数据集
- **自定义数据集**: 支持自定义数据格式

### 数据格式

请按以下格式组织您的数据：

```
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── annotations/
│       ├── image1.json
│       └── image2.json
├── val/
└── test/
```

标注文件格式（JSON）：

```json
{
  "image_id": "image1.jpg",
  "annotations": [
    {
      "category": "ulcer",
      "bbox": [x, y, w, h],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "score": 0.95
    }
  ]
}
```

### 数据准备

运行数据预处理脚本：

```bash
python tools/prepare_data.py \
    --input raw_data/ \
    --output data/ \
    --split 0.8 0.1 0.1
```

---

## 🔧 训练

### 配置训练参数

编辑配置文件 `configs/train_config.yaml`：

```yaml
model:
  name: GastroAgent_base
  backbone: resnet50
  pretrained: true

data:
  train_path: data/train
  val_path: data/val
  batch_size: 16
  num_workers: 4

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: AdamW
  scheduler: CosineAnnealingLR
```

### 开始训练

单GPU训练：

```bash
python train.py --config configs/train_config.yaml
```

多GPU分布式训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    train.py --config configs/train_config.yaml
```

### 断点续训

```bash
python train.py \
    --config configs/train_config.yaml \
    --resume checkpoints/last.pth
```

---

## 📈 评估

### 运行评估

```bash
python eval.py \
    --model weights/GastroAgent_base.pth \
    --data data/test \
    --output results/evaluation.json
```

### 评估指标

支持以下评估指标：

- **检测指标**: mAP, Precision, Recall, F1-Score
- **分割指标**: IoU, Dice Score
- **分类指标**: Accuracy, AUC-ROC

### 生成评估报告

```bash
python tools/generate_report.py \
    --results results/evaluation.json \
    --output report.html
```

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

---

## 🎨 可视化

### 结果可视化

```python
from GastroAgent.visualization import visualize_results

# 可视化检测结果
visualize_results(
    image='path/to/image.jpg',
    predictions=results,
    output='output.jpg',
    show_labels=True,
    show_scores=True
)
```

### 注意力图可视化

```bash
python tools/visualize_attention.py \
    --model weights/GastroAgent_base.pth \
    --image path/to/image.jpg \
    --output attention_map.jpg
```

---

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
