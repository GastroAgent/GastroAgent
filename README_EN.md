# GastroAgent

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1+-orange.svg)](https://pytorch.org/)

**Geometry-aware multimodal AI resolves the long-tail paradox in gastrointestinal diagnostics**

*Multimodal AI medical assistant for upper gastrointestinal endoscopy*

[English](README_EN.md) | [简体中文](README.md)

</div>

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Changelog](#-changelog)
- [System Requirements](#system-requirements)
- [Installation Guide](#-installation-guide)
- [Model Weights](#-model-weights)
- [Demo](#-demo)
  - [Demo 1: Inference (Quick Start)](#demo-1-inference-quick-start)
  - [Demo 2: Training Pipeline](#demo-2-training-pipeline)
- [Usage](#-usage)
- [Datasets](#-datasets)
- [Training](#-training)
  - [GastroMLLM](#gastromllm)
  - [Flow-Match Generator](#flow-match-generator)
  - [Wasserstein-GastroFlow](#wasserstein-gastroflow)
  - [Full Integration: GastroAgent](#full-integration-gastroagent)
- [Evaluation](#-evaluation)
- [Project Structure](#-project-structure)
- [Visualization](#-visualization)
- [Citation](#-citation)
- [Acknowledgements](#-acknowledgements)
- [License](#-license)

---

## 📖 Introduction

GastroAgent is a multimodal AI medical assistant system for upper gastrointestinal (GI) endoscopy. It is designed for real-world clinical scenarios where you need both **stable recognition of common lesions** and **reliable coverage of long-tail rare conditions**.

GI endoscopy faces a long-tail disease distribution: many clinically important diagnoses are relatively rare. Traditional AI systems are often optimized for common lesions only, leaving insufficient coverage for uncommon-but-critical diseases—one of the key barriers to safe clinical deployment.

We propose **GastroAgent (the overall diagnostic framework in this repository)**, which dynamically fuses the general reasoning ability of a multimodal large language model (**GastroMLLM**) with the geometric matching evidence from a few-shot learning module (**Wasserstein-GastroFlow**) via an **entropy-aware adaptive weight controller**. The few-shot module uses optimal transport as a similarity metric: along a learned generative trajectory, it transforms a query image and uses the transport cost required to map it to labeled support samples as evidence, providing interpretable support for uncommon diseases.

Across four GI benchmark datasets, this integrated workflow outperforms metric-learning and diffusion baselines, achieving **93.7%** diagnostic accuracy on the standardized Kvasir dataset. The adaptive fusion significantly improves long-tail cohorts while maintaining performance on common diseases: **81.4%** for uncommon esophageal lesions, **84.8%** for gastric lesions, and **83.8%** for duodenal lesions. By unifying multimodal report generation with robust few-shot recognition, GastroAgent provides a more complete, clinically aligned, broad-coverage endoscopic intelligence solution.

- **Overall workflow (diagram)**

![GastroAgent workflow ](/assets/figures/overview-ill.pdf)

## System Requirements

Ensure your environment meets the following requirements before installation.

### Hardware

| Item | Requirement |
|------|-------------|
| **CPU** | x86_64 (multi-core recommended) |
| **Memory** | ≥ 32 GB RAM recommended |
| **GPU** | NVIDIA GPU, ≥ 24 GB VRAM recommended (e.g. A100, V100, RTX 3090/4090) |
| **Storage** | ≥ 40 GB free space recommended (including models and datasets) |

### Software

| Item | Version |
|------|---------|
| **OS** | Linux (Ubuntu 20.04 / 22.04 recommended) |
| **Python** | ≥ 3.11 |
| **PyTorch** | ≥ 2.5.1 |
| **CUDA** | ≥ 12.1 (required for GPU training and inference) |
| **cuDNN** | Match your CUDA version |

### Optional Dependencies

- **Flash Attention**: Speeds up attention computation; install the appropriate wheel for your CUDA and PyTorch versions.
- Other Python dependencies are listed in the project root `requirements.txt`.

---

## 🛠️ Installation Guide

Follow the steps below to set up the environment and install dependencies.

### 1. Clone the repository

```bash
git clone https://github.com/GastroAgent/GastroAgent.git
cd GastroAgent
```

### 2. Create and activate a virtual environment

Using Conda (recommended):

```bash
conda create -n GastroAgent python=3.11
conda activate GastroAgent
```

Or using venv:

```bash
python3.11 -m venv venv
source venv/bin/activate   # Linux/macOS
# or on Windows: venv\Scripts\activate
```

### 3. Install PyTorch (with CUDA)

Choose the install command that matches your CUDA version from the [PyTorch website](https://pytorch.org/get-started/locally/). For example, CUDA 12.1:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

**Note**: If `requirements.txt` includes local-path dependencies (e.g. Flash Attention or custom packages), adjust those lines for your environment or install them separately before running the above command.

### 5. Verify installation

From the project root, check that PyTorch and CUDA are available:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

If the output shows `CUDA available: True`, the GPU environment is set up correctly.

---

## 📦 Model Weights

### Pretrained models

We provide the following trained model weights: GastroMLLM, Flow-Match generator, and Wasserstein-GastroFlow:
[Hugging Face](https://huggingface.co/GastroAgent/GastroAgent)

Specify the weight paths in the corresponding configuration files.

---

## 🎬 Demo

This section describes two runnable demos: **Inference (Quick Start)** and **Training pipeline**.

### Prerequisites

- Environment set up according to the [Installation Guide](#-installation-guide)
- Pretrained weights downloaded from [Model Weights](#-model-weights) and extracted to the path referred to as `your_path` in the commands below

---

### Demo 1: Inference (Quick Start)

Run batch inference on endoscopy images using the pretrained weights (run from the project root; replace `your_path` with your actual path):

```bash
conda activate GastroAgent

python wasserstein-gastroFlow/wass_flow_train_Kvasir/eval/cal_wass.py \
  --data_path ./dataset/eval_data/exam_dataset_extra_flatten.json \
  --checkpoint your_path/base-flow-match_vae/otcfm/otcfm_weights_step_55000.pt \
  --output_dir your_path/wass_flow_train_Kvasir/result \
  --wass_model_path your_path/best_flow_weights/attention_dy_tsy.pt \
  --sim_model_path your_path/discriminator/latent_model_weight/convnext2.pt
```

Or run directly:

```bash
sbatch wasserstein-gastroFlow/wass_flow_train_Kvasir/eval/cal_wass.sh
```

After inference, results are saved under `--output_dir` (default output is `result.json`). The terminal will show lesion type, site, confidence, and a short summary of few-shot matching evidence.

---

### Demo: GastroMLLM (multimodal LLM) training pipeline

- **SFT fine-tuning**:  
  `bash /mnt/inaisfs/data/home/tansy_criait/GasAgent-main/VLM-R1/src/open-r1-multimodal/run_scripts/sft_data_v1/stage3_generate_lora_run.sh`  
  Or submit via Slurm: `sbatch VLM-R1/src/open-r1-multimodal/run_scripts/sft_data_v1/stage3_submmit.sh`

- **RL fine-tuning**:  
  `bash VLM-R1/src/open-r1-multimodal/run_scripts/rl/run_grpo_my_server_qwen_8gpu_nodes_run.sh`  
  Or submit via Slurm: `sbatch VLM-R1/src/open-r1-multimodal/run_scripts/rl/run_grpo_my_server_qwen_8gpu_nodes_summit.sh`

### Demo: Wasserstein-GastroFlow training pipeline

The full training pipeline has three stages (replace `your_path` with your actual path):

**Stage 1: Train the Flow-Match generator**

```bash
conda activate GastroAgent

python train_flow_by_vae/train_kvasir_Disease.py \
  --data_path ./dataset \
  --output_dir your_path/base-flow-match_vae/otcfm \
  --epochs 100
```

**Stage 2: Train Wasserstein-GastroFlow**

```bash
python wasserstein-gastroFlow/wass_flow_train_Kvasir/train/flow_matcher.py \
  --data_path ./dataset \
  --checkpoint your_path/base-flow-match_vae/otcfm/otcfm_weights_step_55000.pt \
  --output_dir your_path/best_flow_weights \
  --epochs 100
```

**Stage 3: Train the discriminator**

```bash
python discriminator/train.py \
  --data_path ./dataset \
  --output_dir your_path/discriminator/latent_model_weight \
  --epochs 50
```

> For exact arguments in each stage, see the `--help` output of the corresponding script. More detailed training options are described in the [Training](#-training) section.

---

## 📘 Usage

General steps for running GastroAgent inference and (optionally) training.

### 1. Environment and data preparation

- Set up the environment according to [System Requirements](#system-requirements) and the [Installation Guide](#-installation-guide).
- Prepare endoscopy images for inference (common formats such as `.jpg`, `.png` are supported). For paper evaluation, prepare the datasets as described in [Datasets](#-datasets).

### 2. Download model weights

- Download pretrained weights from the link in [Model Weights](#-model-weights) (e.g. Hugging Face).
- Set the weight paths in the project config (e.g. `configs/inference_config.yaml` or the config for each module).

### 3. Run inference

- **Single or batch images**: Use the provided inference script (e.g. `inference.py`) with image path(s) and output directory.
- **Full diagnostic pipeline**: For the fused framework, configure weights for GastroMLLM, Flow-Match, and Wasserstein-GastroFlow; the Agent pipeline will run multimodal reasoning and few-shot evidence fusion automatically.

### 4. View results and visualization

- Inference output typically includes: diagnostic class, confidence, optional report text, and evidence visualization.
- For plots and result figures, see [Visualization](#-visualization) and [Evaluation](#-evaluation).

### 5. Advanced: training and evaluation

- For training or fine-tuning, follow the sub-module instructions (GastroMLLM, Flow-Match, Wasserstein-GastroFlow) in [Training](#-training).
- Metrics and reproduction steps are in [Evaluation](#-evaluation).

> **Disclaimer**: This project is for research only and must not replace clinical diagnosis. Any medical decision must be made by qualified physicians.

---

## 🔧 Training

Training in this project is organized into **three core parts** (module-level):

- **GastroMLLM**: Multimodal large language model (medical reasoning and report generation)
- **Flow-Match generator**: Generative model that learns trajectories / transformation paths
- **Wasserstein-GastroFlow**: Few-shot module based on optimal transport cost (uses Flow-Match trajectories as evidence paths)

Overall, **GastroAgent (full diagnostic framework) = GastroMLLM + Flow-Match generator + Wasserstein-GastroFlow + entropy-aware adaptive weight controller (fusion)**. The fusion stage usually does not require “training a new model from scratch”; it is mainly **loading the three module weights and performing necessary calibration on a validation set (e.g. thresholds)**.

> Below we describe each part by “module role + training inputs/outputs + artifacts”. For exact CLI options, refer to the training entry scripts (e.g. `train.py`) and configs (e.g. `configs/train_config.yaml`) in the repository.

### GastroMLLM

- **Goal**: Obtain a multimodal model with endoscopic scene understanding and medical report generation as the base for “medical reasoning + text generation”.
- **Typical training data**:
  - Endoscopy images / video frames (or keyframes) + structured labels (lesion type, site, attributes)
  - Text supervision (reports, conclusions, conversational instruction data, etc.)
- **Artifacts**: GastroMLLM weights.
- **Relation to other modules**: Can be trained **independently** of Flow-Match / Wasserstein-GastroFlow; used at inference in GastroAgent fusion to generate explanations and reports.

### Flow-Match Generator

- **Goal**: Learn a “query sample → reference distribution” generative trajectory / transformation path to provide an interpretable path for subsequent optimal transport cost computation.
- **Typical training data**: Mainly endoscopy images (optionally split by organ/site/lesion category) to learn stable trajectories.
- **Artifacts**: Flow-Match generator weights (for trajectories / intermediate states).
- **Relation to other modules**: Supplies trajectories (or cost-evaluation paths) to **Wasserstein-GastroFlow**; we recommend **training this module first** before training or evaluating the few-shot module.

### Wasserstein-GastroFlow

- **Goal**: Under few-shot / long-tail settings, use “optimal transport cost along the generative trajectory” as a similarity measure for robust geometric matching and interpretable evidence.
- **Typical training setup**:
  - Few-shot support set (labeled) and query set (to be recognized)
  - Use Flow-Match trajectories to compute transport cost / matching score from each query to each candidate class support set
- **Artifacts**: Few-shot module parameters (if any), support-set feature index, cost-metric config, and cost statistics / visualizations for evaluation.
- **Relation to other modules**: **Depends on the Flow-Match generator**; its output is a main source of “few-shot evidence” for the GastroAgent fusion controller.

### Full Integration: GastroAgent

- **Goal**: Adaptively fuse evidence from **GastroMLLM (multimodal medical model)** and **Wasserstein-GastroFlow (few-shot learning)** to better balance common-class stability and long-tail coverage.
- **Typical workflow**:
  - Load GastroMLLM, Wasserstein-GastroFlow (and Flow-Match generator) weights and configs
  - Calibrate **entropy/confidence-related fusion hyperparameters** on a validation set, or use our provided hyperparameters

---

## 📈 Evaluation

> It is recommended to keep result figures under `assets/figures/`. The paths below are placeholders; replace with the actual files when available.

### Result figures

- **Doctor dataset**

![Benchmark comparison](assets/figures/doctor-dataset.pdf)

- **Few-shot learning example**

![Long-tail performance](assets/figures/kshot-case.pdf)

- **Workflow results**

![Evidence visualization](assets/figures/workflow-result-ill.pdf)

---

## 📁 Project Structure

```
GastroAgent/
├── abnormal_dectect/           # lesion region detection
├── assets/                     # static assets (figures, etc.)
│   └── figures/                # README result/workflow figures
├── conditional_flow_matcher/   # conditional flow matching
├── dataset/                    # data processing scripts
│   ├── eval_data/              # evaluation data
│   ├── xxx                     # other test scripts
├── discriminator/              # discriminator training scripts
├── GasAgenteent/               # Agent trigger scripts
│   ├── Agent_pipeline_result/ # test results
├── model_utils/                # model helper functions
├── my_models/                  # Flow-Match model structure files
├── train_clip/                 # medical visual encoder
├── train_flow_by_vae/          # train Flow-Match generator
├── train_vae/                  # latent-space encoder-decoder
├── utils/                      # helper utilities
├── VLM-R1/                     # MLLM training framework
├── requirements.txt            # dependency list
├── wass_flow_match_duodenum/   # Wasserstein-GastroFlow
└── README.md                   # readme
```

## 📄 Citation

If this project is helpful for your research, please cite:

```bibtex
@article{GastroAgent2026,
  title={GastroAgent: Geometry-aware multimodal AI resolves the long-tail paradox in gastrointestinal diagnostics},
  author={Your Name and Others},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```

---

## 🙏 Acknowledgements

- Thanks to all data providers and medical experts for their support
- This project is built with [PyTorch](https://pytorch.org/) and [MMDetection](https://github.com/open-mmlab/mmdetection)
- Thanks to the open-source community

---

## 📜 License

This project is released under the [MIT License](LICENSE).

**Disclaimer**: This project is for research use only and must not be used for clinical diagnosis. Any medical decision should be made by qualified physicians.

---

## 📮 Contact

- **Issues**: [GitHub Issues](https://github.com/GastroAgent/GastroAgent/issues)
- **Email**: shuyuetan0@gmail.com
- **Homepage**: [Project homepage](https://yourproject.github.io)

---

## 🌟 Star History

If this project helps you, please give us a ⭐️!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/GastroAgent&type=Date)](https://star-history.com/#yourusername/GastroAgent&Date)
