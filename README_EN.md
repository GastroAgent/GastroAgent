# GastroAgent

<div align="center">

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5.1+-orange.svg)](https://pytorch.org/)

**Geometry-aware multimodal AI resolves the long-tail paradox in gastrointestinal diagnostics**

*面向上消化道内窥镜的多模态人工智能医学助手*

English | [简体中文](README.md)

</div>

---

## 📋 Table of Contents

- [Introduction](#-introduction)
- [Key Features](#-key-features)
- [Changelog](#-changelog)
- [Installation](#-installation)
- [Model Weights](#-model-weights)
- [Quick Start](#-quick-start)
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

## ✨ Key Features

> This section is currently a placeholder in `README.md`. Feel free to expand it with bullet points such as “multimodal report generation”, “few-shot long-tail recognition”, “evidence visualization”, etc.

## 🧾 Changelog

> This section is currently a placeholder in `README.md`. Consider adding versioned release notes (e.g., `v0.1.0`).

## 🛠️ Installation

### Requirements

- Python >= 3.11
- PyTorch >= 2.5.1
- CUDA >= 12.1 (GPU recommended)
- See `requirements.txt` for other dependencies

### Steps

1. **Clone the repository**

```bash
git clone https://github.com/GastroAgent/GastroAgent.git
cd GastroAgent
```

2. **Create a virtual environment**

```bash
conda create -n GastroAgent python=3.11
conda activate GastroAgent
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Install the project**

```bash
pip install -e .
```

---

## 📦 Model Weights

### Pretrained Models

We provide trained weights for GastroMLLM, the Flow-Match generator, and Wasserstein-GastroFlow:
[Hugging Face](https://huggingface.co/GastroAgent/GastroAgent)

Specify the weight paths in your configuration file.

---

## 🚀 Quick Start

### Inference Example

> This section is currently empty in `README.md`. Consider adding a runnable command example (inputs, config path, and output directory).

## 🗂️ Datasets

> This section is currently a placeholder in `README.md`. Consider documenting dataset download links, preprocessing steps, and expected directory layout.

## 🔧 Training

The **training in this project is organized into three core parts** (module-level training):

- **GastroMLLM**: multimodal LLM (medical reasoning and report generation)
- **Flow-Match Generator**: a generative model that learns trajectories / transformation paths
- **Wasserstein-GastroFlow**: an optimal-transport-cost-based few-shot module (uses Flow-Match trajectories as evidence paths)

Overall, **GastroAgent (the full diagnostic framework) = GastroMLLM + Flow-Match Generator + Wasserstein-GastroFlow + an entropy-aware adaptive weight controller (fusion)**. The fusion stage usually does not require “training a brand-new model from scratch”; it is more about **loading the three module weights and performing necessary calibration on a validation set (e.g., thresholds)**.

> Note: Below we describe each part in terms of “module responsibility + training inputs/outputs + artifacts” so you can train and reproduce only what you need. For exact CLI arguments, follow the repository’s training entry points (e.g., `train.py`) and configuration files (e.g., `configs/train_config.yaml`).

### GastroMLLM

- **Goal**: obtain a multimodal model capable of endoscopic scene understanding and medical report generation as the foundation for “medical reasoning + text generation”.
- **Typical training data**:
  - Endoscopy images / video frames (or keyframes) + structured labels (lesion type / site / attributes)
  - Text supervision (reports, conclusions, conversational instruction data, etc.)
- **Artifacts**: GastroMLLM weights.
- **Relation to other modules**: can be trained **independently** from Flow-Match / Wasserstein-GastroFlow; called during GastroAgent inference fusion to generate explanations and reports.

### Flow-Match Generator

- **Goal**: learn a “generative trajectory / transformation path” from query samples to a reference distribution, providing an interpretable evidence path for subsequent optimal transport cost computation.
- **Typical training data**: primarily endoscopy images (can be split by organ/site/lesion category into sub-domains) to learn stable trajectories.
- **Artifacts**: Flow-Match generator weights (used to generate trajectories / intermediate states).
- **Relation to other modules**: provides trajectories (or cost-evaluation paths along trajectories) to **Wasserstein-GastroFlow**; thus it is generally recommended to **train this module first** before training/evaluating the few-shot module.

### Wasserstein-GastroFlow

- **Goal**: under few-shot / long-tail settings, use the “optimal transport cost along a generative trajectory” as a similarity metric for robust geometric matching with interpretable evidence outputs.
- **Typical training/building setup**:
  - Few-shot support set (labeled) and query set (to be recognized)
  - Combine Flow-Match trajectories to compute transport costs / matching scores from each query to each candidate class support set
- **Artifacts**: few-shot module parameters (if any) and/or support-feature indices and cost-metric configs, plus cost statistics and visualizations for evaluation.
- **Relation to other modules**: **depends on the Flow-Match generator**; its outputs are an important “few-shot evidence” source for the GastroAgent fusion controller.

### Full Integration: GastroAgent

- **Goal**: adaptively fuse evidence from **GastroMLLM (multimodal medical model)** and **Wasserstein-GastroFlow (few-shot learning)** to better balance stability on common classes and coverage on long-tail classes.
- **Typical workflow**:
  - Load weights/configs for GastroMLLM and Wasserstein-GastroFlow (and the Flow-Match generator)
  - Calibrate **entropy/confidence-related fusion hyperparameters** on a validation set, or directly use our provided hyperparameters

> If you’d like me to make the “CLI examples / parameter names” here extremely specific (e.g., scripts, arguments, output directories for each module), I can continue based only on what appears in `README.md` (entry file names) and write examples in a way that **does not require reading other files**, using placeholder variables to avoid misleading commands.

## 📈 Evaluation

> It is recommended to keep all result figures under `assets/figures/`. The paths below are placeholders; replace them with files of the same names when you have the final figures.

### Result Figures

- **Doctor Datasets**

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
├── assets/                     # static assets (figures for docs, etc.)
│   └── figures/                # README result/workflow figures
├── conditional_flow_matcher/   # conditional flow matching
├── dataset/                    # data processing scripts
│   ├── eval_data/              # evaluation data
│   ├── xxx                     # other testing scripts
├── discriminator/              # scripts for training the stopping discriminator
├── GasAgenteent/               # agent trigger scripts
│   ├── Agent_pipeline_result/  # test outputs
├── model_utils/                # model helper functions
├── my_models/                  # Flow-Match model architectures
├── train_clip/                 # medical visual encoder
├── train_flow_by_vae/          # train the Flow-Match generator
├── train_vae/                  # latent-space encoder/decoder
├── utils/                      # helper utilities
├── VLM-R1/                     # MLLM training framework
├── requirements.txt            # dependency list
├── wass_flow_match_duodenum/   # Wasserstein-GastroFlow
└── README.md                   # readme
```

## 🖼️ Visualization

> This section is currently a placeholder in `README.md`. You can move the “Result Figures” above here, or add additional evidence visualizations.

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
- Built with [PyTorch](https://pytorch.org/) and [MMDetection](https://github.com/open-mmlab/mmdetection)
- Thanks to the open-source community

---

## 📜 License

This project is released under the [MIT License](LICENSE).

**Disclaimer**: This project is for research use only and must not be used for clinical diagnosis. Any medical decision should be made by licensed professionals.

---

## 📮 Contact

- **Issue tracker**: [GitHub Issues](https://github.com/yourusername/GastroAgent/issues)
- **Email**: your.email@example.com
- **Homepage**: [Project homepage](https://yourproject.github.io)

---

## 🌟 Star History

If this project helps you, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/GastroAgent&type=Date)](https://star-history.com/#yourusername/GastroAgent&Date)
