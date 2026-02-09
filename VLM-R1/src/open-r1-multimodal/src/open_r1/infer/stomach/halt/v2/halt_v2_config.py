"""
================================================================================
HALT V2 幻觉检测系统 - 配置文件
================================================================================
重新设计的HALT系统，解决V1版本的核心问题：
1. 特征区分度差 -> 多层特征融合 + 注意力机制
2. 零真阴性问题 -> 动态阈值校准 + 类别平衡
3. 风险分数聚集 -> 对比学习 + 特征增强
4. 单一问题类型 -> 问题类型感知的检测
================================================================================
"""

# ===== 核心HALT V2配置 =====
HALT_V2_CONFIG = {
    # 是否启用HALT V2检测
    "enabled": True,

    # ===== 特征提取策略 =====
    "feature_extraction": {
        # 多层特征融合
        "use_multi_layer": True,
        "layer_positions": [0.25, 0.5, 0.75],  # 使用浅层、中层、深层

        # 特征类型
        "feature_types": {
            "hidden_states": True,      # 隐藏状态
            "attention_weights": True,  # 注意力权重
            "token_entropy": True,      # token级别的熵
            "layer_variance": True,     # 层间方差
        },

        # 特征增强
        "feature_enhancement": {
            "use_contrastive": True,    # 对比学习特征
            "use_uncertainty": True,    # 不确定性特征
            "use_consistency": True,    # 一致性特征（多次采样）
        },
    },

    # ===== 探针模型配置 =====
    "probe_model": {
        "architecture": "attention_mlp",  # "simple_mlp", "attention_mlp", "transformer"

        # 注意力MLP配置
        "attention_mlp": {
            "hidden_dims": [512, 256, 128],
            "num_attention_heads": 4,
            "dropout": 0.2,
            "use_layer_norm": True,
            "use_residual": True,
        },

        # 简单MLP配置（备用）
        "simple_mlp": {
            "hidden_dims": [256, 128],
            "dropout": 0.1,
        },

        # 预训练模型路径
        "pretrained_path": None,
    },

    # ===== 动态阈值校准 =====
    "dynamic_threshold": {
        "enabled": True,
        "method": "adaptive",  # "fixed", "adaptive", "per_question_type"

        # 自适应阈值
        "adaptive": {
            "initial_threshold": 0.5,
            # "calibration_samples": 100,  # 用于校准的样本数
            "target_precision": 0.85,    # 目标精确率
            "target_recall": 0.90,       # 目标召回率
            "update_frequency": 50,      # 每N个样本更新一次
        },

        # 按问题类型的阈值
        "per_question_type": {
            "Disease Diagnosis": 0.45,
            "Modality Recognition": 0.55,
            "Anatomy Identification": 0.50,
            "default": 0.50,
        },
    },

    # ===== 训练策略 =====
    "training": {
        # 类别平衡
        "class_balance": {
            "method": "focal_loss",  # "weighted", "focal_loss", "oversample"
            "focal_loss_gamma": 2.0,
            "focal_loss_alpha": 0.25,
        },

        # 对比学习
        "contrastive_learning": {
            "enabled": True,
            "temperature": 0.07,
            "negative_samples": 5,
        },

        # 数据增强
        "data_augmentation": {
            "enabled": True,
            "methods": ["dropout", "noise", "mixup"],
            "dropout_rate": 0.1,
            "noise_std": 0.01,
            "mixup_alpha": 0.2,
        },

        # 正则化
        "regularization": {
            "l2_weight": 1e-4,
            "gradient_clip": 1.0,
        },
    },
}

# ===== 集成学习配置 =====
ENSEMBLE_CONFIG = {
    "enabled": True,
    "method": "stacking",  # "voting", "stacking", "boosting"

    # 基础检测器
    "base_detectors": [
        {
            "name": "halt_v2_shallow",
            "layer_positions": [0.25],
            "weight": 0.2,
        },
        {
            "name": "halt_v2_middle",
            "layer_positions": [0.5],
            "weight": 0.4,
        },
        {
            "name": "halt_v2_deep",
            "layer_positions": [0.75],
            "weight": 0.3,
        },
        {
            "name": "trigger_final",
            "type": "baseline",
            "weight": 0.1,
        },
    ],

    # Stacking元学习器
    "meta_learner": {
        "model": "logistic_regression",  # "logistic_regression", "xgboost", "mlp"
        "use_cross_validation": True,
        "cv_folds": 5,
    },
}

# ===== 评估配置 =====
EVALUATION_CONFIG = {
    # 评估指标
    "metrics": [
        "accuracy", "precision", "recall", "f1",
        "auc_roc", "auc_pr", "confusion_matrix"
    ],

    # 阈值扫描
    "threshold_sweep": {
        "enabled": True,
        "range": [0.1, 0.9],
        "step": 0.05,
    },

    # 按问题类型评估
    "per_question_type": True,

    # 错误分析
    "error_analysis": {
        "enabled": True,
        "save_misclassified": True,
        "analyze_risk_distribution": True,
    },
}

# ===== 可视化配置 =====
VISUALIZATION_CONFIG = {
    "enabled": True,

    # 图表类型
    "plots": {
        "risk_distribution": True,      # 风险分数分布
        "roc_curve": True,              # ROC曲线
        "pr_curve": True,               # PR曲线
        "confusion_matrix": True,       # 混淆矩阵
        "feature_importance": True,     # 特征重要性
        "threshold_analysis": True,     # 阈值分析
        "calibration_curve": True,      # 校准曲线
    },

    # 输出格式
    "output_format": ["png", "pdf"],
    "dpi": 300,
    "style": "seaborn",
}

# ===== 实验配置 =====
EXPERIMENT_CONFIG = {
    # 消融实验
    "ablation_study": {
        "enabled": True,
        "experiments": [
            {"name": "baseline", "features": ["hidden_states"]},
            {"name": "+attention", "features": ["hidden_states", "attention_weights"]},
            {"name": "+entropy", "features": ["hidden_states", "attention_weights", "token_entropy"]},
            {"name": "full", "features": ["hidden_states", "attention_weights", "token_entropy", "layer_variance"]},
        ],
    },

    # 超参数搜索
    "hyperparameter_search": {
        "enabled": False,
        "method": "optuna",  # "grid", "random", "optuna"
        "n_trials": 100,
        "search_space": {
            "learning_rate": [1e-5, 1e-3],
            "hidden_dim": [128, 256, 512],
            "dropout": [0.1, 0.3],
            "focal_loss_gamma": [1.0, 3.0],
        },
    },
}

# ===== 性能优化配置 =====
PERFORMANCE_CONFIG = {
    "use_fp16": True,
    "gradient_checkpointing": False,
    "batch_size": 32,
    "num_workers": 4,
    "pin_memory": True,
    "prefetch_factor": 2,
}

# ===== 日志配置 =====
LOGGING_CONFIG = {
    "level": "INFO",  # "DEBUG", "INFO", "WARNING", "ERROR"
    "save_to_file": True,
    "log_dir": "./logs",
    "verbose": True,
}

# ===== 路径配置 =====
PATH_CONFIG = {
    "data_dir": "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/data/Eval/halt/Llava-Qwen2-7B-tune-med-v0-mmtag-mul-Latest-all-RL-2500/merge",
    "model_dir": "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/stomach/halt/models",
    "output_dir": "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/stomach/halt/v2/outputs",
    "checkpoint_dir": "/mnt/inaisfs/data/home/tansy_criait/VLM-R1/src/open-r1-multimodal/src/open_r1/infer/stomach/halt/v2/checkpoints",
}

# ===== 使用示例 =====
"""
from halt_v2_config import HALT_V2_CONFIG, ENSEMBLE_CONFIG

# 启用HALT V2
if HALT_V2_CONFIG["enabled"]:
    feature_config = HALT_V2_CONFIG["feature_extraction"]
    probe_config = HALT_V2_CONFIG["probe_model"]

    # 初始化特征提取器
    feature_extractor = MultiLayerFeatureExtractor(
        layer_positions=feature_config["layer_positions"],
        feature_types=feature_config["feature_types"]
    )

    # 初始化探针模型
    probe_model = AttentionMLPProbe(
        **probe_config["attention_mlp"]
    )
"""
