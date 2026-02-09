# HALT幻觉检测配置文件
# 在step1_model_inference.py中使用这些配置

# ===== 核心HALT配置 =====
HALT_CONFIG = {
    # 是否启用HALT检测
    "enabled": True,

    # 中间层位置（0.0-1.0，推荐0.5）
    # 0.3: 较浅层，捕获早期语义
    # 0.5: 中间层，平衡语义和推理（推荐）
    # 0.7: 较深层，接近最终输出
    "middle_layer_ratio": 0.5,

    # 探针模型配置
    "probe": {
        "hidden_dim": 256,           # 探针隐藏层维度
        "dropout": 0.1,              # Dropout率
        "model_path": None,          # 预训练探针路径（None=随机初始化）
    },

    # 风险阈值（0.0-1.0）
    # 较低值(0.3): 更保守，更多样本被标记为高风险
    # 中等值(0.5): 平衡（推荐）
    # 较高值(0.7): 更激进，只标记明显高风险样本
    "risk_threshold": 0.5,
}

# ===== 代理路由配置 =====
ROUTING_CONFIG = {
    # 是否启用代理路由
    "enabled": True,

    # 验证管道类型
    # "strong_model": 路由到更强大的模型
    # "rag": 使用检索增强生成
    # "cross_validation": 多模型交叉验证
    "verification_pipeline": "strong_model",

    # 强模型配置（当verification_pipeline="strong_model"时使用）
    "strong_model": {
        "model_id": "/path/to/stronger/model",
        "enabled": False,  # 是否实际调用（False则仅标记）
    },

    # RAG配置（当verification_pipeline="rag"时使用）
    "rag": {
        "retriever_path": "/path/to/retriever",
        "knowledge_base": "/path/to/knowledge/base",
        "enabled": False,
    },

    # 交叉验证配置（当verification_pipeline="cross_validation"时使用）
    "cross_validation": {
        "models": [
            "/path/to/model1",
            "/path/to/model2",
            "/path/to/model3",
        ],
        "voting_strategy": "majority",  # "majority" or "weighted"
        "enabled": False,
    },
}

# ===== 训练配置（用于train_halt_probe.py）=====
TRAINING_CONFIG = {
    # 数据路径
    "train_data_path": "/path/to/train_with_labels.json",
    "val_data_path": "/path/to/val_with_labels.json",

    # 训练参数
    "batch_size": 32,
    "learning_rate": 1e-4,
    "num_epochs": 10,
    "early_stopping_patience": 3,

    # 模型保存
    "save_path": "/path/to/halt_probe.pth",
    "save_best_only": True,
    "metric": "f1",  # "f1", "accuracy", "auc"
}

# ===== 实验配置 =====
EXPERIMENT_CONFIG = {
    # 多层实验：测试不同中间层位置
    "multi_layer_experiment": {
        "enabled": False,
        "layer_ratios": [0.3, 0.4, 0.5, 0.6, 0.7],
    },

    # 阈值扫描：找到最优风险阈值
    "threshold_sweep": {
        "enabled": False,
        "thresholds": [0.3, 0.4, 0.5, 0.6, 0.7],
    },

    # 消融实验：对比不同方法
    "ablation_study": {
        "enabled": False,
        "methods": [
            "halt_middle_layer",      # HALT中间层（本方法）
            "halt_last_layer",        # HALT最后一层
            "entropy_baseline",       # 熵基线
            "confidence_baseline",    # 置信度基线
        ],
    },
}

# ===== 日志和调试 =====
DEBUG_CONFIG = {
    # 是否打印详细日志
    "verbose": True,

    # 随机打印样本的概率
    "sample_print_prob": 0.05,

    # 是否保存中间结果
    "save_intermediate": True,

    # 是否可视化风险分布
    "visualize_risk_distribution": False,
}

# ===== 性能优化 =====
PERFORMANCE_CONFIG = {
    # 是否使用混合精度
    "use_fp16": True,

    # 是否使用梯度检查点（节省内存）
    "gradient_checkpointing": False,

    # 批处理大小（推理）
    "inference_batch_size": 1,

    # 并行worker数量
    "num_workers": 4,
}

# ===== 使用示例 =====
"""
在step1_model_inference.py中使用：

from halt_config import HALT_CONFIG, ROUTING_CONFIG

HALT_ENABLED = HALT_CONFIG["enabled"]
HALT_MIDDLE_LAYER_RATIO = HALT_CONFIG["middle_layer_ratio"]
HALT_PROBE_HIDDEN_DIM = HALT_CONFIG["probe"]["hidden_dim"]
HALT_RISK_THRESHOLD = HALT_CONFIG["risk_threshold"]
HALT_PROBE_MODEL_PATH = HALT_CONFIG["probe"]["model_path"]

ENABLE_AGENTIC_ROUTING = ROUTING_CONFIG["enabled"]
VERIFICATION_PIPELINE = ROUTING_CONFIG["verification_pipeline"]
"""
