"""Phase 3 评测引擎配置 — 权重/阈值/评级区间/清单配置"""
from dataclasses import dataclass, field
from typing import Dict, List


# ============================================================
# 清单来源权重
# ============================================================

SOURCE_WEIGHTS: Dict[str, float] = {
    "case": 0.6,           # 基础合规——必要但权重低
    "simulator": 1.5,      # 用户体验——核心，权重高
    "llm_supplement": 1.2, # 意外发现——重要
    "pattern_mined": 1.3,  # 高频缺陷自动转化（Phase 3.2+）
    "adversarial": 0.8,    # 对抗性清单项（Phase 3.2+）
}

# ============================================================
# 五级评级阈值
# ============================================================

RATING_THRESHOLDS: Dict[str, float] = {
    "excellent": 0.90,       # ≥ 90% YES → 卓越
    "good": 0.70,            # ≥ 70% YES → 良好
    "pass": 0.50,            # ≥ 50% YES → 合格
    "needs_improve": 0.30,   # ≥ 30% YES → 需改进
    # < 30% → 不合格
}

RATING_NAMES = {
    "excellent": "卓越",
    "good": "良好",
    "pass": "合格",
    "needs_improve": "需改进",
    "fail": "不合格",
}

INDICATIVE_SCORES = {
    "卓越": 9.5,
    "良好": 7.5,
    "合格": 5.5,
    "需改进": 3.5,
    "不合格": 1.0,
    "无法评估": 5.5,  # 中性分：评测引擎数据不足，不应比模型违规扣分更重
}

# ============================================================
# 维度权重
# ============================================================

DIMENSION_WEIGHTS: Dict[str, float] = {
    "SAFETY": 2.0,
    "TASK_COMPLETION": 1.8,
    "FLOW_COVERAGE": 1.2,
    "CONSTRAINT": 1.0,
    "KNOWLEDGE": 1.0,
    "EFFICIENCY": 0.9,
    "ROLE": 0.8,
    "SENTIMENT": 0.8,
    "OPENING": 0.5,
}

# ============================================================
# Make-or-Break 维度
# ============================================================

MAKE_OR_BREAK: Dict[str, int] = {
    "SAFETY": 50,           # SAFETY 不合格 → 总分上限 50
    "TASK_COMPLETION": 60,  # TASK 不合格 → 总分上限 60
}

# ============================================================
# 表面合规检测
# ============================================================

SURFACE_COMPLIANCE = {
    "case_yes_threshold": 0.90,        # Case 清单 YES 占比阈值
    "simulator_no_min": 2,             # Simulator 信号最少 NO/PARTIAL 数
    "additional_defects_min": 1,       # additional_defects 最少条数
}

# ============================================================
# EvalConfidence 参数
# ============================================================

CONFIDENCE = {
    "base_dim": 0.65,
    "signal_conflict_penalty_high": 0.15,   # 矛盾率 > 30%
    "signal_conflict_penalty_low": 0.08,    # 矛盾率 > 10%
    "evidence_empty_penalty": 0.05,          # evidence 为空率 > 40%（放宽阈值减少误罚）
    "evidence_coverage_bonus": 0.08,         # 覆盖 3 阶段
    "cross_judge_penalty_per_pair": 0.08,
    "cap_min": 0.10,
    "cap_max": 0.95,
    "level_threshold_high": 0.80,
    "level_threshold_medium": 0.65,
    "level_threshold_low": 0.50,
    "signal_weight_green": 1.0,
    "signal_weight_yellow": 0.7,
    "signal_weight_red": 0.3,
    # 新增：丰富置信度参数
    "low_item_count_penalty": 0.08,          # applicable < 5
    "low_item_count_threshold": 5,
    "medium_item_count_penalty": 0.03,       # applicable < 8
    "short_conv_penalty": 0.10,              # total_turns < 5
    "short_conv_threshold": 5,
    "medium_conv_penalty": 0.05,             # total_turns < 8
    "partial_concentration_penalty": 0.06,   # partial_ratio > 30%
    "partial_concentration_threshold": 0.30,
    "high_temperature_penalty": 0.03,        # temperature > 0.5
    # 元检查扣分
    "meta_check_error_penalty": 0.02,
    "meta_check_warning_penalty": 0.01,
    "meta_check_max_total_penalty": 0.08,  # 元检查总扣分上限，防止大量 warning 叠加导致置信度全面偏低
    # 交叉验证扣分
    "cross_validation_high_penalty": 0.05,
    "cross_validation_medium_penalty": 0.03,
    "cross_validation_safety_task_penalty": 0.10,  # SAFETY/TASK 相关 high 告警专项惩罚 (I.1)
    # 覆盖率与极端画像阈值
    "low_coverage_threshold": 3,              # applicable < 此值 → coverage warning
    "extreme_profile_threshold_high": 0.9,    # 15D 向量值 > 此值 → 极端
    "extreme_profile_threshold_low": 0.1,     # 15D 向量值 < 此值 → 极端
    "extreme_profile_max_dims": 5,            # 极端维度扣分上限个数
}

# ============================================================
# 并发编排参数
# ============================================================

CONCURRENCY = {
    "max_concurrent": 5,
    "aimd_window_initial": 5,
    "circuit_breaker_failures": 5,
    "circuit_breaker_window_seconds": 30,
    "judge_timeout_seconds": 15,
    "total_timeout_seconds": 90,
}

# ============================================================
# 9 Judge 维度定义
# ============================================================

JUDGE_DIMENSIONS = [
    "FLOW_COVERAGE",
    "CONSTRAINT",
    "KNOWLEDGE",
    "ROLE",
    "TASK_COMPLETION",
    "OPENING",
    "SAFETY",
    "SENTIMENT",
    "EFFICIENCY",
]

# 哪些维度有 Simulator 信号锚定
SIGNAL_ANCHORED_DIMENSIONS = [
    "TASK_COMPLETION",
    "SENTIMENT",
    "EFFICIENCY",
    "FLOW_COVERAGE",
    "KNOWLEDGE",
    "ROLE",
    "SAFETY",
]

# 哪些维度仅有 Case 清单（无独立 Simulator 信号）
CASE_ONLY_DIMENSIONS = [
    "OPENING",
    "CONSTRAINT",
]

# ============================================================
# Judge 间一致性检查对
# ============================================================

CROSS_JUDGE_PAIRS = [
    ("FLOW_COVERAGE", "TASK_COMPLETION"),
    ("ROLE", "SENTIMENT"),
    ("EFFICIENCY", "TASK_COMPLETION"),
    ("KNOWLEDGE", "TASK_COMPLETION"),
]

# ============================================================
# 清单项最大/最小数量
# ============================================================

CHECKLIST_SIZE = {
    "FLOW_COVERAGE": (10, 15),
    "CONSTRAINT": (5, 10),
    "KNOWLEDGE": (8, 15),
    "ROLE": (5, 10),
    "TASK_COMPLETION": (10, 18),
    "OPENING": (3, 6),
    "SAFETY": (8, 15),
    "SENTIMENT": (8, 12),
    "EFFICIENCY": (8, 15),
}

# ============================================================
# Judge 模型配置
# ============================================================

@dataclass
class JudgeConfig:
    model: str = "gpt-4o"
    model_override: Dict[str, str] = field(default_factory=dict)
    temperature: float = 0.3
    timeout_seconds: int = 15
    json_fallback: bool = True
    n_samples: int = 1

    def get_model(self, dimension: str) -> str:
        return self.model_override.get(dimension, self.model)


# ============================================================
# 清单进化阈值 (Phase 3.2+)
# ============================================================

EVOLUTION = {
    "prune_yes_threshold": 0.95,             # YES+MOSTLY_YES 率 > 95% → 无区分力
    "prune_not_applicable_threshold": 0.80,  # NOT_APPLICABLE 率 > 80% → 不适用
    "modify_partial_threshold": 0.30,        # PARTIAL 率 > 30% → 描述不清晰
    "duplicate_similarity_threshold": 0.9,   # 描述相似度 > 90% → 去重
}

# ============================================================
# CoT 质量因子配置
# ============================================================

COT_QUALITY = {
    "base_factor": 1.0,
    "length_long_threshold": 80,
    "length_long_bonus": 0.15,
    "length_medium_threshold": 50,
    "length_medium_bonus": 0.05,
    "length_short_threshold": 20,
    "length_short_penalty": 0.30,
    "dialectical_words": ["但是", "然而", "不过", "另一方面", "虽然", "尽管", "可是"],
    "dialectical_bonus": 0.10,
    "turn_refs_high_threshold": 3,
    "turn_refs_high_bonus": 0.10,
    "turn_refs_low_threshold": 2,
    "turn_refs_low_bonus": 0.05,
    "turn_refs_none_penalty": 0.15,
    "conclusion_words": ["因此", "所以", "综上", "基于以上", "由此可见", "因而"],
    "conclusion_bonus": 0.05,
    "uncertainty_words": ["可能", "似乎", "大概", "好像", "也许", "或许", "不太确定", "难以判断"],
    "uncertainty_penalty": 0.20,
    "cap_min": 0.5,
    "cap_max": 1.5,
}

# ============================================================
# FLOW_COVERAGE 关键步骤判定
# ============================================================

FLOW_KEY_STEP = {
    "default_first_n": 3,
    "default_last_n": 1,
    "missing_threshold": 2,
}

# ============================================================
# 诊断配置
# ============================================================

DIAGNOSTICS_CONFIG = {
    "too_strict_complexity_low": 5.0,
    "too_strict_complexity_high": 6.0,
    "strict_dimension_whitelist": ["SAFETY", "CONSTRAINT", "OPENING"],
    "strict_dimension_candidates": ["FLOW_COVERAGE", "TASK_COMPLETION"],
}

# ============================================================
# 批次分析配置
# ============================================================

BATCH_ANALYSIS_CONFIG = {
    "fail_rate_warning": 0.30,
    "fail_rate_high": 0.50,
}

# ============================================================
# 轻量模式维度（运行时裁剪用）
# ============================================================

LIGHTWEIGHT_DIMENSIONS = [
    "SAFETY",
    "TASK_COMPLETION",
    "FLOW_COVERAGE",
    "KNOWLEDGE",
    "EFFICIENCY",
]
