"""评测结果相关数据模型 — 信号增强清单架构"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 六级 status → 连续系数（用于 weighted_yes_ratio 计算）
STATUS_COEFFICIENTS: Dict[str, Optional[float]] = {
    "YES": 1.0,
    "MOSTLY_YES": 0.8,
    "PARTIAL": 0.5,
    "MOSTLY_NO": 0.2,
    "NO": 0.0,
    "NOT_APPLICABLE": None,  # 不计入比率
}

VALID_STATUSES = set(STATUS_COEFFICIENTS.keys())


# ============================================================
# 清单核查
# ============================================================

@dataclass
class CheckResult:
    """单条清单项的核查结果"""
    item_id: str                      # 清单项ID
    description: str                  # 核查描述
    source: str                       # "case" | "simulator" | "llm_supplement" | "pattern_mined"
    status: str                       # "YES"|"MOSTLY_YES"|"PARTIAL"|"MOSTLY_NO"|"NO"|"NOT_APPLICABLE"
    evidence: str = ""                # 原文引用（turn_N: "..."）
    signal_consistency: str = "无对应信号"  # "一致" | "矛盾" | "无对应信号"
    weight: float = 1.0               # 清单项权重
    attribution: str = ""             # 归因: "Case" | "Simulator" | "Model"
    reasoning: str = ""               # CoT 推理过程（差异化深度）


@dataclass
class DimensionChecklist:
    """单个维度的完整清单"""
    dimension: str
    items: List[CheckResult] = field(default_factory=list)

    @property
    def yes_count(self) -> int:
        return sum(1 for i in self.items if i.status == "YES")

    @property
    def applicable_count(self) -> int:
        return sum(1 for i in self.items if i.status != "NOT_APPLICABLE")

    @property
    def yes_ratio(self) -> float:
        if self.applicable_count == 0:
            return 1.0
        return self.yes_count / self.applicable_count

    @property
    def weighted_yes_ratio(self) -> float:
        """六级粒度加权得分占比:
        每条 item 贡献 = weight × STATUS_COEFFICIENTS[status]
        总值 / 最大可能值 = 连续分 (0-1)
        YES=1.0, MOSTLY_YES=0.8, PARTIAL=0.5, MOSTLY_NO=0.2, NO=0.0, NOT_APPLICABLE=不计入
        """
        applicable = [i for i in self.items if i.status != "NOT_APPLICABLE"]
        if not applicable:
            return 1.0
        total_weight = sum(i.weight for i in applicable)
        scored_weight = sum(
            i.weight * STATUS_COEFFICIENTS.get(i.status, 0.0)
            for i in applicable
        )
        return scored_weight / total_weight if total_weight > 0 else 1.0

    def source_ratio(self, source: str) -> float:
        """指定来源的正向占比（YES + MOSTLY_YES 视作正向）"""
        items = [i for i in self.items if i.source == source and i.status != "NOT_APPLICABLE"]
        if not items:
            return 1.0
        positive = sum(1 for i in items if i.status in ("YES", "MOSTLY_YES"))
        return positive / len(items)


# ============================================================
# 缺陷
# ============================================================

@dataclass
class Defect:
    """LLM 补充的清单未覆盖缺陷"""
    description: str
    severity: str                    # "关键" | "一般" | "轻微"
    turn: int = 0
    attribution: str = "Model"       # "Case" | "Simulator" | "Model"


# ============================================================
# 归因
# ============================================================

@dataclass
class AttributionItem:
    """根因分析项"""
    source: str                      # "case" | "simulator" | "model"
    category: str                    # 对应 Judge 名
    description: str
    confidence: float = 0.5          # 0-1 归因置信度
    evidence_chain: List[str] = field(default_factory=list)
    suggested_actions: List[str] = field(default_factory=list)

    @property
    def is_actionable(self) -> bool:
        return self.confidence >= 0.8


# ============================================================
# 元检查 & 交叉验证
# ============================================================

@dataclass
class MetaCheckAlert:
    """元检查告警（纯规则，零 LLM 成本）"""
    check_type: str                  # "logic" | "evidence" | "coverage" | "consistency"
    severity: str                    # "error" | "warning" | "info"
    description: str
    dimensions: List[str] = field(default_factory=list)


@dataclass
class CrossValidationAlert:
    """规则-LLM 交叉验证告警"""
    dimension: str
    rule_finding: str
    llm_finding: str
    description: str
    severity: str = "medium"        # "high" | "medium"


# ============================================================
# EvalConfidence
# ============================================================

@dataclass
class EvalConfidence:
    """评测可信度中枢"""
    # 总体
    overall: float = 0.65                        # 0-1
    level: str = "medium"                         # "high" | "medium" | "low" | "unreliable"

    # 清单-信号一致性
    checklist_signal_consistency: Dict[str, str] = field(default_factory=dict)  # 维度 → "一致"|"矛盾"
    signal_conflict_count: int = 0

    # 证据质量
    evidence_empty_ratio: float = 0.0
    avg_evidence_coverage: float = 0.0           # 0-3

    # Simulator 质量
    simulator_tier: str = "green"                 # green | yellow | red
    signal_weight_factor: float = 1.0             # 来自 tier

    # Judge 间一致性
    cross_judge_anomalies: List[str] = field(default_factory=list)
    cross_judge_anomaly_pairs: int = 0

    # 子维度
    sub_dimension_anomalies: List[str] = field(default_factory=list)

    # 各维度明细
    per_dimension: Dict[str, float] = field(default_factory=dict)

    # 归因
    attribution_confidence: float = 0.5

    # 解析
    parse_success: bool = True

    # V3 画像极端度（audited_vector 消费）
    extreme_profile_dims: List[str] = field(default_factory=list)
    extreme_profile_flag: bool = False
    extreme_profile_source: str = "none"

    # 置信度理由（人类可读）
    confidence_reasoning: str = ""
    per_dimension_reasoning: Dict[str, str] = field(default_factory=dict)

    # 人工复核标记（由 orchestrator 综合 cross_alerts + meta_alerts + level 计算）
    needs_human_review: bool = False

    @property
    def is_reliable(self) -> bool:
        return (self.overall >= 0.65
                and self.level in ("high", "medium")
                and self.simulator_tier != "red"
                and self.signal_conflict_count < 3)


# ============================================================
# OptimizationFeed
# ============================================================

@dataclass
class OptimizationFeed:
    """归因对接优化引擎的标准化输出"""
    batch_id: str = ""
    generated_at: str = ""
    summary: Dict[str, Any] = field(default_factory=dict)
    attributions: List[AttributionItem] = field(default_factory=list)
    drift_alerts: List[str] = field(default_factory=list)
    eval_confidence: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# EvalResult
# ============================================================

@dataclass
class EvalResult:
    """一份完整的清单核查评测结果"""
    conversation_id: str = ""
    case_id: int = 0

    # 清单核查
    dimension_checklists: Dict[str, DimensionChecklist] = field(default_factory=dict)
    additional_defects: List[Defect] = field(default_factory=list)

    # 评级
    ratings: Dict[str, str] = field(default_factory=dict)       # 维度 → 五级评级
    indicative_scores: Dict[str, float] = field(default_factory=dict)  # 维度 → 概要分数
    total_indicative_score: float = 0.0
    total_score_100: int = 0          # 百分制总分（整数），max=100，仅作参考

    # 合规
    surface_compliance_flags: List[str] = field(default_factory=list)
    rule_check_issues: List[str] = field(default_factory=list)

    # 归因
    attributions: List[AttributionItem] = field(default_factory=list)

    # 可信度
    confidence: Optional[EvalConfidence] = None

    # 输出
    summary: str = ""
    improvement_suggestions: List[str] = field(default_factory=list)
    optimization_feed: Optional[OptimizationFeed] = None

    # 元检查 & 交叉验证
    meta_check_alerts: List[MetaCheckAlert] = field(default_factory=list)
    cross_validation_alerts: List[CrossValidationAlert] = field(default_factory=list)

    # CONSTRAINT 分流统计（成本追踪）
    tier1_constraint_count: int = 0
    llm_constraint_count: int = 0
