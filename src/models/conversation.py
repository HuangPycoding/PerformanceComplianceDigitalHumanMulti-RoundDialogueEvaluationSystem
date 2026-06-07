"""对话记录相关数据模型"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from src.models.evaluation import EvalConfidence, EvalResult


@dataclass
class Turn:
    """一轮对话"""
    turn_number: int                  # 轮次序号
    speaker: Literal["system", "user"]  # 说话者
    content: str                      # 说话内容
    timestamp: float = 0.0            # 相对对话开始的时间戳（秒）
    # 参数化模拟器标签（memory/thought/state/emotion_curve/risk_flag/model_behavior/conversation_quality）
    parsed_tags: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conversation:
    """一场完整的对话记录"""
    id: str                           # 对话ID，如 "case5_profile_cooperative_t1"
    case_id: int                      # 对应的 Case ID
    user_profile: str = ""            # 用户画像标签
    turns: List[Turn] = field(default_factory=list)
    status: Literal["用户挂断", "超时", "异常中断"] = "用户挂断"
    total_turns: int = 0
    duration_seconds: float = 0.0     # 对话总时长
    # 15D 参数向量
    sampled_vector: Optional[List[float]] = None    # 原始采样值 S
    verified_vector: Optional[List[float]] = None   # 自检验证值 V
    audited_vector: Optional[List[float]] = None    # 行为审计值 A
    # 元数据
    profile_label: str = ""                         # 可读标签
    adversarial_strategies: List[str] = field(default_factory=list)
    complexity_score: float = 0.0                   # Case 复杂度评分
    consistency: Dict[str, Any] = field(default_factory=dict)   # {d_sv, d_va, d_sa, tier, method}
    branch_coverage: Dict[str, Any] = field(default_factory=dict)  # {expected, triggered, untriggered}
    model_breakdown_count: int = 0

    # Phase 3 评测结果
    eval_result: Optional[EvalResult] = None
    eval_confidence: Optional[EvalConfidence] = None
    hangup_context: Dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """将对话格式化为纯文本（客服-用户交替）"""
        lines = []
        for turn in self.turns:
            label = "客服" if turn.speaker == "system" else "用户"
            lines.append(f"T{turn.turn_number}: {label}: {turn.content}")
        return "\n".join(lines)
