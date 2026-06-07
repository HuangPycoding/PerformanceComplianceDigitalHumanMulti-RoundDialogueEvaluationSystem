"""用户画像 — UserProfile 数据类 + 参数化画像工厂函数"""
from dataclasses import dataclass, field
from typing import List, Optional

from src.llm.prompts import (
    ADVERSARIAL_PROBE, ADVERSARIAL_INJECTION, ADVERSARIAL_CONTRADICTION,
    ADVERSARIAL_AUTHORITY, ADVERSARIAL_EMOTION,
)
from src.utils.helpers import stable_hash

_ADVERSARIAL_MAP = {
    "probe": ADVERSARIAL_PROBE,
    "injection": ADVERSARIAL_INJECTION,
    "contradiction": ADVERSARIAL_CONTRADICTION,
    "authority": ADVERSARIAL_AUTHORITY,
    "emotion": ADVERSARIAL_EMOTION,
}


@dataclass
class UserProfile:
    """用户画像 — 支持固定画像和参数化画像两种模式"""
    type: str = ""                                   # 画像类型标签
    description: str = ""                            # 画像 prompt 全文
    adversarial_mode: Optional[str] = None
    adversarial_instruction: str = ""
    # 参数化画像字段
    sampled_vector: Optional[List[float]] = None     # 15D 原始采样值 S
    verified_vector: Optional[List[float]] = None    # 15D 自检验证值 V
    persona_text: str = ""                           # LLM 生成的画像文本
    adversarial_strategy: List[str] = field(default_factory=list)  # 自动挂钩的对抗策略
    anchor_description: str = ""                     # 锚点翻译文本
    self_check_d_sv: Optional[float] = None          # 自检 S-V 归一化欧氏距离

    @property
    def is_parameterized(self) -> bool:
        """是否由参数向量生成的画像"""
        return self.sampled_vector is not None

    @property
    def effective_description(self) -> str:
        """返回最佳可用的画像描述"""
        return self.persona_text or self.description

    @property
    def label(self) -> str:
        if self.is_parameterized:
            h = stable_hash(self.sampled_vector or [])
            return f"param_{h}"
        if self.adversarial_mode:
            return f"{self.type}+{self.adversarial_mode}"
        return self.type


def build_profile_from_vector(
    vector: List[float],
    persona_text: str = "",
    verified_vector: Optional[List[float]] = None,
) -> UserProfile:
    """工厂函数：从 15D 参数向量创建 UserProfile"""
    from src.simulator.profile_params import (
        get_adversarial_strategies,
        translate_vector_to_anchor,
    )
    strategies = get_adversarial_strategies(vector)
    adv_instruction = _build_adversarial_instruction_for_strategies(strategies)

    return UserProfile(
        type="parameterized",
        description="",
        adversarial_mode=None,
        adversarial_instruction=adv_instruction,
        sampled_vector=list(vector),
        verified_vector=list(verified_vector) if verified_vector else None,
        persona_text=persona_text,
        adversarial_strategy=strategies,
        anchor_description=translate_vector_to_anchor(vector),
    )


def build_adversarial_instruction_for_vector(vector: List[float]) -> str:
    """从参数向量生成对抗指令（自动挂钩）"""
    from src.simulator.profile_params import get_adversarial_strategies
    return _build_adversarial_instruction_for_strategies(
        get_adversarial_strategies(vector)
    )


def _build_adversarial_instruction_for_strategies(strategies: List[str]) -> str:
    """将对抗策略列表拼装为指令文本"""
    if not strategies:
        return ""
    parts = []
    for s in strategies:
        text = _ADVERSARIAL_MAP.get(s, "")
        if text:
            parts.append(text)
    return "\n\n".join(parts)
