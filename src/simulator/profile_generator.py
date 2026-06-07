"""画像生成管线: CO-STAR + Contrastive + 自检回路"""
from typing import List, Optional, Tuple

from src.llm.client import LLMClient
from src.llm.model_manager import get_generator_client
from src.llm.prompts import (
    PROFILE_GENERATION_PROMPT,
    SELF_CHECK_PROMPT,
    SELF_CHECK_CORRECTION_PROMPT,
)
from src.simulator.profile_params import (
    DIMENSIONS,
    DIM_NAME_TO_INDEX,
    ANCHORS,
    translate_vector_to_anchor,
)
from src.simulator.profiles import UserProfile, build_profile_from_vector
from src.utils.helpers import _safe


def compute_self_check_thresholds(complexity_score: float) -> Tuple[float, float, int]:
    """根据 case 复杂度动态计算自检阈值

    复杂度越高 → 阈值越宽松（高维向量更难在自然语言中精确编码）。
    Returns: (max_dev_limit, d_sv_limit, max_retries)
    """
    if complexity_score >= 9:
        return (0.45, 0.30, 5)
    elif complexity_score >= 7:
        return (0.40, 0.28, 4)
    elif complexity_score >= 5:
        return (0.38, 0.28, 5)  # max_dev 0.35→0.38，15维向量每维偏差仅增0.002
    else:
        return (0.30, 0.20, 3)


class ProfileGenerator:
    """从 15D 参数向量生成自然连贯的用户画像文本"""

    def __init__(self, client: Optional[LLMClient] = None,
                 max_dev_limit: float = 0.35, d_sv_limit: float = 0.25,
                 max_retries: int = 3):
        self.client = client or get_generator_client(temperature=0.7)
        self.max_dev_limit = max_dev_limit
        self.d_sv_limit = d_sv_limit
        self.max_retries = max_retries

    def generate_persona_text(self, vector: List[float]) -> str:
        """Step 1: 使用 CO-STAR + Contrastive 框架生成画像文本"""
        anchor_text = translate_vector_to_anchor(vector)
        contrastive = self._build_contrastive_examples(vector)
        numerical = self._build_numerical_summary(vector)
        user_message = PROFILE_GENERATION_PROMPT.format(
            anchor_description=anchor_text,
            contrastive_examples=contrastive,
            numerical_summary=numerical,
        )
        return self.client.chat(
            system_prompt="你是一个专业的用户画像撰写者。",
            user_message=user_message,
        )

    def self_check(
        self, persona_text: str, sampled_vector: List[float]
    ) -> Tuple[List[float], bool]:
        """Step 2: LLM 独立读取画像文本，对 15 维重新打分

        Returns: (verified_vector, passed)
          使用实例阈值判定（可通过 compute_self_check_thresholds 按复杂度动态设置）。
        """
        rubric = self._build_scoring_rubric()
        user_message = SELF_CHECK_PROMPT.format(
            persona_text=_safe(persona_text),
            scoring_rubric=rubric,
        )

        dim_names = [dim.name for dim in DIMENSIONS]
        schema = {name: 0.0 for name in dim_names}

        result = self.client.chat_structured(
            system_prompt="你是一个画像质量验证器。请输出JSON。",
            user_message=user_message,
            output_schema=schema,
        )
        if result is None:
            return [0.5] * len(dim_names), False, 1.0  # 解析失败→默认不通过自检

        verified = [max(0.0, min(1.0, float(result.get(name, 0.5)))) for name in dim_names]
        max_dev = max(
            abs(sampled_vector[i] - verified[i]) for i in range(len(sampled_vector))
        )
        # 欧氏距离（归一化到 0-1）
        d_sv = (sum((sampled_vector[i] - verified[i]) ** 2 for i in range(15)) / 15) ** 0.5
        passed = max_dev < self.max_dev_limit and d_sv < self.d_sv_limit
        return verified, passed, d_sv

    def _build_scoring_rubric(self) -> str:
        """构建自检评分基准，取每个维度的 0.0 / 0.25 / 0.5 / 0.75 / 1.0 锚点描述"""
        lines = []
        for dim in DIMENSIONS:
            anchors = ANCHORS.get(dim.name, {})
            lines.append(f"**{dim.display_name} ({dim.name})**:")
            for q, label in [(0.0, "极低"), (0.25, "偏低"), (0.5, "中等"), (0.75, "偏高"), (1.0, "极高")]:
                desc = anchors.get(q, "")
                lines.append(f"  - {label}(~{q}): {desc}")
            lines.append("")
        return "\n".join(lines)

    def generate_with_retry(
        self, vector: List[float], max_retries: Optional[int] = None
    ) -> UserProfile:
        """完整管线: 生成 + 自检 + 锚点描述修正重试"""
        if max_retries is None:
            max_retries = self.max_retries
        persona_text = self.generate_persona_text(vector)
        passed = False

        last_d_sv = None
        for attempt in range(max_retries + 1):
            verified, passed, d_sv = self.self_check(persona_text, vector)
            last_d_sv = d_sv

            if passed:
                p = build_profile_from_vector(
                    vector, persona_text=persona_text, verified_vector=verified,
                )
                p._self_check_passed = True
                p.self_check_d_sv = d_sv
                return p

            if attempt < max_retries:
                max_dev_idx = max(
                    range(len(vector)),
                    key=lambda i: abs(vector[i] - verified[i]),
                )
                dim_def = DIMENSIONS[max_dev_idx]

                current_anchor = _find_nearest_anchor(
                    dim_def.name, verified[max_dev_idx]
                )
                target_anchor = _find_nearest_anchor(
                    dim_def.name, vector[max_dev_idx]
                )

                correction_prompt = SELF_CHECK_CORRECTION_PROMPT.format(
                    persona_text=_safe(persona_text),
                    dimension_display_name=dim_def.display_name,
                    current_anchor=current_anchor,
                    target_anchor=target_anchor,
                )
                persona_text = self.client.chat(
                    system_prompt="你是一个用户画像撰写者。",
                    user_message=correction_prompt,
                )

        # 超过最大重试数
        p = build_profile_from_vector(
            vector, persona_text=persona_text, verified_vector=verified,
        )
        p._self_check_passed = False
        p.self_check_d_sv = last_d_sv
        return p

    def batch_generate(
        self, vectors: List[List[float]], verbose: bool = True
    ) -> List[UserProfile]:
        """批量生成画像"""
        profiles = []
        total = len(vectors)
        for i, vector in enumerate(vectors):
            if verbose:
                print(f"[画像生成] {i + 1}/{total} ...", end=" ", flush=True)
            profile = self.generate_with_retry(vector)
            profiles.append(profile)
            if verbose:
                status = "PASS" if getattr(profile, '_self_check_passed', False) else "MAX_RETRY"
                print(status)
        return profiles

    def _build_contrastive_examples(self, vector: List[float]) -> str:
        """构建排除法对比示例"""
        parts = []
        for i, val in enumerate(vector):
            if val <= 0.15 or val >= 0.85:
                dim = DIMENSIONS[i]
                opposite_val = 1.0 if val <= 0.15 else 0.0
                opposite_anchor = ANCHORS[dim.name].get(opposite_val, "")
                short = (
                    opposite_anchor[:80] + "..."
                    if len(opposite_anchor) > 80
                    else opposite_anchor
                )
                parts.append(f"- {dim.display_name}方面，你不是这样的: {short}")

        if not parts:
            parts.append("- 你是一个有鲜明个性的人，不是一个'平均人'。请避免生成中性、模糊的描述。")
        return "\n".join(parts)

    def _build_numerical_summary(self, vector: List[float]) -> str:
        """构建 15D 数值摘要，帮助 LLM 校准行为强度"""
        lines = []
        for i, val in enumerate(vector):
            dim = DIMENSIONS[i]
            if val <= 0.15:
                label = "很低"
            elif val <= 0.35:
                label = "偏低"
            elif val <= 0.65:
                label = "中等"
            elif val <= 0.85:
                label = "偏高"
            else:
                label = "很高"
            lines.append(f"  {dim.display_name}({dim.name}) = {val:.2f} → {label}")
        return "\n".join(lines)


def _find_nearest_anchor(dim_name: str, value: float) -> str:
    """找到离给定值最近的分位锚点描述"""
    anchors = ANCHORS.get(dim_name, {})
    if not anchors:
        return ""
    quantiles = sorted(anchors.keys())
    nearest = min(quantiles, key=lambda q: abs(q - value))
    return anchors[nearest]
