"""Phase 2 审计: Path A (基于 state 标签，零 LLM 成本) + Path B (LLM 行为审计)"""
import math
from typing import Any, Dict, List, Optional, Tuple

from src.llm.client import LLMClient
from src.llm.model_manager import get_auditor_client
from src.llm.prompts import BEHAVIOR_AUDIT_PROMPT
from src.models.conversation import Conversation
from src.simulator.profile_params import (
    DIMENSIONS,
    DIM_NAME_TO_INDEX,
    ANCHORS,
    translate_vector_to_anchor,
)
from src.utils.helpers import _safe


def _estimate_from_state_trajectory(state_trajectory: List[Dict[str, Any]]) -> List[float]:
    """从 <state> 标签轨迹中启发式估算 15D 向量

    可估算维度 (~8/15):
      - initial_mood / mood_volatility: 情绪标量 + 跨轮变化率
      - agreeableness: 从 stance 推断（配合→高, 怀疑/敌对→低）
      - neuroticism: 从 emotion_intensity 均值推断
      - politeness: 从 stance 推断
      - assertiveness: 从 stance 推断（强硬→高, 顺从→低）
      - boundary_testing: 从 branch_triggered 非 none 比例推断
      - truth_consistency: 从 change_justified=false 比例推断

    不可靠维度（回退 None → 在归因时视为 0.5）:
      conscientiousness, extraversion, openness, patience, verbosity,
      information_verification, domain_knowledge
    """
    n = 15
    vec = [None] * n  # None = 不可估算，在归因时回退 0.5
    estimated_count = 0

    if not state_trajectory:
        return [0.5] * n

    # --- initial_mood (index 11) ---
    first = state_trajectory[0]
    mood_val = _mood_to_scalar(first)
    vec[DIM_NAME_TO_INDEX["initial_mood"]] = mood_val
    estimated_count += 1

    # --- mood_volatility (index 12) ---
    if len(state_trajectory) >= 2:
        mood_vals = [_mood_to_scalar(s) for s in state_trajectory]
        flips = sum(
            1 for i in range(1, len(mood_vals))
            if abs(mood_vals[i] - mood_vals[i - 1]) > 0.3
        )
        volatility = min(flips / max(len(mood_vals) - 1, 1), 1.0)
        vec[DIM_NAME_TO_INDEX["mood_volatility"]] = volatility
        estimated_count += 1
    else:
        vec[DIM_NAME_TO_INDEX["mood_volatility"]] = 0.3
        estimated_count += 1

    # --- neuroticism (index 2): 情绪强度均值 ---
    intensities = []
    for s in state_trajectory:
        ei = s.get("emotion_intensity") or s.get("情绪强度") or ""
        try:
            intensities.append(float(ei))
        except (ValueError, TypeError):
            pass
    if intensities:
        avg_intensity = sum(intensities) / len(intensities)
        vec[DIM_NAME_TO_INDEX["neuroticism"]] = avg_intensity
        estimated_count += 1

    # --- 从 stance 推断 agreeableness / politeness / assertiveness ---
    stances = [str(s.get("stance", "") or s.get("立场", "")) for s in state_trajectory]
    stance_text = " ".join(stances)

    # agreeableness (0): 配合→高, 怀疑/敌对/警惕→低
    agree_score = _stance_to_agreeableness(stance_text)
    if agree_score is not None:
        vec[DIM_NAME_TO_INDEX["agreeableness"]] = agree_score
        estimated_count += 1

    # politeness (7): 礼貌/友好→高, 粗鲁/命令→低
    politeness_score = _stance_to_politeness(stance_text)
    if politeness_score is not None:
        vec[DIM_NAME_TO_INDEX["politeness"]] = politeness_score
        estimated_count += 1

    # assertiveness (8): 强硬/推回→高, 顺从/接受→低
    assertiveness_score = _stance_to_assertiveness(stance_text)
    if assertiveness_score is not None:
        vec[DIM_NAME_TO_INDEX["assertiveness"]] = assertiveness_score
        estimated_count += 1

    # --- boundary_testing (14): branch_triggered 非 none 比例 ---
    branch_count = sum(
        1 for s in state_trajectory
        if str(s.get("branch_triggered", "none") or "none").lower() != "none"
    )
    bt = min(branch_count / max(len(state_trajectory), 1), 1.0)
    vec[DIM_NAME_TO_INDEX["boundary_testing"]] = bt
    estimated_count += 1

    # --- truth_consistency (15): change_justified=false 比例 ---
    justified_flags = []
    for s in state_trajectory:
        cj = str(s.get("change_justified", "true") or "true").lower()
        justified_flags.append(cj in ("true", "yes"))
    false_count = sum(1 for f in justified_flags if not f)
    tc = min(false_count / max(len(justified_flags), 1), 1.0)
    vec[DIM_NAME_TO_INDEX["truth_consistency"]] = tc
    estimated_count += 1

    # 回退不可估算维度为 0.5
    for i in range(n):
        if vec[i] is None:
            vec[i] = 0.5

    return vec


def _stance_to_agreeableness(stance_text: str):
    """从立场文本推断宜人性"""
    low = ["怀疑", "敌对", "警惕", "不信任", "刁难"]
    high = ["配合", "信任", "友好", "合作"]
    return _score_from_keywords(stance_text, low, high)


def _stance_to_politeness(stance_text: str):
    """从立场文本推断礼貌度"""
    low = ["粗鲁", "命令", "不耐烦", "催促"]
    high = ["礼貌", "客气", "友好", "尊重"]
    return _score_from_keywords(stance_text, low, high)


def _stance_to_assertiveness(stance_text: str):
    """从立场文本推断主见性"""
    low = ["顺从", "接受", "配合", "被动"]
    high = ["强硬", "推回", "坚持", "主导", "试探"]
    return _score_from_keywords(stance_text, low, high)


def _score_from_keywords(text: str, low_kw: list, high_kw: list):
    """基于关键词计分: low→0.2, high→0.8, 混合→0.5, 无信号→None"""
    low_count = sum(1 for kw in low_kw if kw in text)
    high_count = sum(1 for kw in high_kw if kw in text)
    if low_count == 0 and high_count == 0:
        return None
    if low_count > 0 and high_count == 0:
        return 0.2
    if high_count > 0 and low_count == 0:
        return 0.8
    return 0.5


# 情绪关键词→标量映射表，按优先级排列（长匹配优先，避免"非常负面"被"负面"误匹配）
_MOOD_KEYWORDS = [
    # (keywords_tuple, score) — 任一关键词子串命中即匹配
    (("非常正面", "very positive", "very_positive"), 0.95),
    (("非常负面", "very negative", "very_negative"), 0.05),
    (("满意", "高兴", "开心", "愉快", "positive", "正面", "积极", "good"), 0.85),
    (("期待", "好奇", "curious", "interested"), 0.65),
    (("平静", "平淡", "中性", "一般", "正常", "普通", "neutral", "calm"), 0.5),
    (("冷淡", "冷漠", "不信任", "怀疑", "cold", "suspicious"), 0.25),
    (("不满", "生气", "烦躁", "不耐烦", "失望", "沮丧", "negative", "负面", "消极", "angry", "bad", "frustrated", "impatient"), 0.15),
]


def _mood_to_scalar(state: Dict[str, Any]) -> float:
    """从 state dict 中提取情绪标量 (0=非常负面, 1=非常正面)

    使用包容性子串匹配：复合情绪如"平静中带点急切"匹配"平静"→0.5。
    长关键词优先，避免误匹配。
    """
    mood = (
        state.get("emotion")
        or state.get("current_mood")
        or state.get("mood")
        or state.get("当前情绪")
        or state.get("情绪状态")
        or ""
    )
    mood_str = str(mood).strip().lower()
    for keywords, score in _MOOD_KEYWORDS:
        for kw in keywords:
            if kw in mood_str:
                return score
    return 0.5


def _euclidean(a: List[float], b: List[float]) -> float:
    """两个 15D 向量的归一化欧氏距离"""
    d2 = sum((a[i] - b[i]) ** 2 for i in range(len(a)))
    return math.sqrt(d2 / len(a))


def _max_dev(a: List[float], b: List[float]) -> float:
    """逐维最大偏差"""
    return max(abs(a[i] - b[i]) for i in range(len(a)))


# Path A 仅能可靠估计这 8 个维度（其余 7 个维度从 state 标签中无法提取信号）
_PATH_A_ESTIMABLE_DIMS = {0, 2, 7, 8, 11, 12, 13, 14}
# agreeableness(0) neuroticism(2) politeness(7) assertiveness(8)
# initial_mood(11) mood_volatility(12) boundary_testing(13) truth_consistency(14)


def _attribute_deviation(
    S: List[float], V: Optional[List[float]], A: List[float],
    estimated_dims: Optional[set] = None,
) -> Dict[str, Any]:
    """三层偏差归因

    d_sv: 采样 vs 验证（画像生成质量）
    d_va: 验证 vs 审计（用户行为一致性/夸大偏差）
    d_sa: 采样 vs 审计（端到端保真度）

    若 estimated_dims 不为 None（Path A），仅在这些维度上计算偏差，
    其余维度标记为 N/A 避免系统性回退 0.5 污染结果。
    """
    # 防御 None: S/A 本身可能为 None、含 None 元素、或长度不匹配
    if S is None:
        S = [0.5] * (len(A) if A else 15)
    if A is None:
        A = [0.5] * (len(S) if S else 15)
    n = max(len(S), len(A), 15)
    if len(S) != n:
        S = list(S) + [0.5] * (n - len(S))
    if len(A) != n:
        A = list(A) + [0.5] * (n - len(A))
    S = [0.5 if v is None else float(v) for v in S]
    A = [0.5 if v is None else float(v) for v in A]
    if V is None:
        default_v = [0.5] * n
    else:
        default_v = [0.5 if v is None else float(v) for v in V]
        if len(default_v) != n:
            default_v = list(default_v) + [0.5] * (n - len(default_v))

    # Path A: 仅可估维度
    if estimated_dims is not None:
        dims = sorted(estimated_dims)
        S_sub = [S[i] for i in dims]
        A_sub = [A[i] for i in dims]
        V_sub = [default_v[i] for i in dims]
        d_sv = _euclidean(S_sub, V_sub)
        d_va = _euclidean(V_sub, A_sub)
        d_sa = _euclidean(S_sub, A_sub)
        n_estimated = len(dims)
        n_unestimated = 15 - n_estimated
    else:
        d_sv = _euclidean(S, default_v)
        d_va = _euclidean(default_v, A)
        d_sa = _euclidean(S, A)
        n_estimated = 15
        n_unestimated = 0

    # Tier 判定 — 以 d_sa（端到端行为保真度）为主导，d_sv 为辅
    # green: 行为与实际高度一致；yellow: 可接受；red: 偏差过大
    if d_sa < 0.25 and d_sv < 0.20:
        tier = "green"
    elif d_sa < 0.35:
        tier = "yellow"
    else:
        tier = "red"

    # 偏差归因
    if d_sv > d_va and d_sv > d_sa:
        primary = "A1_画像生成质量"
    elif d_va > d_sv and d_va > d_sa:
        primary = "A2_用户行为一致性(夸大偏差)"
    else:
        primary = "A3_审计噪声(多轮稀释)"

    return {
        "d_sv": round(d_sv, 4),
        "d_va": round(d_va, 4),
        "d_sa": round(d_sa, 4),
        "max_sv_dev": round(_max_dev(S, default_v), 4),
        "max_va_dev": round(_max_dev(default_v, A), 4),
        "tier": tier,
        "primary_deviation": primary,
        "method": ("path_a_state" if V is None else "path_b_llm"),
        "n_estimated_dims": n_estimated,
        "n_unestimated_dims": n_unestimated,
    }


class ProfileAuditor:
    """对话后用户画像审计器"""

    def __init__(self, client: Optional[LLMClient] = None):
        self.client = client or get_auditor_client(temperature=0.0)

    # ---- Path A: 基于 state 标签的循环一致性（零 LLM 成本）----

    def audit_path_a(
        self, conv: Conversation, state_trajectory: List[Dict[str, Any]],
    ) -> None:
        """从 state 轨迹估算 15D，与采样向量对比

        只能可靠估算情绪维度 (~2/15)，其余回退 0.5。
        校准机制: corr(Path_A, Path_B) > 0.8 → Path A 可信。
        """
        if not conv.sampled_vector:
            return

        if not state_trajectory:
            conv.consistency = {
                "tier": "no_data",
                "d_sv": None, "d_va": None, "d_sa": None,
                "method": "path_a_state",
                "reason": "state_trajectory为空，无法估算行为向量",
            }
            return

        estimated = _estimate_from_state_trajectory(state_trajectory)
        # 对抗画像修正: 对抗行为中的"试探"被 Path A 误读为低宜人，
        # 检测 thought 标签中是否有对抗策略执行记录，若有则微调 agreeableness
        adv_turns = 0
        for turn in conv.turns:
            if turn.speaker == "user" and turn.parsed_tags:
                thought = turn.parsed_tags.get("thought", "")
                if isinstance(thought, str) and "对抗策略执行" in thought and "已执行" in thought:
                    adv_turns += 1
        if adv_turns >= 2:
            adv_ratio = min(adv_turns / max(len(state_trajectory), 1), 0.5)
            idx_agree = DIM_NAME_TO_INDEX["agreeableness"]
            current = estimated[idx_agree]
            estimated[idx_agree] = min(current + adv_ratio * 0.15, 1.0)
        result = _attribute_deviation(conv.sampled_vector, None, estimated,
                                      estimated_dims=_PATH_A_ESTIMABLE_DIMS)
        result["method"] = "path_a_state"
        result["path_a_note"] = (
            "仅估算 ~8/15 维(情绪+stance+边界+一致性)，d_sa 仅计算可估维度"
            "其余维度回退0.5。Path A 用于快速筛查，Path B 为权威审计。"
            "Path A 系统性偏悲观是已知局限，非异常。"
        )
        # 不覆盖 conv.verified_vector——Path A 估计不完整，覆盖会破坏 Path B 的 d_sv/d_va 归因
        conv.consistency = result

    # ---- Path B: LLM 行为审计（抽样 ~10%）----

    def audit_path_b(self, conv: Conversation) -> None:
        """让 LLM 阅读对话文本，独立对 15 维重新打分"""
        if not conv.sampled_vector:
            return

        anchor_desc = translate_vector_to_anchor(conv.sampled_vector)
        conv_text = _format_conversation_for_audit(conv)
        user_message = BEHAVIOR_AUDIT_PROMPT.format(
            conversation_text=_safe(conv_text),
            anchor_description=_safe(anchor_desc),
        )

        dim_names = [d.name for d in DIMENSIONS]
        schema = {name: 0.0 for name in dim_names}

        try:
            result = self.client.chat_structured(
                system_prompt="你是一个用户行为审计器。请输出JSON。",
                user_message=user_message,
                output_schema=schema,
            )
            if result is None:
                print(f"[审计错误] {conv.id}: chat_structured JSON解析失败")
                return
            audited = [max(0.0, min(1.0, float(result.get(name, 0.5)))) for name in dim_names]
            conv.audited_vector = audited

            # 三层偏差归因
            S = conv.sampled_vector
            V = conv.verified_vector
            A = audited
            attr = _attribute_deviation(S, V, A)
            attr["method"] = "path_b_llm"
            # 如果已有 Path A 结果，保留 Path A 数据后合并
            if conv.consistency:
                path_a_data = {
                    f"path_a_{k}": v
                    for k, v in conv.consistency.items()
                    if not k.startswith("path_")
                }
                conv.consistency = {**path_a_data, **attr}
            else:
                conv.consistency = attr

        except Exception as e:
            print(f"[审计错误] {conv.id}: {e}")

    # ---- 校准: Path A vs Path B 交叉验证 ----

    def calibrate(self, conversations: List[Conversation]) -> Dict[str, Any]:
        """对同时有自检结果和 Path B 审计结果的对话计算相关性

        比较 verified_vector（画像生成自检回路）与 audited_vector（Path B LLM 行为审计）
        若平均相关 > 0.8 → 自检回路可信；否则 → state 标签质量不足

        Returns: {corr, n, verdict}
        """
        a_scores, b_scores = [], []
        for conv in conversations:
            A = conv.verified_vector  # 自检回路输出
            B = conv.audited_vector   # Path B LLM 行为审计
            if A and B:
                a_scores.append(A)
                b_scores.append(B)

        n = len(a_scores)
        if n < 3:
            return {"corr": None, "n": n, "verdict": "insufficient_data"}

        # 按维度计算平均相关
        dim_corrs = []
        for dim_i in range(15):
            a_dim = [s[dim_i] for s in a_scores]
            b_dim = [s[dim_i] for s in b_scores]
            corr = _pearson(a_dim, b_dim)
            dim_corrs.append(corr)

        avg_corr = sum(dim_corrs) / len(dim_corrs)
        verdict = (
            "generation_behavior_consistent"
            if avg_corr > 0.8
            else "generation_behavior_divergent"
        )
        return {
            "corr": round(avg_corr, 4),
            "dim_corrs": {DIMENSIONS[i].name: round(c, 4) for i, c in enumerate(dim_corrs)},
            "n": n,
            "verdict": verdict,
        }


def _format_conversation_for_audit(conv: Conversation) -> str:
    """将对话格式化为审计可读文本"""
    lines = []
    for turn in conv.turns:
        label = "客服" if turn.speaker == "system" else "用户"
        lines.append(f"[{turn.turn_number}] {label}: {turn.content}")
    return "\n".join(lines)


def _pearson(xs: List[float], ys: List[float]) -> float:
    """Pearson 相关系数"""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx < 1e-10 or sy < 1e-10:
        return 0.0
    cov = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return cov / (sx * sy)
