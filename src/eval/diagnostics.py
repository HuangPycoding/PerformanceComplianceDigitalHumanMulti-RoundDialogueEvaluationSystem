"""诊断引擎 — CaseDX / SimDX / ModelDX / EfficiencyDX + 归因 + 归因置信度"""
import re
from typing import Any, Dict, List, Optional, Tuple

from src.eval.config import DIAGNOSTICS_CONFIG
from src.models.case import Case
from src.models.conversation import Conversation
from src.models.evaluation import AttributionItem, CheckResult, Defect, DimensionChecklist


def run_diagnostics(
    conv: Conversation,
    case: Case,
    checklists: Dict[str, DimensionChecklist],
    additional_defects: List[Defect],
    ratings: Dict[str, str],
    tier1: Dict[str, Any],
    signals: Dict[str, Any],
) -> List[AttributionItem]:
    """主入口：综合四类诊断，输出归因列表"""
    attributions = []

    # 逐维度归因
    for dim, checklist in checklists.items():
        dim_attrs = _attribute_dimension(dim, checklist, case, conv, tier1, signals)
        attributions.extend(dim_attrs)

    # additional_defects 归因
    for defect in additional_defects:
        attributions.append(AttributionItem(
            source=(defect.attribution or "model").lower(),
            category="additional_defect",
            description=defect.description,
            confidence=0.7,
            evidence_chain=[f"T{defect.turn}: severity={defect.severity}"],
            suggested_actions=_suggest_model_fix(defect.description),
        ))

    # V3 audited_vector 归因消费 (G.2): 用户极端画像 → Simulator 偏差归因
    v3_attrs = _attribute_simulator_bias(conv, ratings)
    attributions.extend(v3_attrs)

    # 全局归因
    attributions.extend(_global_attribution(ratings, attributions, case))

    # 计算归因置信度
    _compute_attribution_confidence(attributions)

    return attributions


def _attribute_dimension(
    dim: str,
    checklist: DimensionChecklist,
    case: Case,
    conv: Conversation,
    tier1: Dict[str, Any],
    signals: Dict[str, Any],
) -> List[AttributionItem]:
    """对单个维度逐条 NO 项归因"""
    attrs = []

    for item in checklist.items:
        if item.status not in ("NO", "MOSTLY_NO", "PARTIAL"):
            continue
        attribution = _classify_item_attribution(item, dim, case, conv, tier1, signals)
        attrs.append(AttributionItem(
            source=attribution["source"],
            category=dim,
            description=f"[{item.item_id}] {item.description}",
            confidence=attribution["confidence"],
            evidence_chain=_build_evidence_chain(item, dim, checklist),
            suggested_actions=attribution["suggestions"],
        ))

    return attrs


def _classify_item_attribution(
    item: CheckResult,
    dim: str,
    case: Case,
    conv: Conversation,
    tier1: Dict[str, Any],
    signals: Dict[str, Any],
) -> Dict[str, Any]:
    """归因分类：判断 NO 项根因是 Case/Simulator/Model"""
    source = item.source

    # Simulator 来源项 → 检查信号一致性
    if source == "simulator":
        if item.signal_consistency == "矛盾":
            # 信号与对话文本不一致 → Simulator 问题
            return {
                "source": "simulator",
                "confidence": 0.75,
                "suggestions": [
                    f"检查 Simulator {dim} 标签生成逻辑",
                    f"复查第{_extract_turn(item.evidence)}轮 parsed_tags 是否准确",
                ],
            }
        else:
            # 信号一致但 NO → Model 问题
            return {
                "source": "model",
                "confidence": 0.80,
                "suggestions": _suggest_model_fix(item.description),
            }

    # Case 来源项
    if source == "case":
        # 检查是否 Case 设计过严
        if _is_case_too_strict(item, case, dim):
            return {
                "source": "case",
                "confidence": 0.70,
                "suggestions": [
                    f"复查 '{item.item_id}' 的 Case 设计是否合理",
                    "考虑降低该清单项的要求或调整权重",
                ],
            }
        else:
            return {
                "source": "model",
                "confidence": 0.85,
                "suggestions": _suggest_model_fix(item.description),
            }

    # LLM 补充 / pattern_mined → Model
    return {
        "source": "model",
        "confidence": 0.65,
        "suggestions": _suggest_model_fix(item.description),
    }


def _is_case_too_strict(item: CheckResult, case: Case, dim: str) -> bool:
    """判断是否 Case 设计过严导致误报"""
    cfg = DIAGNOSTICS_CONFIG
    # 高 complexity_score + 该维度多个 NO → 可能是 Case 过严
    if case.complexity_score < cfg["too_strict_complexity_low"]:
        return False
    # 约束类、安全类通常不过严
    if dim in cfg["strict_dimension_whitelist"]:
        return False
    # 复杂度高时 task/flows 可能定义过细
    return case.complexity_score >= cfg["too_strict_complexity_high"] and \
        dim in cfg["strict_dimension_candidates"]


def _build_evidence_chain(
    item: CheckResult,
    dim: str,
    checklist: DimensionChecklist,
) -> List[str]:
    """构建归因证据链"""
    chain = []
    if item.evidence:
        chain.append(f"清单证据: {item.evidence}")
    if item.signal_consistency:
        chain.append(f"信号一致性: {item.signal_consistency}")
    # 同维度其他项 YES 比例
    yes_ratio = checklist.yes_ratio
    chain.append(f"维度 YES 占比: {yes_ratio:.0%}")
    return chain


def _global_attribution(
    ratings: Dict[str, str],
    dim_attributions: List[AttributionItem],
    case: Case,
) -> List[AttributionItem]:
    """全局归因：整个对话的问题集中在哪"""
    attrs = []

    # 统计各来源分布
    source_counts: Dict[str, int] = {}
    for a in dim_attributions:
        source_counts[a.source] = source_counts.get(a.source, 0) + 1

    total = sum(source_counts.values()) or 1

    # Case 设计问题
    if source_counts.get("case", 0) / total > 0.4:
        case_dims = [a.category for a in dim_attributions if a.source == "case"]
        attrs.append(AttributionItem(
            source="case",
            category="global",
            description=f"多条清单项归因于 Case 设计（涉及 {', '.join(set(case_dims))}）",
            confidence=0.7,
            evidence_chain=[f"Case 归因占比 {source_counts['case']/total:.0%}"],
            suggested_actions=["复查 Case 指令设计", "调整过严的清单项"],
        ))

    # 模型能力短板
    model_dims = set(a.category for a in dim_attributions if a.source == "model")
    if model_dims:
        attrs.append(AttributionItem(
            source="model",
            category="global",
            description=f"模型在 {', '.join(sorted(model_dims)[:3])} 维度存在能力短板",
            confidence=0.75,
            evidence_chain=[f"涉及 {len(model_dims)} 个维度"],
            suggested_actions=["针对能力短板维度做定向优化", "增加该维度的少样本示例"],
        ))

    # 批量归因优先级
    if ratings.get("SAFETY") == "不合格":
        attrs.append(AttributionItem(
            source="model",
            category="SAFETY",
            description="安全维度不合格——最高优先级修复",
            confidence=0.95,
            suggested_actions=["立即检查安全合规", "排查是否绕过身份核实或泄露信息"],
        ))

    return attrs


def _compute_attribution_confidence(attributions: List[AttributionItem]) -> None:
    """根据归因间的相互印证调整置信度"""
    if len(attributions) < 2:
        return

    model_attrs = [a for a in attributions if a.source == "model"]
    case_attrs = [a for a in attributions if a.source == "case"]

    # 模型归因互相印证 → 提升置信度
    if len(model_attrs) >= 3:
        for a in model_attrs:
            a.confidence = min(0.95, a.confidence + 0.1)

    # Case 归因彼此独立 → 降低置信度（底值 0.15 允许多次衰减）
    if len(case_attrs) >= 2:
        for a in case_attrs:
            a.confidence = max(0.15, a.confidence - 0.08)


def _attribute_simulator_bias(
    conv: Conversation,
    ratings: Dict[str, str],
) -> List[AttributionItem]:
    """V3 audited_vector 归因消费 (G.2):
    当维度评分低时，检查是否因用户画像极端导致 Simulator 信号偏差。
    映射: SENTIMENT+neuroticism / EFFICIENCY+verbosity / SAFETY+boundary_testing
    """
    attrs = []
    audited = getattr(conv, "audited_vector", None)
    sampled = getattr(conv, "sampled_vector", None)
    vector = audited or sampled
    if not vector or len(vector) < 15:
        return attrs

    vector_source = "audited" if audited else "sampled_unverified"

    # 画像维度索引: 0=agreeableness, 1=patience, 2=neuroticism, 3=conscientiousness,
    #   4=openness, 5=politeness, 6=extraversion, 7=verbosity, 8=assertiveness,
    #   9=information_verification, 10=urgency, 11=initial_mood, 12=mood_volatility,
    #   13=boundary_testing, 14=truth_consistency
    bias_map = {
        "SENTIMENT": (2, 0.7, "neuroticism", "sentiment_bias",
                      "用户神经质水平高——Simulator 情感信号可能过度敏感"),
        "EFFICIENCY": (7, 0.7, "verbosity", "efficiency_bias",
                       "用户高话痨倾向——Simulator 效率信号可能过度严格"),
        "SAFETY": (13, 0.7, "boundary_testing", "safety_bias",
                   "用户边界试探倾向高——Simulator 安全信号可能过度告警"),
    }

    for dim, (vec_idx, threshold, trait_name, bias_type, reason) in bias_map.items():
        rating = ratings.get(dim, "")
        if rating not in ("需改进", "不合格"):
            continue
        val = vector[vec_idx] if vec_idx < len(vector) else 0.5
        if val > threshold:
            confidence = 0.65 if vector_source == "sampled_unverified" else 0.75
            attrs.append(AttributionItem(
                source="simulator",
                category=dim,
                description=f"[{bias_type}] {reason} ({trait_name}={val:.2f})",
                confidence=confidence,
                evidence_chain=[
                    f"audited_vector[{vec_idx}]={val:.2f} > {threshold}",
                    f"vector_source={vector_source}",
                    f"{dim}_rating={rating}",
                ],
                suggested_actions=[
                    f"复查 {dim} Simulator 标签在 {trait_name} 极端用户上的准确性",
                    "考虑按用户画像分层统计该维度评分分布",
                ],
            ))

    return attrs


def _suggest_model_fix(description: str) -> List[str]:
    """根据缺陷描述生成模型优化建议"""
    suggestions = []
    desc_lower = description.lower()
    if any(kw in desc_lower for kw in ["遗漏", "缺失", "未执行", "跳过", "漏"]):
        suggestions.append("检查 prompt 中对应步骤指令是否明确")
    if any(kw in desc_lower for kw in ["错误", "不对", "不一致", "矛盾"]):
        suggestions.append("增加该场景的少样本示例或微调数据")
    if any(kw in desc_lower for kw in ["敷衍", "一笔带过", "不充实", "机械", "模板"]):
        suggestions.append("优化生成温度或多样性参数")
    if any(kw in desc_lower for kw in ["重复", "卡死", "循环"]):
        suggestions.append("检查重复检测及策略切换逻辑")
    if any(kw in desc_lower for kw in ["态度", "情绪", "忽略", "未回应"]):
        suggestions.append("增强情感感知及回应策略")
    if not suggestions:
        suggestions.append("分析该缺陷的具体上下文并针对性优化")
    return suggestions


def _extract_turn(evidence: str) -> str:
    """从 evidence 文本提取轮次号"""
    m = re.search(r'T(\d+)', evidence)
    return m.group(1) if m else "?"
