"""评测叙述性报告生成器 — 以文字解说分析为主，数字评分作为可参考辅助数据

每条对话生成一份完整的叙述性报告，包含:
  1. 对话概要 (总评分 + 核心结论)
  2. 逐维度分析 (每个维度: 发生了什么 + 评了什么 + 为什么)
  3. 对话证据引用 (引用具体轮次内容作为评测依据)
  4. 置信度说明 (本次评测是否可靠 + 原因)
  5. 优化建议 (基于归因分析的具体改进方向)
  6. 完整对话文本 (作为客观证据)
  7. 详细评分数据 (数字作为参考附录)
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.case import Case
from src.models.conversation import Conversation
from src.models.evaluation import (
    EvalResult, EvalConfidence, AttributionItem,
    CrossValidationAlert, MetaCheckAlert, Defect,
)

# 维度中文名 + 解释
DIM_EXPLANATIONS = {
    "SAFETY": ("安全合规", "是否完成了身份核实、是否保护了敏感信息、是否输出了安全内容"),
    "TASK_COMPLETION": ("任务达成", "核心任务目标是否达成、收尾是否完整、跨轮是否自洽"),
    "FLOW_COVERAGE": ("流程覆盖", "Case定义的流程步骤是否被执行、执行质量如何"),
    "KNOWLEDGE": ("知识准确性", "回答是否与知识点一致、是否编造了不存在的信息"),
    "CONSTRAINT": ("约束遵守", "是否遵守了Case定义的语义约束和规则限制"),
    "EFFICIENCY": ("对话效率", "轮次是否在合理范围内、是否存在卡死或冗余"),
    "SENTIMENT": ("情感适配", "是否察觉并恰当回应了用户的情绪变化"),
    "ROLE": ("角色一致性", "是否始终保持在Case定义的角色身份内、是否自然"),
    "OPENING": ("开场白合规", "是否使用了规定的开场白、关键要素是否齐全"),
}

RATING_EXPLANATIONS = {
    "卓越": "该维度表现优异，无明显问题",
    "良好": "该维度总体良好，仅有轻微瑕疵",
    "合格": "该维度基本合格，存在可改进之处",
    "需改进": "该维度有明显不足，需要关注和优化",
    "不合格": "该维度存在严重问题，必须立即改进",
    "无法评估": "该维度因数据不足无法评估",
}


def generate_narrative_report(
    result: EvalResult,
    conv: Conversation,
    case: Case,
    conv_index: int = 1,
    conv_time: float = 0,
    eval_time: float = 0,
) -> str:
    """生成单条对话的完整叙述性评测报告"""

    L = []
    c = result.confidence

    # ===== 标题 =====
    L.append(f"# 评测报告 — 对话 {conv_index}")
    L.append(f"")
    L.append(f"**Case #{case.id}**: {case.title}")
    L.append(f"**对话ID**: {conv.id} | **轮次**: {conv.total_turns} | **状态**: {conv.status}")
    L.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"")

    # ===== 一、总评概要 =====
    L.append(f"## 一、总评概要")
    L.append(f"")

    # 核心结论
    worst_dims = sorted(
        [(d, r) for d, r in result.ratings.items() if r in ("不合格", "需改进")],
        key=lambda x: {"不合格": 0, "需改进": 1}.get(x[1], 2)
    )
    best_dims = [d for d, r in result.ratings.items() if r == "卓越"]

    score_level = "优秀" if result.total_indicative_score >= 80 else \
                  "良好" if result.total_indicative_score >= 65 else \
                  "一般" if result.total_indicative_score >= 50 else "较差"

    L.append(f"**综合评价**: {score_level}（{result.total_indicative_score:.1f} 分 / 百分制 **{result.total_score_100}** 分）。")

    # 标注无法评估的维度
    unevaluable = [d for d, r in result.ratings.items() if r == "无法评估"]
    if unevaluable:
        dim_names = "、".join([DIM_EXPLANATIONS.get(d, (d,))[0] for d in unevaluable])
        L.append(f"**⚠️ 无法评估**: {dim_names}（评测引擎数据不足，该维度不参与评分判断，默认取中性分）。")
        L.append(f"")

    if worst_dims:
        dim_names = "、".join([f"{DIM_EXPLANATIONS[d][0]}({r})" for d, r in worst_dims[:3]])
        L.append(f"**主要问题**: {dim_names}。")
    if best_dims:
        dim_names = "、".join([DIM_EXPLANATIONS[d][0] for d in best_dims[:3]])
        L.append(f"**表现良好**: {dim_names}。")

    # SAFETY make-or-break
    if result.ratings.get("SAFETY") == "不合格":
        L.append(f"**⚠️ SAFETY 不合格触发总分钳制**: 安全不合格时总分配上限为 50 分，当前 {result.total_indicative_score:.1f} 分已受此限制。")

    # Confidence
    if c:
        conf_desc = {"high": "本次评测结果可信度高，可用于优化决策",
                     "medium": "本次评测结果基本可信，可参考使用",
                     "low": "本次评测结果可信度偏低，建议人工复核",
                     "unreliable": "本次评测结果可信度不足，不建议直接用于优化决策"}
        L.append(f"**评测可信度**: {c.level}（{c.overall:.2f}）。{conf_desc.get(c.level, '')}")
        if c.needs_human_review:
            L.append(f"**⚠️ 建议人工复核**: 系统标记该条评测需要人工审核。")

    L.append(f"")

    # ===== 二、逐维度分析 =====
    L.append(f"## 二、逐维度分析")
    L.append(f"")

    # 按评分从低到高排列（优先展示问题维度）
    dim_order = sorted(result.ratings.items(),
                       key=lambda x: {"不合格": 0, "需改进": 1, "合格": 2, "良好": 3, "卓越": 4, "无法评估": 5}.get(x[1], 99))

    for dim, rating in dim_order:
        dim_name, dim_explain = DIM_EXPLANATIONS.get(dim, (dim, ""))
        score = result.indicative_scores.get(dim, 0)
        L.append(f"### {dim_name}（{dim}）: {rating}（{score:.1f} 分）")
        L.append(f"")
        L.append(f"> {dim_explain}")
        L.append(f"")
        L.append(f"**{RATING_EXPLANATIONS.get(rating, '')}**")
        L.append(f"")

        # 维度专属分析：从 checklist 和 alerts 中提取关键信息
        checklist = result.dimension_checklists.get(dim)
        if checklist:
            # 关键负面项
            negative_items = [i for i in checklist.items
                            if i.status in ("NO", "MOSTLY_NO", "PARTIAL")]
            positive_items = [i for i in checklist.items
                            if i.status in ("YES", "MOSTLY_YES")]

            if negative_items:
                L.append(f"**发现的问题**（共 {len(negative_items)} 项）:")
                for item in negative_items[:5]:
                    ev = item.evidence[:80] if item.evidence else "无证据引用"
                    L.append(f"- **{item.item_id}**: {item.description[:60]} → 判定为 {item.status}")
                    if item.evidence:
                        L.append(f"  - 证据: {ev}")
                if len(negative_items) > 5:
                    L.append(f"- ... 还有 {len(negative_items) - 5} 项")
                L.append(f"")

            if positive_items and rating in ("卓越", "良好"):
                L.append(f"**良好表现**（共 {len(positive_items)} 项）:")
                for item in positive_items[:3]:
                    L.append(f"- **{item.item_id}**: {item.description[:60]} → 判定为 {item.status}")
                L.append(f"")

        # 交叉验证告警（本维度相关）
        dim_alerts = [a for a in result.cross_validation_alerts
                      if dim in (a.dimension or "")]
        for a in dim_alerts:
            L.append(f"**交叉验证**: [{a.severity}] {a.description}")
            L.append(f"")

    # ===== 三、对话证据 =====
    L.append(f"## 三、对话证据")
    L.append(f"")
    L.append(f"以下为完整对话文本，作为上述评测的客观依据。每轮标注了说话者和内容。")
    L.append(f"")

    for turn in conv.turns:
        sp = "客服" if turn.speaker == "system" else "用户"
        tags_info = ""
        if turn.parsed_tags:
            state = turn.parsed_tags.get("state", {})
            if isinstance(state, dict) and state.get("emotion"):
                tags_info = f" [情绪: {state.get('emotion')}]"
        L.append(f"**T{turn.turn_number} [{sp}]**{tags_info}: {turn.content}")
        L.append(f"")

    # ===== 四、置信度详细说明 =====
    if c:
        L.append(f"## 四、评测可信度详细说明")
        L.append(f"")
        L.append(f"| 指标 | 值 | 说明 |")
        L.append(f"|------|-----|------|")
        L.append(f"| 综合置信度 | {c.overall:.2f} | 综合 9 维度置信度的加权平均 |")
        L.append(f"| 置信度等级 | {c.level} | high(≥0.80) / medium(≥0.65) / low(≥0.50) / unreliable(<0.50) |")
        L.append(f"| Simulator 质量 | {c.simulator_tier} | green(高) / yellow(中) / red(低) |")
        L.append(f"| 信号矛盾数 | {c.signal_conflict_count} | Simulator信号与对话文本不一致的次数 |")
        L.append(f"| 证据空率 | {c.evidence_empty_ratio:.0%} | 未引用原文证据的清单项占比 |")
        L.append(f"| 证据阶段覆盖 | {c.avg_evidence_coverage:.1f}/3 | 证据覆盖对话前/中/后段的程度 |")
        L.append(f"| Judge间不一致 | {c.cross_judge_anomaly_pairs} 对 | 跨维度评分差异 ≥ 2 级的维度对 |")
        L.append(f"| 归因置信度 | {c.attribution_confidence:.2f} | 归因分析的可靠程度 |")
        L.append(f"| 解析成功 | {'是' if c.parse_success else '否'} | LLM 输出是否成功解析 |")
        if c.extreme_profile_flag:
            L.append(f"| 极端画像 | {', '.join(c.extreme_profile_dims[:3])} | 用户画像存在极端维度 |")
        L.append(f"")

        if c.confidence_reasoning:
            L.append(f"**置信度理由**: {c.confidence_reasoning}")
            L.append(f"")

    # ===== 五、归因与建议 =====
    L.append(f"## 五、归因分析与优化建议")
    L.append(f"")

    model_attrs = [a for a in result.attributions if a.source == "model"]
    case_attrs = [a for a in result.attributions if a.source == "case"]
    sim_attrs = [a for a in result.attributions if a.source == "simulator"]

    if model_attrs:
        L.append(f"### 模型（Assistant）相关 ({len(model_attrs)} 项)")
        L.append(f"")
        for a in model_attrs[:5]:
            L.append(f"- **{a.category}** (置信度 {a.confidence:.0%}): {a.description}")
            if a.suggested_actions:
                for s in a.suggested_actions[:2]:
                    L.append(f"  - 建议: {s}")
        L.append(f"")

    if case_attrs:
        L.append(f"### Case 设计相关 ({len(case_attrs)} 项)")
        L.append(f"")
        for a in case_attrs[:3]:
            L.append(f"- **{a.category}**: {a.description}")
        L.append(f"")

    if sim_attrs:
        L.append(f"### Simulator 相关 ({len(sim_attrs)} 项)")
        L.append(f"")
        for a in sim_attrs[:3]:
            L.append(f"- **{a.category}**: {a.description}")
        L.append(f"")

    if result.improvement_suggestions:
        L.append(f"### 改进建议")
        L.append(f"")
        for s in result.improvement_suggestions[:5]:
            L.append(f"- {s}")
        L.append(f"")

    # ===== 六、Cross-Validation Alerts =====
    if result.cross_validation_alerts:
        L.append(f"## 六、交叉验证告警")
        L.append(f"")
        for a in result.cross_validation_alerts[:5]:
            L.append(f"- **[{a.severity}] {a.dimension}**: {a.description}")
        L.append(f"")

    # ===== 七、Meta Check Alerts =====
    if result.meta_check_alerts:
        logic_alerts = [a for a in result.meta_check_alerts if a.check_type == "logic"]
        coverage_alerts = [a for a in result.meta_check_alerts if a.check_type == "coverage"]
        evidence_alerts = [a for a in result.meta_check_alerts if a.check_type == "evidence"]

        L.append(f"## 七、元检查告警")
        L.append(f"")
        L.append(f"- 逻辑检查: {len(logic_alerts)} 条")
        L.append(f"- 覆盖检查: {len(coverage_alerts)} 条")
        L.append(f"- 证据检查: {len(evidence_alerts)} 条")
        L.append(f"")

        if logic_alerts:
            L.append(f"### 逻辑异常")
            for a in logic_alerts:
                L.append(f"- [{a.severity}] {a.description}")
            L.append(f"")

    # ===== 八、约束分流 =====
    L.append(f"## 八、成本追踪")
    L.append(f"")
    total_cons = result.tier1_constraint_count + result.llm_constraint_count
    if total_cons > 0:
        rule_pct = result.tier1_constraint_count / total_cons * 100
        L.append(f"- Case 共 {total_cons} 条约束")
        L.append(f"- Tier1 规则可检: {result.tier1_constraint_count} 条（{rule_pct:.0f}%），规则直接判定，零 LLM 成本")
        L.append(f"- LLM 语义核查: {result.llm_constraint_count} 条，需 Judge 逐条判断")
    L.append(f"")

    # ===== 附录：完整评分数据 =====
    L.append(f"---")
    L.append(f"")
    L.append(f"## 附录：完整评分数据")
    L.append(f"")
    L.append(f"### 维度评分")
    L.append(f"| 维度 | 评级 | 得分 | 置信度 |")
    L.append(f"|------|------|------|--------|")
    for dim in ["SAFETY","TASK_COMPLETION","FLOW_COVERAGE","KNOWLEDGE","CONSTRAINT",
                 "EFFICIENCY","SENTIMENT","ROLE","OPENING"]:
        r = result.ratings.get(dim, "N/A")
        s = result.indicative_scores.get(dim, 0)
        dc = c.per_dimension.get(dim, 0) if c else 0
        L.append(f"| {DIM_EXPLANATIONS.get(dim, (dim,))[0]} | {r} | {s:.1f} | {dc:.2f} |")
    L.append(f"")
    L.append(f"**总分**: {result.total_indicative_score:.1f}（百分制: {result.total_score_100}）")
    L.append(f"")

    return "\n".join(L)


def generate_batch_narrative(
    conversations: List[Conversation],
    eval_results: List[EvalResult],
    case: Case,
    profiles_info: Optional[Dict[str, Any]] = None,
    conv_time: float = 0,
    eval_time: float = 0,
) -> str:
    """生成批次级别的叙述性汇总报告"""

    L = []
    valid_results = [r for r in eval_results if r is not None]

    L.append(f"# 批次评测汇总报告 — Case #{case.id} {case.title}")
    L.append(f"")
    L.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"**对话数**: {len(conversations)} | **完成评测**: {len(valid_results)}")
    L.append(f"**对话耗时**: {conv_time:.0f}s | **评测耗时**: {eval_time:.0f}s")
    L.append(f"")

    # 核心发现
    L.append(f"## 核心发现")
    L.append(f"")

    all_scores = [r.total_indicative_score for r in valid_results]
    all_scores_100 = [r.total_score_100 for r in valid_results]
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    avg_score_100 = round(sum(all_scores_100) / len(all_scores_100)) if all_scores_100 else 0
    L.append(f"本次测试对 Case #{case.id}（{case.title}）共生成 {len(conversations)} 条对话，评测 {len(valid_results)} 条。")
    L.append(f"平均分 {avg_score:.1f}（百分制 {avg_score_100}），最高 {max(all_scores):.1f}，最低 {min(all_scores):.1f}。")

    # 问题维度识别
    all_ratings: Dict[str, List[str]] = {}
    for r in valid_results:
        for dim, rating in r.ratings.items():
            all_ratings.setdefault(dim, []).append(rating)

    problem_dims = []
    for dim in ["SAFETY","TASK_COMPLETION","FLOW_COVERAGE","KNOWLEDGE","CONSTRAINT",
                 "EFFICIENCY","SENTIMENT","ROLE","OPENING"]:
        ratings = all_ratings.get(dim, [])
        fail_rate = (ratings.count("不合格") + ratings.count("需改进")) / max(len(ratings), 1)
        if fail_rate >= 0.3:
            dim_name = DIM_EXPLANATIONS.get(dim, (dim,))[0]
            problem_dims.append(f"**{dim_name}**（{fail_rate:.0%} 需改进/不合格）")

    if problem_dims:
        L.append(f"")
        L.append(f"**主要短板**: {', '.join(problem_dims)}。")
        L.append(f"")

    # 最佳/最差对话
    if len(valid_results) >= 2:
        best_idx = all_scores.index(max(all_scores))
        worst_idx = all_scores.index(min(all_scores))
        L.append(f"**最佳对话**（#{best_idx+1}）: {max(all_scores):.1f} 分")
        L.append(f"**最差对话**（#{worst_idx+1}）: {min(all_scores):.1f} 分")
        L.append(f"")

    # 逐维度分析
    L.append(f"## 维度详情")
    L.append(f"")
    for dim in ["SAFETY","TASK_COMPLETION","FLOW_COVERAGE","KNOWLEDGE","CONSTRAINT",
                 "EFFICIENCY","SENTIMENT","ROLE","OPENING"]:
        dim_name = DIM_EXPLANATIONS.get(dim, (dim,))[0]
        ratings = all_ratings.get(dim, [])
        fail_rate = (ratings.count("不合格") + ratings.count("需改进")) / max(len(ratings), 1)
        dist = f"卓越{ratings.count('卓越')} 良好{ratings.count('良好')} 合格{ratings.count('合格')} 需改进{ratings.count('需改进')} 不合格{ratings.count('不合格')}"
        L.append(f"- **{dim_name}** ({dim}): {dist}")
        if fail_rate >= 0.3:
            L.append(f"  - ⚠️ 需改进/不合格率 {fail_rate:.0%}")
    L.append(f"")

    # 置信度汇总
    conf_levels = [r.confidence.level for r in valid_results if r.confidence]
    needs_review = sum(1 for r in valid_results if r.confidence and r.confidence.needs_human_review)
    L.append(f"## 置信度汇总")
    L.append(f"")
    L.append(f"- 需人工复核: {needs_review}/{len(valid_results)} 条")
    L.append(f"- 分布: high={conf_levels.count('high')} medium={conf_levels.count('medium')} low={conf_levels.count('low')} unreliable={conf_levels.count('unreliable')}")
    L.append(f"")

    return "\n".join(L)
