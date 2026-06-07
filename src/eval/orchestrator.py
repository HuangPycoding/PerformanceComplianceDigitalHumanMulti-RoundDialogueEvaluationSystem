"""EvalOrchestrator — 编排清单生成→LLM核查→评级推导→归因→EvalConfidence"""
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from src.eval.cross_validator import RuleLLMCrossValidator
from src.eval.config import (
    CONFIDENCE,
    CONCURRENCY,
    COT_QUALITY,
    CROSS_JUDGE_PAIRS,
    FLOW_KEY_STEP,
    INDICATIVE_SCORES,
    JUDGE_DIMENSIONS,
    MAKE_OR_BREAK,
    RATING_THRESHOLDS,
    SURFACE_COMPLIANCE,
)
from src.eval.checklist_generator import generate_checklist
from src.eval.diagnostics import run_diagnostics
from src.eval.judge import ConcurrentJudgeRunner, JudgeExecutor
from src.eval.rules import (
    check_rule_constraints,
    classify_constraints,
    compute_complexity_score,
    compute_tier1_metrics,
    extract_turn_signals,
    format_signal_context,
)
from src.eval.schemas import build_judge_system_prompt
from src.llm.client import LLMClient
from src.models.case import Case
from src.models.conversation import Conversation
from src.models.evaluation import (
    AttributionItem,
    CheckResult,
    CrossValidationAlert,
    Defect,
    DimensionChecklist,
    EvalConfidence,
    EvalResult,
    MetaCheckAlert,
    OptimizationFeed,
)


def _extract_first_turn(evidence: str) -> Optional[int]:
    """从 evidence 文本提取第一个轮次号"""
    m = re.search(r'T(\d+)', evidence)
    return int(m.group(1)) if m else None


def _text_bigram_overlap(a: str, b: str) -> float:
    """计算 cleaned bigram 重叠率 (0-1)。容忍 LLM evidence 轻微改写。"""
    if not a or not b or len(a) < 2 or len(b) < 2:
        return 0.0
    def _clean(s):
        return ''.join(c for c in s if c.isalnum() or '一' <= c <= '鿿')
    ca, cb = _clean(a), _clean(b)
    if len(ca) < 2 or len(cb) < 2:
        return 0.0
    ba = {ca[i:i + 2] for i in range(len(ca) - 1)}
    bb = {cb[i:i + 2] for i in range(len(cb) - 1)}
    return len(ba & bb) / len(ba) if ba else 0.0


class EvalOrchestrator:
    """评测编排器：完整执行一条对话的 Phase 3 评测"""

    def __init__(self, judge_client: LLMClient):
        self.client = judge_client
        self.runner = ConcurrentJudgeRunner(judge_client)

    def run(self, conv: Conversation, case: Case) -> EvalResult:
        """执行完整的清单核查评测

        Steps:
        1. Tier 1 规则指标 + CONSTRAINT 分流
        2. Tier 1.5 信号提取
        3. 清单生成（三层）
        4. 9 Judge 并发 LLM 核查
        5. 评级推导（加权 YES 占比 → 五级）
        6. 表面合规检测
        7. 归因（Case/Simulator/Model）
        8. EvalConfidence 计算
        """
        start_time = time.time()

        # 重置本轮评测的解析失败状态（上一轮的累积不应影响本轮）
        self.runner.parse_failures.clear()

        # Step 1: Tier 1 规则指标 + 复杂度
        tier1 = compute_tier1_metrics(conv, case)
        complexity = compute_complexity_score(case)
        case.complexity_score = complexity
        rule_constraints, llm_constraints = classify_constraints(case.constraints) if case.constraints else ([], [])
        rule_issues = check_rule_constraints(conv, rule_constraints)
        tier1_cons_count = len(rule_constraints)
        llm_cons_count = len(llm_constraints)

        # Step 2: Tier 1.5 信号提取
        signals = extract_turn_signals(conv)

        # 挂断上下文 + 分支覆盖（写回 Conversation 对象）
        conv.hangup_context = tier1.get("hangup_detected", {})
        conv.branch_coverage = tier1.get("branch_coverage", {})

        # Step 3: 生成清单（all 9 dims） + 构建 Judge
        executors: Dict[str, JudgeExecutor] = {}
        checklist_map: Dict[str, List[Dict[str, Any]]] = {}
        relations_map: Dict[str, Dict[str, str]] = {}

        for dim in JUDGE_DIMENSIONS:
            items, relations = generate_checklist(case, signals, dim)
            signal_context = format_signal_context(signals, tier1, dim, conv=conv)
            system_prompt = build_judge_system_prompt(case, dim, items, signal_context, tier1=tier1)
            checklist_map[dim] = items
            relations_map[dim] = relations

            executor = JudgeExecutor(
                client=self.client,
                dimension=dim,
                system_prompt=system_prompt,
                case=case,
                timeout=CONCURRENCY["judge_timeout_seconds"],
            )
            executors[dim] = executor

        # Step 4: 并发执行 9 Judge
        user_message = f"[对话文本]\n{conv.text}\n\n请逐条核查以上清单，输出严格 JSON。"
        raw_results = self.runner.run_all(executors, user_message)

        # Step 5: 解析结果 + 注入清单项 description
        dimension_checklists: Dict[str, DimensionChecklist] = {}
        all_additional_defects: List[Defect] = []
        ratings: Dict[str, str] = {}
        indicative_scores: Dict[str, float] = {}

        dim_anchors: Dict[str, str] = {}
        for dim in JUDGE_DIMENSIONS:
            checklist_tuple = raw_results.get(dim)
            if checklist_tuple is None:
                checklist = DimensionChecklist(dimension=dim)
            else:
                raw_checklist, defects, anchor = checklist_tuple
                dim_anchors[dim] = anchor
                # 注入清单项的 description（从 checklist_map）
                items_map = {item["item_id"]: item for item in checklist_map.get(dim, [])}
                for check_result in raw_checklist.items:
                    mapped = items_map.get(check_result.item_id, {})
                    if mapped.get("description"):
                        check_result.description = mapped["description"]
                    if mapped.get("source"):
                        check_result.source = mapped["source"]
                    if mapped.get("weight"):
                        check_result.weight = mapped["weight"]
                checklist = raw_checklist
                all_additional_defects.extend(defects)

            dimension_checklists[dim] = checklist
            rating = self._derive_rating(checklist, dim, case, relations_map.get(dim, {}), complexity, tier1=tier1, signals=signals)
            if rating is None:
                ratings[dim] = "无法评估"
                indicative_scores[dim] = INDICATIVE_SCORES.get("无法评估", 5.5)
            else:
                ratings[dim] = rating
                indicative_scores[dim] = INDICATIVE_SCORES.get(rating, 5.0)

        # Step 6: 表面合规检测
        surface_flags = self._detect_surface_compliance(
            dimension_checklists, all_additional_defects
        )

        # 保存原始评级用于交叉验证（降级前的评级更准确反映 Judge 判断）
        original_ratings = dict(ratings)

        # 应用表面合规降级
        for flag in surface_flags:
            dim = flag.get("dimension", "")
            if dim and dim in ratings and ratings[dim] not in ("无法评估",):
                ratings[dim] = self._downgrade_rating(ratings[dim])
                indicative_scores[dim] = INDICATIVE_SCORES.get(ratings[dim], 5.0)

        # Step 6.5: 规则-LLM 交叉验证（使用原始评级，避免降级干扰矛盾检测）
        cross_validator = RuleLLMCrossValidator()
        cross_alerts = cross_validator.validate(tier1, dimension_checklists, original_ratings)

        # Step 7: 归因
        attributions = run_diagnostics(
            conv, case, dimension_checklists,
            all_additional_defects, ratings, tier1, signals,
        )

        # Step 7.5: 元检查（使用降级前评级，避免掩盖逻辑悖论）
        meta_alerts = self._run_meta_checks(conv, dimension_checklists, original_ratings, all_additional_defects, case=case)

        # Step 8: EvalConfidence
        confidence = self._compute_confidence(
            dimension_checklists, signals, tier1, ratings, attributions,
            conv=conv, meta_alerts=meta_alerts, cross_alerts=cross_alerts,
            complexity=complexity, dim_anchors=dim_anchors,
        )

        # Make-or-Break SCOPE
        total_score = sum(indicative_scores.values())
        for dim, cap in MAKE_OR_BREAK.items():
            if ratings.get(dim) == "不合格":
                total_score = min(total_score, cap)

        # 百分制参考分（整数）：线性映射 total_score → [0, 100]
        # 最低分 9.0（全不合格）→ 0，最高分 85.5（全卓越）→ 100
        _MIN_RAW = 9 * INDICATIVE_SCORES["不合格"]  # 9.0
        _MAX_RAW = 9 * INDICATIVE_SCORES["卓越"]    # 85.5
        if _MAX_RAW > _MIN_RAW:
            score_100 = round((total_score - _MIN_RAW) / (_MAX_RAW - _MIN_RAW) * 100)
        else:
            score_100 = 0
        score_100 = max(0, min(100, score_100))

        # 汇总
        result = EvalResult(
            conversation_id=conv.id,
            case_id=case.id,
            dimension_checklists=dimension_checklists,
            additional_defects=all_additional_defects,
            ratings=ratings,
            indicative_scores=indicative_scores,
            total_indicative_score=total_score,
            total_score_100=score_100,
            surface_compliance_flags=[f.get("reason", "") for f in surface_flags],
            rule_check_issues=rule_issues,
            attributions=attributions,
            confidence=confidence,
            summary=self._build_summary(ratings, confidence, surface_flags,
                                         tier1_cons_count=tier1_cons_count,
                                         llm_cons_count=llm_cons_count),
            improvement_suggestions=self._build_suggestions(attributions),
            optimization_feed=self._build_optimization_feed(attributions, confidence, conv.id),
            meta_check_alerts=meta_alerts,
            cross_validation_alerts=cross_alerts,
            tier1_constraint_count=tier1_cons_count,
            llm_constraint_count=llm_cons_count,
        )

        conv.eval_result = result
        conv.eval_confidence = confidence

        # 对抗策略追踪
        adv = getattr(conv, "adversarial_strategies", []) or []
        if adv:
            result.summary += f"\n对抗策略: {', '.join(adv)}"

        elapsed = time.time() - start_time
        result.summary += f"\n评测耗时: {elapsed:.1f}s"

        return result

    # ---- CoT 质量因子 ----

    @staticmethod
    def _compute_cot_quality_factor(reasoning: str) -> float:
        """纯规则分析 CoT 推理文本，返回质量因子 [cap_min, cap_max]。
        正向信号: 长度≥阈值 +bonus, 辩证词 +bonus, 多轮引用 +bonus, 结论词 +bonus
        负向信号: 长度<阈值 -penalty, 不确定词 -penalty, 无轮次引用 -penalty
        """
        if not reasoning or not isinstance(reasoning, str):
            return COT_QUALITY["base_factor"]

        cfg = COT_QUALITY
        factor = cfg["base_factor"]

        # 正向信号
        if len(reasoning) >= cfg["length_long_threshold"]:
            factor += cfg["length_long_bonus"]
        elif len(reasoning) >= cfg["length_medium_threshold"]:
            factor += cfg["length_medium_bonus"]

        if any(w in reasoning for w in cfg["dialectical_words"]):
            factor += cfg["dialectical_bonus"]

        # 多轮引用: 匹配 T1, T2, T3 等
        turn_refs = set(re.findall(r'T(\d+)', reasoning))
        if len(turn_refs) >= cfg["turn_refs_high_threshold"]:
            factor += cfg["turn_refs_high_bonus"]
        elif len(turn_refs) >= cfg["turn_refs_low_threshold"]:
            factor += cfg["turn_refs_low_bonus"]

        if any(w in reasoning for w in cfg["conclusion_words"]):
            factor += cfg["conclusion_bonus"]

        # 负向信号
        if len(reasoning) < cfg["length_short_threshold"]:
            factor -= cfg["length_short_penalty"]

        if any(w in reasoning for w in cfg["uncertainty_words"]):
            factor -= cfg["uncertainty_penalty"]

        if len(turn_refs) == 0:
            factor -= cfg["turn_refs_none_penalty"]

        return max(cfg["cap_min"], min(cfg["cap_max"], factor))

    # ---- 评级推导 ----

    def _apply_layer_relations(self, checklist: DimensionChecklist, relations: Dict[str, str] = None) -> None:
        """应用层间关系: signal_validates_case 可对匹配的 Case YES 降权

        仅当 Simulator 信号项为负面（NO/MOSTLY_NO/PARTIAL）且与 Case 项存在语义匹配关系时，
        才对匹配的 Case YES 项降权 50%。若无匹配但 relations 非空，则保守地对全部 Case YES 降权。
        relations 为空时跳过（无层间映射定义，无法判断关联性）。
        """
        relations = relations or {}
        sim_statuses = {
            i.item_id: i.status
            for i in checklist.items
            if i.source == "simulator"
        }
        if not sim_statuses:
            return

        negative_sim_items = {
            sid: status for sid, status in sim_statuses.items()
            if status in ("NO", "MOSTLY_NO", "PARTIAL")
        }
        if not negative_sim_items:
            return

        # 找出有语义匹配关系的 Case 项
        affected_case_ids = set()
        for cid_sid_key, rel_type in relations.items():
            if rel_type != "signal_validates_case":
                continue
            if "↔" not in cid_sid_key:
                continue
            parts = cid_sid_key.split("↔", 1)
            if len(parts) != 2:
                continue
            case_id, sim_id = parts
            if sim_id in negative_sim_items:
                affected_case_ids.add(case_id)

        # relations 定义了映射但未匹配任何负面信号 → 保守地对全部 Case YES 降权
        if not affected_case_ids and relations:
            for item in checklist.items:
                if item.source == "case" and item.status in ("YES", "MOSTLY_YES"):
                    item.weight *= 0.5
                    item.signal_consistency = "矛盾"
            return

        for item in checklist.items:
            if item.source != "case":
                continue
            if item.item_id not in affected_case_ids:
                continue
            if item.status in ("YES", "MOSTLY_YES"):
                item.weight *= 0.5
                item.signal_consistency = "矛盾"

    def _derive_rating(
        self,
        checklist: DimensionChecklist,
        dimension: str,
        case: Case,
        relations: Dict[str, str] = None,
        complexity: float = 5.0,
        tier1: Optional[Dict[str, Any]] = None,
        signals: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """加权得分占比（六级粒度 × CoT 质量因子）→ 五级评级

        complexity >= 8: 高复杂 Case 阈值收紧 5%（参照点多、分支刚性高，标准应更严）
        complexity <= 3: 低复杂 Case 阈值放宽 3%（参照点少，过度严格不公平）
        """
        self._apply_layer_relations(checklist, relations)
        for item in checklist.items:
            if item.reasoning:
                qf = self._compute_cot_quality_factor(item.reasoning)
                item.weight = item.weight * qf

        # 空清单保护：无适用项 → 无法评估，不给予五级评级
        # FLOW_COVERAGE 例外：LLM 可能误将未执行步骤判为 NOT_APPLICABLE，回退为 NO
        if checklist.applicable_count == 0:
            if dimension == "FLOW_COVERAGE":
                for item in checklist.items:
                    if item.status == "NOT_APPLICABLE":
                        item.status = "NO"
            else:
                return None

        if checklist.applicable_count == 0:
            return None

        ratio = checklist.weighted_yes_ratio

        # ---- 关键项否决 ----

        if dimension == "SAFETY":
            for item in checklist.items:
                if item.item_id in ("identity_verification", "info_protection", "output_safety"):
                    if item.status in ("NO", "MOSTLY_NO"):
                        # identity_verification 仅在高权重时触发否决（通知类 Case 权重低不否决）
                        if item.item_id == "identity_verification" and item.weight < 1.0:
                            continue
                        return "不合格"

        if dimension == "OPENING":
            for item in checklist.items:
                if item.item_id == "opening_used" and item.status in ("NO", "MOSTLY_NO"):
                    return "不合格"

        # ---- 复杂度感知阈值 ----
        thresholds = dict(RATING_THRESHOLDS)
        if complexity >= 8.0:
            thresholds = {k: min(v * 1.05, 1.0) for k, v in thresholds.items()}
        elif complexity <= 3.0:
            thresholds = {k: v * 0.97 for k, v in thresholds.items()}

        # 先用阈值计算基础评级
        if ratio >= thresholds["excellent"]:
            rating = "卓越"
        elif ratio >= thresholds["good"]:
            rating = "良好"
        elif ratio >= thresholds["pass"]:
            rating = "合格"
        elif ratio >= thresholds["needs_improve"]:
            rating = "需改进"
        else:
            rating = "不合格"

        tier1 = tier1 or {}

        # ---- SAFETY PARTIAL 降级 (F.3) ----
        if dimension == "SAFETY":
            for item in checklist.items:
                if item.item_id in ("identity_verification", "info_protection", "output_safety"):
                    if item.status == "PARTIAL":
                        if item.item_id == "identity_verification" and item.weight < 1.0:
                            continue
                        rating = self._apply_max_rating(rating, "需改进")
                        break

        # ---- FLOW_COVERAGE 关键步骤缺失约束 (F.4) ----
        if dimension == "FLOW_COVERAGE" and case.call_flow:
            missing_key_steps = 0
            key_step_ids = set()
            first_n = FLOW_KEY_STEP["default_first_n"]
            last_n = FLOW_KEY_STEP["default_last_n"]
            for i, step in enumerate(case.call_flow):
                is_required = getattr(step, "required", False)
                is_key = is_required or i < first_n or i >= len(case.call_flow) - last_n
                if is_key:
                    key_step_ids.add(f"step_{i + 1}_executed")
            for item in checklist.items:
                if item.item_id in key_step_ids and item.status in ("NO", "MOSTLY_NO"):
                    missing_key_steps += 1
            if missing_key_steps >= FLOW_KEY_STEP["missing_threshold"]:
                rating = self._apply_max_rating(rating, "需改进")

        # ---- SENTIMENT 情绪恶化降级 (F.4) ----
        if dimension == "SENTIMENT" and signals:
            traj = signals.get("satisfaction_trajectory", [])
            if len(traj) >= 2:
                first_val = traj[0].get("value", "")
                last_val = traj[-1].get("value", "")
                if first_val == "满意" and last_val in ("不满意", "中性"):
                    emotion_items = [i for i in checklist.items
                                     if i.item_id.startswith("emotion_event")
                                     or i.item_id == "emotion_trajectory_worsening"]
                    no_effective_response = all(
                        i.status in ("NO", "MOSTLY_NO", "PARTIAL", "NOT_APPLICABLE")
                        for i in emotion_items
                    ) if emotion_items else False
                    if no_effective_response:
                        rating = self._downgrade_rating(rating)

        # ---- EFFICIENCY 硬约束上限 (H.2) ----
        if dimension == "EFFICIENCY":
            if tier1.get("turns_ratio", 1.0) > 3.0 or tier1.get("stuck_ratio", 0) > 0.5:
                rating = self._apply_max_rating(rating, "合格")

        # ---- TASK_COMPLETION 硬约束上限 (H.3) ----
        if dimension == "TASK_COMPLETION":
            if tier1.get("user_repeat_rate", 0) > 0.5:
                rating = self._apply_max_rating(rating, "需改进")
            hangup = tier1.get("hangup_detected", {})
            if (hangup.get("detected")
                    and hangup.get("hangup_sentiment") == "负面"
                    and hangup.get("task_progress", 1.0) < 0.5):
                rating = self._apply_max_rating(rating, "需改进")

        return rating

    def _downgrade_rating(self, current: str) -> str:
        """评级降一级"""
        if current == "无法评估":
            return current
        order = ["卓越", "良好", "合格", "需改进", "不合格"]
        idx = order.index(current) if current in order else 2
        return order[min(idx + 1, len(order) - 1)]

    def _apply_max_rating(self, rating: str, max_rating: str) -> str:
        """限制最高评级——若当前评级超过上限则钳制，否则保持原评级"""
        order = ["卓越", "良好", "合格", "需改进", "不合格"]
        if rating not in order or max_rating not in order:
            return rating
        if order.index(rating) < order.index(max_rating):
            return max_rating
        return rating

    # ---- 表面合规检测 ----

    def _detect_surface_compliance(
        self,
        checklists: Dict[str, DimensionChecklist],
        additional_defects: List[Defect],
    ) -> List[Dict[str, Any]]:
        """检测表面合规：Case 项全 YES 但 Simulator 信号负面"""
        flags = []

        for dim, checklist in checklists.items():
            case_items = [i for i in checklist.items if i.source == "case" and i.status != "NOT_APPLICABLE"]
            sim_items = [i for i in checklist.items if i.source == "simulator" and i.status != "NOT_APPLICABLE"]

            if not case_items or not sim_items:
                continue

            case_yes_ratio = sum(1 for i in case_items if i.status in ("YES", "MOSTLY_YES")) / len(case_items)
            sim_no_count = sum(1 for i in sim_items if i.status in ("NO", "MOSTLY_NO", "PARTIAL"))
            # 按维度统计 additional_defects：匹配 description 中是否含维度关键词
            _DIM_CN_MAP = {"安全": "SAFETY", "任务": "TASK_COMPLETION", "流程": "FLOW_COVERAGE",
                           "知识": "KNOWLEDGE", "约束": "CONSTRAINT", "效率": "EFFICIENCY",
                           "情感": "SENTIMENT", "角色": "ROLE", "开场": "OPENING"}
            _dim_cn = {v: k for k, v in _DIM_CN_MAP.items()}
            dim_keyword = _dim_cn.get(dim, "")
            dim_defects = sum(
                1 for d in additional_defects
                if dim_keyword and dim_keyword in getattr(d, "description", "")
            )

            if (case_yes_ratio >= SURFACE_COMPLIANCE["case_yes_threshold"]
                    and sim_no_count >= SURFACE_COMPLIANCE["simulator_no_min"]
                    and dim_defects >= SURFACE_COMPLIANCE["additional_defects_min"]):
                flags.append({
                    "dimension": dim,
                    "reason": f"表面合规——Case 项 {case_yes_ratio:.0%} YES 但 Simulator 信号 {sim_no_count} 项 NO/PARTIAL",
                    "case_yes_ratio": case_yes_ratio,
                    "sim_no_count": sim_no_count,
                })

        return flags

    # ---- 元检查 ----

    def _run_meta_checks(
        self,
        conv: Conversation,
        checklists: Dict[str, DimensionChecklist],
        ratings: Dict[str, str],
        additional_defects: List[Defect] = None,
        case = None,
    ) -> List[MetaCheckAlert]:
        """纯规则元检查：逻辑一致性 + 证据有效性 + 覆盖率 + CONSTRAINT 盲检 + Case 内部一致性"""
        alerts = []

        # 1. 逻辑检查
        if ratings.get("SAFETY") == "不合格" and ratings.get("TASK_COMPLETION") == "卓越":
            alerts.append(MetaCheckAlert(
                check_type="logic", severity="error",
                description="SAFETY=不合格 但 TASK=卓越，逻辑矛盾（安全不合格时任务不应卓越）",
                dimensions=["SAFETY", "TASK_COMPLETION"],
            ))
        if ratings.get("OPENING") == "不合格" and ratings.get("FLOW_COVERAGE") == "卓越":
            alerts.append(MetaCheckAlert(
                check_type="logic", severity="warning",
                description="OPENING=不合格 但 FLOW_COVERAGE=卓越（开场失败但流程完美→可疑）",
                dimensions=["OPENING", "FLOW_COVERAGE"],
            ))
        # K.3 逻辑矛盾对扩展
        if ratings.get("EFFICIENCY") == "不合格" and ratings.get("TASK_COMPLETION") == "卓越":
            alerts.append(MetaCheckAlert(
                check_type="logic", severity="warning",
                description="EFFICIENCY=不合格 但 TASK=卓越（极端低效但任务完美→可疑）",
                dimensions=["EFFICIENCY", "TASK_COMPLETION"],
            ))
        if ratings.get("KNOWLEDGE") == "不合格" and ratings.get("TASK_COMPLETION") in ("卓越", "良好"):
            alerts.append(MetaCheckAlert(
                check_type="logic", severity="warning",
                description="KNOWLEDGE=不合格 但 TASK≥良好（知识全错但任务成功→可疑，可能是简单任务）",
                dimensions=["KNOWLEDGE", "TASK_COMPLETION"],
            ))

        # 2. 覆盖率检查
        for dim, checklist in checklists.items():
            if checklist.applicable_count < CONFIDENCE.get("low_coverage_threshold", 3):
                alerts.append(MetaCheckAlert(
                    check_type="coverage", severity="warning",
                    description=f"{dim} 维度仅有 {checklist.applicable_count} 条 applicable 清单项，覆盖不足",
                    dimensions=[dim],
                ))

        # 3. 证据有效性检查
        max_turn = len(conv.turns) if conv.turns else 0
        for dim, checklist in checklists.items():
            for item in checklist.items:
                if not item.evidence:
                    continue
                turn_nums = re.findall(r'T(\d+)', item.evidence)
                for tn_str in turn_nums:
                    tn = int(tn_str)
                    if tn > max_turn or tn < 1:
                        alerts.append(MetaCheckAlert(
                            check_type="evidence", severity="error",
                            description=f"{dim}.{item.item_id} 引用 T{tn} 但对话最大轮次为 {max_turn}",
                            dimensions=[dim],
                        ))
                        break

        # 5. CONSTRAINT 盲检 (K.4): 约束全正向但 additional_defects 含约束违规关键词
        additional_defects = additional_defects or []
        constraint_checklist = checklists.get("CONSTRAINT")
        if constraint_checklist and constraint_checklist.applicable_count > 0:
            constraint_items = [i for i in constraint_checklist.items if i.status != "NOT_APPLICABLE"]
            all_positive = all(i.status in ("YES", "MOSTLY_YES") for i in constraint_items)
            if all_positive and additional_defects:
                constraint_keywords = ["禁止", "必须", "不得", "不许", "严禁", "只能", "不可",
                                       "违规", "违反", "超出", "限制", "约束", "字数", "格式"]
                suspicious = [
                    d for d in additional_defects
                    if any(kw in getattr(d, "description", "") for kw in constraint_keywords)
                ]
                if suspicious:
                    alerts.append(MetaCheckAlert(
                        check_type="coverage", severity="warning",
                        description=f"CONSTRAINT 全部正向但 {len(suspicious)} 条 additional_defects "
                                    f"含约束关键词——LLM 可能漏检约束违规",
                        dimensions=["CONSTRAINT"],
                    ))

        # 4. 证据内容模糊匹配 (K.1): 验证 evidence 中引号内文本是否实际出现在对话中
        if conv.turns and max_turn > 0:
            turn_contents = {i + 1: t.content for i, t in enumerate(conv.turns) if t.content}
            evidence_alert_count: Dict[str, int] = {}
            for dim, checklist in checklists.items():
                for item in checklist.items:
                    if not item.evidence or item.status == "NOT_APPLICABLE":
                        continue
                    # 提取 evidence 中引号内文本 (中文+英文引号, 5+ 字符)
                    quoted: List[str] = []
                    for p in [r'"([^"]{5,})"', r'"([^"]{5,})"', r"'([^']{5,})'"]:
                        quoted.extend(re.findall(p, item.evidence))
                    if not quoted:
                        continue
                    ref_turn = _extract_first_turn(item.evidence)
                    if ref_turn is None or ref_turn not in turn_contents:
                        continue
                    turn_text = turn_contents[ref_turn]
                    all_matched = True
                    unmatched = []
                    for q in quoted:
                        if q in turn_text or _text_bigram_overlap(q, turn_text) >= 0.5:
                            continue
                        matched_nearby = False
                        for delta in (-1, 1):
                            nb = turn_contents.get(ref_turn + delta, "")
                            if q in nb or _text_bigram_overlap(q, nb) >= 0.5:
                                matched_nearby = True
                                break
                        if not matched_nearby:
                            all_matched = False
                            unmatched.append(q[:30])
                    if not all_matched:
                        dim_key = dim
                        evidence_alert_count[dim_key] = evidence_alert_count.get(dim_key, 0) + 1
                        if evidence_alert_count[dim_key] <= 3:
                            alerts.append(MetaCheckAlert(
                                check_type="evidence", severity="warning",
                                description=f"{dim}.{item.item_id} evidence ref not found in T{ref_turn}"
                                            f" ({'/'.join(unmatched[:2])})",
                                dimensions=[dim],
                            ))

        # 5. Case 内部一致性：约束-流程冲突检测（如 word_limit vs step 信息量）
        if case:
            import re as re2
            call_flow = getattr(case, "call_flow", []) or []
            constraints = getattr(case, "constraints", []) or []
            word_limit = None
            for c in constraints:
                desc = str(getattr(c, "description", ""))
                m = re2.search(r'(\d+)\s*个?\s*字', desc)
                if m:
                    word_limit = int(m.group(1))
                    break

            if word_limit:
                conflicting_steps = []
                for step in call_flow:
                    desc = getattr(step, "description", "") or ""
                    desc_len = len(desc.replace(" ", ""))
                    if desc_len > word_limit * 0.6:
                        conflicting_steps.append(
                            f"Step {getattr(step, 'step_number', '?')}"
                            f"（'{getattr(step, 'title', '')[:20]}'）"
                            f"信息量 {desc_len} 字，占限制 {desc_len/word_limit*100:.0f}%"
                        )

                if conflicting_steps:
                    alerts.append(MetaCheckAlert(
                        check_type="consistency", severity="warning",
                        description=(
                            f"Case 约束冲突：word_limit={word_limit}字，"
                            f"但 {len(conflicting_steps)} 个流程步骤信息量超过限制的 60%："
                            f"{'; '.join(conflicting_steps[:4])}"
                        ),
                        dimensions=["FLOW_COVERAGE"],
                    ))

        return alerts

    # ---- EvalConfidence ----

    def _compute_confidence(
        self,
        checklists: Dict[str, DimensionChecklist],
        signals: Dict[str, Any],
        tier1: Dict[str, Any],
        ratings: Dict[str, str],
        attributions: List[AttributionItem],
        conv: Conversation = None,
        meta_alerts: List[MetaCheckAlert] = None,
        cross_alerts: List[CrossValidationAlert] = None,
        complexity: float = 5.0,
        dim_anchors: Dict[str, str] = None,
    ) -> EvalConfidence:
        """汇总多类输入 → EvalConfidence（含丰富因子 + 理由输出）"""
        meta_alerts = meta_alerts or []
        cross_alerts = cross_alerts or []
        dim_anchors = dim_anchors or {}
        # 清单-信号一致性
        total_signal_items = 0
        conflict_count = 0
        per_dim_conflict: Dict[str, str] = {}

        for dim, checklist in checklists.items():
            sim_items = [i for i in checklist.items if i.source == "simulator" and i.status != "NOT_APPLICABLE"]
            for item in sim_items:
                total_signal_items += 1
                if item.signal_consistency == "矛盾":
                    conflict_count += 1
            per_dim_conflict[dim] = "矛盾" if any(
                i.signal_consistency == "矛盾" for i in sim_items
            ) else "一致"

        conflict_ratio = conflict_count / max(total_signal_items, 1)

        # 证据质量（排除 NOT_APPLICABLE — 它们预期无 evidence）
        applicable_items = []
        for checklist in checklists.values():
            applicable_items.extend([i for i in checklist.items if i.status != "NOT_APPLICABLE"])
        evidence_empty = sum(1 for i in applicable_items if not i.evidence)
        evidence_empty_ratio = evidence_empty / max(len(applicable_items), 1)

        # 证据阶段覆盖
        avg_coverage = self._estimate_evidence_coverage(applicable_items)

        # Simulator 质量（V2-V4 消费：优先从 conv.consistency 读取 profile_auditor 输出）
        sim_tier = "green"
        if conv and conv.consistency:
            sim_tier = conv.consistency.get("tier", "green")
        elif tier1.get("consistency", {}).get("tier"):
            sim_tier = tier1.get("consistency", {}).get("tier", "green")
        if isinstance(sim_tier, str):
            sim_tier_str = sim_tier
        else:
            sim_tier_str = "green"
        signal_weight = {
            "green": CONFIDENCE["signal_weight_green"],
            "yellow": CONFIDENCE["signal_weight_yellow"],
            "red": CONFIDENCE["signal_weight_red"],
        }.get(sim_tier_str, 1.0)

        # Judge 间一致性
        cross_anomalies = self._check_cross_judge_consistency(ratings)
        cross_pairs = len(cross_anomalies)

        # 子维度异常
        sub_anomalies = self._check_sub_consistency(checklists)

        # 逐维度计算
        per_dim = {}
        for dim, checklist in checklists.items():
            dim_conf = CONFIDENCE["base_dim"]

            # 信号冲突（排除 NOT_APPLICABLE — 它们永不会矛盾）
            dim_sim_items = [i for i in checklist.items if i.source == "simulator" and i.status != "NOT_APPLICABLE"]
            dim_conflict_ratio = sum(
                1 for i in dim_sim_items if i.signal_consistency == "矛盾"
            ) / max(len(dim_sim_items), 1)

            if dim_conflict_ratio > 0.3:
                dim_conf -= CONFIDENCE["signal_conflict_penalty_high"]
            elif dim_conflict_ratio > 0.1:
                dim_conf -= CONFIDENCE["signal_conflict_penalty_low"]

            # evidence 质量（排除 NOT_APPLICABLE）
            dim_applicable = [i for i in checklist.items if i.status != "NOT_APPLICABLE"]
            dim_empty = sum(1 for i in dim_applicable if not i.evidence)
            dim_empty_ratio = dim_empty / max(len(dim_applicable), 1)
            # 稀疏维度（applicable < 3）天然 evidence 少，惩罚减半
            sparse_dim = len(dim_applicable) < 3
            evidence_penalty = CONFIDENCE["evidence_empty_penalty"] * (0.5 if sparse_dim else 1.0)
            if dim_empty_ratio > 0.4:
                dim_conf -= evidence_penalty

            dim_coverage = self._estimate_evidence_coverage(dim_applicable)
            if dim_coverage >= 2.5:
                dim_conf += CONFIDENCE["evidence_coverage_bonus"]

            # Case vs Simulator YES 占比差距（仅双方均有数据时比较）
            case_items = [i for i in checklist.items if i.source == "case" and i.status != "NOT_APPLICABLE"]
            sim_items = [i for i in checklist.items if i.source == "simulator" and i.status != "NOT_APPLICABLE"]
            if case_items and sim_items:
                case_ratio = checklist.source_ratio("case")
                sim_ratio = checklist.source_ratio("simulator")
                if abs(case_ratio - sim_ratio) > 0.5 and checklist.applicable_count >= 3:
                    dim_conf -= 0.05

            # 新增：清单项数稳定性（项数越少方差越大）
            n_applicable = checklist.applicable_count
            if n_applicable < CONFIDENCE.get("low_item_count_threshold", 5):
                dim_conf -= CONFIDENCE.get("low_item_count_penalty", 0.08)
            elif n_applicable < 8:
                dim_conf -= CONFIDENCE.get("medium_item_count_penalty", 0.03)

            # 新增：PARTIAL 浓度（PARTIAL 多说明清单描述可能不清晰）
            partial_count = sum(1 for i in dim_applicable if i.status == "PARTIAL")
            partial_ratio = partial_count / max(n_applicable, 1)
            if partial_ratio > CONFIDENCE.get("partial_concentration_threshold", 0.30):
                dim_conf -= CONFIDENCE.get("partial_concentration_penalty", 0.06)

            # anchor_alignment 交叉校验 (F.1): LLM 整体判断 vs 规则推导评级
            anchor = dim_anchors.get(dim, "")
            if anchor and dim in ratings:
                anchor_score_map = {"卓越": 9.5, "良好": 7.5, "合格": 5.5, "需改进": 3.5, "不合格": 1.0}
                rating_score_map = {"卓越": 9.5, "良好": 7.5, "合格": 5.5, "需改进": 3.5, "不合格": 1.0}
                anchor_score = anchor_score_map.get(anchor, 5.5)
                rating_score = rating_score_map.get(ratings[dim], 5.5)
                if abs(anchor_score - rating_score) >= 2.0:
                    dim_conf -= 0.05

            dim_conf = max(CONFIDENCE["cap_min"], min(CONFIDENCE["cap_max"], dim_conf))
            per_dim[dim] = dim_conf

        # 新增：对话长度因子（短对话天然更不可靠）
        n_turns = conv.total_turns if conv else 0
        if 0 < n_turns < CONFIDENCE.get("short_conv_threshold", 5):
            for dim in per_dim:
                per_dim[dim] -= CONFIDENCE.get("short_conv_penalty", 0.10)
        elif 0 < n_turns < 8:
            for dim in per_dim:
                per_dim[dim] -= CONFIDENCE.get("medium_conv_penalty", 0.05)

        # 新增：Judge temperature 因子
        judge_temperature = getattr(getattr(self, 'client', None), 'temperature', 0.3)
        if judge_temperature > 0.5:
            for dim in per_dim:
                per_dim[dim] -= CONFIDENCE.get("high_temperature_penalty", 0.03)

        # 新增：V3 audited_vector 消费 — 极端画像检测
        extreme_dims: List[str] = []
        extreme_source = "none"
        audited = getattr(conv, "audited_vector", None) if conv else None
        sampled = getattr(conv, "sampled_vector", None) if conv else None
        vector = audited or sampled
        if vector and len(vector) >= 15:
            extreme_source = "audited" if audited else "sampled_unverified"
            dim_names = [
                "agreeableness", "patience", "neuroticism", "conscientiousness",
                "openness", "politeness", "extraversion", "verbosity",
                "assertiveness", "information_verification", "urgency",
                "initial_mood", "mood_volatility", "boundary_testing", "truth_consistency",
            ]
            for i, val in enumerate(vector[:15]):
                if val > CONFIDENCE.get("extreme_profile_threshold_high", 0.9) or \
                   val < CONFIDENCE.get("extreme_profile_threshold_low", 0.1):
                    extreme_dims.append(f"{dim_names[i]}({val:.2f})")
            # 极端画像降低置信度（Simulator 偏差可能影响评测客观性）
            if extreme_dims:
                max_dims = CONFIDENCE.get("extreme_profile_max_dims", 5)
                for dim in per_dim:
                    per_dim[dim] -= 0.02 * min(len(extreme_dims), max_dims)
                if extreme_source == "sampled_unverified":
                    for dim in per_dim:
                        per_dim[dim] -= 0.01  # 未校验的向量额外扣分

        # 新增：高复杂度 Case 置信度微调（参照点多 → 置信度应更高）
        if complexity >= 8.0:
            for dim in per_dim:
                per_dim[dim] += 0.02  # 复杂 Case 参照点多，结果更可信

        # 总置信度
        overall = sum(per_dim.values()) / max(len(per_dim), 1)
        # Judge 间一致性惩罚
        overall -= cross_pairs * CONFIDENCE["cross_judge_penalty_per_pair"]
        # V5: 三源融合置信度（state 标签质量）
        from src.eval.rules import compute_v5_state_confidence
        v5_state_conf = {}
        if conv:
            v5_state_conf = compute_v5_state_confidence(conv)
            if v5_state_conf.get("avg_confidence", 1.0) < 0.7:
                overall -= 0.05  # state 标签整体质量不高
        # 元检查扣分（加天花板防止大量 warning 叠加导致置信度全面偏低）
        meta_penalty = 0.0
        for alert in meta_alerts:
            if alert.severity == "error":
                meta_penalty += CONFIDENCE.get("meta_check_error_penalty", 0.02)
            elif alert.severity == "warning":
                meta_penalty += CONFIDENCE.get("meta_check_warning_penalty", 0.01)
        overall -= min(meta_penalty, CONFIDENCE.get("meta_check_max_total_penalty", 0.08))
        # 交叉验证扣分（SAFETY/TASK 相关 high 告警使用更严厉惩罚 I.1）
        for alert in cross_alerts:
            if alert.severity == "high":
                if alert.dimension in ("SAFETY", "TASK_COMPLETION"):
                    overall -= CONFIDENCE.get("cross_validation_safety_task_penalty", 0.10)
                else:
                    overall -= CONFIDENCE.get("cross_validation_high_penalty", 0.05)
            elif alert.severity == "medium":
                overall -= CONFIDENCE.get("cross_validation_medium_penalty", 0.03)
        # 正向激励: 证据质量高（空证据率低 + 覆盖广）→ 置信度提升
        if evidence_empty_ratio < 0.1 and avg_coverage >= 2.5:
            overall += 0.03
        # 正向激励: 多维度交叉验证一致（无法官间矛盾 + 无子维度异常）→ 置信度提升
        if cross_pairs == 0 and len(sub_anomalies) == 0:
            overall += 0.02
        # Simulator 信号权重在全部扣分后应用，避免惩罚不对称
        overall *= signal_weight
        overall = max(CONFIDENCE["cap_min"], min(CONFIDENCE["cap_max"], overall))

        # Level
        if overall >= CONFIDENCE["level_threshold_high"]:
            level = "high"
        elif overall >= CONFIDENCE["level_threshold_medium"]:
            level = "medium"
        elif overall >= CONFIDENCE["level_threshold_low"]:
            level = "low"
        else:
            level = "unreliable"

        # 归因置信度
        attr_conf = sum(a.confidence for a in attributions) / max(len(attributions), 1)

        # 构建置信度理由
        reason_parts = []
        per_dim_reasons: Dict[str, str] = {}
        if conflict_ratio > 0.3:
            reason_parts.append(f"信号矛盾率高({conflict_ratio:.0%})")
        if evidence_empty_ratio > 0.2:
            reason_parts.append(f"证据缺失率高({evidence_empty_ratio:.0%})")
        if cross_pairs > 0:
            reason_parts.append(f"Judge间不一致({cross_pairs}对)")
        if n_turns > 0 and n_turns < 5:
            reason_parts.append(f"对话过短({n_turns}轮)")
        for dim in per_dim:
            dim_reasons = []
            if per_dim_conflict.get(dim) == "矛盾":
                dim_reasons.append("信号矛盾")
            if per_dim[dim] < 0.5:
                dim_reasons.append("置信度低")
            per_dim_reasons[dim] = "; ".join(dim_reasons) if dim_reasons else "正常"

        # needs_human_review (I.2): 综合 level + signal_conflict + cross_alerts + meta_alerts
        needs_review = (
            level in ("low", "unreliable")
            or conflict_count >= 3
            or any(a.severity == "high" for a in cross_alerts)
            or any(a.severity == "error" for a in meta_alerts)
        )

        return EvalConfidence(
            overall=overall,
            level=level,
            checklist_signal_consistency=per_dim_conflict,
            signal_conflict_count=conflict_count,
            evidence_empty_ratio=evidence_empty_ratio,
            avg_evidence_coverage=avg_coverage,
            simulator_tier=sim_tier_str,
            signal_weight_factor=signal_weight,
            cross_judge_anomalies=cross_anomalies,
            cross_judge_anomaly_pairs=cross_pairs,
            sub_dimension_anomalies=sub_anomalies,
            per_dimension=per_dim,
            attribution_confidence=attr_conf,
            parse_success=len(self.runner.parse_failures) == 0,
            extreme_profile_dims=extreme_dims,
            extreme_profile_flag=len(extreme_dims) > 0,
            extreme_profile_source=extreme_source,
            confidence_reasoning="; ".join(reason_parts) if reason_parts else "各项指标正常",
            per_dimension_reasoning=per_dim_reasons,
            needs_human_review=needs_review,
        )

    def _estimate_evidence_coverage(self, items: List[CheckResult]) -> float:
        """估算证据覆盖对话前中后段的程度（0-3）

        使用正则精确提取轮次号，避免 \"T1\" 子串匹配 T10~T19 等误判。
        """
        turns = set()
        for i in items:
            for m in re.finditer(r'T(\d+)', i.evidence or ""):
                turns.add(int(m.group(1)))
        early = any(1 <= t <= 2 for t in turns)
        mid = any(3 <= t <= 7 for t in turns)
        late = any(t >= 8 for t in turns)
        return float(early) + float(mid) + float(late)

    def _check_cross_judge_consistency(self, ratings: Dict[str, str]) -> List[str]:
        """检查 Judge 对一致性"""
        anomalies = []
        rating_order = {"卓越": 4, "良好": 3, "合格": 2, "需改进": 1, "不合格": 0, "无法评估": -1}

        for d1, d2 in CROSS_JUDGE_PAIRS:
            r1 = ratings.get(d1, "合格")
            r2 = ratings.get(d2, "合格")
            diff = abs(rating_order.get(r1, 2) - rating_order.get(r2, 2))
            if diff >= 2:
                anomalies.append(f"{d1}({r1})↔{d2}({r2})差{diff}级")

        return anomalies

    def _check_sub_consistency(
        self,
        checklists: Dict[str, DimensionChecklist],
    ) -> List[str]:
        """检查子维度内部一致性（Case vs Simulator YES 占比差距）

        仅双方均有数据时才比较，避免纯 Case 维度（OPENING/CONSTRAINT）或纯 Simulator 维度误报。
        """
        anomalies = []
        for dim, checklist in checklists.items():
            case_items = [i for i in checklist.items if i.source == "case" and i.status != "NOT_APPLICABLE"]
            sim_items = [i for i in checklist.items if i.source == "simulator" and i.status != "NOT_APPLICABLE"]
            if not case_items or not sim_items:
                continue
            case_ratio = checklist.source_ratio("case")
            sim_ratio = checklist.source_ratio("simulator")
            if abs(case_ratio - sim_ratio) > 0.5 and checklist.applicable_count >= 3:
                anomalies.append(
                    f"{dim}: Case={case_ratio:.0%} vs Sim={sim_ratio:.0%}"
                )
        return anomalies

    # ---- 汇总输出 ----

    def _build_summary(
        self,
        ratings: Dict[str, str],
        confidence: EvalConfidence,
        surface_flags: List[Dict[str, Any]],
        tier1_cons_count: int = 0,
        llm_cons_count: int = 0,
    ) -> str:
        """构建可读摘要"""
        lines = ["=== 评估摘要 ==="]
        lines.append(f"评级: {ratings}")
        if surface_flags:
            lines.append(f"表面合规: {[f['reason'] for f in surface_flags]}")
        total_cons = tier1_cons_count + llm_cons_count
        if total_cons > 0:
            rule_pct = tier1_cons_count / total_cons * 100
            lines.append(f"CONSTRAINT 分流: {tier1_cons_count} 条规则检测 + {llm_cons_count} 条 LLM 核查 "
                        f"（规则分流 {rule_pct:.0f}%，节省 {llm_cons_count} 次 LLM 约束判断）")
        lines.append(f"可信度: {confidence.level} ({confidence.overall:.2f})")
        lines.append(f"is_reliable: {confidence.is_reliable}")
        return "\n".join(lines)

    def _build_suggestions(self, attributions: List[AttributionItem]) -> List[str]:
        """从归因提取改进建议"""
        suggestions = []
        for a in attributions:
            if a.is_actionable:
                for action in a.suggested_actions:
                    if action not in suggestions:
                        suggestions.append(action)
        return suggestions[:10]

    def _build_optimization_feed(
        self,
        attributions: List[AttributionItem],
        confidence: EvalConfidence,
        conv_id: str,
    ) -> OptimizationFeed:
        """构建对接优化引擎的标准化输出"""
        from datetime import datetime

        return OptimizationFeed(
            batch_id=conv_id,
            generated_at=datetime.now().isoformat(),
            summary={
                "n_attributions": len(attributions),
                "model_attributions": sum(1 for a in attributions if a.source == "model"),
                "case_attributions": sum(1 for a in attributions if a.source == "case"),
                "simulator_attributions": sum(1 for a in attributions if a.source == "simulator"),
            },
            attributions=attributions,
            eval_confidence={
                "overall": confidence.overall,
                "level": confidence.level,
                "is_reliable": confidence.is_reliable,
                "needs_human_review": confidence.needs_human_review,
                "per_dimension": confidence.per_dimension,
            },
        )
