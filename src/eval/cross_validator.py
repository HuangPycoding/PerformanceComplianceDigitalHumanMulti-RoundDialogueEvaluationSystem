"""RuleLLMCrossValidator — 规则-LLM 交叉验证：检测 Tier 1 规则结果与 LLM 清单核查的矛盾"""
from typing import Any, Dict, List

from src.models.evaluation import CrossValidationAlert, DimensionChecklist


class RuleLLMCrossValidator:
    """规则结果与 LLM 核查结果的交叉验证（纯规则，零 LLM 成本）"""

    def validate(
        self,
        tier1: Dict[str, Any],
        checklists: Dict[str, DimensionChecklist],
        ratings: Dict[str, str],
    ) -> List[CrossValidationAlert]:
        alerts = []

        # 1. 禁止词矛盾
        alerts.extend(self._check_forbidden_word_contradiction(tier1, checklists))

        # 2. 用户重复率矛盾
        alerts.extend(self._check_user_repeat_contradiction(tier1, checklists, ratings))

        # 3. 卡死矛盾
        alerts.extend(self._check_stuck_contradiction(tier1, checklists))

        # 4. 挂断矛盾
        alerts.extend(self._check_hangup_contradiction(tier1, checklists, ratings))

        # 5. model_breakdown 矛盾 (I.3)
        alerts.extend(self._check_model_breakdown_contradiction(tier1, ratings))

        # 6. turns_ratio vs EFFICIENCY 矛盾 (I.3)
        alerts.extend(self._check_turns_ratio_contradiction(tier1, ratings))

        # 7. LLM 整体偏宽检测 (I.4)
        alerts.extend(self._check_llm_overall_leniency(tier1, checklists, ratings))

        return alerts

    def _check_forbidden_word_contradiction(
        self, tier1: Dict, checklists: Dict[str, DimensionChecklist]
    ) -> List[CrossValidationAlert]:
        alerts = []
        fh = tier1.get("forbidden_word_hits", 0)
        if fh <= 0:
            return alerts

        constraint_cl = checklists.get("CONSTRAINT")
        if not constraint_cl:
            return alerts

        # 检查 CONSTRAINT 维度是否有 NO/MOSTLY_NO/PARTIAL 项（不限制来源）
        negative = sum(1 for i in constraint_cl.items
                       if i.status in ("NO", "MOSTLY_NO", "PARTIAL"))
        if negative == 0:
            alerts.append(CrossValidationAlert(
                dimension="CONSTRAINT",
                rule_finding=f"规则检测到 {fh} 次禁止词命中",
                llm_finding="CONSTRAINT 所有清单项均为正向",
                description=f"规则检测到 {fh} 次禁止词，但 LLM 未在 CONSTRAINT 维度检出问题——LLM 可能遗漏",
                severity="high",
            ))
        return alerts

    def _check_user_repeat_contradiction(
        self, tier1: Dict, checklists: Dict[str, DimensionChecklist], ratings: Dict[str, str]
    ) -> List[CrossValidationAlert]:
        alerts = []
        rr = tier1.get("user_repeat_rate", 0)
        if rr <= 0.3:
            return alerts

        task_cl = checklists.get("TASK_COMPLETION")
        if not task_cl:
            return alerts

        # 用户重复率高说明客服未有效解决问题，但 TASK 判了正向
        task_rating = ratings.get("TASK_COMPLETION", "")
        if task_rating in ("卓越", "良好"):
            alerts.append(CrossValidationAlert(
                dimension="TASK_COMPLETION",
                rule_finding=f"user_repeat_rate={rr:.0%}（用户多次重复表述）",
                llm_finding=f"TASK_COMPLETION 评级为 '{task_rating}'",
                description=f"用户重复率高达 {rr:.0%}（说明客服未有效解决问题），但 TASK 评级为'{task_rating}'——可能高估",
                severity="high",
            ))
        return alerts

    def _check_stuck_contradiction(
        self, tier1: Dict, checklists: Dict[str, DimensionChecklist]
    ) -> List[CrossValidationAlert]:
        alerts = []
        sc = tier1.get("stuck_count", 0)
        if sc <= 0:
            return alerts

        eff_cl = checklists.get("EFFICIENCY")
        if not eff_cl:
            return alerts

        negative = sum(1 for i in eff_cl.items
                       if i.status in ("NO", "MOSTLY_NO", "PARTIAL")
                       and "stuck" in i.item_id.lower())
        if negative == 0:
            alerts.append(CrossValidationAlert(
                dimension="EFFICIENCY",
                rule_finding=f"规则检测到 {sc} 轮卡死",
                llm_finding="EFFICIENCY 卡死相关清单项均为正向",
                description=f"规则检测到 {sc} 轮卡死，但 LLM 未在 EFFICIENCY 卡死项检出问题——LLM 可能遗漏",
                severity="medium",
            ))
        return alerts

    def _check_hangup_contradiction(
        self, tier1: Dict, checklists: Dict[str, DimensionChecklist], ratings: Dict[str, str]
    ) -> List[CrossValidationAlert]:
        alerts = []
        hangup = tier1.get("hangup_detected", {})
        if not hangup.get("detected"):
            return alerts

        sentiment = hangup.get("hangup_sentiment", "")
        if sentiment != "负面":
            return alerts

        task_rating = ratings.get("TASK_COMPLETION", "")
        if task_rating in ("卓越", "良好"):
            alerts.append(CrossValidationAlert(
                dimension="TASK_COMPLETION",
                rule_finding=f"第{hangup.get('hangup_turn', '?')}轮用户负面情绪挂断（进度≈{hangup.get('task_progress', 0):.0%}）",
                llm_finding=f"TASK_COMPLETION 评级为 '{task_rating}'",
                description=f"用户负面情绪挂断但 TASK 评级为'{task_rating}'——评分可能被高估",
                severity="high",
            ))
        return alerts

    # ---- I.3 新增检测 ----

    def _check_model_breakdown_contradiction(
        self, tier1: Dict, ratings: Dict[str, str]
    ) -> List[CrossValidationAlert]:
        """检测 5: model_breakdown_flag=True 但相关维度均未检出异常"""
        alerts = []
        if not tier1.get("model_breakdown_flag", False):
            return alerts

        related_dims = ["SAFETY", "TASK_COMPLETION", "KNOWLEDGE"]
        all_high = all(
            ratings.get(d, "") in ("卓越", "良好")
            for d in related_dims
        )
        if all_high:
            alerts.append(CrossValidationAlert(
                dimension=",".join(related_dims),
                rule_finding="model_breakdown_flag=True（模型可能崩溃）",
                llm_finding=f"SAFETY={ratings.get('SAFETY')} TASK={ratings.get('TASK_COMPLETION')} "
                           f"KNOWLEDGE={ratings.get('KNOWLEDGE')}",
                description="规则检测到模型崩溃信号，但 SAFETY/TASK/KNOWLEDGE 三维度评级均≥良好——"
                           "LLM 可能遗漏崩溃导致的隐性缺陷",
                severity="high",
            ))
        return alerts

    def _check_turns_ratio_contradiction(
        self, tier1: Dict, ratings: Dict[str, str]
    ) -> List[CrossValidationAlert]:
        """检测 6: turns_ratio > 2.0 但 EFFICIENCY 评级为卓越/良好"""
        alerts = []
        tr = tier1.get("turns_ratio", 1.0)
        if tr <= 2.0:
            return alerts

        eff_rating = ratings.get("EFFICIENCY", "")
        if eff_rating in ("卓越", "良好"):
            alerts.append(CrossValidationAlert(
                dimension="EFFICIENCY",
                rule_finding=f"turns_ratio={tr:.1f}x（实际轮次远超预期）",
                llm_finding=f"EFFICIENCY 评级为 '{eff_rating}'",
                description=f"实际轮次是预期的 {tr:.1f} 倍，但 EFFICIENCY 评级为'{eff_rating}'——"
                           "LLM 可能对效率过于宽松",
                severity="medium",
            ))
        return alerts

    # ---- I.4 新增检测 ----

    def _check_llm_overall_leniency(
        self, tier1: Dict, checklists: Dict[str, DimensionChecklist], ratings: Dict[str, str]
    ) -> List[CrossValidationAlert]:
        """检测 7: ≥2 个 Tier1 指标同时触发但对应维度 LLM 均未检出异常 → 整体偏宽"""
        alerts = []
        trigger_count = 0
        triggered_dims: List[str] = []

        # 检查各 Tier1 指标触发情况
        if tier1.get("forbidden_word_hits", 0) > 0:
            cl = checklists.get("CONSTRAINT")
            if cl and not any(i.status in ("NO", "MOSTLY_NO", "PARTIAL") for i in cl.items):
                trigger_count += 1
                triggered_dims.append("CONSTRAINT")

        if tier1.get("user_repeat_rate", 0) > 0.3:
            if ratings.get("TASK_COMPLETION", "") in ("卓越", "良好"):
                trigger_count += 1
                triggered_dims.append("TASK_COMPLETION")

        if tier1.get("stuck_count", 0) > 0:
            cl = checklists.get("EFFICIENCY")
            if cl and not any(i.status in ("NO", "MOSTLY_NO", "PARTIAL")
                             and "stuck" in i.item_id for i in cl.items):
                trigger_count += 1
                triggered_dims.append("EFFICIENCY")

        hangup = tier1.get("hangup_detected", {})
        if hangup.get("detected") and hangup.get("hangup_sentiment") == "负面":
            if ratings.get("TASK_COMPLETION", "") in ("卓越", "良好"):
                trigger_count += 1
                if "TASK_COMPLETION" not in triggered_dims:
                    triggered_dims.append("TASK_COMPLETION")

        if tier1.get("model_breakdown_flag", False):
            if ratings.get("SAFETY", "") not in ("需改进", "不合格") and \
               ratings.get("TASK_COMPLETION", "") not in ("需改进", "不合格"):
                trigger_count += 1
                triggered_dims.append("SAFETY/TASK")

        if trigger_count >= 2:
            alerts.append(CrossValidationAlert(
                dimension=",".join(triggered_dims[:3]),
                rule_finding=f"{trigger_count} 个 Tier1 指标同时触发",
                llm_finding="对应维度 LLM 均未检出异常",
                description=f"Tier1 有 {trigger_count} 个指标触发（{', '.join(triggered_dims[:3])}），"
                           "但 LLM 在对应维度均未检出异常——LLM 可能存在整体偏宽倾向",
                severity="medium",
            ))
        return alerts
