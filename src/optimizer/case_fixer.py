"""Case 设计完善器 — 规则分类 + LLM 生成修改建议

输入: case.json + 评测归因 (source=case) + 多对话评分分布
输出: Case 定义的具体修改建议
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.llm.client import LLMClient
from src.optimizer.optimizer import DefectCluster, OptimizationAction
from src.optimizer.prompts import CASE_FIX_SYSTEM, CASE_FIX_USER, DIMENSION_CN
from src.optimizer.utils import classify_severity


class CaseFixer:
    """Case 设计完善器（规则分类 + LLM 生成）"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_fixes(
        self,
        clusters: List[DefectCluster],
        case_path: Path,
        score_distribution: Optional[Dict] = None,
    ) -> List[OptimizationAction]:
        """为 Case 缺陷聚类生成优化建议。

        规则引擎先分类问题类型（六种），LLM 再进行具体文本生成。
        """
        case_data = json.loads(case_path.read_text(encoding="utf-8")) if case_path.exists() else {}
        actions = []

        for cluster in clusters:
            if cluster.source != "case":
                continue

            # 规则分类
            issue_type = self._classify_issue(cluster)

            if issue_type in ("overly_strict", "missing_branch"):
                # 规则可处理
                action = self._rule_based_fix(cluster, issue_type, case_data)
                if action:
                    actions.append(action)
            else:
                # LLM 生成
                action = self._llm_based_fix(cluster, issue_type, case_data, score_distribution)
                if action:
                    actions.append(action)

        return actions

    def _classify_issue(self, cluster: DefectCluster) -> str:
        """规则引擎分类 Case 问题类型。"""
        pattern = cluster.defect_pattern.lower()
        if any(kw in pattern for kw in ["过严", "complexity", "过高", "过多"]):
            return "overly_strict"
        if any(kw in pattern for kw in ["缺失", "未定义", "branch", "分支"]):
            return "missing_branch"
        if any(kw in pattern for kw in ["不清晰", "模糊", "ambiguous"]):
            return "unclear_description"
        if any(kw in pattern for kw in ["约束", "constraint"]):
            return "constraint_issue"
        if any(kw in pattern for kw in ["知识点", "knowledge", "kp"]):
            return "missing_knowledge"
        return "flow_issue"

    def _rule_based_fix(
        self, cluster: DefectCluster, issue_type: str, case_data: Dict,
    ) -> Optional[OptimizationAction]:
        """规则驱动的 Case 修改建议（零 LLM 成本）。"""
        if issue_type == "overly_strict":
            return OptimizationAction(
                action_id=f"case-rule-{cluster.cluster_id}",
                source="case",
                dimension=cluster.dimension,
                priority=cluster.frequency * 1.2,
                title=f"降低 {cluster.dimension} 维度的 Case 复杂度要求",
                constitutional_principle=f"Case 设计合理性——{cluster.dimension}",
                violation_evidence=cluster.evidence_samples[:3],
                critique_analysis=(
                    f"{cluster.dimension} 维度在 {len(cluster.conversation_ids)} 场对话中"
                    f"累计 {cluster.frequency} 项缺陷被归因为 Case 设计问题。"
                    f"可能是流程步骤/约束定义过严。"
                ),
                revision_proposal=(
                    f"1. 检查 {cluster.dimension} 相关的 call_flow 步骤是否过多/过细\n"
                    f"2. 考虑将部分步骤标记为 optional\n"
                    f"3. 考虑降低该维度的约束数量或放宽阈值"
                ),
                target_location=f"Case call_flow / constraints（{cluster.dimension} 维度）",
                expected_impact=f"预计可减少 {cluster.frequency} 项 Case 归因缺陷",
                effort_estimate="低",
                level="中建议",
                path="A",
            )

        if issue_type == "missing_branch":
            return OptimizationAction(
                action_id=f"case-branch-{cluster.cluster_id}",
                source="case",
                dimension=cluster.dimension,
                priority=cluster.frequency * 1.0,
                title=f"Case 流程可能缺少分支定义——{cluster.dimension}",
                constitutional_principle="Case call_flow 分支完整性",
                violation_evidence=cluster.evidence_samples[:3],
                critique_analysis=(
                    f"{cluster.dimension} 维度的分支相关清单项持续 NO，"
                    f"可能是 call_flow 中缺少对应的 branching 定义。"
                ),
                revision_proposal=(
                    f"1. 检查 {cluster.dimension} 相关步骤的 branching 定义\n"
                    f"2. 补充缺失的分支条件和处理动作\n"
                    f"3. 确保分支覆盖所有常见用户行为"
                ),
                target_location="Case call_flow branching",
                expected_impact=f"预计可修正 {cluster.frequency} 项分支缺失",
                effort_estimate="中",
                level="中建议",
                path="A",
            )

        return None

    def _llm_based_fix(
        self, cluster: DefectCluster, issue_type: str,
        case_data: Dict, score_distribution: Optional[Dict],
    ) -> Optional[OptimizationAction]:
        """LLM 生成 Case 修改建议。"""
        # 构建规则发现上下文
        rule_findings = (
            f"问题类型: {issue_type}\n"
            f"出现频率: {cluster.frequency}\n"
            f"涉及对话: {len(cluster.conversation_ids)} 场\n"
            f"严重程度: {cluster.severity}\n"
            f"评分分布: {score_distribution or '无'}"
        )

        try:
            response = self.llm.chat(
                CASE_FIX_SYSTEM,
                CASE_FIX_USER.format(
                    case_attributions="\n".join(
                        f"- [{g.dimension}] {g.description}"
                        for g in cluster.gradients[:5]
                    ),
                    case_definition=json.dumps(case_data, ensure_ascii=False, indent=2),
                    score_distribution=str(score_distribution or {}),
                    rule_findings=rule_findings,
                ),
            )

            return OptimizationAction(
                action_id=f"case-llm-{cluster.cluster_id}",
                source="case",
                dimension=cluster.dimension,
                priority=cluster.frequency * 0.9,
                title=f"Case 修改: {cluster.defect_pattern[:40]}",
                constitutional_principle=f"Case 定义——{cluster.dimension}",
                violation_evidence=cluster.evidence_samples[:2],
                critique_analysis=f"类型: {issue_type}。{cluster.defect_pattern[:100]}",
                revision_proposal=response[:1000],
                target_location=f"Case 定义（{cluster.dimension} 相关段）",
                expected_impact=f"预期改善 {cluster.dimension} 维度评分准确性",
                effort_estimate="中",
                level="中建议",
                path="B",
            )
        except Exception:
            return None
