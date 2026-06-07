"""对话模型 Prompt 优化器 — LLM 批量生成候选修改方案

输入: case.json (prompt 结构) + 评测归因 (source=model) + 对话文本
输出: 具体的 prompt 修改建议 (diff 形式)
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.client import LLMClient
from src.optimizer.optimizer import DefectCluster, OptimizationAction
from src.optimizer.prompts import (
    DIMENSION_CN,
    PROMPT_OPTIMIZE_SYSTEM,
    PROMPT_OPTIMIZE_USER,
)
from src.optimizer.utils import (
    DIM_TO_PROMPT_SECTIONS,
    extract_prompt_section,
    get_prompt_sections_for_dimension,
)


class PromptOptimizer:
    """被评测对话模型 Prompt 优化器（路径 B，LLM 驱动）"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self._trajectory: List[str] = []  # OPRO 优化轨迹
        self._conv_context_text: str = ""  # 当前对话上下文（用于证据回写）

    def generate_candidates(
        self,
        clusters: List[DefectCluster],
        case_path: Path,
        conv_dir: Path,
        n_candidates: int = 5,
        path_a_context: str = "",
    ) -> List[OptimizationAction]:
        """DSPy 编译式：为每个 Model 缺陷聚类生成 N 条候选 prompt 修改。

        Args:
            clusters: Model 归因的缺陷聚类列表
            case_path: case.json 路径
            conv_dir: conversation_*.json 所在目录
            n_candidates: 每个聚类生成的候选方案数
        """
        # 加载 Case 定义
        case_data = json.loads(case_path.read_text(encoding="utf-8")) if case_path.exists() else {}
        current_prompt = self._build_current_prompt(case_data)

        actions = []
        for cluster in clusters:
            if cluster.source != "model":
                continue

            # 获取对话上下文
            conv_context = self._get_conversation_context(
                cluster, conv_dir,
            )

            # 构建 prompt
            system = PROMPT_OPTIMIZE_SYSTEM.replace("{n_candidates}", str(n_candidates))
            user = PROMPT_OPTIMIZE_USER.format(
                optimization_trajectory=self._build_trajectory(),
                dimension_cn=DIMENSION_CN.get(cluster.dimension, cluster.dimension),
                dimension=cluster.dimension,
                defect_pattern=cluster.defect_pattern,
                frequency=cluster.frequency,
                conversation_count=len(cluster.conversation_ids),
                severity=cluster.severity,
                evidence_samples="\n".join(f"- {e}" for e in cluster.evidence_samples[:3]),
                path_a_context=path_a_context,
                current_prompt=current_prompt,
                conversation_context=conv_context,
                n_candidates=n_candidates,
            )

            try:
                response = self.llm.chat(system, user)
                parsed = self._parse_batch_response(response, cluster)
                actions.extend(parsed)
                # 更新轨迹
                for a in parsed:
                    self._trajectory.append(
                        f"[得分{a.priority:.1f}] 修改 {a.target_location}: "
                        f"{a.revision_proposal[:100]}..."
                    )
            except Exception as e:
                # LLM 调用失败时标记
                actions.append(OptimizationAction(
                    action_id=f"llm-fail-{cluster.cluster_id}",
                    source="model",
                    dimension=cluster.dimension,
                    priority=0.1,
                    title=f"[LLM 不可用] {cluster.defect_pattern[:40]}",
                    critique_analysis=f"LLM 调用失败: {e}",
                    revision_proposal="建议人工分析对话文本和评测归因，手动修改 prompt",
                    path="B",
                    level="弱建议",
                ))

        return actions

    def _build_current_prompt(self, case_data: Dict) -> str:
        """从 case.json 重建当前 Assistant System Prompt 文本。"""
        parts = []

        if case_data.get("role"):
            parts.append(f"# 你的角色\n{case_data['role']}")

        if case_data.get("task"):
            parts.append(f"\n# 任务目标\n{case_data['task']}")

        if case_data.get("opening_line"):
            parts.append(f"\n# 开场白\n通话开始时请使用以下开场白:\n「{case_data['opening_line']}」")

        call_flow = case_data.get("call_flow", [])
        if call_flow:
            parts.append("\n# 通话流程")
            for step in call_flow:
                parts.append(f"\n## Step {step.get('step_number', '?')}: {step.get('title', '')}")
                if step.get("description"):
                    parts.append(f"{step['description']}")
                branches = step.get("branches", [])
                for b in branches:
                    parts.append(f"  - 如果{b.get('condition', '')} → {b.get('action', '')}")

        kps = case_data.get("knowledge_points", [])
        if kps:
            parts.append("\n# 知识点（FAQ）")
            for kp in kps:
                parts.append(f"- {kp.get('topic', '')}: {kp.get('content', '')}")

        constraints = case_data.get("constraints", [])
        if constraints:
            parts.append("\n# 约束条件")
            for c in constraints:
                parts.append(f"- {c.get('description', '')}")

        parts.append("\n请严格按照以上指令完成通话。保持自然、专业。")
        return "\n".join(parts)

    def _get_conversation_context(
        self, cluster: DefectCluster, conv_dir: Path,
    ) -> str:
        """从对话文件中提取失败片段上下文。"""
        excerpts = []
        for g in cluster.gradients[:3]:  # 最多 3 个 gradient
            conv_id = g.conversation_id
            conv_file = conv_dir / f"conversation_{conv_id}.json"
            if not conv_file.exists():
                # 尝试模糊匹配
                try:
                    candidates = [f for f in conv_dir.iterdir()
                                  if f.suffix == '.json' and conv_id in f.name]
                    conv_file = candidates[0] if candidates else None
                except Exception:
                    pass
            if not conv_file or not conv_file.exists():
                continue

            try:
                conv_data = json.loads(conv_file.read_text(encoding="utf-8"))
                turns = conv_data.get("turns", [])
                for evidence in g.evidence[:2]:
                    import re
                    turn_nums = re.findall(r'T(\d+)', evidence)
                    for tn in turn_nums[:2]:
                        t = int(tn)
                        for turn in turns:
                            if turn.get("turn") in range(max(1, t - 3), min(len(turns), t + 4)):
                                speaker = "客服" if turn.get("speaker") == "system" else "用户"
                                line = f"T{turn['turn']} [{speaker}]: {turn.get('content', '')[:200]}"
                                # 附加 thought/memory/state 标签（如果存在）
                                tags = turn.get("parsed_tags", {})
                                if isinstance(tags, dict):
                                    extra = []
                                    if tags.get("thought"):
                                        extra.append(f"用户想法: {str(tags['thought'])[:100]}")
                                    if tags.get("state"):
                                        extra.append(f"用户状态: {str(tags['state'])[:80]}")
                                    if tags.get("model_behavior"):
                                        extra.append(f"模型行为: {str(tags['model_behavior'])[:80]}")
                                    if extra:
                                        line += " | " + " | ".join(extra)
                                excerpts.append(line)
            except Exception:
                continue

        if excerpts:
            self._conv_context_text = "\n".join(excerpts[:20])
            return "## 相关对话片段\n\n" + self._conv_context_text
        self._conv_context_text = ""
        return ""

    def _build_trajectory(self) -> str:
        """构建 OPRO 优化轨迹文本。"""
        if not self._trajectory:
            return "（尚无已生成的候选方案——这是本轮的第一条）"
        return "\n".join(f"- {t}" for t in self._trajectory[-10:])

    def _parse_batch_response(
        self, response: str, cluster: DefectCluster,
    ) -> List[OptimizationAction]:
        """解析 LLM 批量生成的候选方案。"""
        actions = []
        parts = response.split("---")
        for i, part in enumerate(parts):
            if "候选方案" not in part and "修改位置" not in part:
                continue

            # 字段提取
            def _extract(field: str, text: str) -> str:
                for line in text.split("\n"):
                    if line.strip().startswith(f"**{field}**"):
                        return line.split("**:", 1)[-1].strip()
                return ""

            target = _extract("修改位置", part)
            proposed = _extract("修改后文本", part)
            explanation = _extract("修改说明", part)
            risk = _extract("副作用风险", part)
            current_text = _extract("当前文本", part)
            cited_evidence = _extract("引用证据", part)
            path_a_relation = _extract("路径A关联", part)

            # 如果没有提取到修改后文本，尝试从整个part中提取
            if not proposed or len(proposed) < 10:
                # 回退：取"修改后文本"之后的所有内容直到下一个段
                for line in part.split("\n"):
                    if "修改后文本" in line and "**" in line:
                        proposed = line.split("**修改后文本**:", 1)[-1].strip()
                        break
                if not proposed or len(proposed) < 10:
                    continue

            # 证据合并：LLM引用的证据 + 原始cluster证据 + 对话上下文
            evidence = []
            if cited_evidence:
                evidence.append(cited_evidence)
            evidence.extend(cluster.evidence_samples[:2])
            if self._conv_context_text:
                # 截取对话上下文的前200字符作为证据摘要
                evidence.append(self._conv_context_text[:300])

            priority = (
                cluster.frequency * 1.5
                if cluster.severity == "major" else
                cluster.frequency * 0.8
            )

            actions.append(OptimizationAction(
                action_id=f"prompt-{cluster.cluster_id}-{i:02d}",
                source="model",
                dimension=cluster.dimension,
                priority=priority,
                title=f"Prompt 优化: {cluster.defect_pattern[:40]}",
                constitutional_principle=f"评测清单项——{cluster.dimension} 维度缺陷",
                violation_evidence=evidence[:5],
                critique_analysis=(
                    f"**当前文本**: {current_text}\n\n"
                    f"**修改说明**: {explanation}\n\n"
                    f"**副作用风险**: {risk}\n\n"
                    f"**路径A关联**: {path_a_relation or '无'}"
                ),
                revision_proposal=proposed,
                target_location=target or f"Assistant Prompt（{cluster.dimension} 相关段）",
                expected_impact=f"预期改善 {cluster.dimension} 维度评分",
                effort_estimate="低" if "保守" in part else "中",
                is_actionable=True,
                level="中建议" if cluster.severity != "major" else "强建议",
                path="B",
            ))

        return actions
