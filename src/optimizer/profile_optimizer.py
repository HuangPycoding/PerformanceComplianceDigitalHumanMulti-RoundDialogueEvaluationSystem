"""用户画像生成器优化器 — LLM + 规则引擎混合

输入: profiles.json + 评测归因 (source=simulator) + CO-STAR/对抗策略 prompt 源码
输出: 画像参数/对抗策略/CO-STAR模板/自检阈值/对话行为 的五类优化建议
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.llm.client import LLMClient
from src.optimizer.optimizer import DefectCluster, OptimizationAction
from src.optimizer.prompts import DIMENSION_CN, PROFILE_OPTIMIZE_SYSTEM, PROFILE_OPTIMIZE_USER


class ProfileOptimizer:
    """用户画像生成器优化器（规则统计 + LLM 生成）"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_fixes(
        self,
        clusters: List[DefectCluster],
        profiles_path: Path,
        conv_dir: Path,
        prompts_source_path: Optional[Path] = None,
    ) -> List[OptimizationAction]:
        """为画像生成器缺陷聚类生成优化建议。

        规则引擎统计：参数偏差相关性、策略触发频率、信号矛盾计数。
        LLM 生成：具体参数调整值、prompt 文本修改。
        """
        # 加载数据
        profile_data = json.loads(profiles_path.read_text(encoding="utf-8")) if profiles_path.exists() else {}
        prompt_templates = self._load_prompt_templates(prompts_source_path)

        # 规则统计
        rule_findings = self._rule_statistics(clusters, profile_data, conv_dir)

        actions: List[OptimizationAction] = []

        for cluster in clusters:
            if cluster.source != "simulator":
                continue

            # 信号矛盾 → 路径 A 直接输出（已在 optimizer.py 的 _sim_signal_contradiction 中处理）
            has_contradiction = any(
                "矛盾" in g.description or "不一致" in g.description
                for g in cluster.gradients
            )
            if has_contradiction:
                continue  # 已由路径 A 覆盖

            # 其他 → 路径 B LLM 分析
            try:
                response = self.llm.chat(
                    PROFILE_OPTIMIZE_SYSTEM,
                    PROFILE_OPTIMIZE_USER.format(
                        simulator_attributions="\n".join(
                            f"- [{g.dimension}] {g.description}"
                            for g in cluster.gradients[:5]
                        ),
                        profile_data=json.dumps(profile_data, ensure_ascii=False, indent=2)[:2000],
                        prompt_templates=prompt_templates,
                        conversation_summary=self._conv_summary(cluster, conv_dir),
                        rule_findings=rule_findings,
                    ),
                )

                actions.append(OptimizationAction(
                    action_id=f"profile-llm-{cluster.cluster_id}",
                    source="simulator",
                    dimension=cluster.dimension,
                    priority=cluster.frequency * 0.8,
                    title=f"画像生成器优化: {cluster.defect_pattern[:40]}",
                    constitutional_principle=f"用户画像生成器——{cluster.dimension}",
                    violation_evidence=cluster.evidence_samples[:2],
                    critique_analysis=f"{cluster.defect_pattern[:150]}",
                    revision_proposal=response[:1500],
                    target_location="画像生成器参数/策略/模板",
                    expected_impact=f"预期改善 {cluster.dimension} 维度评测准确性",
                    effort_estimate="中",
                    level="中建议",
                    path="B",
                ))
            except Exception:
                continue

        # 对话行为优化——纯规则（路径 A）
        actions.extend(self._behavior_rule_fixes(clusters, conv_dir))

        return actions

    def _rule_statistics(
        self, clusters: List[DefectCluster],
        profile_data: Dict, conv_dir: Path,
    ) -> str:
        """规则引擎统计画像相关指标。"""
        lines = []

        # 统计 Simulator 聚类总数
        sim_clusters = [c for c in clusters if c.source == "simulator"]
        lines.append(f"Simulator 归因聚类数: {len(sim_clusters)}")

        # 统计信号矛盾
        contradiction_count = sum(
            1 for c in sim_clusters
            for g in c.gradients
            if "矛盾" in g.description or "不一致" in g.description
        )
        lines.append(f"信号矛盾数: {contradiction_count}")

        # 统计各维度 Simulator 归因数
        dim_counts: Dict[str, int] = {}
        for c in sim_clusters:
            dim_counts[c.dimension] = dim_counts.get(c.dimension, 0) + c.frequency
        for dim, count in sorted(dim_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {dim}: {count} 次")

        return "\n".join(lines)

    def _load_prompt_templates(self, source_path: Optional[Path]) -> str:
        """从源码文件加载 CO-STAR 和对抗策略 prompt 模板。"""
        if not source_path or not source_path.exists():
            return "（未找到 prompt 模板源码文件）"

        try:
            content = source_path.read_text(encoding="utf-8")
            # 提取关键模板
            sections = []
            for marker in [
                "PROFILE_GENERATION_PROMPT",
                "SELF_CHECK_PROMPT",
                "ADVERSARIAL_PROBE",
                "ADVERSARIAL_INJECTION",
                "ADVERSARIAL_CONTRADICTION",
                "ADVERSARIAL_AUTHORITY",
                "ADVERSARIAL_EMOTION",
            ]:
                idx = content.find(marker)
                if idx >= 0:
                    end = content.find("\n\n\n", idx)
                    if end < 0:
                        end = min(idx + 2000, len(content))
                    sections.append(content[idx:end].strip()[:1500])
            return "\n\n---\n\n".join(sections) if sections else "（未找到已知模板）"
        except Exception:
            return "（加载模板失败）"

    def _conv_summary(self, cluster: DefectCluster, conv_dir: Path) -> str:
        """对话数据摘要。"""
        lines = [f"涉及 {len(cluster.conversation_ids)} 场对话"]
        # 统计 should_end_mismatch
        mismatch_count = 0
        total_conv = 0
        for cid in cluster.conversation_ids[:5]:
            conv_file = conv_dir / f"conversation_{cid}.json"
            if not conv_file.exists():
                try:
                    candidates = [f for f in conv_dir.iterdir()
                                  if f.suffix == '.json' and cid in f.name]
                    conv_file = candidates[0] if candidates else None
                except Exception:
                    pass
            if not conv_file or not conv_file.exists():
                continue
            try:
                conv_data = json.loads(conv_file.read_text(encoding="utf-8"))
                total_conv += 1
                turns = conv_data.get("turns", [])
                for t in turns:
                    tags = t.get("parsed_tags", {})
                    se = tags.get("should_end") if isinstance(tags, dict) else None
                    should_end = False
                    if isinstance(se, dict):
                        val = str(se.get("本轮是否想结束对话", "")).strip().strip("[]")
                        should_end = val in ("是", "true", "yes")
                    if should_end:
                        if t.get("turn", 0) < len(turns) - 2:
                            mismatch_count += 1
                        break
            except Exception:
                continue

        if total_conv > 0:
            rate = mismatch_count / total_conv * 100
            lines.append(f"should_end_mismatch: {mismatch_count}/{total_conv} ({rate:.0f}%)")
            if rate > 30:
                lines.append("⚠️ 超过 30% 阈值——建议收紧匹配逻辑")
        return "\n".join(lines)

    def _behavior_rule_fixes(
        self, clusters: List[DefectCluster], conv_dir: Path,
    ) -> List[OptimizationAction]:
        """对话行为优化——纯规则路径 A。"""
        actions = []
        for cluster in clusters:
            if cluster.source != "simulator":
                continue

            # 统计 should_end_mismatch
            mismatch = 0
            early_end = 0
            late_end = 0
            total = 0

            for cid in cluster.conversation_ids[:10]:
                conv_file = conv_dir / f"conversation_{cid}.json"
                if not conv_file.exists():
                    try:
                        candidates = [f for f in conv_dir.iterdir()
                                      if f.suffix == '.json' and cid in f.name]
                        conv_file = candidates[0] if candidates else None
                    except Exception:
                        pass
                if not conv_file or not conv_file.exists():
                    continue
                try:
                    conv_data = json.loads(conv_file.read_text(encoding="utf-8"))
                    total += 1
                    turns = conv_data.get("turns", [])
                    total_turns = len(turns)
                    for t in turns:
                        tags = t.get("parsed_tags", {})
                        se = tags.get("should_end") if isinstance(tags, dict) else None
                        should_end = False
                        if isinstance(se, dict):
                            val = str(se.get("本轮是否想结束对话", "")).strip().strip("[]")
                            should_end = val in ("是", "true", "yes")
                        if should_end:
                            tn = t.get("turn", 0)
                            if tn < total_turns - 2:
                                mismatch += 1
                            if tn < getattr(conv_data, "min_turns", 8):
                                early_end += 1
                            break
                except Exception:
                    continue

            if total == 0:
                continue

            mr = mismatch / total * 100
            er = early_end / total * 100

            if mr > 30 or er > 20:
                proposals = []
                if mr > 30:
                    proposals.append(f"should_end_mismatch={mr:.0f}% > 30%，建议收紧匹配逻辑")
                if er > 20:
                    proposals.append(f"过早结束={er:.0f}% > 20%，建议增加 END_KEYWORDS 条目")

                actions.append(OptimizationAction(
                    action_id=f"behavior-{cluster.cluster_id}",
                    source="simulator",
                    dimension=cluster.dimension,
                    priority=cluster.frequency * 0.5,
                    title=f"对话行为优化: should_end 逻辑调整",
                    constitutional_principle="Simulator 对话结束行为",
                    violation_evidence=cluster.evidence_samples[:1],
                    critique_analysis=f"should_end_mismatch={mr:.0f}%, 过早结束={er:.0f}%",
                    revision_proposal="\n".join(proposals),
                    target_location="src/simulator/simulator.py should_end_conversation()",
                    expected_impact="提升对话结束准确性",
                    effort_estimate="低",
                    level="弱建议",
                    path="A",
                ))

        return actions
