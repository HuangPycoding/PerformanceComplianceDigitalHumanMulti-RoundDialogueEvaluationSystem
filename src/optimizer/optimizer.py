"""优化引擎主编排器 — 消费评测输出，产出优化建议

Phase 1 内部管线:
  Step 0: 加载四路数据 + 质量门控
  Step 1: 归因按 source 分组 → 路由到四个优化器
  Step 2: 规则引擎预处理（聚类、排序、统计）
  Step 3: 双路径输出（路径 A 规则直接输出 / 路径 B 调用 LLM 生成）
  Step 4: 综合去重 & 合并
  Step 5: 报告导出
"""

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.llm.client import LLMClient
from src.optimizer.utils import (
    DIMENSION_WEIGHTS,
    SEVERITY_WEIGHTS,
    bigram_jaccard,
    calc_priority,
    classify_severity,
    cluster_by_similarity,
    compute_discrimination_stats,
    frequency_distribution,
)
from src.optimizer.eval_optimizer import (
    analyze_defect_conversion,
    load_evolver_stats,
)


# ============================================================
# 数据类
# ============================================================

@dataclass
class TextualGradient:
    """文本梯度——从归因项转化而来的结构化批评信号"""
    source: str               # "model" | "case" | "simulator"
    dimension: str            # 维度名
    dimension_cn: str         # 维度中文名
    description: str          # 自然语言批评
    evidence: List[str]       # 证据支撑
    confidence: float         # 经质量门控调整后的可信度
    severity: str             # "major" | "moderate" | "minor"
    is_actionable: bool       # 是否达到触发优化门槛
    conversation_id: str = ""


@dataclass
class DefectCluster:
    """缺陷聚类"""
    cluster_id: str
    source: str
    dimension: str
    defect_pattern: str
    gradients: List[TextualGradient] = field(default_factory=list)
    frequency: int = 0
    severity: str = "minor"
    conversation_ids: List[str] = field(default_factory=list)
    evidence_samples: List[str] = field(default_factory=list)


@dataclass
class OptimizationAction:
    """优化动作——CAI 三段式输出"""
    action_id: str
    source: str
    dimension: str
    priority: float
    title: str
    constitutional_principle: str = ""
    violation_evidence: List[str] = field(default_factory=list)
    critique_analysis: str = ""
    revision_proposal: str = ""
    target_location: str = ""
    expected_impact: str = ""
    effort_estimate: str = "中"
    is_actionable: bool = False
    level: str = "中建议"       # "强建议" | "中建议" | "弱建议"
    path: str = "A"            # "A"(规则) | "B"(LLM)


# 维度中文名映射
_DIM_CN = {
    "SAFETY": "安全合规", "TASK_COMPLETION": "任务达成",
    "FLOW_COVERAGE": "流程覆盖", "KNOWLEDGE": "知识准确性",
    "CONSTRAINT": "约束遵守", "EFFICIENCY": "对话效率",
    "SENTIMENT": "情感适配", "ROLE": "角色一致性",
    "OPENING": "开场白合规",
}


# ============================================================
# 主编排器
# ============================================================

class OptimizationEngine:
    """优化引擎主编排器"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self.gradients: List[TextualGradient] = []
        self.clusters: Dict[str, List[DefectCluster]] = {}
        self.actions: List[OptimizationAction] = []
        self.per_conv: Dict[str, Dict] = {}  # 逐对话上下文

    # ---- 主入口 ----

    def run(self, input_dir: str, output_dir: str) -> Dict[str, Any]:
        """主编排入口——从导出目录加载数据，生成优化建议并导出报告。

        Args:
            input_dir: 评测导出目录（含 case.json/profiles.json/conversation_*.json/
                       optimization_feed.json）
            output_dir: 优化报告输出目录

        Returns:
            {"actions": List[OptimizationAction], "report_path": str, ...}
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Step 0: 加载 & 质量门控
        self._load_and_gate(input_path)

        # Step 1: 归因按 source 分组
        grouped = self._group_by_source()

        # Step 2: 规则引擎预处理（聚类 + 排序 + 统计）
        self.clusters = self._preprocess(grouped)

        # Step 3: 双路径输出
        self.actions = self._generate_suggestions(input_path)

        # Step 4: 综合去重 & 合并
        self.actions = self._deduplicate(self.actions)

        # Step 5: 报告导出
        report_path = self._export(output_path)

        return {
            "actions": self.actions,
            "report_path": str(report_path),
            "n_gradients": len(self.gradients),
            "n_clusters": sum(len(c) for c in self.clusters.values()),
            "n_actions": len(self.actions),
        }

    # ---- Step 0: 加载 & 质量门控 ----

    def _load_and_gate(self, input_path: Path) -> None:
        """加载四路数据 + 原始评测 JSON，应用质量门控，构建 TextualGradient 列表。

        除 optimization_feed.json 外，还会读取 evaluation_*.json 提取真实证据文本，
        以及 conversation_*.json 提取对话片段。确保证据字段包含具体引用内容而非仅评分。
        """
        feed_path = input_path / "optimization_feed.json"
        if not feed_path.exists():
            raise FileNotFoundError(
                f"optimization_feed.json 不存在于 {input_path}。"
            )

        with open(feed_path, encoding="utf-8") as f:
            feed = json.load(f)

        self.per_conv = feed.get("per_conversation", {})
        attributions = feed.get("attributions", [])

        # 预加载原始评测 JSON 中的真实证据
        raw_evidence = self._load_raw_evaluation_evidence(input_path)

        for attr in attributions:
            conv_id = attr.get("conversation_id", "")
            conv_info = self.per_conv.get(conv_id, {})
            is_reliable = conv_info.get("is_reliable", True)

            raw_conf = attr.get("confidence", 0.5)
            if raw_conf < 0.5:
                continue

            if not is_reliable:
                effective_conf = raw_conf * 0.5
                is_actionable = False
            else:
                effective_conf = raw_conf
                is_actionable = raw_conf >= 0.8

            severity = classify_severity(
                attr.get("description", ""),
                attr.get("category", ""),
                effective_conf,
            )

            # 合并 feed 中的 evidence_chain 和原始评测 JSON 中的真实证据
            evidence = list(attr.get("evidence_chain", []))
            raw_ev = raw_evidence.get(conv_id, {})
            category = attr.get("category", "")

            # 添加该维度的真实清单项证据（含具体对话文本）
            dim_evidence = raw_ev.get("dimension_evidence", {}).get(category, [])
            for de in dim_evidence[:3]:
                evidence.append(de)

            # 添加交叉验证告警和元检查告警
            for alert in raw_ev.get("alerts", []):
                if category in alert.get("dimension", "") or category in alert.get("description", ""):
                    evidence.append(f"[{alert.get('type', '')}] {alert.get('description', '')}")

            # 去重
            evidence = list(dict.fromkeys(evidence))

            self.gradients.append(TextualGradient(
                source=attr.get("source", "model"),
                dimension=category,
                dimension_cn=_DIM_CN.get(category, category),
                description=attr.get("description", ""),
                evidence=evidence,
                confidence=effective_conf,
                severity=severity,
                is_actionable=is_actionable,
                conversation_id=conv_id,
            ))

    def _load_raw_evaluation_evidence(self, input_path: Path) -> Dict[str, Any]:
        """从 evaluation_*.json 中提取真实证据文本。

        Returns:
            {conversation_id: {
                "dimension_evidence": {dim: ["T1: 客服'...' T2: 用户'...' → NO", ...]},
                "alerts": [{type, description, dimension}, ...],
            }}
        """
        result: Dict[str, Any] = {}
        eval_files = sorted(input_path.glob("evaluation_*.json"))

        for ef in eval_files:
            try:
                d = json.loads(ef.read_text(encoding="utf-8"))
            except Exception:
                continue

            cid = d.get("conversation_id", "")
            if not cid:
                continue

            dim_evidence: Dict[str, List[str]] = {}

            # 从 dimension_checklists 提取每个清单项的证据 + CoT reasoning
            for dim, checklist in d.get("dimension_checklists", {}).items():
                items = checklist.get("items", [])
                for item in items:
                    status = item.get("status", "")
                    evidence_text = item.get("evidence", "")
                    desc = item.get("description", "")
                    reasoning = item.get("reasoning", "")  # CoT 推理全文
                    if status in ("NO", "MOSTLY_NO", "PARTIAL") and evidence_text:
                        entry = f"[{status}] {desc}: {evidence_text[:200]}"
                        if reasoning:
                            entry += f" | Judge推理: {reasoning[:150]}"
                        dim_evidence.setdefault(dim, []).append(entry)
                    elif status in ("YES", "MOSTLY_YES") and evidence_text:
                        dim_evidence.setdefault(dim, []).append(
                            f"[{status}] {desc}: {evidence_text[:150]}"
                        )

            # 收集 Tier1 规则中间数据
            tier1_data = {}
            for issue in d.get("rule_check_issues", []):
                if "turns_ratio" in str(issue) or "轮次" in str(issue):
                    tier1_data["efficiency_issue"] = issue
                if "禁止词" in str(issue) or "forbidden" in str(issue):
                    tier1_data["constraint_issue"] = issue

            # 收集告警
            alerts = []
            # Tier1 指标作为告警
            for key, val in tier1_data.items():
                alerts.append({
                    "type": "Tier1规则",
                    "description": str(val)[:200],
                    "dimension": "EFFICIENCY" if "efficiency" in key else "CONSTRAINT",
                })
            for a in d.get("cross_validation_alerts", []):
                alerts.append({
                    "type": "交叉验证",
                    "description": a.get("description", ""),
                    "dimension": a.get("dimension", ""),
                })
            for a in d.get("meta_check_alerts", []):
                dims = a.get("dimensions", [])
                dim_str = dims[0] if dims else a.get("dimension", "")
                alerts.append({
                    "type": f"元检查({a.get('check_type', '')})",
                    "description": a.get("description", ""),
                    "dimension": dim_str,
                })

            result[cid] = {
                "dimension_evidence": dim_evidence,
                "alerts": alerts,
            }

        return result

    # ---- Step 1: 按 source 分组 ----

    def _group_by_source(self) -> Dict[str, List[TextualGradient]]:
        """按 source 字段分组，返回三组 gradient。"""
        grouped: Dict[str, List[TextualGradient]] = defaultdict(list)
        for g in self.gradients:
            grouped[g.source].append(g)
        return dict(grouped)

    # ---- Step 2: 规则引擎预处理 ----

    def _preprocess(
        self, grouped: Dict[str, List[TextualGradient]]
    ) -> Dict[str, List[DefectCluster]]:
        """对每组 gradient 做聚类 + 优先级排序。"""
        all_clusters: Dict[str, List[DefectCluster]] = {}

        for source, grads in grouped.items():
            # 按维度再分
            by_dim: Dict[str, List[TextualGradient]] = defaultdict(list)
            for g in grads:
                by_dim[g.dimension].append(g)

            source_clusters: List[DefectCluster] = []
            cluster_idx = 0

            for dim, dim_grads in by_dim.items():
                # 转为 dict 列表供聚类使用
                items = [{"description": g.description, "_gradient": g} for g in dim_grads]
                raw_clusters = cluster_by_similarity(items, threshold=0.5)

                for rc in raw_clusters:
                    grads_in_cluster = [it["_gradient"] for it in rc]
                    descriptions = [it["description"] for it in rc]

                    # 聚类标签取频率最高的描述
                    freq_desc = max(set(descriptions), key=descriptions.count)

                    # 严重程度取最严重的
                    severities = [g.severity for g in grads_in_cluster]
                    max_sev = "major" if "major" in severities else \
                              "moderate" if "moderate" in severities else "minor"

                    avg_conf = sum(g.confidence for g in grads_in_cluster) / len(grads_in_cluster)

                    cluster = DefectCluster(
                        cluster_id=f"{source}-{dim}-{cluster_idx:03d}",
                        source=source,
                        dimension=dim,
                        defect_pattern=freq_desc,
                        gradients=grads_in_cluster,
                        frequency=len(grads_in_cluster),
                        severity=max_sev,
                        conversation_ids=list(set(g.conversation_id for g in grads_in_cluster)),
                        evidence_samples=list(dict.fromkeys([
                            e for g in grads_in_cluster
                            for e in g.evidence[:3]  # 每个 gradient 取前 3 条证据
                        ]))[:5],  # 去重后最多 5 条
                    )
                    source_clusters.append(cluster)
                    cluster_idx += 1

            # 优先级排序（降序）
            source_clusters.sort(
                key=lambda c: calc_priority(c.severity, c.dimension, c.frequency,
                                            sum(g.confidence for g in c.gradients) / len(c.gradients)),
                reverse=True,
            )

            # 每个 source 最多保留 top-10 聚类
            all_clusters[source] = source_clusters[:10]

        return all_clusters

    # ---- Step 3: 双路径输出 ----

    def _generate_suggestions(self, input_path: Path) -> List[OptimizationAction]:
        """生成优化建议——路径 A（规则直接输出）+ 路径 B（LLM）。"""
        actions: List[OptimizationAction] = []

        # 路径 A: 规则引擎——确定性发现直接输出
        path_a_actions = self._path_a_suggestions(input_path)
        actions.extend(path_a_actions)

        # 路径A→B 联动：收集路径A的统计发现，作为路径B LLM的分析上下文
        path_a_findings = self._summarize_path_a(path_a_actions)

        # 路径 B: LLM——调用子模块生成深度建议（含路径A上下文）
        if self.llm:
            try:
                from src.optimizer.prompt_optimizer import PromptOptimizer
                po = PromptOptimizer(self.llm)
                actions.extend(po.generate_candidates(
                    self.clusters.get("model", []), input_path / "case.json", input_path,
                    path_a_context=path_a_findings,
                ))
            except Exception as e:
                actions.append(self._llm_fail_action("prompt_optimizer", str(e)))

            try:
                from src.optimizer.fewshot_generator import FewshotGenerator
                fg = FewshotGenerator(self.llm)
                ratings_data = {
                    cid: info.get("ratings", {})
                    for cid, info in self.per_conv.items()
                }
                actions.extend(fg.generate_examples(
                    self.clusters.get("model", []), input_path, ratings_data,
                ))
            except Exception as e:
                actions.append(self._llm_fail_action("fewshot_generator", str(e)))

            try:
                from src.optimizer.case_fixer import CaseFixer
                cf = CaseFixer(self.llm)
                actions.extend(cf.generate_fixes(
                    self.clusters.get("case", []), input_path / "case.json",
                ))
            except Exception as e:
                actions.append(self._llm_fail_action("case_fixer", str(e)))

            try:
                from src.optimizer.profile_optimizer import ProfileOptimizer
                pfo = ProfileOptimizer(self.llm)
                actions.extend(pfo.generate_fixes(
                    self.clusters.get("simulator", []), input_path / "profiles.json", input_path,
                    Path("src/llm/prompts.py"),
                ))
            except Exception as e:
                actions.append(self._llm_fail_action("profile_optimizer", str(e)))

        return actions

    def _summarize_path_a(self, actions: List[OptimizationAction]) -> str:
        """汇总路径A的统计发现，作为路径B LLM的结构化分析上下文。

        返回结构化文本（非纯文字描述），LLM prompt 中强制引用。
        """
        if not actions:
            return ""

        lines = [
            "## 路径A 规则分析发现（请在你的分析中引用以下数据）",
            "",
        ]
        for a in actions:
            if a.path != "A":
                continue
            lines.append(f"### 发现: {a.title}")
            lines.append(f"- 涉及维度: {a.dimension}")
            lines.append(f"- 分析结论: {a.critique_analysis[:200]}")
            lines.append(f"- 建议方向: {a.revision_proposal[:200]}")
            lines.append("")
            lines.append("**请在你的建议中明确判断**: 该缺陷是真正的模型问题，还是路径A发现的Case/画像问题导致的假信号？")
            lines.append("")

        return "\n".join(lines)

    def _llm_fail_action(self, module: str, error: str) -> OptimizationAction:
        return OptimizationAction(
            action_id=f"fail-{module}",
            source="model",
            dimension="",
            priority=0.1,
            title=f"[LLM 不可用] {module} 调用失败",
            critique_analysis=f"{module} LLM 调用失败: {error}",
            revision_proposal=f"建议人工分析对话文本和评测归因，手动修改 {module} 相关配置",
            path="B",
            level="弱建议",
        )

    def _path_a_suggestions(self, input_path: Optional[Path] = None) -> List[OptimizationAction]:
        """路径 A: 规则引擎直接输出确定性发现。"""
        actions: List[OptimizationAction] = []

        actions.extend(self._eval_engine_path_a())
        actions.extend(self._sim_signal_contradiction())
        actions.extend(self._case_overly_strict(input_path))
        actions.extend(self._case_meta_check_fixes(input_path))
        actions.extend(self._eval_checklist_evolution())

        return actions

    def _case_meta_check_fixes(self, input_path: Optional[Path] = None) -> List[OptimizationAction]:
        """读取评测引擎 meta_check_alerts 中的 Case 内部一致性问题，生成优化建议。

        直接读取原始 evaluation JSON 中的 consistency 类型 meta_check 告警，
        不依赖 gradient 证据链（因为 meta_check 告警可能不匹配任何单一归因项的 category）。
        """
        actions = []
        if not input_path:
            return actions

        seen = set()
        for ef in sorted(input_path.glob("evaluation_*.json")):
            try:
                d = json.loads(ef.read_text(encoding="utf-8"))
            except Exception:
                continue
            for a in d.get("meta_check_alerts", []):
                if a.get("check_type") != "consistency":
                    continue
                desc = a.get("description", "")
                if ("约束冲突" not in desc and "word_limit" not in desc):
                    continue
                if desc in seen:
                    continue
                seen.add(desc)

                # 提取冲突详情（去掉前缀"Case 约束冲突："）
                detail = desc.replace("Case 约束冲突：", "").strip()
                actions.append(OptimizationAction(
                    action_id="case-meta-consistency",
                    source="case",
                    dimension="FLOW_COVERAGE",
                    priority=4.0,
                    title="Case 约束冲突：字数限制与流程步骤信息量不匹配（评测引擎 meta_check 检出）",
                    constitutional_principle="Case 定义合理性——约束与流程一致性（评测引擎 meta_check）",
                    violation_evidence=[desc],
                    critique_analysis=(
                        f"评测引擎在 meta_check（Case 内部一致性检查）中检测到约束-流程冲突。"
                        f"约束限制与流程步骤的信息量存在矛盾，"
                        f"导致模型在执行流程步骤时受到约束限制，系统性跳过某些步骤。\n\n"
                        f"具体检测结果: {detail}"
                    ),
                    revision_proposal=(
                        "建议修改方案（任选其一）：\n"
                        "1. 放宽字数限制（如 30→60 字）\n"
                        "2. 拆分信息量大的步骤为子步骤\n"
                        "3. 对信息密集型步骤标记为不受字数限制"
                    ),
                    target_location="Case 定义 constraints + call_flow",
                    expected_impact="消除约束冲突后，流程步骤覆盖率预期显著提升（当前 FLOW 90% 不合格与此直接相关）",
                    effort_estimate="低",
                    level="强建议",
                    path="A",
                ))
        return actions

    def _eval_engine_path_a(self) -> List[OptimizationAction]:
        """评测引擎优化——路径 A（基于真实评分分布/覆盖率/置信度数据）。

        不需要 ChecklistEvolver 跨批次积累——v1 阶段直接从当前批次的
        评分分布、元检查告警、置信度数据中分析评测引擎自身的问题。
        """
        actions = []

        # 1. 评分分布异常检测
        dim_ratings: Dict[str, List[str]] = {}
        for info in self.per_conv.values():
            for dim, rating in info.get("ratings", {}).items():
                dim_ratings.setdefault(dim, []).append(rating)

        for dim, ratings in dim_ratings.items():
            n = len(ratings)
            excellent = sum(1 for r in ratings if r == "卓越")
            good = sum(1 for r in ratings if r == "良好")
            fail_or_improve = sum(1 for r in ratings if r in ("不合格", "需改进"))
            single_rating_pct = max(excellent, good, fail_or_improve) / n if n > 0 else 0

            # 评分分布极度集中 → 区分力不足
            if single_rating_pct >= 0.80 and n >= 5:
                dominant = "卓越" if excellent == max(excellent, good, fail_or_improve) else \
                          "良好" if good == max(excellent, good, fail_or_improve) else "需改进/不合格"
                actions.append(OptimizationAction(
                    action_id=f"eval-dist-{dim}",
                    source="eval",
                    dimension=dim,
                    priority=2.0 + (1.5 if dim == "SAFETY" else 0),
                    title=f"评测引擎区分力不足——{dim} 维度 {single_rating_pct:.0%} 集中在'{dominant}'",
                    constitutional_principle=f"评测引擎清单区分力——{dim}",
                    violation_evidence=[
                        f"{dim} 维度评分分布: {dict((r, ratings.count(r)) for r in ['卓越','良好','合格','需改进','不合格'])}",
                        f"{single_rating_pct:.0%} 的对话集中在同一评级，区分力不足",
                    ],
                    critique_analysis=(
                        f"{dim} 维度在 {n} 场对话中 {single_rating_pct:.0%} 都是'{dominant}'评级。"
                        f"评测引擎的清单项或 Judge prompt 可能缺乏区分力——无法有效区分不同质量的 Assistant。"
                    ),
                    revision_proposal=(
                        f"1. 检查 {dim} 维度的清单项是否过于简单（几乎人人都能通过）或过于困难\n"
                        f"2. 考虑增加该维度的清单项数量或调整判定粒度\n"
                        f"3. 优化该维度的 Judge prompt，增加行为锚点的区分度"
                    ),
                    target_location=f"src/eval/config.py CHECKLIST_SIZE + src/eval/schemas.py Judge prompt（{dim}）",
                    expected_impact=f"提升 {dim} 维度对不同质量 Assistant 的区分能力",
                    effort_estimate="中",
                    level="中建议",
                    path="A",
                ))

        # 2. 置信度异常检测
        conf_levels = []
        for info in self.per_conv.values():
            level = info.get("confidence_level", "unknown")
            if level != "unknown":
                conf_levels.append(level)
        if conf_levels:
            unreliable_pct = conf_levels.count("unreliable") / len(conf_levels)
            low_or_worse = (conf_levels.count("unreliable") + conf_levels.count("low")) / len(conf_levels)
            if low_or_worse >= 0.5:
                actions.append(OptimizationAction(
                    action_id="eval-confidence",
                    source="eval",
                    dimension="",
                    priority=3.0,
                    title=f"评测引擎置信度偏低——{low_or_worse:.0%} 对话为 unreliable/low",
                    constitutional_principle="评测引擎可信度（EvalConfidence）",
                    violation_evidence=[
                        f"置信度分布: {dict((l, conf_levels.count(l)) for l in set(conf_levels))}",
                        f"unreliable 占比: {unreliable_pct:.0%}",
                    ],
                    critique_analysis=(
                        f"{low_or_worse:.0%} 的对话评测结果被标记为低可信度。"
                        f"评测引擎的置信度计算可能过于保守，或者评测数据本身质量不足。"
                    ),
                    revision_proposal=(
                        f"1. 检查 CONFIDENCE 因子配置（signal_conflict_penalty/evidence_empty_penalty 等）\n"
                        f"2. 考虑放宽部分置信度惩罚项（当前过于保守）\n"
                        f"3. 增加 ChecklistEvolver 数据积累以校准置信度因子"
                    ),
                    target_location="src/eval/config.py CONFIDENCE 配置块",
                    expected_impact="提升评测结果的可信度水平，减少人工复核需求",
                    effort_estimate="中",
                    level="中建议",
                    path="A",
                ))

        # 3. 元检查告警汇总
        coverage_dims: Dict[str, int] = {}
        evidence_issues = 0
        for grad in self.gradients:
            for ev in grad.evidence:
                if "仅有" in ev and "条 applicable" in ev:
                    for dim in _DIM_CN:
                        if dim in ev:
                            coverage_dims[dim] = coverage_dims.get(dim, 0) + 1
                if "evidence ref not found" in ev:
                    evidence_issues += 1

        if coverage_dims:
            dims_str = "、".join(f"{d}({c}次)" for d, c in sorted(coverage_dims.items(), key=lambda x: -x[1])[:3])
            actions.append(OptimizationAction(
                action_id="eval-coverage",
                source="eval",
                dimension="",
                priority=2.5,
                title=f"评测引擎清单覆盖不足——{dims_str}",
                constitutional_principle="评测引擎清单覆盖率",
                violation_evidence=[
                    f"覆盖不足的维度: {dict(coverage_dims)}",
                    f"证据引用失败: {evidence_issues} 次",
                ] if evidence_issues > 0 else [
                    f"覆盖不足的维度: {dict(coverage_dims)}",
                ],
                critique_analysis=(
                    f"多个维度的清单项 applicable 数量偏低，评测引擎的覆盖率不足。"
                    f"可能导致评测结果不全面。"
                ),
                revision_proposal=(
                    f"1. 检查 CHECKLIST_SIZE 配置，增加低覆盖维度的清单项数量下限\n"
                    f"2. 针对覆盖不足的维度（{dims_str}）补充清单项\n"
                    f"3. 检查 Judge prompt 中是否有导致 NOT_APPLICABLE 判定过多的逻辑"
                ),
                target_location="src/eval/config.py CHECKLIST_SIZE",
                expected_impact="提升评测引擎的清单覆盖率",
                effort_estimate="中",
                level="中建议",
                path="A",
            ))

        return actions

    def _eval_checklist_evolution(self) -> List[OptimizationAction]:
        """清单进化分析——路径 A：消费 ChecklistEvolver 跨批次统计数据，
        将高频 additional_defects 转化为正式清单项建议。

        需要 data/checklist_evolution/accumulated_defects.jsonl 存在。
        """
        actions = []
        try:
            stats = load_evolver_stats()
        except Exception:
            return actions

        defect_freq = stats.get("defect_frequencies", {})
        if not defect_freq:
            return actions

        total = stats.get("total_defects", 0)
        candidates = analyze_defect_conversion(defect_freq, min_frequency=5)

        # 汇总缺陷分布
        dim_freq = stats.get("defects_by_dimension", {})
        top_dims = sorted(dim_freq.items(), key=lambda x: -x[1])[:5]

        if candidates:
            for c in candidates[:5]:  # 最多 5 条转化建议
                actions.append(OptimizationAction(
                    action_id=f"evolve-{c['suggested_item_id']}",
                    source="eval",
                    dimension="",
                    priority=min(8.0, 2.0 + c["frequency"] * 0.5),
                    title=f"高频缺陷转化——'{c['description'][:60]}'（出现 {c['frequency']} 次）应转化为正式清单项",
                    constitutional_principle="评测引擎清单进化——缺陷积累→自动转化",
                    violation_evidence=[
                        f"跨批次累计缺陷: {total} 条",
                        f"该缺陷出现 {c['frequency']} 次，超过转化阈值（5 次）",
                        f"缺陷 Top-5 维度分布: {dict(top_dims)}" if top_dims else "",
                    ],
                    critique_analysis=(
                        f"ChecklistEvolver 跨批次统计发现，缺陷'{c['description'][:80]}'"
                        f"在多个对话中累计出现 {c['frequency']} 次。"
                        f"根据清单进化机制（Phase 3.2），高频缺陷（≥5 次）应自动转化为正式清单项，"
                        f"来源标注为 pattern_mined，初始权重 1.3。"
                    ),
                    revision_proposal=(
                        f"建议新增清单项:\n"
                        f"  item_id: {c['suggested_item_id']}\n"
                        f"  description: {c['description'][:200]}\n"
                        f"  source: pattern_mined\n"
                        f"  weight: {c['suggested_weight']}\n"
                        f"转化后执行 run_evolution_cycle() 完成增删改+校准。"
                    ),
                    target_location="src/eval/checklist_evolver.py → run_evolution_cycle()",
                    expected_impact=f"新增清单项后，该缺陷模式将被系统化检测，不再依赖 LLM 自由补充",
                    effort_estimate="低",
                    level="中建议",
                    path="A",
                ))

        # 全局缺陷统计
        if total >= 20:
            actions.append(OptimizationAction(
                action_id="evolve-summary",
                source="eval",
                dimension="",
                priority=3.0,
                title=f"清单进化统计——跨批次累计 {total} 条缺陷，Top 维度: {', '.join(f'{d}({c})' for d, c in top_dims[:3])}",
                constitutional_principle="评测引擎清单进化——缺陷积累态势",
                violation_evidence=[
                    f"累计缺陷总数: {total}",
                    f"缺陷维度分布: {dict(top_dims)}",
                ],
                critique_analysis=(
                    f"ChecklistEvolver 已积累 {total} 条 additional_defects，"
                    f"主要集中在 {top_dims[0][0] if top_dims else 'N/A'} 等维度。"
                    f"建议定期执行 run_evolution_cycle() 进行清单增删改+校准。"
                ),
                revision_proposal=(
                    "执行 ChecklistEvolver.run_evolution_cycle() 触发全流程编排: "
                    "分析→转化→裁剪→校准，自动将高频缺陷转为清单项并标记低区分力项。"
                ),
                target_location="src/eval/checklist_evolver.py",
                expected_impact="清单覆盖率提升，减少对 LLM 自由补充缺陷的依赖",
                effort_estimate="低",
                level="弱建议",
                path="A",
            ))

        return actions

    def _sim_signal_contradiction(self) -> List[OptimizationAction]:
        """Simulator 信号矛盾检测——路径 A 直接输出。

        当 simulator 归因项中存在信号矛盾（signal_consistency="矛盾"）时，
        直接建议检查 Simulator 标签生成逻辑。
        """
        actions = []
        sim_clusters = self.clusters.get("simulator", [])

        for cluster in sim_clusters:
            # 检测信号矛盾
            has_contradiction = any(
                "矛盾" in g.description or "不一致" in g.description
                for g in cluster.gradients
            )
            if not has_contradiction:
                continue

            priority = calc_priority(
                cluster.severity, cluster.dimension,
                cluster.frequency,
                sum(g.confidence for g in cluster.gradients) / len(cluster.gradients),
            )

            actions.append(OptimizationAction(
                action_id=f"sim-signal-{cluster.cluster_id}",
                source="simulator",
                dimension=cluster.dimension,
                priority=priority,
                title=f"Simulator 信号矛盾——{cluster.dimension} 维度",
                constitutional_principle=f"Simulator 标注信号与对话文本一致性——{cluster.dimension}",
                violation_evidence=(
                    [e for g in cluster.gradients for e in g.evidence[:2] if e and len(e) > 10][:5]
                    or cluster.evidence_samples[:5]
                ),
                critique_analysis=(
                    f"{cluster.dimension} 维度检测到 {cluster.frequency} 次信号矛盾。"
                    f"Simulator 标注的信号（emotion/model_behavior 等）与对话文本不一致，"
                    f"可能导致评测误判。"
                ),
                revision_proposal=(
                    f"1. 复查 parsed_tags 中 {cluster.dimension} 相关标签的生成逻辑\n"
                    f"2. 对比对话文本与信号标注，定位不一致的轮次\n"
                    f"3. 修正对应标签的判定阈值或生成 prompt"
                ),
                target_location="src/simulator/simulator.py parsed_tags 生成逻辑",
                expected_impact=f"预计可修正 {cluster.frequency} 次信号矛盾，提升评测准确性",
                effort_estimate="中",
                is_actionable=True,
                level="中建议",
                path="A",
            ))

        return actions

    def _case_overly_strict(self, input_path: Optional[Path] = None) -> List[OptimizationAction]:
        """Case 过严检测——路径 A 直接输出。

        当 case 归因项中多对话在某维度低分时，建议降低复杂度要求。
        与 case_fixer._rule_based_fix 共享同一检测逻辑，二者不会重复输出。
        """
        actions = []
        case_clusters = self.clusters.get("case", [])

        # 加载 Case 定义以引用实际内容
        case_content = {}
        if input_path:
            case_file = input_path / "case.json"
            if case_file.exists():
                try:
                    case_data = json.loads(case_file.read_text(encoding="utf-8"))
                    # 提取关键段
                    call_flow = case_data.get("call_flow", [])
                    if call_flow:
                        case_content["call_flow_summary"] = [
                            f"Step {s.get('step_number','?')}: {s.get('title','')} - {s.get('description','')[:100]}"
                            for s in call_flow[:10]
                        ]
                    constraints = case_data.get("constraints", [])
                    if constraints:
                        case_content["constraints"] = [
                            c.get("description", "")[:150] for c in constraints[:10]
                        ]
                    case_content["complexity_score"] = case_data.get("complexity_score", "N/A")
                except Exception:
                    pass

        for cluster in case_clusters:
            if cluster.frequency < 3:
                continue  # 孤例不构成过严

            priority = calc_priority(
                cluster.severity, cluster.dimension,
                cluster.frequency,
                sum(g.confidence for g in cluster.gradients) / len(cluster.gradients),
            )

            # 统计该维度在各评级中的分布
            rating_dist = {}
            for cid in cluster.conversation_ids:
                info = self.per_conv.get(cid, {})
                rating = info.get("ratings", {}).get(cluster.dimension, "未知")
                score = info.get("indicative_scores", {}).get(cluster.dimension, 0)
                rating_dist[rating] = rating_dist.get(rating, 0) + 1

            rating_summary = "、".join(f"{r}{c}场" for r, c in sorted(rating_dist.items()))

            # 构建证据：优先文本证据 → 回退统计证据
            real_evidence = []
            for g in cluster.gradients[:5]:
                for e in g.evidence[:2]:
                    if e and len(e) > 20 and e not in real_evidence:
                        real_evidence.append(e)
            # 文本证据不足时，补充评分分布作为统计证据 + Case 实际内容
            if len(real_evidence) < 2:
                real_evidence.append(
                    f"评分分布: {dict((r, rating_dist.get(r, 0)) for r in ['卓越','良好','合格','需改进','不合格'])}"
                )
                # 注入 Case 实际定义内容
                if case_content:
                    if case_content.get("complexity_score"):
                        real_evidence.append(f"Case 复杂度评分: {case_content['complexity_score']}")
                    for step in case_content.get("call_flow_summary", [])[:3]:
                        real_evidence.append(f"Case 流程: {step}")
                    for c in case_content.get("constraints", [])[:2]:
                        real_evidence.append(f"Case 约束: {c}")
                real_evidence.append(
                    f"涉及 {len(cluster.conversation_ids)} 场对话，累计 {cluster.frequency} 项缺陷"
                )

            # 矛盾检测：该维度是否同时在评测引擎建议中被标记为"区分力不足"
            contradiction_note = ""
            eval_dims_failing = set()
            for info in self.per_conv.values():
                for dim, rating in info.get("ratings", {}).items():
                    if rating in ("需改进", "不合格"):
                        eval_dims_failing.add(dim)
            # 计算该维度 model 归因占比
            model_count = sum(1 for c in self.clusters.get("model", []) if c.dimension == cluster.dimension)
            case_count = cluster.frequency
            if cluster.dimension in eval_dims_failing and model_count > case_count:
                contradiction_note = (
                    f"\n\n**⚠️ 矛盾提示**: 该维度在 {len(eval_dims_failing)} 个维度中低分比例较高，"
                    f"且 model 归因({model_count}项)多于 case 归因({case_count}项)。"
                    f"更可能是**模型真实问题**而非 Case 过严——建议优先处理对话模型优化建议"
                    f"（见 §四），再考虑调整 Case 定义。"
                )

            actions.append(OptimizationAction(
                action_id=f"case-overly-strict-{cluster.dimension}",
                source="case",
                dimension=cluster.dimension,
                priority=priority,
                title=f"Case 设计可能过严——{cluster.dimension} 维度 {cluster.frequency} 项缺陷",
                constitutional_principle=f"Case 定义合理性——{cluster.dimension}",
                violation_evidence=real_evidence[:5],
                critique_analysis=(
                    f"{cluster.dimension} 维度在 {len(cluster.conversation_ids)} 场对话中"
                    f"累计 {cluster.frequency} 项缺陷被归因为 Case 设计问题"
                    f"（评级分布: {rating_summary}）。"
                    f"可能是 Case 的流程步骤/约束定义过严，导致模型频繁被判定为不合格。"
                    f"{contradiction_note}"
                ),
                revision_proposal=(
                    f"1. 检查 {cluster.dimension} 相关的 call_flow 步骤是否过多/过细\n"
                    f"2. 考虑将部分步骤标记为 optional\n"
                    f"3. 考虑降低该维度的约束数量或放宽阈值"
                ),
                target_location=f"Case call_flow / constraints（{cluster.dimension} 维度）",
                expected_impact=f"预计可减少 {cluster.frequency} 项 Case 归因缺陷",
                effort_estimate="低",
                is_actionable=True,
                level="中建议",
                path="A",
            ))

        return actions

    # ---- Step 4: 综合去重 & 合并 ----

    def _deduplicate(self, actions: List[OptimizationAction]) -> List[OptimizationAction]:
        """去重相似建议，合并互补建议，按优先级排序，限制 top-20 输出。

        两阶段去重：
        1. 聚类级：每个缺陷聚类（source+dimension+defect_pattern 前缀）只保留 top-2
        2. 全局级：相同 target_location + 相似维度 → 合并 evidence，保留优先级最高
        """
        if not actions:
            return []

        # 阶段1: 聚类级去重——每个聚类只保留最优 2 条
        cluster_groups: Dict[str, List[OptimizationAction]] = {}
        for a in actions:
            # 用 action_id 的前两部分作为聚类键（如 "prompt-model-SAFETY-000"）
            parts = a.action_id.rsplit("-", 2)
            cluster_key = parts[0] if len(parts) >= 2 else a.action_id
            cluster_groups.setdefault(cluster_key, []).append(a)

        deduped: List[OptimizationAction] = []
        for ck, group in cluster_groups.items():
            group.sort(key=lambda x: -x.priority)
            deduped.extend(group[:2])  # 每个聚类最多 2 条

        # 阶段2: 全局去重——按 source+dimension 合并
        seen: Dict[str, OptimizationAction] = {}
        for a in sorted(deduped, key=lambda x: -x.priority):
            # Case 建议按维度去重（不同路径产出的同维度建议合并）
            if a.source == "case":
                key = f"case|{a.dimension}"
            else:
                key = f"{a.source}|{a.dimension}|{a.title[:30]}"

            if key not in seen:
                seen[key] = a
            else:
                existing = seen[key]
                existing.violation_evidence = list(set(
                    existing.violation_evidence + a.violation_evidence
                ))[:5]
                # 保留更高优先级的标题
                if a.priority > existing.priority:
                    existing.title = a.title
                    existing.priority = a.priority
                    existing.critique_analysis = a.critique_analysis

        result = sorted(seen.values(), key=lambda x: -x.priority)
        return result[:25]

    # ---- Step 5: 报告导出 ----

    def _export(self, output_path: Path) -> Path:
        """导出优化报告（Markdown + JSON）。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Markdown 报告
        md_path = output_path / "optimization_report.md"
        md_content = self._build_markdown_report(timestamp)
        md_path.write_text(md_content, encoding="utf-8")

        # JSON 数据
        json_path = output_path / "optimization_actions.json"
        json_data = self._build_json_output(timestamp)
        json_path.write_text(
            json.dumps(json_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # MANIFEST
        manifest_path = output_path / "MANIFEST.md"
        manifest_path.write_text(
            f"# 优化报告 — MANIFEST\n\n"
            f"**生成时间**: {timestamp}\n\n"
            f"## 文件列表\n\n"
            f"| 文件 | 说明 |\n"
            f"|------|------|\n"
            f"| [optimization_report.md](optimization_report.md) | 人类可读优化报告 |\n"
            f"| [optimization_actions.json](optimization_actions.json) | 结构化优化建议 |\n",
            encoding="utf-8",
        )

        return md_path

    def _build_markdown_report(self, timestamp: str) -> str:
        """构建 Markdown 优化报告。"""
        lines = [
            f"# 优化报告",
            f"",
            f"**生成时间**: {timestamp}",
            f"",
            f"## 阅读说明",
            f"",
            f"本报告由优化引擎自动生成，基于评测引擎对 {len(self.per_conv)} 场模拟对话的诊断结果，",
            f"结合原始 Case 定义和对话文本，为以下四个优化对象提供改进建议：",
            f"Case 定义、用户画像生成器、被评测对话模型、评测引擎自身。",
            f"",
            f"### 关键术语",
            f"",
            f"| 术语 | 含义 |",
            f"|------|------|",
            f"| **缺陷信号** | 评测引擎从单场对话中发现的单个问题（如\"身份核实步骤在第 3 轮被跳过\"）。本批次共 {len(self.gradients)} 条信号 |",
            f"| **缺陷主题** | 多个相似缺陷信号归并后的问题类别（如\"SAFETY 身份核实缺失\"在 5 场对话中反复出现）。本批次共 {sum(len(c) for c in self.clusters.values())} 个主题 |",
            f"| **规则分析（路径A）** | 纯统计计算得出的结论——通过率异常、相关性偏离、信号矛盾等。零 LLM 成本，结果确定 |",
            f"| **深度分析（路径B）** | LLM 综合分析原始数据生成的建议——prompt 改写、根因分析、副作用评估等。含具体修改文本 |",
            f"| **优先级** | 数值越高越应优先处理。公式：严重程度 × 维度权重 × 频次 × 置信度 |",
            f"| **建议等级** | 强建议（不修则评测无效）/ 中建议（修了有明确提升）/ 弱建议（锦上添花） |",
            f"",
            f"### 建议结构（CAI 三段式）",
            f"",
            f"每条建议按\"宪法原则 → 违规证据 → 批判分析 → 修正建议\"组织：",
            f"- **📜 宪法原则**：该建议针对的评测清单项（为什么这是问题）",
            f"- **📋 违规证据**：支撑该判定的对话片段或数据",
            f"- **🔍 批判分析**：根因诊断——是模型 prompt 问题？Case 设计问题？还是画像失真？",
            f"- **✏️ 修正建议**：具体、可执行的修改方案",
            f"",
            f"---",
            f"",
            f"**缺陷信号总数**: {len(self.gradients)} | **缺陷主题数**: {sum(len(c) for c in self.clusters.values())} | **建议数**: {len(self.actions)}",
            f"",
        ]

        # 执行摘要
        lines.append("## 一、执行摘要")
        lines.append("")

        # 按优先级排序
        sorted_actions = sorted(self.actions, key=lambda x: -x.priority)
        if sorted_actions:
            lines.append("| 优先级 | 对象 | 维度 | 标题 | 等级 | 路径 |")
            lines.append("|--------|------|------|------|------|------|")
            for a in sorted_actions[:10]:
                path_label = "规则分析" if a.path == "A" else "深度分析"
                lines.append(
                    f"| {a.priority:.1f} | {a.source} | {a.dimension} | "
                    f"{a.title[:40]} | {a.level} | {path_label} |"
                )
            lines.append("")

        # 按优化对象分组
        by_source: Dict[str, List[OptimizationAction]] = defaultdict(list)
        for a in sorted_actions:
            by_source[a.source].append(a)

        section_map = [
            ("case", "二、Case 定义优化建议"),
            ("simulator", "三、用户画像生成器优化建议"),
            ("model", "四、对话模型优化建议"),
            ("eval", "五、评测引擎自身优化建议"),
        ]

        for source, title in section_map:
            source_actions = by_source.get(source, [])
            if not source_actions:
                continue
            lines.append(f"## {title}")
            lines.append("")
            for a in source_actions:
                lines.extend(self._format_action_md(a))

        # 附录
        lines.append("## 六、附录：统计数据")
        lines.append("")
        lines.append(f"- 缺陷信号总数: {len(self.gradients)}")
        lines.append(f"- 缺陷主题总数: {sum(len(c) for c in self.clusters.values())}")
        for source, clusters in self.clusters.items():
            source_cn = {"model": "被评测模型", "case": "Case 定义", "simulator": "画像生成器"}.get(source, source)
            lines.append(f"- {source_cn} 相关主题: {len(clusters)} 个")
        lines.append(f"- 优化建议总数: {len(self.actions)}")
        lines.append(f"- 规则分析（路径A）: {sum(1 for a in self.actions if a.path == 'A')} 条")
        lines.append(f"- 深度分析（路径B）: {sum(1 for a in self.actions if a.path == 'B')} 条")

        return "\n".join(lines)

    def _format_action_md(self, a: OptimizationAction) -> List[str]:
        """将单条 OptimizationAction 格式化为 CAI 三段式 Markdown。"""
        return [
            f"### [{a.dimension}] {a.title} — Priority {a.priority:.1f}",
            f"",
            f"**📜 宪法原则**: {a.constitutional_principle}",
            f"",
            f"**📋 违规证据**:",
            *[f"- {e}" for e in a.violation_evidence[:3]],
            f"",
            f"**🔍 批判分析**: {a.critique_analysis}",
            f"",
            f"**✏️ 修正建议**: {a.revision_proposal}",
            f"",
            f"**预期效果**: {a.expected_impact}",
            f"**实施工作量**: {a.effort_estimate} | **等级**: {a.level} | **分析方式**: {'规则分析' if a.path == 'A' else '深度分析（LLM）'}",
            f"",
        ]

    def _build_json_output(self, timestamp: str) -> Dict[str, Any]:
        """构建结构化 JSON 输出。"""
        return {
            "generated_at": timestamp,
            "summary": {
                "n_gradients": len(self.gradients),
                "n_clusters": sum(len(c) for c in self.clusters.values()),
                "n_actions": len(self.actions),
                "by_source": {
                    source: len(clusters)
                    for source, clusters in self.clusters.items()
                },
                "by_path": {
                    "A": sum(1 for a in self.actions if a.path == "A"),
                    "B": sum(1 for a in self.actions if a.path == "B"),
                },
            },
            "actions": [
                {
                    "action_id": a.action_id,
                    "source": a.source,
                    "dimension": a.dimension,
                    "priority": round(a.priority, 2),
                    "title": a.title,
                    "constitutional_principle": a.constitutional_principle,
                    "violation_evidence": a.violation_evidence,
                    "critique_analysis": a.critique_analysis,
                    "revision_proposal": a.revision_proposal,
                    "target_location": a.target_location,
                    "expected_impact": a.expected_impact,
                    "effort_estimate": a.effort_estimate,
                    "level": a.level,
                    "path": a.path,
                }
                for a in self.actions
            ],
        }


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="优化引擎 v1 — 评测驱动的优化建议生成")
    parser.add_argument(
        "--input-dir", required=True,
        help="评测导出目录（含 optimization_feed.json 等）",
    )
    parser.add_argument(
        "--output", default="data/optimization/latest",
        help="优化报告输出目录",
    )
    args = parser.parse_args()

    engine = OptimizationEngine()
    result = engine.run(args.input_dir, args.output)

    print(f"优化引擎完成:")
    print(f"  梯度: {result['n_gradients']}")
    print(f"  聚类: {result['n_clusters']}")
    print(f"  建议: {result['n_actions']}")
    print(f"  报告: {result['report_path']}")


if __name__ == "__main__":
    main()
