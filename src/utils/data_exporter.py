"""数据导出模块 — 评测链路各节点数据输出为结构化文件，方便人工浏览和判断

支持的导出节点:
  NODE_NAMES = ["case", "profiles", "conversation", "evaluation", "batch_summary", "full_pipeline"]

导出格式:
  - JSON (机器可读，结构化)
  - Markdown (人类可读，适合浏览)
  - CSV (表格数据，适合 Excel)

用法:
  from src.utils.data_exporter import DataExporter
  exporter = DataExporter(output_dir="data/exports/batch_001")
  exporter.export_case(case)           # 导出 Case 指令
  exporter.export_profiles(profiles)   # 导出用户画像
  exporter.export_conversation(conv)   # 导出单条对话
  exporter.export_evaluation(result)   # 导出单条评测结果
  exporter.export_batch(convs, results) # 导出批次汇总
"""

import json
import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.models.case import Case
from src.models.conversation import Conversation
from src.models.evaluation import EvalResult

NODE_NAMES = ["case", "profiles", "conversation", "evaluation", "batch_summary", "optimization_feed", "full_pipeline"]

# 维度中文名映射（用于 optimization_feed 可读性）
_DIM_CN = {
    "SAFETY": "安全合规", "TASK_COMPLETION": "任务达成", "FLOW_COVERAGE": "流程覆盖",
    "KNOWLEDGE": "知识准确性", "CONSTRAINT": "约束遵守", "EFFICIENCY": "对话效率",
    "SENTIMENT": "情感适配", "ROLE": "角色一致性", "OPENING": "开场白合规",
}

# 严重程度关键词（用于归因项自动分级）
_SEVERITY_MAJOR_KW = ["不合格", "安全", "泄露", "严重", "必须立即", "钳制"]
_SEVERITY_MODERATE_KW = ["需改进", "偏差", "矛盾", "遗漏", "缺失", "错误", "跳过"]


def _derive_severity(attr) -> str:
    """从归因项的 description + category 推导严重程度"""
    text = f"{attr.category} {getattr(attr, 'description', '')}".lower()
    if any(kw in text for kw in _SEVERITY_MAJOR_KW):
        return "major"
    if any(kw in text for kw in _SEVERITY_MODERATE_KW):
        return "moderate"
    if getattr(attr, 'confidence', 0.5) >= 0.8:
        return "moderate"
    return "minor"


def _count_by_field(items: list, field: str, key_func=None) -> dict:
    """统计列表中某字段的频次分布（支持 dict 和 object 两种 item 类型）"""
    counts: Dict[str, int] = {}
    for item in items:
        if key_func:
            key = key_func(item)
        elif isinstance(item, dict):
            key = item.get(field, "unknown")
        else:
            key = getattr(item, field, "unknown")
        if isinstance(key, str):
            key = key  # keep as-is
        else:
            key = str(key)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


class DataExporter:
    """评测链路数据导出器 — 每个节点独立导出，支持 JSON/MD/CSV 三种格式"""

    def __init__(self, output_dir: str = "data/exports", batch_id: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.batch_id = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.node_dir = self.output_dir / self.batch_id
        os.makedirs(self.node_dir, exist_ok=True)
        self._manifest: List[Dict[str, str]] = []

    # ---- 节点 1: Case 指令 ----

    def export_case(self, case: Case, fmt: str = "md") -> str:
        """导出 Case 指令定义"""
        data = {
            "id": case.id,
            "title": case.title,
            "business_line": case.business_line,
            "complexity_score": case.complexity_score,
            "role": case.role,
            "task": case.task,
            "opening_line": case.opening_line,
            "call_flow": [
                {
                    "step_number": s.step_number,
                    "title": s.title,
                    "description": getattr(s, "description", ""),
                    "is_optional": getattr(s, "is_optional", False),
                    "branches": [
                        {"condition": b.condition, "action": getattr(b, "action", ""),
                         "target_step": b.target_step}
                        for b in getattr(s, "branching", [])
                    ],
                }
                for s in case.call_flow
            ],
            "knowledge_points": [
                {"topic": k.topic, "content": k.content} for k in case.knowledge_points
            ],
            "constraints": [
                {"type": c.type, "description": c.description,
                 "checkable_by_rule": c.checkable_by_rule}
                for c in case.constraints
            ],
        }
        path = self._write("case", data, fmt)
        self._add_manifest("case", path)
        return path

    # ---- 节点 2: 用户画像 ----

    def export_profiles(self, profiles: Dict[int, List[Any]], fmt: str = "json") -> str:
        """导出所有画像"""
        data = {}
        for cid, plist in profiles.items():
            data[str(cid)] = [
                {
                    "label": getattr(p, "label", f"P{i+1}"),
                    "persona_text": getattr(p, "persona_text", ""),
                    "sampled_vector": getattr(p, "sampled_vector", None),
                    "verified_vector": getattr(p, "verified_vector", None),
                    "adversarial_strategy": getattr(p, "adversarial_strategy", []) or [],
                    "self_check_d_sv": getattr(p, "self_check_d_sv", None),
                }
                for i, p in enumerate(plist)
            ]
        path = self._write("profiles", data, fmt)
        self._add_manifest("profiles", path)
        return path

    # ---- 节点 3: 对话 ----

    def export_conversation(self, conv: Conversation, index: int = 0, fmt: str = "md") -> str:
        """导出单条对话（含完整文本和元数据）"""
        conv_id = conv.id or f"conv_{index}"
        data = {
            "id": conv_id,
            "case_id": conv.case_id,
            "status": conv.status,
            "total_turns": conv.total_turns,
            "duration_seconds": getattr(conv, "duration_seconds", 0),
            "adversarial_strategies": getattr(conv, "adversarial_strategies", []) or [],
            "sampled_vector": conv.sampled_vector,
            "consistency": conv.consistency,
            "turns": [
                {
                    "turn": t.turn_number,
                    "speaker": t.speaker,
                    "content": t.content,
                    "parsed_tags": t.parsed_tags if t.parsed_tags else {},
                }
                for t in conv.turns
            ],
        }

        if fmt == "md":
            path = self._write_conv_md(conv, conv_id)
        elif fmt == "json":
            path = self._write_json(f"conversation_{conv_id}", data)
        else:
            path = self._write_csv_conv(conv, conv_id)

        self._add_manifest("conversation", path)
        return path

    def _write_conv_md(self, conv: Conversation, conv_id: str) -> str:
        """对话专属 Markdown 格式"""
        lines = [f"# 对话: {conv_id}", f"", f"**Case ID**: {conv.case_id}",
                 f"**状态**: {conv.status} | **轮次**: {conv.total_turns}",
                 f"**对抗策略**: {', '.join(getattr(conv, 'adversarial_strategies', []) or ['无'])}",
                 f"**一致性 d_sa**: {conv.consistency.get('path_a_d_sa', 'N/A') if conv.consistency else 'N/A'}",
                 f""]
        for t in conv.turns:
            sp = "客服" if t.speaker == "system" else "用户"
            lines.append(f"**T{t.turn_number} [{sp}]**: {t.content}")
            lines.append(f"")
        path = self.node_dir / f"conversation_{conv_id}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)

    def _write_csv_conv(self, conv: Conversation, conv_id: str) -> str:
        """对话 CSV 格式（每轮一行）"""
        path = self.node_dir / f"conversation_{conv_id}.csv"
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["turn", "speaker", "content", "parsed_tags_keys"])
            for t in conv.turns:
                tags_keys = ",".join(t.parsed_tags.keys()) if t.parsed_tags else ""
                writer.writerow([t.turn_number, t.speaker, t.content, tags_keys])
        return str(path)

    # ---- 节点 4: 评测结果 ----

    def export_evaluation(self, result: EvalResult, conv_id: str = "", fmt: str = "json") -> str:
        """导出单条评测结果"""
        data = {
            "conversation_id": conv_id or result.conversation_id,
            "case_id": result.case_id,
            "ratings": result.ratings,
            "indicative_scores": result.indicative_scores,
            "total_indicative_score": result.total_indicative_score,
            "total_score_100": result.total_score_100,
            "confidence": {
                "overall": result.confidence.overall,
                "level": result.confidence.level,
                "needs_human_review": result.confidence.needs_human_review,
                "signal_conflict_count": result.confidence.signal_conflict_count,
                "evidence_empty_ratio": result.confidence.evidence_empty_ratio,
                "simulator_tier": result.confidence.simulator_tier,
                "per_dimension": result.confidence.per_dimension,
            } if result.confidence else None,
            "surface_compliance_flags": result.surface_compliance_flags,
            "rule_check_issues": result.rule_check_issues,
            "cross_validation_alerts": [
                {"severity": a.severity, "dimension": a.dimension, "description": a.description}
                for a in result.cross_validation_alerts
            ],
            "meta_check_alerts": [
                {"severity": a.severity, "check_type": a.check_type, "description": a.description,
                 "dimensions": a.dimensions}
                for a in result.meta_check_alerts
            ],
            "tier1_constraint_count": result.tier1_constraint_count,
            "llm_constraint_count": result.llm_constraint_count,
            "dimension_checklists": {
                dim: {
                    "items": [
                        {
                            "item_id": i.item_id,
                            "description": i.description,
                            "source": i.source,
                            "status": i.status,
                            "evidence": i.evidence,
                            "reasoning": i.reasoning,  # CoT 推理全文
                            "signal_consistency": i.signal_consistency,
                            "weight": i.weight,
                        }
                        for i in cl.items
                    ],
                }
                for dim, cl in result.dimension_checklists.items()
            },
            "attributions": [
                {
                    "source": a.source,
                    "category": a.category,
                    "description": a.description,
                    "confidence": a.confidence,
                    "is_actionable": a.is_actionable,
                    "evidence_chain": a.evidence_chain,
                    "suggested_actions": a.suggested_actions,
                    "severity": _derive_severity(a),
                }
                for a in result.attributions
            ],
            "attributions_summary": {
                "total": len(result.attributions),
                "model": sum(1 for a in result.attributions if a.source == "model"),
                "case": sum(1 for a in result.attributions if a.source == "case"),
                "simulator": sum(1 for a in result.attributions if a.source == "simulator"),
                "actionable": sum(1 for a in result.attributions if a.is_actionable),
                "by_dimension": _count_by_field(result.attributions, "category"),
                "by_severity": _count_by_field(result.attributions, "severity",
                    key_func=_derive_severity),
            },
            "improvement_suggestions": result.improvement_suggestions[:10],
        }
        path = self._write(f"evaluation_{conv_id}", data, fmt)
        self._add_manifest("evaluation", path)
        return path

    # ---- 节点 5: 批次汇总 ----

    def export_batch_summary(
        self, conversations: List[Conversation], eval_results: List[EvalResult],
        batch_report: Optional[Dict] = None, fmt: str = "md"
    ) -> str:
        """导出批次汇总报告"""
        # 评分汇总
        scores = [r.total_indicative_score for r in eval_results if r is not None]
        scores_100 = [r.total_score_100 for r in eval_results if r is not None]
        all_ratings: Dict[str, List[str]] = {}
        for r in eval_results:
            if r is None:
                continue
            for dim, rating in r.ratings.items():
                all_ratings.setdefault(dim, []).append(rating)
        conf_levels = [r.confidence.level for r in eval_results if r and r.confidence]

        data = {
            "batch_id": self.batch_id,
            "n_conversations": len(conversations),
            "n_evaluated": len([r for r in eval_results if r is not None]),
            "score_stats": {
                "mean": sum(scores) / len(scores) if scores else 0,
                "min": min(scores) if scores else 0,
                "max": max(scores) if scores else 0,
                "median": sorted(scores)[len(scores)//2] if scores else 0,
            },
            "score_stats_100": {
                "mean": round(sum(scores_100) / len(scores_100)) if scores_100 else 0,
                "min": min(scores_100) if scores_100 else 0,
                "max": max(scores_100) if scores_100 else 0,
                "median": sorted(scores_100)[len(scores_100)//2] if scores_100 else 0,
            },
            "rating_distribution": {
                dim: {
                    "卓越": ratings.count("卓越"),
                    "良好": ratings.count("良好"),
                    "合格": ratings.count("合格"),
                    "需改进": ratings.count("需改进"),
                    "不合格": ratings.count("不合格"),
                }
                for dim, ratings in all_ratings.items()
            },
            "confidence_distribution": {
                "high": conf_levels.count("high"),
                "medium": conf_levels.count("medium"),
                "low": conf_levels.count("low"),
                "unreliable": conf_levels.count("unreliable"),
            },
            "needs_human_review_count": sum(
                1 for r in eval_results if r and r.confidence and r.confidence.needs_human_review
            ),
            "batch_report": batch_report,
        }
        path = self._write("batch_summary", data, fmt)
        self._add_manifest("batch_summary", path)
        return path

    # ---- 节点 6: 优化引擎对接数据 ----

    def export_optimization_feed(
        self, eval_results: List[EvalResult], conversations: Optional[List[Conversation]] = None
    ) -> str:
        """导出聚合的 OptimizationFeed JSON —— 优化引擎的唯一输入源。

        将批次内所有 EvalResult 的归因数据汇聚为一份结构化 JSON，
        含完整的 AttributionItem 列表、逐对话评分、置信度分布、
        维度缺陷统计。供下游优化引擎独立读取。

        Args:
            eval_results: 本批次全部评测结果
            conversations: 对话列表（可选，用于附加轮次/对抗策略等上下文）

        Returns:
            导出文件路径
        """
        all_attributions: List[Dict] = []
        per_conv: Dict[str, Dict] = {}
        score_dist: List[float] = []
        conf_levels: List[str] = []
        reliable_count = 0

        for i, r in enumerate(eval_results):
            if r is None:
                continue

            conv_id = r.conversation_id or f"conv_{i}"

            # 收集归因项
            conv_attrs = []
            for a in r.attributions:
                attr_dict = {
                    "source": a.source,
                    "category": a.category,
                    "dimension_cn": _DIM_CN.get(a.category, a.category),
                    "description": a.description,
                    "confidence": a.confidence,
                    "is_actionable": a.is_actionable,
                    "evidence_chain": a.evidence_chain,
                    "suggested_actions": a.suggested_actions,
                    "severity": _derive_severity(a),
                }
                all_attributions.append(attr_dict)
                conv_attrs.append(attr_dict)

            # 逐对话上下文
            conv_info: Dict[str, Any] = {
                "total_score": r.total_indicative_score,
                "total_score_100": r.total_score_100,
                "ratings": r.ratings,
                "indicative_scores": r.indicative_scores,
                "confidence_level": r.confidence.level if r.confidence else "unknown",
                "is_reliable": r.confidence.is_reliable if r.confidence else False,
                "n_attributions": len(conv_attrs),
                "attributions_by_source": {
                    "model": sum(1 for a in conv_attrs if a["source"] == "model"),
                    "case": sum(1 for a in conv_attrs if a["source"] == "case"),
                    "simulator": sum(1 for a in conv_attrs if a["source"] == "simulator"),
                },
            }

            # 附加对话元数据
            if conversations and i < len(conversations):
                c = conversations[i]
                conv_info["total_turns"] = c.total_turns
                conv_info["adversarial_strategies"] = getattr(c, "adversarial_strategies", []) or []
                conv_info["turn_texts"] = [
                    f"T{t.turn_number} [{t.speaker}]: {t.content[:200]}"
                    for t in c.turns[-6:]  # 最后 6 轮供快速浏览
                ]

            per_conv[conv_id] = conv_info
            score_dist.append(r.total_indicative_score)

            if r.confidence:
                conf_levels.append(r.confidence.level)
                if r.confidence.is_reliable:
                    reliable_count += 1

        # 维度级缺陷分布
        dim_defects: Dict[str, Dict[str, int]] = {}
        for a in all_attributions:
            dim = a["dimension_cn"]
            source = a["source"]
            sev = a["severity"]
            dim_defects.setdefault(dim, {"model": 0, "case": 0, "simulator": 0,
                                          "major": 0, "moderate": 0, "minor": 0})
            dim_defects[dim][source] = dim_defects[dim].get(source, 0) + 1
            dim_defects[dim][sev] = dim_defects[dim].get(sev, 0) + 1

        # 构建完整 feed
        n_results = len([r for r in eval_results if r is not None])
        feed = {
            "batch_id": self.batch_id,
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "n_results": n_results,
                "n_attributions": len(all_attributions),
                "by_source": _count_by_field(all_attributions, "source"),
                "by_dimension": _count_by_field(all_attributions, "dimension_cn"),
                "by_severity": _count_by_field(all_attributions, "severity"),
                "actionable_count": sum(1 for a in all_attributions if a["is_actionable"]),
                "reliable_count": reliable_count,
                "reliable_ratio": round(reliable_count / n_results, 2) if n_results > 0 else 0,
            },
            "score_distribution": {
                "mean": round(sum(score_dist) / len(score_dist), 1) if score_dist else 0,
                "min": min(score_dist) if score_dist else 0,
                "max": max(score_dist) if score_dist else 0,
                "median": round(sorted(score_dist)[len(score_dist) // 2], 1) if score_dist else 0,
            },
            "confidence_overview": {
                "level_distribution": _count_by_field(
                    [{"level": l} for l in conf_levels], "level"
                ),
                "average_confidence": round(
                    sum(a["confidence"] for a in all_attributions) / len(all_attributions), 3
                ) if all_attributions else 0,
            },
            "dimension_defect_matrix": dim_defects,
            "attributions": all_attributions,
            "per_conversation": per_conv,
            "drift_alerts": [],  # v1 占位，Phase 3.1+ 填充
        }

        path = self._write("optimization_feed", feed, "json")
        self._add_manifest("optimization_feed", path)
        return path

    # ---- 全链路导出 ----

    def export_full_pipeline(
        self, case: Case, profiles: Dict[int, List[Any]],
        conversations: List[Conversation], eval_results: List[EvalResult],
        conv_time: float = 0, eval_time: float = 0
    ) -> str:
        """一键导出全链路所有节点数据"""
        self.export_case(case)
        self.export_profiles(profiles)

        for i, (conv, result) in enumerate(zip(conversations, eval_results)):
            conv_id = conv.id or f"conv_{i}"
            self.export_conversation(conv, i)
            if result:
                self.export_evaluation(result, conv_id)

        from src.eval.drift_monitor import BatchAnalyzer
        analyzer = BatchAnalyzer()
        ra = [
            {"ratings": r.ratings, "total_score": r.total_indicative_score,
             "is_reliable": r.confidence.is_reliable if r.confidence else False,
             "confidence_level": r.confidence.level if r.confidence else "unknown"}
            for r in eval_results if r is not None
        ]
        batch_report = analyzer.analyze(ra) if ra else {}
        self.export_batch_summary(conversations, eval_results, batch_report)

        # 导出优化引擎对接数据（OptimizationFeed JSON）
        self.export_optimization_feed(eval_results, conversations)

        return self._write_manifest(conv_time, eval_time)

    # ---- 内部方法 ----

    def _write(self, name: str, data: Any, fmt: str) -> str:
        if fmt == "json":
            return self._write_json(name, data)
        elif fmt == "md":
            return self._write_md(name, data)
        elif fmt == "csv":
            return self._write_csv(name, data)
        else:
            return self._write_json(name, data)

    def _write_json(self, name: str, data: Any) -> str:
        path = self.node_dir / f"{name}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)

    def _write_md(self, name: str, data: Any) -> str:
        """简单 Markdown 表格导出"""
        path = self.node_dir / f"{name}.md"
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"# {name}\n\n")
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(v, (str, int, float, bool)):
                        f.write(f"**{k}**: {v}\n\n")
                    elif isinstance(v, dict):
                        f.write(f"## {k}\n\n")
                        for k2, v2 in v.items():
                            f.write(f"- **{k2}**: {v2}\n")
                        f.write("\n")
                    elif isinstance(v, list):
                        f.write(f"## {k} ({len(v)} items)\n\n")
                        for item in v[:20]:
                            f.write(f"- {item}\n")
                        f.write("\n")
        return str(path)

    def _write_csv(self, name: str, data: Any) -> str:
        """简单 CSV 导出"""
        path = self.node_dir / f"{name}.csv"
        if isinstance(data, dict) and all(isinstance(v, dict) for v in data.values()):
            # dict of dicts -> CSV
            all_keys = sorted(set().union(*(d.keys() for d in data.values())))
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["key"] + all_keys)
                writer.writeheader()
                for k, v in data.items():
                    row = {"key": k}
                    row.update(v)
                    writer.writerow(row)
        return str(path)

    def _add_manifest(self, node_name: str, path: str):
        self._manifest.append({"node": node_name, "path": path, "format": Path(path).suffix})

    def _write_manifest(self, conv_time: float = 0, eval_time: float = 0) -> str:
        """写入清单文件（索引所有导出文件）"""
        path = self.node_dir / "MANIFEST.md"
        lines = [
            f"# 导出清单 — {self.batch_id}",
            f"",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**对话耗时**: {conv_time:.0f}s | **评测耗时**: {eval_time:.0f}s",
            f"",
            f"## 节点数据",
            f"",
            f"| 节点 | 格式 | 文件 |",
            f"|------|------|------|",
        ]
        for item in self._manifest:
            fname = Path(item["path"]).name
            lines.append(f"| {item['node']} | {item['format']} | [{fname}]({fname}) |")
        lines.append(f"")
        lines.append(f"## 目录结构")
        lines.append(f"```")
        lines.append(f"{self.node_dir}/")
        for item in self._manifest:
            fname = Path(item["path"]).name
            lines.append(f"  ├── {fname}")
        lines.append(f"  └── MANIFEST.md")
        lines.append(f"```")
        lines.append(f"")
        lines.append(f"**总计**: {len(self._manifest)} 个文件")
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path)
