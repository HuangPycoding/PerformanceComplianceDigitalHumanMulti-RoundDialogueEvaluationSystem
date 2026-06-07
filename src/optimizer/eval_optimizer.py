"""评测引擎自身优化器 — 路径 A（规则驱动）

消费 ChecklistEvolver 跨批次统计数据 + SelfReliabilityChecker 自验证报告，
产出评测引擎自身的优化建议（清单项校准/权重校准/缺陷转化/区分力分析）。

纯规则驱动，零 LLM 成本。仅在需要 Judge prompt 文本改写时走路径 B。
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.optimizer.utils import (
    DIMENSION_WEIGHTS,
    compute_discrimination_stats,
    frequency_distribution,
    pearson_r,
)


# 维度中文名
_DIM_CN = {
    "SAFETY": "安全合规", "TASK_COMPLETION": "任务达成",
    "FLOW_COVERAGE": "流程覆盖", "KNOWLEDGE": "知识准确性",
    "CONSTRAINT": "约束遵守", "EFFICIENCY": "对话效率",
    "SENTIMENT": "情感适配", "ROLE": "角色一致性",
    "OPENING": "开场白合规",
}


def analyze_checklist_health(
    pass_rates: Dict[str, float],
    na_rates: Optional[Dict[str, float]] = None,
    partial_rates: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """分析清单项健康度——路径 A 直接输出。

    Args:
        pass_rates: {item_id: YES+MOSTLY_YES 占比}
        na_rates: {item_id: NOT_APPLICABLE 占比}
        partial_rates: {item_id: PARTIAL 占比}

    Returns:
        区分力分析结果，含建议降权/删除/改写的项列表
    """
    stats = compute_discrimination_stats(pass_rates, na_rates, partial_rates)

    result = {
        "summary": {},
        "recommendations": [],
    }

    if stats["low_discrimination"]:
        result["summary"]["low_discrimination_count"] = len(stats["low_discrimination"])
        result["recommendations"].append({
            "type": "prune_or_reduce_weight",
            "items": stats["low_discrimination"],
            "reason": "通过率 > 95%，区分力不足，建议降权或删除",
            "severity": "moderate",
        })

    if stats["high_discrimination"]:
        result["summary"]["high_discrimination_count"] = len(stats["high_discrimination"])
        result["recommendations"].append({
            "type": "check_definition",
            "items": stats["high_discrimination"],
            "reason": "通过率 < 30%，可能定义过严，建议检查清单项定义是否合理",
            "severity": "moderate",
        })

    if stats["high_na"]:
        result["summary"]["high_na_count"] = len(stats["high_na"])
        result["recommendations"].append({
            "type": "narrow_scope",
            "items": stats["high_na"],
            "reason": "不适用率 > 80%，过于场景特定，建议缩小适用范围",
            "severity": "minor",
        })

    if stats["unclear"]:
        result["summary"]["unclear_count"] = len(stats["unclear"])
        result["recommendations"].append({
            "type": "rewrite_description",
            "items": stats["unclear"],
            "reason": "PARTIAL 率 > 30%，清单项措辞可能模糊，建议改写",
            "severity": "minor",
        })

    return result


def analyze_weight_calibration(
    source_yes_ratios: Dict[str, Dict[str, float]],
    total_scores: Dict[str, float],
    dim_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """分析权重校准建议——路径 A 直接输出。

    计算 SourceWeights 中各 source 的 YES 率与总分的 Pearson 相关性，
    判断当前权重是否合理。

    Args:
        source_yes_ratios: {conversation_id: {"case": ratio, "simulator": ratio, ...}}
        total_scores: {conversation_id: total_score}
        dim_scores: {dimension: {conversation_id: score}}

    Returns:
        权重校准建议，含 Pearson r 值和调整方向
    """
    result = {"source_weights": {}, "dimension_weights": {}}

    if len(source_yes_ratios) < 5:
        result["_note"] = "样本不足（<5），跳过权重校准"
        return result

    # SourceWeights 相关性
    for source in ["case", "simulator", "llm_supplement"]:
        ratios = []
        scores = []
        for cid, src_ratios in source_yes_ratios.items():
            if cid in total_scores and source in src_ratios:
                ratios.append(src_ratios[source])
                scores.append(total_scores[cid])

        if len(ratios) >= 5:
            r = pearson_r(ratios, scores)
            result["source_weights"][source] = {
                "pearson_r": round(r, 3),
                "n_samples": len(ratios),
                "recommendation": _interpret_source_correlation(source, r),
            }

    # DimWeights 相关性
    if dim_scores:
        for dim in DIMENSION_WEIGHTS:
            dim_vals = []
            total_vals = []
            for cid in total_scores:
                if dim in dim_scores and cid in dim_scores[dim]:
                    dim_vals.append(dim_scores[dim][cid])
                    total_vals.append(total_scores[cid])

            if len(dim_vals) >= 5:
                r = pearson_r(dim_vals, total_vals)
                result["dimension_weights"][dim] = {
                    "pearson_r": round(r, 3),
                    "n_samples": len(dim_vals),
                    "current_weight": DIMENSION_WEIGHTS.get(dim, 1.0),
                    "recommendation": _interpret_dim_correlation(dim, r),
                }

    return result


def _interpret_source_correlation(source: str, r: float) -> str:
    """解释 SourceWeight 的 Pearson 相关性。"""
    current = {"case": 0.6, "simulator": 1.5, "llm_supplement": 1.2}.get(source, 1.0)

    if abs(r) < 0.3:
        return f"相关性弱 (r={r:.3f})，当前权重 {current} 与评分关联不明显，建议保持观察"
    if r > 0.3:
        return f"正相关 (r={r:.3f})，当前权重 {current} 偏高，建议降至 {max(0.3, current - 0.2):.1f}"
    else:
        return f"负相关 (r={r:.3f})，当前权重 {current} 可能反向，建议提升至 {min(2.0, current + 0.3):.1f}"


def _interpret_dim_correlation(dim: str, r: float) -> str:
    """解释 DimWeight 的 Pearson 相关性。"""
    cn = _DIM_CN.get(dim, dim)
    current = DIMENSION_WEIGHTS.get(dim, 1.0)

    if abs(r) < 0.2:
        return f"{cn} 与总分相关性弱 (r={r:.3f})，当前权重 {current} 合理"
    if r > 0.2:
        return f"{cn} 与总分正相关 (r={r:.3f})，当前权重 {current} 合理"
    else:
        return f"{cn} 与总分负相关 (r={r:.3f})，当前权重 {current} 可能偏低，建议升至 {min(2.0, current + 0.2):.1f}"


def analyze_defect_conversion(
    defect_frequencies: Dict[str, int],
    min_frequency: int = 5,
) -> List[Dict[str, Any]]:
    """分析高频 additional_defects 是否应转化为正式清单项。

    Args:
        defect_frequencies: {defect_description: count}
        min_frequency: 最低转化频率阈值

    Returns:
        建议转化的缺陷列表，含建议的 item_id/weight
    """
    candidates = []
    for desc, freq in sorted(defect_frequencies.items(), key=lambda x: -x[1]):
        if freq < min_frequency:
            break
        candidates.append({
            "description": desc,
            "frequency": freq,
            "suggested_item_id": f"evolved_{_slugify(desc[:30])}",
            "suggested_weight": 1.3,  # pattern_mined 默认权重
            "source": "pattern_mined",
        })
    return candidates


def _slugify(text: str) -> str:
    """简单的中文 slug 生成。"""
    return "".join(c for c in text if c.isalnum() or c in "_-")


def load_evolver_stats(storage_dir: str = "data/checklist_evolution") -> Dict[str, Any]:
    """加载 ChecklistEvolver 的跨批次统计数据。

    Returns:
        {
            "defect_frequencies": {desc: count},
            "total_defects": int,
            "defects_by_dimension": {dim: count},
        }
    """
    storage = Path(storage_dir)
    defects_file = storage / "accumulated_defects.jsonl"

    if not defects_file.exists():
        return {"defect_frequencies": {}, "total_defects": 0, "defects_by_dimension": {}}

    defect_freq: Dict[str, int] = {}
    dim_freq: Dict[str, int] = {}
    total = 0

    with open(defects_file, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                desc = record.get("description", "")
                dim = record.get("dimension", "")
                if desc:
                    defect_freq[desc] = defect_freq.get(desc, 0) + 1
                if dim:
                    dim_freq[dim] = dim_freq.get(dim, 0) + 1
                total += 1
            except json.JSONDecodeError:
                continue

    return {
        "defect_frequencies": defect_freq,
        "total_defects": total,
        "defects_by_dimension": dim_freq,
    }
