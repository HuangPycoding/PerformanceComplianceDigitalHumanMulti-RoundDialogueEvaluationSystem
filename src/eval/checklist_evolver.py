"""清单进化引擎 — v1 数据积累 + Phase 3.1 半自动分析 + Phase 3.2 自动转化"""
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.config import DATA_DIR


class ChecklistEvolver:
    """清单进化管理器

    v1: 被动积累 additional_defects → 落盘 JSON
    Phase 3.1: 半自动分析（文本去重 + 频率统计，零 LLM）
    Phase 3.2: 规则自动转化（高频缺陷 ≥5 次 → 自动转清单项）
    """

    def __init__(self, storage_dir: Optional[str] = None):
        if storage_dir is None:
            storage_dir = str(DATA_DIR / "checklist_evolution")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.defects_file = self.storage_dir / "accumulated_defects.jsonl"
        self.checklist_file = self.storage_dir / "evolved_checklist_items.json"
        self.stats_file = self.storage_dir / "frequency_stats.json"

    # ---- Phase 3.0 (v1): 被动积累 ----

    def accumulate_defects(
        self,
        conversation_id: str,
        case_id: int,
        dimension: str,
        additional_defects: List[Dict[str, Any]],
    ) -> None:
        """将 additional_defects 追加入 JSONL 文件"""
        if not additional_defects:
            return

        timestamp = datetime.now().isoformat()
        for defect in additional_defects:
            record = {
                "timestamp": timestamp,
                "conversation_id": conversation_id,
                "case_id": case_id,
                "dimension": dimension,
                "description": defect.get("description", ""),
                "severity": defect.get("severity", "一般"),
                "turn": defect.get("turn", 0),
                "attribution": defect.get("attribution", "Model"),
            }
            with open(self.defects_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def accumulate_from_result(self, result: Dict[str, Any]) -> None:
        """从单条 EvalResult 积累所有维度的 additional_defects"""
        conv_id = result.get("conversation_id", "")
        case_id = result.get("case_id", 0)

        for dim, defects in result.get("additional_defects_by_dim", {}).items():
            if isinstance(defects, list):
                self.accumulate_defects(conv_id, case_id, dim, defects)

    # ---- Phase 3.1: 半自动分析 ----

    def analyze_defects(
        self,
        min_frequency: int = 3,
        similarity_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """半自动分析：文本去重 + 频率统计 + 转化候选

        Returns:
            {
                "total_defects": int,
                "unique_defects": int,
                "top_defects": [...],
                "conversion_candidates": [...],
            }
        """
        all_defects = self._load_all_defects()
        if not all_defects:
            return {"total_defects": 0, "unique_defects": 0, "top_defects": [], "conversion_candidates": []}

        # 按描述去重（零 LLM：嵌入相似度）+ 时间衰减
        descriptions = [d["description"] for d in all_defects]
        timestamps = [d.get("timestamp", "") for d in all_defects]
        clusters = self._cluster_by_similarity(descriptions, similarity_threshold, timestamps, all_defects)

        # 频率统计（使用时间衰减后的 weighted_count）
        freq_stats = []
        for cluster in clusters:
            representative = cluster["representative"]
            wcount = cluster.get("weighted_count", cluster.get("count", 1))
            raw_count = cluster.get("raw_count", wcount)
            severities = cluster["severities"]
            dimensions = cluster["dimensions"]
            freq_stats.append({
                "description": representative,
                "count": round(wcount, 1),
                "raw_count": raw_count,
                "severities": Counter(severities),
                "dimensions": Counter(dimensions),
            })

        freq_stats.sort(key=lambda x: x["count"], reverse=True)

        # 转化候选：衰减后频率 ≥ min_frequency
        candidates = [fs for fs in freq_stats if fs["count"] >= min_frequency]
        # 去掉与 checklist_items 中已有项高相似度的
        candidates = self._filter_existing(candidates)

        # 保存统计
        stats = {
            "analyzed_at": datetime.now().isoformat(),
            "total_defects": len(all_defects),
            "unique_clusters": len(clusters),
            "top_defects": freq_stats[:20],
            "conversion_candidates": candidates[:10],
        }
        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        return stats

    def _load_all_defects(self) -> List[Dict[str, Any]]:
        """加载全部积累的缺陷"""
        if not self.defects_file.exists():
            return []
        defects = []
        with open(self.defects_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        defects.append(json.loads(line))
                    except json.JSONDecodeError:
                        import logging as _logging
                        _logging.warning(f"checklist_evolver: skipping malformed JSON line in {self.defects_file}")
                        continue
        return defects

    def _time_weight(self, timestamp_str: str) -> float:
        """时间衰减权重：exp(-days/30)，今天=1.0，30天前≈0.37，60天前≈0.14"""
        if not timestamp_str:
            return 0.5
        try:
            ts = datetime.fromisoformat(timestamp_str)
            days_ago = (datetime.now() - ts).days
            if days_ago < 0:
                return 1.0
            import math
            return math.exp(-days_ago / 30.0)
        except (ValueError, TypeError):
            return 0.5

    def _cluster_by_similarity(
        self,
        descriptions: List[str],
        threshold: float,
        timestamps: List[str] = None,
        all_defects: List[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """文本相似度聚类（零 LLM），支持时间衰减权重"""
        if not descriptions:
            return []

        timestamps = timestamps or [""] * len(descriptions)
        all_defects = all_defects or []
        clusters = []
        for i, desc in enumerate(descriptions):
            tw = self._time_weight(timestamps[i]) if i < len(timestamps) else 1.0
            matched = False
            for cluster in clusters:
                if self._text_similarity(desc, cluster["representative"]) >= threshold:
                    cluster["members"].append(i)
                    cluster["weighted_count"] += tw
                    cluster["raw_count"] += 1
                    matched = True
                    break
            if not matched:
                clusters.append({
                    "representative": desc,
                    "members": [i],
                    "weighted_count": tw,
                    "raw_count": 1,
                    "severities": [],
                    "dimensions": [],
                })

        # 聚合 severities 和 dimensions（需要原始 defects 数据）
        for cluster in clusters:
            for idx in cluster["members"]:
                if idx < len(all_defects):
                    cluster["severities"].append(all_defects[idx].get("severity", "一般"))
                    cluster["dimensions"].append(all_defects[idx].get("dimension", ""))

        return clusters

    def _text_similarity(self, a: str, b: str) -> float:
        """中文 bigram Jaccard 相似度（对同义描述聚类效果优于字符级）"""
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        # Bigram: 相邻字符对，"客服态度不好" → ["客服","服态","态度","度不","不好"]
        bigrams_a = {a[i:i+2] for i in range(len(a) - 1)}
        bigrams_b = {b[i:i+2] for i in range(len(b) - 1)}
        if not bigrams_a or not bigrams_b:
            # 退化为字符级 Jaccard（单字情况）
            return len(set(a) & set(b)) / len(set(a) | set(b))
        return len(bigrams_a & bigrams_b) / len(bigrams_a | bigrams_b)

    def _filter_existing(self, candidates: List[Dict]) -> List[Dict]:
        """过滤与已有清单项高相似的候选"""
        existing = self._load_evolved_items()
        existing_descs = [item.get("description", "") for item in existing]

        from src.eval.config import EVOLUTION as _EVOL
        dup_threshold = _EVOL.get("duplicate_similarity_threshold", 0.85)

        filtered = []
        for c in candidates:
            is_new = True
            for ed in existing_descs:
                if self._text_similarity(c["description"], ed) >= dup_threshold:
                    is_new = False
                    break
            if is_new:
                filtered.append(c)
        return filtered

    # ---- Phase 3.2: 自动转化 ----

    def convert_to_checklist_items(
        self,
        min_frequency: int = 5,
    ) -> List[Dict[str, Any]]:
        """高频缺陷自动转为清单项（Phase 3.2）

        Returns:
            新增的清单项列表
        """
        stats = self.analyze_defects(min_frequency=min_frequency)
        candidates = stats.get("conversion_candidates", [])

        new_items = []
        for c in candidates:
            item = {
                "item_id": f"evolved_{_slugify(c['description'][:30])}",
                "description": c["description"],
                "source": "pattern_mined",
                "weight": 1.3,
                "frequency": c["count"],
                "primary_dimension": c["dimensions"].most_common(1)[0][0] if c["dimensions"] else "",
                "created_at": datetime.now().isoformat(),
            }
            new_items.append(item)

        if new_items:
            existing = self._load_evolved_items()
            existing.extend(new_items)
            with open(self.checklist_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, ensure_ascii=False, indent=2)

        return new_items

    def _load_evolved_items(self) -> List[Dict[str, Any]]:
        """加载已转化的清单项"""
        if not self.checklist_file.exists():
            return []
        with open(self.checklist_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- 清单裁剪 ----

    def compute_pass_rates(
        self,
        results: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """计算各清单项的通过率（用于检测过松/过严项）"""
        item_counts: Dict[str, int] = {}
        item_yes_counts: Dict[str, int] = {}

        for r in results:
            for dim, checklist in r.get("dimension_checklists", {}).items():
                items = checklist.get("items", [])
                for item in items:
                    item_id = item.get("item_id", "")
                    if not item_id:
                        continue
                    item_counts[item_id] = item_counts.get(item_id, 0) + 1
                    if item.get("status") == "YES":
                        item_yes_counts[item_id] = item_yes_counts.get(item_id, 0) + 1

        pass_rates = {}
        for item_id, total in item_counts.items():
            pass_rates[item_id] = item_yes_counts.get(item_id, 0) / total if total > 0 else 1.0

        return pass_rates

    def prune_recommendations(
        self,
        pass_rates: Dict[str, float],
        too_easy_threshold: float = 0.95,
        too_hard_threshold: float = 0.05,
    ) -> Dict[str, List[str]]:
        """给出清单裁剪建议"""
        too_easy = [k for k, v in pass_rates.items() if v >= too_easy_threshold]
        too_hard = [k for k, v in pass_rates.items() if v <= too_hard_threshold]
        return {"too_easy": too_easy, "too_hard": too_hard}


def _slugify(text: str) -> str:
    """简单中文兼容的 slug 化"""
    import re
    text = re.sub(r'[^\w一-鿿]', '_', text)
    return text.strip('_').lower() or "unknown"


# ---- Phase 3.2 扩展: 清单删除 ----

def prune_low_discrimination_items(
    pass_rates: Dict[str, float],
    not_applicable_rates: Dict[str, float] = None,
    yes_threshold: float = 0.95,
    na_threshold: float = 0.80,
) -> Dict[str, List[Dict[str, Any]]]:
    """识别应删除的低区分力清单项

    Returns: {"to_remove": [{"item_id": str, "reason": str, "rate": float}, ...]}
    """
    not_applicable_rates = not_applicable_rates or {}
    to_remove = []

    for item_id, rate in pass_rates.items():
        if rate >= yes_threshold:
            to_remove.append({
                "item_id": item_id,
                "reason": f"YES+MOSTLY_YES 率 {rate:.1%} ≥ {yes_threshold:.0%}，无区分力",
                "rate": rate,
            })

    for item_id, rate in not_applicable_rates.items():
        if rate >= na_threshold:
            # 避免重复
            if not any(r["item_id"] == item_id for r in to_remove):
                to_remove.append({
                    "item_id": item_id,
                    "reason": f"NOT_APPLICABLE 率 {rate:.1%} ≥ {na_threshold:.0%}，不适用于大多数对话",
                    "rate": rate,
                })

    return {"to_remove": to_remove}


# ---- Phase 3.2 扩展: 清单修改 ----

def suggest_modifications(
    partial_rates: Dict[str, float],
    descriptions: Dict[str, str] = None,
    partial_threshold: float = 0.30,
) -> Dict[str, List[Dict[str, Any]]]:
    """识别需要修改措辞的清单项（PARTIAL 率高说明描述不清晰）

    Returns: {"to_modify": [{"item_id": str, "current_description": str, "partial_rate": float}, ...]}
    """
    descriptions = descriptions or {}
    to_modify = []

    for item_id, rate in partial_rates.items():
        if rate >= partial_threshold:
            to_modify.append({
                "item_id": item_id,
                "current_description": descriptions.get(item_id, ""),
                "partial_rate": rate,
                "suggestion": f"PARTIAL 率 {rate:.1%} ≥ {partial_threshold:.0%}——描述可能不清晰，建议重新措辞或拆分",
            })

    return {"to_modify": to_modify}


# ---- Phase 3.2 扩展: 权重校准 ----

def calibrate_source_weights(
    results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """按维度计算 Case/Simulator YES 占比与总体分的相关性，输出校准后权重

    Returns: {dimension: {"case": float, "simulator": float}}
    """
    from collections import defaultdict

    dim_data = defaultdict(lambda: {"case_pairs": [], "sim_pairs": []})  # [(ratio, score), ...]

    for r in results:
        score = r.get("total_indicative_score", 5.0)
        for dim, checklist in r.get("dimension_checklists", {}).items():
            items = checklist.get("items", [])
            case_items = [i for i in items if i.get("source") == "case" and i.get("status") != "NOT_APPLICABLE"]
            sim_items = [i for i in items if i.get("source") == "simulator" and i.get("status") != "NOT_APPLICABLE"]

            if case_items:
                cr = sum(1 for i in case_items if i.get("status") in ("YES", "MOSTLY_YES")) / len(case_items)
                dim_data[dim]["case_pairs"].append((cr, score))
            if sim_items:
                sr = sum(1 for i in sim_items if i.get("status") in ("YES", "MOSTLY_YES")) / len(sim_items)
                dim_data[dim]["sim_pairs"].append((sr, score))

    calibrated = {}
    for dim, data in dim_data.items():
        case_pairs = data["case_pairs"]
        sim_pairs = data["sim_pairs"]
        if len(case_pairs) < 5 and len(sim_pairs) < 5:
            continue

        case_ratios = [p[0] for p in case_pairs]
        case_scores = [p[1] for p in case_pairs]
        sim_ratios = [p[0] for p in sim_pairs]
        sim_scores = [p[1] for p in sim_pairs]

        case_corr = _pearson_corr(case_ratios, case_scores) if len(case_pairs) >= 5 else None
        sim_corr = _pearson_corr(sim_ratios, sim_scores) if len(sim_pairs) >= 5 else None

        # 相关性高者权重上调
        case_w = 0.6 + max(0, case_corr) * 0.3 if case_corr else 0.6
        sim_w = 1.5 + max(0, sim_corr) * 0.3 if sim_corr else 1.5

        calibrated[dim] = {
            "case_weight": round(case_w, 2),
            "simulator_weight": round(sim_w, 2),
            "case_correlation": round(case_corr, 3) if case_corr else 0,
            "simulator_correlation": round(sim_corr, 3) if sim_corr else 0,
        }

    return calibrated


def _pearson_corr(x: List[float], y: List[float]) -> Optional[float]:
    """Pearson 相关系数"""
    n = len(x)
    if n < 3 or n != len(y):
        return None
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = (sum((xi - mean_x) ** 2 for xi in x) ** 0.5)
    std_y = (sum((yi - mean_y) ** 2 for yi in y) ** 0.5)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


# ---- 进化周期编排 ----

def compute_detailed_pass_rates(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算详细的清单项通过率（含 YES, PARTIAL, NOT_APPLICABLE 分类）"""
    item_counts: Dict[str, int] = {}
    item_positive: Dict[str, int] = {}
    item_partial: Dict[str, int] = {}
    item_na: Dict[str, int] = {}
    item_descriptions: Dict[str, str] = {}

    for r in results:
        for dim, checklist in r.get("dimension_checklists", {}).items():
            for item in checklist.get("items", []):
                iid = item.get("item_id", "")
                if not iid:
                    continue
                item_counts[iid] = item_counts.get(iid, 0) + 1
                item_descriptions[iid] = item.get("description", "")
                status = item.get("status", "")
                if status in ("YES", "MOSTLY_YES"):
                    item_positive[iid] = item_positive.get(iid, 0) + 1
                if status == "PARTIAL":
                    item_partial[iid] = item_partial.get(iid, 0) + 1
                if status == "NOT_APPLICABLE":
                    item_na[iid] = item_na.get(iid, 0) + 1

    positive_rates = {iid: item_positive.get(iid, 0) / item_counts[iid] for iid in item_counts}
    partial_rates = {iid: item_partial.get(iid, 0) / item_counts[iid] for iid in item_counts}
    na_rates = {iid: item_na.get(iid, 0) / item_counts[iid] for iid in item_counts}

    return {
        "positive_rates": positive_rates,
        "partial_rates": partial_rates,
        "not_applicable_rates": na_rates,
        "descriptions": item_descriptions,
        "item_counts": item_counts,
    }


def run_evolution_cycle(
    results: List[Dict[str, Any]],
    min_samples: int = 50,
    min_modify_samples: int = 20,
) -> Dict[str, Any]:
    """编排完整的清单进化周期：分析通过率 → 删除建议 → 修改建议 → 权重校准

    Returns: {
        "delete_suggestions": [...],
        "modify_suggestions": [...],
        "weight_calibration": {...},
        "summary": str,
    }
    """
    n = len(results)
    report = {"n_results": n, "actions_taken": []}

    # 1. 分析通过率
    rates = compute_detailed_pass_rates(results)
    report["pass_rate_analysis"] = rates

    # 2. 删除建议（需足够样本量）
    if n >= min_samples:
        from src.eval.config import EVOLUTION
        del_result = prune_low_discrimination_items(
            rates["positive_rates"],
            rates["not_applicable_rates"],
            yes_threshold=EVOLUTION.get("prune_yes_threshold", 0.95),
            na_threshold=EVOLUTION.get("prune_not_applicable_threshold", 0.80),
        )
        report["delete_suggestions"] = del_result["to_remove"]
        if del_result["to_remove"]:
            report["actions_taken"].append(f"建议删除 {len(del_result['to_remove'])} 条低区分力清单项")
    else:
        report["delete_suggestions"] = []
        report["actions_taken"].append(f"样本量不足({n}<{min_samples})，跳过删除分析")

    # 3. 修改建议
    if n >= min_modify_samples:
        from src.eval.config import EVOLUTION
        mod_result = suggest_modifications(
            rates["partial_rates"],
            rates["descriptions"],
            partial_threshold=EVOLUTION.get("modify_partial_threshold", 0.30),
        )
        report["modify_suggestions"] = mod_result["to_modify"]
        if mod_result["to_modify"]:
            report["actions_taken"].append(f"建议修改 {len(mod_result['to_modify'])} 条描述不清晰的清单项")
    else:
        report["modify_suggestions"] = []
        report["actions_taken"].append(f"样本量不足({n}<{min_modify_samples})，跳过修改分析")

    # 4. 权重校准
    if n >= 10:
        report["weight_calibration"] = calibrate_source_weights(results)
        report["actions_taken"].append("完成源权重校准分析")
    else:
        report["weight_calibration"] = {}
        report["actions_taken"].append("样本量不足(<10)，跳过权重校准")

    report["summary"] = "; ".join(report["actions_taken"])
    return report
