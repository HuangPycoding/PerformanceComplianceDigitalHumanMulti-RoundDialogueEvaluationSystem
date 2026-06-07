"""BatchAnalyzer — 批次聚合分析 + 自验证集成

原 DriftMonitor 概念已合并到 BatchAnalyzer 中。
漂移检测功能由以下已有机制提供：
- SelfReliabilityChecker.check_score_distribution() — 区分力漂移
- SelfReliabilityChecker.check_inter_judge_agreement() — 维度冗余
- BatchAnalyzer.analyze() 的异常维度告警
- EvalConfidence 中逐条消费 Simulator tier 分布
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.eval.config import BATCH_ANALYSIS_CONFIG
from src.eval.self_reliability import SelfReliabilityChecker


class DriftMonitor:
    """[已弃用] 评测引擎内部漂移监控器。

    原 DriftMonitor 功能已被 BatchAnalyzer + SelfReliabilityChecker 替代。
    请使用 BatchAnalyzer.analyze() 进行批次聚合分析。
    """

    def __init__(self):
        warnings.warn(
            "DriftMonitor 已弃用，请使用 BatchAnalyzer",
            DeprecationWarning, stacklevel=2,
        )
        self.baseline_ratings: Dict[str, Dict[str, float]] = {}
        self.baseline_checklist_pass: Dict[str, float] = {}
        self.baseline_signal_tiers: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []

    def set_baseline(self, results: List[Dict[str, Any]]) -> None:
        """[已弃用] 从一批结果建立 baseline"""
        warnings.warn("DriftMonitor.set_baseline 已弃用", DeprecationWarning)
        pass

    def check_drift(self, new_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """[已弃用] 检查新一批结果是否有漂移

        Returns:
            {"has_drift": bool, "alerts": [...], "details": {...}}
        """
        warnings.warn("DriftMonitor.check_drift 已弃用", DeprecationWarning)
        return {"has_drift": False, "alerts": [], "details": {}}

    def add_to_history(self, batch_results: List[Dict[str, Any]]) -> None:
        """[已弃用] 记录批次结果到历史"""
        warnings.warn("DriftMonitor.add_to_history 已弃用", DeprecationWarning)
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "n_results": len(batch_results),
            "results": batch_results,
        })


class BatchAnalyzer:
    """批次聚合分析层 — v1 纯统计零 LLM + Phase 4 自验证集成"""

    def __init__(self):
        self.reliability_checker = SelfReliabilityChecker()

    def analyze(
        self,
        results: List[Dict[str, Any]],
        retest_results: Optional[List[Dict[str, Any]]] = None,
        conv_texts: Optional[List[str]] = None,
        max_turns_list: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """对一批 EvalResult 做聚合分析 + 自验证

        Args:
            results: List of EvalResult dicts
            retest_results: 重测结果（同一批对话重跑后的结果）
            conv_texts: 对话文本列表（用于证据引用验证）
            max_turns_list: 每场对话的轮次数（用于证据引用验证）

        Returns:
            summary dict with:
                - rating_distribution: 各维度五级评级占比
                - anomaly_dimensions: 异常维度告警
                - is_reliable_ratio: is_reliable=False 占比
                - complexity_strata: 按复杂度分层统计
                - alerts: 告警列表
                - self_reliability: 评测系统自身可靠性报告
        """
        n = len(results)

        # 评级分布
        if n == 0:
            return {
                "n_results": 0,
                "rating_distribution": {},
                "anomaly_dimensions": [],
                "is_reliable_ratio": 0.0,
                "unreliable_count": 0,
                "complexity_strata": {"low": {"count": 0}, "medium": {"count": 0}, "high": {"count": 0}},
                "alerts": ["批次无结果——可能数据未生成或流水线失败"],
                "self_reliability": {},
            }
        dim_ratings: Dict[str, Dict[str, int]] = {}
        for r in results:
            for dim, rating in r.get("ratings", {}).items():
                dim_ratings.setdefault(dim, {}).setdefault(rating, 0)
                dim_ratings[dim][rating] += 1

        rating_distribution = {}
        for dim, counts in dim_ratings.items():
            rating_distribution[dim] = {
                label: counts.get(label, 0) / n for label in ["卓越", "良好", "合格", "需改进", "不合格"]
            }

        # 异常维度告警
        anomaly_dims = []
        bac = BATCH_ANALYSIS_CONFIG
        for dim, dist in rating_distribution.items():
            fail_rate = dist.get("不合格", 0)
            if fail_rate > bac["fail_rate_warning"]:
                anomaly_dims.append({
                    "dimension": dim,
                    "reason": f"不合格率 = {fail_rate:.0%}",
                    "severity": "high" if fail_rate > bac["fail_rate_high"] else "medium",
                })

        # is_reliable 占比
        unreliable_count = sum(1 for r in results if not r.get("is_reliable", True))
        unreliable_ratio = unreliable_count / n

        # 按复杂度分层
        complexity_strata = {"low": [], "medium": [], "high": []}
        for r in results:
            cs = r.get("complexity_score", 0)
            if cs <= 3:
                complexity_strata["low"].append(r)
            elif cs <= 7:
                complexity_strata["medium"].append(r)
            else:
                complexity_strata["high"].append(r)

        strata_summary = {}
        for level, items in complexity_strata.items():
            if not items:
                strata_summary[level] = {"count": 0}
                continue
            avg_score = sum(i.get("total_indicative_score", 0) for i in items) / len(items)
            strata_summary[level] = {"count": len(items), "avg_score": round(avg_score, 2)}

        # 告警
        alerts = []
        if unreliable_ratio > 0.5:
            alerts.append(f"批次 is_reliable=False 占比 {unreliable_ratio:.0%}——建议检查 Judge 配置或 Simulator 质量")
        for ad in anomaly_dims:
            alerts.append(f"[{ad['severity']}] {ad['dimension']}: {ad['reason']}")

        # Phase 4: 自验证
        self_reliability = self.reliability_checker.run_full_check(
            results_list=results,
            results_retest=retest_results,
            conv_texts=conv_texts,
            max_turns=max_turns_list,
        )

        # 自验证告警
        if not self_reliability.get("is_healthy", True):
            alerts.append(f"[自验证] 评测系统健康度 {self_reliability['overall_health']:.2f} < 0.80——建议检查")

        dist_check = self_reliability.get("score_distribution", {})
        for alert in dist_check.get("alerts", []):
            alerts.append(f"[自验证] {alert['dimension']}: {alert['description']}")

        ija = self_reliability.get("inter_judge_agreement", {})
        for pair in ija.get("redundant_pairs", []):
            alerts.append(f"[自验证] 维度冗余: {pair['dim_a']} ↔ {pair['dim_b']} (Spearman ρ={pair['spearman_rho']:.3f})")

        tr = self_reliability.get("test_retest", {})
        if tr and not tr.get("is_reliable", True):
            alerts.append(f"[自验证] 重测信度 {tr['overall_reliability']:.2f} < 0.85——评测结果不稳定")

        return {
            "n_results": n,
            "rating_distribution": rating_distribution,
            "anomaly_dimensions": anomaly_dims,
            "is_reliable_ratio": unreliable_ratio,
            "unreliable_count": unreliable_count,
            "complexity_strata": strata_summary,
            "alerts": alerts,
            "self_reliability": self_reliability,
        }
