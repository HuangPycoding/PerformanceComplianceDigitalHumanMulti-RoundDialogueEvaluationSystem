"""SelfReliabilityChecker — 无人工标注的评测系统自验证（纯规则，零 LLM 成本）"""
from typing import Any, Dict, List, Optional, Tuple


class SelfReliabilityChecker:
    """评测系统自身可靠性验证器"""

    @staticmethod
    def check_test_retest(
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """重测信度：同一对话跑两次，计算一致率"""
        if len(results_a) != len(results_b) or len(results_a) == 0:
            return {"is_reliable": False, "overall_reliability": 0.0, "error": "结果列表长度不一致或为空"}

        total_items = 0
        agree_items = 0
        rating_agreements = 0
        total_dims = 0

        for ra, rb in zip(results_a, results_b):
            checklists_a = ra.get("dimension_checklists", {})
            checklists_b = rb.get("dimension_checklists", {})

            for dim in checklists_a:
                items_a = {i["item_id"]: i.get("status", "")
                           for i in checklists_a.get(dim, {}).get("items", [])}
                items_b = {i["item_id"]: i.get("status", "")
                           for i in checklists_b.get(dim, {}).get("items", [])}

                for iid in items_a:
                    if iid in items_b:
                        total_items += 1
                        if items_a[iid] == items_b[iid]:
                            agree_items += 1

            ratings_a = ra.get("ratings", {})
            ratings_b = rb.get("ratings", {})
            for dim in ratings_a:
                if dim in ratings_b:
                    total_dims += 1
                    if ratings_a[dim] == ratings_b[dim]:
                        rating_agreements += 1

        status_agreement = agree_items / max(total_items, 1)
        rating_agreement = rating_agreements / max(total_dims, 1)
        overall = (status_agreement * 0.6 + rating_agreement * 0.4)

        return {
            "status_agreement_rate": status_agreement,
            "rating_agreement_rate": rating_agreement,
            "overall_reliability": overall,
            "total_check_items": total_items,
            "total_dimensions": total_dims,
            "is_reliable": overall >= 0.85,
        }

    @staticmethod
    def validate_evidence_citations(
        result: Dict[str, Any],
        max_turn: int,
    ) -> Dict[str, Any]:
        """验证 evidence 中轮次引用是否在有效范围内"""
        import re

        invalid_citations = []
        valid_count = 0
        total_count = 0

        checklists = result.get("dimension_checklists", {})
        for dim, cl in checklists.items():
            for item in cl.get("items", []):
                evidence = item.get("evidence", "")
                if not evidence:
                    continue
                total_count += 1

                turn_nums = re.findall(r'T(\d+)', evidence)
                all_valid = True
                for tn_str in turn_nums:
                    tn = int(tn_str)
                    if tn > max_turn or tn < 1:
                        invalid_citations.append({
                            "dimension": dim,
                            "item_id": item.get("item_id", ""),
                            "evidence": evidence[:80],
                            "reason": f"T{tn} 不在对话范围 (1-{max_turn})",
                        })
                        all_valid = False
                        break

                if all_valid:
                    valid_count += 1

        return {
            "total_citations": total_count,
            "valid_citations": valid_count,
            "invalid_citations": invalid_citations,
            "validity_ratio": valid_count / max(total_count, 1),
        }

    @staticmethod
    def check_score_distribution(
        results_list: List[Dict[str, Any]],
        min_conversations: int = 20,
    ) -> Dict[str, Any]:
        """检查评分分布是否异常（区分力检测）"""
        from collections import Counter

        n = len(results_list)
        if n < min_conversations:
            return {"warning": f"样本量不足 ({n} < {min_conversations})，跳过分布检查", "alerts": []}

        alerts = []
        dim_ratings: Dict[str, List[str]] = {}

        for r in results_list:
            for dim, rating in r.get("ratings", {}).items():
                dim_ratings.setdefault(dim, []).append(rating)

        for dim, ratings_list in dim_ratings.items():
            counter = Counter(ratings_list)
            most_common_ratio = counter.most_common(1)[0][1] / len(ratings_list)
            if most_common_ratio > 0.95:
                alerts.append({
                    "dimension": dim,
                    "issue": "no_discrimination",
                    "description": f"95%+ 对话评级为 '{counter.most_common(1)[0][0]}'——该维度无区分力",
                    "distribution": dict(counter),
                })

        return {"alerts": alerts, "n_conversations": n}

    @staticmethod
    def check_inter_judge_agreement(
        results_list: List[Dict[str, Any]],
        rho_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        """检查 Judge 间评分一致性（冗余检测）"""
        n = len(results_list)
        if n < 10:
            return {"warning": f"样本量不足 ({n} < 10)，跳过一致性检查", "redundant_pairs": []}

        rating_order = {"卓越": 4, "良好": 3, "合格": 2, "需改进": 1, "不合格": 0}
        dim_scores: Dict[str, List[int]] = {}

        for r in results_list:
            for dim, rating in r.get("ratings", {}).items():
                dim_scores.setdefault(dim, []).append(rating_order.get(rating, 2))

        redundant_pairs = []
        dims = list(dim_scores.keys())
        for i in range(len(dims)):
            for j in range(i + 1, len(dims)):
                if len(dim_scores[dims[i]]) != len(dim_scores[dims[j]]):
                    continue
                rho = SelfReliabilityChecker._spearman_rho(
                    dim_scores[dims[i]], dim_scores[dims[j]]
                )
                if rho is not None and rho > rho_threshold:
                    redundant_pairs.append({
                        "dim_a": dims[i],
                        "dim_b": dims[j],
                        "spearman_rho": rho,
                    })

        return {"redundant_pairs": redundant_pairs, "n_conversations": n}

    @staticmethod
    def _spearman_rho(a: List[int], b: List[int]) -> Optional[float]:
        """计算 Spearman 秩相关系数"""
        n = len(a)
        if n < 3:
            return None

        def rank(data):
            sorted_data = sorted((v, i) for i, v in enumerate(data))
            ranks = [0] * n
            i = 0
            while i < n:
                j = i
                while j < n and sorted_data[j][0] == sorted_data[i][0]:
                    j += 1
                avg_rank = (i + j - 1) / 2 + 1
                for k in range(i, j):
                    ranks[sorted_data[k][1]] = avg_rank
                i = j
            return ranks

        rank_a = rank(a)
        rank_b = rank(b)
        d2 = sum((ra - rb) ** 2 for ra, rb in zip(rank_a, rank_b))
        return 1 - (6 * d2) / (n * (n * n - 1))

    def run_full_check(
        self,
        results_list: List[Dict[str, Any]],
        results_retest: Optional[List[Dict[str, Any]]] = None,
        conv_texts: Optional[List[str]] = None,
        max_turns: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """运行完整的自验证检查"""
        report = {}

        # 1. 评分分布检查
        report["score_distribution"] = self.check_score_distribution(results_list)

        # 2. Judge 间一致性
        report["inter_judge_agreement"] = self.check_inter_judge_agreement(results_list)

        # 3. 重测信度（可选）
        if results_retest:
            report["test_retest"] = self.check_test_retest(results_list, results_retest)

        # 4. 证据引用验证（单条抽样）
        if conv_texts and max_turns and results_list:
            ev_results = []
            for i, (r, mt) in enumerate(zip(results_list[:5], max_turns[:5])):
                ev_results.append(self.validate_evidence_citations(r, mt))
            avg_validity = sum(e["validity_ratio"] for e in ev_results) / max(len(ev_results), 1)
            report["evidence_validity"] = {
                "sample_size": len(ev_results),
                "avg_validity_ratio": avg_validity,
                "details": ev_results,
            }

        # 5. 综合可靠性分数
        scores = []
        dist = report.get("score_distribution", {})
        if dist.get("alerts"):
            scores.append(max(0.0, 1.0 - len(dist["alerts"]) * 0.1))
        else:
            scores.append(1.0)

        ija = report.get("inter_judge_agreement", {})
        if ija.get("redundant_pairs"):
            scores.append(max(0.0, 1.0 - len(ija["redundant_pairs"]) * 0.05))
        else:
            scores.append(1.0)

        tr = report.get("test_retest", {})
        if tr:
            scores.append(tr.get("overall_reliability", 1.0))

        ev = report.get("evidence_validity", {})
        if ev:
            scores.append(ev.get("avg_validity_ratio", 1.0))

        report["overall_health"] = sum(scores) / max(len(scores), 1)
        report["is_healthy"] = report["overall_health"] >= 0.80

        return report
