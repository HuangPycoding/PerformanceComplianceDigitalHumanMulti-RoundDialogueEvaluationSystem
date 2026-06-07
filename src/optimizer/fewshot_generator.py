"""Few-shot 示例生成器 — MMR 多样性选择 + LLM 生成正负例

触发条件: 维度"合格"以下占比 > 30% 或缺陷聚类 frequency ≥ 3
输出: 正例 + 负例 + 修正例 (ChatML 三元组格式)
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.llm.client import LLMClient
from src.optimizer.optimizer import DefectCluster, OptimizationAction
from src.optimizer.prompts import DIMENSION_CN, FEWSHOT_GEN_SYSTEM, FEWSHOT_GEN_USER
from src.optimizer.utils import bigram_jaccard


class FewshotGenerator:
    """Few-shot 示例生成器（路径 B，LLM 驱动 + MMR 选择）"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate_examples(
        self,
        clusters: List[DefectCluster],
        conv_dir: Path,
        ratings_data: Dict[str, List[str]],
    ) -> List[OptimizationAction]:
        """为弱维度生成 few-shot 训练示例。

        Args:
            clusters: 缺陷聚类列表
            conv_dir: conversation_*.json 所在目录
            ratings_data: {conversation_id: {dimension: rating}}
        """
        actions = []

        for cluster in clusters:
            if not self._should_generate(cluster, ratings_data):
                continue

            # MMR 多样性选择对话片段
            excerpts = self._mmr_select_excerpts(cluster, conv_dir)

            if not excerpts:
                continue

            # LLM 生成示例
            try:
                response = self.llm.chat(
                    FEWSHOT_GEN_SYSTEM,
                    FEWSHOT_GEN_USER.format(
                        dimension_cn=DIMENSION_CN.get(cluster.dimension, cluster.dimension),
                        dimension=cluster.dimension,
                        defect_pattern=cluster.defect_pattern,
                        conversation_excerpts=excerpts,
                        evaluation_findings="\n".join(
                            f"- {e}" for e in cluster.evidence_samples[:3]
                        ),
                    ),
                )

                actions.append(OptimizationAction(
                    action_id=f"fewshot-{cluster.cluster_id}",
                    source="model",
                    dimension=cluster.dimension,
                    priority=cluster.frequency * 0.6,
                    title=f"Few-shot 示例: {cluster.defect_pattern[:40]}",
                    constitutional_principle=f"评测维度——{cluster.dimension}",
                    violation_evidence=cluster.evidence_samples[:2],
                    critique_analysis=f"该维度在 {len(cluster.conversation_ids)} 场对话中出现系统性低分",
                    revision_proposal=response,
                    target_location="Assistant System Prompt（作为 few-shot 示例嵌入）",
                    expected_impact=f"预期改善 {cluster.dimension} 维度评分",
                    effort_estimate="低",
                    is_actionable=True,
                    level="中建议",
                    path="B",
                ))
            except Exception:
                continue

        return actions

    def _should_generate(
        self, cluster: DefectCluster, ratings_data: Dict[str, List[str]],
    ) -> bool:
        """判断是否需要生成 few-shot 示例。"""
        if cluster.frequency >= 3:
            return True

        # 统计该维度"需改进"+"不合格"的占比
        dim = cluster.dimension
        total = 0
        fail = 0
        for ratings in ratings_data.values():
            if dim in ratings:
                total += 1
                if ratings[dim] in ("需改进", "不合格"):
                    fail += 1

        return total > 0 and fail / total > 0.30

    def _mmr_select_excerpts(
        self, cluster: DefectCluster, conv_dir: Path,
        k: int = 3, lambda_param: float = 0.7,
    ) -> str:
        """MMR 多样性选择——从失败对话中选取高信息量片段。"""
        excerpts: Dict[str, str] = {}

        for g in cluster.gradients[:5]:
            conv_id = g.conversation_id
            conv_file = self._find_conv_file(conv_dir, conv_id)
            if not conv_file:
                continue

            try:
                conv_data = json.loads(conv_file.read_text(encoding="utf-8"))
                turns = conv_data.get("turns", [])

                # 提取证据轮次前后
                turn_nums = set()
                for evidence in g.evidence[:2]:
                    for tn in re.findall(r'T(\d+)', evidence):
                        t = int(tn)
                        for dt in range(max(1, t - 3), min(len(turns), t + 4)):
                            turn_nums.add(dt)

                if not turn_nums:
                    # 无证据轮次 → 取对话中段
                    mid = len(turns) // 2
                    turn_nums = set(range(max(1, mid - 3), min(len(turns), mid + 4)))

                lines = []
                for turn in turns:
                    if turn.get("turn") in turn_nums:
                        speaker = "客服" if turn.get("speaker") == "system" else "用户"
                        lines.append(
                            f"T{turn['turn']} [{speaker}]: {turn.get('content', '')[:200]}"
                        )

                if lines:
                    excerpts[conv_id] = "\n".join(lines)
            except Exception:
                continue

        if len(excerpts) <= k:
            return "\n---\n".join(excerpts.values())

        # MMR 贪心选择
        keys = list(excerpts.keys())
        scores = {
            cid: bigram_jaccard(text, cluster.defect_pattern)
            for cid, text in excerpts.items()
        }

        selected = []
        remaining = list(keys)
        first = max(remaining, key=lambda cid: scores[cid])
        selected.append(first)
        remaining.remove(first)

        for _ in range(k - 1):
            if not remaining:
                break
            best = max(remaining, key=lambda cid:
                lambda_param * scores[cid] -
                (1 - lambda_param) * max(
                    bigram_jaccard(excerpts[cid], excerpts[s])
                    for s in selected
                ) if selected else lambda_param * scores[cid]
            )
            selected.append(best)
            remaining.remove(best)

        return "\n---\n".join(excerpts[cid] for cid in selected)

    def _find_conv_file(self, conv_dir: Path, conv_id: str) -> Path | None:
        """查找对话 JSON 文件。"""
        # 精确匹配
        conv_file = conv_dir / f"conversation_{conv_id}.json"
        if conv_file.exists():
            return conv_file
        # 模糊匹配
        try:
            candidates = [f for f in conv_dir.iterdir()
                          if f.suffix == '.json' and conv_id in f.name]
            return candidates[0] if candidates else None
        except Exception:
            return None
