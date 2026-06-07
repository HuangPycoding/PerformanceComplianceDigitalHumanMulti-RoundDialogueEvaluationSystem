"""优化引擎 v1 — 评测驱动的优化建议生成模块

消费评测引擎输出（optimization_feed.json + case.json + profiles.json + conversation_*.json），
结合 LLM 和规则引擎，为 Case 定义、用户画像生成器、对话模型、评测引擎提供优化建议。

只出建议，不动代码。
"""

from src.optimizer.utils import (
    bigram_jaccard,
    cluster_by_similarity,
    calc_priority,
    classify_severity,
    extract_prompt_section,
    pearson_r,
    compute_discrimination_stats,
)

__all__ = [
    "bigram_jaccard",
    "cluster_by_similarity",
    "calc_priority",
    "classify_severity",
    "extract_prompt_section",
    "pearson_r",
    "compute_discrimination_stats",
]
