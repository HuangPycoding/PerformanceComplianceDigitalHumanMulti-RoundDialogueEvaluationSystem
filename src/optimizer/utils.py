"""优化引擎规则工具 — 聚类、排序、相关性、区分力计算等零 LLM 成本的统计函数"""
import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# 中文文本相似度
# ============================================================

def bigram_jaccard(a: str, b: str) -> float:
    """计算两个中文文本的 bigram Jaccard 相似度 (0-1)。

    用于缺陷聚类和 MMR 多样性选择。容忍 LLM 生成的 evidence 轻微改写。
    对英文混合文本也适用——中文字符和英文单词统一处理。
    """
    if not a or not b:
        return 0.0

    def _clean(s: str) -> str:
        return ''.join(c for c in s if c.isalnum() or '一' <= c <= '鿿')

    ca, cb = _clean(a), _clean(b)
    if len(ca) < 2 or len(cb) < 2:
        return 0.0

    ba = {ca[i:i + 2] for i in range(len(ca) - 1)}
    bb = {cb[i:i + 2] for i in range(len(cb) - 1)}

    union = ba | bb
    if not union:
        return 0.0
    return len(ba & bb) / len(union)


# ============================================================
# 缺陷聚类
# ============================================================

def cluster_by_similarity(
    items: List[Dict[str, Any]],
    threshold: float = 0.5,
    key_field: str = "description",
) -> List[List[Dict[str, Any]]]:
    """基于 bigram 相似度做贪心聚类，将相似归因项归并为聚类。

    Args:
        items: 待聚类的归因项列表（每项至少含 key_field）
        threshold: bigram Jaccard 最低阈值
        key_field: 用于比较的字段名

    Returns:
        聚类列表，每个聚类是归因项的子列表
    """
    if not items:
        return []

    clusters: List[List[Dict]] = []
    used = [False] * len(items)

    for i, item in enumerate(items):
        if used[i]:
            continue
        cluster = [item]
        used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            sim = bigram_jaccard(
                item.get(key_field, ""),
                items[j].get(key_field, ""),
            )
            if sim >= threshold:
                cluster.append(items[j])
                used[j] = True
        clusters.append(cluster)

    return clusters


# ============================================================
# 优先级计算
# ============================================================

# 严重程度权重
SEVERITY_WEIGHTS = {"major": 3.0, "moderate": 2.0, "minor": 1.0}

# 维度权重（与 src/eval/config.py DIMENSION_WEIGHTS 保持一致）
DIMENSION_WEIGHTS = {
    "SAFETY": 2.0,
    "TASK_COMPLETION": 1.8,
    "FLOW_COVERAGE": 1.2,
    "CONSTRAINT": 1.0,
    "KNOWLEDGE": 1.0,
    "EFFICIENCY": 0.9,
    "ROLE": 0.8,
    "SENTIMENT": 0.8,
    "OPENING": 0.5,
}


def calc_priority(
    severity: str,
    dimension: str,
    frequency: int,
    avg_confidence: float,
) -> float:
    """计算缺陷聚类的优化优先级分数。

    公式: severity_weight × dimension_weight × ln(freq + 1) × avg_confidence
    """
    sw = SEVERITY_WEIGHTS.get(severity, 1.0)
    dw = DIMENSION_WEIGHTS.get(dimension, 1.0)
    freq_factor = math.log(frequency + 1)
    return sw * dw * freq_factor * max(avg_confidence, 0.1)


# ============================================================
# 严重程度分类
# ============================================================

_SEVERITY_MAJOR_KW = ["不合格", "安全", "泄露", "严重", "必须立即", "钳制", "SCOPE"]
_SEVERITY_MODERATE_KW = ["需改进", "偏差", "矛盾", "遗漏", "缺失", "错误", "跳过",
                          "偏低", "不足", "模糊", "过严", "过宽"]


def classify_severity(description: str, category: str = "", confidence: float = 0.5) -> str:
    """从归因项的 description + category 推导严重程度 (major/moderate/minor)。"""
    text = f"{category} {description}".lower()
    if any(kw in text for kw in _SEVERITY_MAJOR_KW):
        return "major"
    if any(kw in text for kw in _SEVERITY_MODERATE_KW):
        return "moderate"
    if confidence >= 0.8:
        return "moderate"
    return "minor"


# ============================================================
# Prompt 段提取
# ============================================================

# Assistant Prompt 7 段结构标记
PROMPT_SECTION_MARKERS = [
    ("# 你的角色", "role"),
    ("# 任务目标", "task"),
    ("# 开场白", "opening"),
    ("# 通话流程", "call_flow"),
    ("# 知识点", "knowledge"),
    ("# 约束条件", "constraints"),
    ("请严格按照以上指令", "closing"),
]

# 维度 → 对应 Prompt 段映射
DIM_TO_PROMPT_SECTIONS = {
    "SAFETY": ["call_flow", "constraints"],
    "TASK_COMPLETION": ["task", "call_flow"],
    "FLOW_COVERAGE": ["call_flow"],
    "KNOWLEDGE": ["knowledge"],
    "CONSTRAINT": ["constraints"],
    "EFFICIENCY": ["call_flow", "closing"],
    "SENTIMENT": ["role", "closing"],
    "ROLE": ["role"],
    "OPENING": ["opening"],
}


def extract_prompt_section(prompt_text: str, section_name: str) -> str:
    """从完整 Assistant Prompt 中提取指定段的文本。

    Args:
        prompt_text: 完整的 7 段 prompt 文本
        section_name: 段名 ("role" | "task" | "opening" | "call_flow" |
                      "knowledge" | "constraints" | "closing")

    Returns:
        提取到的段文本，未找到时返回空字符串
    """
    marker_map = {tag: marker for marker, tag in PROMPT_SECTION_MARKERS}
    if section_name not in marker_map:
        return ""

    start_marker = marker_map[section_name]
    lines = prompt_text.split("\n")

    start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith(start_marker):
            start_idx = i
            break

    if start_idx == -1:
        return ""

    # 找到下一个段的起始位置
    all_markers = [m for m, _ in PROMPT_SECTION_MARKERS]
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        for marker in all_markers:
            if lines[i].strip().startswith(marker) and marker != start_marker:
                end_idx = i
                break
        if end_idx != len(lines):
            break

    return "\n".join(lines[start_idx:end_idx]).strip()


def get_prompt_sections_for_dimension(dimension: str) -> List[str]:
    """获取某评测维度对应的 Prompt 段名列表。"""
    return DIM_TO_PROMPT_SECTIONS.get(dimension, [])


# ============================================================
# 统计工具
# ============================================================

def pearson_r(x: List[float], y: List[float]) -> float:
    """计算 Pearson 相关系数。

    用于权重校准——分析 SourceWeights 或 DimWeights 与总分的相关性。
    """
    n = len(x)
    if n < 3 or n != len(y):
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))

    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def compute_discrimination_stats(
    pass_rates: Dict[str, float],
    na_rates: Optional[Dict[str, float]] = None,
    partial_rates: Optional[Dict[str, float]] = None,
) -> Dict[str, List[str]]:
    """计算清单项的区分力统计。

    Args:
        pass_rates: {item_id: YES+MOSTLY_YES 占比}
        na_rates: {item_id: NOT_APPLICABLE 占比}
        partial_rates: {item_id: PARTIAL 占比}

    Returns:
        {
            "low_discrimination": [通过率 > 95% 的项——建议降权或删除],
            "high_discrimination": [通过率 < 30% 且跨 case 一致的项——建议检查定义],
            "high_na": [不适用率 > 80% 的项——建议缩小适用范围],
            "unclear": [PARTIAL 率 > 30% 的项——建议改写措辞],
        }
    """
    result = {
        "low_discrimination": [],
        "high_discrimination": [],
        "high_na": [],
        "unclear": [],
    }

    for item_id, rate in pass_rates.items():
        if rate >= 0.95:
            result["low_discrimination"].append(item_id)
        elif rate < 0.30:
            result["high_discrimination"].append(item_id)

    if na_rates:
        for item_id, rate in na_rates.items():
            if rate >= 0.80:
                result["high_na"].append(item_id)

    if partial_rates:
        for item_id, rate in partial_rates.items():
            if rate >= 0.30:
                result["unclear"].append(item_id)

    return result


def frequency_distribution(items: List[Dict], field: str) -> Dict[str, int]:
    """统计列表中某字段的频次分布，按频次降序排列。"""
    counts = Counter(item.get(field, "unknown") for item in items)
    return dict(counts.most_common())


def threshold_check(value: float, threshold: float, op: str = "gt") -> bool:
    """通用阈值检查。op: "gt"(大于) | "lt"(小于) | "gte" | "lte"。"""
    ops = {"gt": lambda v, t: v > t, "lt": lambda v, t: v < t,
           "gte": lambda v, t: v >= t, "lte": lambda v, t: v <= t}
    return ops.get(op, ops["gt"])(value, threshold)
