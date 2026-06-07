"""解析模拟器标签输出格式"""
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_simulator_output(text: str) -> Tuple[Dict[str, Any], str]:
    """解析模拟器的标签输出

    Returns:
        (tags_dict, clean_text) where:
        - tags_dict: memory/thought 为 str, state/emotion_curve/risk_flag/model_behavior/conversation_quality 为 dict
        - clean_text: '--' 分隔符后的纯文本回复
    """
    if not text:
        return {}, ""

    tags: Dict[str, Any] = {
        "memory": None,
        "thought": None,
        "state": None,
        "emotion_curve": None,
        "risk_flag": None,
        "model_behavior": None,
        "conversation_quality": None,
        "should_end": None,
    }

    # 按顺序提取每个标签块
    tag_defs = [
        ("memory", r"<memory>\s*(.*?)\s*</memory>"),
        ("thought", r"<thought>\s*(.*?)\s*</thought>"),
        ("state", r"<state>\s*(.*?)\s*</state>"),
        ("emotion_curve", r"<emotion_curve>\s*(.*?)\s*</emotion_curve>"),
        ("risk_flag", r"<risk_flag>\s*(.*?)\s*</risk_flag>"),
        ("model_behavior", r"<model_behavior>\s*(.*?)\s*</model_behavior>"),
        ("conversation_quality", r"<conversation_quality>\s*(.*?)\s*</conversation_quality>"),
        ("should_end", r"<should_end>\s*(.*?)\s*</should_end>"),
    ]

    remaining = text
    for tag_name, pattern in tag_defs:
        match = re.search(pattern, remaining, re.DOTALL | re.IGNORECASE)
        if match:
            raw = match.group(1).strip()
            tags[tag_name] = _parse_tag_content(tag_name, raw)
            # 用切片移除已匹配标签（避免 replace 误删重复内容）
            remaining = remaining[:match.start()] + remaining[match.end():]

    # 提取纯文本回复: '--' 分隔符之后
    clean_text = ""
    # 仅在剩余内容不含 XML 标签结构时信任 -- 分隔符
    if "--" in remaining and not re.search(r"</?[a-zA-Z_一-鿿][a-zA-Z0-9_一-鿿]*\s*/?>", remaining):
        parts = remaining.rsplit("--", 1)
        clean_text = parts[1].strip()
    else:
        clean_text = _strip_all_tags(remaining).strip()

    # P2: 为缺失的结构化标签填充安全默认值，避免输出 JSON 中出现 null
    _fill_defaults(tags)
    # P4: 清理 LLM 偶尔输出的 bracket 包裹值 (如 "[是]" "[否]")
    _strip_brackets(tags)

    return tags, clean_text


def _parse_tag_content(tag_name: str, raw: str) -> Any:
    """解析标签内容: memory/thought 返回 str, 结构化标签返回 dict"""
    if tag_name in ("state", "emotion_curve", "risk_flag", "model_behavior", "conversation_quality", "should_end"):
        result = _parse_key_value_block(raw)
        # P3: 移除 LLM 误输出的标签名作为 key（如 state 内容里的 "state": ""）
        _clean_tag_name_keys(result, tag_name)
        return result
    return raw


def _clean_tag_name_keys(result: dict, tag_name: str) -> None:
    """移除 LLM 偶尔在标签内容中重复输出的标签名 key"""
    keys_to_remove = {
        "state", "emotion_curve", "risk_flag", "model_behavior",
        "conversation_quality", "should_end",
    }
    for k in list(result.keys()):
        if k in keys_to_remove and (not result[k] or result[k].strip() == ""):
            del result[k]


def _parse_key_value_block(raw: str) -> dict:
    """解析 key: value 格式的文本块为 dict"""
    result: dict = {}
    lines = raw.strip().split("\n")
    current_key: Optional[str] = None
    current_value: List[str] = []

    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        # 检测新 key: value 或 key：value
        match = re.match(
            r"^([a-zA-Z_一-鿿][a-zA-Z0-9_一-鿿]*)\s*[:：]\s*(.*)", line_s
        )
        if match:
            if current_key:
                result[current_key] = "\n".join(current_value).strip()
            current_key = match.group(1)
            current_value = [match.group(2)] if match.group(2) else []
        elif current_key:
            current_value.append(line_s)

    if current_key:
        result[current_key] = "\n".join(current_value).strip()

    # 去除值两侧的引号（LLM 常输出 emotion: "满意" 带引号）
    for k, v in result.items():
        if isinstance(v, str) and len(v) >= 2:
            if (v.startswith('"') and v.endswith('"')) or \
               (v.startswith("'") and v.endswith("'")):
                result[k] = v[1:-1]

    return result


def _fill_defaults(tags: Dict[str, Any]) -> None:
    """为缺失的结构化标签填充安全默认值"""
    defaults = {
        "conversation_quality": {"本轮是否自然": "是", "是否卡死": "否"},
        "state": {"emotion": "未知", "emotion_intensity": "0.5"},
        "emotion_curve": {"轨迹": "未知", "趋势": "未知"},
        "should_end": {"本轮是否想结束对话": "否", "原因": "未输出标签"},
    }
    for key, default in defaults.items():
        if not tags.get(key) or not isinstance(tags.get(key), dict):
            tags[key] = default


def _strip_brackets(tags: Dict[str, Any]) -> None:
    """清理 dict 标签值中 LLM 偶尔输出的 bracket 包裹 (如 '[是]' → '是')"""
    for key, val in tags.items():
        if isinstance(val, dict):
            for k, v in val.items():
                if isinstance(v, str) and len(v) >= 3 and v[0] == '[' and v[-1] == ']':
                    val[k] = v[1:-1]


def _strip_all_tags(text: str) -> str:
    """移除所有 <tag>...</tag> 块"""
    return re.sub(r"</?[a-z_]+\s*>", "", text, flags=re.IGNORECASE)


def get_conversation_quality_issues(tags: Dict[str, Any]) -> Optional[str]:
    """检测对话质量标签是否标记异常

    Returns: 异常描述字符串，正常则返回 None
    """
    cq = tags.get("conversation_quality")
    if not isinstance(cq, dict):
        return None
    is_natural = str(cq.get("本轮是否自然", cq.get("is_natural", ""))).strip().strip("[]")
    is_stuck = str(cq.get("是否卡死", cq.get("is_stuck", ""))).strip().strip("[]")
    not_natural_kw = {"否", "false", "no", "不自然", "不太自然", "不", "异常"}
    is_stuck_kw = {"是", "true", "yes", "卡死", "卡住了", "死循环", "重复"}
    if is_natural.lower() in not_natural_kw or is_stuck.lower() in is_stuck_kw:
        return f"natural={is_natural}, stuck={is_stuck}"
    return None


def get_should_end(tags: Dict[str, Any]) -> bool:
    """从标签中提取用户是否想结束对话的判断

    Returns: True 表示用户本轮想结束对话
    """
    se = tags.get("should_end")
    if isinstance(se, dict):
        val = str(se.get("本轮是否想结束对话", "")).strip().strip("[]")
        return val in ("是", "true", "yes")
    return False
