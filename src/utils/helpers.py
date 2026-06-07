"""共享工具函数"""
import hashlib
from typing import List, Optional


def _safe(text: str) -> str:
    """转义花括号，防止 LLM 生成文本中的 { } 导致 .format() KeyError"""
    return text.replace("{", "{{").replace("}", "}}")


def stable_hash(vector: List[float]) -> str:
    """确定性哈希（跨进程一致），用于画像 label 生成"""
    raw = ",".join(f"{v:.6f}" for v in (vector or []))
    return hashlib.md5(raw.encode()).hexdigest()[:4]
