"""对话加载器 — 从 JSON 文件反序列化 Conversation 对象（用于跨 session 历史回放）"""
import json
import os
from typing import List

from src.models.conversation import Conversation, Turn


def load_conversation_from_json(filepath: str) -> Conversation:
    """从 JSON 文件加载单个 Conversation"""
    with open(filepath, "r", encoding="utf-8") as f:
        return load_conversation_from_dict(json.load(f))


def load_conversation_from_dict(data: dict) -> Conversation:
    """从 dict 反序列化 Conversation"""
    turns = [
        Turn(
            turn_number=t["turn_number"],
            speaker=t["speaker"],
            content=t["content"],
            timestamp=t.get("timestamp", 0.0),
            parsed_tags=t.get("parsed_tags") or {},
        )
        for t in data["turns"]
    ]
    return Conversation(
        id=data["id"],
        case_id=data["case_id"],
        user_profile=data.get("user_profile", ""),
        turns=turns,
        status=data.get("status", "用户挂断"),
        total_turns=data.get("total_turns", len(turns)),
        duration_seconds=data.get("duration_seconds", 0.0),
        complexity_score=data.get("complexity_score", 0.0),
        profile_label=data.get("profile_label", ""),
        adversarial_strategies=data.get("adversarial_strategies", []),
        consistency=data.get("consistency", {}),
        branch_coverage=data.get("branch_coverage", {}),
        model_breakdown_count=data.get("model_breakdown_count", 0),
    )


def load_conversations_from_dir(dirpath: str) -> List[Conversation]:
    """从目录批量加载所有 JSON 对话文件"""
    conversations = []
    for filename in sorted(os.listdir(dirpath)):
        if filename.endswith(".json"):
            filepath = os.path.join(dirpath, filename)
            conv = load_conversation_from_json(filepath)
            conversations.append(conv)
    return conversations
