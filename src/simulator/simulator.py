"""UserSimulator — 根据画像+背景+历史生成用户回复"""
import re
import time
from typing import Any, Dict, List, Tuple

from src.models.case import Case
from src.models.conversation import Turn
from src.llm.client import LLMClient
from src.llm.prompts import SIMULATOR_SYSTEM_PROMPT, TAGGED_OUTPUT_SIMULATOR_PROMPT
from src.simulator.profiles import UserProfile
from src.utils.helpers import _safe


# ============================================================
# 终止检测（规则）
# ============================================================

END_KEYWORDS = [
    # 直接告别
    "再见", "拜拜", "挂了", "先挂了", "挂了啊",
    # 总结结束
    "没别的事了", "没有别的事了", "没事了",
    # 主动挂断
    "那我先忙了", "就这样吧", "那先这样",
    # 感谢结束
    "谢谢啊",
    # 无更多问题
    "没什么问题了", "没问题了",
]


def should_end_conversation(user_text: str) -> bool:
    """规则检测：用户是否表达了挂断/结束意图"""
    text = user_text.strip()
    clean = text.rstrip("。！？，、！?.,;； ")

    for kw in END_KEYWORDS:
        # 精确 endswith 匹配
        if clean.endswith(kw):
            return True
        # 窗口匹配：关键词在末尾附近出现（如"再见站长"中的"再见"）
        if kw in clean[-len(kw)-6:]:
            return True

    # 短文本全串匹配
    if len(text) <= 10:
        for kw in END_KEYWORDS:
            if kw in text:
                return True

    return False


# ============================================================
# UserSimulator
# ============================================================

class UserSimulator:
    """模拟用户，根据画像 + 案例背景 + 对话历史 → 生成下一句回复"""

    def __init__(self, profile: UserProfile, case: Case, client: LLMClient):
        self.profile = profile
        self.case = case
        self.client = client

    def respond(self, history: List[Turn]) -> Tuple[str, Dict[str, Any]]:
        """生成下一句用户回复。返回 (clean_text, parsed_tags)。"""
        system_prompt = self._build_system_prompt()
        system_prompt = system_prompt.replace("{N}", str(len(history) + 1))
        user_message = self._format_history(history)
        raw_response = self.client.chat(system_prompt, user_message)

        from src.simulator.output_parser import parse_simulator_output
        tags, clean_text = parse_simulator_output(raw_response)
        return clean_text, tags

    def _build_system_prompt(self) -> str:
        """组装 system prompt — 参数化画像使用标签输出格式"""
        user_context = self._build_user_context()

        persona = self.profile.effective_description
        adv = _safe(self.profile.adversarial_instruction or "无")
        if persona:
            return TAGGED_OUTPUT_SIMULATOR_PROMPT.format(
                persona_text=_safe(persona),
                user_context=_safe(user_context),
                adversarial_instruction=adv,
            )
        else:
            # 兜底：画像文本为空时使用传统 prompt
            return SIMULATOR_SYSTEM_PROMPT.format(
                profile_description="你是一个美团用户，接到客服来电。",
                user_context=_safe(user_context),
                adversarial_instruction=adv,
            )

    def _build_user_context(self) -> str:
        """从 Case 提取用户背景信息"""
        parts = []
        if self.case.role:
            # Case.role 描述的是客服角色，提取用户相关部分
            parts.append(f"客服身份: {self.case.role}")
        if self.case.task:
            parts.append(f"这通电话的目的: {self.case.task}")
        return "\n".join(parts) if parts else "你接到了一通来自美团客服的电话。"

    def _format_history(self, history: List[Turn]) -> str:
        """把对话历史格式化为文本"""
        next_turn = len(history) + 1

        if not history:
            lines = ["当前是第1轮对话。", "对话刚开始，客服说了开场白。请直接回复。"]
        else:
            lines = [f"当前是第{next_turn}轮对话。以下是目前的对话记录:"]
            for turn in history:
                label = "客服" if turn.speaker == "system" else "你"
                lines.append(f"{label}: {turn.content}")
            lines.append("")

        lines.append("请按系统提示输出所需标签，然后用'--'分隔，输出你的回复。")
        return "\n".join(lines)
