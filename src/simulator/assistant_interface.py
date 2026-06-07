"""Assistant 抽象接口 — 解耦被评测对话模型与编排逻辑"""
from abc import ABC, abstractmethod
from typing import List

from src.models.case import Case
from src.models.conversation import Turn
from src.llm.client import LLMClient


class AssistantInterface(ABC):
    """被评测客服对话模型的抽象接口

    Phase 2: LLMAssistant 实现（基于 LLM 扮演）
    Phase 3: APIAssistant 实现（通过 HTTP API 接入真实模型）
    """

    @abstractmethod
    def respond(self, history: List[Turn]) -> str:
        """根据对话历史生成下一句客服回复"""
        ...


class LLMAssistant(AssistantInterface):
    """基于 LLM 的客服对话模型 — 用 case 指令作为 system prompt"""

    def __init__(self, case: Case, client: LLMClient, use_raw_prompt: bool = True):
        self.case = case
        self.client = client
        self.use_raw_prompt = use_raw_prompt

    def respond(self, history: List[Turn]) -> str:
        system_prompt = self._build_system_prompt()
        user_message = self._format_history(history)
        return self.client.chat(system_prompt, user_message)

    def _build_system_prompt(self) -> str:
        if self.use_raw_prompt and self.case.raw_instruction:
            return self.case.raw_instruction
        return self._build_structured_prompt()

    def _build_structured_prompt(self) -> str:
        parts = []

        if self.case.role:
            parts.append(f"# 你的角色\n{self.case.role}")

        if self.case.task:
            parts.append(f"\n# 任务目标\n{self.case.task}")

        if self.case.opening_line:
            parts.append(f"\n# 开场白\n通话开始时请使用以下开场白:\n「{self.case.opening_line}」")

        if self.case.call_flow:
            parts.append("\n# 通话流程")
            for step in self.case.call_flow:
                parts.append(f"\n## Step {step.step_number}: {step.title}")
                if step.description:
                    parts.append(f"{step.description}")
                if step.sub_steps:
                    for ss in step.sub_steps:
                        parts.append(f"  - {ss}")
                if step.branching:
                    for b in step.branching:
                        parts.append(f"  - 如果{b.condition} → {b.action}")
                if step.reference_script:
                    parts.append(f"  参考话术: {step.reference_script}")

        if self.case.knowledge_points:
            parts.append("\n# 知识点（FAQ）")
            for kp in self.case.knowledge_points:
                parts.append(f"- {kp.topic}: {kp.content}")

        if self.case.constraints:
            parts.append("\n# 约束条件")
            for c in self.case.constraints:
                parts.append(f"- {c.description}")

        parts.append("\n请严格按照以上指令完成通话。保持自然、专业。")
        return "\n".join(parts)

    @staticmethod
    def _format_history(history: List[Turn]) -> str:
        lines = ["以下是当前通话记录:"]
        for turn in history:
            label = "你(客服)" if turn.speaker == "system" else "用户"
            lines.append(f"{label}: {turn.content}")
        lines.append("\n请以客服身份回复下一句。只输出你要说的话，不要加角色标签。")
        return "\n".join(lines)


class APIAssistant(AssistantInterface):
    """通过 HTTP API 接入的真实对话模型（Phase 3 实现）

    Args:
        endpoint: API 地址
        api_key: API 密钥
        timeout: 请求超时（秒）
    """

    def __init__(self, endpoint: str, api_key: str = "", timeout: int = 30):
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout = timeout

    def respond(self, history: List[Turn]) -> str:
        raise NotImplementedError("APIAssistant 将在阶段三实现")
