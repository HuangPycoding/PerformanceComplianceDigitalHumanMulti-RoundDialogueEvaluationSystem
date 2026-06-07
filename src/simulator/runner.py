"""DialogueRunner — 编排 Assistant ↔ UserSimulator 多轮对话"""
import re
import random
import time
import traceback
from typing import Optional, Dict

from src.models.case import Case
from src.models.conversation import Turn, Conversation
from src.llm.client import LLMClient
from src.llm.model_manager import get_simulator_client
from src.simulator.profiles import UserProfile
from src.simulator.simulator import UserSimulator, should_end_conversation
from src.simulator.output_parser import get_conversation_quality_issues, get_should_end
from src.simulator.assistant_interface import AssistantInterface, LLMAssistant


class DialogueRunner:
    """单场对话运行器：Assistant(被评测模型) ↔ UserSimulator(模拟用户)"""

    def __init__(self, case: Case, profile: UserProfile,
                 assistant: AssistantInterface,
                 simulator_client: Optional[LLMClient] = None,
                 use_llm_end_detection: bool = True):
        self.case = case
        self.assistant = assistant
        self.simulator_client = simulator_client or get_simulator_client(temperature=0.7)
        self.simulator = UserSimulator(profile, case, self.simulator_client)
        self.use_llm_end_detection = use_llm_end_detection

    @classmethod
    def create_with_llm(cls, case: Case, profile: UserProfile,
                        assistant_client: LLMClient,
                        simulator_client: Optional[LLMClient] = None,
                        use_raw_prompt: bool = True,
                        use_llm_end_detection: bool = True) -> "DialogueRunner":
        """便捷工厂：用 LLMClient 创建 LLMAssistant 并返回 DialogueRunner"""
        assistant = LLMAssistant(case, assistant_client, use_raw_prompt=use_raw_prompt)
        return cls(case, profile, assistant, simulator_client,
                   use_llm_end_detection=use_llm_end_detection)

    def run(self, max_turns: int = 20) -> Conversation:
        """执行一场完整对话"""
        if max_turns < 2:
            raise ValueError(f"max_turns 至少为 2，当前值: {max_turns}")

        conv_id = f"case{self.case.id}_{self.simulator.profile.label}_{int(time.time())}"
        conv = Conversation(
            id=conv_id,
            case_id=self.case.id,
            user_profile=self.simulator.profile.label,
        )

        # 从 profile 填充元数据
        conv.sampled_vector = self.simulator.profile.sampled_vector
        conv.verified_vector = self.simulator.profile.verified_vector
        conv.adversarial_strategies = self.simulator.profile.adversarial_strategy or []
        conv.profile_label = self.simulator.profile.label

        start_time = time.time()
        user_end_signals = 0
        turns_since_last_end_signal = 0
        consecutive_breakdowns = 0
        min_steps = len(self.case.call_flow) if self.case.call_flow else 5
        min_turns = min(max(min_steps + 1, 5), max_turns // 2)

        try:
            # Turn 1: Assistant 开场白（解析 ${variable} 占位符后使用）
            opening = _resolve_placeholders(self.case.opening_line or "您好，我是美团客服。")
            conv.turns.append(Turn(turn_number=1, speaker="system",
                                   content=opening, timestamp=0.0))

            # Turn 2+: 交替对话
            for turn_num in range(2, max_turns + 1):
                is_user_turn = (turn_num % 2 == 0)

                if is_user_turn:
                    text, parsed_tags = self.simulator.respond(conv.turns)

                    # 后处理: 修正 emotion_change 中"上轮情绪"为上一轮实际的 emotion 文本
                    _fix_emotion_change(parsed_tags, conv.turns)

                    ts = time.time() - start_time
                    conv.turns.append(Turn(
                        turn_number=turn_num, speaker="user",
                        content=text, timestamp=ts,
                        parsed_tags=parsed_tags,
                    ))

                    # R7: 模型崩溃检测
                    issue = get_conversation_quality_issues(parsed_tags)
                    if issue:
                        consecutive_breakdowns += 1
                    else:
                        consecutive_breakdowns = 0
                    if consecutive_breakdowns >= 2:
                        conv.status = "异常中断"
                        conv.model_breakdown_count = consecutive_breakdowns
                        break

                    # 结束检测：LLM 标签优先，关键词兜底
                    user_wants_end = False
                    if self.use_llm_end_detection and parsed_tags:
                        user_wants_end = get_should_end(parsed_tags)
                    if not user_wants_end:
                        user_wants_end = should_end_conversation(text)

                    if user_wants_end:
                        user_end_signals += 1
                        turns_since_last_end_signal = 0
                    else:
                        turns_since_last_end_signal += 1
                        # 连续3轮用户未表达结束意图 → 重置信号计数
                        if turns_since_last_end_signal >= 3:
                            user_end_signals = 0
                            turns_since_last_end_signal = 0

                    if user_end_signals >= 2 and turn_num >= min_turns:
                        conv.status = "用户挂断"
                        break
                else:
                    text = self.assistant.respond(conv.turns)
                    ts = time.time() - start_time
                    conv.turns.append(Turn(
                        turn_number=turn_num, speaker="system",
                        content=text, timestamp=ts,
                    ))
            else:
                conv.status = "超时"

        except Exception as e:
            conv.status = "异常中断"
            conv.turns.append(Turn(
                turn_number=len(conv.turns) + 1,
                speaker="system",
                content=f"[对话异常: {str(e)}]",
                timestamp=time.time() - start_time,
            ))

        conv.total_turns = len(conv.turns)
        conv.duration_seconds = round(time.time() - start_time, 1)
        return conv


# 中文常见姓氏 + 名字生成
_SURNAMES = ["张", "李", "王", "刘", "陈", "杨", "赵", "黄", "周", "吴",
             "徐", "孙", "马", "朱", "胡", "郭", "何", "高", "林", "罗"]
_GIVEN = ["伟", "强", "磊", "军", "勇", "杰", "涛", "明", "超", "华",
          "丽", "敏", "静", "秀英", "芳", "娜", "婷", "雪", "玲", "红",
          "建国", "志强", "文博", "浩然", "子涵", "宇轩", "一鸣"]

# 数值占位符默认值映射
_DEFAULT_VALUES: Dict[str, str] = {
    "X": "8", "Y": "12", "Z": "20", "W": "7",
    "x": "8", "y": "12", "z": "20", "w": "7",
    "rider_name": None,  # 动态生成
}


def _resolve_placeholders(text: str) -> str:
    """替换文本中的 ${variable} 占位符。

    - ${rider_name} / ${rider} → 随机生成中文姓名
    - ${X}, ${Y} 等数值占位符 → 使用默认值
    - 未知占位符保留原样
    """
    if not text or "${" not in text:
        return text

    def _replacer(match: re.Match) -> str:
        var = match.group(1).strip()
        # rider name
        if var.lower() in ("rider_name", "rider", "ridername"):
            return random.choice(_SURNAMES) + random.choice(_GIVEN)
        # numeric defaults
        if var in _DEFAULT_VALUES and _DEFAULT_VALUES[var] is not None:
            return _DEFAULT_VALUES[var]
        # unknown — keep as-is
        return match.group(0)

    return re.sub(r'\$\{(\w+)\}', _replacer, text)


def _fix_emotion_change(parsed_tags: dict, history: list) -> None:
    """后处理: 修正 emotion_change 中"上轮情绪"和"本轮情绪"的不一致

    LLM 在每个 turn 独立生成 emotion_change 时倾向于简化"上轮情绪"文本，
    且可能写出与当前 state.emotion 不同的"本轮情绪"。
    此函数利用历史数据和当前 state 将两侧都修正为实际值。
    """
    state = parsed_tags.get("state")
    if not isinstance(state, dict):
        return
    ec = str(state.get("emotion_change", ""))
    if "→" not in ec:
        return
    current_emotion = str(state.get("emotion", ""))
    # 找到上一轮 user turn 的 state
    prev_emotion = None
    for t in reversed(history):
        if t.speaker == "user" and t.parsed_tags:
            ps = t.parsed_tags.get("state")
            if isinstance(ps, dict):
                prev_emotion = str(ps.get("emotion", ""))
                break
    if not prev_emotion:
        return
    # 计算修正后的 emotion_change
    _, after = ec.split("→", 1)
    new_from = prev_emotion
    new_to = current_emotion if current_emotion else after
    fixed = f"{new_from}→{new_to}"
    if fixed != ec:
        state["emotion_change"] = fixed
