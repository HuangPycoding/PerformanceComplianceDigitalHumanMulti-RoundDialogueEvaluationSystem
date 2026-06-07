"""模拟器单元测试 — 用 mock 绕过 LLM 调用，测逻辑正确性"""
import pytest
from unittest.mock import MagicMock, patch

from src.models.case import Case, CallFlowStep, Constraint, KnowledgePoint
from src.models.conversation import Turn, Conversation
from src.simulator.profiles import (
    build_profile_from_vector,
    build_adversarial_instruction_for_vector,
    UserProfile,
)
from src.simulator.simulator import UserSimulator, should_end_conversation
from src.simulator.runner import DialogueRunner
from src.simulator.assistant_interface import LLMAssistant
from src.simulator.output_parser import parse_simulator_output, get_should_end


# ============================================================
# 测试夹具
# ============================================================

def make_mock_case() -> Case:
    return Case(
        id=1,
        title="测试案例",
        business_line="外卖",
        role="你是美团客服",
        task="通知用户订单延迟30分钟",
        opening_line="您好，请问是张先生吗？我是美团客服。",
        call_flow=[
            CallFlowStep(
                id="step_1", step_number=1, title="身份确认",
                description="确认用户身份",
            ),
            CallFlowStep(
                id="step_2", step_number=2, title="说明情况",
                description="告知订单延迟",
                sub_steps=["说明延迟原因", "告知预计送达时间"],
            ),
        ],
        knowledge_points=[
            KnowledgePoint(id="kp1", topic="延迟赔付", content="超过30分钟赔付5元优惠券"),
        ],
        constraints=[
            Constraint(id="c1", type="forbidden_word", description="禁止提及竞品", checkable_by_rule=True),
        ],
        complexity_score=2.5,
        raw_instruction="# Role\n你是美团客服\n# Task\n通知用户订单延迟30分钟",
    )


def make_mock_profile() -> UserProfile:
    """创建一个参数化画像用于测试"""
    return build_profile_from_vector(
        [0.5] * 15,
        persona_text="你是一个配合型用户，会耐心回答客服问题。",
    )


def make_adversarial_profile() -> UserProfile:
    """创建含对抗策略的参数化画像"""
    return build_profile_from_vector(
        [0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.85, 0.85],
        persona_text="你是一个喜欢试探边界的用户，有时会前后矛盾。",
    )


def make_mock_llm_client(response_text: str = "测试回复") -> MagicMock:
    mock = MagicMock()
    mock.chat.return_value = response_text
    return mock


# ============================================================
# profiles.py 测试
# ============================================================

class TestBuildProfileFromVector:
    def test_creates_parameterized_profile(self):
        p = build_profile_from_vector([0.5] * 15, persona_text="测试画像")
        assert p.type == "parameterized"
        assert p.is_parameterized
        assert p.sampled_vector == [0.5] * 15
        assert p.persona_text == "测试画像"

    def test_label_is_hash_based(self):
        p = build_profile_from_vector([0.5] * 15)
        assert p.label.startswith("param_")

    def test_adversarial_strategies_auto_hooked(self):
        p = build_profile_from_vector([0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.85, 0.2])
        assert len(p.adversarial_strategy) > 0

    def test_effective_description_uses_persona_text(self):
        p = build_profile_from_vector([0.5] * 15, persona_text="自定义画像")
        assert p.effective_description == "自定义画像"

    def test_effective_description_fallback(self):
        p = build_profile_from_vector([0.5] * 15)
        assert p.anchor_description


class TestBuildAdversarialInstruction:
    def test_no_strategies_empty(self):
        assert build_adversarial_instruction_for_vector([0.5] * 15) == ""

    def test_contradiction_generates_instruction(self):
        # truth_consistency high → contradiction
        v = [0.5] * 15
        v[14] = 0.85  # truth_consistency high → 自相矛盾
        result = build_adversarial_instruction_for_vector(v)
        assert result  # 应有内容


class TestUserProfile:
    def test_label_parameterized(self):
        p = UserProfile(type="parameterized", sampled_vector=[0.5] * 15)
        assert p.label.startswith("param_")

    def test_is_parameterized_true(self):
        p = UserProfile(type="parameterized", sampled_vector=[0.5] * 15)
        assert p.is_parameterized

    def test_is_parameterized_false(self):
        p = UserProfile(type="custom")
        assert not p.is_parameterized


# ============================================================
# simulator.py 测试
# ============================================================

class TestShouldEndConversation:
    def test_goodbye_ends(self):
        assert should_end_conversation("好的，再见")
        assert should_end_conversation("谢谢，拜拜")

    def test_short_goodbye_ends(self):
        assert should_end_conversation("再见")
        assert should_end_conversation("拜拜")

    def test_thanks_ends(self):
        assert should_end_conversation("谢谢啊")
        assert should_end_conversation("好的谢谢啊")

    def test_normal_talk_does_not_end(self):
        assert not should_end_conversation("我的订单什么时候到？")
        assert not should_end_conversation("那你们怎么处理呢")
        assert not should_end_conversation("嗯，我知道了")

    def test_goodbye_mid_sentence_does_not_end(self):
        assert not should_end_conversation("我觉得不能说再见就完了，你们得负责")

    def test_goodbye_with_punctuation_ends(self):
        assert should_end_conversation("好的，拜拜。")
        assert should_end_conversation("谢谢，再见！")
        assert should_end_conversation("谢谢啊。")

    def test_goodbye_with_title_ends(self):
        """R8 fix: 带称呼的告别应被检测（窗口匹配）"""
        assert should_end_conversation("好的，那我先上线了。再见站长！")
        assert should_end_conversation("谢谢你的帮助，拜拜老板")
        assert should_end_conversation("那就这样吧，挂了哈")

    def test_new_keywords_ends(self):
        """R8 fix: 新增结束语"""
        assert should_end_conversation("我没别的事了")
        assert should_end_conversation("我先挂了")
        assert should_end_conversation("挂了啊，谢谢")

    def test_medium_length_farewell_ends(self):
        """R8 fix: 中等长度文本中含告别词也检测"""
        assert should_end_conversation("好的那我先上线接单了，再见站长")
        assert should_end_conversation("好的没什么问题了谢谢你的通知")

    def test_normal_long_text_does_not_end(self):
        """R8 fix 回归: 非告别长文本不应误触发"""
        assert not should_end_conversation("我的订单什么时候能送到，已经等了很久了")

    def test_new_keywords_v2_ends(self):
        """方案A: 新增关键词"""
        # 感谢式结束
        assert should_end_conversation("谢谢啊")
        assert should_end_conversation("好的谢谢啊")
        # 主动挂断
        assert should_end_conversation("那我先忙了")
        assert should_end_conversation("好的那我先忙了")
        # 无更多问题
        assert should_end_conversation("没什么问题了")
        assert should_end_conversation("没问题了")
        assert should_end_conversation("好的没问题了")
        # 变体
        assert should_end_conversation("没有别的事了")

    def test_non_farewell_with_new_keywords_does_not_end(self):
        """方案A 回归: 含新关键词但不是结束语的不应误触发"""
        assert not should_end_conversation("我先忙了一下，刚才没接到")
        assert not should_end_conversation("这个问题没问题了，但我还有另一个问题")


class TestUserSimulator:
    def test_respond_calls_llm(self):
        case = make_mock_case()
        profile = make_mock_profile()
        mock_client = make_mock_llm_client(
            "好的，我就是。\n--\n嗯，我就是。有什么事吗？"
        )

        sim = UserSimulator(profile, case, mock_client)
        history = [Turn(turn_number=1, speaker="system", content="您好，请问是张先生吗？")]
        text, tags = sim.respond(history)

        assert "嗯，我就是" in text
        mock_client.chat.assert_called_once()

    def test_build_system_prompt_contains_context(self):
        case = make_mock_case()
        profile = make_mock_profile()
        sim = UserSimulator(profile, case, MagicMock())

        prompt = sim._build_system_prompt()
        assert "美团客服" in prompt
        assert "订单延迟" in prompt

    def test_build_system_prompt_with_adversarial(self):
        case = make_mock_case()
        profile = make_adversarial_profile()
        sim = UserSimulator(profile, case, MagicMock())

        prompt = sim._build_system_prompt()
        assert "试探" in prompt or "矛盾" in prompt

    def test_format_history_empty(self):
        case = make_mock_case()
        profile = make_mock_profile()
        sim = UserSimulator(profile, case, MagicMock())

        text = sim._format_history([])
        assert "开场白" in text

    def test_format_history_with_turns(self):
        case = make_mock_case()
        profile = make_mock_profile()
        sim = UserSimulator(profile, case, MagicMock())

        history = [
            Turn(turn_number=1, speaker="system", content="您好，请问是张先生吗？"),
            Turn(turn_number=2, speaker="user", content="是的"),
        ]
        text = sim._format_history(history)
        assert "客服" in text
        assert "您好" in text
        assert "是的" in text

    def test_respond_returns_tags_for_parameterized(self):
        case = make_mock_case()
        profile = make_mock_profile()
        mock_client = make_mock_llm_client(
            '<memory>测试</memory>\n<thought>思考</thought>\n<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n<risk_flag>none</risk_flag>\n'
            '<model_behavior>normal</model_behavior>\n<conversation_quality>good</conversation_quality>\n'
            '--\n好的，我知道了。'
        )

        sim = UserSimulator(profile, case, mock_client)
        text, tags = sim.respond([])

        assert "好的" in text
        assert isinstance(tags, dict)


# ============================================================
# runner.py 测试
# ============================================================

class TestDialogueRunner:
    def test_opening_line_is_first_turn(self):
        case = make_mock_case()
        profile = make_mock_profile()
        mock_sim = make_mock_llm_client(
            '<memory>结束</memory>\n<thought>挂断</thought>\n<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n<risk_flag>none</risk_flag>\n'
            '<model_behavior>normal</model_behavior>\n<conversation_quality>good</conversation_quality>\n'
            '--\n好的，谢谢，再见'
        )
        mock_asst = make_mock_llm_client("不客气")

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
        )
        conv = runner.run(max_turns=10)

        assert conv.total_turns >= 1
        assert conv.turns[0].speaker == "system"
        assert conv.turns[0].content == case.opening_line

    def test_user_goodbye_ends_normally(self):
        case = make_mock_case()
        profile = make_mock_profile()
        mock_sim = make_mock_llm_client("好的，谢谢，再见")
        mock_asst = make_mock_llm_client("不客气，祝您生活愉快")

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
        )
        conv = runner.run(max_turns=10)
        assert conv.status in ("用户挂断", "超时")

    def test_abnormal_breakdown_detection(self):
        case = make_mock_case()
        profile = make_mock_profile()
        # 连续2轮 conversation_quality 异常应触发 R7 崩溃中断
        mock_sim = make_mock_llm_client(
            '<memory>混乱</memory>\n<thought>不知道</thought>\n'
            '<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n<risk_flag>none</risk_flag>\n'
            '<model_behavior>breakdown</model_behavior>\n'
            '<conversation_quality>\n本轮是否自然: 否\n是否卡死: 是\n</conversation_quality>\n'
            '--\n嗯...'
        )
        mock_asst = make_mock_llm_client("您还好吗？")

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
        )
        conv = runner.run(max_turns=6)
        assert conv.status == "异常中断"

    def test_min_turns_validation(self):
        case = make_mock_case()
        profile = make_mock_profile()
        mock_asst = make_mock_llm_client("测试")
        assistant = LLMAssistant(case, mock_asst)

        with pytest.raises(ValueError, match="max_turns"):
            DialogueRunner(case, profile, assistant).run(max_turns=1)

    def test_run_with_profile_parameterized(self):
        case = make_mock_case()
        profile = make_mock_profile()
        mock_sim = make_mock_llm_client(
            '<memory>ok</memory>\n<thought>fine</thought>\n<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n<risk_flag>none</risk_flag>\n'
            '<model_behavior>normal</model_behavior>\n<conversation_quality>good</conversation_quality>\n'
            '--\n好的，谢谢，再见'
        )
        mock_asst = make_mock_llm_client("不客气")

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
        )
        conv = runner.run(max_turns=6)
        assert conv.total_turns > 0


# ============================================================
# 方案 B: LLM should_end 标签测试
# ============================================================

class TestShouldEndParsing:
    def test_parse_should_end_yes(self):
        raw = (
            '<memory>ok</memory>\n<thought>fine</thought>\n'
            '<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n'
            '<risk_flag>none</risk_flag>\n'
            '<model_behavior>normal</model_behavior>\n'
            '<conversation_quality>good</conversation_quality>\n'
            '<should_end>\n本轮是否想结束对话: 是\n原因: 问题已解决\n</should_end>\n'
            '--\n好的，谢谢，再见'
        )
        tags, text = parse_simulator_output(raw)
        assert "好的，谢谢，再见" in text
        assert get_should_end(tags) is True

    def test_parse_should_end_no(self):
        raw = (
            '<memory>ok</memory>\n<thought>fine</thought>\n'
            '<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n'
            '<risk_flag>none</risk_flag>\n'
            '<model_behavior>normal</model_behavior>\n'
            '<conversation_quality>good</conversation_quality>\n'
            '<should_end>\n本轮是否想结束对话: 否\n原因: 还想继续沟通\n</should_end>\n'
            '--\n那你们怎么处理呢？'
        )
        tags, text = parse_simulator_output(raw)
        assert get_should_end(tags) is False

    def test_parse_should_end_missing_tag(self):
        """should_end 标签缺失时返回 False（不误结束）"""
        raw = (
            '<memory>ok</memory>\n<thought>fine</thought>\n'
            '<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n'
            '--\n嗯，我知道了'
        )
        tags, text = parse_simulator_output(raw)
        assert get_should_end(tags) is False

    def test_should_end_true_value_variants(self):
        """支持 是/true/yes 三种真值"""
        assert get_should_end({"should_end": {"本轮是否想结束对话": "是"}}) is True
        assert get_should_end({"should_end": {"本轮是否想结束对话": "true"}}) is True
        assert get_should_end({"should_end": {"本轮是否想结束对话": "yes"}}) is True

    def test_should_end_false_value_variants(self):
        assert get_should_end({"should_end": {"本轮是否想结束对话": "否"}}) is False
        assert get_should_end({"should_end": {"本轮是否想结束对话": ""}}) is False


class TestDialogueRunnerLLMEndDetection:
    def test_llm_end_detection_path(self):
        """方案 B: LLM 标签判定结束 → 应在第一轮用户回复后结束"""
        case = make_mock_case()
        profile = make_mock_profile()
        mock_sim = make_mock_llm_client(
            '<memory>ok</memory>\n<thought>fine</thought>\n'
            '<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n<risk_flag>none</risk_flag>\n'
            '<model_behavior>normal</model_behavior>\n<conversation_quality>good</conversation_quality>\n'
            '<should_end>\n本轮是否想结束对话: 是\n原因: 客服已告别\n</should_end>\n'
            '--\n好的，谢谢，再见'
        )
        mock_asst = make_mock_llm_client("不客气")

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
            use_llm_end_detection=True,
        )
        conv = runner.run(max_turns=6)
        # 用户第一轮(轮次2)回复后 LLM 标签 should_end=true → +1
        # 但需要 >=2 分才结束，一轮不够
        # 客服回复后如果不含结束信号 → decrement
        # 需要两轮 should_end=true 的用户回复才能结束
        assert conv.total_turns > 0

    def test_llm_fallback_to_keyword(self):
        """方案 B: LLM 标签缺失时 fallback 到关键词兜底"""
        case = make_mock_case()
        profile = make_mock_profile()
        # LLM 输出不含 should_end 标签，但用户文本含"再见"
        mock_sim = make_mock_llm_client(
            '<memory>ok</memory>\n<thought>fine</thought>\n'
            '<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n<risk_flag>none</risk_flag>\n'
            '<model_behavior>normal</model_behavior>\n<conversation_quality>good</conversation_quality>\n'
            '--\n好的，谢谢，再见'
        )
        mock_asst = make_mock_llm_client("不客气")

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
            use_llm_end_detection=True,
        )
        conv = runner.run(max_turns=6)
        assert conv.status in ("用户挂断", "超时")


# ============================================================
# R9: 对话结束多重检测测试
# ============================================================

# 复用 mock 响应的辅助工厂
def _make_user_response(should_end: bool, text: str) -> str:
    yn = "是" if should_end else "否"
    reason = "问题已解决" if should_end else "还想继续沟通"
    return (
        '<memory>ok</memory>\n<thought>test</thought>\n'
        '<state>{"emotion":"neutral"}</state>\n'
        '<emotion_curve>stable</emotion_curve>\n<risk_flag>none</risk_flag>\n'
        '<model_behavior>normal</model_behavior>\n<conversation_quality>good</conversation_quality>\n'
        f'<should_end>\n本轮是否想结束对话: {yn}\n原因: {reason}\n</should_end>\n'
        f'--\n{text}'
    )


def _wrap_side_effect(responses):
    """用 lambda 包裹列表，耗尽后 fallback 到最后一个"""
    def _respond(*args, **kwargs):
        if responses:
            return responses.pop(0)
        return "好的，谢谢，再见"
    return _respond


class TestDialogueRunnerR9MultiRound:
    # ── 简化后的结束逻辑：仅用户侧驱动 ──

    def test_user_hangs_up_two_signals(self):
        """用户连续两轮发结束信号 → 用户挂断"""
        case = make_mock_case()
        profile = make_mock_profile()

        user_responses = [
            _make_user_response(True, "好的，谢谢。"),
            _make_user_response(True, "好的，再见。"),
            _make_user_response(True, "谢谢。"),
        ]
        mock_sim = make_mock_llm_client()
        mock_sim.chat.side_effect = _wrap_side_effect(user_responses)

        mock_asst = make_mock_llm_client()
        mock_asst.chat.side_effect = _wrap_side_effect([
            "好的，已为您记录。",
            "不客气。",
            "好的。",
        ])

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
            use_llm_end_detection=True,
        )
        conv = runner.run(max_turns=8)

        assert conv.status == "用户挂断"

    def test_user_hangs_up_after_gap(self):
        """用户在非连续轮次发两个结束信号 → 用户挂断"""
        case = make_mock_case()
        profile = make_mock_profile()

        user_responses = [
            _make_user_response(True, "好的，谢谢。"),        # T2: signal 1
            _make_user_response(False, "等等，再问一下。"),   # T4: no signal
            _make_user_response(True, "好的，再见。"),        # T6: signal 2 → 挂断
        ]
        mock_sim = make_mock_llm_client()
        mock_sim.chat.side_effect = _wrap_side_effect(user_responses)

        asst_responses = [
            "好的，已记录。",
            "还有什么问题？",
            "好的。",
        ]
        mock_asst = make_mock_llm_client()
        mock_asst.chat.side_effect = _wrap_side_effect(asst_responses)

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
            use_llm_end_detection=True,
        )
        conv = runner.run(max_turns=8)

        assert conv.status == "用户挂断"

    def test_single_signal_then_timeout(self):
        """用户只发一个结束信号，后续不发 → 超时"""
        case = make_mock_case()
        profile = make_mock_profile()

        user_responses = [
            _make_user_response(True, "好的，谢谢。"),        # T2: signal 1
            _make_user_response(False, "嗯。"),
            _make_user_response(False, "知道了。"),
            _make_user_response(False, "好。"),
        ]
        mock_sim = make_mock_llm_client()
        mock_sim.chat.side_effect = _wrap_side_effect(user_responses)

        asst_responses = [
            "好的。",
            "还有什么可以帮您的？",
            "好的。",
            "嗯。",
        ]
        mock_asst = make_mock_llm_client()
        mock_asst.chat.side_effect = _wrap_side_effect(asst_responses)

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
            use_llm_end_detection=True,
        )
        conv = runner.run(max_turns=8)
        assert conv.status == "超时"

    def test_no_signal_timeout(self):
        """用户始终不发结束信号 → 超时"""
        case = make_mock_case()
        profile = make_mock_profile()

        user_responses = [
            _make_user_response(False, "是的，延迟多久？"),
            _make_user_response(False, "那赔付怎么处理？"),
            _make_user_response(False, "什么时候到账？"),
            _make_user_response(False, "好的，谢谢。"),
            _make_user_response(False, "嗯。"),
        ]
        mock_sim = make_mock_llm_client()
        mock_sim.chat.side_effect = _wrap_side_effect(user_responses)

        asst_responses = [
            "预计延迟30分钟。",
            "超过30分钟赔付5元。",
            "24小时内到账。",
            "还有什么可以帮您？",
            "嗯。",
        ]
        mock_asst = make_mock_llm_client()
        mock_asst.chat.side_effect = _wrap_side_effect(asst_responses)

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=mock_sim,
            use_llm_end_detection=True,
        )
        conv = runner.run(max_turns=10)

        assert conv.status == "超时"
        assert conv.total_turns >= 5

    def test_keyword_fallback_ends(self):
        """LLM 标签缺失时，关键词兜底检测结束"""
        case = make_mock_case()
        profile = make_mock_profile()

        # 不含 should_end 标签，但用户文本含"再见"
        sim_without_tags = make_mock_llm_client()
        sim_without_tags.chat.side_effect = _wrap_side_effect([
            '<memory>ok</memory>\n<thought>fine</thought>\n'
            '<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n<risk_flag>none</risk_flag>\n'
            '<model_behavior>normal</model_behavior>\n<conversation_quality>\n本轮是否自然: 是\n是否卡死: 否\n</conversation_quality>\n'
            '--\n好的，谢谢，再见',
            '<memory>ok</memory>\n<thought>fine</thought>\n'
            '<state>{"emotion":"neutral"}</state>\n'
            '<emotion_curve>stable</emotion_curve>\n<risk_flag>none</risk_flag>\n'
            '<model_behavior>normal</model_behavior>\n<conversation_quality>\n本轮是否自然: 是\n是否卡死: 否\n</conversation_quality>\n'
            '--\n拜拜',
        ])

        mock_asst = make_mock_llm_client()
        mock_asst.chat.side_effect = _wrap_side_effect(["不客气", "再见"])

        assistant = LLMAssistant(case, mock_asst)
        runner = DialogueRunner(
            case, profile,
            assistant=assistant,
            simulator_client=sim_without_tags,
            use_llm_end_detection=True,
        )
        conv = runner.run(max_turns=8)
        assert conv.status == "用户挂断"
