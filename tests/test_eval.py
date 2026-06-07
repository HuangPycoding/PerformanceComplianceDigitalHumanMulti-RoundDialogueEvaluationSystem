"""Phase 3 集成测试 — 清单生成 / 核查 / 评级 / 表面合规 / 归因 / CONSTRAINT 分流"""
import pytest

from src.eval.checklist_generator import generate_checklist
from src.eval.config import (
    CONFIDENCE,
    INDICATIVE_SCORES,
    RATING_THRESHOLDS,
    SOURCE_WEIGHTS,
)
from src.eval.rules import (
    check_rule_constraints,
    classify_constraints,
    compute_complexity_score,
    compute_tier1_metrics,
    extract_turn_signals,
    format_signal_context,
)
from src.models.case import (
    Branch,
    CallFlowStep,
    Case,
    Constraint,
    KnowledgePoint,
)
from src.models.conversation import Conversation, Turn
from src.models.evaluation import (
    AttributionItem,
    CheckResult,
    Defect,
    DimensionChecklist,
    EvalConfidence,
    EvalResult,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def sample_case() -> Case:
    """构造一个包含完整指令的测试 Case"""
    return Case(
        id=2,
        title="外卖超时赔付",
        business_line="外卖",
        role="美团外卖客服",
        task="处理用户关于订单超时的投诉并完成赔付",
        opening_line="您好，我是美团客服，请问有什么可以帮您？",
        call_flow=[
            CallFlowStep(
                id="step_1", step_number=1, title="身份确认",
                description="确认用户手机号后四位",
                sub_steps=["询问手机号", "系统验证"],
            ),
            CallFlowStep(
                id="step_2", step_number=2, title="了解问题",
                description="询问订单号和具体情况",
                branching=[
                    Branch(condition="用户愿意等", target_step=3, action="安抚并告知预计时间"),
                    Branch(condition="用户不愿等", target_step=4, action="直接进入赔付流程"),
                ],
            ),
            CallFlowStep(
                id="step_3", step_number=3, title="查询订单",
                description="查询订单配送状态",
            ),
            CallFlowStep(
                id="step_4", step_number=4, title="赔付处理",
                description="确认赔付金额并执行",
            ),
        ],
        knowledge_points=[
            KnowledgePoint(id="kp1", topic="超时赔付标准", content="超时30分钟赔付5元，超时60分钟赔付10元"),
            KnowledgePoint(id="kp2", topic="赔付到账时间", content="赔付将在1-3个工作日内到账"),
        ],
        constraints=[
            Constraint(id="c1", type="safety", description="必须核实用户身份后才能操作订单",
                       checkable_by_rule=False),
            Constraint(id="c2", type="tone", description="保持礼貌和同理心",
                       checkable_by_rule=False),
            Constraint(id="c3", type="word_limit", description="单轮回复不超过200字",
                       checkable_by_rule=True, rule_pattern="200"),
            Constraint(id="c4", type="forbidden_word", description="禁止使用'不知道'一词",
                       checkable_by_rule=True, rule_pattern="不知道"),
        ],
        complexity_score=5.0,
    )


@pytest.fixture
def sample_conversation() -> Conversation:
    """构造一个测试对话"""
    conv = Conversation(
        id="test_conv_1",
        case_id=2,
        user_profile="急躁型用户",
        status="用户挂断",
        total_turns=6,
        turns=[
            Turn(turn_number=1, speaker="system",
                 content="您好，我是美团客服，请问有什么可以帮您？",
                 parsed_tags={"state": {"emotion": "中性", "emotion_intensity": 0.3}}),
            Turn(turn_number=2, speaker="user",
                 content="我点的外卖超了一个小时了还没到！你们怎么回事？",
                 parsed_tags={
                     "state": {"emotion": "生气", "emotion_intensity": 0.8},
                     "model_behavior": {"用户评价": "不满", "是否改变态度": "否"},
                 }),
            Turn(turn_number=3, speaker="system",
                 content="非常抱歉给您带来不便。请问您的手机号后四位是多少？我帮您查询一下。",
                 parsed_tags={"state": {"emotion": "中性", "emotion_intensity": 0.3}}),
            Turn(turn_number=4, speaker="user",
                 content="138****5678。你们每次都这样，我要投诉！",
                 parsed_tags={
                     "state": {"emotion": "愤怒", "emotion_intensity": 0.9},
                     "model_behavior": {"用户评价": "不满", "是否改变态度": "是"},
                     "should_end": {"本轮是否想结束对话": "否"},
                 }),
            Turn(turn_number=5, speaker="system",
                 content="好的，我已经查到您的订单。根据我们的规定，超时60分钟赔付10元，我马上为您处理。",
                 parsed_tags={
                     "state": {"emotion": "中性", "emotion_intensity": 0.3},
                     "conversation_quality": {"是否卡死": "否"},
                 }),
            Turn(turn_number=6, speaker="user",
                 content="好吧，那快点处理。",
                 parsed_tags={
                     "state": {"emotion": "不满", "emotion_intensity": 0.5},
                     "model_behavior": {"用户评价": "满意"},
                     "should_end": {"本轮是否想结束对话": "是"},
                 }),
        ],
    )
    return conv


# ============================================================
# 清单生成测试
# ============================================================

class TestChecklistGeneration:
    """测试三层清单生成"""

    def test_flow_coverage_items(self, sample_case):
        """FLOW_COVERAGE 应生成步骤+质量+分支+顺序项"""
        signals = {"satisfaction_trajectory": []}
        items, relations = generate_checklist(sample_case, signals, "FLOW_COVERAGE")
        item_ids = [i["item_id"] for i in items]

        assert len(items) >= 8, f"FLOW_COVERAGE 清单项过少: {len(items)}"
        assert "step_1_executed" in item_ids
        assert "step_1_quality" in item_ids
        assert "branch_2_1_correct" in item_ids  # step_2 有一个分支
        assert "sequence_order" in item_ids

    def test_safety_items(self, sample_case):
        """SAFETY 应包含关键安全项"""
        signals = {}
        items, _ = generate_checklist(sample_case, signals, "SAFETY")
        item_ids = [i["item_id"] for i in items]

        assert len(items) >= 5
        assert "identity_verification" in item_ids
        assert "info_protection" in item_ids
        assert "output_safety" in item_ids

    def test_knowledge_items(self, sample_case):
        """KNOWLEDGE 应从 knowledge_points 生成核查项"""
        signals = {}
        items, _ = generate_checklist(sample_case, signals, "KNOWLEDGE")
        item_ids = [i["item_id"] for i in items]

        assert "kp_1_accuracy" in item_ids
        assert "kp_2_accuracy" in item_ids
        assert "no_hallucination" in item_ids

    def test_constraint_items_checkable_filtered(self, sample_case):
        """CONSTRAINT 应过滤 checkable_by_rule=True 的项"""
        signals = {}
        items, _ = generate_checklist(sample_case, signals, "CONSTRAINT")
        item_ids = [i["item_id"] for i in items]

        # c1 (safety, checkable=False) 应被包含
        # c2 (tone, checkable=False) 应被包含
        # c3 (word_limit, checkable=True) 应被过滤
        # c4 (forbidden_word, checkable=True) 应被过滤
        for item in items:
            assert "200" not in item["description"], "字数限制应由规则层处理"
            assert "不知道" not in item["description"], "禁止词应由规则层处理"

    def test_checklist_size_capped(self, sample_case):
        """清单项数量不应超过 CHECKLIST_SIZE 上限"""
        signals = {"satisfaction_trajectory": []}
        for dim in ["FLOW_COVERAGE", "TASK_COMPLETION", "SAFETY"]:
            items, _ = generate_checklist(sample_case, signals, dim)
            from src.eval.config import CHECKLIST_SIZE
            max_size = CHECKLIST_SIZE.get(dim, (5, 12))[1]
            assert len(items) <= max_size, f"{dim} 清单项 {len(items)} 超上限 {max_size}"


# ============================================================
# Tier 1 规则指标测试
# ============================================================

class TestTier1Metrics:
    """测试 11 个规则指标"""

    def test_turns_ratio(self, sample_conversation, sample_case):
        metrics = compute_tier1_metrics(sample_conversation, sample_case)
        assert "turns_ratio" in metrics
        assert metrics["turns_ratio"] > 0

    def test_stuck_detection(self, sample_conversation, sample_case):
        metrics = compute_tier1_metrics(sample_conversation, sample_case)
        assert "stuck_count" in metrics
        assert "stuck_ratio" in metrics

    def test_hangup_detected(self, sample_conversation, sample_case):
        metrics = compute_tier1_metrics(sample_conversation, sample_case)
        assert "hangup_detected" in metrics
        assert isinstance(metrics["hangup_detected"], dict)

    def test_forbidden_word_hits(self, sample_conversation, sample_case):
        metrics = compute_tier1_metrics(sample_conversation, sample_case)
        assert "forbidden_word_hits" in metrics
        # 对话中没有 "不知道"
        assert metrics["forbidden_word_hits"] == 0


# ============================================================
# Tier 1.5 信号提取测试
# ============================================================

class TestSignalExtraction:
    """测试 Turn 级信号提取"""

    def test_satisfaction_trajectory(self, sample_conversation):
        signals = extract_turn_signals(sample_conversation)
        traj = signals["satisfaction_trajectory"]
        assert len(traj) >= 1
        values = [s["value"] for s in traj]
        assert "不满意" in values or "满意" in values

    def test_emotion_curve(self, sample_conversation):
        signals = extract_turn_signals(sample_conversation)
        emotions = signals["emotion_curve"]
        assert len(emotions) >= 2
        assert emotions[1]["intensity"] == 0.8  # Turn 2 用户生气

    def test_attitude_changes(self, sample_conversation):
        signals = extract_turn_signals(sample_conversation)
        changes = signals.get("attitude_changes", [])
        # Turn 4 用户态度改变
        assert len(changes) >= 1

    def test_stuck_turns(self, sample_conversation):
        signals = extract_turn_signals(sample_conversation)
        stuck = signals["stuck_turns"]
        # 本对话无卡死
        assert len(stuck) == 0

    def test_should_end_turns(self, sample_conversation):
        signals = extract_turn_signals(sample_conversation)
        ends = signals["should_end_turns"]
        # Turn 6 用户表达结束意愿
        assert 6 in ends


# ============================================================
# CONSTRAINT 分流测试
# ============================================================

class TestConstraintRouting:
    """测试 CONSTRAINT 分流"""

    def test_classify_constraints(self, sample_case):
        rule_checks, llm_checks = classify_constraints(sample_case.constraints)
        # c3 (word_limit), c4 (forbidden_word) → 规则可检
        assert len(rule_checks) == 2
        # c1 (safety), c2 (tone) → LLM
        assert len(llm_checks) == 2

    def test_check_rule_constraints_no_violation(self, sample_conversation, sample_case):
        rule_cs, _ = classify_constraints(sample_case.constraints)
        issues = check_rule_constraints(sample_conversation, rule_cs)
        # 对话中没有违规
        assert len(issues) == 0


# ============================================================
# 信号上下文格式化测试
# ============================================================

class TestSignalContext:
    """测试信号→清单输入格式化"""

    def test_task_context_includes_satisfaction(self, sample_conversation, sample_case):
        signals = extract_turn_signals(sample_conversation)
        tier1 = compute_tier1_metrics(sample_conversation, sample_case)
        context = format_signal_context(signals, tier1, "TASK_COMPLETION")
        assert "满意" in context

    def test_sentiment_context_includes_emotion(self, sample_conversation, sample_case):
        signals = extract_turn_signals(sample_conversation)
        tier1 = compute_tier1_metrics(sample_conversation, sample_case)
        context = format_signal_context(signals, tier1, "SENTIMENT")
        assert "情绪" in context

    def test_efficiency_context_includes_metrics(self, sample_conversation, sample_case):
        signals = extract_turn_signals(sample_conversation)
        tier1 = compute_tier1_metrics(sample_conversation, sample_case)
        context = format_signal_context(signals, tier1, "EFFICIENCY")
        assert "轮次比" in context or "卡死" in context


# ============================================================
# 评级推导测试
# ============================================================

class TestRatingDerivation:
    """测试加权 YES 占比 → 五级评级"""

    def test_dimension_checklist_properties(self):
        """测试 DimensionChecklist 的属性计算"""
        cl = DimensionChecklist(
            dimension="TEST",
            items=[
                CheckResult(item_id="i1", description="", source="case", status="YES", weight=1.0),
                CheckResult(item_id="i2", description="", source="case", status="YES", weight=1.0),
                CheckResult(item_id="i3", description="", source="case", status="NO", weight=1.0),
                CheckResult(item_id="i4", description="", source="simulator", status="PARTIAL", weight=1.5),
                CheckResult(item_id="i5", description="", source="case", status="NOT_APPLICABLE", weight=1.0),
            ],
        )
        assert cl.yes_count == 2
        assert cl.applicable_count == 4
        assert cl.yes_ratio == 0.5
        # 六级粒度: YES=1.0×1.0 + YES=1.0×1.0 + NO=1.0×0.0 + PARTIAL=1.5×0.5 = 2.75 / 4.5 ≈ 0.611
        assert 0.55 < cl.weighted_yes_ratio < 0.65

    def test_source_ratio(self):
        """测试按来源统计 YES 占比"""
        cl = DimensionChecklist(
            dimension="TEST",
            items=[
                CheckResult(item_id="c1", description="", source="case", status="YES", weight=0.6),
                CheckResult(item_id="c2", description="", source="case", status="NO", weight=0.6),
                CheckResult(item_id="s1", description="", source="simulator", status="YES", weight=1.5),
                CheckResult(item_id="s2", description="", source="simulator", status="YES", weight=1.5),
            ],
        )
        assert cl.source_ratio("case") == 0.5
        assert cl.source_ratio("simulator") == 1.0


# ============================================================
# EvalConfidence 测试
# ============================================================

class TestEvalConfidence:
    """测试 EvalConfidence 计算"""

    def test_high_confidence(self):
        conf = EvalConfidence(
            overall=0.85,
            level="high",
            simulator_tier="green",
            signal_conflict_count=1,
        )
        assert conf.is_reliable
        assert not conf.needs_human_review

    def test_low_confidence(self):
        conf = EvalConfidence(
            overall=0.45,
            level="low",
            simulator_tier="yellow",
            signal_conflict_count=5,
        )
        assert not conf.is_reliable
        assert conf.needs_human_review

    def test_red_tier_unreliable(self):
        conf = EvalConfidence(
            overall=0.70,
            level="medium",
            simulator_tier="red",
            signal_conflict_count=1,
        )
        assert not conf.is_reliable

    def test_cap_min_max(self):
        conf = EvalConfidence(overall=0.05)
        # 直接测试 cap 逻辑在 orchestrator._compute_confidence 中
        assert 0.0 <= conf.overall <= 1.0


# ============================================================
# complexity_score 测试
# ============================================================

class TestComplexityScore:
    """测试复杂度计算"""

    def test_simple_case(self):
        simple = Case(id=99, title="简单", business_line="测试", role="客服", task="问候", opening_line="你好")
        score = compute_complexity_score(simple)
        assert score == 0.0

    def test_complex_case_with_branches(self, sample_case):
        score = compute_complexity_score(sample_case)
        # 2 分支 × 1.5 = 3.0 + 4 constraints × 0.5 = 2.0 + safety +1 = 6.0
        assert score >= 3.0
        assert score <= 10.0

    def test_score_capped_at_10(self):
        """复杂度评分上限 10"""
        # 构造极端复杂的 case
        heavy = Case(
            id=100, title="复杂", business_line="测试", role="", task="",
            opening_line="",
            call_flow=[
                CallFlowStep(id=f"s{i}", step_number=i, title=f"步骤{i}",
                             branching=[Branch(condition="条件", target_step=i + 1)])
                for i in range(10)
            ],
            constraints=[Constraint(id=f"c{i}", type="safety", description=f"约束{i}")
                         for i in range(20)],
            knowledge_points=[KnowledgePoint(id=f"k{i}", topic=f"知识{i}", content=f"内容{i}")
                            for i in range(10)],
        )
        score = compute_complexity_score(heavy)
        assert score <= 10.0


# ============================================================
# AttributionItem 测试
# ============================================================

class TestAttribution:
    """测试归因数据模型"""

    def test_actionable_requires_high_confidence(self):
        attr = AttributionItem(
            source="model", category="TEST",
            description="测试", confidence=0.85,
            suggested_actions=["修复"],
        )
        assert attr.is_actionable

    def test_not_actionable_with_low_confidence(self):
        attr = AttributionItem(
            source="model", category="TEST",
            description="测试", confidence=0.5,
        )
        assert not attr.is_actionable


# ============================================================
# EvalResult 数据模型测试
# ============================================================

class TestEvalResult:
    """测试 EvalResult 结构"""

    def test_empty_result(self):
        result = EvalResult(conversation_id="test", case_id=2)
        assert result.total_indicative_score == 0.0
        assert result.summary == ""

    def test_result_with_ratings(self):
        result = EvalResult(
            conversation_id="test", case_id=2,
            ratings={"SAFETY": "卓越", "TASK_COMPLETION": "良好"},
            indicative_scores={"SAFETY": 9.5, "TASK_COMPLETION": 7.5},
            total_indicative_score=17.0,
            confidence=EvalConfidence(overall=0.75, level="medium"),
        )
        assert result.confidence is not None
        assert result.confidence.is_reliable
