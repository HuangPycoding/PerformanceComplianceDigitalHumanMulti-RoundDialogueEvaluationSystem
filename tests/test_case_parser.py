"""解析器单元测试"""
from src.loader.case_parser import (
    parse_instruction, parse_role, parse_task, parse_opening_line,
    parse_call_flow, parse_knowledge_points, parse_constraints,
    detect_business_line, _classify_constraint,
)


# 格式A样例（#5 闪购配送延迟）
INSTRUCTION_A = """# Role
你是美团闪购客服专员，负责配送异常主动外呼。

# Task
致电用户李女士，通知她下单的永辉超市订单因暴雨天气将延迟送达，预计延迟40分钟，并提供补偿方案。

# Opening Line
您好，请问是李女士吗？我是美团闪购客服。关于您刚下的永辉超市订单，非常抱歉需要通知您配送会有延迟。

# Call Flow
1. 说明延迟原因和预计时间：因暴雨导致骑手配送速度受限，预计比原定时间晚40分钟左右。
2. 确认用户是否仍需要此订单：
   - 愿意等 → 感谢理解，告知可实时查看骑手位置，进入第3步
   - 不想等 → 协助取消，全额退款，进入第4步
3. 告知补偿方案：系统已自动发放一张10元闪购优惠券到账户。
4. 如订单含生鲜商品，主动询问是否需要备注优先冷藏配送。
5. 确认无其他问题后结束通话。

# Knowledge Points (FAQ)
- 优惠券自动发放，无需手动领取，可在"我的-优惠券"中查看。
- 延迟超过60分钟可申请"超时赔付"，每单最高赔付15元余额。

# Constraints
- 语气优先共情，先道歉再说事。
- 每次回复控制在30字以内。
- 不主动承诺具体送达时间，用"预计"和"左右"。
- 不与其他配送平台做对比。
"""

# 格式B样例（#2 课程平台）
INSTRUCTION_B = """# Role: Customer Support Specialist for Course Publishing Platform

## Task: 告知机构客户直播选项升级，鼓励选择低延迟直播。

# Constraints:
- 每次回复极简——最多15-20个字
- 不说"好的"、"哈哈"、"嘿嘿"、"嘻嘻"等语气词
- 若老板说忙，说"就1分钟，保证简短"后继续简短说明

# Opening Line: 您好，请问您是贵培训机构/校区的负责人吗？

# Conversation Flow:

## Step 1: 身份确认
- 若是负责人 → 进入第2步
- 若不是 → 请其转达，然后进入第2步

**参考话术：** 我们对直播产品做了升级，新增了独立的"低延迟直播"选项。

## Step 2: 确认是否知情
**询问：** 您之前选的是标准直播，但我们后台其实已为您走低延迟线路以保障质量，您知道吗？

- 若不知情 → 说明前端当时未开放
- 若已知情 → 进入第3步

## Step 3: 传达升级内容
**参考话术：** 之后发布页会分开显示两个选项，根据课程类型自行选择即可。
"""


class TestParseRole:
    def test_format_a(self):
        role = parse_role(INSTRUCTION_A)
        assert "闪购客服" in role

    def test_format_b(self):
        role = parse_role(INSTRUCTION_B)
        assert "Customer Support" in role


class TestParseTask:
    def test_format_a(self):
        task = parse_task(INSTRUCTION_A)
        assert "李女士" in task
        assert "延迟" in task

    def test_format_b(self):
        task = parse_task(INSTRUCTION_B)
        assert "低延迟直播" in task


class TestParseOpeningLine:
    def test_format_a(self):
        opening = parse_opening_line(INSTRUCTION_A)
        assert "您好" in opening
        assert "李女士" in opening

    def test_format_b(self):
        opening = parse_opening_line(INSTRUCTION_B)
        assert "您好" in opening
        assert "培训机构" in opening


class TestParseCallFlow:
    def test_format_a_steps(self):
        steps = parse_call_flow(INSTRUCTION_A)
        assert len(steps) == 5
        assert steps[0].step_number == 1
        assert len(steps[1].branching) == 2  # 第2步有2个分支

    def test_format_b_steps(self):
        steps = parse_call_flow(INSTRUCTION_B)
        assert len(steps) == 3
        assert steps[0].title is not None
        assert len(steps[0].branching) == 2  # 身份确认有2个分支

    def test_format_b_reference_script(self):
        steps = parse_call_flow(INSTRUCTION_B)
        assert any("直播产品" in s.reference_script for s in steps)


class TestParseKnowledgePoints:
    def test_format_a(self):
        kps = parse_knowledge_points(INSTRUCTION_A)
        assert len(kps) == 2
        assert any("优惠券" in kp.topic for kp in kps)
        assert all(kp.content for kp in kps)

    def test_empty(self):
        kps = parse_knowledge_points(INSTRUCTION_B)
        assert len(kps) == 0


class TestParseConstraints:
    def test_format_a(self):
        constraints = parse_constraints(INSTRUCTION_A)
        assert len(constraints) == 4
        assert any(c.type == "word_limit" for c in constraints)
        assert any(c.type == "tone" for c in constraints)

    def test_format_b(self):
        constraints = parse_constraints(INSTRUCTION_B)
        assert len(constraints) == 3
        assert any(c.type == "word_limit" for c in constraints)
        assert any(c.type == "forbidden_word" for c in constraints)


class TestClassifyConstraint:
    def test_word_limit(self):
        t, checkable, _ = _classify_constraint("每次回复控制在30字以内")
        assert t == "word_limit"
        assert checkable

    def test_forbidden_word(self):
        t, checkable, pattern = _classify_constraint("不说\"好的\"、\"哈哈\"")
        assert t == "forbidden_word"
        assert checkable
        assert pattern is not None

    def test_tone(self):
        t, checkable, _ = _classify_constraint("语气温柔、关怀")
        assert t == "tone"
        assert not checkable

    def test_behavior(self):
        t, _, _ = _classify_constraint("不主动承诺具体送达时间")
        assert t == "behavior"


class TestDetectBusinessLine:
    def test_flash_purchase(self):
        biz = detect_business_line(INSTRUCTION_A)
        assert biz == "外卖"

    def test_course_platform(self):
        biz = detect_business_line(INSTRUCTION_B)
        assert biz == "课程平台"


class TestFullParse:
    def test_format_a(self):
        case = parse_instruction(INSTRUCTION_A, 5, "闪购配送延迟通知")
        assert case.id == 5
        assert case.title == "闪购配送延迟通知"
        assert len(case.call_flow) == 5
        assert len(case.constraints) == 4
        assert len(case.knowledge_points) == 2
        assert case.opening_line

    def test_format_b(self):
        case = parse_instruction(INSTRUCTION_B, 2, "课程平台直播选项升级")
        assert case.id == 2
        assert len(case.call_flow) == 3
        assert len(case.constraints) == 3
