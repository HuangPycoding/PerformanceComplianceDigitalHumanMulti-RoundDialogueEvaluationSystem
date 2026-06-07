"""指令复杂度量化"""
from src.models.case import Case, CallFlowStep
from src.eval.rules import compute_complexity_score  # 统一使用评测引擎的复杂度函数

# 约束类型权重: safety/behavior 比 tone/word_limit 更难满足
CONSTRAINT_TYPE_WEIGHT = {
    "safety": 1.5, "behavior": 1.3, "other": 1.0,
    "tone": 0.7, "word_limit": 0.5, "format": 0.5,
}


def count_total_branches(steps: list[CallFlowStep]) -> int:
    """统计所有步骤中的条件分支总数"""
    return sum(len(s.branching) for s in steps)


def has_nested_branching(steps: list[CallFlowStep]) -> bool:
    """检测是否存在嵌套分支（分支内部还有分支）"""
    for step in steps:
        for branch in step.branching:
            # 如果这条分支引用的目标步骤本身也有分支，视为嵌套
            for other in steps:
                if other.step_number == branch.target_step and other.branching:
                    return True
    return False


def calculate_complexity(case: Case) -> float:
    """量化指令复杂度 (0-10)

    六因子加权:
      流程分支数     0~3分
      约束数量       0~2分
      约束类型多样性 0~1.5分
      知识点数量     0~1分
      流程步骤数     0~1.5分
      嵌套分支       0~1分
    """
    score = 0.0

    # 1. 分支数 (0~3分) — 分母 /4 对齐实际分布中位数 ~3
    branch_count = count_total_branches(case.call_flow)
    score += min(branch_count / 4, 1.0) * 3

    # 2. 约束数量 (0~2分) — 分母 /6 对齐实际 5-6 个
    score += min(len(case.constraints) / 6, 1.0) * 2

    # 3. 约束类型多样性 (0~1.5分) — 加权求和，safety/behavior 权重更高
    weight_sum = 0.0
    for c in case.constraints:
        weight_sum += CONSTRAINT_TYPE_WEIGHT.get(c.type, 1.0)
    score += min(weight_sum / 5, 1.0) * 1.5

    # 4. 知识点数量 (0~1分) — 分母 /5 对齐实际
    score += min(len(case.knowledge_points) / 5, 1.0) * 1.0

    # 5. 流程步骤数 (0~1.5分)
    score += min(len(case.call_flow) / 7, 1.0) * 1.5

    # 6. 嵌套分支 (0~1分)
    if has_nested_branching(case.call_flow):
        score += 1.0

    return round(min(score, 10.0), 1)


def compute_max_turns(case: Case) -> int:
    """根据 case 复杂度动态计算建议最大对话轮次

    考虑因素:
      - 流程步骤数 (每步基础2轮)
      - 分支数 (每个分支额外1轮决策)
      - 字数限制约束 (回复短→需要更多轮次)
      - 综合复杂度评分

    返回 8-40 之间的整数值。
    """
    steps = len(case.call_flow) if case.call_flow else 1
    branches = count_total_branches(case.call_flow)
    complexity = compute_complexity_score(case)

    # 约束类型影响因子：safety/behavior 增加轮次预算，word_limit 大幅增加
    constraint_factor = 1.0
    has_word_limit = False
    for c in getattr(case, 'constraints', []):
        ct = getattr(c, 'type', '')
        if ct == 'word_limit':
            constraint_factor += 0.3  # 短回复→需要更多轮次（从 0.5 调低避免过度膨胀）
            has_word_limit = True
        elif ct in ('safety', 'behavior'):
            constraint_factor += 0.15
        elif ct == 'forbidden_word':
            constraint_factor += 0.15  # 禁用词可能迫使重述（从 0.1 提升）
    constraint_factor = min(constraint_factor, 2.0)  # 上限防过度膨胀

    base = steps * 2 + 2          # 每步2轮 + 开头/结尾
    branch_extra = branches        # 分支决策额外轮次
    complexity_extra = max(0, (complexity - 5) * 1.5)

    turns = int(base * constraint_factor + branch_extra + complexity_extra)
    # 动态上限：复杂度越高，允许的轮次上限越高
    if complexity >= 9.5:
        cap = 40
    elif complexity >= 8:
        cap = 35
    elif complexity >= 6:
        cap = 30
    else:
        cap = 25
    # 约束感知下限：防止简单 case 因约束多而轮次不够
    lower = 14 if has_word_limit else 10
    return max(lower, min(turns, cap))
