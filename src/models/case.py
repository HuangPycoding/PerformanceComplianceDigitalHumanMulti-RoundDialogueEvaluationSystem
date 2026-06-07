"""指令案例相关数据模型"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Branch:
    """流程分支条件"""
    condition: str           # 触发条件，如 "用户愿意等"
    target_step: int         # 跳转目标步骤编号
    action: str = ""         # 执行动作，如 "协助取消，全额退款"


@dataclass
class CallFlowStep:
    """通话流程中的一个步骤"""
    id: str                           # 如 "step_1"
    step_number: int                  # 步骤序号
    title: str                        # 步骤标题，如 "身份确认"
    description: str = ""             # 该步骤做什么
    branching: List[Branch] = field(default_factory=list)   # 分支条件
    sub_steps: List[str] = field(default_factory=list)      # 子步骤
    reference_script: str = ""        # 参考话术
    is_optional: bool = False         # 是否为可选步骤


@dataclass
class Constraint:
    """一条约束条件"""
    id: str                           # 如 "c1"
    type: str                         # word_limit / forbidden_word / tone / behavior / safety / other
    description: str                  # 约束原文
    checkable_by_rule: bool = False   # 能否用规则自动检查
    rule_pattern: Optional[str] = None  # 正则表达式（可规则检查时）


@dataclass
class KnowledgePoint:
    """一条FAQ知识点"""
    id: str                           # 如 "kp1"
    topic: str                        # 主题，如 "超时赔付标准"
    content: str                      # 标准答案


@dataclass
class Case:
    """一条完整的指令测试案例"""
    id: int
    title: str                        # 简短标题
    business_line: str                # 业务线：外卖/酒店/闪购/医美/打车...
    role: str                         # 角色描述
    task: str                         # 任务描述
    opening_line: str                 # 开场白
    call_flow: List[CallFlowStep] = field(default_factory=list)
    knowledge_points: List[KnowledgePoint] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    complexity_score: float = 0.0     # 复杂度评分 (0-10)
    raw_instruction: str = ""         # 原始指令文本
