"""端到端测试：验证全量数据加载"""
import json
from pathlib import Path

from src.loader.case_loader import load_cases, load_raw_cases
from src.loader.complexity import calculate_complexity
from src.models.case import Case


def test_load_raw_cases():
    """应该读取到所有JSON文件的数据"""
    raw = load_raw_cases()
    assert len(raw) >= 20  # 至少20条
    for entry in raw:
        assert "id" in entry
        assert "instruction" in entry
        assert isinstance(entry["id"], int)


def test_load_cases():
    """解析后应该都是有效Case对象"""
    cases = load_cases()
    assert len(cases) >= 20

    for case in cases:
        assert isinstance(case, Case)
        assert case.id > 0
        assert case.title
        assert case.business_line
        assert case.role
        assert case.task
        # 每个Case至少有一个步骤
        assert len(case.call_flow) >= 1

    # 验证复杂度在合理范围
    for case in cases:
        assert 0 <= case.complexity_score <= 10
        # 复杂度应该已经被计算
        assert case.complexity_score > 0


def test_complexity_distribution():
    """复杂度应该分布在低/中/高三个区间"""
    cases = load_cases()
    low = [c for c in cases if c.complexity_score <= 3]
    mid = [c for c in cases if 3 < c.complexity_score <= 7]
    high = [c for c in cases if c.complexity_score > 7]

    # 应该至少有中复杂度案例
    assert len(mid) >= 1


def test_call_flow_integrity():
    """流程步骤完整性检查"""
    cases = load_cases()
    for case in cases:
        step_numbers = [s.step_number for s in case.call_flow]
        if len(step_numbers) >= 1:
            # 应该从1开始
            assert min(step_numbers) == 1


def test_constraint_types():
    """约束类型应该都是有效值"""
    valid_types = {"word_limit", "forbidden_word", "tone", "behavior", "safety", "other"}
    cases = load_cases()
    for case in cases:
        for c in case.constraints:
            assert c.type in valid_types, f"无效约束类型: {c.type}"


def test_knowledge_points_have_content():
    """知识点都应该有内容"""
    cases = load_cases()
    for case in cases:
        for kp in case.knowledge_points:
            assert kp.topic
            assert kp.content
