"""批量加载入口：读取4个JSON → 解析 → 合并 → 输出"""
import json
import os

from src.config import CASES_PARSED_PATH, PROJECT_ROOT
from src.loader.case_parser import parse_instruction
from src.eval.rules import compute_complexity_score as calculate_complexity
from src.models.case import Case


def load_raw_cases() -> list[dict]:
    """读取合并后的JSON文件，返回原始数据列表"""
    path = PROJECT_ROOT / "generated_cases_all.json"
    with open(path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)
    print(f"  已读取 generated_cases_all.json: {len(all_cases)} 条")
    return all_cases


def load_cases() -> list[Case]:
    """读取并解析全部案例"""
    raw = load_raw_cases()
    cases: list[Case] = []
    for entry in raw:
        try:
            cid = entry["id"]
            title = entry.get("title", "")
            instruction = entry["instruction"]
            case = parse_instruction(instruction, cid, title)
            case.complexity_score = calculate_complexity(case)
            cases.append(case)
        except (KeyError, TypeError, ValueError) as e:
            print(f"[跳过] 条目解析失败: {e}")
    return cases


def export_cases(cases: list[Case], path: str | None = None):
    """将 Case 列表导出为 JSON"""
    if path is None:
        path = str(CASES_PARSED_PATH)
    data = []
    for c in cases:
        data.append({
            "id": c.id,
            "title": c.title,
            "business_line": c.business_line,
            "role": c.role,
            "task": c.task,
            "opening_line": c.opening_line,
            "complexity_score": c.complexity_score,
            "call_flow": [
                {
                    "id": s.id,
                    "step_number": s.step_number,
                    "title": s.title,
                    "description": s.description,
                    "branching": [{"condition": b.condition, "target_step": b.target_step, "action": b.action} for b in s.branching],
                    "sub_steps": s.sub_steps,
                    "reference_script": s.reference_script,
                    "is_optional": s.is_optional,
                }
                for s in c.call_flow
            ],
            "knowledge_points": [
                {"id": kp.id, "topic": kp.topic, "content": kp.content}
                for kp in c.knowledge_points
            ],
            "constraints": [
                {"id": ct.id, "type": ct.type, "description": ct.description, "checkable_by_rule": ct.checkable_by_rule, "rule_pattern": ct.rule_pattern}
                for ct in c.constraints
            ],
        })
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_summary(cases: list[Case]):
    """打印解析汇总"""
    print()
    print("=" * 60)
    print(f"解析完成: {len(cases)} 条成功")

    low = [c for c in cases if c.complexity_score <= 3]
    mid = [c for c in cases if 3 < c.complexity_score <= 7]
    high = [c for c in cases if c.complexity_score > 7]
    print(f"复杂度分布:")
    print(f"  低 (0-3):   {len(low)} 条")
    print(f"  中 (3-7):   {len(mid)} 条")
    print(f"  高 (7-10):  {len(high)} 条")
    print(f"输出: {CASES_PARSED_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    cases = load_cases()
    export_cases(cases)
    print_summary(cases)
