"""Phase 3 评测引擎 — 信号增强清单评估

核心入口: EvalOrchestrator — 编排一条对话的完整评测流程

数据流:
  1. Tier 1 规则层 (rules.py) — 11 指标 + CONSTRAINT 分流
  2. Tier 1.5 信号提取 (rules.py) — 7 Turn 级信号
  3. 清单生成 (checklist_generator.py) — 三层清单
  4. LLM 核查 (judge.py) — 9 Judge 并发
  5. 评级推导 (orchestrator.py) — 加权 YES 占比
  6. 归因 (diagnostics.py) — Case/Simulator/Model
  7. EvalConfidence (orchestrator.py) — 五类输入
  8. 批次聚合 (drift_monitor.py) — BatchAnalyzer
  9. 清单进化 (checklist_evolver.py) — v1 积累 / 3.1+ 转化
"""
from src.eval.checklist_evolver import ChecklistEvolver
from src.eval.drift_monitor import BatchAnalyzer, DriftMonitor
from src.eval.orchestrator import EvalOrchestrator
from src.eval.rules import (
    check_rule_constraints,
    classify_constraints,
    compute_complexity_score,
    compute_tier1_metrics,
    extract_turn_signals,
    format_signal_context,
)
from src.eval.schemas import build_judge_system_prompt

__all__ = [
    "EvalOrchestrator",
    "BatchAnalyzer",
    "DriftMonitor",
    "ChecklistEvolver",
    # Rules
    "compute_tier1_metrics",
    "extract_turn_signals",
    "format_signal_context",
    "check_rule_constraints",
    "classify_constraints",
    "compute_complexity_score",
    # Schemas
    "build_judge_system_prompt",
]
