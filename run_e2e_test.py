"""端到端测试: case 2 × 5 画像 → 对话生成 → 评测 → MD 报告"""
import json, os, sys, time, random
from datetime import datetime
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(__file__))

from src.llm.client import LLMClient
from src.llm.model_manager import get_generator_client, get_auditor_client
from src.loader.case_loader import load_cases
from src.loader.complexity import compute_max_turns
from src.models.case import Case
from src.models.conversation import Conversation
from src.simulator.profiles import build_profile_from_vector
from src.simulator.profile_params import (
    lhs_sample, subspace_lhs, deduplicate_vectors,
    compute_profile_count, extract_branch_constraints,
)
from src.simulator.profile_generator import ProfileGenerator, compute_self_check_thresholds
from src.simulator.profile_auditor import ProfileAuditor
from src.simulator.runner import DialogueRunner
from src.simulator.assistant_interface import LLMAssistant
from src.eval.orchestrator import EvalOrchestrator
from src.eval.drift_monitor import BatchAnalyzer

os.makedirs("data/e2e_test", exist_ok=True)

# ============================================================
# Config
# ============================================================
CASE_ID = 2
N_GLOBAL = 3   # 全空间基线画像
N_BRANCH = 2   # 分支画像
TOTAL_PROFILES = N_GLOBAL + N_BRANCH  # 5 profiles
BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"data/e2e_test/{BATCH_ID}"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"=" * 60)
print(f"E2E Test: Case {CASE_ID} x {TOTAL_PROFILES} profiles")
print(f"Batch: {BATCH_ID}")
print(f"=" * 60)

# ============================================================
# Step 1: Load Case
# ============================================================
print("\n[Step 1] Loading case...")
cases = load_cases()
case = next(c for c in cases if c.id == CASE_ID)
print(f"  Case #{case.id}: {case.title}")
print(f"  Role: {case.role}")
print(f"  Call flow steps: {len(case.call_flow)}")
print(f"  Constraints: {len(case.constraints)}")
print(f"  Complexity: {case.complexity_score}")

# ============================================================
# Step 2: Generate Profiles (Phase 0)
# ============================================================
print("\n[Step 2] Generating profiles...")
gen_client = get_generator_client(temperature=0.7)

# Compute self-check thresholds based on complexity
md_limit, dsv_limit, retries = compute_self_check_thresholds(case.complexity_score)
gen = ProfileGenerator(gen_client, max_dev_limit=md_limit, d_sv_limit=dsv_limit, max_retries=retries)

# P1: Global LHS
vectors = lhs_sample(N_GLOBAL, 15)

# P2: Branch-constrained LHS
branch_constraints = extract_branch_constraints(case.call_flow)
if branch_constraints:
    free_dims = [d for d in range(15) if d not in branch_constraints]
    constrained = subspace_lhs(N_BRANCH, branch_constraints)
    constrained = deduplicate_vectors(constrained, min_distance=0.3, free_dim_indices=free_dims)
    vectors.extend(constrained)
    # Add 2 extreme profiles
    for pole in ("low", "high"):
        v_ext = [random.random() for _ in range(15)]
        for dim_idx, (lo, hi) in branch_constraints.items():
            if pole == "low":
                v_ext[dim_idx] = lo if lo > 0.1 else max(0.0, lo + 0.02)
            else:
                v_ext[dim_idx] = hi if hi < 0.9 else min(1.0, hi - 0.02)
        vectors.append(v_ext)
else:
    vectors.extend(lhs_sample(max(N_BRANCH, 2), 15))

print(f"  Generated {len(vectors)} vectors, generating persona texts...")
profiles = gen.batch_generate(vectors, verbose=True)
print(f"  {len(profiles)} profiles generated with self-check")

# ============================================================
# Step 3: Run Conversations (Phase 1)
# ============================================================
print("\n[Step 3] Running conversations...")
sim_client = LLMClient(temperature=0.7)
assistant_client = LLMClient(temperature=0.3)
auditor_client = get_auditor_client(temperature=0.3)
auditor = ProfileAuditor(auditor_client)
max_turns = compute_max_turns(case)

conversations: List[Conversation] = []
for i, profile in enumerate(profiles):
    print(f"  [{i+1}/{len(profiles)}] {profile.label} ... ", end="", flush=True)
    try:
        assistant = LLMAssistant(case, assistant_client, use_raw_prompt=False)
        runner = DialogueRunner(case, profile, assistant, sim_client)
        conv = runner.run(max_turns=max_turns)
        # Run Path A audit from state trajectory
        state_traj = [t.parsed_tags.get("state", {}) for t in conv.turns
                      if t.parsed_tags and t.parsed_tags.get("state")]
        if conv.sampled_vector is None:
            conv.sampled_vector = profile.vector
        auditor.audit_path_a(conv, state_traj)
        conversations.append(conv)
        print(f"{conv.status} ({conv.total_turns}t)")
    except Exception as e:
        print(f"ERROR: {e}")
        conv = Conversation(id=f"case{CASE_ID}_profile{i}", case_id=CASE_ID)
        conv.status = "异常中断"
        conversations.append(conv)

print(f"  {len(conversations)} conversations completed")

# ============================================================
# Step 4: Run Evaluation (Phase 3)
# ============================================================
print("\n[Step 4] Running evaluation...")
eval_client = LLMClient(temperature=0.3)
orch = EvalOrchestrator(eval_client)

eval_results = []
for i, conv in enumerate(conversations):
    if conv.total_turns == 0:
        eval_results.append(None)
        continue
    print(f"  [{i+1}/{len(conversations)}] Evaluating conv {conv.id} ({conv.total_turns}t) ... ",
          end="", flush=True)
    try:
        result = orch.run(conv, case)
        eval_results.append(result)
        c = result.confidence
        print(f"Score={result.total_indicative_score:.0f} Conf={c.level}({c.overall:.2f})")
    except Exception as e:
        print(f"ERROR: {e}")
        eval_results.append(None)

# Run batch analysis
analyzer = BatchAnalyzer()
reliable_convs = [c for c, r in zip(conversations, eval_results) if r is not None]
reliable_results = [r for r in eval_results if r is not None]
# Convert EvalResult to dict for analyzer
results_for_analyzer = []
for r in reliable_results:
    results_for_analyzer.append({
        "ratings": r.ratings,
        "total_score": r.total_indicative_score,
        "is_reliable": r.confidence.is_reliable if r.confidence else False,
        "confidence_level": r.confidence.level if r.confidence else "unknown",
    })
batch_report = analyzer.analyze(results_for_analyzer) if results_for_analyzer else {}

# ============================================================
# Step 5: Generate MD Report
# ============================================================
print("\n[Step 5] Generating report...")

lines = []
lines.append(f"# E2E 评测报告 — Case {CASE_ID}")
lines.append(f"")
lines.append(f"**批次**: {BATCH_ID}")
lines.append(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append(f"**画像数**: {len(profiles)} | **对话数**: {len(conversations)} | **完成评测**: {len(reliable_results)}")
lines.append(f"")

# ---- Case Info ----
lines.append(f"## 一、Case 信息")
lines.append(f"")
lines.append(f"| 属性 | 值 |")
lines.append(f"|------|-----|")
lines.append(f"| ID | {case.id} |")
lines.append(f"| 标题 | {case.title} |")
lines.append(f"| 业务线 | {case.business_line} |")
lines.append(f"| 复杂度 | {case.complexity_score} |")
lines.append(f"| 流程步骤数 | {len(case.call_flow)} |")
lines.append(f"| 约束数 | {len(case.constraints)} |")
lines.append(f"| 知识点数 | {len(case.knowledge_points)} |")

if case.call_flow:
    lines.append(f"")
    lines.append(f"### 流程步骤")
    for step in case.call_flow:
        branches = getattr(step, 'branching', [])
        branch_info = f" ({len(branches)}分支)" if branches else ""
        lines.append(f"- **{step.title}**{branch_info}: {getattr(step, 'description', '')[:100]}")

lines.append(f"")

# ---- Profile Info ----
lines.append(f"## 二、用户画像")
lines.append(f"")
for i, p in enumerate(profiles):
    lines.append(f"### P{i+1}: {p.label}")
    lines.append(f"")
    lines.append(f"```")
    lines.append(p.persona_text[:500])
    if len(p.persona_text) > 500:
        lines.append("...(truncated)")
    lines.append(f"```")
    lines.append(f"")

# ---- Conversation Details ----
lines.append(f"## 三、对话详情")
lines.append(f"")

for i, (conv, result) in enumerate(zip(conversations, eval_results)):
    profile = profiles[i] if i < len(profiles) else None
    p_label = profile.label if profile else f"P{i+1}"

    lines.append(f"### 对话 {i+1}: {p_label}")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 状态 | {conv.status} |")
    lines.append(f"| 轮次 | {conv.total_turns} |")
    _dsa = conv.consistency.get('path_a_d_sa', 'N/A') if conv.consistency else 'N/A'
    if isinstance(_dsa, (int, float)):
        _dsa = f"{_dsa:.3f}"
    lines.append(f"| 一致性 d_sa | {_dsa} |")

    if result:
        lines.append(f"| 评测总分 | {result.total_indicative_score:.1f} |")
        if result.confidence:
            lines.append(f"| 可信度 | {result.confidence.level} ({result.confidence.overall:.2f}) |")
            lines.append(f"| 需人工复核 | {result.confidence.needs_human_review} |")

    lines.append(f"")
    lines.append(f"#### 对话文本")
    lines.append(f"")
    lines.append(f"```")
    for turn in conv.turns:
        speaker = "客服" if turn.speaker == "system" else "用户"
        lines.append(f"T{turn.turn_number} [{speaker}]: {turn.content}")
    lines.append(f"```")
    lines.append(f"")

    if result:
        lines.append(f"#### 评测结果")
        lines.append(f"")
        lines.append(f"| 维度 | 评级 | 得分 |")
        lines.append(f"|------|------|------|")
        for dim in ['SAFETY','TASK_COMPLETION','FLOW_COVERAGE','KNOWLEDGE','CONSTRAINT',
                     'EFFICIENCY','SENTIMENT','ROLE','OPENING']:
            r = result.ratings.get(dim, 'N/A')
            s = result.indicative_scores.get(dim, 0)
            lines.append(f"| {dim} | {r} | {s:.1f} |")
        lines.append(f"| **总分** | | **{result.total_indicative_score:.1f}** |")
        lines.append(f"")

        if result.cross_validation_alerts:
            lines.append(f"**交叉验证告警**:")
            for a in result.cross_validation_alerts:
                lines.append(f"- [{a.severity}] {a.dimension}: {a.description}")
            lines.append(f"")

        if result.meta_check_alerts:
            lines.append(f"**元检查告警** ({len(result.meta_check_alerts)}条):")
            for a in result.meta_check_alerts[:10]:
                lines.append(f"- [{a.severity}][{a.check_type}] {a.description}")
            if len(result.meta_check_alerts) > 10:
                lines.append(f"- ... 还有 {len(result.meta_check_alerts) - 10} 条")
            lines.append(f"")

        if result.surface_compliance_flags:
            lines.append(f"**表面合规标记**: {result.surface_compliance_flags}")
            lines.append(f"")

    lines.append(f"---")
    lines.append(f"")

# ---- Aggregate Analysis ----
lines.append(f"## 四、聚合分析")
lines.append(f"")

if reliable_results:
    # Rating distribution
    all_ratings: Dict[str, List[str]] = {}
    for r in reliable_results:
        for dim, rating in r.ratings.items():
            all_ratings.setdefault(dim, []).append(rating)

    lines.append(f"### 评级分布")
    lines.append(f"")
    lines.append(f"| 维度 | 卓越 | 良好 | 合格 | 需改进 | 不合格 |")
    lines.append(f"|------|------|------|------|--------|--------|")
    for dim in ['SAFETY','TASK_COMPLETION','FLOW_COVERAGE','KNOWLEDGE','CONSTRAINT',
                 'EFFICIENCY','SENTIMENT','ROLE','OPENING']:
        ratings = all_ratings.get(dim, [])
        lines.append(f"| {dim} | {ratings.count('卓越')} | {ratings.count('良好')} | "
                     f"{ratings.count('合格')} | {ratings.count('需改进')} | {ratings.count('不合格')} |")
    lines.append(f"")

    # Score summary
    scores = [r.total_indicative_score for r in reliable_results]
    lines.append(f"### 分数汇总")
    lines.append(f"")
    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 均值 | {sum(scores)/len(scores):.1f} |")
    lines.append(f"| 最高 | {max(scores):.1f} |")
    lines.append(f"| 最低 | {min(scores):.1f} |")
    lines.append(f"| 中位数 | {sorted(scores)[len(scores)//2]:.1f} |")
    lines.append(f"")

    # Confidence summary
    conf_levels = [r.confidence.level for r in reliable_results if r.confidence]
    lines.append(f"### 置信度分布")
    lines.append(f"")
    lines.append(f"| Level | 数量 |")
    lines.append(f"|-------|------|")
    for level in ['high', 'medium', 'low', 'unreliable']:
        lines.append(f"| {level} | {conf_levels.count(level)} |")
    lines.append(f"")

    lines.append(f"| 指标 | 值 |")
    lines.append(f"|------|-----|")
    lines.append(f"| 需人工复核 | {sum(1 for r in reliable_results if r.confidence and r.confidence.needs_human_review)} |")
    lines.append(f"| CONSTRAINT 分流 (tier1) | {sum(r.tier1_constraint_count for r in reliable_results)} |")
    lines.append(f"| CONSTRAINT 分流 (llm)  | {sum(r.llm_constraint_count for r in reliable_results)} |")
    lines.append(f"")

    # Batch report summary
    if batch_report:
        lines.append(f"### 批次分析")
        lines.append(f"")
        for k, v in batch_report.items():
            if isinstance(v, dict):
                lines.append(f"**{k}**:")
                for k2, v2 in v.items():
                    lines.append(f"- {k2}: {v2}")
            elif isinstance(v, list):
                lines.append(f"**{k}** ({len(v)} items):")
                for item in v[:5]:
                    lines.append(f"- {item}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append(f"")

# ---- Known Issues ----
lines.append(f"## 五、说明")
lines.append(f"")
lines.append(f"- 本次测试对话的 parsed_tags 可能为空（对话引擎输出解析状态取决于运行环境），导致 Simulator 信号相关检查输出较少")
lines.append(f"- evidence 模糊匹配告警较多说明 LLM 的 evidence 引用格式与实际对话文本存在差异")
lines.append(f"- Confidence 为 unreliable 时说明评测系统自身认为该条评测结果不应用于驱动优化决策")

# Write report
report_path = os.path.join(OUT_DIR, "e2e_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))

# Also save raw data
raw_path = os.path.join(OUT_DIR, "raw_data.json")
raw_data = {
    "batch_id": BATCH_ID,
    "case_id": CASE_ID,
    "case_title": case.title,
    "n_profiles": len(profiles),
    "conversations": [],
}
for i, (conv, result) in enumerate(zip(conversations, eval_results)):
    conv_data = {
        "id": conv.id,
        "status": conv.status,
        "total_turns": conv.total_turns,
        "consistency": conv.consistency,
        "turns": [{"turn": t.turn_number, "speaker": t.speaker, "content": t.content}
                   for t in conv.turns],
    }
    if result:
        conv_data["ratings"] = result.ratings
        conv_data["indicative_scores"] = result.indicative_scores
        conv_data["total_score"] = result.total_indicative_score
        conv_data["confidence"] = {
            "overall": result.confidence.overall,
            "level": result.confidence.level,
            "needs_human_review": result.confidence.needs_human_review,
        } if result.confidence else None
        conv_data["meta_alerts_count"] = len(result.meta_check_alerts)
        conv_data["cv_alerts_count"] = len(result.cross_validation_alerts)
    raw_data["conversations"].append(conv_data)

with open(raw_path, 'w', encoding='utf-8') as f:
    json.dump(raw_data, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 60}")
print(f"E2E Test Complete!")
print(f"  Report: {report_path}")
print(f"  Raw data: {raw_path}")
print(f"  Conversations: {len(conversations)}")
print(f"  Evaluated: {len(reliable_results)}")
if reliable_results:
    scores = [r.total_indicative_score for r in reliable_results]
    print(f"  Avg score: {sum(scores)/len(scores):.1f}")
print(f"{'=' * 60}")
