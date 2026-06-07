"""Case 1 完整链路 — 画像生成 → 对话 → 评测 → MD 报告"""
import json, os, sys, time
from datetime import datetime
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from src.loader.case_loader import load_cases
from src.llm.client import LLMClient
from src.llm.model_manager import get_generator_client
from src.simulator.batch_runner import BatchRunner
from src.eval.drift_monitor import BatchAnalyzer

BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"data/full_test/{BATCH_ID}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Batch: {BATCH_ID}")

cases = load_cases()
case1 = next(c for c in cases if c.id == 1)
print(f"Case 1: {case1.title}  Complexity={case1.complexity_score}")

sim_client = LLMClient(temperature=0.7)
asm_client = LLMClient(temperature=0.3)
eval_client = LLMClient(temperature=0.3)
runner = BatchRunner(cases=[case1], assistant_client=asm_client, simulator_client=sim_client,
                     eval_client=eval_client, use_raw_prompt=False)

# Phase 0: Profiles
print("Generating profiles...")
gen_client = get_generator_client(temperature=0.7)
case_profiles = runner.generate_profiles(n_global=5, gen_client=gen_client, verbose=True)
total = sum(len(ps) for ps in case_profiles.values())
print(f"Generated {total} profiles")

# Phase 1+2: Conversations
print("Running conversations...")
t0 = time.time()
convs = runner.run_all(parallel=False, max_turns=30, profiles_dict=case_profiles, run_eval=False)
t1 = time.time()

# Phase 3: Evaluation
print("Running evaluation...")
results = runner.run_phase3(convs, eval_client=eval_client)
t2 = time.time()

# Batch analysis
analyzer = BatchAnalyzer()
ra = [{"ratings": r.ratings, "total_score": r.total_indicative_score,
       "is_reliable": r.confidence.is_reliable if r.confidence else False,
       "confidence_level": r.confidence.level if r.confidence else "unknown"}
      for r in results if r is not None]
batch = analyzer.analyze(ra) if ra else {}

# Build report
L = []
L.append(f"# Case 1 全链路评测报告")
L.append(f"")
L.append(f"**批次**: {BATCH_ID} | **画像数**: {total} | **对话数**: {len(convs)} | **评测数**: {len(results)}")
L.append(f"**对话耗时**: {t1-t0:.0f}s | **评测耗时**: {t2-t1:.0f}s")
L.append(f"")

# Case info
L.append(f"## 一、Case 指令")
L.append(f"")
L.append(f"| 属性 | 值 |")
L.append(f"|------|-----|")
L.append(f"| ID | {case1.id} |")
L.append(f"| 标题 | {case1.title} |")
L.append(f"| 业务线 | {case1.business_line} |")
L.append(f"| 复杂度 | {case1.complexity_score} |")
L.append(f"| 角色 | {case1.role} |")
L.append(f"| 任务 | {case1.task} |")
L.append(f"| 开场白 | {case1.opening_line[:150]} |")
L.append(f"")
L.append(f"### 流程步骤 ({len(case1.call_flow)} 步)")
for s in case1.call_flow:
    L.append(f"- **{s.title}**{': '+s.description[:120] if getattr(s,'description','') else ''}")
L.append(f"")
L.append(f"### 知识点 ({len(case1.knowledge_points)} 条)")
for k in case1.knowledge_points:
    L.append(f"- **{k.topic}**: {k.content}")
L.append(f"")
L.append(f"### 约束条件 ({len(case1.constraints)} 条)")
for c in case1.constraints:
    L.append(f"- [{c.type}] {c.description}")
L.append(f"")

# Profiles
L.append(f"## 二、用户画像 ({total} 个)")
for cid, profiles in case_profiles.items():
    for i, p in enumerate(profiles):
        adv = ','.join(getattr(p,'adversarial_strategy',[]) or []) or '无'
        L.append(f"### P{i+1}: {p.label} (对抗: {adv})")
        L.append(f"```")
        L.append(p.persona_text[:500])
        L.append(f"```")
        L.append(f"")

# Conversations + Evaluations
L.append(f"## 三、对话详情与评测结果")
all_ratings = {}
all_scores = []
conf_levels = []

for i, (conv, result) in enumerate(zip(convs, results)):
    profiles_list = case_profiles.get(conv.case_id, [])
    p_label = profiles_list[i].label if i < len(profiles_list) else f"P{i+1}"

    L.append(f"### 对话 {i+1}: {p_label}")
    L.append(f"| 状态 | 轮次 | 总分 | 可信度 |")
    L.append(f"|------|------|------|--------|")
    sc = result.total_indicative_score if result else 0
    cl = f"{result.confidence.level}({result.confidence.overall:.2f})" if (result and result.confidence) else "N/A"
    L.append(f"| {conv.status} | {conv.total_turns} | {sc:.1f} | {cl} |")
    L.append(f"")

    L.append(f"#### 对话文本")
    L.append(f"```")
    for turn in conv.turns:
        sp = "[客服]" if turn.speaker == "system" else "[用户]"
        L.append(f"T{turn.turn_number} {sp}: {turn.content}")
    L.append(f"```")
    L.append(f"")

    if result:
        all_scores.append(result.total_indicative_score)
        if result.confidence:
            conf_levels.append(result.confidence.level)
        L.append(f"#### 评测结果")
        L.append(f"| 维度 | 评级 | 得分 |")
        L.append(f"|------|------|------|")
        for dim in ['SAFETY','TASK_COMPLETION','FLOW_COVERAGE','KNOWLEDGE','CONSTRAINT',
                     'EFFICIENCY','SENTIMENT','ROLE','OPENING']:
            r = result.ratings.get(dim, 'N/A')
            s = result.indicative_scores.get(dim, 0)
            L.append(f"| {dim} | {r} | {s:.1f} |")
            all_ratings.setdefault(dim, []).append(r)
        L.append(f"| **总分** | | **{result.total_indicative_score:.1f}** |")
        L.append(f"")

        if result.cross_validation_alerts:
            L.append(f"**交叉验证告警**:")
            for a in result.cross_validation_alerts:
                L.append(f"- [{a.severity}] {a.dimension}: {a.description[:150]}")
            L.append(f"")
        if result.meta_check_alerts:
            L.append(f"**元检查告警** ({len(result.meta_check_alerts)}条, 前10):")
            for a in result.meta_check_alerts[:10]:
                L.append(f"- [{a.severity}][{a.check_type}] {a.description[:150]}")
            L.append(f"")
    L.append(f"---")
    L.append(f"")

# Aggregate
L.append(f"## 四、聚合分析")
L.append(f"")
L.append(f"### 评级分布")
L.append(f"| 维度 | 卓越 | 良好 | 合格 | 需改进 | 不合格 |")
L.append(f"|------|------|------|------|--------|--------|")
for dim in ['SAFETY','TASK_COMPLETION','FLOW_COVERAGE','KNOWLEDGE','CONSTRAINT',
             'EFFICIENCY','SENTIMENT','ROLE','OPENING']:
    rs = all_ratings.get(dim, [])
    L.append(f"| {dim} | {rs.count('卓越')} | {rs.count('良好')} | {rs.count('合格')} | {rs.count('需改进')} | {rs.count('不合格')} |")
L.append(f"")

if all_scores:
    L.append(f"### 分数汇总")
    avg = sum(all_scores)/len(all_scores)
    L.append(f"| 均值 | 最高 | 最低 | 中位数 |")
    L.append(f"|------|------|------|--------|")
    L.append(f"| {avg:.1f} | {max(all_scores):.1f} | {min(all_scores):.1f} | {sorted(all_scores)[len(all_scores)//2]:.1f} |")
    L.append(f"")

if conf_levels:
    L.append(f"### 置信度分布")
    for lv in ['high','medium','low','unreliable']:
        L.append(f"- {lv}: {conf_levels.count(lv)}")
    L.append(f"")

# Analysis
L.append(f"## 五、效果分析")
L.append(f"")
L.append(f"### Case 指令覆盖")
L.append(f"- 流程步骤 {len(case1.call_flow)} 步 → FLOW_COVERAGE 维度覆盖")
L.append(f"- 知识点 {len(case1.knowledge_points)} 条 → KNOWLEDGE 维度逐条核查")
L.append(f"- 约束 {len(case1.constraints)} 条 → CONSTRAINT 维度覆盖（含 Tier1 规则分流）")
L.append(f"- 开场白 → OPENING 维度核查")
L.append(f"- 角色/任务 → ROLE / TASK_COMPLETION 维度覆盖")
L.append(f"")

L.append(f"### 用户模拟器效果")
L.append(f"- 画像多样性: {total} 个画像，覆盖 50% 对抗策略")
L.append(f"- 对话正常率: {sum(1 for c in convs if c.status != '异常中断')}/{len(convs)}")
L.append(f"- 平均轮次: {sum(c.total_turns for c in convs)/max(len(convs),1):.1f}")
L.append(f"")

L.append(f"### 评测可量化性")
if all_scores:
    L.append(f"- 分数范围: {min(all_scores):.0f} - {max(all_scores):.0f} (区分度 {max(all_scores)-min(all_scores):.0f})")
L.append(f"- 9 维度 × 5 级 = 45 级量化粒度")
if batch:
    L.append(f"- unreliable 占比: {batch.get('unreliable_count',0)}/{batch.get('n_results',1)}")
L.append(f"")

# Write
report_path = os.path.join(OUT_DIR, "full_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

# Raw data
raw_path = os.path.join(OUT_DIR, "raw_data.json")
rd = {"batch_id": BATCH_ID, "case_id": 1, "total_profiles": total,
      "conversations": []}
for conv, result in zip(convs, results):
    cd = {"id": conv.id, "status": conv.status, "total_turns": conv.total_turns,
          "turns": [{"turn": t.turn_number, "speaker": t.speaker, "content": t.content}
                     for t in conv.turns]}
    if result:
        cd["ratings"] = result.ratings
        cd["total_score"] = result.total_indicative_score
        cd["confidence"] = {"overall": result.confidence.overall, "level": result.confidence.level} if result.confidence else None
    rd["conversations"].append(cd)
with open(raw_path, 'w', encoding='utf-8') as f:
    json.dump(rd, f, ensure_ascii=False, indent=2)

print(f"\nDone! Report: {report_path}")
print(f"Conv: {len(convs)} Eval: {len(results)}")
if all_scores:
    print(f"Score: {min(all_scores):.0f}-{max(all_scores):.0f} (avg {sum(all_scores)/len(all_scores):.1f})")
