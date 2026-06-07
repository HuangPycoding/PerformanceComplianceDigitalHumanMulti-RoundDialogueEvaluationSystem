"""Case 2 全链路测试 — batch_runner 画像→对话→评测→MD报告"""
import json, os, sys, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from src.loader.case_loader import load_cases
from src.llm.client import LLMClient
from src.llm.model_manager import get_generator_client
from src.simulator.batch_runner import BatchRunner
from src.eval.drift_monitor import BatchAnalyzer
from src.eval.checklist_generator import _is_notification_call

BATCH_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_DIR = f"data/full_test/{BATCH_ID}"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Batch: {BATCH_ID}")

cases = load_cases()
case2 = next(c for c in cases if c.id == 2)
print(f"Case 2: {case2.title}  Complexity={case2.complexity_score}")
print(f"Steps: {len(case2.call_flow)} KPs: {len(case2.knowledge_points)} Constraints: {len(case2.constraints)}")
print(f"Notification call: {_is_notification_call(case2)}")

sim_client = LLMClient(temperature=0.7)
asm_client = LLMClient(temperature=0.3)
eval_client = LLMClient(temperature=0.3)
runner = BatchRunner(cases=[case2], assistant_client=asm_client, simulator_client=sim_client,
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

# Build MD report
L = []
L.append(f"# Case 2 全链路评测报告")
L.append(f"")
L.append(f"**批次**: {BATCH_ID} | **画像数**: {total} | **对话数**: {len(convs)} | **评测数**: {len(results)}")
L.append(f"**对话耗时**: {t1-t0:.0f}s | **评测耗时**: {t2-t1:.0f}s")
L.append(f"**通知类 Case**: {_is_notification_call(case2)}")
L.append(f"")

# Case info
L.append(f"## 一、Case 指令")
L.append(f"")
L.append(f"| 属性 | 值 |")
L.append(f"|------|-----|")
L.append(f"| ID | {case2.id} |")
L.append(f"| 标题 | {case2.title} |")
L.append(f"| 业务线 | {case2.business_line} |")
L.append(f"| 复杂度 | {case2.complexity_score} |")
L.append(f"| 角色 | {case2.role[:100]} |")
L.append(f"| 任务 | {case2.task} |")
L.append(f"| 开场白 | {case2.opening_line[:120]} |")
L.append(f"")
L.append(f"### 流程步骤 ({len(case2.call_flow)} 步)")
for s in case2.call_flow:
    branches = getattr(s, 'branching', [])
    opt = '(可选)' if getattr(s, 'is_optional', False) else ''
    L.append(f"- **{s.title}**{opt}{': '+getattr(s,'description','')[:120] if getattr(s,'description','') else ''} ({len(branches)}分支)")
L.append(f"")
L.append(f"### 知识点 ({len(case2.knowledge_points)} 条)")
for k in case2.knowledge_points:
    L.append(f"- **{k.topic}**: {k.content}")
if not case2.knowledge_points:
    L.append(f"- (无知识点)")
L.append(f"")
L.append(f"### 约束条件 ({len(case2.constraints)} 条)")
for c in case2.constraints:
    L.append(f"- [{c.type}] {c.description[:120]}")
L.append(f"")

# Profiles
L.append(f"## 二、用户画像 ({total} 个)")
for cid, profiles in case_profiles.items():
    for i, p in enumerate(profiles):
        adv = ','.join(getattr(p, 'adversarial_strategy', []) or []) or '无'
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
needs_review_count = 0

for i, (conv, result) in enumerate(zip(convs, results)):
    profiles_list = case_profiles.get(conv.case_id, [])
    p_label = profiles_list[i].label if i < len(profiles_list) else f"P{i+1}"
    adv_strats = getattr(conv, 'adversarial_strategies', []) or []

    L.append(f"### 对话 {i+1}: {p_label}")
    L.append(f"| 状态 | 轮次 | 总分 | 可信度 | 对抗策略 |")
    L.append(f"|------|------|------|--------|----------|")
    sc = result.total_indicative_score if result else 0
    cl = f"{result.confidence.level}({result.confidence.overall:.2f})" if (result and result.confidence) else "N/A"
    adv_str = ','.join(adv_strats) if adv_strats else '无'
    L.append(f"| {conv.status} | {conv.total_turns} | {sc:.1f} | {cl} | {adv_str} |")
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
            if result.confidence.needs_human_review:
                needs_review_count += 1
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
            L.append(f"**交叉验证告警** ({len(result.cross_validation_alerts)}条):")
            for a in result.cross_validation_alerts:
                L.append(f"- [{a.severity}] {a.dimension}: {a.description[:150]}")
            L.append(f"")
        if result.meta_check_alerts:
            evidence_alerts = [a for a in result.meta_check_alerts if a.check_type == 'evidence']
            coverage_alerts = [a for a in result.meta_check_alerts if a.check_type == 'coverage']
            logic_alerts = [a for a in result.meta_check_alerts if a.check_type == 'logic']
            L.append(f"**元检查告警** ({len(result.meta_check_alerts)}条: ev:{len(evidence_alerts)} cov:{len(coverage_alerts)} logic:{len(logic_alerts)}):")
            for a in logic_alerts:
                L.append(f"- [{a.severity}][{a.check_type}] {a.description[:150]}")
            for a in coverage_alerts[:3]:
                L.append(f"- [{a.severity}][{a.check_type}] {a.description[:150]}")
            if evidence_alerts:
                L.append(f"- [evidence] ... ({len(evidence_alerts)} evidence mismatch alerts, bigram matching applied)")
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
    avg = sum(all_scores)/len(all_scores)
    L.append(f"### 分数汇总")
    L.append(f"| 均值 | 最高 | 最低 | 中位数 | 需复核 |")
    L.append(f"|------|------|------|--------|--------|")
    L.append(f"| {avg:.1f} | {max(all_scores):.1f} | {min(all_scores):.1f} | {sorted(all_scores)[len(all_scores)//2]:.1f} | {needs_review_count} |")
    L.append(f"")

if conf_levels:
    L.append(f"### 置信度分布")
    for lv in ['high','medium','low','unreliable']:
        L.append(f"- {lv}: {conf_levels.count(lv)}")
    L.append(f"")

# Quality Analysis
L.append(f"## 五、效果分析")
L.append(f"")
L.append(f"### Case 指令覆盖")
L.append(f"- 流程步骤 {len(case2.call_flow)} 步 ({sum(1 for s in case2.call_flow if getattr(s,'branching',[]))} 含分支) → FLOW_COVERAGE 维度")
L.append(f"- 知识点 {len(case2.knowledge_points)} 条 → KNOWLEDGE 维度{'（无KP，仅基础核查项）' if not case2.knowledge_points else ' 逐条核查'}")
L.append(f"- 约束 {len(case2.constraints)} 条 → CONSTRAINT 维度 + Tier1 规则分流")
L.append(f"- 通知类 Case → SAFETY 身份核实权重已降低 (0.5x)")
L.append(f"- 开场白 → OPENING 维度")
L.append(f"")

L.append(f"### 用户模拟器效果")
L.append(f"- 画像多样性: {total} 个画像")
L.append(f"- 对抗策略覆盖: {sum(1 for ps in case_profiles.values() for p in ps if getattr(p,'adversarial_strategy',[]))} 个画像含对抗策略")
L.append(f"- 对话正常率: {sum(1 for c in convs if c.status != '异常中断')}/{len(convs)}")
L.append(f"- 平均轮次: {sum(c.total_turns for c in convs)/max(len(convs),1):.1f}")
L.append(f"")

L.append(f"### 模拟对话模型效果")
if all_scores:
    L.append(f"- 分数范围: {min(all_scores):.0f} - {max(all_scores):.0f} (区分度 {max(all_scores)-min(all_scores):.0f})")
L.append(f"- SAFETY 评分分布: {all_ratings.get('SAFETY', [])}")
L.append(f"- OPENING 评分分布: {all_ratings.get('OPENING', [])}")
L.append(f"")

L.append(f"### 评测可量化性")
L.append(f"- 9 维度 × 5 级 = 45 级量化粒度")
if batch:
    L.append(f"- is_reliable 占比: {batch.get('is_reliable_ratio', 0):.0%}")
    L.append(f"- 批次健康度: {batch.get('self_reliability', {}).get('overall_health', 'N/A')}")
L.append(f"")

L.append(f"## 六、发现的问题与建议")
L.append(f"")
if _is_notification_call(case2):
    L.append(f"- ⚠️ 此为通知类 Case，SAFETY 身份核实权重已自动降低（0.5×），避免不合理否决")
if not case2.knowledge_points:
    L.append(f"- ⚠️ 此 Case 无知识点，KNOWLEDGE 仅使用基础核查项（无幻觉+跨轮一致+不确定诚实）")
L.append(f"- evidence 模糊匹配已使用 bigram 重叠算法（≥50%阈值），减少假阳性告警")
L.append(f"")

# Write
report_path = os.path.join(OUT_DIR, "full_report.md")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L))

# Raw data
raw_path = os.path.join(OUT_DIR, "raw_data.json")
rd = {"batch_id": BATCH_ID, "case_id": 2, "case_title": case2.title,
     "is_notification": _is_notification_call(case2),
     "total_profiles": total, "conversations": []}
for conv, result in zip(convs, results):
    cd = {"id": conv.id, "status": conv.status, "total_turns": conv.total_turns,
          "adversarial_strategies": getattr(conv, 'adversarial_strategies', []),
          "turns": [{"turn": t.turn_number, "speaker": t.speaker, "content": t.content}
                     for t in conv.turns]}
    if result:
        cd["ratings"] = result.ratings
        cd["total_score"] = result.total_indicative_score
        cd["confidence"] = {"overall": result.confidence.overall, "level": result.confidence.level,
                            "needs_human_review": result.confidence.needs_human_review} if result.confidence else None
        cd["tier1_constraint_count"] = result.tier1_constraint_count
        cd["llm_constraint_count"] = result.llm_constraint_count
    rd["conversations"].append(cd)
with open(raw_path, 'w', encoding='utf-8') as f:
    json.dump(rd, f, ensure_ascii=False, indent=2)

print(f"\nDone! Report: {report_path}")
print(f"Profiles: {total}  Conv: {len(convs)}  Eval: {len(results)}")
if all_scores:
    print(f"Score: {min(all_scores):.0f}-{max(all_scores):.0f} (avg {avg:.1f})")
