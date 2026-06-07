"""BatchRunner — 批量运行多场对话，串行或线程池并行"""
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional

from src.config import DATA_DIR
from src.loader.case_loader import load_cases
from src.loader.complexity import compute_max_turns
from src.llm.client import LLMClient
from src.llm.model_manager import get_generator_client, get_auditor_client
from src.models.case import Case
from src.models.conversation import Conversation
from src.models.evaluation import EvalResult
from src.simulator.profiles import build_profile_from_vector, UserProfile
from src.simulator.profile_params import (
    lhs_sample, subspace_lhs, deduplicate_vectors,
    compute_profile_count, extract_branch_constraints,
    translate_vector_to_anchor,
)
from src.simulator.profile_generator import ProfileGenerator
from src.simulator.profile_auditor import ProfileAuditor
from src.simulator.runner import DialogueRunner


# G6 策略池：每种组合覆盖不同对抗维度，保证多样性
_ADVERSARIAL_STRATEGY_POOL = [
    # 边界试探类 (需 bt 高)
    ["probe", "authority"],       # 试探 + 越权
    ["probe", "contradiction"],   # 试探 + 矛盾
    ["injection", "probe"],       # 注入 + 试探（injection 必须出现）
    ["authority", "emotion"],     # 越权 + 情绪
    # 一致性/情绪类 (需 tc 高 或 mv 高)
    ["contradiction", "emotion"], # 矛盾 + 情绪
    ["contradiction"],            # 仅矛盾
    ["emotion"],                  # 仅情绪
    ["probe"],                    # 仅试探
]

# 每种策略组合需要设置的维度
_STRATEGY_DIM_MAP = {
    "probe":         [(13, 0.61, 0.70)],   # boundary_testing: (0.6, 0.7], 仅触发 probe
    "authority":     [(13, 0.71, 0.80)],   # boundary_testing: (0.7, 0.8], 触发 probe+authority
    "injection":     [(13, 0.81, 0.95)],   # boundary_testing: (0.8, 1.0], 触发 probe+injection
    "contradiction": [(14, 0.71, 0.90)],   # truth_consistency: > 0.7
    "emotion":       [(12, 0.71, 0.85)],   # mood_volatility: > 0.7
}


def _group_defects_by_dim(defects) -> Dict[str, List[Dict]]:
    """将 Defect 列表按归因分组"""
    result: Dict[str, List[Dict]] = {}
    for d in defects:
        dim = getattr(d, "attribution", "unknown")
        result.setdefault(dim, []).append({
            "description": getattr(d, "description", ""),
            "severity": getattr(d, "severity", "一般"),
            "turn": getattr(d, "turn", 0),
            "attribution": getattr(d, "attribution", "Model"),
        })
    return result


def compute_max_turns_for_runner(case: Case) -> int:
    """后备: 如果 loader.complexity 不可用，直接用 call_flow 长度"""
    if case.call_flow:
        return max(8, len(case.call_flow) * 4)
    return 12


def _assign_adversarial_strategies(vectors: list) -> None:
    """为 50% 向量分配对抗策略，保证策略类型均匀分布

    每个对抗画像随机从策略池取一种组合（不重复取完一轮再洗牌），
    按策略需求设置对应维度的值。
    自然已触发对抗的向量优先计入 50% 配额。
    """
    n = len(vectors)
    n_target = n // 2  # 目标 50%

    # 统计已自然触发对抗的向量（mv > 0.7 或 bt > 0.6 或 tc > 0.7）
    naturally_adversarial = []
    for i, v in enumerate(vectors):
        if v[12] > 0.7 or v[13] > 0.6 or v[14] > 0.7:
            naturally_adversarial.append(i)

    # 从策略池分配 — 保证策略类型均匀覆盖
    # 方法：按唯一类型优先级排序池条目，确保小样本时最大限度覆盖所有类型
    pool = list(_ADVERSARIAL_STRATEGY_POOL)
    random.shuffle(pool)

    # 构建类型优先级：稀有类型优先
    all_strategy_types = {"probe", "authority", "injection", "contradiction", "emotion"}
    # 统计每种类型在池中的出现次数
    type_freq = {t: sum(1 for entry in pool if t in entry) for t in all_strategy_types}
    # 按"最稀有类型"排序池条目——含稀有类型的条目排前面
    pool.sort(key=lambda entry: min(type_freq[s] for s in entry))
    # 轻微再次 shuffle 保持同优先级内的随机性
    # 分组：同"最稀有度"的条目内部随机
    i = 0
    while i < len(pool):
        j = i
        while j < len(pool) and min(type_freq[s] for s in pool[j]) == min(type_freq[s] for s in pool[i]):
            j += 1
        group = pool[i:j]
        random.shuffle(group)
        pool[i:j] = group
        i = j

    # 确定哪些向量需要主动赋对抗策略
    remaining = n_target - len(naturally_adversarial)
    if remaining < 0:
        # 自然对抗太多，随机降级一部分以保持 50%
        random.shuffle(naturally_adversarial)
        to_downgrade = naturally_adversarial[: -remaining]
        for idx in to_downgrade:
            v = vectors[idx]
            for dim_idx in (12, 13, 14):
                v[dim_idx] = random.uniform(0.1, 0.5)
        return
    elif remaining == 0:
        # 全自然对抗：检查策略类型多样性，缺失类型从池中补充
        all_types_in_natural = set()
        for idx in naturally_adversarial:
            v = vectors[idx]
            bt, tc, mv = v[13], v[14], v[12]
            if bt > 0.8:
                all_types_in_natural.update(["probe", "injection"])
            elif bt > 0.7:
                all_types_in_natural.update(["probe", "authority"])
            elif bt > 0.6:
                all_types_in_natural.add("probe")
            if tc > 0.7:
                all_types_in_natural.add("contradiction")
            if mv > 0.7:
                all_types_in_natural.add("emotion")
        desired_types = {"probe", "authority", "injection", "contradiction", "emotion"}
        missing = desired_types - all_types_in_natural
        # 将缺失类型分散到多个自然对抗向量上（每个向量加1种新类型，避免cap冲突）
        if missing and len(naturally_adversarial) >= 1:
            to_fix = list(missing)
            random.shuffle(to_fix)
            # 每个缺失类型分配给一个不同的自然对抗向量
            for i, s in enumerate(to_fix):
                idx = naturally_adversarial[i % len(naturally_adversarial)]
                v = vectors[idx]
                for dim_idx, lo, hi in _STRATEGY_DIM_MAP.get(s, []):
                    if v[dim_idx] < lo:
                        v[dim_idx] = random.uniform(lo, hi)
        return

    # 排除已有自然对抗的向量
    eligible = [i for i in range(n) if i not in naturally_adversarial]
    random.shuffle(eligible)
    selected = eligible[:remaining]

    # 策略分配：保证每种策略类型至少覆盖一次
    # 提取所有独特策略类型并按类型分组
    strategy_groups = {}  # type_name -> [pool_indices]
    for pi, strat_list in enumerate(pool):
        for s in strat_list:
            strategy_groups.setdefault(s, []).append(pi)

    # 收集必须覆盖的策略类型（probe 和 authority 可能二选一，优先 probe）
    all_types = sorted(strategy_groups.keys())
    # 轮替分配：首轮确保每种类型至少一次
    assigned_types = set()
    assigned = []
    pool_idx = 0
    for j in range(remaining):
        if pool_idx < len(pool):
            strategies = pool[pool_idx]
        else:
            # 所有池条目已用完，重新洗牌
            random.shuffle(pool)
            pool_idx = 0
            strategies = pool[pool_idx]
        assigned.append(strategies)
        for s in strategies:
            assigned_types.add(s)
        pool_idx += 1

    # 如果首轮未覆盖所有类型，替换最后一个为缺失类型
    missing = set(all_types) - assigned_types
    if missing and len(assigned) >= 1:
        # 找包含缺失类型的池条目
        for pi, strat_list in enumerate(pool):
            if any(s in missing for s in strat_list):
                assigned[-1] = strat_list
                break

    for j, idx in enumerate(selected):
        strategies = assigned[j % len(assigned)]
        v = vectors[idx]
        for s in strategies:
            for dim_idx, lo, hi in _STRATEGY_DIM_MAP.get(s, []):
                if v[dim_idx] < lo:
                    v[dim_idx] = random.uniform(lo, hi)


def _run_single_v2(case: Case, profile: UserProfile,
                    assistant_client: LLMClient, simulator_client: LLMClient,
                    use_raw_prompt: bool, max_turns: int) -> Conversation:
    """运行单场对话（供线程池调用）"""
    runner = DialogueRunner.create_with_llm(
        case=case,
        profile=profile,
        assistant_client=assistant_client,
        simulator_client=simulator_client,
        use_raw_prompt=use_raw_prompt,
    )
    conv = runner.run(max_turns=max_turns)
    conv.complexity_score = getattr(case, 'complexity_score', 0.0)
    return conv


class BatchRunner:
    """批量对话运行器 — 参数化画像路径"""

    def __init__(self, cases: List[Case],
                 assistant_client: Optional[LLMClient] = None,
                 simulator_client: Optional[LLMClient] = None,
                 eval_client: Optional[LLMClient] = None,
                 use_raw_prompt: bool = True):
        self.cases = cases
        self.assistant_client = assistant_client or LLMClient(temperature=0.0)
        self.simulator_client = simulator_client or LLMClient(temperature=0.7)
        self.eval_client = eval_client  # Phase 3 Judge client
        self.use_raw_prompt = use_raw_prompt
        self._case_profiles: Dict[int, List[UserProfile]] = {}

        print(f"BatchRunner 初始化: {len(cases)} cases")

    # ---- Phase 0: 画像生成 ----

    def generate_profiles(
        self,
        n_global: int = 2,
        gen_client: Optional[LLMClient] = None,
        verbose: bool = True,
    ) -> Dict[int, List[UserProfile]]:
        """Phase 0: 为所有 case 生成参数化用户画像

        P1: 全空间 LHS baseline (n_global per case)
        P2: 分支约束子空间 LHS + 极端画像
        对抗比例: 50% 向量对抗维度提升到高值区间
        Returns: {case_id: [UserProfile, ...]}
        """
        gen_client = gen_client or get_generator_client(temperature=0.7)
        case_profiles: Dict[int, List[UserProfile]] = {}

        for case in self.cases:
            # 根据 case 复杂度动态调整自检严格度
            from src.simulator.profile_generator import compute_self_check_thresholds
            md_limit, dsv_limit, retries = compute_self_check_thresholds(
                case.complexity_score
            )
            gen = ProfileGenerator(gen_client,
                                   max_dev_limit=md_limit,
                                   d_sv_limit=dsv_limit,
                                   max_retries=retries)
            vectors_for_case: List[List[float]] = []

            # P1: 全空间 LHS baseline
            vectors_for_case.extend(lhs_sample(n_global, 15))

            # P2: 分支约束子空间 LHS
            branch_constraints = extract_branch_constraints(case.call_flow)
            n_per_branch = compute_profile_count(case)

            if branch_constraints:
                free_dims = [d for d in range(15) if d not in branch_constraints]
                constrained_vectors = subspace_lhs(n_per_branch, branch_constraints)
                constrained_vectors = deduplicate_vectors(
                    constrained_vectors, min_distance=0.3, free_dim_indices=free_dims,
                )
                vectors_for_case.extend(constrained_vectors)

                # G2: 极端画像 — 每个约束维度各自取其区间内的极端值
                for extreme_pole in ("low", "high"):
                    v_extreme = [random.random() for _ in range(15)]
                    for dim_idx, (lo, hi) in branch_constraints.items():
                        mid = (lo + hi) / 2
                        if extreme_pole == "low":
                            # 取区间内尽可能低的值，但若约束整体偏高则取 lo
                            v_extreme[dim_idx] = lo if lo > 0.1 else max(0.0, lo + 0.02)
                        else:
                            # 取区间内尽可能高的值，但若约束整体偏低则取 hi
                            v_extreme[dim_idx] = hi if hi < 0.9 else min(1.0, hi - 0.02)
                    vectors_for_case.append(v_extreme)
            else:
                # 无分支约束时：额外全空间 LHS + 语义维度对极端画像
                vectors_for_case.extend(lhs_sample(max(n_per_branch, 2), 15))
                # G2 策略B: 从 5 组客服场景语义维度对中随机选 1 组，生成双低+双高对比画像
                _DIM_PAIRS = [
                    (0, 5),   # agreeableness + patience: 敌意急躁 ↔ 信任耐心
                    (2, 7),   # neuroticism + politeness: 焦虑粗鲁 ↔ 冷静礼貌
                    (8, 12),  # assertiveness + mood_volatility: 被动稳定 ↔ 强硬波动
                    (9, 11),  # information_verification + initial_mood: 轻信乐观 ↔ 怀疑悲观
                    (13, 14), # boundary_testing + truth_consistency: 守规一致 ↔ 越界矛盾
                ]
                dim_a, dim_b = random.choice(_DIM_PAIRS)
                # 双低极端画像
                v_ll = [random.random() for _ in range(15)]
                v_ll[dim_a] = max(0.0, random.uniform(0, 0.1))
                v_ll[dim_b] = max(0.0, random.uniform(0, 0.1))
                vectors_for_case.append(v_ll)
                # 双高极端画像
                v_hh = [random.random() for _ in range(15)]
                v_hh[dim_a] = min(1.0, random.uniform(0.9, 1.0))
                v_hh[dim_b] = min(1.0, random.uniform(0.9, 1.0))
                vectors_for_case.append(v_hh)

            # G6: 50% 画像分配对抗策略，每种策略类型均匀覆盖
            _assign_adversarial_strategies(vectors_for_case)

            if verbose:
                print(f"Case {case.id}: 生成 {len(vectors_for_case)} 个画像文本...")
            generated = gen.batch_generate(vectors_for_case, verbose=verbose)
            case_profiles[case.id] = generated

        self._case_profiles = case_profiles
        return case_profiles

    # ---- 运行 ----

    def run_all(self, parallel: bool = False, max_workers: int = 5,
                max_turns: Optional[int] = None,
                profiles_dict: Optional[Dict[int, List[UserProfile]]] = None,
                run_eval: bool = False,
                verify_reliability: bool = False,
                ) -> List[Conversation]:
        """执行所有对话

        Args:
            parallel: True 使用 ThreadPoolExecutor 并行
            max_workers: 并行线程数
            max_turns: 每场对话最大轮次，None 时取所有 case 的自动计算结果最大值
            profiles_dict: 参数化画像 {case_id: [UserProfile, ...]}
                           传入则走参数化路径, 不传则走传统路径
            run_eval: Phase 2 完成后自动运行 Phase 3 评测（同 session，零 JSON 开销）
            verify_reliability: 重测信度验证——抽样 5 条重跑评测并计算一致率
        """
        if max_turns is None:
            print(f"max_turns: 每个 case 自动计算")
        results: List[Conversation] = []
        start_time = time.time()
        self._case_profiles = profiles_dict if profiles_dict is not None else self._case_profiles

        if self._case_profiles:
            results = self._run_parameterized(parallel, max_workers, max_turns)
        else:
            print("请先调用 generate_profiles() 或传入 profiles_dict")

        if run_eval:
            self.run_phase3(results, verify_reliability=verify_reliability)

        elapsed = time.time() - start_time
        print(f"\n全部完成: {len(results)} 场对话, 耗时 {elapsed:.0f}s")
        self._print_status_summary(results)
        return results

    def _run_parameterized(self, parallel: bool, max_workers: int,
                           max_turns: Optional[int]) -> List[Conversation]:
        """参数化画像路径（每 case 使用各自的 max_turns）"""
        tasks = []
        for case in self.cases:
            case_mt = max_turns if max_turns is not None else compute_max_turns(case)
            for profile in self._case_profiles.get(case.id, []):
                tasks.append((case, profile, case_mt))

        total = len(tasks)
        print(f"参数化模式: {total} 场对话")

        if not parallel:
            results = []
            for idx, (case, profile, case_mt) in enumerate(tasks):
                print(f"[{idx + 1}/{total}] case{case.id} x {profile.label} (max={case_mt}) ... ",
                      end="", flush=True)
                try:
                    conv = _run_single_v2(
                        case, profile,
                        self.assistant_client, self.simulator_client,
                        self.use_raw_prompt, case_mt,
                    )
                    results.append(conv)
                    print(f"{conv.status} ({conv.total_turns}轮)")
                except Exception as e:
                    print(f"异常: {e}")
            return results
        else:
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        _run_single_v2, case, profile,
                        self.assistant_client, self.simulator_client,
                        self.use_raw_prompt, case_mt,
                    ): (case.id, profile.label)
                    for case, profile, case_mt in tasks
                }
                completed = 0
                for future in as_completed(futures):
                    case_id, label = futures[future]
                    try:
                        conv = future.result()
                        results.append(conv)
                    except Exception as e:
                        print(f"[异常] case{case_id} x {label}: {e}")
                    completed += 1
                    if completed % 10 == 0:
                        print(f"进度: {completed}/{total} 场完成")
            return results

    # ---- Phase 3: 评测 ----

    def run_phase3(
        self,
        conversations: List[Conversation],
        eval_client: Optional[LLMClient] = None,
        verify_reliability: bool = False,
    ) -> List[EvalResult]:
        """Phase 3: 信号增强清单评测

        对每场对话执行完整的清单核查评测流程：
        Tier 1 规则 → Tier 1.5 信号 → 清单生成 → 9 Judge 并发 LLM 核查
        → 评级推导 → 表面合规 → 归因 → EvalConfidence

        Args:
            conversations: Phase 2 完成的对话列表
            eval_client: Judge LLM 客户端（不传则用 self.assistant_client）
            verify_reliability: 重测信度验证——抽样 5 条重跑评测并计算一致率

        Returns:
            eval_results 列表，同时结果写入 conv.eval_result
        """
        from src.eval.orchestrator import EvalOrchestrator
        from src.eval.drift_monitor import BatchAnalyzer
        from src.eval.checklist_evolver import ChecklistEvolver

        client = eval_client or self.eval_client or self.assistant_client
        orchestrator = EvalOrchestrator(client)
        evolver = ChecklistEvolver()

        eval_results = []
        total = len(conversations)
        print(f"\nPhase 3 评测: {total} 场对话")

        for idx, conv in enumerate(conversations):
            case = self._get_case_by_id(conv.case_id)
            if case is None:
                print(f"[{idx + 1}/{total}] {conv.id} — 跳过 (找不到 Case {conv.case_id})")
                continue

            try:
                result = orchestrator.run(conv, case)
                eval_results.append(result)

                # 积累 additional_defects 到进化引擎
                evolver.accumulate_from_result(self._result_to_dict(result))

                is_rel = "可靠" if result.confidence and result.confidence.is_reliable else "低可信"
                total_score = result.total_indicative_score
                print(f"[{idx + 1}/{total}] {conv.id} — {total_score:.1f}分 [{is_rel}] "
                      f"({result.confidence.level if result.confidence else '?'})")
            except Exception as e:
                print(f"[{idx + 1}/{total}] {conv.id} — 评测异常: {e}")

        # 批次聚合分析
        result_dicts = [
            self._result_to_dict(r, conv)
            for r, conv in zip(eval_results, conversations)
        ]

        # Phase 4: 重测信度验证（可选）
        retest_results = None
        conv_texts = None
        max_turns_list = None
        if verify_reliability:
            sample_size = min(5, total)
            sample_indices = random.sample(range(total), sample_size)
            retest_convs = [conversations[i] for i in sample_indices]

            print(f"\n  重测信度验证: 抽样 {sample_size} 场重跑评测...")
            retest_results = []
            for idx, (i, conv) in enumerate(zip(sample_indices, retest_convs)):
                case = self._get_case_by_id(conv.case_id)
                if case is None:
                    continue
                try:
                    retry_result = orchestrator.run(conv, case)
                    retest_results.append(self._result_to_dict(retry_result, conv))
                    print(f"    [{idx + 1}/{sample_size}] {conv.id} — 重测完成")
                except Exception as e:
                    print(f"    [{idx + 1}/{sample_size}] {conv.id} — 重测异常: {e}")

            # 提取对话文本和轮次数用于证据引用验证
            conv_texts = []
            max_turns_list = []
            for i in sample_indices:
                conv = conversations[i]
                # 拼接所有轮次内容
                texts = []
                for turn in conv.turns:
                    texts.append(f"[T{turn.turn_number}] {turn.speaker}: {turn.content}")
                conv_texts.append("\n".join(texts))
                max_turns_list.append(conv.total_turns)

        analyzer = BatchAnalyzer()
        batch_summary = analyzer.analyze(
            result_dicts,
            retest_results=retest_results,
            conv_texts=conv_texts,
            max_turns_list=max_turns_list,
        )
        self._last_batch_summary = batch_summary

        # 打印批次汇总
        self._print_batch_summary(batch_summary)
        if verify_reliability:
            self._print_reliability_report(batch_summary)

        # 数据导出：全链路节点数据自动保存
        self._export_results(conversations, eval_results, batch_summary)

        return eval_results

    def _export_results(self, conversations, eval_results, batch_summary):
        """自动导出全链路数据到 data/exports/"""
        try:
            from src.utils.data_exporter import DataExporter
            exporter = DataExporter(output_dir="data/exports",
                                    batch_id=datetime.now().strftime("%Y%m%d_%H%M%S"))

            # 导出 Case 信息
            if self.cases:
                for case in self.cases:
                    exporter.export_case(case)

            # 导出画像
            if self._case_profiles:
                exporter.export_profiles(self._case_profiles)

            # 逐条导出对话 + 评测结果
            conv_time = 0
            eval_time = 0
            for i, (conv, result) in enumerate(zip(conversations, eval_results)):
                conv_id = conv.id or f"conv_{i}"
                exporter.export_conversation(conv, i)           # MD 格式（人类可读）
                exporter.export_conversation(conv, i, fmt="json")  # JSON 格式（含完整 parsed_tags，供优化引擎消费）
                if result:
                    exporter.export_evaluation(result, conv_id)
                    conv_time += getattr(conv, "duration_seconds", 0)

            # 批次汇总
            exporter.export_batch_summary(conversations, eval_results, batch_summary)

            # 优化引擎对接数据（OptimizationFeed JSON）
            exporter.export_optimization_feed(eval_results, conversations)

            # 生成叙述性评测报告（文字解说为主）
            try:
                from src.eval.report_generator import generate_narrative_report, generate_batch_narrative
                reports_dir = exporter.node_dir / "narrative_reports"
                reports_dir.mkdir(exist_ok=True)

                for i, (conv, result) in enumerate(zip(conversations, eval_results)):
                    if result is None:
                        continue
                    case = self._get_case_by_id(conv.case_id)
                    if case is None:
                        continue
                    report_md = generate_narrative_report(result, conv, case, i + 1)
                    rpath = reports_dir / f"report_conv_{i+1}.md"
                    rpath.write_text(report_md, encoding="utf-8")

                # 批次叙述性汇总
                batch_case = self._get_case_by_id(conversations[0].case_id) if conversations else None
                if batch_case:
                    batch_narrative = generate_batch_narrative(
                        conversations, eval_results, batch_case,
                        conv_time=sum(getattr(c, "duration_seconds", 0) for c in conversations),
                        eval_time=0)
                    (reports_dir / "batch_narrative.md").write_text(batch_narrative, encoding="utf-8")

                print(f"  叙述性报告: data/exports/{exporter.node_dir.name}/narrative_reports/")
            except Exception as e:
                print(f"  叙述性报告生成失败: {e}")

            # 写入清单
            manifest_path = exporter._write_manifest(conv_time, eval_time=0)
            print(f"  数据导出: data/exports/{Path(manifest_path).parent.name}/")
        except Exception as e:
            print(f"  数据导出失败: {e}")

    def replay_and_eval(self, replay_dir: str) -> List[Conversation]:
        """从历史 JSON 加载对话并运行 Phase 3 评测（跨 session 历史回放）"""
        from src.loader.conversation_loader import load_conversations_from_dir

        conversations = load_conversations_from_dir(replay_dir)
        print(f"从 {replay_dir} 加载 {len(conversations)} 场历史对话")

        self.run_phase3(conversations)
        return conversations

    def _get_case_by_id(self, case_id: int) -> Optional[Case]:
        """根据 case_id 查找 Case"""
        for case in self.cases:
            if case.id == case_id:
                return case
        return None

    @staticmethod
    def _result_to_dict(result: EvalResult, conv: Conversation = None) -> Dict:
        """将 EvalResult 转为可序列化的 dict（用于批次分析和进化引擎）"""
        return {
            "conversation_id": result.conversation_id,
            "case_id": result.case_id,
            "ratings": result.ratings,
            "indicative_scores": result.indicative_scores,
            "total_indicative_score": result.total_indicative_score,
            "total_score_100": result.total_score_100,
            "surface_compliance_flags": result.surface_compliance_flags,
            "dimension_checklists": {
                dim: {
                    "items": [
                        {
                            "item_id": i.item_id,
                            "description": i.description,
                            "source": i.source,
                            "status": i.status,
                            "evidence": i.evidence,
                            "signal_consistency": i.signal_consistency,
                            "weight": i.weight,
                        }
                        for i in cl.items
                    ],
                }
                for dim, cl in result.dimension_checklists.items()
            },
            "additional_defects_by_dim": _group_defects_by_dim(result.additional_defects),
            "complexity_score": conv.complexity_score if conv else 0,
            "is_reliable": result.confidence.is_reliable if result.confidence else True,
            "confidence_level": result.confidence.level if result.confidence else "medium",
        }

    @staticmethod
    def _print_batch_summary(summary: Dict) -> None:
        """打印批次聚合摘要"""
        print(f"\n{'='*60}")
        print("Phase 3 批次评测汇总")
        print(f"{'='*60}")
        print(f"  评测数: {summary.get('n_results', 0)}")

        if summary.get("alerts"):
            print("  告警:")
            for alert in summary["alerts"]:
                try:
                    print(f"    - {alert}")
                except UnicodeEncodeError:
                    print(f"    - (encoding issue, skipped)")
        else:
            print("  告警: 无")

        dist = summary.get("rating_distribution", {})
        if dist:
            print("  各维度不合格率:")
            for dim, ratios in sorted(dist.items()):
                fail = ratios.get("不合格", 0)
                if fail > 0:
                    print(f"    {dim}: {fail:.0%}")
            # 仅显示有问题的，如果全OK也标注
            all_ok = all(ratios.get("不合格", 0) == 0 for ratios in dist.values())
            if all_ok:
                print("    全部维度无不合格")

        print(f"  低可信占比: {summary.get('is_reliable_ratio', 0):.0%}")
        print(f"  复杂度分层: {summary.get('complexity_strata', {})}")

        # 自验证摘要
        sr = summary.get("self_reliability", {})
        if sr:
            print(f"  评测系统健康度: {sr.get('overall_health', 1.0):.2f} "
                  f"{'[健康]' if sr.get('is_healthy', True) else '[需关注]'}")
            tr = sr.get("test_retest", {})
            if tr:
                print(f"  重测信度: {tr.get('overall_reliability', 1.0):.2f}")
            ev = sr.get("evidence_validity", {})
            if ev:
                print(f"  证据有效性: {ev.get('avg_validity_ratio', 1.0):.0%}")
        print(f"{'='*60}")

    @staticmethod
    def _print_reliability_report(summary: Dict) -> None:
        """打印详细的重测信度报告"""
        sr = summary.get("self_reliability", {})
        if not sr:
            return
        print(f"\n{'='*60}")
        print("评测系统自身可靠性报告")
        print(f"{'='*60}")
        print(f"  整体健康度: {sr.get('overall_health', 1.0):.2f} "
              f"{'[健康]' if sr.get('is_healthy', True) else '[需关注]'}")

        tr = sr.get("test_retest", {})
        if tr:
            print(f"  重测信度: status一致率={tr.get('status_agreement_rate', 0):.1%}, "
                  f"评级一致率={tr.get('rating_agreement_rate', 0):.1%}")
            print(f"  综合重测信度: {tr.get('overall_reliability', 0):.2f} "
                  f"{'[稳定]' if tr.get('is_reliable', True) else '[不稳定]'}")

        dist = sr.get("score_distribution", {})
        if dist.get("alerts"):
            print("  区分力告警:")
            for a in dist["alerts"]:
                print(f"    - {a['dimension']}: {a['description']}")

        ija = sr.get("inter_judge_agreement", {})
        if ija.get("redundant_pairs"):
            print("  维度冗余告警:")
            for p in ija["redundant_pairs"]:
                print(f"    - {p['dim_a']} ↔ {p['dim_b']} (ρ={p['spearman_rho']:.3f})")

        ev = sr.get("evidence_validity", {})
        if ev:
            print(f"  证据引用验证: 抽样{ev.get('sample_size', 0)}条, "
                  f"有效性={ev.get('avg_validity_ratio', 0):.1%}")
            for detail in ev.get("details", []):
                if detail.get("invalid_citations"):
                    for ic in detail["invalid_citations"]:
                        print(f"    - {ic['dimension']}/{ic['item_id']}: {ic['reason']}")

        print(f"{'='*60}")

    # ---- 保存 ----

    def save_results(self, conversations: List[Conversation],
                     output_dir: Optional[str] = None):
        """每场对话存为独立 JSON 文件"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = str(DATA_DIR / "conversations" / f"batch_{timestamp}")

        os.makedirs(output_dir, exist_ok=True)

        for conv in conversations:
            data = {
                "id": conv.id,
                "case_id": conv.case_id,
                "user_profile": conv.user_profile,
                "status": conv.status,
                "total_turns": conv.total_turns,
                "duration_seconds": conv.duration_seconds,
                "complexity_score": conv.complexity_score,
                "profile_label": conv.profile_label,
                "sampled_vector": conv.sampled_vector,
                "verified_vector": conv.verified_vector,
                "audited_vector": conv.audited_vector,
                "adversarial_strategies": conv.adversarial_strategies,
                "consistency": conv.consistency,
                "branch_coverage": conv.branch_coverage,
                "model_breakdown_count": conv.model_breakdown_count,
                "state_trajectory": [
                    t.parsed_tags.get("state")
                    for t in conv.turns
                    if t.speaker == "user" and t.parsed_tags
                    and isinstance(t.parsed_tags.get("state"), dict)
                ],
                "turns": [
                    {
                        "turn_number": t.turn_number,
                        "speaker": t.speaker,
                        "content": t.content,
                        "timestamp": t.timestamp,
                        "parsed_tags": t.parsed_tags if t.parsed_tags else None,
                    }
                    for t in conv.turns
                ],
            }
            # Phase 3 评测结果
            if conv.eval_result:
                er = conv.eval_result
                data["eval"] = {
                    "ratings": er.ratings,
                    "indicative_scores": er.indicative_scores,
                    "total_indicative_score": er.total_indicative_score,
                    "surface_compliance_flags": er.surface_compliance_flags,
                    "rule_check_issues": er.rule_check_issues,
                    "dimension_checklists": {
                        dim: {
                            "yes_ratio": cl.yes_ratio,
                            "weighted_yes_ratio": round(cl.weighted_yes_ratio, 3),
                            "items": [
                                {
                                    "item_id": i.item_id,
                                    "source": i.source,
                                    "status": i.status,
                                    "evidence": i.evidence,
                                    "signal_consistency": i.signal_consistency,
                                }
                                for i in cl.items
                            ],
                        }
                        for dim, cl in er.dimension_checklists.items()
                    },
                    "additional_defects": [
                        {
                            "description": d.description,
                            "severity": d.severity,
                            "turn": d.turn,
                            "attribution": d.attribution,
                        }
                        for d in er.additional_defects
                    ],
                    "attributions": [
                        {
                            "source": a.source,
                            "category": a.category,
                            "description": a.description,
                            "confidence": a.confidence,
                            "is_actionable": a.is_actionable,
                            "suggested_actions": a.suggested_actions,
                        }
                        for a in er.attributions
                    ],
                    "summary": er.summary,
                    "improvement_suggestions": er.improvement_suggestions,
                }
                if er.confidence:
                    data["eval"]["confidence"] = {
                        "overall": er.confidence.overall,
                        "level": er.confidence.level,
                        "is_reliable": er.confidence.is_reliable,
                        "simulator_tier": er.confidence.simulator_tier,
                        "signal_conflict_count": er.confidence.signal_conflict_count,
                        "cross_judge_anomalies": er.confidence.cross_judge_anomalies,
                    }

            filepath = os.path.join(output_dir, f"{conv.id}.json")
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"对话已保存: {output_dir}/ ({len(conversations)} 个文件)")

    # ---- Phase 2: 审计 ----

    def audit_results(
        self,
        conversations: List[Conversation],
        audit_client: Optional[LLMClient] = None,
        sample_ratio: float = 0.1,
    ) -> None:
        """Phase 2: 对话后行为审计

        Path A: 全量 — 基于 state 标签的循环一致性（零 LLM 成本）
        Path B: 抽样 — LLM 行为审计交叉验证
        """
        auditor = ProfileAuditor(audit_client or get_auditor_client(temperature=0.0))

        # Path A: 所有对话
        path_a_ok = 0
        for conv in conversations:
            try:
                state_trajectory = []
                for turn in conv.turns:
                    if turn.speaker == "user" and turn.parsed_tags:
                        state_dict = turn.parsed_tags.get("state")
                        if state_dict and isinstance(state_dict, dict):
                            state_trajectory.append(state_dict)
                auditor.audit_path_a(conv, state_trajectory)
                path_a_ok += 1
            except Exception as e:
                print(f"[审计警告] Path A {conv.id}: {e}")

        # Path B: 抽样
        path_b_ok = 0
        sample_size = max(int(len(conversations) * sample_ratio), 1)
        sampled = random.sample(conversations, min(sample_size, len(conversations)))

        for conv in sampled:
            try:
                auditor.audit_path_b(conv)
                path_b_ok += 1
            except Exception as e:
                print(f"[审计警告] Path B {conv.id}: {e}")

        print(f"审计完成: Path A={path_a_ok}/{len(conversations)}场, Path B={path_b_ok}/{len(sampled)}场")

    @staticmethod
    def _print_status_summary(results: List[Conversation]):
        """打印状态汇总"""
        status_counts = {}
        for conv in results:
            status_counts[conv.status] = status_counts.get(conv.status, 0) + 1
        print("状态分布:")
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count} 场")

        avg_turns = sum(c.total_turns for c in results) / max(len(results), 1)
        print(f"平均轮次: {avg_turns:.1f}")


# ============================================================
# 便捷入口: 单 case / 多 case 一键测试
# ============================================================

def run_case(
    case_id: int,
    n_global: int = 2,
    api_key: Optional[str] = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    parallel: bool = False,
    save: bool = True,
    run_eval: bool = False,
    verify_reliability: bool = False,
    judge_model: Optional[str] = None,
) -> dict:
    """对单个 case 运行完整测试矩阵（全空间 + 分支 + 极端画像）

    Args:
        case_id: 目标 case ID
        n_global: 全空间 LHS 采样数
        api_key: LLM API key，不传则从环境变量读取
        base_url: API 地址
        model: 模型名
        parallel: 是否并行跑对话
        save: 是否保存结果 JSON

    Returns:
        {"case_id": int, "n_profiles": int, "n_conversations": int,
         "status_distribution": dict, "tier_distribution": dict,
         "adversarial_ratio": float, "strategy_distribution": dict,
         "avg_turns": float, "abnormal_count": int, "results": [...]}
    """
    if api_key is None:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("API_KEY", "")

    all_cases = load_cases()
    cases = [c for c in all_cases if c.id == case_id]
    if not cases:
        raise ValueError(f"未找到 case_id={case_id}（可用: {[c.id for c in all_cases[:10]]}...）")

    gen_client = LLMClient(api_key=api_key, base_url=base_url, model=model, temperature=0.7)
    sim_client = LLMClient(api_key=api_key, base_url=base_url, model=model, temperature=0.7)
    asst_client = LLMClient(api_key=api_key, base_url=base_url, model=model, temperature=0.0)
    audit_client = LLMClient(api_key=api_key, base_url=base_url, model=model, temperature=0.0)

    # Phase 3 Judge 客户端（独立于 Simulator，可选不同模型家族）
    j_model = judge_model or model
    judge_client = LLMClient(api_key=api_key, base_url=base_url, model=j_model, temperature=0.3)

    runner = BatchRunner(cases, asst_client, sim_client, eval_client=judge_client)

    # Phase 0
    profiles = runner.generate_profiles(n_global=n_global, gen_client=gen_client)

    # Phase 1
    conversations = runner.run_all(parallel=parallel)

    # Phase 2
    try:
        runner.audit_results(conversations, audit_client, sample_ratio=0.3)
    except Exception as e:
        print(f"[警告] 审计阶段异常: {e}")

    # Phase 3
    eval_results = None
    if run_eval:
        try:
            eval_results = runner.run_phase3(conversations, judge_client, verify_reliability=verify_reliability)
        except Exception as e:
            print(f"[警告] 评测阶段异常: {e}")

    if save:
        try:
            runner.save_results(conversations)
        except Exception as e:
            print(f"[警告] 保存失败: {e}")

    # Compile summary
    status_counts = {}
    tier_counts = {}
    total_turns = 0
    abnormal = 0
    adv_count = 0
    strat_types = Counter()
    detail = []

    for conv in conversations:
        status_counts[conv.status] = status_counts.get(conv.status, 0) + 1
        consistency = conv.consistency or {}
        tier = consistency.get("tier", "?")
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        total_turns += conv.total_turns
        if conv.status == "异常中断":
            abnormal += 1
        adv = conv.adversarial_strategies or []
        if adv:
            adv_count += 1
        for s in adv:
            strat_types[s] += 1
        d_sv = consistency.get("d_sv")
        d_sa = consistency.get("d_sa")
        detail_entry = {
            "profile_label": conv.profile_label,
            "status": conv.status,
            "total_turns": conv.total_turns,
            "tier": tier,
            "adversarial": adv,
            "d_sv": round(d_sv, 4) if d_sv is not None else None,
            "d_sa": round(d_sa, 4) if d_sa is not None else None,
        }
        if conv.eval_result:
            detail_entry["eval"] = {
                "total_score": conv.eval_result.total_indicative_score,
                "ratings": conv.eval_result.ratings,
                "is_reliable": conv.eval_result.confidence.is_reliable if conv.eval_result.confidence else True,
            }
        detail.append(detail_entry)

    n = len(conversations)
    summary = {
        "case_id": case_id,
        "n_profiles": sum(len(v) for v in profiles.values()),
        "n_conversations": n,
        "status_distribution": status_counts,
        "tier_distribution": tier_counts,
        "adversarial_ratio": adv_count / n if n else 0,
        "strategy_distribution": dict(strat_types),
        "avg_turns": total_turns / n if n else 0,
        "abnormal_count": abnormal,
        "results": detail,
    }

    # Print summary table
    print(f"\n{'='*60}")
    print(f"Case #{case_id} 测试完成")
    print(f"{'='*60}")
    print(f"  画像数: {summary['n_profiles']}  |  对抗比例: {summary['adversarial_ratio']:.0%}")
    print(f"  策略分布: {summary['strategy_distribution']}")
    print(f"  对话数: {n}  |  状态: {status_counts}")
    print(f"  一致性: {tier_counts}  |  异常中断: {abnormal}")
    print(f"  平均轮次: {summary['avg_turns']:.1f}")
    print(f"{'='*60}")

    return summary


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    import argparse
    import os as _os
    from dotenv import load_dotenv as _load_dotenv

    parser = argparse.ArgumentParser(
        description="BatchRunner — 批量画像测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m src.simulator.batch_runner --case-id 2
  python -m src.simulator.batch_runner --case-id 2 --n-global 3
  python -m src.simulator.batch_runner --case-id 1,2,3 --parallel
        """,
    )
    parser.add_argument("--case-id", type=str, required=True,
                        help="目标 case ID，多个用逗号分隔（如 1,2,3）")
    parser.add_argument("--n-global", type=int, default=5,
                        help="全空间 LHS 采样数（default: 5）")
    parser.add_argument("--parallel", action="store_true",
                        help="是否并行跑对话")
    parser.add_argument("--run-eval", action="store_true",
                        help="运行 Phase 3 评测")
    parser.add_argument("--verify-reliability", action="store_true",
                        help="Phase 3 评测后抽样 5 条重跑，计算重测信度和系统自验证")
    parser.add_argument("--replay-dir", type=str, default=None,
                        help="从历史 JSON 目录加载对话并运行 Phase 3（跨 session）")
    parser.add_argument("--judge-model", type=str, default=None,
                        help="Phase 3 Judge 模型（不传则用同一模型）")

    args = parser.parse_args()
    _load_dotenv()

    # 跨 session 历史回放模式
    if args.replay_dir:
        from src.simulator.batch_runner import BatchRunner
        from src.loader.case_loader import load_cases
        from src.llm.client import LLMClient

        api_key = _os.getenv("API_KEY", "")
        j_model = args.judge_model or "deepseek-chat"
        judge_client = LLMClient(api_key=api_key, base_url="https://api.deepseek.com",
                                 model=j_model, temperature=0.3)
        all_cases = load_cases()
        runner = BatchRunner(all_cases, eval_client=judge_client)
        runner.replay_and_eval(args.replay_dir)
        import sys
        sys.exit(0)

    case_ids = [int(x.strip()) for x in args.case_id.split(",")]

    for cid in case_ids:
        try:
            run_case(case_id=cid, n_global=args.n_global, parallel=args.parallel,
                     run_eval=args.run_eval, verify_reliability=args.verify_reliability,
                     judge_model=args.judge_model)
        except Exception as e:
            print(f"Case #{cid} 失败: {e}")
