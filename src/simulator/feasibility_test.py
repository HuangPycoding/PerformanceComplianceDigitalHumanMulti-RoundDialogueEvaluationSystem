"""FeasibilityTestRunner — 可行性小规模验证入口

跑通第一个 case 的所有分支对话（参数化画像），
将完整对话内容记录到一份 markdown 文件中。

用法:
    1. 在 .env 中设置 API_KEY=你的key
    2. python -m src.simulator.feasibility_test

    或直接传参:
    from src.simulator.feasibility_test import feasibility_run
    feasibility_run(api_key="sk-xxx", case_index=0)
"""
import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.config import PROJECT_ROOT
from src.llm.client import LLMClient
from src.loader.case_loader import load_cases
from src.loader.complexity import calculate_complexity, compute_max_turns
from src.models.case import Case
from src.models.conversation import Conversation, Turn
from src.simulator.profile_params import (
    lhs_sample,
    translate_vector_to_anchor,
    get_adversarial_strategies,
    extract_branch_constraints,
    subspace_lhs,
    compute_profile_count,
)
from src.simulator.profile_generator import ProfileGenerator
from src.simulator.profile_auditor import ProfileAuditor
from src.simulator.profiles import (
    build_profile_from_vector,
    build_adversarial_instruction_for_vector,
    UserProfile,
)
from src.simulator.runner import DialogueRunner

FEASIBILITY_OUTPUT_DIR = PROJECT_ROOT / "data" / "feasibility_output"


# ============================================================
# 结果数据类
# ============================================================

@dataclass
class BranchDialogueResult:
    """单个分支对话的完整结果"""
    profile_label: str
    profile_type: str = ""
    profile_description: str = ""
    adversarial_strategies: List[str] = field(default_factory=list)
    persona_text: str = ""
    sampled_vector: Optional[List[float]] = None
    verified_vector: Optional[List[float]] = None
    # 对话
    conversation: Optional[Conversation] = None
    status: str = ""
    total_turns: int = 0
    duration_seconds: float = 0.0
    turns: List[Dict[str, Any]] = field(default_factory=list)
    # 审计
    consistency: Dict[str, Any] = field(default_factory=dict)
    # 错误
    error: Optional[str] = None


@dataclass
class FeasibilityReport:
    """一份可行性验证报告"""
    case_id: int
    case_title: str
    complexity_score: float = 0.0
    timestamp: str = ""
    model: str = ""
    # 所有分支对话结果
    branch_results: List[BranchDialogueResult] = field(default_factory=list)
    # 汇总
    success_count: int = 0
    fail_count: int = 0
    total_turns_all: int = 0
    total_duration: float = 0.0


# ============================================================
# FeasibilityTestRunner
# ============================================================

class FeasibilityTestRunner:
    """可行性验证运行器

    为一个 case 跑多个画像分支的对话，输出完整 markdown 报告。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        # 三个角色共用同一模型（可行性验证模式）
        self.gen_client = LLMClient(
            api_key=api_key, base_url=base_url, model=model, temperature=0.7,
        )
        self.sim_client = LLMClient(
            api_key=api_key, base_url=base_url, model=model, temperature=0.7,
        )
        self.asst_client = LLMClient(
            api_key=api_key, base_url=base_url, model=model, temperature=0.0,
        )

    # ================================================================
    # 主入口
    # ================================================================

    def run_case_all_branches(
        self,
        case_index: int = 0,
        max_turns: Optional[int] = None,
        include_parameterized: bool = True,
        n_parameterized: int = 2,
    ) -> FeasibilityReport:
        """对第 case_index 个 case 跑所有画像分支

        Args:
            case_index: case 索引
            max_turns: 每场对话最大轮次，None 时自动计算
            include_parameterized: 是否包含参数化画像
            n_parameterized: 参数化画像数量
        """
        cases = load_cases()
        if case_index >= len(cases):
            raise ValueError(f"case_index={case_index} 超出范围 (共 {len(cases)} 个)")
        case = cases[case_index]

        complexity = calculate_complexity(case)
        if max_turns is None:
            max_turns = compute_max_turns(case)

        report = FeasibilityReport(
            case_id=case.id,
            case_title=case.title,
            complexity_score=complexity,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            model=self.model,
        )

        print(f"{'='*60}")
        print(f"  可行性验证 — Case #{case.id}: {case.title}")
        print(f"{'='*60}")
        print(f"  复杂度: {complexity:.1f}/10")
        print(f"  步骤: {len(case.call_flow)}, 约束: {len(case.constraints)}, "
              f"知识点: {len(case.knowledge_points)}")
        print(f"  API 模型: {self.model}")
        print(f"  max_turns: {max_turns} (自动)")
        print()

        # ---- 收集所有待测画像 ----
        profiles_to_test: List[tuple[str, UserProfile]] = self._collect_profiles(
            case, include_parameterized, n_parameterized,
        )

        print(f"共 {len(profiles_to_test)} 个画像分支待测试\n")

        # ---- 逐个跑对话 ----
        for idx, (label, profile) in enumerate(profiles_to_test):
            print(f"[{idx + 1}/{len(profiles_to_test)}] 运行: {label} ... ", end="", flush=True)

            result = BranchDialogueResult(
                profile_label=label,
                profile_type=profile.type,
                profile_description=profile.effective_description,
                adversarial_strategies=profile.adversarial_strategy,
                persona_text=profile.persona_text,
                sampled_vector=profile.sampled_vector,
                verified_vector=profile.verified_vector,
            )

            try:
                conv = self._run_one_dialogue(case, profile, max_turns)
                result.conversation = conv
                result.status = conv.status
                result.total_turns = conv.total_turns
                result.duration_seconds = conv.duration_seconds

                # 提取对话轮次
                for turn in conv.turns:
                    speaker = "客服" if turn.speaker == "system" else "用户"
                    turn_data = {
                        "turn": turn.turn_number,
                        "speaker": speaker,
                        "content": turn.content,
                    }
                    if turn.parsed_tags:
                        turn_data["parsed_tags"] = turn.parsed_tags
                    result.turns.append(turn_data)

                # Path A 审计
                auditor = ProfileAuditor(client=self.asst_client)
                state_trajectory = []
                for turn in conv.turns:
                    if turn.speaker == "user" and turn.parsed_tags:
                        state_dict = turn.parsed_tags.get("state")
                        if state_dict and isinstance(state_dict, dict):
                            state_trajectory.append(state_dict)
                auditor.audit_path_a(conv, state_trajectory)
                result.consistency = conv.consistency or {}

                print(f"OK ({conv.status}, {conv.total_turns}轮, {conv.duration_seconds:.0f}s)")
                report.success_count += 1

            except Exception as e:
                result.error = f"{type(e).__name__}: {e}"
                result.status = "异常中断"
                print(f"FAIL — {result.error}")
                traceback.print_exc()
                report.fail_count += 1

            report.branch_results.append(result)
            report.total_turns_all += result.total_turns
            report.total_duration += result.duration_seconds

        # ---- 保存 markdown ----
        md_path = self._save_markdown(report, case)
        print(f"\n报告已保存: {md_path}")

        # ---- 打印汇总 ----
        self._print_summary(report)

        return report

    # ================================================================
    # 内部方法
    # ================================================================

    def _collect_profiles(
        self, case: Case, include_parameterized: bool, n_parameterized: int,
    ) -> List[tuple[str, UserProfile]]:
        """收集所有待测画像 — 全部为参数化画像"""
        if not include_parameterized:
            return []
        profiles: List[tuple[str, UserProfile]] = []
        gen = ProfileGenerator(self.gen_client)

        # P1: 全空间 LHS baseline
        n_global = max(n_parameterized, 4)
        global_vectors = lhs_sample(min(n_global, 6), 15)
        for i, v in enumerate(global_vectors):
            label = f"param_global_{i+1}"
            strategies = get_adversarial_strategies(v)
            try:
                profile = gen.generate_with_retry(v, max_retries=2)
                profile.adversarial_strategy = strategies
                profile.adversarial_instruction = build_adversarial_instruction_for_vector(v)
                profiles.append((label, profile))
            except Exception as e:
                print(f"  [警告] 参数化画像生成失败 ({label}): {e}")

        # P2: 分支约束子空间（如有）
        branch_constraints = extract_branch_constraints(case.call_flow)
        if branch_constraints:
            n_branch = max(compute_profile_count(case), 1)
            constrained_vectors = subspace_lhs(n_branch, branch_constraints)
            for i, v in enumerate(constrained_vectors):
                label = f"param_branch_{i+1}"
                strategies = get_adversarial_strategies(v)
                try:
                    profile = gen.generate_with_retry(v, max_retries=2)
                    profile.adversarial_strategy = strategies
                    profile.adversarial_instruction = build_adversarial_instruction_for_vector(v)
                    profiles.append((label, profile))
                except Exception as e:
                    print(f"  [警告] 分支画像生成失败 ({label}): {e}")

        return profiles

    def _run_one_dialogue(
        self, case: Case, profile: UserProfile, max_turns: int,
    ) -> Conversation:
        """跑一场对话"""
        runner = DialogueRunner.create_with_llm(
            case=case,
            profile=profile,
            assistant_client=self.asst_client,
            simulator_client=self.sim_client,
            use_raw_prompt=True,
        )
        conv = runner.run(max_turns=max_turns)
        # 附加元数据
        conv.sampled_vector = profile.sampled_vector
        conv.verified_vector = profile.verified_vector
        conv.adversarial_strategies = profile.adversarial_strategy or []
        conv.complexity_score = calculate_complexity(case)
        return conv

    # ================================================================
    # Markdown 输出
    # ================================================================

    def _save_markdown(self, report: FeasibilityReport, case: Case) -> Path:
        """将报告保存为 markdown 文件"""
        os.makedirs(str(FEASIBILITY_OUTPUT_DIR), exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feasibility_case{case.id}_{timestamp}.md"
        filepath = FEASIBILITY_OUTPUT_DIR / filename

        lines = self._build_markdown(report, case)

        with open(str(filepath), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return filepath

    def _build_markdown(self, report: FeasibilityReport, case: Case) -> List[str]:
        """构建 markdown 内容"""
        L = []  # lines accumulator

        # ---- 标题 ----
        L.append(f"# 可行性验证报告 — Case #{case.id}: {case.title}")
        L.append("")

        # ---- 测试概览 ----
        L.append("## 测试概览")
        L.append("")
        L.append(f"| 项目 | 值 |")
        L.append(f"|------|-----|")
        L.append(f"| 测试时间 | {report.timestamp} |")
        L.append(f"| API 模型 | `{report.model}` |")
        L.append(f"| 画像总数 | {len(report.branch_results)} |")
        L.append(f"| 成功 | {report.success_count} |")
        L.append(f"| 失败 | {report.fail_count} |")
        L.append(f"| 总轮次 | {report.total_turns_all} |")
        L.append(f"| 总耗时 | {report.total_duration:.1f}s |")
        if report.success_count > 0:
            avg_turns = report.total_turns_all / max(report.success_count, 1)
            L.append(f"| 平均轮次 | {avg_turns:.1f} |")
        L.append("")

        # ---- Case 信息 ----
        L.append("## Case 信息")
        L.append("")
        L.append(f"- **复杂度评分**: {report.complexity_score:.1f}/10")
        L.append(f"- **流程步骤**: {len(case.call_flow)}")
        L.append(f"- **约束条件**: {len(case.constraints)}")
        L.append(f"- **知识点**: {len(case.knowledge_points)}")
        L.append("")

        # 角色 & 任务
        if case.role:
            L.append(f"### 客服角色")
            L.append("")
            L.append(case.role)
            L.append("")
        if case.task:
            L.append(f"### 任务目标")
            L.append("")
            L.append(case.task)
            L.append("")
        if case.opening_line:
            L.append(f"### 开场白")
            L.append("")
            L.append(f"> {case.opening_line}")
            L.append("")

        # 通话流程
        if case.call_flow:
            L.append("### 通话流程")
            L.append("")
            for step in case.call_flow:
                L.append(f"**Step {step.step_number}: {step.title}**")
                if step.description:
                    L.append(f"- {step.description}")
                if step.sub_steps:
                    for ss in step.sub_steps:
                        L.append(f"  - {ss}")
                if step.branching:
                    for b in step.branching:
                        L.append(f"  - 分支: 如果 {b.condition} → {b.action}")
                if step.reference_script:
                    L.append(f"  - 参考话术: {step.reference_script}")
                L.append("")

        # 约束
        if case.constraints:
            L.append("### 约束条件")
            L.append("")
            for c in case.constraints:
                L.append(f"- [{c.type}] {c.description}")
            L.append("")

        # 知识点
        if case.knowledge_points:
            L.append("### 知识点")
            L.append("")
            for kp in case.knowledge_points:
                L.append(f"- **{kp.topic}**: {kp.content}")
            L.append("")

        L.append("---")
        L.append("")

        # ---- 每个分支对话 ----
        for idx, br in enumerate(report.branch_results):
            L.append(f"## 对话 {idx + 1}: {br.profile_label}")
            L.append("")

            if br.error:
                L.append(f"**状态**: ❌ 异常中断 — `{br.error}`")
                L.append("")
                L.append("---")
                L.append("")
                continue

            # 画像信息
            L.append("### 画像信息")
            L.append("")
            L.append(f"| 属性 | 值 |")
            L.append(f"|------|-----|")
            L.append(f"| 画像标签 | `{br.profile_label}` |")
            L.append(f"| 画像类型 | {br.profile_type} |")
            if br.adversarial_strategies:
                L.append(f"| 对抗策略 | {', '.join(br.adversarial_strategies)} |")
            L.append(f"| 对话状态 | {br.status} |")
            L.append(f"| 总轮次 | {br.total_turns} |")
            L.append(f"| 耗时 | {br.duration_seconds:.1f}s |")

            # 一致性
            if br.consistency:
                tier = br.consistency.get("tier", "N/A")
                d_sv = br.consistency.get("d_sv", "N/A")
                L.append(f"| 一致性 | tier={tier}, d_sv={d_sv} |")

            L.append("")

            # 画像描述
            if br.persona_text:
                L.append("<details>")
                L.append("<summary>画像文本（点击展开）</summary>")
                L.append("")
                L.append("```")
                L.append(br.persona_text)
                L.append("```")
                L.append("")
                L.append("</details>")
                L.append("")
            elif br.profile_description:
                L.append("<details>")
                L.append("<summary>画像描述（点击展开）</summary>")
                L.append("")
                L.append("```")
                L.append(br.profile_description[:500])
                if len(br.profile_description) > 500:
                    L.append("... (截断)")
                L.append("```")
                L.append("")
                L.append("</details>")
                L.append("")

            # 15D 向量
            if br.sampled_vector:
                L.append("<details>")
                L.append("<summary>15D 参数向量（点击展开）</summary>")
                L.append("")
                L.append("| 维度 | S (采样) | V (验证) |")
                L.append("|------|----------|----------|")
                from src.simulator.profile_params import DIMENSIONS
                dim_names = [d.name for d in DIMENSIONS]
                for di in range(15):
                    sv = br.sampled_vector[di] if di < len(br.sampled_vector) else "?"
                    vv = br.verified_vector[di] if br.verified_vector and di < len(br.verified_vector) else "-"
                    L.append(f"| {dim_names[di]} | {sv:.3f} | {vv} |")
                L.append("")
                L.append("</details>")
                L.append("")

            # 对话记录
            L.append("### 对话记录")
            L.append("")
            L.append("| # | 角色 | 内容 | 状态标签 |")
            L.append("|---|------|------|----------|")
            for turn in br.turns:
                content = turn["content"].replace("\n", " ").replace("|", "\\|")
                if len(content) > 200:
                    content = content[:200] + "..."
                tags = ""
                if turn.get("parsed_tags"):
                    state = turn["parsed_tags"].get("state", {})
                    if isinstance(state, dict):
                        emotion = state.get("emotion", "")
                        if emotion:
                            tags = f"情绪:{emotion}"
                L.append(f"| {turn['turn']} | {turn['speaker']} | {content} | {tags} |")
            L.append("")

            # 详细对话（折叠）
            L.append("<details>")
            L.append("<summary>完整对话文本（点击展开）</summary>")
            L.append("")
            for turn in br.turns:
                L.append(f"**[{turn['turn']}] {turn['speaker']}**")
                L.append("")
                L.append(turn["content"])
                L.append("")
                if turn.get("parsed_tags"):
                    L.append("```json")
                    L.append(json.dumps(turn["parsed_tags"], ensure_ascii=False, indent=2))
                    L.append("```")
                    L.append("")
            L.append("</details>")
            L.append("")

            L.append("---")
            L.append("")

        # ---- 附录: 原始 JSON ----
        L.append("## 附录: 原始对话数据 (JSON)")
        L.append("")
        L.append("<details>")
        L.append("<summary>点击展开 JSON</summary>")
        L.append("")
        L.append("```json")
        json_data = {
            "case_id": report.case_id,
            "case_title": report.case_title,
            "complexity_score": report.complexity_score,
            "timestamp": report.timestamp,
            "model": report.model,
            "success_count": report.success_count,
            "fail_count": report.fail_count,
            "total_turns": report.total_turns_all,
            "total_duration": report.total_duration,
            "branches": [
                {
                    "profile_label": br.profile_label,
                    "profile_type": br.profile_type,
                    "status": br.status,
                    "total_turns": br.total_turns,
                    "duration_seconds": br.duration_seconds,
                    "adversarial_strategies": br.adversarial_strategies,
                    "sampled_vector": br.sampled_vector,
                    "verified_vector": br.verified_vector,
                    "consistency": br.consistency,
                    "error": br.error,
                    "turns": [
                        {
                            "turn": t["turn"],
                            "speaker": t["speaker"],
                            "content": t["content"],
                            "parsed_tags": t.get("parsed_tags"),
                        }
                        for t in br.turns
                    ],
                }
                for br in report.branch_results
            ],
        }
        L.append(json.dumps(json_data, ensure_ascii=False, indent=2))
        L.append("```")
        L.append("")
        L.append("</details>")
        L.append("")

        return L

    # ================================================================
    # 汇总
    # ================================================================

    def _print_summary(self, report: FeasibilityReport):
        """打印控制台汇总"""
        print(f"\n{'='*60}")
        print(f"  可行性验证完成")
        print(f"{'='*60}")
        print(f"  Case #{report.case_id}: {report.case_title}")
        print(f"  画像数: {len(report.branch_results)}")
        print(f"  成功: {report.success_count} / 失败: {report.fail_count}")
        print(f"  总轮次: {report.total_turns_all}")
        print(f"  总耗时: {report.total_duration:.1f}s")
        print()

        for br in report.branch_results:
            status_icon = "OK" if not br.error else "FAIL"
            print(f"  [{status_icon}] {br.profile_label:30s} | {br.status:8s} | {br.total_turns:2d}轮 | {br.duration_seconds:5.1f}s")
            if br.error:
                print(f"       Error: {br.error}")

        print(f"\n报告文件: {FEASIBILITY_OUTPUT_DIR}/")
        print(f"{'='*60}")


# ============================================================
# 快捷函数
# ============================================================

def feasibility_run(
    api_key: str,
    case_index: int = 0,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    max_turns: Optional[int] = None,
    include_parameterized: bool = True,
    n_parameterized: int = 2,
) -> FeasibilityReport:
    """一行跑通可行性验证

    Args:
        api_key: API 密钥
        case_index: case 索引，默认 0（第一个 case）
        base_url: API 地址
        model: 模型名称
        max_turns: 每场对话最大轮次
        include_parameterized: 是否包含参数化画像
        n_parameterized: 参数化画像数量
    """
    runner = FeasibilityTestRunner(
        api_key=api_key, base_url=base_url, model=model,
    )
    return runner.run_case_all_branches(
        case_index=case_index,
        max_turns=max_turns,
        include_parameterized=include_parameterized,
        n_parameterized=n_parameterized,
    )


# ============================================================
# CLI 入口
# ============================================================

if __name__ == "__main__":
    load_dotenv()

    # 优先从环境变量取，兼容 API_KEY / MODEL_API_KEY 两种命名
    api_key = os.getenv("API_KEY", "") or os.getenv("MODEL_API_KEY", "")
    base_url = os.getenv("BASE_URL", "https://api.deepseek.com")
    model = os.getenv("MODEL", "deepseek-chat")

    if not api_key:
        print("=" * 60)
        print("  未检测到 API_KEY")
        print("  请在 .env 文件中设置: API_KEY=你的密钥")
        print("  或直接调用: feasibility_run(api_key='你的密钥')")
        print("=" * 60)
        exit(1)

    print(f"API Key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    print()

    feasibility_run(
        api_key=api_key,
        case_index=0,
        base_url=base_url,
        model=model,
        max_turns=None,
        include_parameterized=True,
        n_parameterized=2,
    )
