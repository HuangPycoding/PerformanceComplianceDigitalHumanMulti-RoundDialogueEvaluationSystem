"""QuickTestRunner — 轻量测试入口：直接传 api_key，跑通一个 case 全流程"""
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
)
from src.simulator.profile_generator import ProfileGenerator
from src.simulator.profile_auditor import ProfileAuditor
from src.simulator.profiles import (
    build_profile_from_vector,
    build_adversarial_instruction_for_vector,
    UserProfile,
)
from src.simulator.runner import DialogueRunner

TEST_OUTPUT_DIR = PROJECT_ROOT / "data" / "test_output"


@dataclass
class QuickTestResult:
    """一次轻量测试的完整结果"""
    case_id: int
    case_title: str = ""
    complexity_score: float = 0.0
    # 画像
    sampled_vector: Optional[List[float]] = None
    verified_vector: Optional[List[float]] = None
    persona_text: str = ""
    adversarial_strategies: List[str] = field(default_factory=list)
    # 对话
    conversation: Optional[Conversation] = None
    conversation_status: str = ""
    total_turns: int = 0
    duration_seconds: float = 0.0
    # 审计
    consistency: Dict[str, Any] = field(default_factory=dict)
    # 原始文本
    turns_text: List[Dict[str, Any]] = field(default_factory=list)
    # 错误
    error: Optional[str] = None


class QuickTestRunner:
    """轻量测试运行器：直接传参数，不依赖 .env / ModelManager

    用法:
        runner = QuickTestRunner(
            api_key="sk-xxx",
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
        )
        result = runner.run_one_case(case_index=0, max_turns=8)
        runner.print_report(result)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        # 三个角色共用同一模型（轻量调试模式）
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
    # 主入口：跑一个 case 全流程
    # ================================================================

    def run_one_case(
        self, case_index: int = 0, max_turns: Optional[int] = None,
        use_llm_end: bool = True,
    ) -> QuickTestResult:
        """挑第 case_index 个 case，走完 画像生成 → 对话 → 审计 全流程

        max_turns 为 None 时根据 case 复杂度自动计算。
        """
        result = QuickTestResult(case_id=-1)

        try:
            # ---- Step 1: 加载 case ----
            cases = load_cases()
            if case_index >= len(cases):
                result.error = f"case_index={case_index} 超出范围 (共 {len(cases)} 个)"
                return result
            case = cases[case_index]

            result.case_id = case.id
            result.case_title = case.title
            result.complexity_score = calculate_complexity(case)

            if max_turns is None:
                max_turns = compute_max_turns(case)

            print(f"Case #{case.id}: {case.title}")
            print(f"  复杂度: {result.complexity_score:.1f}/10")
            print(f"  流程步骤: {len(case.call_flow)}")
            print(f"  约束: {len(case.constraints)}, 知识点: {len(case.knowledge_points)}")
            print(f"  max_turns: {max_turns} (自动)")

            # ---- Step 2: 生成画像 ----
            print("\n--- Phase 0: 画像生成 ---")
            profile = self._generate_one_profile(case)
            result.sampled_vector = profile.sampled_vector
            result.verified_vector = profile.verified_vector
            result.persona_text = profile.persona_text
            result.adversarial_strategies = profile.adversarial_strategy or []

            print(f"  画像标签: {profile.label}")
            print(f"  对抗策略: {result.adversarial_strategies or '无'}")
            print(f"  画像文本 ({len(profile.persona_text)}字):")
            print(f"  ---")
            for line in profile.persona_text.split("\n")[:8]:
                print(f"  {line}")
            if len(profile.persona_text.split("\n")) > 8:
                print(f"  ... (截断)")
            print(f"  ---")

            # ---- Step 3: 跑对话 ----
            print(f"\n--- Phase 1: 对话模拟 (max_turns={max_turns}) ---")
            conv = self._run_dialogue(case, profile, max_turns, use_llm_end)
            result.conversation = conv
            result.conversation_status = conv.status
            result.total_turns = conv.total_turns
            result.duration_seconds = conv.duration_seconds

            for turn in conv.turns:
                speaker = "客服" if turn.speaker == "system" else "用户"
                tag_info = ""
                if turn.speaker == "user" and turn.parsed_tags:
                    state = turn.parsed_tags.get("state", {})
                    if isinstance(state, dict):
                        tag_info = f" [情绪:{state.get('emotion','?')}]"
                print(f"  [{turn.turn_number}] {speaker}: {turn.content[:80]}{'...' if len(turn.content)>80 else ''}{tag_info}")
                result.turns_text.append({
                    "turn": turn.turn_number,
                    "speaker": speaker,
                    "content": turn.content,
                    "parsed_tags": turn.parsed_tags if turn.parsed_tags else None,
                })

            # ---- Step 4: 审计 ----
            print(f"\n--- Phase 2: 审计 ---")
            auditor = ProfileAuditor(client=self.asst_client)

            # Path A: 从 state 标签提取轨迹
            state_trajectory = []
            for turn in conv.turns:
                if turn.speaker == "user" and turn.parsed_tags:
                    state_dict = turn.parsed_tags.get("state")
                    if state_dict and isinstance(state_dict, dict):
                        state_trajectory.append(state_dict)
            auditor.audit_path_a(conv, state_trajectory)

            # Path B: LLM 行为审计
            auditor.audit_path_b(conv)

            result.consistency = conv.consistency or {}
            print(f"  一致性 tier: {result.consistency.get('tier', 'N/A')}")
            print(f"  d_sv: {result.consistency.get('d_sv', 'N/A')}")
            print(f"  d_sa: {result.consistency.get('d_sa', 'N/A')}")
            if conv.audited_vector:
                print(f"  audited_vector[:3]: {[round(v,3) for v in conv.audited_vector[:3]]}...")

            # ---- Step 5: 保存 ----
            self._save_result(result)

            print(f"\n{'='*50}")
            print(f"测试完成: case#{result.case_id} | 状态={result.conversation_status} | {result.total_turns}轮 | 耗时{result.duration_seconds:.0f}s")

        except Exception as e:
            import traceback
            result.error = f"{type(e).__name__}: {e}"
            print(f"\n[异常] {result.error}")
            traceback.print_exc()

        return result

    # ================================================================
    # 内部方法
    # ================================================================

    def _generate_one_profile(self, case: Case) -> UserProfile:
        """为单个 case 生成 1 个参数化画像"""
        gen = ProfileGenerator(self.gen_client)

        # LHS 采样一个 15D 向量
        vectors = lhs_sample(1, 15)
        vector = vectors[0]

        # 分支约束（有则用子空间 LHS）
        branch_constraints = extract_branch_constraints(case.call_flow)
        if branch_constraints:
            from src.simulator.profile_params import subspace_lhs
            constrained = subspace_lhs(1, branch_constraints)
            vector = constrained[0]

        # 对抗策略自动挂钩
        strategies = get_adversarial_strategies(vector)

        # 生成画像文本（含自检回路）
        profile = gen.generate_with_retry(vector, max_retries=3)
        profile.adversarial_strategy = strategies
        profile.adversarial_instruction = build_adversarial_instruction_for_vector(vector)

        return profile

    def _run_dialogue(
        self, case: Case, profile: UserProfile, max_turns: int,
        use_llm_end: bool = True,
    ) -> Conversation:
        """跑一场对话"""
        runner = DialogueRunner.create_with_llm(
            case=case,
            profile=profile,
            assistant_client=self.asst_client,
            simulator_client=self.sim_client,
            use_raw_prompt=True,
            use_llm_end_detection=use_llm_end,
        )
        conv = runner.run(max_turns=max_turns)
        # 附加元数据
        conv.sampled_vector = profile.sampled_vector
        conv.verified_vector = profile.verified_vector
        conv.adversarial_strategies = profile.adversarial_strategy or []
        conv.complexity_score = calculate_complexity(case)
        return conv

    def _save_result(self, result: QuickTestResult):
        """保存测试结果到 JSON 文件"""
        os.makedirs(str(TEST_OUTPUT_DIR), exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_case{result.case_id}_{timestamp}.json"
        filepath = TEST_OUTPUT_DIR / filename

        data = {
            "case_id": result.case_id,
            "case_title": result.case_title,
            "complexity_score": result.complexity_score,
            "sampled_vector": result.sampled_vector,
            "verified_vector": result.verified_vector,
            "persona_text": result.persona_text,
            "adversarial_strategies": result.adversarial_strategies,
            "conversation_status": result.conversation_status,
            "total_turns": result.total_turns,
            "duration_seconds": result.duration_seconds,
            "consistency": result.consistency,
            "turns": result.turns_text,
            "error": result.error,
        }

        with open(str(filepath), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"\n结果已保存: {filepath}")

    # ================================================================
    # 报告
    # ================================================================

    def print_report(self, result: QuickTestResult):
        """打印可读的测试报告"""
        print("\n" + "=" * 60)
        print(f"  轻量测试报告")
        print("=" * 60)
        print(f"  Case:       #{result.case_id} — {result.case_title}")
        print(f"  复杂度:     {result.complexity_score:.1f}/10")
        print(f"  画像标签:   {len(result.persona_text)} 字")
        print(f"  对抗策略:   {result.adversarial_strategies or '无'}")
        print(f"  对话状态:   {result.conversation_status}")
        print(f"  总轮次:     {result.total_turns}")
        print(f"  耗时:       {result.duration_seconds:.1f}s")
        if result.consistency:
            print(f"  一致性:     tier={result.consistency.get('tier','?')} "
                  f"d_sv={result.consistency.get('d_sv','?')} "
                  f"d_sa={result.consistency.get('d_sa','?')}")
        if result.error:
            print(f"  错误:       {result.error}")
        print("=" * 60)


# ================================================================
# 快捷函数
# ================================================================

def quick_run(
    api_key: str,
    case_index: int = 0,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    max_turns: Optional[int] = None,
) -> QuickTestResult:
    """一行跑通全流程"""
    runner = QuickTestRunner(api_key=api_key, base_url=base_url, model=model)
    result = runner.run_one_case(case_index=case_index, max_turns=max_turns)
    runner.print_report(result)
    return result


if __name__ == "__main__":
    import os
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        print("请设置 API_KEY 环境变量或在 .env 中配置")
        print("用法: python -m src.simulator.quick_test")
        exit(1)

    quick_run(api_key=api_key, case_index=0)
