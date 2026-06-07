"""Tier 1 规则层 + Tier 1.5 信号提取 + CONSTRAINT 分流 + 信号→清单映射"""
import re
from typing import Any, Dict, List, Optional, Tuple

from src.models.case import Case
from src.models.conversation import Conversation, Turn
from src.simulator.simulator import should_end_conversation


# ============================================================
# Tier 1: 11 个规则指标
# ============================================================

def compute_tier1_metrics(conv: Conversation, case: Optional[Case] = None) -> Dict[str, Any]:
    """计算全部 11 个 Tier 1 规则指标"""
    metrics = {}
    if not conv.turns:
        return {
            "turns_ratio": 0.0, "stuck_count": 0, "stuck_ratio": 0.0,
            "should_end_mismatch": 0, "repetition_score": 0.0,
            "word_count_violations": 0, "forbidden_word_hits": 0,
            "step_order_ok": True, "model_breakdown_flag": False,
            "user_repeat_rate": 0.0, "hangup_detected": {"detected": False},
            "branch_coverage": {"expected": [], "triggered": [], "untriggered": [], "coverage_ratio": 0.0},
        }
    total = len(conv.turns)
    system_turns = [t for t in conv.turns if t.speaker == "system"]

    # 1. turns_ratio
    expected_min = 4  # 默认最小期望轮数
    if case and case.call_flow:
        expected_min = max(4, len(case.call_flow) * 2)
    metrics["turns_ratio"] = total / expected_min if expected_min > 0 else 1.0

    # 2. stuck_count
    stuck_count = 0
    for t in conv.turns:
        cq = t.parsed_tags.get("conversation_quality", {})
        if isinstance(cq, dict) and cq.get("是否卡死") == "是":
            stuck_count += 1
    metrics["stuck_count"] = stuck_count
    metrics["stuck_ratio"] = stuck_count / total if total > 0 else 0.0

    # 4. should_end_mismatch: 用户表示结束后紧接着的 system 回复应引导结束
    mismatch = 0
    for i in range(len(conv.turns) - 1):
        if conv.turns[i].speaker != "user":
            continue
        se = conv.turns[i].parsed_tags.get("should_end", {})
        is_end = False
        if isinstance(se, dict):
            is_end = se.get("本轮是否想结束对话") == "是"
        elif isinstance(se, str):
            is_end = "是" in se
        if is_end and conv.turns[i + 1].speaker == "system":
            mismatch += 1
    metrics["should_end_mismatch"] = mismatch

    # 5. repetition_score
    metrics["repetition_score"] = _compute_repetition_score(system_turns)

    # 6. word_count_violations
    violations = 0
    if case and case.constraints:
        for t in system_turns:
            for c in case.constraints:
                if c.checkable_by_rule and c.type == "word_limit" and c.rule_pattern:
                    try:
                        limit = int(c.rule_pattern)
                        if len(t.content) > limit:
                            violations += 1
                    except ValueError:
                        pass
    metrics["word_count_violations"] = violations

    # 7. forbidden_word_hits
    forbidden_hits = 0
    if case and case.constraints:
        for t in system_turns:
            for c in case.constraints:
                if c.checkable_by_rule and c.type == "forbidden_word" and (c.rule_pattern or ""):
                    try:
                        if re.search((c.rule_pattern or ""), t.content):
                            forbidden_hits += 1
                    except re.error:
                        pass
    metrics["forbidden_word_hits"] = forbidden_hits

    # 8. step_order_ok (简单状态机 vs 预期顺序)
    metrics["step_order_ok"] = _check_step_order(conv, case)

    # 9. model_breakdown_flag
    metrics["model_breakdown_flag"] = conv.model_breakdown_count > 0

    # 10. user_repeat_rate
    metrics["user_repeat_rate"] = _compute_user_repeat_rate(conv)

    # 11. hangup_detected
    metrics["hangup_detected"] = _detect_hangup(conv)

    # V1: 分支覆盖
    metrics["branch_coverage"] = compute_branch_coverage(conv, case)

    return metrics


def _compute_repetition_score(turns: List[Turn]) -> float:
    """相邻系统轮 n-gram 重叠率"""
    if len(turns) < 2:
        return 0.0
    scores = []
    for i in range(len(turns) - 1):
        a = set(_ngrams(turns[i].content, 2))
        b = set(_ngrams(turns[i + 1].content, 2))
        if len(a | b) > 0:
            scores.append(len(a & b) / len(a | b))
    return sum(scores) / len(scores) if scores else 0.0


def _ngrams(text: str, n: int) -> List[str]:
    """提取 n-gram"""
    chars = list(text)
    return ["".join(chars[i:i + n]) for i in range(len(chars) - n + 1)]


def _compute_user_repeat_rate(conv: Conversation) -> float:
    """用户连续两轮表述相似度 > 0.7 的轮次占比"""
    user_turns = [t for t in conv.turns if t.speaker == "user"]
    if len(user_turns) < 2:
        return 0.0
    repeat_count = 0
    for i in range(len(user_turns) - 1):
        similarity = _text_similarity(user_turns[i].content, user_turns[i + 1].content)
        if similarity > 0.7:
            repeat_count += 1
    return repeat_count / (len(user_turns) - 1)


def _text_similarity(a: str, b: str) -> float:
    """简单字符级 Jaccard 相似度"""
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    return len(set_a & set_b) / len(set_a | set_b)


def _detect_hangup(conv: Conversation) -> Dict[str, Any]:
    """检测用户挂断事件并提取上下文"""
    user_turns = [(i, t) for i, t in enumerate(conv.turns) if t.speaker == "user"]
    for idx, turn in reversed(user_turns):
        if should_end_conversation(turn.content):
            # 提取挂断上下文
            progress = _estimate_task_progress(conv, turn.turn_number)
            sentiment = _get_sentiment_at_turn(turn)
            return {
                "detected": True,
                "hangup_turn": turn.turn_number,
                "task_progress": progress,
                "hangup_sentiment": sentiment,
                "hangup_phrase": turn.content.strip(),
            }
    return {"detected": False}


def _extract_facts_from_memory(memory) -> List[str]:
    """从 memory 中提取关键事实列表，兼容 str（当前生产格式）和 dict（未来格式）"""
    if not memory:
        return []
    if isinstance(memory, dict):
        facts = memory.get("关键事实", "")
        if isinstance(facts, str) and facts.strip():
            return [l.strip("- ") for l in facts.split("\n") if l.strip("- ").strip()]
        return []
    if isinstance(memory, str):
        facts = []
        in_facts_section = False
        for line in memory.split("\n"):
            stripped = line.strip()
            if stripped.startswith("关键事实"):
                in_facts_section = True
                continue
            if in_facts_section:
                if "进展追踪" in stripped or "上次决策" in stripped:
                    break
                if stripped.startswith("- "):
                    facts.append(stripped[2:])
        return facts
    return []


def _estimate_task_progress(conv: Conversation, turn_number: int) -> float:
    """估算挂断时的任务进度（基于 memory 标签中关键事实的积累）"""
    facts_collected = 0
    for t in conv.turns:
        if t.turn_number > turn_number:
            break
        memory = t.parsed_tags.get("memory")
        facts = _extract_facts_from_memory(memory)
        facts_collected = max(facts_collected, len(facts))
    return max(0.0, min(1.0, facts_collected / 5.0))


def _get_sentiment_at_turn(turn: Turn) -> str:
    """获取某轮的 sentiment 状态"""
    state = turn.parsed_tags.get("state", {})
    if isinstance(state, dict):
        emotion = state.get("emotion", "")
        if any(w in str(emotion) for w in ["生气", "愤怒", "不满", "烦躁", "失望"]):
            return "负面"
        elif any(w in str(emotion) for w in ["满意", "高兴", "感谢", "开心"]):
            return "正面"
    return "中性"


def _check_step_order(conv: Conversation, case: Optional[Case]) -> bool:
    """检查对话是否遗漏或颠倒了 Case 定义的前置步骤"""
    if not case or not case.call_flow or len(case.call_flow) < 2:
        return True

    # 提取步骤关键词（优先取前 4 字，若重复则用完整 title 避免关键词冲突）
    step_keywords = []
    seen_kws = set()
    for step in case.call_flow:
        if not step.title:
            continue
        kw = step.title[:4]
        if kw in seen_kws:
            kw = step.title  # 重复时用完整 title
        seen_kws.add(kw)
        step_keywords.append(kw)

    if len(step_keywords) < 2:
        return True

    # 在系统回复中检测关键词出现顺序
    system_turns = [t for t in conv.turns if t.speaker == "system"]
    all_system_text = " ".join(t.content for t in system_turns)

    # 检查每对相邻步骤：前一步应在后一步之前出现（允许部分乱序）
    positions = []
    for kw in step_keywords:
        pos = all_system_text.find(kw)
        positions.append(pos if pos >= 0 else float('inf'))

    # 统计逆序对
    inversions = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if positions[i] > positions[j] and positions[i] != float('inf') and positions[j] != float('inf'):
                inversions += 1

    # 超过一半的步骤对逆序 → 步骤顺序有问题
    max_pairs = len(positions) * (len(positions) - 1) / 2
    return max_pairs == 0 or inversions / max_pairs <= 0.5


# ============================================================
# CONSTRAINT 分流
# ============================================================

def classify_constraints(constraints) -> Tuple[List, List]:
    """将约束分为规则可检和需要 LLM 核查两类"""
    rule_checkable = []
    llm_checkable = []
    for c in constraints:
        if getattr(c, "checkable_by_rule", False):
            rule_checkable.append(c)
        else:
            llm_checkable.append(c)
    return rule_checkable, llm_checkable


def check_rule_constraints(conv: Conversation, rule_constraints) -> List[str]:
    """规则检测约束，返回违规列表"""
    issues = []
    system_turns = [t for t in conv.turns if t.speaker == "system"]
    for c in rule_constraints:
        ctype = getattr(c, "type", "")
        pattern = getattr(c, "rule_pattern", "") or ""
        desc = getattr(c, "description", "")
        if ctype == "forbidden_word" and pattern:
            for t in system_turns:
                try:
                    if re.search(pattern, t.content):
                        issues.append(f"T{t.turn_number}: 触发禁止词 '{pattern}' — {desc}")
                except re.error:
                    pass
        elif ctype == "word_limit" and pattern:
            try:
                limit = int(pattern)
                for t in system_turns:
                    if len(t.content) > limit:
                        issues.append(f"T{t.turn_number}: 字数 {len(t.content)} 超限 {limit} — {desc}")
            except ValueError:
                pass
    return issues


# ============================================================
# Tier 1.5: 7 个 Turn 级信号提取
# ============================================================

def extract_turn_signals(conv: Conversation) -> Dict[str, Any]:
    """从 parsed_tags 提取 7 个 Turn 级信号"""
    signals = {}

    # 1. 满意度轨迹
    satisfaction = []
    for t in conv.turns:
        mb = t.parsed_tags.get("model_behavior", {})
        if isinstance(mb, dict):
            rating = mb.get("用户评价", "")
            if "不满" in str(rating):
                satisfaction.append({"turn": t.turn_number, "value": "不满意"})
            elif "满意" in str(rating):
                satisfaction.append({"turn": t.turn_number, "value": "满意"})
            else:
                satisfaction.append({"turn": t.turn_number, "value": "中性"})
    signals["satisfaction_trajectory"] = satisfaction

    # 2. 卡死/不自然
    stuck_turns = []
    for t in conv.turns:
        cq = t.parsed_tags.get("conversation_quality", {})
        if isinstance(cq, dict) and cq.get("是否卡死") == "是":
            stuck_turns.append(t.turn_number)
    signals["stuck_turns"] = stuck_turns

    # 3. 结束意愿
    end_turns = []
    for t in conv.turns:
        se = t.parsed_tags.get("should_end", {})
        is_end = False
        if isinstance(se, dict):
            is_end = se.get("本轮是否想结束对话") == "是"
        elif isinstance(se, str):
            is_end = "是" in se
        if is_end:
            end_turns.append(t.turn_number)
    signals["should_end_turns"] = end_turns

    # 4. 情绪曲线
    emotions = []
    for t in conv.turns:
        state = t.parsed_tags.get("state", {})
        if isinstance(state, dict):
            raw_intensity = state.get("emotion_intensity", 0.5)
            try:
                intensity = float(raw_intensity)
            except (ValueError, TypeError):
                intensity = 0.5
            emotions.append({
                "turn": t.turn_number,
                "emotion": state.get("emotion", ""),
                "intensity": intensity,
            })
    signals["emotion_curve"] = emotions

    # 5. 上下文记忆
    memory_facts = []
    for t in conv.turns:
        facts = _extract_facts_from_memory(t.parsed_tags.get("memory"))
        if facts:
            memory_facts.append({"turn": t.turn_number, "facts": "; ".join(facts)})
    signals["memory_facts"] = memory_facts

    # 6. 态度转变事件
    attitude_changes = []
    for t in conv.turns:
        mb = t.parsed_tags.get("model_behavior", {})
        if isinstance(mb, dict) and mb.get("是否改变态度") == "是":
            attitude_changes.append({
                "turn": t.turn_number,
                "detail": mb.get("是否改变态度", ""),
            })
    signals["attitude_changes"] = attitude_changes

    # 7. 信息采集进度
    signals["info_collection_progress"] = _compute_info_progress(conv)

    return signals


def _compute_info_progress(conv: Conversation) -> Dict[str, Any]:
    """从 memory 标签估算信息采集进度"""
    facts_by_turn = {}
    collected_keys = set()
    for t in conv.turns:
        facts = _extract_facts_from_memory(t.parsed_tags.get("memory"))
        for f in facts:
            collected_keys.add(f[:20])  # 用前20字符做key去重
        facts_by_turn[t.turn_number] = len(collected_keys)

    max_facts = max(facts_by_turn.values()) if facts_by_turn else 0
    return {
        "per_turn": facts_by_turn,
        "total_collected": max_facts,
        "stagnation_turn": _find_stagnation(facts_by_turn),
    }


def _find_stagnation(facts_by_turn: Dict[int, int]) -> Optional[int]:
    """找到信息采集停滞的起始轮次（连续3个值不变=停滞）"""
    turns = sorted(facts_by_turn.keys())
    if len(turns) < 3:
        return None
    values = [facts_by_turn[t] for t in turns]
    for i in range(len(values) - 2):
        if values[i] == values[i + 1] == values[i + 2] and values[i] > 0:
            return turns[i]
    return None


# ============================================================
# 信号 → 清单上下文格式化
# ============================================================

def format_signal_context(signals: Dict[str, Any], tier1: Dict[str, Any], dimension: str, conv=None) -> str:
    """将信号格式化为注入 Judge prompt 的 [Simulator 信号上下文] 段落"""
    parts = []

    if dimension in ("TASK_COMPLETION", "EFFICIENCY"):
        # 满意度轨迹
        traj = signals.get("satisfaction_trajectory", [])
        if traj:
            summary = " → ".join(f"T{s['turn']}{s['value']}" for s in traj)
            parts.append(f"满意度轨迹: {summary}")

        # 态度转变
        changes = signals.get("attitude_changes", [])
        if changes:
            for c in changes:
                parts.append(f"态度转变: 第{c['turn']}轮用户态度发生变化")

        # 信息采集进度
        prog = signals.get("info_collection_progress", {})
        if prog.get("stagnation_turn"):
            parts.append(f"信息采集停滞: 第{prog['stagnation_turn']}轮起无新增信息")

        # 挂断上下文
        hangup = tier1.get("hangup_detected", {})
        if hangup.get("detected"):
            parts.append(
                f"挂断事件: 第{hangup['hangup_turn']}轮用户主动挂断"
                f"（进度={hangup.get('task_progress', 0):.0%}，"
                f"情绪={hangup.get('hangup_sentiment', '中性')}）"
                f" 原文: \"{hangup.get('hangup_phrase', '')}\""
            )

        # 用户重复率
        repeat_rate = tier1.get("user_repeat_rate", 0)
        if repeat_rate > 0.3:
            parts.append(f"用户重复率: {repeat_rate:.0%}（说明客服未有效解决问题）")

    if dimension == "SENTIMENT":
        emotions = signals.get("emotion_curve", [])
        if emotions:
            key_events = [e for e in emotions if e.get("intensity", 0) > 0.6]
            if key_events:
                parts.append(f"情绪关键事件: " + ", ".join(
                    f"T{e['turn']}{e['emotion']}(强度{e['intensity']})" for e in key_events
                ))
            # 情绪趋势
            intensities = [e.get("intensity", 0) for e in emotions]
            if len(intensities) >= 2:
                trend = "上升" if intensities[-1] > intensities[0] else "下降" if intensities[-1] < intensities[0] else "稳定"
                parts.append(f"情绪趋势: {trend}（{intensities[0]:.1f}→{intensities[-1]:.1f}）")

    if dimension == "EFFICIENCY":
        parts.append(f"轮次比: {tier1.get('turns_ratio', 1):.1f}x 预期")
        parts.append(f"卡死轮次: {tier1.get('stuck_count', 0)}轮")
        parts.append(f"should_end不匹配: {tier1.get('should_end_mismatch', 0)}次")
        parts.append(f"话术重复度: {tier1.get('repetition_score', 0):.2f}")
        end_turns = signals.get("should_end_turns", [])
        if end_turns:
            parts.append(f"用户表达结束意愿: 第{','.join(map(str, end_turns))}轮")

    if dimension in ("FLOW_COVERAGE", "ROLE"):
        # model_behavior 一致性
        mb_issues = []
        if conv:
            for t in conv.turns:
                mb = t.parsed_tags.get("model_behavior", {})
                if isinstance(mb, dict) and mb.get("是否改变态度") == "是":
                    mb_issues.append(f"T{t.turn_number}")
        if mb_issues:
            parts.append(f"用户态度变化: {', '.join(mb_issues)}")

    if dimension == "SAFETY":
        risk_turns = []
        if conv:
            for t in conv.turns:
                rf = t.parsed_tags.get("risk_flag", {})
                if isinstance(rf, dict) and rf:
                    risk_turns.append(str(t.turn_number))
        if risk_turns:
            parts.append(f"风险标记触发: 第{','.join(risk_turns)}轮")

    if dimension == "KNOWLEDGE":
        mem_facts = signals.get("memory_facts", [])
        if mem_facts:
            parts.append(f"用户提及关键事实: {len(mem_facts)}处")
            for mf in mem_facts[-3:]:  # 最近3个
                parts.append(f"  T{mf['turn']}: {mf['facts'][:80]}...")

    return "\n".join(parts) if parts else "无特殊信号上下文。"


# ============================================================
# V1: 分支覆盖计算
# ============================================================

def compute_branch_coverage(conv: Conversation, case: Optional[Case] = None) -> Dict[str, Any]:
    """V1: 从 parsed_tags 提取实际触发分支，与 Case 预期分支对比"""
    # 提取预期分支
    expected_branches = []
    if case and case.call_flow:
        for i, step in enumerate(case.call_flow):
            for j, branch in enumerate(getattr(step, "branching", []) or []):
                cond = getattr(branch, "condition", "")
                action = getattr(branch, "action", "")
                expected_branches.append(f"step_{i + 1}_branch_{j + 1}: {cond} → {action}")

    # 提取实际触发分支
    triggered_branches = []
    for t in conv.turns:
        state = t.parsed_tags.get("state", {})
        if isinstance(state, dict):
            bt = state.get("branch_triggered", "none")
            if bt and bt != "none":
                triggered_branches.append({
                    "turn": t.turn_number,
                    "branch": bt,
                    "confidence": state.get("branch_trigger_confidence", "medium"),
                })

    triggered_names = {tb["branch"] for tb in triggered_branches}
    matched_expected = set()
    for expected in expected_branches:
        prefix = expected.split(":")[0].strip() if ":" in expected else expected
        for tname in triggered_names:
            if tname in expected or prefix.lower() in tname.lower():
                matched_expected.add(expected)
                break
    untriggered = [b for b in expected_branches if b not in matched_expected]

    return {
        "expected": expected_branches,
        "triggered": sorted(triggered_names),
        "untriggered": untriggered,
        "coverage_ratio": len(matched_expected) / max(len(expected_branches), 1),
    }


# ============================================================
# V5: 三源融合置信度
# ============================================================

def compute_v5_state_confidence(conv: Conversation) -> Dict[str, Any]:
    """V5: 三源融合——每轮 state 标签的置信度评估（纯规则，零 LLM）

    Source 1: state 标签完整性（emotion/stance 是否存在）
    Source 2: 内部一致性（emotion_change 是否匹配 emotion 值序列）
    Source 3: 跨轮一致性（branch_triggered 序列逻辑）
    """
    per_turn = {}
    prev_emotion = None

    for i, turn in enumerate(conv.turns):
        state = turn.parsed_tags.get("state", {})
        if not isinstance(state, dict) or not state:
            per_turn[turn.turn_number] = {"confidence": 0.5, "sources": 0}
            continue

        # Source 1: state 标签完整性
        s1 = bool(state.get("emotion") and state.get("stance"))

        # Source 2: 内部一致性
        ec = state.get("emotion_change", "")
        s2 = prev_emotion is not None  # 无法校验时保守处理
        if ec and "→" in ec and prev_emotion:
            parts = ec.split("→")
            if len(parts) == 2:
                prev_from_tag = parts[0].strip()
                s2 = (prev_from_tag == prev_emotion)
            else:
                s2 = False  # 多箭头格式异常，不信任
        # 第一轮无法校验 emotion_change → 默认有效
        if i == 0:
            s2 = True

        # Source 3: 跨轮一致性
        bt = state.get("branch_triggered", "none")
        s3 = bt is not None and bt != "none"  # 有分支触发标记说明与其他轮有逻辑关联

        sources = sum([s1, s2, s3])
        conf = {3: 1.0, 2: 0.85}.get(sources, 0.5)
        per_turn[turn.turn_number] = {
            "confidence": conf,
            "sources": sources,
            "s1_valid": s1,
            "s2_valid": s2,
            "s3_valid": s3,
        }

        # 记录本轮 emotion 供下轮校验
        prev_emotion = state.get("emotion", "")

    avg_conf = sum(v["confidence"] for v in per_turn.values()) / max(len(per_turn), 1)
    low_turns = [t for t, v in per_turn.items() if v["confidence"] < 0.6]

    return {
        "per_turn": per_turn,
        "avg_confidence": avg_conf,
        "low_confidence_turns": low_turns,
        "low_confidence_ratio": len(low_turns) / max(len(per_turn), 1),
    }


# ============================================================
# complexity_score 计算
# ============================================================

def compute_complexity_score(case: Case) -> float:
    """根据 Case 结构 + 语义深度计算复杂度评分 0-10

    第一阶段: 结构复杂度（cap 6）— 5 个因子
    第二阶段: 语义深度（cap 4）— 3 个因子
    """
    # ---- 第一阶段: 结构复杂度（cap 6）----
    structural_score = 0.0

    # 1. call_flow 分支点数 × 1.5（上限 6）
    if case.call_flow:
        branch_count = sum(1 for step in case.call_flow if getattr(step, "branching", []) or [])
        structural_score += min(branch_count * 1.5, 6)

    # 2. constraints 数量 × 0.5（上限 2）
    if case.constraints:
        structural_score += min(len(case.constraints) * 0.5, 2)

    # 3. 有 safety 约束 +1
    if case.constraints:
        has_safety = any(getattr(c, "type", "") == "safety" for c in case.constraints)
        if has_safety:
            structural_score += 1

    # 4. knowledge_points 数量 ≥ 5 时 +1
    if case.knowledge_points and len(case.knowledge_points) >= 5:
        structural_score += 1

    # 5. 有 adversarial 标记的 constraints +1
    if case.constraints:
        has_adversarial = any("adversarial" in getattr(c, "description", "").lower() for c in case.constraints)
        if has_adversarial:
            structural_score += 1

    structural_score = min(structural_score, 6.0)

    # ---- 第二阶段: 语义深度（cap 4）----
    semantic_score = 0.0

    # 6. knowledge_content_depth: 知识点的内容深度
    if case.knowledge_points:
        avg_len = sum(len(getattr(kp, "content", str(kp))) for kp in case.knowledge_points) / len(case.knowledge_points)
        if avg_len >= 500:
            semantic_score += 2
        elif avg_len >= 200:
            semantic_score += 1

    # 7. constraint_semantic_depth: 深层约束（非 word_limit/forbidden_word 类型）
    if case.constraints:
        deep_constraints = [
            c for c in case.constraints
            if getattr(c, "type", "") not in ("word_limit", "forbidden_word")
        ]
        semantic_score += min(len(deep_constraints) * 0.3, 2.0)

    # 8. branch_nesting_depth: 最大子步骤嵌套层级
    if case.call_flow:
        max_nesting = 0
        for step in case.call_flow:
            sub = getattr(step, "sub_steps", []) or []
            if sub:
                max_nesting = max(max_nesting, 1)
                for s in sub:
                    if isinstance(s, dict) and s.get("sub_steps"):
                        max_nesting = max(max_nesting, 2)
        if max_nesting >= 2:
            semantic_score += 1

    semantic_score = min(semantic_score, 4.0)

    return min(structural_score + semantic_score, 10.0)
