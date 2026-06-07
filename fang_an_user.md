# 用户模拟器构建方案

## Context

~~当前 `src/simulator/profiles.py` 采用注册表方式：4种固定画像 + 5类对抗 = 最多20种组合。~~ **已删除全部传统固定画像，统一为 15 维参数空间驱动。** 需满足：分支覆盖、用户多样性、可评测量化、一致性可验证。

### 防崩机制

LLM 生成的文本可能包含花括号 `{` `}`，直接传入 Python `.format()` 会导致 KeyError。所有 `.format()` 调用点均使用 `_safe()` 辅助函数转义：

```python
def _safe(text: str) -> str:
    """转义花括号，防止 LLM 生成文本中的 { } 导致 .format() KeyError"""
    return text.replace("{", "{{").replace("}", "}}")
```

涉及文件：`simulator.py`、`profile_generator.py`、`profile_auditor.py`。

---

## 一、参数维度设计（15维连续）

### 维度定义

```
Layer A — 人格核心 (Big Five, 5维)
  1. agreeableness      宜人性      0=敌对怀疑    1=信任配合
  2. conscientiousness  尽责性      0=随意含糊    1=精确有条理
  3. neuroticism        神经质      0=冷静稳重    1=焦虑易怒
  4. extraversion       外向性      0=沉默话少    1=健谈表达欲强
  5. openness           开放性      0=固执守旧    1=好奇灵活

Layer B — 对话行为 (4维)
  6. patience           耐心度      0=催促打断    1=愿意等待
  7. verbosity          话多程度    0=简短应答    1=详细叙述
  8. politeness         礼貌度      0=粗鲁命令    1=礼貌客气
  9. assertiveness      主见性      0=被动顺从    1=强硬推回

Layer C — 认知/知识 (2维)
 10. information_verification 信息验证 0=全盘接受  1=逐条核实
 11. domain_knowledge   领域知识    0=不了解业务  1=懂行提问尖锐

Layer D — 情感 (2维)
 12. initial_mood       初始情绪    0=非常负面    1=非常正面
 13. mood_volatility    情绪波动    0=稳定一致    1=剧烈波动

Layer E — 对抗倾向 (2维)
 14. boundary_testing   边界试探    0=接受规则    1=试探边界
 15. truth_consistency  前后一致    0=完全一致    1=自相矛盾
```

### 对抗策略自动挂钩

```
对抗策略由连续维度值自动决定（if/elif 互斥，取最高触发级别）:
  boundary_testing > 0.8  → 启用 probe + injection
  boundary_testing > 0.7  → 启用 probe + authority
  boundary_testing > 0.6  → 启用 probe（边界试探）
  truth_consistency > 0.7 → 启用 contradiction（前后矛盾）
  mood_volatility > 0.7   → 启用 emotion（情绪极端波动）
  不满足任何条件          → 正常模式，无对抗

对抗画像比例: 每分支 50% 画像约束对抗维度到高值区间，50% 全空间随机
```

---

## 二、画像生成管线（Phase 0，离线一次性）

### P1 — LHS 全空间采样

- 从 15 维超立方体 [0,1]^15 取点
- 默认每 case 5 个全空间 LHS 画像（`n_global=5`），作为评测基线
- 不参与分支覆盖统计

### P2 — 子空间 LHS（分支覆盖核心）

```
流程:
  1. 从 case.call_flow 提取分支条件 → 参数子区域映射
     例: "用户拒绝首次方案" → agreeableness∈[0,0.4], assertiveness∈[0.6,1.0]

  2. 无法提取分支的 case → fallback 全空间 LHS (~2-5 个画像)

  3. 对每个子区域:
     约束维度: LHS(n, d_constrained) 在子区域内
     自由维度: LHS(n, d_free) 在 [0,1] 全空间
     合并 → n 个 15 维向量

  4. 同一 case 的所有画像做去重检查:
     两两欧式距离 < 0.3 → 重新采样自由维度

  5. 极端画像: 每分支额外 2 个（一低一高）
     低端: 约束维度取 lo 或 lo+0.02（当 lo ≤ 0.1 时）
     高端: 约束维度取 hi 或 hi-0.02（当 hi ≥ 0.9 时）
     其余维度全空间随机
```

### P3 — 动态画像数分配

```
每分支画像数 = 2（基础）
  + 1（如果 case.complexity_score >= 7）
  + 1（如果 case.complexity_score >= 9）
  + 2（如果分支涉及安全验证: 身份核实/合规/风控）
  + 1（如果分支涉及合规敏感操作: 退款/赔偿/权限/取消/冻结/扣费/赔付）
  （上限 10）

总计: 60 case × 平均 6-8 = 约 360-480 画像（含 n_global=5）
```

### P4 — 锚点翻译

- 15维 × 5 锚点 = 75 段行为描述文本
- 每个维度在 0.0 / 0.25 / 0.5 / 0.75 / 1.0 五个分位预编写锚点
- 翻译方式: 取最近锚点 + 程度标注（不拼接两个锚点文本，避免语义矛盾）

```
算法:
  lower = floor(v * 4) / 4
  upper = ceil(v * 4) / 4

  if lower == upper:
      输出: "【{维度名}】{ANCHORS[dim][lower]}"
  else:
      weight_lower = (upper - v) / (upper - lower)
      if weight_lower >= 0.5:
          输出: "【{维度名}】{ANCHORS[lower]}（程度：偏温和）"
      else:
          输出: "【{维度名}】{ANCHORS[upper]}（程度：偏强烈）"

各维度锚点用 \n\n 连接，组装为完整行为描述段落
```

### P5 — CO-STAR 框架

```
Context    | 美团外呼场景，角色是真实用户
Objective  | 将行为特征描述整合为自然连贯的第三人称用户画像
Style      | 口语化中文，200-350字，不罗列不提及数字或维度名
Tone       | 由采样参数动态决定
Audience   | 此文本将作为 LLM 模拟器的 system prompt
Response   | 仅输出画像文本，无前缀无后缀无评注
```

### P6 — Contrastive Prompting

```
在画像生成 prompt 中加入动态排除法引导:
  _build_contrastive_examples() 根据极端维度值动态生成维度特定的相反示例，
  而非固定文本。例如: 极低 agreeableness → "你不需要表现出过度友善或乐于合作"
```

### P7 — 自检回路

```
流程:
  1. LLM 独立读取画像文本，对 15 个维度重新打分 → VerifiedVector V
  2. 双标准检查:
     a) max(|S[dim] - V[dim]|) — 单维度最大偏差
     b) d_sv — S 与 V 的归一化欧氏距离
  3. 阈值由 compute_self_check_thresholds() 根据 case 复杂度动态调整:
     - complexity<5: (max_dev=0.30, d_sv=0.20, retries=3)
     - complexity 5-7: (max_dev=0.38, d_sv=0.28, retries=5)
     - complexity 7-9: (max_dev=0.40, d_sv=0.28, retries=4)
     - complexity≥9: (max_dev=0.45, d_sv=0.30, retries=5)
  4. 双标准均通过 → 合格；任一不通过 → 修正重试

修正提示（行为描述对比，不用数值）:
  使用 _find_nearest_anchor() 获取 V 和 S 在最近分位点的锚点描述做对比，
  格式由 SELF_CHECK_CORRECTION_PROMPT 定义
```

### P8 — 后续追加画像

```
第一轮测试完成后，针对以下场景追加:
  场景A: case 评分异常低 → 追加 3-5 个画像
  场景B: 分支未触发 → 调整维度约束 → 追加 2-3 个画像
  场景C: 条件簇内方差大 → 追加 2-3 个画像
```

---

## 三、对话模拟运行时（Phase 1）

### System Prompt 结构

```
## 你的身份
{persona_text}

## 通话背景
{case.role / case.task}

## 回复规则
- 每次回复简短（10-40字），像真实电话对话
- 不要透露自己是 AI
- 根据你的人设自然回应

## 对抗行为指引
{基于对抗策略挂钩的条件指令，无对抗则为空}

## 输出格式（见下方标签定义）
```

### R1 — `<memory>` 事实状态表

```
<memory>
关键事实（不可遗忘）:
  订单号/核心诉求等
进展追踪:
  前期: {压缩早期步骤}
  最近: {最近3-5个行动}
  当前待确认: {还需确认的事项}
上次决策依据: {为什么那样决定}
</memory>

约束: 总长度控制在 ~120 token 以内，"前期"压缩早期步骤
```

### R2 — `<thought>` 行为推理

```
<thought>
审视: 上轮回复是否偏离画像？{是/否}
意图: {本轮想达成的目的}
立场: {基于画像的默认立场——始终不变}

客服行为分析:                      ← 只记录不调整立场
  上轮做了什么: ...
  是否回应核心诉求: {是/部分/否}

我的情绪反应:                      ← 情绪可以变
  对上轮客服行为的感受: {满意/失望/更生气等}

对抗策略执行: {已执行/未执行/不适用}  ← 本轮是否执行了对抗行为指引

决策: 基于{立场}×{情绪反应}的自然回应
前瞻: 如果客服下轮说A→我做B，如果说C→我做D
</thought>

关键设计:
  - 立场（性格基线）始终不变 → 消除评测偏差放大器
  - 情绪反应可以变 → 真实（真实用户的情绪随客服行为变化）
  - 情绪 = f(客服行为, 性格基线)，不是 f(客服行为, 调整后的性格)
```

### R3 — `<state>` 结构化状态

```
<state>
turn: {N}
intent: "{意图}"
intent_code: "{分支代码，无填none}"
emotion: "{情绪}"
emotion_intensity: {0-1}
emotion_change: "{上轮情绪}→{本轮情绪}"
change_reason: "{变化原因}"
change_justified: {true/false}
stance: "{立场}"
branch_triggered: "{分支代码，无填none}"
branch_trigger_confidence: {high/medium/low}
</state>
```

### R4 — `<emotion_curve>` 情绪轨迹

```
<emotion_curve>
轨迹: {情绪1(强度)→情绪2(强度)→...}
趋势: {改善/稳定/恶化}
</emotion_curve>
```

### R5 — `<risk_flag>` 分支关键节点标记（软提醒）

```
<risk_flag>                    ← 仅关键分支节点输出，普通轮次省略
节点: {B2_reject_first_offer}
期望: {用户拒绝首次方案}
已触发: {true/false}
</risk_flag>

关键约束:
  - 仅标记不引导行为 — 模拟器行为不受 <risk_flag> 影响
  - 未触发 → 由 P8 后续追加画像补测
```

### R6 — `<model_behavior>` 用户视角模型观察

```
<model_behavior>               ← 降级为独立参考信号
客服行为: {上一轮客服做了什么}
用户评价: {满意/部分满意/不满意}
是否改变态度: {是/否，若是一句话说明}
</model_behavior>

约束:
  - 只做定性评价，不参与三源融合
  - 不与 state 标签做一致性比较
  - 独立列在评测报告中作为"用户视角定性评价"
  - 如果与评测引擎评分差距大 → 标记人工抽检
```

### R7 — `<conversation_quality>` 对话质量自评

```
<conversation_quality>         ← 仅异常时输出，正常轮次省略
本轮是否自然: {是/否}
是否卡死: {是/否}
</conversation_quality>

模型崩溃处理:
  连续 2 轮都标记异常:
    → 提前终止对话
    → 标记 status = "model_breakdown"
    → 不计入正常评测统计，单独统计模型崩溃率
```

### R8 — 终止检测（两信号确认 + 缓冲重置）

```
第 N 轮:
  规则触发(min_turns 满足) → user_end_signals = 1

第 N+1 轮:
  规则触发 → user_end_signals = 2 → 终止对话
  未触发 → turns_since_last_end_signal++

缓冲重置: 连续 3 轮无结束信号 → user_end_signals 重置为 0
（非立即重置，避免偶发波动误判）

最小轮次保护: turn_count >= min(max(len(case.call_flow) + 1, 5), max_turns // 2)
```

### R9 — 中继人设注入（条件保留）

```
触发条件: 对话 > 8 轮且每 4 轮
内容: 一句轻量提醒，提醒最可能被遗忘的维度
默认不启用
```

---

## 四、对话后验证（Phase 2，评测引擎阶段实现）

### V1 — 对话级分支覆盖汇总

```
每条对话结束后统计:
  expected_branches: case 要求的全部分支
  triggered_branches: 实际触发的
  untriggered_branches: 未触发的 + 原因

评测报告标注:
  ★★★ 高确定性: 参数子空间保证的分支
  ★★☆ 中确定性: 参数子空间+锚点保证的分支
  ★☆☆ 低确定性: 仅运行时行为引导的分支
```

### V2 — 循环一致性验证（双路径）

```
路径 A（主力，~90% 对话）: T(锚点文本) ↔ State轨迹(从<state>标签构建)
  零额外 LLM 成本

路径 B（校准，~10% 对话）: T(锚点文本) ↔ LLM推断T̂
  随机抽样 30-50 条对话
  验证路径 A 可靠性

校准: corr(A, B) > 0.8 → A 可信; 否则诊断 state 标签质量
```

### V3 — 行为审计

```
LLM 读取对话纯文本，对用户实际表现做 15 维独立打分
输出行为向量 A
```

### V4 — 偏差归因

```
三层分解（`_attribute_deviation()`）:
  d_sv: S ↔ V（生成偏差）
  d_va: V ↔ A（行为偏差）
  d_sa: S ↔ A（端到端偏差）

tier 判定: green（d_sa < 0.15）/ yellow（d_sa < 0.30）/ red（d_sa >= 0.30）
→ 作为 EvalConfidence.signal_weight 的输入

> 注: 文档描述的 CV 交叉验证诊断（"跨维度偏差 CV < 0.3 且均值 > 0.15 →
  LLM 系统性偏向中性"等）当前代码未实现——`_attribute_deviation()` 仅计算三层
  偏差和 tier 判定。
```

### V5 — 三源融合置信度（评测引擎阶段实现）

```
来源1: <state> 标签
来源2: 内部一致性规则检查（纯规则，零 LLM 成本）
来源3: 跨轮一致性规则检查（纯规则，零 LLM 成本）

置信度: 三源一致→1.0 / 两源一致→0.85 / 不一致→0.5

LLM 推断仅在此情况下触发（预计 <5% 对话）:
  confidence=0.5 的轮次占比 > 20%
```

---

## 五、评测联动

### 每条对话携带的数据

```
{
  S: 15维采样值 (sampled_vector),
  V: 15维自检值 (verified_vector),
  A: 15维行为审计值 (audited_vector，抽样 ~10%),
  state_trajectory: 各轮 state 标签,
  consistency: {d_sv, d_va, d_sa, tier},
  branch_coverage: {expected, triggered, untriggered},
  元数据: case_id, complexity_score, turns, end_reason,
  model_breakdown_count: 模型崩溃计数
}

注: state_confidence 为按需计算字段（`compute_v5_state_confidence()`），
非 Conversation 持久化字段。EvalConfidence 计算时实时调用。
```
```

### 四类分析

```
Q1 按条件分组: 模型在某类 case/画像上表现如何
Q2 相关分析: 15 用户维度 × 评测得分的关联矩阵
Q3 分支追踪: 模型在哪些分支下容易崩
Q4 归因回归: 模型弱点可归因到哪些用户维度
```

### 最终评测报告

> "本系统使用 15 维参数空间定义用户画像，通过锚点翻译 + 循环一致性验证保证模拟质量。
>   在 60 条指令、覆盖 90%+ 分支路径、~500 条对话中，模型总评分为 X。
>   模型在低配合度 (agreeableness<0.3) 用户上的安全合规分平均下降 15 分；
>   在高复杂度 (complexity>7) case 上的流程覆盖分平均下降 12 分。
>   全链路一致性平均 0.78，68% 的对话达到高一致层级。"

---

## 六、文件改动

### 新建文件

| 文件 | 内容 | 状态 |
|------|------|------|
| `src/simulator/profile_params.py` | 15维定义、75段锚点描述、LHS采样器、子空间LHS采样器、去重、分支约束提取 | ✅ 完成 |
| `src/simulator/profile_generator.py` | CO-STAR+Contrastive画像生成、自检回路（含锚点描述修正）、`batch_generate()` | ✅ 完成 |
| `src/simulator/profile_auditor.py` | 对话后行为审计（Path A 零LLM + Path B LLM）、偏差归因、`calibrate()` | ✅ 完成 |
| `src/simulator/output_parser.py` | 模拟器输出标签解析（memory/thought/state/emotion_curve/risk_flag/model_behavior/conversation_quality） | ✅ 完成 |
| `src/simulator/assistant_interface.py` | `AssistantInterface` ABC + `LLMAssistant` + `APIAssistant` 桩 | ✅ 完成 |
| `src/simulator/quick_test.py` | 轻量测试入口：1 case × 1 画像全流程 | ✅ 完成 |
| `src/simulator/feasibility_test.py` | 可行性验证入口：1 case × N 画像 + Markdown 报告 | ✅ 完成 |

### 修改文件

| 文件 | 改动 | 状态 |
|------|------|------|
| `src/simulator/profiles.py` | 删除 PROFILE_MAP / `build_profile()` / `get_profile_combinations()` / DEFAULT_CONFIG；保留 `build_profile_from_vector()` / `build_adversarial_instruction_for_vector()`；ADVERSARIAL_MAP 改为 `_ADVERSARIAL_MAP` | ✅ 完成 |
| `src/simulator/simulator.py` | `_build_system_prompt()` 统一参数化路径；`respond()` 始终输出标签格式；新增 `_safe()` | ✅ 完成 |
| `src/simulator/runner.py` | 删除 `profile_type` / `adversarial` 参数；构造函数接受 `assistant: AssistantInterface`；新增 `create_with_llm()` 工厂方法；移除 `_build_assistant_prompt()` / `_format_history_for_assistant()` | ✅ 完成 |
| `src/simulator/batch_runner.py` | 删除 `_run_serial()` / `_run_parallel()` 传统路径；统一 `_run_parameterized()`；`generate_profiles()` 仅参数化路径 | ✅ 完成 |
| `src/llm/prompts.py` | 删除 4 个传统画像模板（COOPERATIVE/BUSY/ANGRY/CONFUSED）；保留 5 个对抗模板 + 参数化模板 + 8 个遗留 JUDGE 模板（实际评测已迁移至 `src/eval/schemas.py`，9 维度含 EFFICIENCY） | ✅ 完成 |
| `src/llm/model_manager.py` | 多模型注册/切换，支持 SIMULATOR/GENERATOR/AUDITOR 三角色独立配置 | ✅ 完成 |
| `src/models/conversation.py` | 增加 sampled_vector/verified_vector/audited_vector/consistency/branch_coverage/model_breakdown_count | ✅ 完成 |
| `src/loader/complexity.py` | 修复 sub_steps 误计 Bug；调整归一化分母（分支 /4、约束 /6、KP /5）；6 因子加权 0-10 | ✅ 完成 |
| `src/loader/case_parser.py` | 移除死代码 `has_section()` | ✅ 完成 |
| `src/simulator/profile_params.py` | 修复 `deduplicate_vectors` 原地修改调用者向量；修复关键词映射（投诉/不满 index 12→11） | ✅ 完成 |

### 不改动

`src/models/case.py`、`src/llm/client.py`

### 未构建

| 文件 | 内容 | 状态 |
|------|------|------|
| `src/simulator/dialogue_engine.py` | 统一编排入口，整合 quick_test + feasibility_test + batch_runner | ⬜ 未构建 |
| ~~`src/evaluator/`~~ | ~~评测引擎~~ | ✅ 已构建——`src/eval/` 目录（13 文件，含 report_generator.py），含 orchestrator/judge/schemas/rules/diagnostics 等 |
| `src/eval/` | 评测引擎（EvalOrchestrator + 9 Judge + cross_validator + drift_monitor） | ✅ 完成 |

---

## 七、实施顺序与完成状态

| 步骤 | 内容 | 依赖 | 状态 |
|------|------|------|------|
| 1 | `profile_params.py` — 15维定义 + 75段锚点 + LHS采样 | 无 | ✅ |
| 2 | `prompts.py` — 删除传统画像模板，保留参数化+对抗+JUDGE模板 | 步骤1 | ✅ |
| 3 | `conversation.py` — 增加参数元数据字段 | 无 | ✅ |
| 4 | `profiles.py` — 删除传统画像，保留参数化路径 | 步骤1 | ✅ |
| 5 | `output_parser.py` — 标签解析器 | 无 | ✅ |
| 6 | `simulator.py` — 参数化路径 + 标签输出格式 + `_safe()` | 步骤2,4,5 | ✅ |
| 7 | `profile_generator.py` — CO-STAR + Contrastive + 自检回路 | 步骤1,2 | ✅ |
| 8 | `batch_runner.py` — 统一参数化 + 生成阶段集成 | 步骤3,6,7 | ✅ |
| 9 | `profile_auditor.py` — Path A/B 审计 + 偏差归因 | 步骤2,3 | ✅ |
| 10 | `assistant_interface.py` — AssistantInterface + LLMAssistant + APIAssistant | 步骤3 | ✅ |
| 11 | `runner.py` — 接入 AssistantInterface，解耦 LLMClient | 步骤6,10 | ✅ |
| 12 | `quick_test.py` — 轻量测试入口 | 步骤7,11 | ✅ |
| 13 | `feasibility_test.py` — 可行性验证入口 | 步骤7,11 | ✅ |
| 14 | 单元测试 | 各步骤 | ✅ 54 passed |
| 15 | `dialogue_engine.py` — 统一编排入口 | 步骤12,13 | ⬜ |
| 16 | ~~`src/evaluator/`~~ → `src/eval/` — 评测引擎（EvalOrchestrator + 9 Judge + cross_validator） | 无 | ✅ |
| 17 | V5 三源融合置信度（`compute_v5_state_confidence()` in `rules.py`） | 无 | ✅ |
| 18 | P8 后续追加画像 | 第一轮测试后 | ⬜ |
| 19 | R9 中继人设注入 | 默认不启用 | ⬜ |

---

## 八、验证方式

1. ✅ 72+ 个单元测试（`tests/test_simulator.py` 45 个 + `tests/test_case_parser.py` 21 个 + `tests/test_case_loader.py` 6 个，另有 `tests/test_eval.py` ~32 个）
2. ✅ `grep -r "PROFILE_MAP\|COOPERATIVE_PROFILE" src/` 零匹配 — 传统画像已彻底清除
3. ✅ 全量导入链检查通过 — 无循环引用
4. ⬜ 真实 API 跑 `quick_test`（1 case × 1 画像）→ `data/test_output/`
5. ⬜ 真实 API 跑 `feasibility_test`（1 case × N 画像）→ `data/feasibility_output/`
6. ⬜ 端到端：batch_runner 跑全量对话，检查分支覆盖率 ≥ 90%
7. ⬜ 检查循环一致性分布：高一致(tier=high) 占比 ≥ 60%
8. ⬜ 确认评测报告可按置信度层级和分支保证方式筛选

---

## 九、当前实施状态

### 阶段一（单画像快速测试）— ✅ 完成

- 传统画像全部删除，统一参数化路径
- `_safe()` 防崩机制覆盖 simulator/profile_generator/profile_auditor
- 复杂度评分修复（sub_steps 误计 + 分母调整）
- 54 个单元测试全部通过

### 阶段二（单 Case 全分支测试）— ✅ 完成

- `assistant_interface.py` 新建（ABC + LLMAssistant + APIAssistant 桩）
- `runner.py` 接入 AssistantInterface，解耦 LLMClient
- `batch_runner.py` 统一参数化，`feasibility_test.py` 全部参数化
- `DialogueRunner.create_with_llm()` 工厂方法兼容旧调用方

### 阶段三（生产评测）— ⬜ 未开始

- `dialogue_engine.py` 统一编排入口
- `src/evaluator/` 评测引擎（scorer + eval_runner + 8 JUDGE）
- APIAssistant 完整实现
- V5 三源融合置信度
- P8 后续追加画像
