 # 优化引擎 v1 开发计划书

---

> **定位**：优化引擎是一个轻量化模块——将原始 Case 定义、对话数据、用户模拟数据、评测结果一并交给 LLM 和规则引擎分析，为 Case 定义、用户画像生成器、对话模型、评测引擎提供具体的优化建议。只出建议，不动代码。

---

## 一、概述

### 1.1 背景

评测引擎 v1 已完成。它能对每场模拟对话产出详细诊断：9 维度评分（五级文字评级 + 数值分数 + 百分制参考分）、清单项逐条判定、缺陷根因归属、可信度评估。

但有一个断层：评测引擎告诉你"哪里病了"，却不会告诉你"怎么治"。归因模块的建议停留在关键词匹配层面——"检查 prompt 中对应步骤指令是否明确"——太泛，无法直接执行。

优化引擎填补这个断层：把原始数据 + 评测诊断一并分析，转化为具体、可执行的优化建议。

### 1.2 目标

根据原始 Case 定义、对话文本、用户模拟数据、评测结果，为以下四个优化对象提供优化建议：

| 优化对象 | 可优化的内容 |
|---------|------------|
| **Case 定义** | call_flow 步骤/分支/reference_script、constraints 约束条款、knowledge_points 知识点、opening_line 开场白、complexity_score 复杂度、raw_instruction 文本 |
| **用户画像生成器** | 15 维画像参数锚点、对抗策略 prompt（PROBE/INJECTION/CONTRADICTION/AUTHORITY/EMOTION）及触发条件、CO-STAR 生成模板、自检阈值（max_dev/d_sv）、Simulator 对话行为参数（END_KEYWORDS、should_end 逻辑、history 格式化） |
| **被评测对话模型** | Assistant System Prompt 各段文本（角色/任务/开场白/流程/知识点/约束/闭合）、raw_instruction 模式下的自由文本、few-shot 示例 |
| **评测引擎自身** | SOURCE_WEIGHTS、DIMENSION_WEIGHTS、评级阈值、置信度因子、清单大小边界、进化阈值、Judge prompt 文本、Rubric 锚点描述、盲点扫描清单 |

### 1.3 范围

| 项目 | 内容 |
|------|------|
| **输入** | `optimization_feed.json`（评测归因）+ `case.json`（Case 定义）+ `profiles.json`（用户画像）+ `conversation_*.json`（完整对话文本，**需 JSON 格式**以包含每轮完整 `parsed_tags` 键值对，非仅 key 名）+ ChecklistEvolver 跨批次统计（`data/checklist_evolution/` 下的 JSONL）。prompt 模板文本（CO-STAR/Judge/对抗策略）从源码文件直接读取。**v1 按单 Case 批次运行，多 Case 场景由外部循环调用** |
| **输出** | `optimization_report.md` + `optimization_actions.json`。**仅此两个文件** |
| **方法** | LLM 生成 + 规则引擎混合——LLM 负责综合分析并生成具体建议文本，规则引擎负责统计计算和确定性诊断 |
| **运行方式** | `python -m src.optimizer.optimizer --input-dir data/exports/xxx/ --output data/optimization/xxx/` |

### 1.4 边界约束

- ❌ **不写任何源文件**：不修改 prompt、不修改 Case YAML、不调整画像参数、不改评测引擎配置
- ❌ **不调用任何修改接口**
- ❌ 不做自动闭环触发（v2）、不做优化效果自动衡量（v2）、不做训练数据生成（v3）

---

## 二、核心理念

```
原始数据（Case + 对话 + 画像 + 评测结果）
        │
        ▼
┌──────────────────────────────────────────────┐
│            优化引擎（LLM + 规则引擎）           │
│                                              │
│  规则引擎：统计计算 + 确定性诊断                │
│        │                                     │
│        ├──→ 路径 A：确定性发现 → 直接进报告     │
│        │         （数值异常/统计结论/高频模式）  │
│        │                                     │
│        └──→ 路径 B：复杂分析 → LLM 深度生成     │
│              LLM 综合原始数据 + 规则发现        │
│                                              │
│  ┌──────────────────────────────────────┐    │
│  │ 对 Case 定义的完善建议                │    │
│  │ 对用户画像生成器的改进建议             │    │
│  │ 对被评测对话模型的优化建议             │    │
│  │ 对评测引擎自身的优化建议               │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  ╔══════════════════════════════════════╗     │
│  ║  优化引擎到此为止                     ║     │
│  ║  以下由人类决定是否执行                ║     │
│  ╚══════════════════════════════════════╝     │
└──────────────────────────────────────────────┘
```

**规则引擎和 LLM 的分工——双路径输出**：

规则引擎的计算结果走两条路径输出。确定性强的直接进报告，需要深度分析的交给 LLM：

```
规则引擎计算结果
        │
        ├──→ 路径 A：确定性发现 → 直接进报告（零 LLM 成本，100% 可复现）
        │
        │    适用场景：
        │    · 通过率异常（>95% 或 <30%）→ "建议降权/检查定义"
        │    · 相关性数值（Pearson r 显著偏离）→ "SOURCE_WEIGHTS 从 X 调至 Y"
        │    · 区分力指标（评分分布极度集中）→ "建议优化 Judge prompt"
        │    · 信号矛盾计数（超出阈值）→ "检查 Simulator 标签生成逻辑"
        │    · 高频缺陷聚类（≥5 次）→ "建议转化为正式清单项"
        │
        └──→ 路径 B：复杂分析上下文 → 放入 LLM prompt → LLM 深度生成
              │
              ▼
         LLM 综合原始数据（Case 全文/对话全文/画像值/prompt 文本）
         + 评测诊断（归因/证据/评分/置信度）
         + 规则发现（聚类/排序/相关性/统计）
              │
              ▼
         生成内容：
         · 具体的修改文本（prompt 段改写、约束条款重拟、话术补充）
         · 因果分析（"为什么会出现这个缺陷"）
         · 效果预估（"修改后预期改善什么"）
         · 副作用评估（"可能引入的新问题"）
```

| 路径 | 负责 | 产出特点 | 成本 | 适用 |
|------|------|---------|------|------|
| A | 规则引擎直接输出 | 确定性、可复现、模板化表述 | **零** | 数值异常、统计结论、高频模式 |
| B | LLM 综合分析后生成 | 具体文本、因果推理、上下文关联 | ~5-7K tokens/次 | 文本修改、根因分析、精细化建议 |

**核心原则**：能算出来的直接说（路径 A），需要理解的交给 LLM（路径 B）。同一优化对象的两类建议在报告中通过"关联 ID"关联——路径 A 给出数据事实，路径 B 基于事实做深度解读。

**动态决策：何时走 A、何时走 B**：

规则和 LLM 不是静态分工——每条数据到达时，根据复杂度动态决定路径：

```
数据到达 → 复杂度判断
  │
  ├── 简单（单一统计异常，无交叉因素）
  │    例：通过率 97%、Pearson r=0.45、信号矛盾计数=5
  │    → 全路径 A，零 LLM 成本
  │
  ├── 中等（需要文本生成但模式明确）
  │    例：清单项描述改写、约束条款调整——模板化但有多种表达
  │    → 路径 A 给出数据事实 + 模板框架，路径 B 做文本润色
  │
  └── 高复杂度（多因素交叉、需要因果推理）
       例：评分低 + 对话中用户情绪正常 + 画像参数偏高
       → 全路径 B，LLM 综合分析所有原始数据做因果推断
```

决策依据：缺陷聚类中的 `frequency`（频次低+多源交叉→高复杂度）、`signal_consistency`（矛盾→需要 LLM 分析原因）、`evidence_chain` 的长度（证据链越长→因素越复杂）。

**跨引擎交叉分析——优化引擎的核心价值**：

优化引擎区别于"单独看评测报告 + 单独看对话 + 单独看 Case"的关键在于：它能同时消费三个引擎的数据做交叉推理。两个典型场景：

```
场景一：区分"真问题"和"假信号"
  评测说 SENTIMENT 评分低
  + 对话中用户情绪轨迹正常（parsed_tags.emotion_curve 平稳）
  + 画像中 neuroticism=0.85（偏高）
  → 不是模型情感适配差，是画像参数偏差导致信号失真
  → 优化建议：降低 neuroticism，而非修改模型 prompt

场景二：区分"模型问题"和"Case 问题"
  评测说 FLOW_COVERAGE step_2 持续 NO
  + 对话中 Assistant 实际说了 step_2 的内容但措辞不同
  + Case 定义中 step_2 描述只有一句话且模糊
  → 不是模型跳步，是 Case 描述不清晰导致 Judge 误判
  → 优化建议：改写 Case step_2 描述，而非修改模型 prompt
```

没有跨引擎分析，这两个场景都会被误判为"模型需要优化"，导致无效的 prompt 修改。

---

## 三、上游模块关联

### 3.1 四优化对象 × 输入数据 × 可优化参数（合并表）

| 优化对象 | 输入来源（文件/路径） | 可优化的参数/内容 |
|---------|-------------------|-----------------|
| **Case 定义** | `case.json` | call_flow 各步骤的 title/description/reference_script、branching 条件与动作、is_optional 标记、constraints 文本、knowledge_points 内容、opening_line 文本、complexity_score、raw_instruction 全文 |
| **画像生成器** | `profiles.json` + `src/llm/prompts.py`（CO-STAR/对抗策略模板）+ `src/simulator/profile_params.py` | 15 维参数锚点值、`min_distance` 去重阈值、`compute_profile_count` 参数、5 个对抗策略 prompt 文本及触发阈值、CO-STAR 模板文本、自检 prompt、`d_sv`/`max_dev` 阈值、`END_KEYWORDS`、`should_end_conversation()` 匹配逻辑 |
| **对话模型** | `case.json` + `conversation_*.json`（含每轮 `parsed_tags`：emotion_curve/model_behavior/should_end/risk_flag，用于跨引擎交叉分析） | Assistant Prompt 7 段全文或 raw_instruction 全文、few-shot 示例 |
| **评测引擎** | `optimization_feed.json`（评测归因）+ `src/eval/config.py` + `src/eval/schemas.py`（Judge prompt）+ `data/checklist_evolution/*.jsonl`（进化统计） | `SOURCE_WEIGHTS`（5 个）、`DIMENSION_WEIGHTS`（9 个）、`RATING_THRESHOLDS`（4 个）、`CONFIDENCE` 因子（25+ 个）、`CHECKLIST_SIZE` 边界、`COT_QUALITY` 因子（15+ 个）、进化阈值（6 个）、9 个 Judge prompt 文本、Rubric 锚点描述、盲点扫描清单 |

### 3.2 raw_instruction 模式的特殊处理

当 `case.raw_instruction` 非空且 `use_raw_prompt=True` 时，Assistant 使用的是一段自由文本而非 7 段结构化 prompt。此时：

- prompt_optimizer 无法按"维度→prompt 段"定位，改为**将 raw_instruction 全文交给 LLM**，让 LLM 直接分析整段文本并提出修改建议
- 输出格式从"修改 # 通话流程 Step 1"变为"修改 raw_instruction 中第 X 段关于 Y 的描述"
- fewshot_generator 不受影响（它基于对话文本而非 prompt 结构）

### 3.3 中间数据消费（除最终评测产出外的可消费数据）

优化引擎不仅消费评测的最终产出（归因+评分），还消费以下中间数据以增强证据质量和分析深度：

| 中间数据 | 来源 | 消费方式 | 增强什么 |
|---------|------|---------|---------|
| **Tier1 规则指标** | `evaluation_*.json` → `rule_check_issues`（turns_ratio/stuck_count/user_repeat_rate/hangup 等） | 路径A 规则计算 | 区分"模型效率低"还是"用户故意拖延"——精确定位 EFFICIENCY 问题根因 |
| **Simulator thought/memory 标签** | `conversation_*.json` → `parsed_tags`（thought/memory 字段） | 路径B LLM 上下文 | 用户在想什么、为什么说这句话——判断画像参数偏差还是策略正确触发 |
| **Checklist item CoT 推理** | `evaluation_*.json` → `dimension_checklists[].reasoning` | 路径B LLM 上下文 | ✅ 已实现——`_load_raw_evaluation_evidence` 提取 reasoning 文本注入 evidence | LLM Judge 的完整推理链 |
| **Profile 自检结果** | `profiles.json` → `self_check_d_sv` | 路径A 规则计算 | ✅ 已修复（v1.1）——`ProfileGenerator.generate_with_retry` 存储 `d_sv` 到 `UserProfile.self_check_d_sv` | 画像质量评分 |
| **Conversation quality 标签** | `conversation_*.json` → `parsed_tags`（model_behavior/risk_flag） | 路径B LLM 上下文 | ✅ 已实现——prompt_optimizer 提取 thought/memory/state/model_behavior | 精确定位问题根因 |
| **Satisfaction trajectory** | `conversation_*.json` → `parsed_tags`（emotion_curve） | 路径B LLM 上下文 | ✅ 已导出 | 用户满意度逐轮变化 |

### 3.4 建议矛盾处理

当不同路径或不同优化对象对同一维度产生矛盾建议时（如 Case 建议"过严→放宽" vs 评测建议"区分力不足"），优化引擎在批判分析中标注矛盾并给出判断方向：
- 如果该维度 ≥80% 对话低分且归因以 model 为主 → 更可能是**模型真实问题**，优先修改 prompt
- 如果低分伴随 Case 归因占比 >40% → 更可能是**Case 设计问题**，优先修改 Case

---

## 四、系统架构

### 4.1 模块结构（8 源文件 + __init__.py）

```
src/optimizer/
│
├── optimizer.py            ← 主编排：加载(含原始评测JSON证据提取)→质量门控→规则分析→聚类→排序→双路径生成→去重→报告导出。报告生成逻辑集成在本文件中
├── case_fixer.py           ← Case 归因 → Case 设计完善建议（规则分类六种问题 + LLM 生成文本）
├── profile_optimizer.py    ← Simulator 归因 → 画像生成器改进建议（五层诊断：参数/策略/CO-STAR/自检/对话行为，规则统计 + LLM 生成）
├── prompt_optimizer.py     ← Model 归因 → 对话模型 prompt 优化建议（DSPy 批量 N=5 候选 + OPRO 轨迹 + 维度→prompt段映射）
├── fewshot_generator.py    ← 弱维度 → Few-shot 示例生成（MMR 多样性选择 + LLM 生成 ChatML 三元组）
├── eval_optimizer.py       ← 评测引擎自身问题 → 清单/Judge/置信度建议（纯规则路径A驱动，LLM 仅在文本生成时介入）
├── prompts.py              ← 5 组 LLM prompt 模板（统一结构：原始数据段 + 评测诊断段 + 规则发现段 + 输出格式要求）
├── utils.py                ← 规则引擎工具：bigram聚类、优先级计算、Pearson相关性、区分力统计、严重度分类、prompt段提取
└── __init__.py
```

LLM 和规则的配比因模块而异——prompt_optimizer/fewshot_generator 以 LLM 为主，case_fixer/profile_optimizer LLM 与规则各半，eval_optimizer 以规则为主。

### 4.2 数据流（7 步管线，与代码一致）

```
Step 0: 上下文融合
  加载 optimization_feed.json + 原始 evaluation_*.json（提取真实证据文本）
  + case.json + profiles.json + conversation_*.json（含 parsed_tags）
  + 源码 prompt 模板
        │
        ▼
Step 1: 质量门控
  is_reliable=False → 置信度×0.5 且不标记 actionable
  confidence < 0.5 → 直接舍弃
        │
        ▼
Step 2: 归因按 source 分组 → 路由到四个优化器
  model → prompt_optimizer + fewshot_generator
  case → case_fixer
  simulator → profile_optimizer
  eval 数据 → eval_optimizer
        │
        ▼
Step 3: 规则引擎预处理
  - 缺陷聚类（bigram Jaccard ≥ 0.5）
  - 优先级排序（severity × dimension_weight × ln(freq+1) × confidence）
  - 信号矛盾检测、区分力统计、评分分布分析
        │
        ▼
Step 4: 生成优化建议（双路径并行，路径A→B联动）
  ┌─────────────────────────────────────────────────────┐
  │ 路径 A（规则分析）：确定性发现 + 统计证据直接进报告  │
  │   · 评分分布异常（80%集中同一评级→区分力不足）       │
  │   · 置信度偏低（>50% unreliable→检查因子配置）       │
  │   · 清单覆盖不足（meta_check_alerts 汇总）           │
  │   · Case 过严检测（多对话低分→统计证据）             │
  │   · Simulator 信号矛盾（证据文本引用）               │
  │   零 LLM 成本，100% 可复现                          │
  ├─────────────────────────────────────────────────────┤
  │ 路径 B（深度分析）：规则→LLM→生成                   │
  │   LLM prompt 含：原始数据 + 评测诊断 + 规则发现       │
  │   + 路径A结构化上下文（LLM 需判断真问题 vs 假信号）  │
  │   产出：具体修改文本 + 因果分析 + 副作用评估         │
  └─────────────────────────────────────────────────────┘
        │
        ▼
Step 5: 综合去重 & 合并
  聚类级去重（每 cluster 最多 2 条）
  + 全局去重（Case 按维度合并、其他按 source+dim+title）
  → 限制 top-25 输出
        │
        ▼
Step 6: 报告导出
  optimization_report.md（含阅读说明 + 四个优化对象章节 + CAI三段式 + 附录统计）
  + optimization_actions.json + MANIFEST.md
```

### 4.3 设计原则

1. **双路径输出 + 联动**：规则分析（路径A，零成本 100% 可复现）产出统计发现和确定性结论 → 深度分析（路径B，LLM）基于路径A的结构化上下文做因果推理和文本生成。路径A发现（如"该维度存在 Case 过严可能"）强制注入路径B prompt，LLM 需明确判断真问题 vs 假信号
2. **真实证据驱动**：`_load_and_gate` 不仅读 `optimization_feed.json`，还读原始 `evaluation_*.json` 的 `dimension_checklists`/`meta_check_alerts`/`cross_validation_alerts` 提取真实证据文本（含对话轮次和内容）。缺文本证据时用评分分布等统计数据替代
3. **建议可执行**：宏观诊断 + 微观参数双层输出，可量化的给数值，不可量化的给方向。每条建议标注强/中/弱三级
4. **证据可追溯**：违规证据优先引用真实对话文本和评测判定（如 `T3: 客服直接进入业务内容` + `identity_verification → MOSTLY_NO`）；无文本证据时展示评分分布统计
5. **零副作用**：只读输入，只写输出，不改源文件
6. **降级容错**：LLM 全部调用失败时，路径A的确定性发现正常输出，路径B标记为"LLM 不可用，仅提供统计数据"
7. **报告自解释**：生成的 `optimization_report.md` 开头含阅读说明（术语表 + 建议结构说明），无需额外文档即可理解

---

## 五、各模块说明

### 5.1 `optimizer.py` — 主编排器

负责全流程串联：

- **质量门控**：`is_reliable=False` 的归因降权，不标记 actionable
- **规则预处理**：调用 utils.py 完成聚类、排序、相关性计算、区分力统计
- **LLM 调度**：为 case_fixer / profile_optimizer / prompt_optimizer / fewshot_generator 构建含原始数据 + 评测诊断 + 规则发现的 prompt，并行调用 LLM；eval_optimizer 以路径 A 为主（确定性发现直接输出），仅在需要文本生成时走路径 B
- **候选评估**：对 LLM 生成的建议从覆盖度、副作用风险、可执行性三维打分
- **最优选择**：得分最高的 1-2 条进入报告；最高分 < 0.4 时回退

优先级公式：`severity_weight × dimension_weight × ln(freq+1) × avg_confidence`（维度权重见附录 A）

**降级容错**：LLM 全部调用失败时，路径 A 的确定性发现照常输出；路径 B 标记为"LLM 不可用，仅提供统计数据"。

**优化优先级标注**：优化引擎在报告中标注"建议优先关注的对象"——Case 归因占比 > 40% → "建议优先处理 Case 问题（Case 缺陷可能导致 Model 评分系统性失真）"；同一维度在多场对话中置信度均 > 0.8 → "建议优先处理该维度（系统性短板）"。

### 5.2 `case_fixer.py` — Case 设计完善

**工作流程**：规则引擎先分类问题类型（六种），LLM 再进行具体文本生成。LLM prompt 包含：case.json 全文（call_flow/constraints/KP/opening_line/raw_instruction）+ 评测归因中 source=case 的项 + 多对话评分分布。

LLM 基于规则分类结果生成六类建议：

| 类型 | LLM 分析内容 |
|------|------------|
| overly_strict | 复杂度高 + 多对话低分 → 建议降低复杂度或标记 optional |
| missing_branch | 分支项持续 NO → 建议补充分支定义 |
| unclear_description | 步骤描述模糊 → 生成更清晰的描述文本 |
| constraint_issue | 约束过宽/过严 → 生成调整后的约束文本 |
| missing_knowledge | KP 不足 → 生成补充知识点 |
| flow_issue | 流程结构问题 → 生成调整建议 |
| **constraint_conflict** | **word_limit 约束与 call_flow 步骤信息量冲突。由评测引擎 meta_check(consistency) 检出，优化引擎 `_case_meta_check_fixes` 读取并生成强建议** |

规则引擎辅助：计算 complexity_score 与评分相关性、统计各步骤的通过率分布，作为 LLM 的分析依据。

### 5.3 `profile_optimizer.py` — 用户画像生成器改进

**LLM prompt 包含**：profiles.json（所有对话的 15 维参数值 + 对抗策略标注 + d_sv 值）+ CO-STAR prompt 模板全文 + 5 个对抗策略 prompt 全文 + 评测归因中 source=simulator 的项 + 信号矛盾数据 + 对话文本。

LLM 综合分析五类可优化内容：

| 优化层面 | 具体内容 | LLM 依据 |
|---------|---------|---------|
| 画像参数 | 15 维锚点值调整建议 | 参数值与评分偏差的相关性 |
| 对抗策略 | 5 个策略 prompt 文本修改 + 触发阈值调整 | 策略触发率 vs 评测维度评分的相关性 |
| CO-STAR 模板 | 生成 prompt 文本优化 | 画像质量趋势（d_sv 分布） |
| 自检阈值 | `d_sv`/`max_dev` 校准建议 | 多对话自检通过率 vs 实际画像质量 |
| 对话行为 | `END_KEYWORDS` 增减、`should_end_conversation()` 匹配逻辑调整、history 格式化优化 | 规则统计：`should_end` 触发轮次 vs `should_end_mismatch` 频率；过早结束（< min_turns）或过晚结束（> max_turns）的对话占比 |

规则引擎辅助：计算画像维度与评测维度的偏差相关性、统计对抗策略触发频率、检测信号矛盾。**具体诊断阈值**：`should_end_mismatch` > 30% 对话 → 建议收紧匹配逻辑；过早结束 > 20% → 建议增加 `END_KEYWORDS` 条目；过晚结束 > 20% → 建议降低 `should_end` 触发敏感度。

### 5.4 `prompt_optimizer.py` — 对话模型 Prompt 优化

**LLM prompt 包含**：case.json 中的 prompt 结构（7 段全文或 raw_instruction 全文）+ 评测归因中 source=model 的项 + 对话文本 + 对话中 Assistant 的实际回复。

**对话片段定位**：从评测归因的 `evidence_chain` 提取目标轮次号（如 `T5`）→ 从 `conversation_*.json` 中截取 T5 前后各 3 轮 → 同时读取该轮次的 `parsed_tags`（emotion/model_behavior）→ 一并放入 LLM prompt。这样 LLM 看到的不仅是"说了什么"，还有"Simulator 标注的状态是什么"，能做跨引擎推理。

LLM 批量生成 N=5 条候选修改方案（激进/保守/创新），覆盖：

- 各 prompt 段的具体修改文本（diff 形式）
- raw_instruction 模式下的自由文本修改
- 维度→prompt 段定位映射：

| 维度 | Prompt 段 |
|------|----------|
| SAFETY | # 通话流程 + # 约束条件 |
| TASK_COMPLETION | # 任务目标 + # 通话流程 |
| FLOW_COVERAGE | # 通话流程 |
| KNOWLEDGE | # 知识点（FAQ） |
| CONSTRAINT | # 约束条件 |
| EFFICIENCY | # 通话流程 + 闭合指令 |
| SENTIMENT | # 你的角色 + 闭合指令 |
| ROLE | # 你的角色 |
| OPENING | # 开场白 |

候选方案经 DSPy 三维评估（覆盖度/副作用风险/可执行性）后选择最优。

### 5.5 `fewshot_generator.py` — Few-shot 示例生成

**触发条件**：维度评级为"需改进"或"不合格"的对话占比 > 30%（即"合格"以下），或缺陷聚类 frequency ≥ 3。

**LLM prompt 包含**：对话文本（MMR 多样性选择后的片段）+ 缺陷描述 + 评测判定。

LLM 生成正例（正确做法）+ 负例（错误做法 + 标注）+ 修正例（负例的正确版本），格式对齐 ChatML 三元组。输出嵌入在 `optimization_report.md` 的 §四.4.3 节和 `optimization_actions.json` 的 `fewshot_examples` 字段中。

### 5.6 `eval_optimizer.py` — 评测引擎自身优化

**数据消费**：以路径A（规则）为主——直接从当前批次的评分分布、置信度数据、元检查告警中分析评测引擎自身问题，**不需要 ChecklistEvolver 跨批次积累即可运行**。仅在需要文本生成时（Judge prompt 改写、置信度校准方向解读）走路径B，此时才将相关段放入 LLM prompt。

三类优化建议（均路径A，零 LLM 成本）：
- **评分分布异常**：某维度 ≥80% 对话集中在同一评级 → 建议增加清单项或优化 Judge prompt 区分力
- **置信度偏低**：>50% 对话为 unreliable/low → 建议检查 CONFIDENCE 因子配置
- **清单覆盖不足**：meta_check_alerts 中 coverage 告警汇总 → 建议调整 CHECKLIST_SIZE

LLM 分析并生成五类建议：

| 类型 | 规则引擎提供 | LLM 生成 |
|------|------------|---------|
| 清单项校准 | 通过率 > 95% 的项列表、NOT_APPLICABLE > 80% 的项列表、PARTIAL > 30% 的项列表 | 建议具体降权/删除/改写方案 |
| Judge prompt 优化 | 评分分布极度集中的维度列表、区分力低的维度列表 | 建议 Judge prompt 的具体修改文本 |
| 权重校准 | Pearson r 显著偏离（|r|>0.3）→ 路径 A 直接输出数值建议；不显著 → 路径 B LLM 分析是否需要保留现状 | 路径 A: "SOURCE_WEIGHTS['case'] 从 0.6→0.5（r=0.42）"；路径 B: "当前权重与评分相关性弱，建议保持观察" |
| 置信度校准 | 各维度 EvalConfidence 均值与分布 | 建议校准方向 + 标注"建议人工验证" |
| 缺陷转化 | 高频 additional_defects（≥5 次）的聚类结果 | 建议转化为正式清单项，含 item_id/description/weight 建议值 |

### 5.7 报告生成（集成在 `optimizer.py` 中）

报告生成逻辑集成在主编排器的 `_build_markdown_report` 和 `_export` 方法中，输出两个文件 + MANIFEST。

**optimization_report.md 结构**：
```
阅读说明（术语表 + CAI三段式说明）
  - 缺陷信号 / 缺陷主题 / 规则分析 / 深度分析 / 优先级 / 建议等级

一、执行摘要（Top-10 优先级排序表，含分析方式标注）

二、Case 定义优化建议
三、用户画像生成器优化建议
四、对话模型优化建议
五、评测引擎自身优化建议
六、附录：统计数据（缺陷信号/主题/建议数按来源和路径分布）
```

每条建议用 CAI 四段式：宪法原则 → 违规证据 → 批判分析 → 修正建议。违规证据含真实对话文本（如 `T3: 客服直接进入业务内容，未追问身份对应关系`）或评分分布统计，批判分析含根因诊断、路径A关联判断、副作用风险。

**optimization_actions.json**：结构化机器可读，含 action_id/对象/维度/优先级/标题/宪法原则/违规证据/批判分析/修正建议/目标位置/预期效果/实施工作量/等级/分析方式。

**建议分级**（每条建议标注等级）：

| 等级 | 触发条件 | 含义 |
|------|---------|------|
| **强建议** | SAFETY/TASK 不合格触发 SCOPE；关键项否决触发 | 不修则评测结果无效或严重失真 |
| **中建议** | 维度评分偏低但未触发 SCOPE；高频缺陷聚类 ≥3 次 | 修了有明确提升 |
| **弱建议** | 微调参数；单次出现的低置信度缺陷 | 锦上添花，可选择性执行 |

### 5.8 `prompts.py` + `utils.py`

- **prompts.py**：5 组 LLM prompt 模板（prompt 优化、few-shot 生成、case 修改、画像+评测引擎优化、综合去重）。每组 prompt 都有固定结构——原始数据段 + 评测诊断段 + 规则发现段 + 输出格式要求
- **utils.py**：规则引擎工具——bigram 聚类、优先级计算、Pearson 相关性、区分力统计、严重程度分类、prompt 段提取

---

## 六、LLM 调用成本

### 6.1 调用场景（双路径后）

| 阶段 | 路径 A（规则，零 LLM） | 路径 B（LLM） | Token/次（路径 B） |
|------|---------------------|-------------|-------------------|
| Case 优化 | 过严检测/分支缺失判断 | 1 次（描述改写/约束调整/KP 补充） | ~4000 in + ~1500 out |
| 画像优化 | 参数偏差相关性计算/策略触发率统计/信号矛盾计数 | 1 次（CO-STAR 模板优化/策略 prompt 改写/自检阈值校准） | ~5000 in + ~2000 out |
| Prompt 优化 | — | 每 Model 聚类 1 次（最多 5 次，批量 N=5） | ~4000 in + ~3000 out |
| Few-shot 生成 | MMR 多样性选择 | top-3 弱维度（最多 3 次） | ~4000 in + ~2500 out |
| 评测引擎优化 | 通过率异常/区分力指标/Pearson 相关性/高频缺陷聚类 → 直接输出 | 1 次（Judge prompt 改写/置信度因子校准方向） | ~4000 in + ~1500 out |
| 综合去重 | — | 1 次 | ~4000 in + ~1500 out |

### 6.2 典型批次（5 对话 × 1 Case）

| 路径 | 次数 | Token |
|------|------|-------|
| 路径 A（规则） | 若干 | **0** |
| 路径 B（LLM） | Case 1 + 画像 1 + Prompt 3 + Fewshot 1 + 评测 1 + 去重 1 = **8** | **~46,500** |
| **合计** | | **~46,500** |

双路径后比全 LLM 方案节约 ~6K tokens（确定性发现不经过 LLM）。约为评测引擎（9 Judge 并发，每对话 ~45-90K）的 7-10%。

成本控制：`--light`（N=3 候选，Prompt 优化节约 40%）/ `--skip-fewshot`/ `--skip-eval-opt`。

---

## 七、开发步骤

| Phase | 内容 | 产出 | 状态 |
|-------|------|------|------|
| 1 | `__init__.py` + `utils.py`（规则引擎工具） | 聚类/排序/相关性/区分力就绪 | ✅ 完成 |
| 2 | `optimizer.py`（主编排 + 数据管道 + 报告生成 + 证据提取）+ `eval_optimizer.py`（路径A 规则部分）。其余子模块用 stub 占位 | 端到端管道可运行（无 LLM，仅规则输出） | ✅ 完成 |
| 3 | `prompts.py` + `prompt_optimizer.py` + `fewshot_generator.py` | LLM 生成就绪（对话模型 + few-shot） | ✅ 完成 |
| 4 | `case_fixer.py` + `profile_optimizer.py` | 全部四个优化对象就绪 | ✅ 完成 |
| 5 | 用已有评测结果运行 + 人工审查 + 迭代优化 | 质量验证通过，报告含四对象建议 + 真实证据 | ✅ 完成 |

---

## 八、验证

- **功能**：8 源文件可导入、规则引擎工具函数正确、LLM prompt 模板格式正确、端到端贯通
- **质量**：建议含具体文本/数值（非泛化）、链接到评测证据、覆盖四对象全部可优化内容、宏微观建议关联正确
- **效果闭环**：优化建议被执行后，重新运行相同 Case 的评测，对比优化前后的 `total_score_100` 变化——提升 >5 分视为有效，降低 >5 分视为退化需回滚，±5 分范围内视为无显著变化

```bash
python -m src.optimizer.optimizer \
  --input-dir data/exports/{batch_id}/ \
  --output data/optimization/{run_id}/
```

---

## 九、后续迭代

| 版本 | 内容 |
|------|------|
| v1.1 | RPO Experience Replay（跨批次历史回放）+ MetaReflection 元指令积累 |
| v2 | 闭环自动触发 + 可选自动应用 + Lumina 遗传优化 |
| v3 | SFT 训练数据生成 + SPIN 自我对弈微调 |

---

## 十、风险

| 风险 | 缓解 |
|------|------|
| 评测引擎不可靠导致优化方向错误 | EvalConfidence 门控——不可靠评估降权且不标记 actionable |
| LLM 生成的建议数值不准确 | 规则引擎计算的统计值作为 prompt 中的硬数据，LLM 基于硬数据推理而非凭空生成 |
| 原始数据量大导致 prompt 超长 | 对话文本截取关键片段（失败轮次前后各 3 轮）；config.py 只传相关段 |
| 过度优化导致 prompt 越来越长 | 长度上限（+50%）、定期审查 |
| 跨 case 泛化失败 | 按业务线分组；元指令提取；A/B 验证 |

---

## 附录 A: 维度权重

| 维度 | 中文 | 权重 |
|------|------|------|
| SAFETY | 安全合规 | 2.0 |
| TASK_COMPLETION | 任务达成 | 1.8 |
| FLOW_COVERAGE | 流程覆盖 | 1.2 |
| CONSTRAINT | 约束遵守 | 1.0 |
| KNOWLEDGE | 知识准确性 | 1.0 |
| EFFICIENCY | 对话效率 | 0.9 |
| ROLE | 角色一致性 | 0.8 |
| SENTIMENT | 情感适配 | 0.8 |
| OPENING | 开场白合规 | 0.5 |

## 附录 B: 建议输出模板

每条建议在 Markdown 报告中的呈现格式（CAI 宪法式）：

> **### [Case] call_flow Step 1 定义不够具体 — Priority 8.2**
>
> **📜 评测清单项**: identity_verification——是否完成了身份核实步骤？
>
> **📋 违规证据**: T1-T3 对话片段 + MOSTLY_NO 判定（3/5 场对话）
>
> **🔍 批判分析**: Model 缺陷（置信度 0.95）。根因——prompt Step 1 只写"确认接听者是房东本人"，未给出具体确认标准和话术分支。当前 boundary_testing 值（0.82/0.78/0.91）偏高但不是主因
>
> **✏️ 修正建议**（微观参数）: 将 Step 1 的 description 从"确认接听者是房东本人"改为 [含 3 种分支的完整话术模板]，同时将 reference_script 从空补充为 [具体话术示例]
>
> **预期效果**: identity_verification MOSTLY_NO → MOSTLY_YES，SAFETY 不合格 → 良好
>
> **实施工作量**: 低

## 附录 C: 设计参考

| 方案 | 出处 | 借鉴点 |
|------|------|--------|
| DSPy MIPROv2 | Stanford, 2024 | 批量生成候选 → 评估 → 选择最优 |
| TextGrad | Stanford, Nature 2025 | 评测归因 = 文本梯度，按 source 反向传播 |
| Constitutional AI | Anthropic, 2024 | 三段式输出：原则 → 证据 → 批判 → 修正 |
| OPRO | Google DeepMind, ICLR 2024 | LLM prompt 嵌入历史轨迹 |
| MMR/DPP | SIGIR 2024 | Few-shot 多样性选择 |

详细调研见 [optimization_engine_research.md](optimization_engine_research.md)。
