# 美团 AI Hackathon — 复杂指令多轮对话评测系统 实施方案

## Context

| 项目 | 内容 |
|------|------|
| 赛道 | 复杂指令下的多轮对话评测系统 |
| 场景 | 履约数字人外呼 — 系统发起通话，对话模型按指令完成任务 |
| 交付 | ①用户模拟器 ②可解释可量化评测报告 |
| 数据 | 60条虚拟示例指令，覆盖18+美团业务线 |

---

## 总路线图

```
Step 1: 基础设施 ─── Step 2: 对话模拟引擎 ─── Step 3: 评测引擎 ─── Step 4: 报告生成
     (当前)              (下一步)                (下下一步)            (最后)

每个 Step 完成后可独立验证，不依赖后续 Step。
```

### 深入方向总览（12个，按实施阶段分布）

| # | 方向 | 融入阶段 | 优先级 | 一句话价值 |
|---|------|---------|--------|----------|
| 1 | 指令复杂度量化：6因子加权公式 | Step1/3 | **必做** | 高复杂度指令低分 ≠ 模型差 |
| 2 | 多维评测（含多轮特有指标） | Step3 | **必做** | 信息持久性/错误自修复别人测不到 |
| 3 | 对抗性用户模拟 | Step2 | **必做** | 专门找模型弱点 |
| 4 | 根因自动归类 | Step3 | **必做** | 不是扣分，是解释为什么扣分 |
| 5 | 多Judge交叉验证 | Step3 | 强烈推荐 | 评测可靠性自证 |
| 6 | 情感/状态曲线 | Step4 | 强烈推荐 | 报告中最亮眼的可视化 |
| 7 | 多粒度聚合分析 | Step4 | 强烈推荐 | 发现隐藏规律 |
| 8 | 评测成本账本 | Step4 | 推荐 | 生产实用主义 |
| 9 | 指令脆弱点发现 | Step3/4 | 推荐 | 反哺指令设计 |
| 10 | 遗漏检测 | Step3 | 推荐 | 测"该做没做" |
| 11 | 反事实分析 | Step4 | 加分 | 学术感最强 |
| 12 | 指令回归测试 | Step4 | 加分 | CI/CD 级思考 |

---

## Step 1: 基础设施 — 指令解析 + 数据模型 + LLM Client

**目标**：把60条非结构化文本指令变成程序可消费的结构化数据，搭好LLM调用基底。

**验证标准**：`python case_loader.py` 跑完输出 30 条结构化 Case 对象，`python -m pytest` 通过所有单元测试。

### 1.1 目录结构

```
美团/
├── .env                          # API Key 配置
├── requirements.txt              # 依赖清单
├── main.py                       # 全链路入口（Step4 串起来）
│
├── data/                         # 原始数据 + 中间产物
│   ├── cases_raw/                # 4个 generated_cases.json (已有)
│   ├── cases_parsed.json         # Step1 输出：60条结构化Case
│   ├── conversations/            # Step2 输出：120场对话JSON
│   ├── evaluations/              # Step3 输出：120份评测结果JSON
│   └── report.html               # Step4 输出：最终评测报告
│
├── src/
│   ├── __init__.py
│   ├── config.py                 # 全局配置（从 .env 读取）
│   │
│   ├── loader/                   # Step1: 指令解析
│   │   ├── __init__.py
│   │   ├── case_parser.py        # 正则解析单条指令 → Case
│   │   ├── case_loader.py        # 批量读取4个JSON → 合并 → 解析 → 输出
│   │   └── complexity.py         # 指令复杂度量化
│   │
│   ├── models/                   # Step1: 数据模型：定义系统中"什么东西长什么样"
│   │   ├── __init__.py
│   │   ├── case.py               # Case / CallFlowStep / Constraint / KnowledgePoint
│   │   ├── conversation.py       # Turn / Conversation
│   │   └── evaluation.py         # EvalResult / DimensionScore / RootCause
│   │
│   ├── llm/                      # Step1: LLM 调用封装
│   │   ├── __init__.py
│   │   ├── client.py             # OpenAI兼容SDK封装（重试/限流/结构化输出）
│   │   └── prompts.py            # 所有 prompt 模板集中管理
│   │
│   ├── simulator/                # Step2: 用户模拟器
│   │   ├── __init__.py
│   │   ├── profiles.py           # 4种用户画像 + 5类对抗场景
│   │   ├── simulator.py          # UserSimulator
│   │   ├── runner.py             # DialogueRunner（单场）
│   │   └── batch_runner.py       # BatchRunner（并行120场）
│   │
│   ├── evaluator/                # Step3: 评测引擎
│   │   ├── __init__.py
│   │   ├── rule_checks.py        # 规则检查器（字数/禁词/竞品/开场白）
│   │   ├── flow_evaluator.py     # 流程覆盖评估
│   │   ├── constraint_evaluator.py # 约束遵守评估
│   │   ├── knowledge_evaluator.py # 知识点准确性评估
│   │   ├── role_evaluator.py     # 角色一致性评估
│   │   ├── root_cause.py         # 根因自动归类
│   │   ├── omission_check.py     # 遗漏检测
│   │   └── judge.py              # LLM Judge 统一接口
│   │
│   └── report/                   # Step4: 报告生成
│       ├── __init__.py
│       ├── aggregator.py         # pandas 统计/聚合/交叉分析
│       ├── template.html         # Jinja2 报告模板
│       └── generator.py          # HTML 报告生成
│
└── tests/
    ├── __init__.py
    ├── test_case_parser.py
    ├── test_case_loader.py
    ├── test_complexity.py
    ├── test_simulator.py
    ├── test_evaluator.py
    └── test_report.py
```

### 1.2 数据模型

```python
# src/models/case.py

@dataclass
class CallFlowStep:
    id: str                    # e.g. "step_1"
    step_number: int           # 1, 2, 3...
    title: str                 # e.g. "身份确认"
    description: str           # 该步骤要做什么
    branching: List[Branch]    # 分支条件
    sub_steps: List[str]       # 子步骤描述
    reference_script: str      # 参考话术
    is_optional: bool          # 是否为可选步骤

@dataclass
class Constraint:
    id: str
    type: str                  # "word_limit" | "forbidden_word" | "tone" | "behavior" | "safety" | "other"
    description: str           # 原文
    checkable_by_rule: bool    # 能否规则检查
    rule_pattern: Optional[str] # 正则（如果可规则检查）

@dataclass  
class KnowledgePoint:
    id: str
    topic: str                 # 主题
    content: str               # 标准答案

@dataclass
class Case:
    id: int
    title: str                 # 从 Task 提取的简短标题
    business_line: str         # 业务线：外卖/酒店/闪购...
    role: str
    task: str
    opening_line: str
    call_flow: List[CallFlowStep]
    knowledge_points: List[KnowledgePoint]
    constraints: List[Constraint]
    complexity_score: float    # 复杂度评分 (0-10)
    raw_instruction: str       # 原始文本
```

### 1.3 指令解析策略

60条指令有**两种格式**：
- 格式A（#1-#20）：Markdown 风格 `# Role` / `# Task` / `# Call Flow` / `# Constraints`
- 格式B（#21-#30）：更结构化的 Markdown `# Role:` / `## Task:` / `## Step 1:` / `### 3.1 小标题`

解析引擎按优先级尝试多个解析器，fallback 到 LLM 辅助解析（仅对无法正则解析的部分）。

```
解析流程:
  1. 正则提取 Role → 总能成功
  2. 正则提取 Task → 总能成功
  3. 正则提取 Opening Line → 总能成功
  4. 正则提取 Call Flow → 格式A用缩进层级，格式B用 ## Step 标记
  5. 正则提取 Knowledge Points → 以 "- " 开头 + FAQ 关键字锚定
  6. 正则提取 Constraints → 以 "- " 开头 + 约束关键字锚定
  7. 提取失败的部分 → 用 LLM 从原文中补提取（fallback，仅在正则失败时调用）
```

### 1.4 指令复杂度量化

```python
# src/loader/complexity.py

def calculate_complexity(case: Case) -> float:
    """量化指令复杂度 (0-10)"""
    score = 0.0
    
    # 1. 流程分支数 (0-3分)
    branch_count = count_total_branches(case.call_flow)
    score += min(branch_count / 5, 1.0) * 3
    
    # 2. 约束数量 (0-2分)
    score += min(len(case.constraints) / 10, 1.0) * 2
    
    # 3. 约束类型多样性 (0-1.5分)
    unique_types = len(set(c.type for c in case.constraints))
    score += min(unique_types / 5, 1.0) * 1.5
    
    # 4. 知识点数量 (0-1分)
    score += min(len(case.knowledge_points) / 6, 1.0) * 1.0
    
    # 5. 流程步骤数 (0-1.5分)
    score += min(len(case.call_flow) / 7, 1.0) * 1.5
    
    # 6. 是否有嵌套分支 (0-1分)
    if has_nested_branching(case.call_flow):
        score += 1.0
    
    return round(min(score, 10.0), 1)
```

### 1.5 LLM Client

```python
# src/llm/client.py

class LLMClient:
    """OpenAI兼容接口封装"""
    
    def __init__(self, model: str, temperature: float = 0.0, max_retries: int = 3):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
    
    async def chat(self, system_prompt: str, user_message: str, 
                   response_format: str = "text") -> str:
        """单次 LLM 调用，内置重试"""
        ...
    
    async def chat_structured(self, system_prompt: str, user_message: str,
                              schema: dict) -> dict:
        """返回结构化 JSON，带 schema 校验"""
        ...
```

两个实例：
- `simulator_client = LLMClient(model="deepseek-chat", temperature=0.7)` — 模拟器
- `judge_client = LLMClient(model="deepseek-chat", temperature=0.0)` — 评测

### 1.6 Step 1 任务清单

- [ ] 创建 `.env` 文件（API_KEY / BASE_URL / MODEL）
- [ ] `requirements.txt`（openai, pandas, python-dotenv, httpx, jinja2, openpyxl）
- [ ] `src/config.py` — 统一配置入口
- [ ] `src/models/case.py` — Case / CallFlowStep / Constraint / KnowledgePoint 数据类
- [ ] `src/models/conversation.py` — Turn / Conversation 数据类
- [ ] `src/models/evaluation.py` — EvalResult / DimensionScore / RootCause 数据类
- [ ] `src/loader/case_parser.py` — 正则解析器
- [ ] `src/loader/case_loader.py` — 批量加载 + 合并4个JSON + 解析 + 输出 cases_parsed.json
- [ ] `src/loader/complexity.py` — 复杂度量化
- [ ] `src/llm/client.py` — LLM 调用封装
- [ ] `src/llm/prompts.py` — Prompt 模板
- [ ] `tests/test_case_parser.py` — 覆盖两种格式 + 边界case
- [ ] `tests/test_case_loader.py` — 端到端验证60条全部解析成功

### 1.7 Step 1 验证清单

```bash
# 1. 跑通解析
python -m src.loader.case_loader

# 预期输出:
#   解析完成: 60/60 条成功
#   复杂度分布: 低(0-3) 0条 / 中(3-7) 41条 / 高(7-10) 19条
#   输出: data/cases_parsed.json

# 2. 跑测试
python -m pytest tests/ -v

# 3. 随便挑一条打印
python -c "
from src.loader.case_loader import load_cases
cases = load_cases()
c = cases[2]
print(f'ID: {c.id}')
print(f'业务线: {c.business_line}')
print(f'复杂度: {c.complexity_score}')
print(f'流程步骤数: {len(c.call_flow)}')
print(f'约束数: {len(c.constraints)}')
print(f'知识点数: {len(c.knowledge_points)}')
for step in c.call_flow:
    print(f'  Step{step.step_number}: {step.title} (分支数: {len(step.branching)})')
"
```

---

## Step 2 预览: 对话模拟引擎（下一步）

- 4 种用户画像 + 5 类对抗场景 Prompt
- UserSimulator: profile + history → next utterance
- DialogueRunner: Assistant(被评测模型) ↔ UserSimulator 交替，终结检测
- BatchRunner: asyncio 并发 60×4=240 场，全部存 JSON

## Step 3 预览: 评测引擎（下下一步）

- 规则检查（字数/禁词/竞品/开场白）→ 零成本 100% 准确
- 9 维度评测：流程覆盖 + 约束遵守 + 知识准确 + 角色一致 + 任务完成 + 信息持久性 + 流程韧度 + 错误自修复 + 轮转合理性
- LLM Judge + 多 Judge 交叉验证（可靠性自证）
- 根因自动归类（情绪触发/流程纠缠/知识盲区/上下文丢失/角色漂移）
- 遗漏检测（对比已覆盖 vs 应覆盖知识点）
- 约束冲突雷达（发现互斥约束对）

## Step 4 预览: 报告生成（最后）

- 多粒度聚合（业务线×复杂度×画像×轮次×约束类型）
- 情感/状态曲线可视化
- 评测成本账本
- 指令脆弱点发现 + 优化建议
- HTML 自包含报告 + JSON 原始数据
