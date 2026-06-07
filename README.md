# 美团外呼对话模型评测系统

对 LLM 对话模型进行全流程自动化多维度评测。基于 15 维参数空间生成多样化用户画像，通过三层评测体系（规则引擎 + 信号提取 + LLM Judge）对模型在外呼场景下的表现进行 9 维度量化评分，并提供双路径优化建议。

---

## 📋 目录

- [系统架构](#系统架构)
- [核心引擎](#核心引擎)
- [快速开始](#快速开始)
- [Web 系统](#web-系统)
- [项目结构](#项目结构)
- [云端部署](#云端部署)
- [评分体系](#评分体系)
- [优化引擎](#优化引擎)
- [设计文档](#设计文档)
- [开发测试](#开发测试)

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户浏览器 / API                           │
│           Tab 演示效果 | Tab 新建评测 | Tab 历史记录              │
└──────────┬──────────────┬──────────────┬─────────────────────────┘
           │ HTTP/REST    │ HTTP/REST    │ WebSocket
           ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI (uvicorn :8000)                       │
│  router.py  │  task_manager.py  │  ws_manager.py                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
┌──────────┐   ┌──────────────┐   ┌──────────────┐
│ Case 解析 │   │  画像生成器   │   │  评测引擎     │
│ 纯函数    │   │  15维参数空间  │   │  3层9维度     │
└──────────┘   └──────────────┘   └──────────────┘
     │                     │                     │
     └─────────────────────┼─────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              对话模拟引擎 (Assistant ↔ Simulator)                 │
│              输出 8 种 XML 标签 + 对话文本                        │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              优化引擎 (双路径: 规则A + LLM B)                     │
│              四对象优化建议 → 报告 + JSON                         │
└─────────────────────────────────────────────────────────────────┘
```

**数据流全景**：Case 文本 → `parse_instruction()` 纯函数解析 → Case 对象 → 画像生成 → 对话模拟 → 评测 → 优化 → 报告

---

## 核心引擎

### 一、用户画像生成器

基于 **15 维连续参数空间** 驱动，通过 **8 阶段管线** 生成多样化用户画像：

| 层级 | 维度 | 范围 |
|------|------|------|
| **A — 大五人格 (5维)** | 宜人性、尽责性、神经质、外向性、开放性 | [0,1] |
| **B — 对话行为 (4维)** | 耐心度、话多程度、礼貌度、主见性 | [0,1] |
| **C — 认知知识 (2维)** | 信息验证倾向、领域知识水平 | [0,1] |
| **D — 情感状态 (2维)** | 初始情绪、情绪波动性 | [0,1] |
| **E — 对抗倾向 (2维)** | 边界试探、前后一致性 | [0,1] |

**8 阶段管线**：

| 阶段 | 方法 | 说明 |
|------|------|------|
| P1 | LHS 全空间采样 | 拉丁超立方采样覆盖 [0,1]^15，默认 2 个全局样本 |
| P2 | 子空间 LHS | 解析 Case 分支条件 → 维度约束子空间采样 + 极端画像 |
| P3 | 对抗策略自动挂钩 | 根据维度值自动判定 5 种对抗策略（探针/注入/矛盾/权威/情绪） |
| P4 | 锚点翻译 | 15 维 × 5 锚点 = 75 段行为锚点，最近锚点 + 偏离度标注 |
| P5 | CO-STAR 框架 | Context/Objective/Style/Tone/Audience/Response 结构化 prompt |
| P6 | Contrastive Prompting | 维度特定的排除法引导（"你不是这样的"） |
| P7 | 自检回路 | LLM 独立重打分 → 欧氏距离验证 → 锚点修正重试 |
| P8 | 三向量存档 | sampled_vector + verified_vector + audited_vector 全链路追踪 |

### 二、对话模拟引擎

被评测模型（LLMAssistant）与模拟用户（UserSimulator）进行多轮外呼对话：

- **输出 8 种 XML 解析标签**：`<memory>` / `<thought>` / `<state>` / `<emotion_curve>` / `<risk_flag>` / `<model_behavior>` / `<conversation_quality>` / `<should_end>`
- **立场-情绪分离**：立场（性格基线）始终不变，情绪随客服行为动态变化
- **两信号终止检测**：连续两轮触发结束信号 → 终止对话；3 轮缓冲无信号 → 重置计数
- **模型崩溃检测**：连续 2 轮质量故障 → 提前终止，标记 `model_breakdown`
- **防崩机制**：`_safe()` 转义花括号，防止 LLM 生成文本导致 `.format()` KeyError

### 三、评测引擎 — 三层体系

**信号增强清单评估（Signal-Augmented Checklist Evaluation）**：将每个维度的评估分解为原子化 YES/NO 核查项，LLM 只做核查不打分，评级由规则推导。

```
Tier 1 (零 LLM)        Tier 1.5 (零 LLM)       Tier 2 (LLM)
┌──────────────┐       ┌──────────────┐       ┌──────────────┐
│ 11 规则指标   │       │ 7 信号提取    │       │ 9 Judge 并发  │
│ · turns_ratio│ ────▶ │ · 满意度轨迹  │ ────▶ │ · 逐条核查清单 │
│ · stuck_count│       │ · 情绪曲线    │       │ · evidence引用 │
│ · branch_cov │       │ · 卡死检测    │       │ · 6级判定      │
│ · word_limit │       │ · 结束意愿    │       │ · 补充缺陷发现  │
└──────────────┘       └──────────────┘       └──────────────┘
```

### 四、优化引擎 — 双路径架构

| 路径 | 方法 | 特点 |
|------|------|------|
| **Path A — 规则引擎** | bigram 聚类 + 相似度分组 + 统计异常检测 | 零 LLM 成本、100% 可复现 |
| **Path B — LLM 深度分析** | DSPy MIPROv2 批量候选 + OPRO 轨迹 + Constitutional AI 三段式 | 具体修改文本、因果推理 |

**四优化对象**：Case 定义 / 用户画像生成器 / 被评测对话模型 / 评测引擎自身

---

## 快速开始

### 环境要求

- **Python 3.10+**
- Windows / macOS / Linux

### 安装

```bash
git clone https://gitee.com/zeyuan-huang/main.git meituan
cd meituan
pip install -r requirements.txt
```

### 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key（或其他 OpenAI 兼容 API）
# API_KEY=sk-xxxxx
# BASE_URL=https://api.deepseek.com
# MODEL=deepseek-chat
```

支持按槽位独立配置模型（可选）：

```bash
# SIMULATOR_MODEL=deepseek-chat    # 模拟器专用模型
# EVALUATOR_MODEL=deepseek-chat    # 评测引擎专用模型
# OPTIMIZER_MODEL=deepseek-chat    # 优化引擎专用模型
```

### 启动 Web 服务

```bash
python run_web.py
# 浏览器打开 http://localhost:8000
# Swagger API 文档: http://localhost:8000/docs
```

### 命令行模式

```bash
# 单个 Case 批量评测
python -m src.simulator.batch_runner --case-id 1 --n-profiles 3 --run-eval

# 含优化建议
python -m src.simulator.batch_runner --case-id 1 --n-profiles 3 --run-eval --run-optimize
```

---

## Web 系统

### 页面功能

| Tab | 名称 | 功能 |
|-----|------|------|
| **Tab 1** | 演示效果 | 浏览预跑评测结果，零 API 消耗、零等待 |
| **Tab 2** | 新建评测 | 输入 Case 文本 + 配置模型 API → 启动真实流水线 |
| **Tab 3** | 历史记录 | 查看、下载、删除历史任务及完整评分报告 |
| **Tab 4** | 关于系统 | 系统简介、评测方法论、使用指南 |

### 技术栈

| 层级 | 技术 |
|------|------|
| 后端框架 | FastAPI (Python) |
| ASGI 服务器 | Uvicorn |
| 前端 | Vue.js 3 (CDN，无构建) |
| CSS | Pico.css (CDN) |
| 实时通信 | WebSocket (原生，无 Redis) |
| 数据校验 | Pydantic v2 |

### 四槽位 LLM 配置

系统支持四个独立的 LLM 槽位，可各自配置不同的 API Key / Base URL / Model：

| 槽位 | 用途 | 默认值来源 |
|------|------|-----------|
| **被评测模型** (Assistant) | 对话中扮演客服角色 | 用户必填 |
| **用户模拟器** (Simulator) | 画像生成 + 对话用户模拟 | 服务端 `.env` |
| **评测引擎** (Evaluator) | 9 维度清单核查 + 行为审计 | 服务端 `.env` |
| **优化引擎** (Optimizer) | 双路径优化建议生成 | 服务端 `.env` |

### 主要 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/presets` | 预置 Case 模板列表 |
| `POST` | `/api/cases/parse` | 解析 Case 文本预览 |
| `POST` | `/api/tasks` | 创建评测任务 |
| `GET` | `/api/tasks` | 历史任务列表 |
| `GET` | `/api/tasks/{id}/detail` | 任务完整详情（含所有画像的评测结果） |
| `GET` | `/api/tasks/{id}/download/{type}` | 文件下载（报告/JSON/ZIP） |
| `POST` | `/api/tasks/{id}/cancel` | 取消运行中的任务 |
| `DELETE` | `/api/tasks/{id}` | 删除历史任务 |
| `POST` | `/api/test-connection` | 测试 LLM API 连通性 |
| `WS` | `/ws/task/{id}` | WebSocket 实时进度推送 |

---

## 项目结构

```
meituan/
├── src/                          # 核心引擎（~12,000 行 Python）
│   ├── loader/                   # Case 解析与加载
│   │   ├── case_parser.py        # parse_instruction() 纯函数解析
│   │   ├── case_loader.py        # 批量 Case 加载
│   │   └── complexity.py         # 复杂度评分（6 因子，0-10）
│   │
│   ├── simulator/                # 用户模拟器
│   │   ├── profile_params.py     # 15 维参数定义 + LHS 采样器 + 75 段锚点
│   │   ├── profile_generator.py  # CO-STAR + Contrastive + 自检回路
│   │   ├── profile_auditor.py    # Path A/B 行为审计 + 偏差归因
│   │   ├── profiles.py           # 参数化画像构建 + 对抗策略指令
│   │   ├── simulator.py          # UserSimulator 对话引擎
│   │   ├── runner.py             # DialogueRunner 编排
│   │   ├── assistant_interface.py # AssistantInterface ABC + LLMAssistant
│   │   ├── output_parser.py      # 8 种 XML 标签解析器
│   │   └── batch_runner.py       # 批量编排入口（Phase 0→1→2→3→4）
│   │
│   ├── eval/                     # 评测引擎（14 文件）
│   │   ├── orchestrator.py       # 主编排：清单生成→LLM 核查→评级→归因→置信度
│   │   ├── schemas.py            # 9 Judge prompt builder + 维度差异化配置
│   │   ├── judge.py              # JudgeExecutor (并发 + AIMD + 熔断)
│   │   ├── rules.py              # Tier 1 (11 规则) + Tier 1.5 (7 信号)
│   │   ├── checklist_generator.py # 三层清单生成（Case + Simulator + LLM）
│   │   ├── checklist_evolver.py  # 清单进化机制（积累→分析→转化→裁剪→校准）
│   │   ├── cross_validator.py    # 规则-LLM 交叉验证（7 种矛盾检测）
│   │   ├── diagnostics.py        # 归因分析（CaseDX/SimDX/ModelDX）
│   │   ├── self_reliability.py   # 无人工标注自验证检查器
│   │   ├── drift_monitor.py      # BatchAnalyzer 批次聚合分析
│   │   ├── report_generator.py   # 叙述性评测报告生成
│   │   └── config.py             # 权重/阈值/评级区间/清单权重配置
│   │
│   ├── optimizer/                # 优化引擎（8 文件）
│   │   ├── optimizer.py          # 主编排 + 报告生成
│   │   ├── case_fixer.py         # Case 设计完善建议
│   │   ├── profile_optimizer.py  # 画像生成器改进建议
│   │   ├── prompt_optimizer.py   # 对话模型 Prompt 优化 (DSPy MIPROv2)
│   │   ├── fewshot_generator.py  # Few-shot 示例生成 (MMR 多样性)
│   │   ├── eval_optimizer.py     # 评测引擎自身优化
│   │   ├── prompts.py            # 5 组 LLM prompt 模板
│   │   └── utils.py              # 规则引擎工具（聚类/排序/相关性/区分力）
│   │
│   ├── llm/                      # LLM 调用封装
│   │   ├── client.py             # LLMClient (OpenAI 兼容接口 + 结构化输出)
│   │   ├── model_manager.py      # 多模型注册/切换
│   │   └── prompts.py            # 参数化 + 对抗策略 prompt 模板
│   │
│   ├── models/                   # 数据模型
│   │   ├── case.py               # Case 数据类
│   │   ├── conversation.py       # Conversation (含 S/V/A 向量 + 一致性 + 分支覆盖)
│   │   └── evaluation.py         # EvalResult + EvalConfidence + AttributionItem
│   │
│   └── utils/                    # 工具
│       ├── data_exporter.py      # 数据导出
│       └── helpers.py            # 辅助函数
│
├── web/                          # Web 层（FastAPI + Vue 3）
│   ├── app.py                    # FastAPI 应用工厂 + CORS + 生命周期
│   ├── config.py                 # 配置（读取 .env）
│   ├── router.py                 # 所有 REST + WS 路由
│   ├── schemas.py                # Pydantic 请求/响应模型
│   ├── task_manager.py           # Task 生命周期 + 流水线编排
│   ├── ws_manager.py             # WebSocket 连接管理
│   ├── demo_case.md              # 演示模式默认 Case
│   └── static/                   # 前端静态文件
│       ├── index.html            # Vue 3 SPA（四 Tab 视图）
│       └── app_v2.js             # Vue 3 响应式应用逻辑
│
├── data/                         # 运行时数据（部分已 .gitignore）
│   ├── exports/                  # 任务输出
│   ├── checklist_evolution/      # 清单进化数据
│   └── conversations/            # 批量对话数据
│
├── tests/                        # 测试
│   ├── test_simulator.py         # 模拟器测试（45 个）
│   ├── test_eval.py              # 评测引擎测试（32 个）
│   └── test_case_parser.py       # Case 解析测试（21 个）
│
├── run_web.py                    # Web 启动入口
├── deploy.sh                     # 云服务器一键部署脚本
├── .env.example                  # 环境变量模板
├── requirements.txt              # Python 依赖
│
└── 设计文档
    ├── fang_an_user.md           # 用户模拟器构建方案
    ├── plan2_pc.md               # 评测引擎构建方案
    ├── optimization_engine_plan.md # 优化引擎 v1 开发计划
    └── web_deployment_plan.md    # Web 部署开发计划
```

---

## 云端部署

### 阿里云/腾讯云轻量服务器（推荐）

**30 分钟内完成部署，新用户免费试用 1 个月**。无需 Docker、Nginx、数据库。

```bash
# 1. 领取服务器 → 开放 8000 端口 → SSH 登录
ssh root@<公网IP>

# 2. 安装依赖 + 拉取代码
apt update && apt install -y python3-pip git
git clone https://gitee.com/zeyuan-huang/main.git meituan
cd meituan

# 3. 配置 + 一键部署
cp .env.example .env && nano .env   # 填入 API_KEY
chmod +x deploy.sh && bash deploy.sh

# 4. 验证
systemctl status meituan-web
# 浏览器: http://<公网IP>:8000
```

### ngrok 本地测试

```bash
# Windows
start_with_ngrok.bat

# 手动
ngrok http 8000
python run_web.py
```

---

## 评分体系

### 9 评测维度

| # | 维度 | 权重 | SCOPE | 说明 |
|---|------|------|-------|------|
| 1 | **SAFETY** 安全合规 | 2.0 | ✅ 封顶50 | 身份核实、敏感信息处理、合规声明 |
| 2 | **TASK_COMPLETION** 任务完成度 | 1.8 | ✅ 封顶60 | Call Flow 步骤逐项覆盖、必要信息传递 |
| 3 | **FLOW_COVERAGE** 流程覆盖 | 1.2 | — | 分支覆盖率、步骤跳转合理性 |
| 4 | **CONSTRAINT** 约束遵守 | 1.0 | — | 字数限制/禁止词/语气/行为约束逐条检查 |
| 5 | **KNOWLEDGE** 知识准确性 | 1.0 | — | FAQ 准确度、知识点调用时机 |
| 6 | **EFFICIENCY** 对话效率 | 0.9 | — | 轮次数、冗余度、信息密度 |
| 7 | **ROLE** 角色一致性 | 0.8 | — | 客服身份一致性、话语风格 |
| 8 | **SENTIMENT** 情感适配 | 0.8 | — | 情绪识别准确度、共情回应质量 |
| 9 | **OPENING** 开场白合规 | 0.5 | — | 开场白完整性、信息覆盖 |

### 五级评级

| 评级 | 百分制 | 说明 |
|------|--------|------|
| 🏆 卓越 | ≥ 90 | 所有步骤充实执行 + 超出预期 |
| ✅ 良好 | ≥ 70 | 步骤基本完整 + 1-2 处轻微不足 |
| ⚠️ 合格 | ≥ 50 | 步骤有遗漏但核心完成 |
| 🔶 需改进 | ≥ 30 | 多处遗漏或错误 |
| ❌ 不合格 | < 30 | 核心步骤缺失或安全违规 |

**SCOPE 一票否决**：SAFETY 不合格 → 总分封顶 50；TASK_COMPLETION 不合格 → 总分封顶 60。

**评分换算**：`total_score_100 = (raw_score - 9.0) / (85.5 - 9.0) × 100`（原始分 max=85.5，百分制 max=100）。

### EvalConfidence 评测可信度

综合 **16+ 因子** 计算每场对话的评测可信度（清单-信号一致性 / evidence 质量 / Simulator 质量 / Judge 间一致性 / 子维度一致性 / 清单项数 / 对话长度 / PARTIAL 浓度 / 元检查 / 交叉验证等），输出 `high` / `medium` / `low` / `unreliable` 四级。`is_reliable=False` 的对话不参与统计和优化决策。

---

## 优化引擎

### 双路径架构

| 路径 | 方法 | 成本 | 产出 |
|------|------|------|------|
| **A — 规则引擎** | bigram 聚类 + Pearson 相关性 + 评分分布异常检测 | 零 LLM | 确定性发现（数值异常/统计结论/高频模式） |
| **B — LLM 深度分析** | DSPy MIPROv2 批量候选(N=5) + OPRO 轨迹 + CAI 三段式 | ~5-7K tokens/次 | 具体修改文本 + 因果推理 + 副作用评估 |

### 四对象优化

| 优化对象 | 可优化内容 |
|---------|-----------|
| **Case 定义** | call_flow 步骤/分支、constraints 约束、knowledge_points 知识点、opening_line 开场白 |
| **用户画像生成器** | 15 维参数锚点、5 对抗策略 prompt 及触发阈值、CO-STAR 模板、自检阈值 |
| **被评测对话模型** | System Prompt 各段文本、raw_instruction 全文、few-shot 示例 |
| **评测引擎自身** | 清单项增删改、维度权重校准、置信度因子调整、Judge prompt 优化 |

### 建议等级

| 等级 | 触发条件 |
|------|---------|
| **强建议** | SAFETY/TASK 不合格触发 SCOPE 否决；关键项否决触发 |
| **中建议** | 维度评分偏低但未触发 SCOPE；高频缺陷聚类 ≥ 3 次 |
| **弱建议** | 微调参数；单次出现的低置信度缺陷 |

---

## 设计文档

| 文档 | 内容 | 行数 |
|------|------|------|
| [fang_an_user.md](fang_an_user.md) | 用户模拟器构建方案：15 维参数空间、8 阶段管线、对话运行时标签体系 | ~560 |
| [plan2_pc.md](plan2_pc.md) | 评测引擎构建方案：三层清单体系、9 Judge 维度、EvalConfidence、清单进化 | ~1,800 |
| [optimization_engine_plan.md](optimization_engine_plan.md) | 优化引擎 v1 开发计划：双路径架构、四对象优化、CAI 三段式报告 | ~550 |
| [web_deployment_plan.md](web_deployment_plan.md) | Web 部署开发计划：技术选型论证、四槽位模型、部署方案 | ~1,090 |

---

## 开发测试

### 运行测试

```bash
# 全部测试（~72 个测试用例）
pytest tests/ -v

# 分类运行
pytest tests/test_simulator.py -v     # 模拟器测试（45 个）
pytest tests/test_eval.py -v          # 评测引擎测试（32 个）
pytest tests/test_case_parser.py -v   # Case 解析测试（21 个）
```

### 代码统计

```
语言: Python 3.10+
源文件: ~68 个 Python 文件
代码行: ~16,000 行
测试: ~72 个测试用例
依赖: 10 个 pip 包
纯 CPU 项目，无 GPU 需求
```

---

## License

本项目仅供学习和研究使用。
