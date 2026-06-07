# LLM 评测系统 Demo Web 部署开发计划书

> 版本: v2.0 | 日期: 2026-06-06 | 状态: 待实施

---

## 一、项目背景与目标

### 1.1 当前状态

本项目是一个完整的 LLM 评测系统（~17,800 行 Python），包含五大引擎：

| 引擎 | 模块 | 耗时 | 说明 |
|------|------|------|------|
| Case 解析 | `src/loader/` | 秒级 | `parse_instruction()` 是纯函数，接受任意 Markdown 文本 → 结构化 `Case` 对象 |
| 画像生成 | `src/simulator/` (profile) | 30s-2min | 15 维参数采样 → LLM 生成画像文本 → 自检回路 |
| 对话模拟 | `src/simulator/` (dialogue) | 1-5min | 被评测模型(LLM) ↔ 用户模拟器(LLM) 多轮对话 |
| 评测 | `src/eval/` | 1-3min | 9 维度清单检查 → 规则引擎 + LLM Judge → 百分制评分 |
| 优化（可选）| `src/optimizer/` | 1-2min | 双路径 (规则+LLM) → 四对象优化建议 |

**关键发现**: `parse_instruction()` 是纯函数——接受任意 raw text string，不依赖文件 IO。`BatchRunner.__init__()` 直接接受 `List[Case]`，不强制文件加载。这意味着 Web 层可以直接将用户输入的 Case 文本解析后传入流水线，**不需要修改任何 `src/` 代码**。

当前系统为 **CLI-first**，无任何 Web 代码。

### 1.2 Web 应用完整流程（核心变更）

```
┌─────────────────────────────────────────────────────────────────────┐
│                        用户操作（浏览器）                              │
│                                                                     │
│  Step 1: 选择模式（默认演示模式，可切自定义）                           │
│  Step 2: 输入/选择 Case 文本（Markdown 格式，演示模式已预填）            │
│  Step 3: 配置模型 API（四槽位，演示模式已预设、自定义模式可独立覆盖）     │
│  Step 4: 设置参数（画像数量、是否评测、是否优化）→ 点击「开始评测」       │
│                                                                     │
│                        系统自动执行（后端）                              │
│                                                                     │
│  Phase 0: parse_instruction(text) → Case 对象                        │
│  Phase 1: 画像生成（Simulator 槽位 gen_client）                        │
│  Phase 2: 对话模拟（Assistant 槽位 ↔ Simulator 槽位 sim_client）       │
│  Phase 3: 评测（Evaluator 槽位 eval_client + audit_client）           │
│  Phase 4: 优化建议生成（可选，Optimizer 槽位 llm_client）               │
│                                                                     │
│  → 实时推送进度（WebSocket，日志含模型名称）→ 展示评分 + 下载            │
└─────────────────────────────────────────────────────────────────────┘
```

**与 v1 计划的关键区别**:
| 维度 | v1 (旧) | v2 (新) |
|------|---------|---------|
| Case 来源 | 从 60 条预置 JSON 中选择 | **用户输入外部 Case 文本** |
| 被评测模型 | 使用服务端 `.env` 的统一模型 | **用户提供自己的模型 API** |
| 启动方式 | 选择 Case ID → 开始 | **输入文本 + 配置 API → 开始** |
| 预置 Case | 唯一数据源 | 保留为「快速体验」可选模板 |

### 1.3 评委评审场景设计

#### 演示模式：预跑结果展示

你（开发者）在评审前提前完成：

```
评审前一天:
  打开网站 → 用 Case id=1 跑 1 case × N 画像
           → 用 Case id=2 跑 1 case × N 画像
           → 结果自动存入历史记录

评审当天评委打开网站:
  Tab 1「演示效果」→ 直接看到两条预跑结果
  ├─ Case 1 完整评分报告（9 维度明细 + 对话记录 + 优化建议）
  ├─ Case 2 完整评分报告
  └─ 评委浏览、下载
  → 零 API 调用、零等待、无限次查看
```

**演示模式不是跑流水线，是展示已经跑好的结果。** 评委通过预跑结果了解系统能产出什么、报告长什么样、评了什么维度。

#### 自定义模式：评委真体验

```
评委切到 Tab 2「自定义评测」→ 输入自己的 Case 文本 + 被评测模型 API
→ 点「开始评测」→ 真实流水线跑起来 → 实时进度 → 自己的评分报告
→ API 费用: 被评测模型走评委自己的 Key，模拟器/评测器走服务端 .env
```

**评委想深度体验 → 自带 Case + 自带模型 API → 得到真实评测结果。** 服务端模拟器和评测器仍用你的 `.env` Key，但这两个调用量不大且使用门槛（评委需自带 Key）天然限制了频率。

#### 为什么这样设计

| | 演示模式 | 自定义模式 |
|------|------|------|
| 目的 | 快速了解系统能力 | 深度体验评测流程 |
| 评委操作 | 浏览 + 下载 | 输入 Case + API → 启动 |
| API 费用 | **¥0**（预跑数据） | 评委的 Key（被评测模型）+ 少量服务端 Key |
| 等待时间 | 0 | 8-10 分钟 |
| 数据 | 预跑的 2 条 Case | 评委自己的 Case |

### 1.4 核心原则

- **纯增量**: 不修改任何现有 `src/` 代码，Web 层为独立 `web/` 包
- **轻量化**: 不引入数据库、消息队列、Redis 等中间件
- **演示模式 = 预跑结果展示**: 评审前用 2 条 Case 预跑并存储结果，评委直接浏览，零 API 消耗
- **自定义模式 = 评委真体验**: 评委自带 Case 和被评测模型 API，真实跑流水线

---

## 二、技术选型

### 2.1 后端框架: FastAPI — 全面对比论证

#### 核心约束: WebSocket 不可妥协

本项目的 Demo 体验核心在于「评委能实时看到流水线进展」。轮询方式延迟高且浪费资源。WebSocket 在流水线每个阶段切换时即时推送，这是必须满足的硬需求。

#### 候选框架逐项对比

| 能力 | 本项目需求 | Django | Django+Channels | Flask | Flask+SocketIO | Sanic | **FastAPI** |
|------|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 原生 async | ✅ | ❌ | ⚠️ 部分 | ❌ | ❌ | ✅ | ✅ |
| 原生 WebSocket | ✅ | ❌ | ✅ 需Redis | ❌ | ⚠️ Long-polling 降级 | ✅ | ✅ |
| 自动 API 文档 | ✅ 演示加分 | ❌ 需插件 | ❌ | ❌ | ❌ | ❌ | ✅ Swagger |
| Pydantic 数据校验 | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ 原生 |
| `asyncio.to_thread` | ✅ | ❌ | ⚠️ | ❌ | ❌ | ✅ | ✅ |
| Windows 兼容 | ✅ | ⚠️ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| 社区+中文资料 | — | 大 | 小 | 大 | 中 | 小 | 大 |
| 学习曲线 | — | 陡 | 很陡 | 平 | 平 | 中 | 中 |
| 最终判定 | — | ❌ 过重 | ❌ 太重 | ❌ 无原生WS | ❌ WS不可靠 | ⚠️ 生态弱 | ✅ 最优 |

#### 逐项淘汰分析

- **Django / Django+Channels**: ORM、模板引擎、admin 等 80% 功能本项目不需要。Channels 支持 WebSocket 但必须配 Redis 做 channel layer，增加不必要的中间件依赖和运维复杂度。

- **Flask / Flask+SocketIO**: Flask 本身不支持 async 和 WebSocket。Flask-SocketIO 在浏览器不支持 WebSocket 时会**降级为 HTTP long-polling**——这不是真正的 WebSocket。在 Windows 上的稳定性也有社区反馈问题。

- **Sanic**: 性能与 FastAPI 相当，也原生支持 WebSocket。但 GitHub stars 约为 FastAPI 的 1/4，中文文档和社区案例少很多。遇到问题的排查成本高。

- **aiohttp**: 太过底层。需要手动处理路由注册、请求校验、API 文档生成——相当于自己造 FastAPI 已经提供的轮子。

- **Litestar**: FastAPI 之外最有竞争力的现代框架，但社区规模差距大（Stars 差 7 倍），中文资料几乎为零。

#### 结论: FastAPI 对本项目是最优解

三条核心原因在别的框架上无法同时满足：

| 优势 | 为什么关键 | 其他框架如何 |
|------|-----------|-------------|
| **WebSocket 一等公民** | 实时进度是 Demo 核心亮点 | Django/Flask 需要额外组件且不可靠 |
| **asyncio.to_thread()** | BatchRunner 同步调用不阻塞 event loop，一行代码搞定 | 同步框架需要手动管理线程池 |
| **Swagger UI (/docs)** | 评委可浏览+测试所有 API，展示工程化水平 | Sanic/aiohttp 需要手写文档 |

#### 潜在劣势与对策

| 潜在问题 | 对策 |
|----------|------|
| 同步 LLM 调用阻塞 event loop | `asyncio.to_thread()` 放入线程池（复用已有 `ThreadPoolExecutor`） |
| WebSocket 断线后状态丢失 | Task 状态存内存 + 重连后推送当前快照 |
| Windows 上 asyncio 特性差异 | 本项目使用的 `asyncio.Queue`/`to_thread`/WebSocket 在 Windows 上均正常 |

### 2.2 前端方案: 深度评估

Demo 的前端需要处理的状态并不简单:

- **四步流程状态机**: idle → config → running → completed，每步有独立的校验逻辑
- **WebSocket 实时数据流**: 进度事件、日志流、对话卡片、阶段切换 —— 多处 UI 需要响应同一数据
- **表单校验联动**: Case 文本 → 实时解析预览 → API 连通性测试 → 参数面板，多步之间有依赖
- **评分数据可视化**: 汇总表、维度明细、评级分布

#### 四个候选方案对比

| 维度 | Vanilla JS | **Vue.js (CDN)** | Alpine.js | React (Vite) |
|------|:--:|:--:|:--:|:--:|
| 响应式状态管理 | ❌ 手动 DOM | ✅ `ref/reactive` | ✅ `x-data` | ✅ hooks |
| 组件化 | ❌ 需自建 | ✅ SFC/模板 | ⚠️ 有限 | ✅ 完整 |
| WebSocket 集成 | ⚠️ 手动 | ✅ watch+更新 | ⚠️ 手动 | ✅ useEffect |
| 构建工具 | 不需要 | 不需要 | 不需要 | 需要 npm/vite |
| 文件体积 | 0 | ~33KB (gzip) | ~15KB (gzip) | ~40KB+ (gzip) |
| 学习曲线 | 低 | 低 | 低 | 中高 |
| 可维护性 | ⚠️ 随复杂度下降 | ✅ 好 | ⚠️ 中 | ✅ 最好 |
| 适合本项目 | ⚠️ 够用但代码质量低 | ✅ 最佳平衡 | ⚠️ 勉强 | ❌ 过度设计 |

#### 逐项分析

- **Vanilla JS (原方案)**: 技术上可行，但四步流程 + WebSocket 实时更新 + 多维数据展示的复杂度下，原生 JS 会导致大量 `document.getElementById`、手动状态同步、DOM 操作分散在各处。代码行数可能膨胀到 800-1000 行难以维护的脚本。**评委视角下，如果前端交互出现细微 bug（如进度条不更新、数据不同步），会严重影响 Demo 印象。**

- **Alpine.js**: 比 Vanilla JS 好，提供了响应式数据绑定。但处理 WebSocket 事件流和复杂状态机时语法会变得笨拙（大量的 `x-on`/`x-effect` 嵌套）。对于有四步流程的复杂 UI，Alpine 的模板会显得很"碎"。

- **React (Vite)**: 功能最强，但需要 Node.js + npm + 构建步骤。对 Demo 项目引入了不必要的工具链复杂度。而且构建后的静态文件需要通过 reverse proxy 或额外配置才能被 FastAPI 托管。

- **Vue.js (CDN, 无构建)**: 只需要一行 `<script src="https://unpkg.com/vue@3/dist/vue.global.prod.js">`，零构建工具。提供了完整的响应式系统 (`ref`/`reactive`/`computed`/`watch`)，WebSocket 数据更新后 UI 自动响应。四步流程可以用 `v-if` 干净地切换。`v-for` 优雅处理对话卡片列表。后端仍然是 FastAPI 托管静态文件，架构不变。**代码量预计从 Vanilla JS 的 800+ 行降到 400-500 行，且可读性和可维护性大幅提升。**

#### 最终建议：Vue.js (CDN) 替代 Vanilla JS

| 对比 | Vanilla JS | Vue.js CDN |
|------|-----------|------------|
| 构建工具 | 不需要 | 不需要（同样） |
| 静态文件 | 可直接托管 | 可直接托管（同样） |
| 新增依赖 | 无 | 1 行 CDN `<script>` |
| 代码量 | ~800 行 | ~500 行 |
| 状态管理 | 手动，易出错 | 响应式，自动同步 |
| 评委体验影响 | 可能存在小 bug | 更流畅、更可靠 |

**这是零成本的提升**——不需要 npm、不需要构建、不需要改变部署架构。只是在原来 `app.js` 的基础上使用 Vue 的响应式系统来组织代码。

### 2.3 部署平台: 全面对比分析

#### 2.3.1 Vercel / Netlify 可行性分析（结论: ❌ 均不可行）

很多开发者会首先想到 Vercel 或 Netlify 这类「一键部署」平台。但本项目的架构需求与它们的底层模型存在根本性冲突。

**核心矛盾: Serverless 函数 vs 长连接有状态服务**

```
┌─────────────────────────────────────────────────────────────────┐
│  Vercel / Netlify 的架构（Serverless 函数）                       │
│                                                                 │
│  浏览器请求 → CDN → Serverless Function → 返回结果 → 销毁实例     │
│              ↑                    ↑                             │
│          无状态、短生命周期       每次请求新建实例，函数间不共享状态   │
│                                                                 │
│  适合场景: API 请求/响应（毫秒~分钟级），静态网站，SSR              │
├─────────────────────────────────────────────────────────────────┤
│  本项目需要的架构（持久服务器进程）                                  │
│                                                                 │
│  浏览器 ←── WebSocket 长连接 ──→ 持久服务器进程（uvicorn）          │
│          ←── 30分钟流水线 ──→   状态在内存中保持（Task 对象）        │
│          ←── 实时进度推送 ──→   事件队列持续工作                    │
│                                                                 │
│  适合场景: 实时通信、长时间任务、有状态服务                          │
└─────────────────────────────────────────────────────────────────┘
```

**这是两种完全不同的架构模型，不可调和。**

**Netlify：三振出局 ❌**

| 致命缺陷 | 详情 |
|----------|------|
| **不支持 Python 运行时** | Netlify Functions 仅支持 JS/TS/Go。Python 只能在构建阶段使用，不能处理运行时请求。FastAPI 根本无法部署 |
| **不支持 WebSocket** | AWS Lambda 架构的硬限制，官方明确不支持 |
| **同步函数最长 26 秒** | Pro 套餐也只能调到 26 秒，Background Functions 最长 15 分钟但同样不支持 Python |

**Vercel：比 Netlify 强，但仍不够 ❌**

Vercel 至少支持 Python 运行时，可以将 FastAPI 部署为 Serverless Function：

| 能力 | Vercel | 本项目需求 | 判定 |
|------|--------|-----------|:--:|
| Python 运行时 | ✅ 原生支持 | ✅ | ✅ |
| FastAPI 部署 | ✅ 通过 vercel.json 配置 | ✅ | ✅ |
| **WebSocket** | ❌ **不支持** | ✅ **核心交互必须有** | ❌ |
| 函数超时 | ⚠️ 最长 800s (Pro, 13.3min) | 最长 30min | ❌ |
| 空闲连接 | ❌ 340s 强制断开 | 需维持 30min | ❌ |
| 持久服务器进程 | ❌ 无状态短生命周期 | WebSocket 需持久连接 | ❌ |

**结论: Vercel、Netlify 及同类 Serverless 平台（AWS Lambda + API Gateway、阿里云函数计算 FC、腾讯云 SCF）均不适合本项目。必须使用能运行持久服务器进程的平台。**

#### 2.3.2 可行部署平台全面对比

| 维度 | 阿里云/腾讯云轻量 | Fly.io | Render | Zeabur.cn | Azure Container Apps | Railway |
|:------|:----------:|:------:|:------:|:------:|:--------------------:|:-------:|
| **WebSocket** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **30min 流水线** | ✅ 无限制 | ✅ 无限制 | ✅ 无限制 | ✅ 无限制 | ✅ 可配至 60min | ✅ 无限制 |
| **Python 原生** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **免费额度** | ✅ 免费1月 | ✅ $5/月信用额 | ❌ $7/月起 | ✅ $5/月免费额 | ✅ 慷慨免费额度 | ❌ $5 试用 |
| **月费(演示级)** | ¥0 | $0-$5 | $7 | $0-$5 | $0-$2 | $5+ |
| **需 Docker** | ❌ | ✅ | ❌ | ❌(自动) | ✅ | ❌(可选) |
| **中国网络友好** | ✅ 国内低延迟 | ⚠️ 新加坡~80ms | ❌ 延迟高 | ⚠️ 海外 | ✅ 中国区域 | ⚠️ 海外 |
| **部署难度** | ⭐⭐ SSH+pip | ⭐⭐⭐ Docker | ⭐⭐ Git push | ⭐⭐ Git push | ⭐⭐⭐ Docker | ⭐⭐ Git push |
| **git push 部署** | ❌ 需手动 | ✅ | ✅ | ✅ | ❌ | ✅ |

#### 2.3.3 分阶段部署建议

```
┌────────────────────────────────────────────────────────────────┐
│  Phase 1 (评委评审/Demo): 阿里云/腾讯云 轻量应用服务器 ⭐            │
│  ├─ 优点: 免费1月、公网IP自带、国内低延迟、7×24稳定、无需Docker    │
│  ├─ 部署: SSH + pip install + uvicorn → 30分钟搞定               │
│  ├─ 费用: ¥0（免费试用）                                          │
│  └─ 适用: 评委评审、Demo 演示、短期展示                             │
│                                                                 │
│  如果追求 "Vercel 式体验"（git push 自动部署）:                      │
│  ├─ Zeabur.cn: 最接近 Vercel 体验，但服务器在海外                    │
│  └─ Render.com: Git push 部署 + WebSocket，但 $7/月 + 海外延迟     │
│                                                                 │
│  Phase 2 (长期在线): Fly.io / 阿里云轻量 / Zeabur                  │
│  └─ 适用: 评审结束后想保留服务长期运行                               │
└────────────────────────────────────────────────────────────────┘
```

**评委评审场景最终推荐: 阿里云/腾讯云轻量应用服务器**（详见第七章部署方案）。

---

## 三、架构设计

### 3.1 整体架构

```
┌──────────────────────────────────────────────────────────────┐
│                      用户浏览器                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Tab 1 演示模式 (默认) — 展示预跑结果                    │   │
│  │  ├─ Case 1 + Case 2 完整评分报告                        │   │
│  │  └─ 零 API 调用、零等待、直接浏览                         │   │
│  │  ────────────────────────────────────                  │   │
│  │  Tab 2 自定义模式 — 评委真体验                           │   │
│  │  ├─ 自行输入 Case 文本                                  │   │
│  │  ├─ 自行填写被评测模型 API Key                           │   │
│  │  └─ 真实流水线: 画像→对话→评测→优化                      │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────┬──────────────┬──────────────┬─────────────────────┘
           │ HTTP/REST    │ HTTP/REST    │ WebSocket
           ▼              ▼              ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI (uvicorn :8000)                     │
│                                                                │
│  ┌─────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │ router.py   │  │ task_manager.py │  │ ws_manager.py    │  │
│  │ REST 端点    │  │ Task CRUD +     │  │ WebSocket 注册   │  │
│  │             │  │ 流水线编排       │  │ + 广播           │  │
│  └─────────────┘  └────────┬────────┘  └──────────────────┘  │
│               │   ├───────────────────┤  │                      │
│               │   │  OptimizationEngine│  │  Slot 4 llm_client   │
│               │   │  → 优化建议        │  │  (可选)              │
│               │   └───────────────────┘  │                      │
│               └─────────────────────────┘                      │
│                                                                │
│   现有 src/ 模块: 零改动                                        │
│   loader/ simulator/ eval/ optimizer/ llm/ models/ utils/     │
└──────────────────────────────────────────────────────────────┘
         │
         │ 云服务器公网 IP / ngrok tunnel
         ▼
   http://你的公网IP:8000
```

### 3.2 关键设计决策

#### a) 四槽位 LLM 配置模型（核心变更）

系统中有四类 LLM 调用，仅在**自定义模式**下使用。演示模式不调 LLM，直接展示预跑结果。

```
┌──────────────────────────────────────────────────────────────┐
│                    四槽位 LLM 配置模型（自定义模式）              │
│                                                              │
│  槽位 1: 被评测模型 (Assistant) ← 评委自己提供                  │
│  ├─ 用途: 对话中扮演客服角色回复用户                             │
│  ├─ API Key: 评委输入（花评委的钱）                              │
│  └─ 示例: gpt-4o / claude-sonnet-4-6 / deepseek-chat         │
│                                                              │
│  槽位 2: 用户模拟器 (Simulator) ← 服务端 .env                  │
│  ├─ 用途: 生成画像文本 + 模拟用户行为 + 自检                    │
│  └─ 默认: 全局 API_KEY / BASE_URL / MODEL                    │
│                                                              │
│  槽位 3: 评测引擎 (Evaluator) ← 服务端 .env                    │
│  ├─ 用途: 9 维度清单核查 + 交叉验证 + 根因诊断                   │
│  └─ 默认: 全局 API_KEY / BASE_URL / MODEL                    │
│                                                              │
│  槽位 4: 优化引擎 (Optimizer) ← 服务端 .env                    │
│  ├─ 用途: 双路径 (规则+LLM) 生成优化建议                        │
│  └─ 默认: 全局 API_KEY / BASE_URL / MODEL                    │
└──────────────────────────────────────────────────────────────┘
```

**核心概念**: 被评测模型走评委自己的 Key（评委花钱），模拟器/评测器/优化器走服务端 .env（你花钱但用量少且有使用门槛限制）。演示模式完全不调 LLM。

**实现方式**:

```python
# task_manager.py 中的实现逻辑 —— 四 UI 槽位展开为六代码槽位
def _build_clients(self, task: Task):
    cfg = task.llm_config  # 四槽位配置字典
    
    # 辅助: 用户填了用用户的，否则用服务端全局默认
    def _make(slot_cfg, default_model, temperature):
        return LLMClient(
            api_key=slot_cfg.api_key or config.API_KEY,
            base_url=slot_cfg.base_url or config.BASE_URL,
            model=slot_cfg.model or default_model or config.MODEL,
            temperature=temperature,
        )

    # Slot 1: 被评测模型 — 评委必须提供（未提供则报错）
    assistant_client = LLMClient(
        api_key=cfg.assistant.api_key,  # 不设默认值，自定义模式必填
        base_url=cfg.assistant.base_url or "https://api.deepseek.com",
        model=cfg.assistant.model or "deepseek-chat",
        temperature=0.0,
    )
    
    # Slot 2: 用户模拟器 — 服务端全局默认
    gen_client = _make(cfg.simulator, config.SIMULATOR_MODEL, temperature=0.7)
    sim_client = _make(cfg.simulator, config.SIMULATOR_MODEL, temperature=0.7)
    
    # Slot 3: 评测引擎 — 服务端全局默认
    eval_client = _make(cfg.evaluator, config.EVALUATOR_MODEL, temperature=0.0)
    audit_client = _make(cfg.evaluator, config.EVALUATOR_MODEL, temperature=0.0)
    
    # Slot 4: 优化引擎 — 服务端全局默认
    optimizer_client = _make(cfg.optimizer, config.OPTIMIZER_MODEL, temperature=0.0)
    
    return (assistant_client, gen_client, sim_client, eval_client, audit_client, optimizer_client)
```

**四槽位 UI 与六代码槽位的映射**:

```
UI 槽位 (前端展示 4 个)         实际代码槽位 (src/ 内部 6 个)
┌──────────────────────┐      ┌──────────────────────────────┐
│ 被评测模型 (Assistant) │ ───→ │ assistant_client              │
│                      │      │ → LLMAssistant.respond()       │
├──────────────────────┤      ├──────────────────────────────┤
│ 用户模拟器 (Simulator) │ ───→ │ gen_client   (画像生成)         │
│                      │      │ → ProfileGenerator()           │
│                      │ ───→ │ sim_client   (对话模拟)         │
│                      │      │ → UserSimulator.respond()      │
├──────────────────────┤      ├──────────────────────────────┤
│ 评测引擎 (Evaluator)  │ ───→ │ eval_client  (9维度 Judge)     │
│                      │      │ → EvalOrchestrator.run()       │
│                      │ ───→ │ audit_client (行为审计)         │
│                      │      │ → ProfileAuditor()             │
├──────────────────────┤      ├──────────────────────────────┤
│ 优化引擎 (Optimizer)  │ ───→ │ llm_client   (Path B LLM)     │
│                      │      │ → OptimizationEngine()         │
└──────────────────────┘      └──────────────────────────────┘
```

**实现方式**: Web 层的 `task_manager.py` 中，slot 2 (Simulator) 的配置会生成两个 `LLMClient` 实例（gen + sim），slot 3 (Evaluator) 同样生成两个（eval + audit）。它们共享同一个 API Key/URL/Model 配置。这不需要修改 `src/` 代码——`BatchRunner.__init__()` 已经接受 `assistant_client`、`simulator_client`、`eval_client` 三个参数；`generate_profiles()` 接受独立的 `gen_client`；`audit_results()` 接受独立的 `audit_client`。审计报告已通过逐行检查 [batch_runner.py](src/simulator/batch_runner.py) 的方法签名验证了这一点。

#### a2) 评测结果中的模型信息展示

每份评测报告（无论演示预跑还是自定义模式）都标注使用的模型，确保评委清楚各槽位的模型:

```
┌──────────────────────────────────────────────────┐
│  本次评测模型配置                                    │
│  ┌─ 被评测模型: deepseek-chat @ api.deepseek.com   │
│  ├─ 用户模拟器: deepseek-chat @ api.deepseek.com   │
│  ├─ 评测引擎:   deepseek-chat @ api.deepseek.com   │
│  └─ 优化引擎:   deepseek-chat @ api.deepseek.com   │
│  ⚠ 提示: 被评测模型与评测引擎使用不同模型可避免        │
│    「自己评自己」的偏差                               │
└──────────────────────────────────────────────────┘
```

流水线执行期间，每个阶段的日志也会标注当前使用的模型:
```
[画像生成] 使用模型 deepseek-chat → 生成 8 个画像文本
[对话模拟] 被评测模型 GPT-4o ←→ 模拟用户 deepseek-chat
[评测] Judge 模型 deepseek-chat → 维度 SAFETY: 8.5/10
```

#### b) 输入: Task 数据结构

每个评测请求创建为一个 `Task`（UUID 标识）:
- **case_text**: 原始 Markdown Case 文本
- **demo_mode**: true=演示模式（展示预跑结果，不调 LLM），false=自定义模式（真实流水线）
- **llm_config**: 四槽位配置，每槽位含 `{api_key, base_url, model}`，demo 模式下全部留空
- **params**: n_profiles、run_eval、run_optimize
- 状态: `pending` → `parsing` → `phase_profiles` → `phase_dialogues` → `phase_eval` → `phase_optimize` → `completed` / `failed`
- 进度: 通过 `asyncio.Queue` 缓冲事件 → WebSocket 推送（含当前槽位使用的模型名称）
- 结果: 完成后指向 `data/exports/{task_id}/` 目录

#### c) 实时进度: WebSocket

- 前端创建 Task 后建立 `ws://host/ws/task/{task_id}`
- 事件类型: `queued` / `phase_start` / `progress` / `log` / `dialogue_card` / `completed` / `error`
- 新增 `queued` 事件: `{"type":"queued","position":2,"estimated_wait":"~5min"}`——任务排队时前端显示预期等待时间，避免白屏
- 新增 `dialogue_card` 事件: `{"type":"dialogue_card","label":"P0_冒险者","status":"用户挂断","turns":8,"model":"deepseek-chat"}`——每完成一个对话推送一张卡片，前端实时渲染
- **心跳**: 服务端每 30s 通过 WebSocket 原生 `ping` 帧发送心跳（Starlette/FastAPI 的 `websocket.iter_json()` 自动处理 pong 响应），无需应用层实现
- 断线自动重连: 前端指数退避（1s→2s→4s→…→30s cap），重连后服务端推送当前 Task 快照

#### d) 并发模型

- `max_workers=1` 串行执行，避免 LLM API 并发限流
- 任务队列: 同时只运行一个，其余显示 `queued` 状态

#### e) Task 生命周期管理

**状态机（完整版）**:
```
queued → pending → parsing → phase_profiles → phase_dialogues
→ phase_eval → phase_optimize → completed / failed
                                     ↓
                         (保留 TASK_RETENTION_HOURS 可查询，默认 72h)
                                     ↓
                         (自动清理: 内存 del + 磁盘删除)
```

**超时机制**: Task 创建后最长存活 **45 分钟**（30min 流水线上限 × 1.5 缓冲）。到期则强制标记为 `timeout`，通过 `asyncio.wait_for()` 包装 `TaskManager.run_task()` 实现。

**清理策略**（可配置）:
- 通过 `.env` 变量 `TASK_RETENTION_HOURS` 控制保留时长（默认 **72 小时**，覆盖整个评审周期）
- 超出保留期的 Task: `del`（内存） + `shutil.rmtree(data/exports/{task_id}/)`（磁盘）
- 清理协程每 10 分钟运行一次
- 单 Task 内存占用估算: ~2MB，50 个 Task 峰值 ~100MB，2C4G 服务器安全
- 手动删除: `POST /api/tasks/{id}/delete`（评委可清理自己的测试）

**服务器重启恢复**: Task 结果持久化在 `data/exports/{task_id}/` 磁盘目录。`TaskManager` 启动时扫描该目录，重建内存中的 Task 索引（仅恢复 completed/failed 状态，不恢复排队中/running 状态的任务）。这样服务器重启或崩溃后，评委的历史记录不会丢失。

**预跑 Demo 策略**: 部署完成后立即用演示模式跑一条任务，结果保留在历史中。评委到场后打开网页 → 历史记录里有现成的完整评分报告 → 即时展示系统能力 → 评委想自己试再跑新的。

#### f) 任务取消机制

Python 线程无法被安全地强制终止（`thread.terminate()` 不存在）。采用**协作式取消**:

```python
# Task 对象持有一个 threading.Event 取消令牌
class Task:
    cancel_event: threading.Event = field(default_factory=threading.Event)

# 流水线中每个阶段完成后检查
def run_task(task):
    for phase in [parse, profiles, dialogues, eval, optimize]:
        if task.cancel_event.is_set():
            task.status = "cancelled"
            return
        await run_phase(phase)
```

- 取消粒度: 每完成一个对话或一个画像后检查取消令牌
- `POST /api/tasks/{id}/cancel` → 设置 `cancel_event` → 当前阶段结束后停止
- 最大取消延迟: 一个 LLM 调用的时间（~30s），不会无限等待
- 取消后结果不保留（已完成的对话数据丢弃）

#### g) 部分故障处理

单个对话失败不影响整体任务:
- 单个对话 LLM 调用异常 → 该对话标记 `status="异常中断"`，继续下一个
- 超过 50% 对话失败 → 整个 Task 标记 `failed`
- 解析/画像生成失败 → 整个 Task 标记 `failed`（无恢复必要）

#### h) demo_case.md 规范

```markdown
# demo_case.md — 评委打开网页时看到的默认示例 Case

格式: 标准 Markdown Case 指令文本（与 generated_cases_all.json 中条目格式一致）
加载时机: web/app.py 启动时由 config.py 读取并缓存到内存
加载路径: Path(__file__).parent / "demo_case.md"
备选方案: 若文件不存在，回退到 generated_cases_all.json 中 id=1 的 Case
内容建议: 选一条外卖或酒店场景、4-6 步 call flow、无复杂分支的 Case，
         确保 3 个画像 × 各 5-8 轮对话 = 约 5-8 分钟完成
```

#### i) Case 文本解析: 直接复用

`parse_instruction()` 已经是纯函数:

```python
from src.loader.case_parser import parse_instruction

# Web 层直接调用，无需任何修改
case = parse_instruction(
    instruction=task.case_text,   # 用户输入的 Markdown 文本
    case_id=0,                     # Web 动态 Case 使用 0 或时间戳
    title=task.case_title or "外部 Case",
)
# → Case 对象可直接传入 BatchRunner
```

---

## 四、页面与 API 设计

### 4.1 前端视图

页面顶部固定一行引导说明，下方三个 Tab 切换：

| Tab | 名称 | 功能 | 核心元素 |
|-----|------|------|----------|
| **Tab 1** | 新建评测 | 创建并运行新任务 | 模式切换（演示/自定义）→ Case 编辑器 + 四槽位配置 + 参数面板 → 实时监控 → 结果面板 |
| **Tab 2** | 历史记录 | 浏览、回溯所有历史测试 | 任务列表（时间、Case 名、评分、画像数、状态）→ **点击展开详情**（完整评分报告 + 9 维度明细 + 对话记录 + 优化建议）→ 下载 / 删除 |
| **Tab 3** | 关于系统 | 项目说明 | 系统简介、评测方法论（9 维度说明）、使用指南 |

**引导语**（页面顶部，始终可见）:
> 美团外呼对话模型评测系统 Demo — Tab 1 浏览演示效果 | Tab 2 自定义评测 | Tab 3 历史记录

**演示模式流程**: 评审前，你用 Case id=1 和 id=2 跑完两条流水线，结果存入历史记录。评委打开网页 → Tab 1「新建评测」→ 演示模式下直接展示这两条预跑结果的完整报告（9 维度明细 + 对话记录 + 优化建议 + 下载）→ **零 API 调用、零等待**。评委想自己体验 → 切到自定义模式 → 输入 Case + API → 真实跑流水线。

**自定义模式流程**: 评委切换后 → 自行输入 Case 文本 + 被评测模型 API → 模拟器/评测器/优化器使用服务端 `.env` 凭据 → 点「开始评测」→ WebSocket 实时进度 → 评分报告。被评测模型走评委自己的 Key，不消耗服务端额度。

**历史记录功能**（完整交互）:
```
Tab 2「历史记录」
├─ 任务列表（按时间倒序）
│   ├─ [2026-06-05 14:30] 外卖Case | GPT-4o | 72分(合格) | 3画像 | ✅
│   ├─ [2026-06-05 13:15] 酒店Case | deepseek-chat | 85分(良好) | 5画像 | ✅
│   └─ [2026-06-05 10:00] Demo预跑 | deepseek-chat | 78分(合格) | 3画像 | ✅
│
├─ 点击任一条 → 展开详情面板
│   ├─ 评分总览（百分制 + 五级评级）
│   ├─ 9 维度明细（每维度分数 + checklist 清单 + Judge 评语）
│   ├─ 优化建议（四对象：Case/画像/模型/评测）
│   ├─ 对话记录（完整多轮对话，标注发言方+模型名称）
│   └─ 操作按钮: [下载报告 MD] [下载报告 JSON] [下载对话 ZIP] [删除此记录]
```

### 4.2 REST API 端点

```
GET  /                          服务 index.html + 静态资源

# ---- Case 相关 ----
GET  /api/presets               获取预置 Case 模板列表（快速体验用）
     Response: [{id, title, business_line, complexity_score}]

GET  /api/presets/{id}          获取预置 Case 模板的完整文本
     Response: {id, title, instruction: "完整 Markdown 文本"}

POST /api/cases/parse           预览解析用户输入的 Case 文本
     Body: {case_text: "..."}
     Response: {title, business_line, role, task, opening_line,
                call_flow_steps: [...], constraints: [...], knowledge_points: [...]}

# ---- 任务相关 ----
POST /api/tasks                 创建评测任务
     Body: {
       case_text: str,              // 必填: 原始 Markdown Case 文本
       case_title: str?,            // 可选: Case 标题
       demo_mode: bool (默认 true), // 演示模式: true=展示预跑结果(不调LLM)，false=真实流水线

       // 四槽位 LLM 配置（自定义模式下用户可逐项覆盖，演示模式下全部留空）
       llm_config: {
         assistant:  {api_key?, base_url?, model?},  // 被评测模型
         simulator:  {api_key?, base_url?, model?},  // 用户模拟器（不填用服务端默认）
         evaluator:  {api_key?, base_url?, model?},  // 评测引擎（不填用服务端默认）
         optimizer:  {api_key?, base_url?, model?}   // 优化引擎（不填用服务端默认）
       },

       n_profiles: int (默认 3),
       run_eval: bool (默认 true),
       run_optimize: bool (默认 false)
     }
     Response: {task_id: "uuid", status: "accepted", mode: "demo|custom",
                effective_models: {assistant, simulator, evaluator, optimizer}}
     // effective_models 返回本次任务实际使用的模型名称，供前端展示

GET  /api/tasks/{id}/models      获取当前任务的模型配置（供前端"当前模型配置"面板使用）
     Response: {assistant: {model, base_url}, simulator: {...}, ...}

GET  /api/tasks                 列出历史任务（按时间倒序，支持 ?limit=20）
     Response: [{task_id, status, created_at, case_title, n_profiles,
                total_score_100?, rating_label?}]

GET  /api/tasks/{id}            获取任务状态和结果摘要（含评分总览）
     Response: {task_id, status, progress, result_summary: {total_score_100, rating_label, dimension_scores}}

GET  /api/tasks/{id}/detail     获取任务完整详情（历史回溯用）
     Response: {task_id, status, created_at, case_title, llm_models_used,
                eval_result: {total_score_100, rating_label, dimension_checklists, ...},
                optimization_suggestions: [...], conversation_summary: {...}}

GET  /api/tasks/{id}/conversations  获取对话全文（轮次级）
     Response: [{turn_index, speaker, content, parsed_tags, model_used}]

DELETE /api/tasks/{id}          删除历史任务（内存 + 磁盘）
     Response: {ok: true}

GET  /api/tasks/{id}/result     获取详细结果 JSON（?summary=true 仅摘要）

GET  /api/tasks/{id}/download/{type}
     type = report_md | report_json | conversations_json | batch_summary | all_zip
     Response: 文件下载

# ---- 工具 ----
POST /api/test-connection       测试指定槽位的 API 连通性
     Body: {slot: "assistant"|"simulator"|"evaluator"|"optimizer",
            api_key?, base_url?, model?}  // 不填则使用对应槽位的服务端默认值
     Response: {ok: bool, message: str, latency_ms: int}

WS   /ws/task/{id}              WebSocket 实时进度
     连接策略: Task 不存在时拒绝连接(4404)。Task 已完成时返回快照后关闭。
     心跳: 服务端每 30s 发送 WebSocket ping 帧，客户端自动 pong。
     Server → Client 事件类型:
       {"type":"queued","position":2,"estimated_wait_sec":300}
       {"type":"phase","phase":"parsing|profiles|dialogues|eval|optimize","status":"started|completed",
        "model_used":"deepseek-chat"}
       {"type":"progress","phase":"dialogues","completed":3,"total":15}
       {"type":"dialogue_card","label":"P0_冒险者","case_id":1,"status":"用户挂断","turns":8,
        "assistant_model":"gpt-4o","simulator_model":"deepseek-chat"}
       {"type":"log","message":"正在解析 Case 文本...","level":"info|warn|error"}
       {"type":"completed","task_id":"...","result_summary":{...}}
       {"type":"error","message":"...","recoverable":true|false}
```

---

## 五、文件清单（全部新建）

```
meituan/
├── web/
│   ├── __init__.py              # 包初始化 (~5 行)
│   ├── config.py                # Web 配置: 全局默认 + 四槽位可选覆盖 + session 临时覆盖 (~50 行)
│   ├── demo_case.md             # 演示模式默认 Case 文本 (~50 行)
│   ├── schemas.py               # Pydantic 模型 (~160 行)
│   │                            #   LLMConfig, LLMConfigSlots (四槽位),
│   │                            #   ParseCaseRequest/Response,
│   │                            #   CreateTaskRequest (含 demo_mode + llm_config),
│   │                            #   TestConnectionRequest, DemoConfigResponse,
│   │                            #   TaskStatusResponse, ProgressEvent (含 model_used)
│   ├── task_manager.py          # Task 抽象 + 四槽位 LLM 构建 + 流水线编排 (~400 行)
│   ├── ws_manager.py            # WebSocket 连接注册 + 广播 + heartbeat (~60 行)
│   ├── router.py                # 所有 REST + WS 路由处理器 (~280 行)
│   ├── app.py                   # FastAPI 应用工厂 + 静态文件挂载 + CORS (~40 行)
│   └── static/
│       ├── index.html           # Vue 3 CDN 单页应用，四步视图 (~500 行)
│       ├── style.css            # Demo 样式 (Pico.css CDN + 自定义) (~200 行)
│       └── app.js               # Vue 3 响应式应用逻辑 (~500 行)
│                                 #   状态: demoMode, llmConfig (四槽位 reactive),
│                                 #         currentStep, taskStatus, progressEvents[]
│                                 #   组件: ModeSelector, CaseEditor, LlmConfigPanel,
│                                 #         ParamPanel, TaskMonitor, ResultsDashboard
├── run_web.py                   # 启动入口 (~15 行)
├── start_with_ngrok.bat         # 开发测试用一键启动 (~10 行)
├── README.md                    # 项目说明：快速开始 + 环境配置 + 本地运行指南 (~80 行)
└── README_WEB.md                # Web 部署运维说明 (~60 行)
```

**预估总代码量**: ~3,600 行（后端 ~1,400 行 + 前端 ~1,900 行 + 脚本 ~25 行 + 文档 ~280 行）

> 上调原因: 前端增加历史记录详情面板 + 关于系统页面（+200 行）；后端增加历史详情/删除/重启恢复逻辑（+100 行）

**现有代码改动**: **零**。所有 `src/` 模块保持不变。四槽位 LLM 配置直接利用 `BatchRunner.__init__()` 已有的 `assistant_client`、`simulator_client`、`eval_client` 参数。

---

## 六、开发步骤（4 阶段）

### Phase 1: 后端骨架 + Case 输入（Day 1-3）

1. 创建 `web/` 包目录结构
2. 实现 `web/config.py` — 读取 `.env` 配置：
   ```bash
   # .env 配置项
   
   # 全局默认（服务端模拟器/评测器/优化器三个槽位的回退值，必填）
   API_KEY=sk-your-deepseek-key
   BASE_URL=https://api.deepseek.com
   MODEL=deepseek-chat
   
   # 可选: 按槽位覆盖模型型号（不设则回退到全局 MODEL）
   # SIMULATOR_MODEL=deepseek-chat    # 模拟器专用模型
   # EVALUATOR_MODEL=deepseek-chat    # 评测引擎专用模型
   # OPTIMIZER_MODEL=deepseek-chat    # 优化引擎专用模型
   ```
   自定义模式下: 被评测模型用评委自己的 Key，模拟器/评测器/优化器用服务端全局默认（可选按槽位覆盖）。演示模式不调 LLM，无需 API Key。
3. 实现 `web/schemas.py` — Pydantic 模型:
   - `LLMConfig` — 单槽位配置 `{api_key?, base_url?, model?}`
   - `LLMConfigSlots` — 四槽位配置 `{assistant?, simulator?, evaluator?, optimizer?}`
   - `ParseCaseRequest/Response` — Case 文本解析预览
   - `CreateTaskRequest` — **含 case_text + demo_mode + llm_config（四槽位）**
   - `TestConnectionRequest/Response` — API 连通性测试（指定槽位）
   - `PresetCaseSummary` — 预置 Case 模板
   - `DemoConfigResponse` — 返回演示模式配置（预填 Case 文本、四槽位模型名称等）
   - `TaskStatusResponse`、`ProgressEvent`（含 `model_used` 字段标注当前阶段使用的模型）
4. 实现 `web/task_manager.py`:
   - `Task` dataclass: id, status, case_text, demo_mode, **llm_config（四槽位）**, progress, event_queue
   - `TaskManager.create_task()` / `get_task()` / `list_tasks()`
   - `TaskManager.run_task()` — 解析 Case 文本 → 四槽位 LLM 客户端构建 → 流水线执行
   - **配置优先级**: 被评测模型用评委自己的 Key（必填），其余槽位用户可选填 > 槽位专用变量(`SIMULATOR_MODEL`等) > 全局 `MODEL`
   - **API Key 安全**: 自定义模式的所有 api_key 仅存内存，Task 完成/失败后立即清除
   - **演示模式默认 Case**: 从 `web/demo_case.md` 加载
5. 实现 `web/ws_manager.py` — WebSocket 连接 + heartbeat
6. 实现 `web/router.py` — 所有端点（含 `POST /api/cases/parse` 解析预览）
7. 实现 `web/app.py` — FastAPI 应用创建
8. 创建 `run_web.py`
9. **验证**: `python run_web.py` → Swagger 可访问 → `POST /api/cases/parse` 返回正确解析结果

### Phase 2: 流水线集成（Day 3-5）

1. 将 `TaskManager.run_task()` 接入真实流水线:
   - 四槽位 LLM 客户端构建: 被评测模型=评委必填；模拟器/评测器/优化器=评委可选填 > 槽位变量 > 全局默认
   - Case 解析: `parse_instruction(task.case_text, 0, title)` → Case 对象
   - 画像生成: `ProfileGenerator` + **Simulator 槽位 (gen_client)** → 每画像推送进度（含模型名称）
   - 对话模拟: `DialogueRunner` → **Assistant 槽位 (asst_client)** ↔ **Simulator 槽位 (sim_client)** → 每对话推送（含两方模型名称）
   - 评测: `EvalOrchestrator` + **Evaluator 槽位 (eval_client)** + `ProfileAuditor` + **Evaluator 槽位 (audit_client)** → 每评测推送评分摘要（含 Judge 模型名称）
   - 优化: `OptimizationEngine` + **optimizer 槽位** → 推送建议数量
2. 结果文件管理: `data/exports/{task_id}/` + `DataExporter`
3. 下载端点实现
4. API 连通性测试: 对每个配置过的槽位发送一条简短测试请求（"回复 OK"），返回延迟和状态
5. 错误处理（分级）:
   - 单个对话异常 → 标记 `异常中断`，推送 `error(recoverable=true)` → 继续下一个
   - 累计超过 50% 对话失败 → 标记 `failed`
   - 解析/画像生成失败 → 立即标记 `failed`
6. **验证**: 输入一条完整 Case 文本 + 单个 API → 端到端走通 → 再测试四槽位不同模型配置 → 测试部分对话失败场景

### Phase 3: 前端（Day 5-8）

技术方案: **Vue.js 3 (CDN, 无构建)** + Pico.css CDN（见 §2.2 论证）。单文件 `index.html` 内嵌 `<script type="module">`。

1. 创建 `web/static/index.html` — Vue 3 单页应用，四步视图:
   - `#view-select` — 模式切换（演示/自定义）+ Case 编辑器 + 预置模板 + 解析预览
   - `#view-config` — 四槽位模型配置面板 + 每个槽位的连通性测试
   - `#view-params` — 画像数量滑块 + 评测/优化开关 + 「当前模型配置」摘要栏
   - `#view-monitor` — 实时进度（WebSocket）+ 阶段进度条 + 日志流（含模型标注）+ 对话卡片
   - `#view-results` — 评分总览 + 9 维度明细 + 优化建议 + 下载
2. 创建 `web/static/style.css` — Pico.css CDN + 自定义 Demo 样式
3. 创建 `web/static/app.js` — Vue 3 应用:
   - `createApp({ data(), methods{}, computed{}, watch{} })` 单文件组织
   - 响应式状态: `currentStep`, `demoMode`, `llmConfig` (四槽位 reactive 对象), `taskStatus`, `progressEvents[]`, `results`
   - 步骤导航（watch 驱动校验）
   - Case 文本编辑 + 模板加载 + 实时解析预览（debounce 500ms）
   - 四槽位 API 配置表单 + 逐槽位连通性测试
   - WebSocket 实时监控 + 事件驱动的 UI 更新
   - 结果面板 + 评分表格 + 下载
4. **验证**: 完整操作: 切换模式 → 输入文本 → 配四槽位 → 启动 → 看实时进度 → 看结果 → 下载

### Phase 4: 打磨与部署（Day 8-10）

1. **历史记录完整功能**:
   - 任务列表页（Tab 2）: 按时间倒序展示，含 Case 名称、评分、模型、画像数、状态标签
   - 点击展开详情: 完整评分报告 + 9 维度明细 + 对话记录 + 优化建议
   - 重新下载: 历史任务同样支持下载报告
   - 删除功能: `POST /api/tasks/{id}/delete` + 前端确认弹窗
2. 服务器重启恢复: `TaskManager` 启动时扫描 `data/exports/` 目录重建历史索引
3. 添加取消任务功能
4. CSS 柱状图展示评分分布
5. 「关于系统」页面（Tab 3）: 项目简介 + 评测方法论 + 使用指南
6. 阿里云部署（§7.2 完整流程）: 领取服务器 → 部署 → 预跑 Demo → 公网验证
7. 编写 `README.md`（项目根目录）: 指导评委本地下载安装运行（见 §7.4）
8. 全链路测试: 演示模式 + 自定义模式 + 历史回溯 + 并发浏览
9. **验证**: 公网链接完整流程 + API Key 安全性 + 历史持久化

---

## 七、部署方案（评委评审场景最终推荐）

### 7.1 方案对比（针对评委评审场景）

核心约束: 评委在国内 → 低延迟 → 公网 URL 稳定 → 零门槛体验。

| 方案 | 稳定性 | 延迟 | 部署难度 | 费用 | 评委体验 |
|------|:--:|:--:|:--:|------|:--:|
| **A: 阿里云/腾讯云轻量服务器** ⭐ | ✅ 7×24 | ⭐ 国内<20ms | ⭐⭐ SSH+pip | **¥0** (免费1月) | 🏆 最佳 |
| B: ngrok + 本地电脑 | ⚠️ 本机不能休眠 | ⭐ 本地 | ⭐ 最简 | $0 | 有风险 |
| C: Fly.io | ✅ 7×24 | ⚠️ 新加坡~80ms | ⭐⭐⭐ 需Docker | $0-$5/月 | 可用 |
| D: Zeabur | ✅ 7×24 | ⚠️ 海外 | ⭐⭐ git push | $5 免费额 | 可用 |

**结论: 方案 A（云轻量服务器）是唯一同时满足「零成本 + 国内低延迟 + 稳定 7×24 + 部署极简 + 无需 Docker」的方案。**

### 7.2 最终推荐: 阿里云/腾讯云轻量应用服务器

#### 为什么选它

1. **完全免费**: 阿里云新用户免费试用 1 个月（2核1G/2核4G），腾讯云免费试用 1 个月（2核2G），**完全覆盖评审周期**
2. **公网 IP 自带**: 不需要 ngrok、不需要域名备案、不需要端口转发
3. **部署极简**: 5 条命令，不需要 Docker，不需要 Nginx
4. **国内低延迟**: 评委打开网页秒级响应，WebSocket 连接不卡顿
5. **用完即弃**: 评审结束停止服务器，零费用。想保留则阿里云 38 元/年续费

#### 具体部署步骤（30 分钟内完成）

```bash
# ==================== Step 1: 领取免费服务器 ====================
# 方式 A: 阿里云 → free.aliyun.com → 领取「轻量应用服务器」1 个月免费试用
# 方式 B: 腾讯云 → 免费体验馆 → 领取「标准型 2核2G」1 个月试用
# 方式 C: 抢不到免费 → 腾讯云 35 元/月 直接买，或阿里云 38 元/年 秒杀

# 选镜像: Ubuntu 22.04（纯系统）
# 设置 root 密码，记录公网 IP

# ==================== Step 2: 云控制台开放端口 ====================
# 阿里云: 安全组 → 添加规则 → 入方向 → TCP 8000 → 授权 0.0.0.0/0
# 腾讯云: 防火墙 → 添加规则 → TCP 8000 → 允许

# ==================== Step 3: SSH 登录，一键部署 ====================
ssh root@你的公网IP

# 安装 Python（Ubuntu 22.04 自带 Python 3.10）
apt update && apt install -y python3-pip git

# 克隆项目
git clone https://github.com/你的用户名/meituan.git
cd meituan

# 安装依赖
pip install -r requirements.txt
pip install fastapi uvicorn[standard] slowapi

# 配置 .env
cat > .env << 'EOF'
API_KEY=sk-your-deepseek-key
BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat
EOF

# 启动服务（后台运行，崩溃自动拉起）
cat > /etc/systemd/system/meituan-web.service << 'EOF'
[Unit]
Description=Meituan LLM Evaluation Web
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/meituan
ExecStart=/usr/bin/python3 -m uvicorn web.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable meituan-web
systemctl start meituan-web

# 确认服务状态
systemctl status meituan-web

# ==================== Step 4: 验证 ====================
# 本地浏览器打开: http://你的公网IP:8000
# 确认页面正常 → Swagger 可访问: http://你的公网IP:8000/docs
# 确认 WebSocket: 浏览器 DevTools → Network → WS → 连接成功

# ==================== Step 5: 预跑演示数据 ====================
# 用 batch_runner 批量跑 2 条 Case，画像数拉满:
python -m src.simulator.batch_runner --case-id 1 --n-profiles 8 --run-eval --run-optimize
python -m src.simulator.batch_runner --case-id 2 --n-profiles 8 --run-eval --run-optimize

# 跑完后导出结果到 data/exports/ 目录（DataExporter 自动处理）
# 评委打开网页 → Tab 1 演示模式 → 看到 2 条 Case × 8 画像的完整评测报告

# ==================== Step 6: 发给评委 ====================
# 公网访问地址: http://你的公网IP:8000
# 评委打开 → Tab 1「新建评测」演示模式 → 看预跑 Case 结果（零等待）
# → Tab 2 切自定义模式 → 自己输入 Case + API → 真实体验
```

#### 可选：绑定域名（提升专业度）

在云控制台为轻量服务器添加 DNS 解析（需已有域名），将 `demo.你的域名.com` 指向服务器公网 IP。评委访问 `http://demo.你的域名.com:8000` 比裸 IP 更专业。

### 7.3 备选方案

#### 备选 A: ngrok + 本地机器（快速测试用）

如果只是自己测试、不需要发给评委，ngrok 是最高效的选择：

```batch
:: 下载 ngrok.exe 放入项目根目录
start_with_ngrok.bat

:: 内容:
:: start "" ngrok http 8000
:: timeout /t 2
:: python run_web.py
```

**评委场景不推荐**，原因: 电脑休眠则链接死掉、网络波动影响体验、URL 每次重启变化。

#### 备选 B: Fly.io（海外稳定方案）

如果已熟悉 Docker，Fly.io 的 $5/月免费额度可覆盖 Demo:

```bash
fly launch  # 自动检测 Python + FastAPI
fly deploy  # 部署到新加坡节点
# URL: https://yourapp.fly.dev
```

#### 备选 C: 如果腾讯云/阿里云都不可用

华为云云耀 L 实例 **95 元/年**（~8 元/月，2核2G/3Mbps/40GB），性价比高。

### 7.4 评委本地运行（README.md 指南）

评委如果想从 GitHub 下载源代码在自己的电脑上运行，项目根目录需提供 `README.md` 指导。内容大纲：

```markdown
# 美团外呼对话模型评测系统

## 快速开始（5 分钟）

### 1. 环境要求
- Python 3.10+
- Windows / macOS / Linux

### 2. 安装
git clone https://github.com/xxx/meituan.git
cd meituan
pip install -r requirements.txt

### 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key（或其他 OpenAI 兼容 API）
# API_KEY=sk-xxxxx
# BASE_URL=https://api.deepseek.com
# MODEL=deepseek-chat

### 4. 启动 Web 服务
python run_web.py
# 浏览器打开 http://localhost:8000

### 5. 命令行模式（可选）
python -m src.simulator.batch_runner --case-id 1
```

评委本地运行和 Web 部署版的核心区别：

| | Web 部署版 | 本地运行 |
|------|------|------|
| 演示模式 | 浏览预跑结果，零门槛 | 不适用（需自己跑） |
| 自定义模式 | 输入 Case + 自己的 Key → 真实流水线 | 同，需要自己装 Python |
| 评委需要 API Key | 仅自定义模式需要（自备） | 需要 |
| 评委需要装 Python | ❌ 不需要 | ✅ 需要 |
| 适用场景 | 快速体验、评审演示 | 深度了解代码、二次开发 |

## 八、现有代码兼容性分析

### 8.1 无需任何修改

| 文件 | 改动 | 原因 |
|------|------|------|
| [case_parser.py](src/loader/case_parser.py) `parse_instruction()` | 当前 `case_id` 参数用于 `detect_business_line()` 中的特殊逻辑（如 case 1/2 的 fallback）。需确保传入 `case_id=0` 或任意值时不会崩溃 | 外部 Case 没有预置 ID。经分析 `case_id` 仅用于 `detect_business_line()` 的 `_BUSINESS_LINE_OVERRIDES` 字典查找（key 不存在时 fallback 到关键词匹配），传入 `case_id=0` 无影响。**无需修改**。 |

经全面审计，**现有代码无需任何修改即可支持动态 Case 输入**。

### 8.2 可选优化（非必须）

| 文件 | 优化 | 优先级 |
|------|------|--------|
| [case_loader.py](src/loader/case_loader.py) | 新增 `load_preset_summaries()` 返回摘要列表（只读 id/title/business_line/complexity），避免 Web 层解析 9064 行的完整 JSON | 低（当前 60 条 Case 的完整加载也在毫秒级） |
| [batch_runner.py](src/simulator/batch_runner.py) | 新增 `progress_callback` 参数，使 Web 层无需重新实现编排循环 | 中（v2 可考虑，当前包装方案可行） |

---

## 九、关键现有文件（接口依赖）

| 文件 | Web 层调用的接口 | 是否需改 |
|------|-----------------|:--:|
| [case_parser.py](src/loader/case_parser.py) | `parse_instruction(text, id, title)` — 纯函数，Web 层核心依赖 | ❌ |
| [batch_runner.py](src/simulator/batch_runner.py) | `BatchRunner(cases, assistant_client=None, simulator_client=None, eval_client=None)` — 直接传入 Case 列表和 3 个 LLMClient；另有 `generate_profiles(gen_client)` + `audit_results(audit_client)` + `run_phase3(eval_client)` 独立参数 | ❌ |
| [client.py](src/llm/client.py) | `LLMClient(api_key, base_url, model)` — 支持动态凭据构建 | ❌ |
| [profile_generator.py](src/simulator/profile_generator.py) | `ProfileGenerator(client)` — 构造函数接受外部 LLMClient，Web 层传入 gen_client（Simulator 槽位） | ❌ |
| [orchestrator.py](src/eval/orchestrator.py) | `EvalOrchestrator(judge_client)` — 构造函数接受外部 LLMClient，Web 层传入 eval_client（Evaluator 槽位） | ❌ |
| [optimizer.py](src/optimizer/optimizer.py) | `OptimizationEngine(llm_client)` — 构造函数接受可选 LLMClient，Web 层传入 optimizer_client（Optimizer 槽位），为 None 则跳过 Path B | ❌ |
| [data_exporter.py](src/utils/data_exporter.py) | `DataExporter` — 复用所有导出逻辑 | ❌ |

---

## 十、安全与容错

### 10.1 API Key 安全

| 风险 | 措施 |
|------|------|
| 用户 API Key 泄露 | 仅存 Task 对象内存中，任务完成/失败/清理后立即 `del`。不写入日志、不落盘。`GET /api/tasks/{id}/models` 仅返回模型名称和 base_url，**不返回 api_key** |
| HTTPS 传输 | 创建任务时 API Key 通过 HTTPS body 传输（加密），之后不在任何请求中出现 |
| 服务端 API Key 泄露 | `.env` 文件已在 `.gitignore`，前端无任何读取路径 |

### 10.2 容错

| 风险 | 措施 |
|------|------|
| 单个对话 LLM 调用失败 | 该对话标记 `异常中断`，继续下一个（Phase 2 已实现） |
| 超过 50% 对话失败 | 整个 Task 标记 `failed` |
| Task 执行超时 | 45min 硬超时（`asyncio.wait_for`），标记 `timeout` |
| 瞬时网络错误 | 复用 `LLMClient` 内置 3 次退避重试；WebSocket 断线前端自动重连 |
| 大结果文件 | 下载端点流式传输 `all_zip`（`StreamingResponse` + `zipfile`），不一次性加载到内存。Dashboard 视图默认返回摘要 |
| 公开端点滥用 | 演示级基本速率限制: `POST /api/tasks` 每 IP 最多 1 次/10秒，`POST /api/test-connection` 每 IP 最多 1 次/5秒。使用 FastAPI `slowapi` 中间件实现（~10 行配置）

---

## 十一、验证清单

- [ ] `python run_web.py` 正常启动，`http://localhost:8000/docs` Swagger 可访问
- [ ] `POST /api/cases/parse` — 输入 Markdown 文本返回正确解析结果
- [ ] `GET /api/presets` — 返回 60 条预置 Case 摘要
- [ ] `POST /api/test-connection` — 四槽位各自连通性测试正确（有效/无效 Key 均正确处理）
- [ ] `POST /api/tasks` — 创建任务成功，demo 模式和 custom 模式均正常
- [ ] WebSocket: 连接→queued事件→phase事件→dialogue_card事件→completed事件 全链路
- [ ] WebSocket: 心跳 ping/pong 正常，断线重连正常
- [ ] 端到端: 输入 Case 文本 + 配四槽位 → 画像 → 对话（含模型名称标注）→ 评测 → 结果
- [ ] 端到端（含优化）: 上述流程 + 优化建议生成
- [ ] 预置模板: 选择预置 Case → 自动填入文本区 → 可修改后启动
- [ ] 部分故障: 单个对话失败 → 继续执行 → 超过 50% → 标记 failed
- [ ] Task 超时: 45min 超时 → 标记 timeout → 前端显示
- [ ] Task 清理: 超过 TASK_RETENTION_HOURS（默认 72h）→ 内存 + 磁盘自动清理
- [ ] 结果下载: report_md / report_json / conversations_json / all_zip 均可下载
- [ ] 云服务器: ngrok/轻量服务器 公网链接可从其他设备访问完整流程
- [ ] **安全**: `GET /api/tasks/{id}/models` 不返回 api_key
- [ ] **安全**: 任务完成后/清理后服务端内存中 api_key 已清除
- [ ] 错误场景: 解析失败/API 不通/对话超时 → 前端正确显示分级错误信息
- [ ] `src/` 目录下所有文件未被修改
