# 美团外呼对话模型评测系统

对 LLM 对话模型进行多维度自动化评测。包含 Case 解析、用户画像生成、对话模拟、9 维度评测和优化建议生成。

## 快速开始

### 环境要求
- Python 3.10+
- Windows / macOS / Linux

### 安装

```bash
git clone <repo-url>
cd meituan
pip install -r requirements.txt
```

### 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key（或其他 OpenAI 兼容 API）:
# API_KEY=sk-xxxxx
# BASE_URL=https://api.deepseek.com
# MODEL=deepseek-chat
```

### 启动 Web 服务

```bash
python run_web.py
# 浏览器打开 http://localhost:8000
```

### 命令行模式

```bash
# 单个 Case 批量评测
python -m src.simulator.batch_runner --case-id 1
```

## Web 部署

详见 [web_deployment_plan.md](web_deployment_plan.md) 部署计划书。

简要步骤:
1. 阿里云/腾讯云领取免费轻量应用服务器
2. SSH 登录 → 克隆项目 → pip install → 启动 uvicorn
3. 安全组/防火墙放行 8000 端口
4. 浏览器访问 `http://公网IP:8000`

## 项目结构

```
meituan/
├── src/
│   ├── loader/       # Case 解析
│   ├── simulator/    # 画像生成 + 对话模拟
│   ├── eval/         # 9 维度评测引擎
│   ├── optimizer/    # 优化建议引擎
│   ├── llm/          # LLM 调用封装
│   ├── models/       # 数据模型
│   └── utils/        # 工具函数
├── web/              # Web 层 (FastAPI + Vue 3)
├── data/             # 运行时数据
└── run_web.py        # Web 启动入口
```
