# 阿里云/腾讯云部署执行计划

> 日期: 2026-06-07 | 状态: 待执行

---

## 一、背景

Web 评测系统已完成构建、Bug 修复和全链路验证，Git 历史干净（4 个提交）。当前系统运行在 `localhost:8000`，需要部署到公网服务器供评委访问。

**项目特点**：
- 纯 CPU 项目，无 GPU 需求（所有 LLM 调用走远程 API）
- FastAPI + Vue.js 3（CDN），无需构建工具
- 10 个 Python 依赖，Python 3.10+
- 不需要 Docker、Nginx、数据库、Redis 等中间件

**待填补缺口**：
- 无 `.env.example` 模板（README.md 引用了它但不存在）
- 无远程 Git 仓库（无法在服务器上 `git clone`）
- 无 `deploy.sh` 一键部署脚本
- `data/checklist_evolution/` 未在 `.gitignore` 中排除

---

## 二、本地方案（6 步，~15 分钟）

### Step 1: 创建 `.env.example`

项目根目录新建 `.env.example`：

```bash
# ========== LLM API 配置（必填）==========
# 所有槽位的全局默认 API 配置
API_KEY=sk-your-api-key-here
BASE_URL=https://api.deepseek.com
MODEL=deepseek-chat

# ========== 按槽位覆盖模型型号（可选）==========
# 不设则回退到全局 MODEL
# SIMULATOR_MODEL=deepseek-chat
# EVALUATOR_MODEL=deepseek-chat
# OPTIMIZER_MODEL=deepseek-chat

# ========== 服务器配置（可选）==========
# WEB_HOST=0.0.0.0
# WEB_PORT=8000

# ========== 任务生命周期（可选）==========
# TASK_TIMEOUT_SEC=2700       (45分钟超时)
# TASK_RETENTION_HOURS=72      (3天保留)
# CLEANUP_INTERVAL_SEC=600     (10分钟清理间隔)
```

### Step 2: 创建 `deploy.sh`

项目根目录新建 `deploy.sh`（云服务器上一键部署）：

```bash
#!/bin/bash
set -e
echo "=== 美团外呼评测系统 — 一键部署 ==="

# 1. 系统依赖
echo "[1/5] 安装系统依赖..."
apt update -qq && apt install -y -qq python3-pip python3-venv git

# 2. Python 依赖
echo "[2/5] 安装 Python 依赖..."
pip3 install -q -r requirements.txt

# 3. 检查 .env
echo "[3/5] 检查配置..."
if [ ! -f .env ]; then
    echo ">>> 未找到 .env，从 .env.example 创建模板..."
    cp .env.example .env
    echo ">>> 请编辑 .env 填入 API Key: nano .env"
    exit 1
fi

# 4. systemd 服务
echo "[4/5] 配置 systemd 服务..."
cat > /etc/systemd/system/meituan-web.service << 'SVC'
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
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
SVC

systemctl daemon-reload
systemctl enable meituan-web
systemctl restart meituan-web

# 5. 验证
echo "[5/5] 检查服务状态..."
sleep 3
systemctl status meituan-web --no-pager
echo ""
echo "=== 部署完成 ==="
echo "公网访问: http://$(curl -s ifconfig.me):8000"
echo "API 文档:  http://$(curl -s ifconfig.me):8000/docs"
```

### Step 3: 更新 `.gitignore`

在 `data/exports/` 段附近添加：

```
data/checklist_evolution/
```

### Step 4: 创建 Gitee 远程仓库并推送

```bash
# 1. 在 https://gitee.com 创建私有仓库（如 meituan-eval）
# 2. 添加远程仓库并推送
git remote add gitee https://gitee.com/<你的用户名>/meituan-eval.git
git push -u gitee main
```

### Step 5: Git 提交本地变更

```bash
git add .env.example deploy.sh .gitignore
git commit -m "chore: 添加部署准备文件 (.env.example, deploy.sh, 更新 .gitignore)"
git push gitee main
```

### Step 6: 确认不需要推送的数据已排除

- `data/exports/` ✅ 已在 .gitignore（24MB 本地测试数据不会推送）
- `.env` ✅ 已在 .gitignore（含真实 API Key，不会泄露）
- `data/checklist_evolution/` → Step 3 中补充排除

---

## 三、云服务器方案（阿里云轻量应用服务器）

### 为什么选阿里云

| 维度 | 详情 |
|------|------|
| **免费额度** | 新用户 1 个月免费试用 |
| **配置** | 2核1G/2核4G（本项目只需 ~200MB 内存） |
| **公网 IP** | 自带，无需域名备案 |
| **国内延迟** | < 20ms，评委打开秒级响应 |
| **部署** | 5 条命令，无需 Docker |
| **续费** | ~38 元/年（秒杀价），用完即弃也可以 |

### 备选方案

| 平台 | 价格 | 适用场景 |
|------|------|----------|
| 腾讯云 | 新用户 1 个月免费 | 阿里云抢不到时使用 |
| 华为云 | 95 元/年 | 想长期运行时性价比高 |
| ngrok | 免费 | 本地测试、临时分享链接 |

---

## 四、服务器部署（6 步，~20 分钟）

### Step 1: 领取服务器 + 开放端口

```
1. 访问 free.aliyun.com → "轻量应用服务器" → 免费试用 1 个月
2. 地域: 离评委最近的地域（如杭州、北京）
3. 镜像: Ubuntu 22.04 LTS（纯系统）
4. 设置 root 密码
5. 记录公网 IP（如 47.xx.xx.xx）
6. 控制台 → 安全组 → 添加规则:
   - 协议: TCP
   - 端口: 8000
   - 授权对象: 0.0.0.0/0
```

### Step 2: SSH 登录

```bash
ssh root@47.xx.xx.xx
```

### Step 3: 安装基础依赖

```bash
apt update && apt install -y python3-pip git
```

### Step 4: 拉取代码

```bash
cd /root
git clone https://gitee.com/<你的用户名>/meituan-eval.git
cd meituan-eval
```

### Step 5: 配置环境变量

```bash
cp .env.example .env
nano .env  # 修改 API_KEY 为真实 Key
```

### Step 6: 执行部署

```bash
chmod +x deploy.sh
bash deploy.sh
```

---

## 五、部署后验证

### 5.1 检查服务

```bash
systemctl status meituan-web
# 应显示 active (running)
```

### 5.2 浏览器验证

```
http://<公网IP>:8000            # 系统主页（Tab 1-4 完整功能）
http://<公网IP>:8000/docs       # Swagger API 文档（测试所有端点）
```

### 5.3 WebSocket 验证

打开浏览器 DevTools → Network → 刷新页面 → 切换到新建评测 → 创建任务 → 确认 WS 连接建立。

### 5.4 预跑演示数据（可选）

```bash
python3 -m src.simulator.batch_runner --case-id 1 --n-profiles 3 --run-eval
python3 -m src.simulator.batch_runner --case-id 2 --n-profiles 3 --run-eval
```

评委打开网页 → Tab 1 演示效果 → 看到预跑评分报告，零等待了解系统能力。

---

## 六、运维命令速查

```bash
# 服务管理
systemctl status meituan-web          # 查看状态
systemctl restart meituan-web         # 重启服务
systemctl stop meituan-web            # 停止服务

# 日志查看
journalctl -u meituan-web -f          # 实时日志
journalctl -u meituan-web -n 50       # 最近 50 行
journalctl -u meituan-web --since "1 hour ago"  # 最近 1 小时

# 代码更新
cd /root/meituan-eval && git pull
systemctl restart meituan-web

# 磁盘清理（72h 旧任务自动清理，手动清理用）
ls data/exports/                      # 查看所有任务数据
rm -rf data/exports/<旧任务ID>        # 手动删除
```

---

## 七、安全加固（可选）

| 措施 | 操作 | 优先级 |
|------|------|--------|
| 改默认端口 | `.env` 中 `WEB_PORT=18888` + 安全组放行 | 低 |
| HTTPS | 用 nginx + Let's Encrypt 反向代理 | 中（Demo 场景可跳过） |
| 限制 CORS | `web/app.py` 中收紧 `allow_origins` | 低 |
| 系统更新 | `apt update && apt upgrade -y` | 部署后立即执行 |
| 访问控制 | 如果只需特定人访问，安全组限制来源 IP | 中 |

---

## 八、文件变更汇总

| 操作 | 文件 | 说明 |
|------|------|------|
| 新增 | `.env.example` | 环境变量模板 |
| 新增 | `deploy.sh` | 云服务器一键部署脚本 |
| 修改 | `.gitignore` | 添加 `data/checklist_evolution/` |
| 不修改 | 所有 `web/` + `src/` 代码 | 功能完整，无需改动 |
