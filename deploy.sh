#!/bin/bash
# ============================================================
# 美团外呼对话模型评测系统 — 一键部署脚本
# 用法: chmod +x deploy.sh && sudo bash deploy.sh
# 适用: Ubuntu 22.04+ (阿里云/腾讯云/华为云轻量应用服务器)
# ============================================================
set -e

echo "=============================================="
echo "  美团外呼评测系统 — 一键部署"
echo "=============================================="
echo ""

# ---- 1. 系统依赖 ----
echo "[1/5] 安装系统依赖..."
apt update -qq
apt install -y -qq python3-pip python3-venv git
echo "  -> 完成"
echo ""

# ---- 2. Python 依赖 ----
echo "[2/5] 安装 Python 依赖..."
pip3 install -q -r requirements.txt
echo "  -> 完成"
echo ""

# ---- 3. 检查 .env ----
echo "[3/5] 检查配置文件..."
if [ ! -f .env ]; then
    echo "  -> 未找到 .env，从 .env.example 创建模板..."
    cp .env.example .env
    echo "  -> ========================================"
    echo "  ->  请先编辑 .env 填入 API Key:"
    echo "  ->  nano .env"
    echo "  ->  然后重新运行: bash deploy.sh"
    echo "  -> ========================================"
    exit 1
fi
echo "  -> .env 已就绪"
echo ""

# ---- 4. systemd 服务 ----
echo "[4/5] 配置 systemd 服务..."

# 获取当前目录的绝对路径
WORKDIR=$(pwd)
PYTHON_PATH=$(which python3)

cat > /etc/systemd/system/meituan-web.service << SVC_END
[Unit]
Description=Meituan LLM Evaluation Web
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=${WORKDIR}
ExecStart=${PYTHON_PATH} -m uvicorn web.app:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

# 安全加固（可选）
# NoNewPrivileges=yes
# PrivateTmp=yes

[Install]
WantedBy=multi-user.target
SVC_END

systemctl daemon-reload
systemctl enable meituan-web
systemctl restart meituan-web
echo "  -> systemd 服务已配置并启动"
echo ""

# ---- 5. 验证 ----
echo "[5/5] 检查服务状态..."
sleep 3

if systemctl is-active --quiet meituan-web; then
    echo "  -> 服务状态: ACTIVE ✓"
else
    echo "  -> 服务状态: FAILED ✗"
    echo "  -> 查看日志: journalctl -u meituan-web -n 20"
fi

PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || echo "未知")
echo ""
echo "=============================================="
echo "  部署完成！"
echo "=============================================="
echo "  公网访问: http://${PUBLIC_IP}:8000"
echo "  API 文档: http://${PUBLIC_IP}:8000/docs"
echo ""
echo "  运维命令:"
echo "    查看状态: systemctl status meituan-web"
echo "    查看日志: journalctl -u meituan-web -f"
echo "    重启服务: systemctl restart meituan-web"
echo "=============================================="
