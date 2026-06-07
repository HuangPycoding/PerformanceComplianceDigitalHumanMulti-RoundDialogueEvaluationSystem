@echo off
echo ========================================
echo  美团 LLM 评测系统 — 本地 Demo 启动
echo ========================================
echo.
echo 确保已安装 ngrok.exe 并放入 PATH 或项目根目录
echo.

REM 启动 ngrok（后台）
start "ngrok" ngrok http 8000

REM 等待 ngrok 初始化
timeout /t 3 >nul

REM 启动 Web 服务
python run_web.py

pause
