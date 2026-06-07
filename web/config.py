"""Web 层配置 — 读取 .env + 定义输出目录 + 四槽位默认值"""
import os
from pathlib import Path

from dotenv import load_dotenv

# 加载项目根目录 .env
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ── 服务器配置 ──
HOST = os.getenv("WEB_HOST", "0.0.0.0")
PORT = int(os.getenv("WEB_PORT", "8000"))
# 预留：uvicorn workers 数（通过 run_web.py --workers 指定，此处仅定义默认值）
MAX_WORKERS = int(os.getenv("WEB_MAX_WORKERS", "1"))

# ── 输出目录 ──
OUTPUT_DIR = _PROJECT_ROOT / "data" / "exports"

# ── 全局默认 LLM 凭据（所有槽位的回退值） ──
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://api.deepseek.com")
MODEL = os.getenv("MODEL", "deepseek-chat")

# ── 槽位专用默认模型（可选覆盖，不设则回退到全局 MODEL） ──
SIMULATOR_MODEL = os.getenv("SIMULATOR_MODEL", "") or MODEL
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "") or MODEL
OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", "") or MODEL

# ── Task 生命周期 ──
TASK_TIMEOUT_SEC = int(os.getenv("TASK_TIMEOUT_SEC", "2700"))       # 45 min
TASK_RETENTION_HOURS = int(os.getenv("TASK_RETENTION_HOURS", "72"))  # 3 天
CLEANUP_INTERVAL_SEC = int(os.getenv("CLEANUP_INTERVAL_SEC", "600"))  # 10 min

# ── 速率限制（预留，未来接入 slowapi 或自定义限流中间件时启用） ──
RATE_LIMIT_TASK = os.getenv("RATE_LIMIT_TASK", "1/10second")
RATE_LIMIT_TEST = os.getenv("RATE_LIMIT_TEST", "1/5second")

# ── 演示模式 ──
DEMO_CASE_PATH = Path(__file__).parent / "demo_case.md"
PRESET_CASES_PATH = _PROJECT_ROOT / "generated_cases_all.json"
