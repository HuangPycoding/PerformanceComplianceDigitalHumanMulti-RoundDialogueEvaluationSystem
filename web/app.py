"""FastAPI 应用工厂"""
import asyncio
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from web.router import router
from web.task_manager import task_manager

# ── 创建应用 ──
app = FastAPI(
    title="美团外呼对话模型评测系统",
    description="LLM 评测系统 Demo Web",
    version="1.0.0",
)

# ── CORS（Demo 级别：允许所有来源） ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 路由 ──
app.include_router(router)

# ── 静态文件 ──
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# ── 持久化任务引用 ──
_cleanup_task = None


# ── 根路径返回 index.html ──
@app.get("/")
async def root():
    index_path = _static_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "Web 服务运行中。静态文件尚未部署。"}


# ── 启动/关闭生命周期 ──
@app.on_event("startup")
async def startup():
    """启动时恢复历史任务 + 启动清理协程"""
    global _cleanup_task
    task_manager.recover_from_disk()
    _cleanup_task = asyncio.create_task(task_manager.cleanup_loop())


@app.on_event("shutdown")
async def shutdown():
    """取消清理协程"""
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
