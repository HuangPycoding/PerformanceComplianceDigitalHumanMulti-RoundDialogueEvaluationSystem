"""FastAPI 路由 — REST + WebSocket 端点"""
import asyncio
import json
import time
import zipfile
import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from src.llm.client import LLMClient
from src.loader.case_loader import load_cases
from src.loader.case_parser import parse_instruction

from web import config
from web.schemas import (
    CreateTaskRequest, TaskCreatedResponse, TaskStatusResponse,
    TaskListItem, TaskDetailResponse, TestConnectionRequest,
    TestConnectionResponse, ParseCaseRequest, ParseCaseResponse,
    PresetCaseSummary, DemoConfigResponse, TaskDeleteResponse,
    ConversationTurn, EffectiveModels,
)
from web.task_manager import task_manager
from web.ws_manager import ws_manager

router = APIRouter()


# ── Case 相关 ──

def _load_all_cases():
    """加载预置 Case"""
    try:
        return load_cases()
    except Exception:
        return []


@router.get("/api/presets", response_model=list[PresetCaseSummary])
async def get_presets():
    """获取预置 Case 模板列表"""
    cases = _load_all_cases()
    return [
        PresetCaseSummary(
            id=c.id,
            title=c.title,
            business_line=c.business_line,
            complexity_score=c.complexity_score,
        )
        for c in cases
    ]


@router.get("/api/presets/{case_id}")
async def get_preset_detail(case_id: int):
    """获取预置 Case 完整文本"""
    cases = _load_all_cases()
    for c in cases:
        if c.id == case_id:
            return {"id": c.id, "title": c.title, "instruction": c.raw_instruction}
    raise HTTPException(404, "Case not found")


@router.post("/api/cases/parse", response_model=ParseCaseResponse)
async def parse_case(req: ParseCaseRequest):
    """预览解析用户输入的 Case 文本"""
    try:
        case = parse_instruction(req.case_text, case_id=0, title="")
        return ParseCaseResponse(
            title=case.title,
            business_line=case.business_line,
            role=case.role,
            task=case.task,
            opening_line=case.opening_line,
            call_flow_steps=[
                {"step": s.step, "action": s.action}
                for s in case.call_flow
            ],
            constraints=[
                {"type": c.type, "description": c.description}
                for c in case.constraints
            ],
            knowledge_points=[
                {"title": k.title, "content": k.content}
                for k in case.knowledge_points
            ],
        )
    except Exception as e:
        raise HTTPException(400, f"Case 解析失败: {e}")


# ── 任务相关 ──

@router.post("/api/tasks")
async def create_task(req: CreateTaskRequest):
    """创建评测任务"""
    if not req.demo_mode and not req.case_text.strip():
        raise HTTPException(400, "Case 文本不能为空")
    if req.demo_mode and not req.case_text.strip():
        # 演示模式无文本时加载 demo_case.md
        if config.DEMO_CASE_PATH.exists():
            req.case_text = config.DEMO_CASE_PATH.read_text(encoding="utf-8")
        else:
            cases = _load_all_cases()
            if cases:
                req.case_text = cases[0].raw_instruction

    task = task_manager.create_task(
        case_text=req.case_text,
        case_title=req.case_title or "",
        demo_mode=req.demo_mode,
        llm_config=req.llm_config,
        n_profiles=req.n_profiles,
        run_eval=req.run_eval,
        run_optimize=req.run_optimize,
    )
    return TaskCreatedResponse(
        task_id=task.task_id,
        status="accepted",
        mode="demo" if task.demo_mode else "custom",
        effective_models=task.effective_models,
    )


@router.get("/api/tasks")
async def list_tasks(limit: int = Query(default=20, le=100)):
    """列出历史任务"""
    tasks = task_manager.list_tasks(limit=limit)
    return [_build_task_list_item(t) for t in tasks]


def _get_rating_from_disk(output_dir: Path) -> Optional[str]:
    """从磁盘评测文件读取评级"""
    for f in sorted(output_dir.glob("evaluation_*.json")):
        try:
            edata = json.loads(f.read_text(encoding="utf-8"))
            if "rating_label" in edata and edata["rating_label"]:
                return edata["rating_label"]
            return _compute_overall_rating(edata)
        except Exception:
            pass
    return None


@router.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """获取任务状态"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return TaskStatusResponse(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
        case_title=task.case_title,
        n_profiles=task.n_profiles,
        mode="demo" if task.demo_mode else "custom",
        progress=task.progress,
        result_summary=task.result_summary,
        effective_models=task.effective_models,
    )


@router.get("/api/tasks/{task_id}/models")
async def get_task_models(task_id: str):
    """获取任务模型配置（不返回 api_key）"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    return {
        "assistant": {"model": task.effective_models.assistant},
        "simulator": {"model": task.effective_models.simulator},
        "evaluator": {"model": task.effective_models.evaluator},
        "optimizer": {"model": task.effective_models.optimizer},
    }


@router.get("/api/tasks/{task_id}/detail")
async def get_task_detail(task_id: str):
    """获取任务完整详情（合并评测+对话+优化）"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")

    eval_results = []
    conversations = []
    optimization = None

    if task.output_dir and task.output_dir.exists():
        # 加载所有评测结果
        for f in sorted(task.output_dir.glob("evaluation_*.json")):
            try:
                edata = json.loads(f.read_text(encoding="utf-8"))
                # 添加评级标签
                if "rating_label" not in edata and "ratings" in edata:
                    edata["rating_label"] = _compute_overall_rating(edata)
                eval_results.append(edata)
            except Exception:
                pass

        # 加载所有对话
        for f in sorted(task.output_dir.glob("conversation_*.json")):
            try:
                cdata = json.loads(f.read_text(encoding="utf-8"))
                conversations.append(cdata)
            except Exception:
                pass

        # 加载优化建议
        opt_dir = task.output_dir / "optimization"
        if opt_dir.exists():
            for f in sorted(opt_dir.glob("*.json")):
                try:
                    optimization = json.loads(f.read_text(encoding="utf-8"))
                    break
                except Exception:
                    pass

    # 合并第一个评测结果 + 第一个对话数据
    primary_eval = eval_results[0] if eval_results else None
    primary_conv = conversations[0] if conversations else None

    # 读取 MD 报告内容
    report_md = ""
    opt_report_md = ""
    if task.output_dir and task.output_dir.exists():
        rm = task.output_dir / "report.md"
        if rm.exists():
            report_md = rm.read_text(encoding="utf-8")[:20000]
        om = task.output_dir / "optimization_report.md"
        if om.exists():
            opt_report_md = om.read_text(encoding="utf-8")[:20000]

    return TaskDetailResponse(
        task_id=task.task_id,
        status=task.status,
        created_at=task.created_at,
        case_title=task.case_title,
        mode="demo" if task.demo_mode else "custom",
        effective_models=task.effective_models,
        eval_result=primary_eval,
        all_eval_results=eval_results if eval_results else None,
        optimization_suggestions=[optimization] if optimization else None,
        conversation_summary=primary_conv,
        all_conversations=conversations if conversations else None,
        report_md=report_md,
        optimization_report_md=opt_report_md,
    )


def _compute_overall_rating(edata: dict) -> str:
    """从 ratings 计算整体评级"""
    ratings = edata.get("ratings", {})
    if not ratings:
        return edata.get("rating_label", "N/A")
    rank = {"卓越": 5, "良好": 4, "合格": 3, "需改进": 2, "不合格": 1}
    values = [rank.get(v, 0) for v in ratings.values() if v in rank]
    if not values:
        return "N/A"
    avg = sum(values) / len(values)
    if avg >= 4.5:
        return "卓越"
    elif avg >= 3.5:
        return "良好"
    elif avg >= 2.5:
        return "合格"
    elif avg >= 1.5:
        return "需改进"
    return "不合格"


@router.get("/api/tasks/{task_id}/conversations")
async def get_task_conversations(task_id: str):
    """获取对话全文"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    if not task.output_dir or not task.output_dir.exists():
        raise HTTPException(404, "Conversation data not found")

    conv_files = sorted(task.output_dir.glob("conversation_*.json"))
    if not conv_files:
        raise HTTPException(404, "No conversation files found")

    all_turns = []
    try:
        data = json.loads(conv_files[0].read_text(encoding="utf-8"))
        if isinstance(data, list):
            all_turns = data
        elif isinstance(data, dict):
            all_turns = data.get("turns", [])
    except Exception:
        pass

    return all_turns


@router.get("/api/tasks/{task_id}/result")
async def get_task_result(task_id: str, summary: bool = Query(default=True)):
    """获取评测结果"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    if task.status in ("queued", "pending", "parsing", "phase_profiles",
                        "phase_dialogues", "phase_eval", "phase_optimize"):
        return {"status": task.status, "message": "任务尚未完成，请等待"}

    if not task.output_dir or not task.output_dir.exists():
        raise HTTPException(404, "Result data not found")

    eval_files = sorted(task.output_dir.glob("evaluation_*.json"))
    if not eval_files:
        raise HTTPException(404, "No evaluation files found")

    try:
        data = json.loads(eval_files[0].read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(500, "Failed to read result file")

    if summary:
        return {
            "task_id": task.task_id,
            "total_score_100": data.get("total_score_100", 0),
            "rating_label": data.get("rating_label", ""),
            "dimension_scores": data.get("dimension_scores", {}),
        }
    return data


@router.get("/api/tasks/{task_id}/download/{file_type}")
async def download_task_file(task_id: str, file_type: str):
    """下载任务文件"""
    task = task_manager.get_task(task_id)
    if task is None:
        raise HTTPException(404, "Task not found")
    if not task.output_dir or not task.output_dir.exists():
        raise HTTPException(404, "Files not found")

    # 单文件下载
    file_map = {
        "report_md": "report.md",
        "report_json": "evaluation_1.json",
        "conversations_json": "conversation_1.json",
        "batch_summary": "batch_summary.json",
    }

    if file_type in file_map:
        file_path = task.output_dir / file_map[file_type]
        if not file_path.exists():
            # 尝试匹配模式
            pattern = file_map[file_type].replace("_1", "_*")
            matches = sorted(task.output_dir.glob(pattern))
            if not matches:
                raise HTTPException(404, f"No {file_type} file found")
            file_path = matches[0]
        return FileResponse(
            file_path,
            filename=file_path.name,
            media_type="application/octet-stream",
        )

    # ZIP 打包下载
    if file_type == "all_zip":
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in task.output_dir.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(task.output_dir))
        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=task_{task_id}.zip"},
        )

    raise HTTPException(400, f"Unknown file type: {file_type}")


@router.post("/api/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    """取消任务"""
    if task_manager.cancel_task(task_id):
        return {"ok": True, "message": "取消请求已发送"}
    raise HTTPException(404, "Task not found or already completed")


@router.delete("/api/tasks/{task_id}", response_model=TaskDeleteResponse)
async def delete_task(task_id: str):
    """删除历史任务"""
    if task_manager.delete_task(task_id):
        return TaskDeleteResponse(ok=True)
    raise HTTPException(404, "Task not found or cannot be deleted (still running)")


# ── 演示模式配置 ──

def _build_task_list_item(t) -> TaskListItem:
    """构建任务列表项（含评级）"""
    score = t.result_summary.get("total_score_100") if t.result_summary else None
    rating = t.result_summary.get("rating_label") if t.result_summary else None
    if not rating and t.output_dir and t.output_dir.exists():
        rating = _get_rating_from_disk(t.output_dir)
    return TaskListItem(
        task_id=t.task_id, status=t.status, created_at=t.created_at,
        case_title=t.case_title, n_profiles=t.n_profiles,
        total_score_100=score, rating_label=rating,
    )


@router.get("/api/demo/config", response_model=DemoConfigResponse)
async def get_demo_config():
    """获取演示模式配置"""
    tasks = task_manager.list_tasks(limit=50)
    completed = [t for t in tasks if t.status == "completed"]
    return DemoConfigResponse(
        available=config.DEMO_CASE_PATH.exists(),
        demo_case_ids=[1, 2],  # 预跑 Case
        demo_tasks=[
            _build_task_list_item(t)
            for t in completed
        ],
        effective_models=EffectiveModels(
            assistant=config.MODEL,
            simulator=config.SIMULATOR_MODEL,
            evaluator=config.EVALUATOR_MODEL,
            optimizer=config.OPTIMIZER_MODEL,
        ),
    )


# ── 工具 ──

@router.post("/api/test-connection")
async def test_connection(req: TestConnectionRequest):
    """测试 API 连通性"""
    # 确定默认值
    default_api_key = config.API_KEY
    default_base_url = config.BASE_URL
    default_model = config.MODEL

    slot_defaults = {
        "assistant": {"model": config.MODEL},
        "simulator": {"model": config.SIMULATOR_MODEL},
        "evaluator": {"model": config.EVALUATOR_MODEL},
        "optimizer": {"model": config.OPTIMIZER_MODEL},
    }
    defaults = slot_defaults.get(req.slot, slot_defaults["assistant"])

    api_key = req.api_key or default_api_key
    base_url = req.base_url or default_base_url
    model = req.model or defaults["model"]

    try:
        client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=0.0,
            max_retries=1,
            timeout=15,
        )
        start = time.time()
        response = await asyncio.to_thread(
            client.chat,
            system_prompt="你是一个API连通性测试助手。",
            user_message="请仅回复 'OK'。",
        )
        latency_ms = int((time.time() - start) * 1000)
        return TestConnectionResponse(
            ok="OK" in response,
            message=f"连接成功，延迟 {latency_ms}ms",
            latency_ms=latency_ms,
        )
    except Exception as e:
        err_msg = str(e)[:200]
        # 尝试遮盖 API Key（安全防护）
        if api_key and len(api_key) > 8:
            err_msg = err_msg.replace(api_key, api_key[:4] + "****" + api_key[-4:])
        return TestConnectionResponse(
            ok=False,
            message=f"连接失败: {err_msg}",
            latency_ms=0,
        )


# ── WebSocket ──

@router.websocket("/ws/task/{task_id}")
async def ws_task(websocket: WebSocket, task_id: str):
    """WebSocket 实时进度"""
    task = task_manager.get_task(task_id)
    if task is None:
        await websocket.close(code=4404)
        return

    await ws_manager.connect(task_id, websocket)

    try:
        # 如果任务已完成，发送快照后关闭
        if task.status in ("completed", "failed", "timeout", "cancelled"):
            await websocket.send_json({
                "type": "completed" if task.status == "completed" else "error",
                "task_id": task_id,
                "status": task.status,
                "result_summary": task.result_summary,
            })
            return

        # 发送当前状态
        await websocket.send_json({
            "type": "status",
            "task_id": task_id,
            "status": task.status,
            "progress": task.progress,
        })

        # 事件通过 ws_manager.broadcast() 直接推送到客户端；
        # 此循环仅负责心跳 + 任务终态检测，避免从 event_queue 读取导致重复发送
        while task.status not in ("completed", "failed", "timeout", "cancelled"):
            await asyncio.sleep(30)
            try:
                await websocket.send_json({"type": "heartbeat"})
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ws_manager.disconnect(task_id, websocket)
