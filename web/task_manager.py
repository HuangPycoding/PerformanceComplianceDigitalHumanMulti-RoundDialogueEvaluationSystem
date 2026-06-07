"""Task 管理 — 创建、四槽位构建、流水线编排、生命周期"""
import asyncio
import datetime
import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.llm.client import LLMClient
from src.loader.case_parser import parse_instruction
from src.simulator.batch_runner import BatchRunner
from src.optimizer.optimizer import OptimizationEngine

from web import config
from web.schemas import (
    EffectiveModels, LLMConfigSlots, LLMSlotConfig,
    TaskListItem, TaskStatusResponse, TaskDetailResponse, ResultSummary,
)
from web.ws_manager import ws_manager


# ── 导出辅助 ──

def _serialize_obj(obj, depth=0):
    """递归序列化任意对象为 JSON 兼容的 dict/list/基本类型"""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if depth > 10:
        return str(obj)[:1000]
    if isinstance(obj, dict):
        return {str(k): _serialize_obj(v, depth+1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_serialize_obj(i, depth+1) for i in obj]
    if hasattr(obj, '__dict__') and not isinstance(obj, type):
        result = {}
        for k, v in obj.__dict__.items():
            if k.startswith('_'):
                continue
            result[k] = _serialize_obj(v, depth+1)
        return result
    try:
        return str(obj)[:5000]
    except Exception:
        return f"<{type(obj).__name__}>"


def _export_results(output_dir: Path, case, profiles_dict: dict,
                    conversations: list, eval_results: list) -> None:
    """将评测结果导出到磁盘"""
    import json as _json
    output_dir.mkdir(parents=True, exist_ok=True)

    # 评测结果 JSON
    if eval_results:
        for i, r in enumerate(eval_results):
            er_dict = _serialize_obj(r) if hasattr(r, '__dict__') else {}
            (output_dir / f"evaluation_{i+1}.json").write_text(
                _json.dumps(er_dict, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

    # 对话 JSON
    if conversations:
        for i, conv in enumerate(conversations):
            if hasattr(conv, '__dict__'):
                conv_dict = _serialize_obj(conv)
            elif hasattr(conv, 'to_dict'):
                conv_dict = conv.to_dict()
            else:
                conv_dict = {}
            (output_dir / f"conversation_{i+1}.json").write_text(
                _json.dumps(conv_dict, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

    # 优化报告 MD
    if eval_results:
        _generate_optimization_report_md(output_dir, conversations, eval_results)

    # 用户画像数据
    if profiles_dict:
        profiles_data = {}
        for cid, plist in profiles_dict.items():
            profiles_data[str(cid)] = []
            for p in plist:
                if hasattr(p, '__dict__'):
                    profiles_data[str(cid)].append(_serialize_obj(p))
                elif isinstance(p, dict):
                    profiles_data[str(cid)].append(p)
                else:
                    profiles_data[str(cid)].append({"label": str(p)})
        try:
            (output_dir / "profiles.json").write_text(
                _json.dumps(profiles_data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

        # 补全对话 JSON 中的字段
        for i, conv in enumerate(conversations or []):
            if hasattr(conv, 'user_profile') and hasattr(conv.user_profile, 'adversarial_strategy'):
                conv.adversarial_strategies = list(conv.user_profile.adversarial_strategy)
            cf = output_dir / f"conversation_{i+1}.json"
            if cf.exists():
                try:
                    cdata = _json.loads(cf.read_text(encoding="utf-8"))
                    if case and hasattr(case, 'complexity_score') and case.complexity_score:
                        cdata.setdefault('complexity_score', case.complexity_score)
                    found_label = False
                    for plist in profiles_dict.values():
                        for p in plist:
                            if hasattr(p, 'profile_label') and p.profile_label:
                                cdata.setdefault('profile_label', str(p.profile_label))
                                found_label = True
                                break
                        if found_label:
                            break
                    cf.write_text(_json.dumps(cdata, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

    # Case 详情
    if case:
        try:
            case_data = _serialize_obj(case) if hasattr(case, '__dict__') else {"title": str(case)}
            (output_dir / "case.json").write_text(
                _json.dumps(case_data, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    # 批次摘要
    scores = []
    for er in (eval_results or []):
        try:
            if hasattr(er, 'total_score_100'):
                val = er.total_score_100
            elif isinstance(er, dict):
                val = er.get('total_score_100', 0)
            else:
                continue
            if val and isinstance(val, (int, float)):
                scores.append(float(val))
        except Exception:
            pass
    batch_summary = {
        "case_title": case.title if case else "",
        "n_profiles": sum(len(v) for v in (profiles_dict or {}).values()),
        "n_conversations": len(conversations),
        "n_evaluations": len(eval_results),
        "average_score": sum(scores) / len(scores) if scores else 0,
        "exported_at": str(datetime.now()),
    }
    try:
        (output_dir / "batch_summary.json").write_text(
            _json.dumps(batch_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        pass

    # MANIFEST
    try:
        manifest_lines = ["# 导出文件清单\n"]
        for f in sorted(output_dir.iterdir()):
            if f.is_file():
                manifest_lines.append(f"- {f.name} ({f.stat().st_size} bytes)")
        (output_dir / "MANIFEST.md").write_text("\n".join(manifest_lines), encoding="utf-8")
    except Exception:
        pass


def _build_summary_section(eval_results: list) -> str:
    """构建综合总结章节"""
    lines = ["## 综合总结", "", "| 维度 | 平均分 | 评价 |", "|------|--------|------|"]
    dim_scores = {}
    for er in (eval_results or []):
        scores = er.get('indicative_scores', {}) if isinstance(er, dict) else getattr(er, 'indicative_scores', {})
        for dim, sc in (scores if isinstance(scores, dict) else {}).items():
            if isinstance(sc, (int, float)):
                dim_scores.setdefault(dim, []).append(sc)
    for dim, vals in dim_scores.items():
        avg = sum(vals) / len(vals)
        level = "优秀" if avg >= 8 else "良好" if avg >= 6 else "一般" if avg >= 4 else "需改进"
        lines.append(f"| {dim} | {avg:.1f} | {level} |")
    return "\n".join(lines)


def _format_single_report_json(idx: int, er, conv) -> str:
    """单个对话的报告（JSON 数据格式 fallback）"""
    lines = [f"## 对话 {idx}"]
    if isinstance(er, dict):
        s = er.get('total_score_100', '?')
        r = er.get('rating_label', '?')
        lines.append(f"- 总分: {s} 分 | 评级: {r}")
        ratings = er.get('ratings', {})
        if ratings:
            lines.append("\n| 维度 | 评级 | 分数 |")
            lines.append("|------|------|------|")
            for dim, rat in ratings.items():
                sc = er.get('indicative_scores', {}).get(dim, '-')
                lines.append(f"| {dim} | {rat} | {sc} |")
            lines.append("")
        imps = er.get('improvement_suggestions', [])
        if imps:
            lines.append("\n### 改进建议")
            for j, imp in enumerate(imps[:10]):
                txt = imp if isinstance(imp, str) else str(imp)[:200]
                lines.append(f"{j+1}. {txt}")
    if hasattr(conv, 'status') and hasattr(conv, 'total_turns'):
        lines.append(f"\n状态: {conv.status} ({conv.total_turns} 轮)")
    return "\n".join(lines)


DIM_NAMES_CN = {
    "SAFETY": "安全合规", "TASK_COMPLETION": "任务完成度", "FLOW_COVERAGE": "流程覆盖",
    "CONSTRAINT": "约束遵守", "KNOWLEDGE": "知识点运用", "ROLE": "角色扮演",
    "OPENING": "开场白", "SENTIMENT": "情感处理", "EFFICIENCY": "交互效率",
}


def _generate_narrative_report_md(output_dir: Path, case, conversations: list, eval_results: list) -> None:
    """生成叙事性详细评测报告 MD"""
    title = case.title if case and hasattr(case, 'title') else "评测任务"
    cx = case.complexity_score if case and hasattr(case, 'complexity_score') else 0
    lines = [f"# 评测报告: {title}", "", f"## 评测概览", "",
             f"- 指令复杂度: {cx}", f"- 评测对话数: {len(eval_results)}", ""]
    all_scores = [er['total_score_100'] for er in (eval_results or []) if isinstance(er, dict) and er.get('total_score_100')]
    if all_scores:
        lines.extend([f"- 平均分: {sum(all_scores)/len(all_scores):.0f} 分",
                      f"- 最高分: {max(all_scores)} 分", f"- 最低分: {min(all_scores)} 分", ""])
    for i, (er, conv) in enumerate(zip(eval_results or [], conversations or [])):
        er_s = er if isinstance(er, dict) else {}
        ratings = er_s.get('ratings', {})
        scores = er_s.get('indicative_scores', {})
        dcl = er_s.get('dimension_checklists', {})
        s = er_s.get('total_score_100', 0)
        rl = er_s.get('rating_label', 'N/A')
        st = getattr(conv, 'status', '?') if hasattr(conv, 'status') else conv.get('status', '?') if isinstance(conv, dict) else '?'
        tt = getattr(conv, 'total_turns', '?') if hasattr(conv, 'total_turns') else conv.get('total_turns', '?') if isinstance(conv, dict) else '?'
        lines.extend([f"## 对话 {i+1}", "", f"**总分**: {s} 分 | **评级**: {rl} | **状态**: {st} ({tt} 轮)", ""])
        best = [d for d, r in ratings.items() if r == '卓越']
        worst = [(d, r) for d, r in ratings.items() if r in ('不合格', '需改进')]
        if best: lines.append(f"**优势维度**: {', '.join(best)} 表现卓越")
        if worst: lines.append(f"**问题维度**: {', '.join(f'{d}({r})' for d, r in worst)} 需要改进")
        lines.append("")
        lines.append("### 维度详细分析")
        for dim in DIM_NAMES_CN:
            r = ratings.get(dim, 'N/A'); sc = scores.get(dim, '-')
            lines.append(f"**{dim} ({DIM_NAMES_CN[dim]})**: 评级 {r} | 分数 {sc}")
            items = dcl.get(dim, [])
            if isinstance(items, list):
                for item in items[:3]:
                    if isinstance(item, dict):
                        lines.append(f"  - [{item.get('status','?')}] {item.get('description','')[:120]}")
                        if item.get('evidence'): lines.append(f"    证据: {item.get('evidence','')[:120]}")
            lines.append("")
        imps = er_s.get('improvement_suggestions', [])
        if imps:
            lines.append("### 改进建议")
            for j, imp in enumerate(imps[:5]):
                lines.append(f"{j+1}. {imp if isinstance(imp, str) else str(imp)[:300]}")
            lines.append("")
        lines.append("---\n")
    lines.append("## 综合总结")
    lines.append("| 维度 | 名称 | 平均分 | 评价 |")
    lines.append("|------|------|--------|------|")
    for dim, name in DIM_NAMES_CN.items():
        vals = [er.get('indicative_scores', {}).get(dim, 0) for er in (eval_results or [])
                if isinstance(er, dict) and isinstance(er.get('indicative_scores', {}).get(dim), (int, float))]
        avg = sum(vals) / len(vals) if vals else 0
        level = "优秀" if avg >= 8 else "良好" if avg >= 6 else "一般" if avg >= 4 else "需改进"
        lines.append(f"| {dim} | {name} | {avg:.1f} | {level} |")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _generate_optimization_report_md(output_dir: Path, conversations: list, eval_results: list) -> None:
    """生成优化建议 MD 报告"""
    lines = ["# 优化建议报告\n", f"## 概览", f"- 对话数量: {len(conversations)}",
             f"- 评测数量: {len(eval_results)}", f"- 生成时间: {datetime.now()}\n"]
    all_imps = []
    for i, er in enumerate(eval_results or []):
        if not isinstance(er, dict): continue
        imps = er.get('improvement_suggestions', [])
        if imps:
            lines.append(f"### 对话 {i+1} 的改进建议")
            for j, imp in enumerate(imps[:10]):
                lines.append(f"{j+1}. {imp if isinstance(imp, str) else str(imp)[:500]}")
            lines.append("")
        all_imps.extend(imps if isinstance(imps, list) else [imps])
    lines.append(f"## 汇总（共 {len(all_imps)} 条建议）")
    (output_dir / "optimization_report.md").write_text("\n".join(lines), encoding="utf-8")


def _generate_basic_report(output_dir: Path, case, conversations: list, eval_results: list) -> None:
    """生成综合 Markdown 报告（含所有可用数据）"""
    lines = [f"# 评测报告\n"]
    title = case.title if case and hasattr(case, 'title') else "评测任务"
    lines.append(f"## Case: {title}")
    if case and hasattr(case, 'complexity_score'):
        lines.append(f"- 指令复杂度: {case.complexity_score}")
    lines.append(f"- 对话数量: {len(conversations)}")
    lines.append(f"- 评测数量: {len(eval_results)}")
    all_scores = []
    all_dim_scores = {}
    for i, er in enumerate(eval_results or []):
        if not isinstance(er, dict): continue
        s = er.get('total_score_100', 0)
        if s: all_scores.append(s)
        lines.append(f"\n## 对话 {i+1}")
        lines.append(f"- 总分: {s} 分")
        if er.get('rating_label'): lines.append(f"- 评级: {er['rating_label']}")
        if i < len(conversations or []):
            conv = conversations[i]
            lines.append(f"- 状态: {getattr(conv, 'status', '?')} ({getattr(conv, 'total_turns', '?')} 轮)")
        ratings = er.get('ratings', {})
        if ratings:
            lines.append("\n| 维度 | 评级 | 分数 |")
            lines.append("|------|------|------|")
            for dim, rating in ratings.items():
                score = er.get('indicative_scores', {}).get(dim, '-')
                lines.append(f"| {dim} | {rating} | {score} |")
                if dim not in all_dim_scores: all_dim_scores[dim] = []
                all_dim_scores[dim].append(score if isinstance(score, (int, float)) else 0)
        if er.get('improvement_suggestions'):
            lines.append(f"\n### 改进建议")
            for j, imp in enumerate(er['improvement_suggestions'][:10]):
                lines.append(f"{j+1}. {imp if isinstance(imp, str) else str(imp)[:200]}")
    if all_scores:
        lines.insert(4, f"- 平均分: {sum(all_scores)/len(all_scores):.0f}")
        lines.insert(5, f"- 最高分: {max(all_scores)}")
        lines.insert(6, f"- 最低分: {min(all_scores)}")
    if all_dim_scores:
        lines.append(f"\n## 维度平均分")
        lines.append("| 维度 | 平均分 |")
        lines.append("|------|--------|")
        for dim, scores in all_dim_scores.items():
            lines.append(f"| {dim} | {sum(scores)/len(scores):.1f} |")
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ── Task 数据类 ──

@dataclass
class Task:
    task_id: str
    case_text: str
    case_title: str
    demo_mode: bool
    llm_config: LLMConfigSlots
    n_profiles: int
    run_eval: bool
    run_optimize: bool
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    progress: Dict[str, Any] = field(default_factory=dict)
    result_summary: Optional[dict] = None
    effective_models: EffectiveModels = field(default_factory=EffectiveModels)
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancel_event: threading.Event = field(default_factory=threading.Event)
    output_dir: Optional[Path] = None
    _client_allocated: bool = False


# ── TaskManager ──

class TaskManager:
    """Task 生命周期管理器（单例）"""

    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._running: Optional[str] = None
        self._queue: List[str] = []

    def create_task(self, case_text: str, case_title: str = "", demo_mode: bool = True,
                    llm_config: Optional[LLMConfigSlots] = None, n_profiles: int = 3,
                    run_eval: bool = True, run_optimize: bool = False) -> Task:
        task_id = uuid.uuid4().hex[:12]
        task = Task(task_id=task_id, case_text=case_text, case_title=case_title or "外部 Case",
                    demo_mode=demo_mode, llm_config=llm_config or LLMConfigSlots(),
                    n_profiles=n_profiles, run_eval=run_eval, run_optimize=run_optimize)
        task.output_dir = config.OUTPUT_DIR / task_id
        with self._lock:
            self._tasks[task_id] = task
        task.effective_models = self._resolve_effective_models(task)
        if self._running is None:
            self._running = task_id
            task.status = "pending"
            asyncio.create_task(self._run_task(task))
        else:
            self._queue.append(task_id)
            task.status = "queued"
            position = len(self._queue) + 1
            asyncio.create_task(self._push_event(task, {
                "type": "queued", "position": position, "estimated_wait_sec": position * 600,
            }))
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        return self._tasks.get(task_id)

    def list_tasks(self, limit: int = 20) -> List[Task]:
        tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    def delete_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None: return False
        if task.status in ("queued", "pending", "parsing", "phase_profiles", "phase_dialogues", "phase_eval", "phase_optimize"):
            return False
        with self._lock: del self._tasks[task_id]
        if task.output_dir and task.output_dir.exists():
            shutil.rmtree(task.output_dir, ignore_errors=True)
        task.llm_config = LLMConfigSlots()
        return True

    def cancel_task(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if task is None: return False
        task.cancel_event.set()
        return True

    def _resolve_effective_models(self, task: Task) -> EffectiveModels:
        cfg = task.llm_config
        return EffectiveModels(
            assistant=(cfg.assistant and cfg.assistant.model) or "评委自填",
            simulator=(cfg.simulator and cfg.simulator.model) or config.SIMULATOR_MODEL,
            evaluator=(cfg.evaluator and cfg.evaluator.model) or config.EVALUATOR_MODEL,
            optimizer=(cfg.optimizer and cfg.optimizer.model) or config.OPTIMIZER_MODEL,
        )

    def _build_clients(self, task: Task):
        cfg = task.llm_config
        def _make(slot, default_model, temperature):
            if slot is None: slot = LLMSlotConfig()
            return LLMClient(api_key=slot.api_key or config.API_KEY,
                             base_url=slot.base_url or config.BASE_URL,
                             model=slot.model or default_model, temperature=temperature)
        assistant_cfg = cfg.assistant or LLMSlotConfig()
        if task.demo_mode:
            assistant_client = LLMClient(api_key=config.API_KEY, base_url=config.BASE_URL, model=config.MODEL, temperature=0.0)
        else:
            if not assistant_cfg.api_key:
                raise ValueError("自定义模式下被评测模型 API Key 为必填项")
            assistant_client = LLMClient(api_key=assistant_cfg.api_key,
                                         base_url=assistant_cfg.base_url or "https://api.deepseek.com",
                                         model=assistant_cfg.model or "deepseek-chat", temperature=0.0)
        gen_client = _make(cfg.simulator, config.SIMULATOR_MODEL, temperature=0.7)
        sim_client = _make(cfg.simulator, config.SIMULATOR_MODEL, temperature=0.7)
        eval_client = _make(cfg.evaluator, config.EVALUATOR_MODEL, temperature=0.0)
        audit_client = _make(cfg.evaluator, config.EVALUATOR_MODEL, temperature=0.0)
        optimizer_client = _make(cfg.optimizer, config.OPTIMIZER_MODEL, temperature=0.0)
        return assistant_client, gen_client, sim_client, eval_client, audit_client, optimizer_client

    async def _push_event(self, task: Task, event: dict) -> None:
        event.setdefault("task_id", task.task_id)
        await ws_manager.broadcast(task.task_id, event)

    async def _run_task(self, task: Task):
        task.status = "pending"
        try:
            await asyncio.wait_for(self._execute_pipeline(task), timeout=config.TASK_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            task.status = "timeout"
            await self._push_event(task, {"type": "error", "message": "任务执行超时（45分钟）", "recoverable": False})
        except Exception as e:
            task.status = "failed"
            await self._push_event(task, {"type": "error", "message": str(e), "recoverable": False})
        finally:
            task._client_allocated = False
            self._on_task_done(task)

    def _on_task_done(self, task: Task):
        task.llm_config = LLMConfigSlots()
        with self._lock:
            if task.task_id == self._running:
                self._running = None
            if self._queue:
                # 跳过队列中已删除的任务，防止队列停滞
                while self._queue:
                    next_id = self._queue.pop(0)
                    next_task = self._tasks.get(next_id)
                    if next_task is not None:
                        self._running = next_id
                        next_task.status = "pending"
                        asyncio.create_task(self._run_task(next_task))
                        break
                else:
                    self._running = None

    async def _execute_pipeline(self, task: Task):
        if task.cancel_event.is_set():
            task.status = "cancelled"
            await self._push_event(task, {"type": "error", "message": "任务已取消"})
            return

        # Phase 0: Case 解析
        task.status = "parsing"
        await self._push_event(task, {"type": "phase", "phase": "parsing", "status": "started"})
        await self._push_event(task, {"type": "log", "message": "正在解析 Case 文本...", "level": "info"})
        case = await asyncio.to_thread(parse_instruction, instruction=task.case_text, case_id=0, title=task.case_title or "外部 Case")
        await self._push_event(task, {"type": "phase", "phase": "parsing", "status": "completed"})
        await self._push_event(task, {"type": "log", "message": f"解析完成: {case.title} | {case.business_line} | {len(case.call_flow)} 步 call flow", "level": "info"})
        if task.cancel_event.is_set():
            task.status = "cancelled"; return

        # 构建客户端
        (assistant_client, gen_client, sim_client, eval_client, audit_client, optimizer_client) = self._build_clients(task)

        # Phase 1: 画像生成
        task.status = "phase_profiles"
        await self._push_event(task, {"type": "phase", "phase": "profiles", "status": "started", "model_used": task.effective_models.simulator})
        await self._push_event(task, {"type": "log", "message": f"[画像生成] 使用模型 {task.effective_models.simulator}，基础画像 {task.n_profiles} 个...", "level": "info"})
        gen_client.timeout = 80; gen_client.max_retries = 2
        runner = BatchRunner([case], assistant_client=assistant_client, simulator_client=sim_client, eval_client=eval_client)
        profiles_dict = await asyncio.to_thread(runner.generate_profiles, n_global=task.n_profiles, gen_client=gen_client, verbose=False)
        total_profiles = sum(len(v) for v in profiles_dict.values())
        task.n_profiles = total_profiles  # 同步实际画像总数（非仅 P1 数量）
        await self._push_event(task, {"type": "phase", "phase": "profiles", "status": "completed"})
        await self._push_event(task, {"type": "log", "message": f"[画像生成] 完成: 共生成 {total_profiles} 个画像", "level": "info"})
        if task.cancel_event.is_set():
            task.status = "cancelled"; return

        # Phase 2: 对话模拟
        task.status = "phase_dialogues"
        await self._push_event(task, {"type": "phase", "phase": "dialogues", "status": "started", "model_used": task.effective_models.assistant})
        await self._push_event(task, {"type": "log", "message": f"[对话模拟] 被评测模型 {task.effective_models.assistant} ←→ 模拟用户 {task.effective_models.simulator}", "level": "info"})
        conversations = await asyncio.to_thread(runner.run_all, parallel=False, profiles_dict=profiles_dict)
        for conv in conversations:
            if case and hasattr(case, 'complexity_score') and case.complexity_score:
                conv.complexity_score = case.complexity_score
            if hasattr(conv, 'user_profile') and hasattr(conv.user_profile, 'adversarial_strategy'):
                conv.adversarial_strategies = list(conv.user_profile.adversarial_strategy)
            # 覆写 profile_label 为人类可读标签（原值为 param_xxxx 哈希）
            try:
                adv_list = getattr(conv, 'adversarial_strategies', []) or []
                if adv_list:
                    _ADV_CN = {"probe": "试探", "injection": "注入", "contradiction": "矛盾", "authority": "权威", "emotion": "情绪"}
                    adv_cn = [_ADV_CN.get(a, a) for a in adv_list]
                    conv.profile_label = f"对抗型（{' + '.join(adv_cn)}）"
                else:
                    conv.profile_label = "普通用户"
            except Exception:
                pass  # 标签生成失败不影响主流程
        failed = sum(1 for c in conversations if c.status == "异常中断")
        for conv in conversations:
            if task.cancel_event.is_set():
                task.status = "cancelled"; return
            await self._push_event(task, {"type": "dialogue_card", "label": getattr(conv, 'profile_label', ''), "case_id": case.id, "status": conv.status, "turns": conv.total_turns, "assistant_model": task.effective_models.assistant, "simulator_model": task.effective_models.simulator})
        task.progress["dialogues"] = {"completed": len(conversations), "failed": failed}
        if failed > len(conversations) / 2:
            task.status = "failed"
            await self._push_event(task, {"type": "error", "message": f"超过 50% 对话失败 ({failed}/{len(conversations)})", "recoverable": False})
            return
        await self._push_event(task, {"type": "phase", "phase": "dialogues", "status": "completed"})

        # Phase 2.5: 行为审计
        try:
            await asyncio.to_thread(runner.audit_results, conversations, audit_client=audit_client, sample_ratio=0.3)
        except Exception:
            pass
        if not task.run_eval:
            task.status = "completed"
            await self._push_event(task, {"type": "completed", "result_summary": {}})
            return
        if task.cancel_event.is_set():
            task.status = "cancelled"; return

        # Phase 3: 评测
        eval_client.timeout = 90; eval_client.max_retries = 2
        task.status = "phase_eval"
        await self._push_event(task, {"type": "phase", "phase": "eval", "status": "started", "model_used": task.effective_models.evaluator})
        eval_results = await asyncio.to_thread(runner.run_phase3, conversations, eval_client=eval_client)
        if eval_results:
            scores_100 = [r.total_score_100 for r in eval_results if r.total_score_100 > 0]
            avg_score = int(sum(scores_100)/len(scores_100)) if scores_100 else 0
            if avg_score >= 90: rating = "卓越"
            elif avg_score >= 70: rating = "良好"
            elif avg_score >= 50: rating = "合格"
            elif avg_score >= 30: rating = "需改进"
            else: rating = "不合格"
            task.result_summary = {"total_score_100": avg_score, "rating_label": rating, "conversations_evaluated": len(eval_results)}
        await self._push_event(task, {"type": "phase", "phase": "eval", "status": "completed"})
        await self._push_event(task, {"type": "log", "message": f"[评测] Judge 模型 {task.effective_models.evaluator} → 完成 {len(eval_results)} 条评测", "level": "info"})
        if task.cancel_event.is_set():
            task.status = "cancelled"; return

        # 导出数据
        task.output_dir.mkdir(parents=True, exist_ok=True)

        # optimization_feed via DataExporter (read actual JSON content, not file path)
        try:
            from src.utils.data_exporter import DataExporter
            exporter = DataExporter()
            feed_path = exporter.export_optimization_feed(
                [r for r in (eval_results or []) if hasattr(r, '__dict__')],
                [c for c in (conversations or []) if hasattr(c, '__dict__')])
            # DataExporter writes to its own managed directory and returns the path;
            # read the actual JSON content from that path, then write to our output dir
            import pathlib
            fp = pathlib.Path(feed_path) if not isinstance(feed_path, pathlib.Path) else feed_path
            if fp.exists():
                feed_content = fp.read_text(encoding="utf-8")
                (task.output_dir / "optimization_feed.json").write_text(feed_content, encoding="utf-8")
            else:
                # Fallback: build feed dict ourselves
                feed = {"case_title": case.title if case else "", "per_conversation": []}
                for i, (er, conv) in enumerate(zip(eval_results or [], conversations or [])):
                    er_dict = _serialize_obj(er) if hasattr(er, '__dict__') else (er if isinstance(er, dict) else {})
                    conv_dict = _serialize_obj(conv) if hasattr(conv, '__dict__') else (conv if isinstance(conv, dict) else {})
                    feed["per_conversation"].append({"conv_index": i + 1, "evaluation": er_dict, "conversation": conv_dict})
                (task.output_dir / "optimization_feed.json").write_text(json.dumps(feed, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        # report.md via generate_narrative_report
        try:
            from src.eval.report_generator import generate_narrative_report
            report_parts = [f"# 评测报告: {case.title}\n" if case and hasattr(case, 'title') else "# 评测报告\n"]
            # 按较长列表迭代，避免 zip 静默截断
            max_n = max(len(eval_results or []), len(conversations or []))
            for i in range(max_n):
                er = eval_results[i] if i < len(eval_results or []) else None
                conv = conversations[i] if i < len(conversations or []) else None
                if hasattr(er, '__dict__') and hasattr(conv, '__dict__'):
                    report_parts.append(generate_narrative_report(er, conv, case, conv_index=i+1))
                elif er is not None or conv is not None:
                    report_parts.append(_format_single_report_json(i+1, er, conv))
            report_parts.append(_build_summary_section(eval_results))
            (task.output_dir / "report.md").write_text("\n\n".join(report_parts), encoding="utf-8")
        except Exception:
            _generate_basic_report(task.output_dir, case, conversations, eval_results)

        # 完整导出
        await asyncio.to_thread(_export_results, task.output_dir, case, profiles_dict, conversations, eval_results)

        # Phase 4: 优化
        if task.run_optimize:
            task.status = "phase_optimize"
            await self._push_event(task, {"type": "phase", "phase": "optimize", "status": "started", "model_used": task.effective_models.optimizer})
            try:
                optimizer = OptimizationEngine(llm_client=optimizer_client)
                optimizer.run(str(task.output_dir), str(task.output_dir / "optimization"))
                opt_md = task.output_dir / "optimization" / "optimization_report.md"
                if opt_md.exists(): shutil.copy2(opt_md, task.output_dir / "optimization_report.md")
                opt_json = task.output_dir / "optimization" / "optimization_actions.json"
                if opt_json.exists(): shutil.copy2(opt_json, task.output_dir / "optimization_actions.json")
            except Exception as e:
                await self._push_event(task, {"type": "log", "message": f"优化引擎异常: {e}", "level": "warn"})
            await self._push_event(task, {"type": "phase", "phase": "optimize", "status": "completed"})

        # 保存元数据
        meta = {"task_id": task.task_id, "created_at": task.created_at, "case_title": task.case_title,
                "n_profiles": task.n_profiles, "mode": "demo" if task.demo_mode else "custom",
                "effective_models": {"assistant": task.effective_models.assistant, "simulator": task.effective_models.simulator,
                                     "evaluator": task.effective_models.evaluator, "optimizer": task.effective_models.optimizer},
                "total_score_100": task.result_summary.get("total_score_100") if task.result_summary else None,
                "rating_label": task.result_summary.get("rating_label") if task.result_summary else None}
        (task.output_dir / "task_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        task.status = "completed"
        await self._push_event(task, {"type": "completed", "result_summary": task.result_summary})

    def recover_from_disk(self):
        if not config.OUTPUT_DIR.exists(): return
        for subdir in config.OUTPUT_DIR.iterdir():
            if not subdir.is_dir(): continue
            meta_file = subdir / "task_meta.json"
            if not meta_file.exists(): continue
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                task_id = meta["task_id"]
                if task_id in self._tasks: continue
                task = Task(task_id=task_id, case_text="", case_title=meta.get("case_title", ""),
                            demo_mode=meta.get("mode") == "demo", llm_config=LLMConfigSlots(),
                            n_profiles=meta.get("n_profiles", 0), run_eval=False, run_optimize=False,
                            status="completed", created_at=meta.get("created_at", ""))
                task.output_dir = subdir
                if meta.get("effective_models"):
                    task.effective_models = EffectiveModels(**meta["effective_models"])
                eval_files = sorted(subdir.glob("evaluation_*.json"))
                if eval_files and meta.get("total_score_100"):
                    task.result_summary = {"total_score_100": meta["total_score_100"]}
                elif eval_files:
                    try:
                        edata = json.loads(eval_files[0].read_text(encoding="utf-8"))
                        task.result_summary = {"total_score_100": edata.get("total_score_100", 0)}
                    except Exception:
                        pass
                self._tasks[task_id] = task
            except Exception:
                continue

    async def cleanup_loop(self):
        while True:
            await asyncio.sleep(config.CLEANUP_INTERVAL_SEC)
            now = datetime.now()
            expired_ids = []
            with self._lock:
                for tid, task in list(self._tasks.items()):
                    if task.status in ("completed", "failed", "timeout", "cancelled"):
                        try:
                            created = datetime.fromisoformat(task.created_at)
                            if now - created > timedelta(hours=config.TASK_RETENTION_HOURS):
                                expired_ids.append(tid)
                        except (ValueError, TypeError):
                            continue
                for tid in expired_ids:
                    task = self._tasks.pop(tid, None)
                    if task and task.output_dir and task.output_dir.exists():
                        shutil.rmtree(task.output_dir, ignore_errors=True)


task_manager = TaskManager()
