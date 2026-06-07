"""Pydantic 请求/响应模型"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ── LLM 配置 ──

class LLMSlotConfig(BaseModel):
    """单个槽位配置"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class LLMConfigSlots(BaseModel):
    """四槽位 LLM 配置"""
    assistant: Optional[LLMSlotConfig] = None
    simulator: Optional[LLMSlotConfig] = None
    evaluator: Optional[LLMSlotConfig] = None
    optimizer: Optional[LLMSlotConfig] = None


class EffectiveModels(BaseModel):
    """实际使用的模型名称（返回给前端展示）"""
    assistant: str = ""
    simulator: str = ""
    evaluator: str = ""
    optimizer: str = ""


# ── Case 相关 ──

class ParseCaseRequest(BaseModel):
    case_text: str


class ParseCaseResponse(BaseModel):
    title: str = ""
    business_line: str = ""
    role: str = ""
    task: str = ""
    opening_line: str = ""
    call_flow_steps: List[Dict[str, Any]] = Field(default_factory=list)
    constraints: List[Dict[str, Any]] = Field(default_factory=list)
    knowledge_points: List[Dict[str, Any]] = Field(default_factory=list)


class PresetCaseSummary(BaseModel):
    id: int
    title: str
    business_line: str
    complexity_score: float


# ── 任务相关 ──

class CreateTaskRequest(BaseModel):
    case_text: str = ""
    case_title: Optional[str] = None
    demo_mode: bool = True  # True=展示预跑结果(不调LLM), False=真实流水线
    llm_config: Optional[LLMConfigSlots] = None
    n_profiles: int = Field(default=2, ge=1, le=20)
    run_eval: bool = True
    run_optimize: bool = False


class TaskCreatedResponse(BaseModel):
    task_id: str
    status: str
    mode: str  # "demo" or "custom"
    effective_models: EffectiveModels = Field(default_factory=EffectiveModels)


class ResultSummary(BaseModel):
    total_score_100: Optional[int] = None
    rating_label: Optional[str] = None
    dimension_scores: Optional[Dict[str, float]] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    created_at: str = ""
    case_title: str = ""
    n_profiles: int = 0
    mode: str = ""
    progress: Dict[str, Any] = Field(default_factory=dict)
    result_summary: Optional[ResultSummary] = None
    effective_models: EffectiveModels = Field(default_factory=EffectiveModels)


class TaskListItem(BaseModel):
    task_id: str
    status: str
    created_at: str = ""
    case_title: str = ""
    n_profiles: int = 0
    total_score_100: Optional[int] = None
    rating_label: Optional[str] = None


class TaskDetailResponse(BaseModel):
    task_id: str
    status: str
    created_at: str = ""
    case_title: str = ""
    mode: str = ""
    effective_models: EffectiveModels = Field(default_factory=EffectiveModels)
    eval_result: Optional[Dict[str, Any]] = None
    all_eval_results: Optional[List[Dict[str, Any]]] = None
    optimization_suggestions: Optional[List[Dict[str, Any]]] = None
    conversation_summary: Optional[Dict[str, Any]] = None
    all_conversations: Optional[List[Dict[str, Any]]] = None
    report_md: str = ""
    optimization_report_md: str = ""


class ConversationTurn(BaseModel):
    turn_index: int
    speaker: str
    content: str
    parsed_tags: Optional[Dict[str, Any]] = None
    model_used: Optional[str] = None


class TaskDeleteResponse(BaseModel):
    ok: bool = True


# ── 工具 ──

class TestConnectionRequest(BaseModel):
    slot: str = "assistant"  # assistant | simulator | evaluator | optimizer
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None


class TestConnectionResponse(BaseModel):
    ok: bool
    message: str
    latency_ms: int = 0


# ── 演示模式 ──

class DemoConfigResponse(BaseModel):
    """演示模式配置（预跑结果 + 模型信息）"""
    available: bool = True
    demo_case_ids: List[int] = Field(default_factory=list)
    demo_tasks: List[TaskListItem] = Field(default_factory=list)
    effective_models: EffectiveModels = Field(default_factory=EffectiveModels)


# ── WebSocket 事件 ──

class ProgressEvent(BaseModel):
    type: str  # queued | phase | progress | dialogue_card | log | completed | error
    phase: Optional[str] = None
    status: Optional[str] = None
    completed: Optional[int] = None
    total: Optional[int] = None
    label: Optional[str] = None
    case_id: Optional[int] = None
    turns: Optional[int] = None
    model_used: Optional[str] = None
    assistant_model: Optional[str] = None
    simulator_model: Optional[str] = None
    message: Optional[str] = None
    level: Optional[str] = None
    recoverable: Optional[bool] = None
    position: Optional[int] = None
    estimated_wait_sec: Optional[int] = None
    result_summary: Optional[ResultSummary] = None
