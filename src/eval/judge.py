"""JudgeExecutor — 单次 LLM 调用 + 清单结果解析 + JSON fallback + 并发编排"""
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from src.eval.config import CONCURRENCY, JUDGE_DIMENSIONS
from src.models.evaluation import VALID_STATUSES
from src.llm.client import LLMClient
from src.models.case import Case
from src.models.evaluation import CheckResult, Defect, DimensionChecklist


class CircuitBreaker:
    """熔断器"""

    def __init__(self, failure_threshold: int = 5, window_seconds: float = 30.0):
        self.threshold = failure_threshold
        self.window = window_seconds
        self.failures: List[float] = []
        self.is_open = False

    def record_failure(self):
        now = time.time()
        self.failures.append(now)
        self.failures = [t for t in self.failures if now - t < self.window]
        if len(self.failures) >= self.threshold:
            self.is_open = True

    def record_success(self):
        now = time.time()
        # 清理过期失败记录
        self.failures = [t for t in self.failures if now - t < self.window]
        # 窗口内无新失败 → 关闭熔断器
        if not self.failures:
            self.is_open = False


class AIMDController:
    """AIMD 并发窗口控制器"""

    def __init__(self, initial_window: int = 5, max_window: int = 10):
        self.window = initial_window
        self.max = max_window
        self.consecutive_successes = 0

    def on_success(self):
        self.consecutive_successes += 1
        if self.consecutive_successes >= 3:
            self.window = min(self.window + 1, self.max)
            self.consecutive_successes = 0

    def on_rate_limit(self):
        self.window = max(1, self.window // 2)
        self.consecutive_successes = 0

    def on_failure(self):
        self.consecutive_successes = 0


class JudgeExecutor:
    """执行单个 Judge 维度的清单核查"""

    def __init__(
        self,
        client: LLMClient,
        dimension: str,
        system_prompt: str,
        case: Case,
        timeout: int = 15,
    ):
        self.client = client
        self.dimension = dimension
        self.system_prompt = system_prompt
        self.case = case
        self.timeout = timeout

    def execute(self, user_message: str) -> Tuple[DimensionChecklist, List[Defect], str]:
        """执行一次清单核查，返回 (checklist, defects, anchor_alignment)"""
        raw = self._call_with_retry(user_message)
        check_results, defects, anchor = self._parse_response(raw)
        checklist = DimensionChecklist(dimension=self.dimension, items=check_results)
        return checklist, defects, anchor

    def _call_with_retry(self, user_message: str) -> str:
        """调用 LLM，含超时和重试"""
        try:
            return self.client.chat(self.system_prompt, user_message)
        except Exception as first_error:
            # 重试 1 次
            try:
                return self.client.chat(self.system_prompt, user_message)
            except Exception as second_error:
                raise RuntimeError(
                    f"Judge LLM call failed after retry: first={first_error}, second={second_error}"
                ) from second_error

    def _parse_response(self, raw: str) -> Tuple[List[CheckResult], List[Defect], str]:
        """解析 LLM 输出的 JSON"""
        # 尝试提取 JSON
        json_str = self._extract_json(raw)

        try:
            data = json.loads(json_str)
            check_results = self._parse_checklist_items(data.get("checklist_results", []))
            defects = self._parse_defects(data.get("additional_defects", []))
            anchor = data.get("anchor_alignment", "合格")
            return check_results, defects, anchor
        except (json.JSONDecodeError, KeyError):
            # Fallback: 尝试更简单 schema
            return self._fallback_parse(raw)

    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON"""
        # 尝试匹配 ```json ... ``` 块
        m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if m:
            return m.group(1).strip()

        # 尝试匹配 { ... }
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            return m.group(0).strip()

        return text.strip()

    def _parse_checklist_items(self, raw_items: List[Dict]) -> List[CheckResult]:
        """解析清单项（六级粒度 + CoT reasoning）"""
        results = []
        for item in raw_items:
            status = item.get("status", "NOT_APPLICABLE")
            if status not in VALID_STATUSES:
                status = "PARTIAL"  # 未知状态降级为 PARTIAL

            results.append(CheckResult(
                item_id=item.get("item_id", ""),
                description=item.get("description", ""),
                source=item.get("source", "llm_supplement"),
                status=status,
                evidence=item.get("evidence", ""),
                signal_consistency=item.get("signal_consistency", "无对应信号"),
                reasoning=item.get("reasoning", ""),
            ))
        return results

    def _parse_defects(self, raw_defects: List[Dict]) -> List[Defect]:
        """解析补充缺陷"""
        defects = []
        for d in raw_defects:
            sev = d.get("severity", "一般")
            if sev not in ("关键", "一般", "轻微"):
                sev = "一般"
            defects.append(Defect(
                description=d.get("description", ""),
                severity=sev,
                turn=d.get("turn", 0),
                attribution=d.get("attribution", "Model"),
            ))
        return defects

    def _fallback_parse(self, raw: str) -> Tuple[List[CheckResult], List[Defect], str]:
        """JSON 解析失败时的降级处理"""
        return [], [], "无法评估"


class ConcurrentJudgeRunner:
    """并发编排：信号量 + AIMD + 熔断"""

    def __init__(self, client: LLMClient):
        self.client = client
        self.breaker = CircuitBreaker(
            failure_threshold=CONCURRENCY["circuit_breaker_failures"],
            window_seconds=CONCURRENCY["circuit_breaker_window_seconds"],
        )
        self.aimd = AIMDController(
            initial_window=CONCURRENCY["aimd_window_initial"],
        )
        self.parse_failures: Dict[str, bool] = {}

    def run_all(
        self,
        executors: Dict[str, JudgeExecutor],
        user_message: str,
    ) -> Dict[str, Tuple[DimensionChecklist, List[Defect], str]]:
        """并发执行所有 Judge 维度"""
        results = {}

        if self.breaker.is_open:
            # 熔断开启，逐个串行执行
            for dim, executor in executors.items():
                try:
                    results[dim] = executor.execute(user_message)
                    self.breaker.record_success()
                except Exception:
                    self.breaker.record_failure()
                    results[dim] = (
                        DimensionChecklist(dimension=dim),
                        [],
                        "无法评估",
                    )
            return results

        # 分批并发（每批不超过 aimd.window）
        dims = list(executors.keys())
        batch_size = self.aimd.window

        for i in range(0, len(dims), batch_size):
            batch = dims[i:i + batch_size]
            batch_results = self._run_batch(
                {d: executors[d] for d in batch},
                user_message,
            )
            results.update(batch_results)

        return results

    def _run_batch(
        self,
        batch_executors: Dict[str, JudgeExecutor],
        user_message: str,
    ) -> Dict[str, Tuple[DimensionChecklist, List[Defect], str]]:
        """执行一批 Judge"""
        results = {}
        with ThreadPoolExecutor(max_workers=len(batch_executors)) as pool:
            futures = {}
            for dim, executor in batch_executors.items():
                future = pool.submit(executor.execute, user_message)
                futures[future] = dim

            from concurrent.futures import TimeoutError as FuturesTimeoutError
            try:
                for future in as_completed(futures, timeout=30):
                    dim = futures[future]
                    try:
                        result = future.result(timeout=CONCURRENCY["judge_timeout_seconds"])
                        results[dim] = result
                        self.aimd.on_success()
                        self.breaker.record_success()
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "rate" in err_str.lower():
                            self.aimd.on_rate_limit()
                        else:
                            self.aimd.on_failure()
                        self.breaker.record_failure()
                        self.parse_failures[dim] = True
                        results[dim] = (
                            DimensionChecklist(dimension=dim),
                            [],
                            "无法评估",
                        )
            except FuturesTimeoutError:
                # 整批超时：取消未完成的 future，标记剩余维度为失败
                for future, dim in futures.items():
                    if not future.done():
                        future.cancel()
                for dim in batch_executors:
                    if dim not in results:
                        self.breaker.record_failure()
                        self.aimd.on_failure()
                        self.parse_failures[dim] = True
                        results[dim] = (
                            DimensionChecklist(dimension=dim),
                            [],
                            "无法评估",
                        )

        return results
