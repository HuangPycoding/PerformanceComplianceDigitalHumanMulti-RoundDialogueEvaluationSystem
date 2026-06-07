"""WebSocket 连接管理 — 注册、广播、心跳"""
import asyncio
import json
from typing import Dict, List, Set

from fastapi import WebSocket


class WSManager:
    """按 task_id 分组管理 WebSocket 连接"""

    def __init__(self):
        # task_id → set of websocket connections
        self._connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, task_id: str, ws: WebSocket) -> None:
        """接受 WebSocket 连接并注册"""
        await ws.accept()
        if task_id not in self._connections:
            self._connections[task_id] = set()
        self._connections[task_id].add(ws)

    def disconnect(self, task_id: str, ws: WebSocket) -> None:
        """移除连接"""
        if task_id in self._connections:
            self._connections[task_id].discard(ws)
            if not self._connections[task_id]:
                del self._connections[task_id]

    def has_connections(self, task_id: str) -> bool:
        """检查是否有活跃连接"""
        return task_id in self._connections and len(self._connections[task_id]) > 0

    async def broadcast(self, task_id: str, event: dict) -> None:
        """向 task_id 的所有连接广播事件"""
        if task_id not in self._connections:
            return
        dead: List[WebSocket] = []
        payload = json.dumps(event, ensure_ascii=False, default=str)
        for ws in self._connections[task_id]:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(task_id, ws)

    async def heartbeat_loop(self, task_id: str, ws: WebSocket, interval: int = 30) -> None:
        """每 interval 秒发送 WebSocket ping 帧（Starlette 自动处理 pong）"""
        try:
            while task_id in self._connections and ws in self._connections.get(task_id, set()):
                await asyncio.sleep(interval)
                await ws.send_json({"type": "heartbeat"})
        except Exception:
            pass
        finally:
            self.disconnect(task_id, ws)


# 全局单例
ws_manager = WSManager()
