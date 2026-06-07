"""模型注册、切换与管理"""
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from dotenv import load_dotenv

from src.llm.client import LLMClient
from src.config import API_KEY, BASE_URL, MODEL as DEFAULT_MODEL

# 确保加载 .env（config.py 已加载，此处保底）
load_dotenv(Path(__file__).parent.parent.parent / ".env")


@dataclass
class ModelConfig:
    """单个模型配置"""
    name: str
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_retries: int = 3


@dataclass
class ModelManager:
    """模型注册与管理

    从 .env 读取多套模型配置，按逻辑名创建 LLMClient 实例。
    缺失的配置项自动回退到默认值。
    """
    _configs: Dict[str, ModelConfig] = field(default_factory=dict)

    def __post_init__(self):
        if not self._configs:
            self._auto_load()

    def _auto_load(self) -> None:
        """从环境变量自动加载模型配置

        命名约定:
          <NAME>_API_KEY  /  <NAME>_BASE_URL  /  <NAME>_MODEL
        例如:
          SIMULATOR_API_KEY=sk-xxx
          SIMULATOR_BASE_URL=https://api.deepseek.com
          SIMULATOR_MODEL=deepseek-chat
        """
        # 1) 默认模型（从已有的环境变量）
        default = ModelConfig(
            name="default",
            api_key=API_KEY,
            base_url=BASE_URL,
            model=DEFAULT_MODEL,
            temperature=0.0,
        )
        self._configs["default"] = default

        # 2) 扫描 COMMON_ROLE 后缀，发现专用模型
        suffixes = {
            "SIMULATOR": "simulator",
            "GENERATOR": "generator",
            "AUDITOR": "auditor",
        }

        for env_suffix, logical_name in suffixes.items():
            key = os.getenv(f"{env_suffix}_API_KEY", "")
            url = os.getenv(f"{env_suffix}_BASE_URL", "")
            model = os.getenv(f"{env_suffix}_MODEL", "")

            if key or url or model:
                self._configs[logical_name] = ModelConfig(
                    name=logical_name,
                    api_key=key or API_KEY,
                    base_url=url or BASE_URL,
                    model=model or DEFAULT_MODEL,
                    temperature=0.0,
                )
            else:
                # 未配置时回退到默认
                self._configs[logical_name] = ModelConfig(
                    name=logical_name,
                    api_key=API_KEY,
                    base_url=BASE_URL,
                    model=DEFAULT_MODEL,
                    temperature=0.0,
                )

    def register(self, config: ModelConfig) -> None:
        """手动注册模型配置"""
        self._configs[config.name] = config

    def get_config(self, name: str = "default") -> ModelConfig:
        """获取指定逻辑名的模型配置"""
        if name not in self._configs:
            raise KeyError(f"未知的模型配置: {name}，可用: {list(self._configs)}")
        return self._configs[name]

    def create_client(
        self, name: str = "default", **overrides
    ) -> LLMClient:
        """根据逻辑名创建 LLMClient 实例

        overrides 可覆盖 temperature 等参数:
            mgr.create_client("simulator", temperature=0.7)
        """
        cfg = self.get_config(name)
        return LLMClient(
            api_key=overrides.pop("api_key", cfg.api_key),
            base_url=overrides.pop("base_url", cfg.base_url),
            model=overrides.pop("model", cfg.model),
            temperature=overrides.pop("temperature", cfg.temperature),
            max_retries=overrides.pop("max_retries", cfg.max_retries),
            **overrides,
        )


# ---- 全局单例 ----

_manager: Optional[ModelManager] = None
_manager_lock = threading.Lock()


def get_manager() -> ModelManager:
    """获取全局单例 ModelManager（线程安全）"""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ModelManager()
    return _manager


def get_simulator_client(temperature: float = 0.7) -> LLMClient:
    """快捷获取模拟器用 LLMClient"""
    return get_manager().create_client("simulator", temperature=temperature)


def get_generator_client(temperature: float = 0.7) -> LLMClient:
    """快捷获取画像生成用 LLMClient"""
    return get_manager().create_client("generator", temperature=temperature)


def get_auditor_client(temperature: float = 0.0) -> LLMClient:
    """快捷获取审计用 LLMClient"""
    return get_manager().create_client("auditor", temperature=temperature)


def get_default_client(temperature: float = 0.0) -> LLMClient:
    """快捷获取默认 LLMClient"""
    return get_manager().create_client("default", temperature=temperature)
