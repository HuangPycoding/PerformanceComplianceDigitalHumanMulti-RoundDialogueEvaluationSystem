"""OpenAI 兼容接口封装，内置重试和结构化输出"""
import json
import re
import time
from typing import Optional

from openai import OpenAI

from src.config import API_KEY, BASE_URL, MODEL


class LLMClient:
    """OpenAI兼容接口封装"""

    def __init__(self, model: str = MODEL, temperature: float = 0.0,
                 max_retries: int = 3, api_key: str = "", base_url: str = "",
                 timeout: int = 120):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = OpenAI(
            api_key=api_key or API_KEY,
            base_url=base_url or BASE_URL,
            timeout=timeout,
        )

    def chat(self, system_prompt: str, user_message: str) -> str:
        """单次 LLM 调用，返回文本。max_retries=3 表示最多 3 次额外重试（共 4 次尝试）。"""
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    timeout=self.timeout,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                if attempt == self.max_retries:
                    raise
                time.sleep(1 * (attempt + 1))
        return ""

    def chat_structured(self, system_prompt: str, user_message: str,
                        output_schema: dict) -> dict | None:
        """返回结构化 JSON。网络重试由 chat() 负责，此处仅重试 JSON 解析错误。

        Returns:
            dict: 成功解析的结构化 JSON
            None: JSON 解析重试 3 次均失败（LLM 返回了文本但格式不正确）
        Raises:
            Exception: 网络调用失败（chat() 重试耗尽后重新抛出）
        """
        schema_text = json.dumps(output_schema, ensure_ascii=False, indent=2)
        full_prompt = f"""{user_message}

请严格按照以下JSON格式输出，不要输出其他内容：
{schema_text}"""

        last_exception = None
        for json_attempt in range(3):
            try:
                text = self.chat(system_prompt, full_prompt)
                json_text = self._extract_json(text)
                return json.loads(json_text)
            except json.JSONDecodeError as e:
                last_exception = e
                if json_attempt < 2:
                    time.sleep(1 * (json_attempt + 1))
            # 网络异常直接向上抛出，不吞掉
        print(f"[警告] chat_structured JSON解析重试3次均失败: {last_exception}")
        return None

    @staticmethod
    def _extract_json(text: str) -> str:
        """从文本中提取 JSON（支持 ```json...``` 代码块围栏）"""
        text = text.strip()
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text


# 全局默认实例（同步版本）
default_client = LLMClient()
