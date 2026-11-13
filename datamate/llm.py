"""Client LLM qua API kiểu OpenAI (vLLM serve chạy cục bộ trong Docker, không ra mạng ngoài)."""
from __future__ import annotations

import json
import re
import time


class LLM:
    def __init__(self, cfg: dict):
        from openai import OpenAI

        self.cfg = cfg
        self.client = OpenAI(base_url=cfg.get("base_url", "http://127.0.0.1:8000/v1"),
                             api_key=cfg.get("api_key", "EMPTY"), timeout=cfg.get("request_timeout_s", 600))
        self.model = cfg["name"]

    def chat(self, messages: list[dict], temperature: float, seed: int, schema: dict | None = None, thinking: bool | None = None) -> str:
        extra = {}
        think = self.cfg.get("enable_thinking") if thinking is None else thinking
        if "enable_thinking" in self.cfg or thinking is not None:
            extra["chat_template_kwargs"] = {"enable_thinking": bool(think)}
        if schema is not None:
            extra["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        max_tokens = self.cfg.get("max_tokens_thinking", 8192) if think else self.cfg.get("max_tokens", 2048)
        if self.cfg.get("top_k") is not None:
            extra["top_k"] = self.cfg["top_k"]
        last = None
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature, seed=seed,
                    top_p=self.cfg.get("top_p", 1.0) if temperature > 0 else 1.0,
                    max_tokens=max_tokens, extra_body=extra or None)
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 — server bận / quá tải thì thử lại
                last = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"LLM lỗi: {last}")


    def chat_json(self, messages: list[dict], schema: dict, seed: int = 42):
        """Đầu ra JSON đúng schema (vLLM ràng buộc lúc sinh); bước này luôn tắt thinking. Không đọc được thì None."""
        return parse_json(self.chat(messages, 0.0, seed, schema=schema, thinking=False))


class DummyLLM:
    """Không có model: luôn trả abstain. Dùng để chạy thử toàn bộ đường ống."""

    def chat(self, messages, temperature, seed, thinking=False):
        return "```python\nANSWER = dm.abstain('dummy backend')\n```"

    def chat_json(self, messages, schema, seed=42):
        return None


def parse_json(text: str):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(text: str) -> str | None:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blocks = _BLOCK.findall(text)
    if blocks:
        return max(blocks, key=lambda b: ("ANSWER" in b, len(b)))
    # không có khối ``` : chỉ nhận cả câu trả lời khi nó là code Python hợp lệ, không thì bắt viết lại
    if "ANSWER" in text:
        try:
            compile(text, "<code>", "exec")
            return text
        except SyntaxError:
            return None
    return None
