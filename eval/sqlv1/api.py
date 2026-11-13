"""Gọi model qua eval/llm.py (guard bật, ledger riêng outputs/synth_sql_v1/requests.sqlite, trần SYNTH_HARD_CAP=12000), cache theo nội dung."""
from __future__ import annotations

import collections
import datetime as dt
import hashlib
import json
import os
import sqlite3
import threading
from pathlib import Path

import llm

LEDGER = Path(__file__).resolve().parents[2] / "outputs" / "synth_sql_v1" / "requests.sqlite"
CAP = 12000
LIMITS = {"gptsol": 6, "haiku": 6, "deepseek": 8, "gemini": 10, "qwen": 6, "gpt": 2}
_lock = threading.Lock()
_configured = [False]
FAIL = collections.Counter()


def setup():
    if not _configured[0]:
        os.environ["SYNTH_HARD_CAP"] = str(CAP)
        llm.configure_run(LEDGER, limit=CAP, provider_limits=LIMITS)
        _configured[0] = True


def used() -> int:
    with sqlite3.connect(LEDGER) as db:
        return db.execute("SELECT COUNT(*) FROM requests").fetchone()[0]


def used_by() -> dict:
    with sqlite3.connect(LEDGER) as db:
        return {f"{p}:{t.split(':')[1] if ':' in t else t}": n for p, t, n in db.execute("SELECT provider, tag, COUNT(*) FROM requests GROUP BY 1, 2").fetchall()}


def digest(x) -> str:
    return hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def call(work: Path, role: str, shop: str, provider: str, messages, *, think="off", max_tokens=12000, temperature=0, timeout=300) -> tuple[dict, bool]:
    """(trả lời, lấy từ cache?). Lỗi mạng/API trả finish='api_error' (không cache)."""
    setup()
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    settings = {"provider": provider, "model": llm.config(provider)["model"], "think": think, "max_tokens": max_tokens, "temperature": temperature}
    key = digest({"messages": messages, "settings": settings})
    path = work / "api" / role / shop / f"{key[:24]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"], True
    try:
        r = llm.chat(provider, messages, think=think, max_tokens=max_tokens, temperature=temperature, retries=1, timeout=timeout,
                     tag=f"sqlv1:{role}:{shop}", check_leak=True)
    except (llm.LeakError, llm.BudgetExceeded):
        raise
    except Exception as e:  # noqa: BLE001
        with _lock:
            FAIL[provider] += 1
        return {"content": "", "finish": "api_error", "error": str(e)[:300], "usage": {}}, False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"role": role, "shop": shop, "messages": messages, "settings": settings, "response": r,
                               "ts": dt.datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return r, False
