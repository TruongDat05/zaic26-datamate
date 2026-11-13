"""Gọi 3 model sinh dữ liệu qua API kiểu OpenAI (chat/completions). Đọc khoá trong .env, không bao giờ in khoá.

    python eval/llm.py check            # thử từng model một câu ngắn (bật/tắt suy nghĩ), in bảng trạng thái
    python eval/llm.py curl             # in lệnh curl tương đương (dùng biến môi trường, không lộ khoá)

Nhà cung cấp (tên ngắn → biến trong .env):
    deepseek  API_URL, API_KEY        deepseek-v4-flash  gateway LiteLLM nội bộ FCI; đường dẫn /chat/completions, không có /v1
    qwen      MK_API_URL, MK_API_KEY  Qwen3.8-27B        FPT Cloud marketplace; URL đã gồm /chat/completions
    gemini    GEMINI_URL, GEMINI_KEY  ag/gemini-3.8-flash-high  proxy bên thứ ba (api.215tohieu.io.vn); reasoning high cố định
    gpt       GPT_URL, GPT_KEY        gpt-6-luna         proxy bên thứ ba (chat.trollllm.xyz) — chỉ gửi nội dung tự soạn/tự sinh

Cả bốn đều là model có suy nghĩ (reasoning). `think="off"` tắt hẳn với deepseek/qwen; gpt chỉ giảm được (reasoning_effort=low);
gemini luôn ở mức high (key chỉ mở model *-high).
Mọi lần gọi ghi vào outputs/llm_calls/<ngày>.jsonl (đã gitignore): prompt, trả lời, token, thời gian, chi phí ước tính.

Chặn rò dữ liệu: `guard()` từ chối prompt có đoạn chép từ rulebook BTC, mã shop BTC (biz_1xx), hoặc tên ngành/khái niệm
chỉ có ngoài train (synth_spec.LOCAL_BLOCKLIST). Đây là lưới an toàn cuối, không thay cho việc chỉ đưa nội dung tự sinh vào prompt.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import fcntl
import sqlite3
from contextlib import contextmanager
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval")]

PROVIDERS = {
    "deepseek": {"url": "API_URL", "key": "API_KEY", "model": "deepseek-v4-flash", "model_env": "API_MODEL",
                 "think": {"off": {"chat_template_kwargs": {"thinking": False, "enable_thinking": False}}, "low": {}, "high": {}},
                 "price_per_m": (0.14, 0.28), "noi_bo": True,
                 "ghi_chu": "rẻ nhất (~11 lần rẻ hơn Qwen3.8 ở token ra), ngữ cảnh 500K–1M; mặc định có suy nghĩ"},
    "qwen": {"url": "MK_API_URL", "key": "MK_API_KEY", "model": "Qwen3.8-27B", "model_env": "MK_MODEL",
             "think": {"off": {"chat_template_kwargs": {"enable_thinking": False}}, "low": {}, "high": {}},
             "price_per_m": (0.45, 3.20), "noi_bo": True,
             "ghi_chu": "cùng họ Qwen với model học (Qwen3-14B); token ra đắt, suy nghĩ dài"},
    "gemini": {"url": "GEMINI_URL", "key": "GEMINI_KEY", "model": "ag/gemini-3.8-flash-high", "model_env": "GEMINI_MODEL",
               "think": {"off": {}, "low": {}, "high": {}}, "think_co_dinh": "high",
               "price_per_m": None, "noi_bo": False,
               "ghi_chu": "proxy bên thứ ba; reasoning high cố định theo tên model; proxy chèn ~2,9K token prompt hệ thống ẩn"},
    "gpt": {"url": "GPT_URL", "key": "GPT_KEY", "model": "gpt-6-luna", "model_env": "GPT_MODEL",
            "think": {"off": {"reasoning_effort": "low"}, "low": {"reasoning_effort": "low"}, "high": {"reasoning_effort": "high"}},
            "price_per_m": None, "noi_bo": False,
            "ghi_chu": "proxy bên thứ ba, giá chưa rõ; không tắt hẳn được suy nghĩ"},
    # thêm 11/11/2026 cho syn_sales_sql_v1: cùng proxy trollllm (GPT_URL/GPT_KEY), TÊN MODEL CỐ ĐỊNH (không đọc từ env)
    "gptsol": {"url": "GPT_URL", "key": "GPT_KEY", "model": "gpt-6.1-sol", "model_env": None,
               "think": {"off": {"reasoning_effort": "low"}, "low": {"reasoning_effort": "low"}, "high": {"reasoning_effort": "high"}},
               "price_per_m": None, "noi_bo": False, "ghi_chu": "proxy trollllm; họ OpenAI; không tắt hẳn được suy nghĩ"},
    "haiku": {"url": "GPT_URL", "key": "GPT_KEY", "model": "claude-haiku-5-5", "model_env": None,
              "think": {"off": {}, "low": {}, "high": {}},
              "price_per_m": None, "noi_bo": False, "ghi_chu": "proxy trollllm; họ Anthropic"},
}
FAMILY = {"gpt": "openai", "gptsol": "openai", "haiku": "anthropic", "gemini": "google", "deepseek": "deepseek", "qwen": "qwen"}
LOG_DIR = ROOT / "outputs" / "llm_calls"
DEFAULT_LEDGER = ROOT / "outputs" / "synth_request_budget.sqlite"
HARD_CAP = 4000  # trần HTTP request cộng dồn trên một ledger (nâng từ 800 ngày 10/11/2026; ledger và lịch sử giữ nguyên)


def hard_cap() -> int:
    """Trần cho ledger mới: SYNTH_HARD_CAP (vd. 12000 cho outputs/synth_sql_v1/requests.sqlite); không đặt thì giữ HARD_CAP = 4000."""
    return int(os.environ.get("SYNTH_HARD_CAP") or HARD_CAP)


class LeakError(ValueError):
    """Prompt có nội dung không được gửi ra API."""



class BudgetExceeded(RuntimeError):
    """The configured run has exhausted its HTTP request budget."""


_RUN = None


def configure_run(ledger_path, limit=HARD_CAP, provider_limits=None, cap=None):
    """Enable durable, process-safe per-attempt accounting and mandatory prompt guarding.

    All processes participating in a run must configure the same ledger path and limits.
    Passing None disables run accounting for ordinary interactive use.
    cap: trần của ledger này (mặc định hard_cap(): env SYNTH_HARD_CAP hoặc 4000). Trần được ghi vào ledger lần đầu
    (bảng ledger_meta); cấu hình sau với trần khác bị từ chối, nên không ledger nào bị nâng trần ngầm.
    """
    global _RUN
    if ledger_path is None:
        _RUN = None
        return
    cap = hard_cap() if cap is None else int(cap)
    if not 0 <= limit <= cap or any(v < 1 for v in (provider_limits or {}).values()):
        raise ValueError("invalid request or provider limit")
    path = Path(ledger_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=60) as db:
        db.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, ts TEXT, tag TEXT, provider TEXT, prompt_sha TEXT, requested_model TEXT, response_model TEXT, think_params TEXT, attempt INTEGER, status TEXT, error TEXT, usage TEXT)")
        has_meta = db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'ledger_meta'").fetchone()[0]
        row = db.execute("SELECT value FROM ledger_meta WHERE key = 'cap'").fetchone() if has_meta else None
        if row is not None and int(row[0]) != cap:
            raise ValueError(f"ledger {path} có trần {row[0]}, cấu hình xin {cap}")
        if row is None and cap != HARD_CAP:  # trần khác mặc định: chỉ cho ledger mới, ghi lại để mọi tiến trình dùng cùng trần
            if db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] > 0:
                raise ValueError(f"ledger {path} đã có request mà chưa ghi trần: không đổi trần cho ledger cũ")
            db.execute("CREATE TABLE IF NOT EXISTS ledger_meta (key TEXT PRIMARY KEY, value TEXT)")
            db.execute("INSERT INTO ledger_meta VALUES ('cap', ?)", (str(cap),))
    _RUN = (path, limit, dict(provider_limits or {}))


@contextmanager
def _provider_slot(provider):
    run = _RUN
    if not run or provider not in run[2]:
        yield
        return
    # Each independently opened descriptor competes across both processes and threads.
    files = [open(str(run[0]) + f".{provider}.{i}.lock", "a") for i in range(run[2][provider])]
    selected = None
    try:
        while selected is None:
            for f in files:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    selected = f
                    break
                except BlockingIOError:
                    pass
            if selected is None:
                time.sleep(0.05)
        yield
    finally:
        if selected:
            fcntl.flock(selected, fcntl.LOCK_UN)
        for f in files:
            f.close()


def _reserve(provider, tag, prompt_sha, body, attempt, think):
    if not _RUN:
        return None
    path, limit, _ = _RUN
    with sqlite3.connect(path, timeout=60, isolation_level=None) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] >= limit:
            db.rollback()
            raise BudgetExceeded(f"HTTP request budget exhausted ({limit})")
        params = {"requested": think, "fixed": PROVIDERS[provider].get("think_co_dinh"), "parameters": {k: v for k, v in body.items() if k not in ("messages", "model", "max_tokens", "temperature")}}
        cur = db.execute("INSERT INTO requests (ts, tag, provider, prompt_sha, requested_model, think_params, attempt, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         (_dt.datetime.now(_dt.timezone.utc).isoformat(), tag, provider, prompt_sha, body['model'], json.dumps(params, ensure_ascii=False), attempt, 'started'))
        db.commit()
        return cur.lastrowid


def _finish_request(row, status, response=None, error=None):
    if row is None:
        return
    with sqlite3.connect(_RUN[0], timeout=60) as db:
        db.execute("UPDATE requests SET status=?, response_model=?, usage=?, error=? WHERE id=?",
                   (status, (response or {}).get('model'), json.dumps((response or {}).get('usage') or {}), error, row))

def load_env(path: Path = ROOT / ".env") -> dict:
    """Đọc KEY=VALUE (bỏ dòng trống, #, dấu nháy). Biến đã có trong môi trường được ưu tiên."""
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip().removeprefix("export ").strip()] = v.strip().strip('"').strip("'")
    env.update({k: v for k, v in os.environ.items() if k in {p[x] for p in PROVIDERS.values() for x in ("url", "key", "model_env")}})
    return env


def endpoint(url: str) -> str:
    url = url.rstrip("/")
    return url if url.endswith("/chat/completions") else url + "/chat/completions"


def config(provider: str, env: dict | None = None) -> dict:
    p = PROVIDERS[provider]
    env = load_env() if env is None else env
    if not env.get(p["url"]) or not env.get(p["key"]):
        raise RuntimeError(f"thiếu {p['url']} hoặc {p['key']} trong .env")
    return {"endpoint": endpoint(env[p["url"]]), "key": env[p["key"]], "model": env.get(p["model_env"]) or p["model"]}


# ---------------------------------------------------------------- chặn rò dữ liệu
_GUARD: dict = {}


def _norm(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def _guard_data():
    if not _GUARD:
        words = set()
        for f in (ROOT / "data/train/rulebook.md", ROOT / "data/synth_source/rules/rulebook_train.md"):
            if f.exists():
                w = _norm(f.read_text(encoding="utf-8"))
                words |= {" ".join(w[i:i + 10]) for i in range(len(w) - 9)}
        _GUARD.update(rule10=words)
    return _GUARD


def guard(text: str) -> None:
    """Từ chối prompt chép ≥ 10 từ liên tiếp của rulebook BTC, có mã shop BTC hoặc từ trong LOCAL_BLOCKLIST
    (tên ngành chỉ tính khi đứng sau từ chỉ ngành — xem synth_checks.blocklist_hits)."""
    from synth_checks import blocklist_hits
    g = _guard_data()
    low = text.lower()
    hits = blocklist_hits(low)
    if hits:
        raise LeakError(f"prompt có từ trong LOCAL_BLOCKLIST: {hits[:3]}")
    if re.search(r"\bbiz_1\d\d\b", low):
        raise LeakError("prompt có mã shop BTC (biz_1xx)")
    w = _norm(text)
    for i in range(len(w) - 9):
        if " ".join(w[i:i + 10]) in g["rule10"]:
            raise LeakError("prompt chép đoạn ≥ 10 từ từ rulebook BTC: " + " ".join(w[i:i + 10]))


# ---------------------------------------------------------------- gọi model
def cost_usd(provider: str, usage: dict | None) -> float | None:
    price = PROVIDERS[provider]["price_per_m"]
    if not price or not usage:
        return None
    return round((usage.get("prompt_tokens", 0) * price[0] + usage.get("completion_tokens", 0) * price[1]) / 1e6, 6)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # each reserved HTTP attempt must send at most one request


def _post(cfg: dict, body: dict, timeout: float, stream: bool) -> dict:
    """POST một lần. stream=True đọc SSE rồi ghép lại như trả lời thường — cần cho câu trả lời dài: FPT Cloud (Cloudflare)
    cắt request không stream sau ~100 giây (HTTP 524). timeout áp cho từng lần đọc socket, không cho cả câu trả lời."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}} if stream else {**body, "stream": False}
    req = urllib.request.Request(cfg["endpoint"], data=json.dumps(body).encode("utf-8"), method="POST", headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {cfg['key']}", "x-api-key": cfg["key"]})
    with urllib.request.build_opener(_NoRedirect()).open(req, timeout=timeout) as r:
        # có proxy trả SSE cả khi không xin stream: đọc theo content-type
        if "event-stream" not in (r.headers.get("content-type") or ""):
            return json.loads(r.read().decode("utf-8"))
        content, reasoning, finish, usage, model = [], [], None, {}, None
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            model = ev.get("model") or model
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                d = ch.get("delta") or {}
                if d.get("content"):
                    content.append(d["content"])
                if d.get("reasoning_content") or d.get("reasoning"):
                    reasoning.append(d.get("reasoning_content") or d.get("reasoning"))
                finish = ch.get("finish_reason") or finish
    return {"model": model, "usage": usage,
            "choices": [{"message": {"content": "".join(content), "reasoning_content": "".join(reasoning)}, "finish_reason": finish}]}


def chat(provider: str, messages: list[dict] | str, *, think: str = "off", max_tokens: int = 4096, temperature: float | None = None,
         timeout: float = 600, retries: int = 3, tag: str = "", log: bool = True, check_leak: bool = True, env: dict | None = None,
         extra: dict | None = None, stream: bool | None = None) -> dict:
    """Một lần gọi. Trả {content, reasoning, usage, cost_usd, latency_s, model, finish, provider}. Lỗi mạng/429/5xx/524 thì thử lại.
    stream mặc định bật khi trả lời có thể dài (max_tokens > 2000)."""
    stream = max_tokens > 2000 if stream is None else stream
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list) or not messages or any(not isinstance(m, dict) or not isinstance(m.get("content"), str) or not isinstance(m.get("role"), str) for m in messages):
        raise ValueError("guarded chat requires string content for every message")
    if set(extra or {}) & {"messages", "model", "stream", "stream_options"}:
        raise ValueError("extra cannot override guarded messages, model or transport")
    if _RUN and not check_leak:
        raise ValueError("configured runs require prompt guarding")
    if check_leak:
        guard("\n".join(m["content"] for m in messages if isinstance(m.get("content"), str)))
    cfg = config(provider, env)
    body = {"model": cfg["model"], "messages": messages, "max_tokens": max_tokens, **PROVIDERS[provider]["think"][think], **(extra or {})}
    if temperature is not None:
        body["temperature"] = temperature
    prompt_sha = hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()
    err, t0 = None, time.time()
    for attempt in range(retries + 1):
        t0 = time.time()
        try:
            with _provider_slot(provider):
                row = _reserve(provider, tag, prompt_sha, body, attempt + 1, think)
                try:
                    d = _post(cfg, body, timeout, stream)
                except BaseException as exc:
                    _finish_request(row, "error", error=type(exc).__name__)
                    raise
                _finish_request(row, "success", response=d)
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            err = f"HTTP {e.code}: {detail}"
            if e.code not in (408, 409, 429, 500, 502, 503, 504, 524) or attempt == retries:
                raise RuntimeError(f"{provider}: {err}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = f"{type(e).__name__}: {e}"
            if attempt == retries:
                raise RuntimeError(f"{provider}: {err}") from None
        time.sleep(min(60, 3 * 2 ** attempt))
    msg = d["choices"][0]["message"]
    out = {"provider": provider, "model": d.get("model", cfg["model"]), "content": msg.get("content") or "",
           "reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "", "finish": d["choices"][0].get("finish_reason"),
           "usage": d.get("usage") or {}, "latency_s": round(time.time() - t0, 2)}
    out["cost_usd"] = cost_usd(provider, out["usage"])
    if log:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        rec = {"ts": _dt.datetime.now().isoformat(timespec="seconds"), "tag": tag, "think": think, "max_tokens": max_tokens,
               "prompt_sha": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()[:16], "messages": messages,
               **out}
        with open(LOG_DIR / f"{_dt.date.today():%Y%m%d}.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def extract_json(text: str):
    """JSON đầu tiên trong trả lời (bỏ ```json). Không parse được thì ValueError."""
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    s = m.group(1) if m else text
    s = s.strip()
    starts = [i for i in (s.find("{"), s.find("[")) if i >= 0]
    if not starts:
        raise ValueError("không thấy JSON")
    s = s[min(starts):]
    dec = json.JSONDecoder(strict=False)  # cho phép xuống dòng thật trong chuỗi (hay gặp ở dòng bảng luật)
    obj, _ = dec.raw_decode(s)
    return obj


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    return (max(blocks, key=len) if blocks else text).strip()


# ---------------------------------------------------------------- CLI
def _check():
    configure_run(ROOT / "outputs/synth_request_budget.sqlite", provider_limits={p: 1 for p in PROVIDERS})
    env = load_env()
    print(f"{'model':<20}{'suy nghĩ':<10}{'HTTP':<6}{'giây':>6}  trả lời   token ra (suy nghĩ)  ghi chú")
    ok = True
    for name, p in PROVIDERS.items():
        for think in (("high",) if p.get("think_co_dinh") else ("off", "high")):
            try:
                r = chat(name, "Chỉ trả lời đúng một từ: OK", think=think, max_tokens=800, tag="check", env=env, retries=1, timeout=120)
                rt = (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens")
                print(f"{r['model']:<20}{think:<10}{'200':<6}{r['latency_s']:>6}  {r['content'].strip()[:8]!r:<9}"
                      f"{r['usage'].get('completion_tokens')!s:>4} ({rt if rt is not None else len(r['reasoning'])})       {p['ghi_chu'] if think == 'off' else ''}")
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"{p['model']:<20}{think:<10}{'LỖI':<6}{'':>6}  {str(e)[:150]}")
    return 0 if ok else 1


def _curl():
    for name, p in PROVIDERS.items():
        think = json.dumps(p["think"]["off"], ensure_ascii=False)[1:-1]
        body = '{"model":"%s","messages":[{"role":"user","content":"Chỉ trả lời đúng một từ: OK"}],"max_tokens":200%s}' % (
            p["model"], ("," + think) if think else "")
        url = f"${p['url']}" if name == "qwen" else f"${{{p['url']}%/}}/chat/completions"
        print(f"# {name}: {p['model']}\nset -a; . ./.env; set +a\ncurl -sS {url} -H \"Authorization: Bearer ${p['key']}\" "
              f"-H \"x-api-key: ${p['key']}\" -H 'Content-Type: application/json' -d '{body}'\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    sys.exit(_check() if cmd == "check" else _curl() or 0)
