"""Dựng lại luồng trả lời của từng câu từ log một lượt chạy: prompt → model viết code → sandbox chạy → sửa lỗi → phiếu → quyết định.

    python eval/show_trace.py OUT/run --id q_abc                              # in ra màn hình
    python eval/show_trace.py OUT/run --decision no_consensus --limit 5
    python eval/show_trace.py OUT/run --gold data/train/answers.jsonl --wrong --html trace.html
    python eval/show_trace.py OUT/run --id q_abc --data /content/hard --cache /content/hard_cache   # kèm prompt đã gửi

Log không lưu prompt; có --data và --cache thì dựng lại prompt từ cấu hình trong logs/run_meta.json (khớp khi code chưa đổi).
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]

from datamate.llm import extract_code  # noqa: E402
from score import match  # noqa: E402

MARK = {1: "ĐÚNG", -1: "SAI", 0: "bỏ trống"}


def _path_of(step: dict, seed0: int) -> str:
    """Log cũ chưa ghi path: suy từ seed (phiếu i dùng seed0 + 1000·i, đường bổ sung thứ k dùng seed0 + 1000·(k + 10))."""
    if step.get("path"):
        return step["path"]
    k = (int(step.get("seed", seed0)) - seed0) // 1000
    return f"base{k}" if k < 10 else f"đường#{k - 10}"


def _short(ans) -> str:
    if not ans:
        return "không ra đáp án"
    shown = {k: v for k, v in ans.items() if k not in ("id", "template")}
    return json.dumps(shown, ensure_ascii=False)[:300]


def build(r: dict, gold: dict | None, seed0: int) -> dict:
    """Một câu thành cấu trúc: các phiếu theo thứ tự chạy, mỗi phiếu gồm các lượt (phản hồi, code, kết quả)."""
    paths: OrderedDict = OrderedDict()
    for st in r.get("steps", []):
        res = st.get("result") or {}
        turn = {
            "turn": st.get("turn", 0),
            "response": st.get("response", ""),
            "code": extract_code(st.get("response", "")),
            "ok": bool(res.get("ok")),
            "answer": res.get("answer"),
            "error": res.get("error"),
            "stdout": res.get("stdout", ""),
            "error_code": st.get("error_code"),
            "repair_type": st.get("repair_type"),
            "verify": st.get("verify"),
        }
        paths.setdefault(_path_of(st, seed0), {"turns": [], "answer": None})["turns"].append(turn)
    for v in r.get("votes", []):
        paths.setdefault(v["path"], {"turns": [], "answer": None})["answer"] = v["answer"]
    if "votes" not in r and isinstance(r.get("candidates"), list) and len(r["candidates"]) == len(paths):
        # Log cũ (≤ v4.2) chưa có votes: candidates xếp đúng thứ tự phiếu, None = phiếu không ra đáp án
        for name, c in zip(list(paths), r["candidates"]):
            if c:
                paths[name]["answer"] = {k: v for k, v in c.items() if k != "id"}
    g = gold.get(r["id"]) if gold else None
    for p in paths.values():
        p["mark"] = match({"id": r["id"], **p["answer"]}, g) if g and p["answer"] else None
    final = r.get("final") or {"type": "abstain"}
    return {
        "id": r["id"], "business_id": r.get("business_id"), "question": r.get("question"), "as_of_date": r.get("as_of_date"),
        "decision": r.get("decision") or ("skipped" if r.get("skipped") else "error" if r.get("error") else "?"),
        "template": r.get("template"), "final": final, "llm_calls": r.get("llm_calls", 0), "elapsed_s": r.get("elapsed_s"),
        "error": r.get("error") or r.get("template_error"), "gold": g, "mark": match(final, g) if g else None,
        "paths": paths, "extra": {k: r[k] for k in ("plan", "plan_error", "judge", "judge_error") if k in r},
    }


class Prompts:
    """Dựng lại prompt ban đầu của mỗi phiếu bằng chính Agent._messages với cấu hình của lượt chạy."""

    def __init__(self, run: Path, data: Path, cache: Path, cfg: dict):
        from datamate import agent as agent_mod, loader
        from datamate.llm import DummyLLM
        from datamate.run import make_fewshot
        self.agent_mod, self.loader, self.run, self.data, self.cache = agent_mod, loader, run, data, cache
        self.agent = agent_mod.Agent(DummyLLM(), cfg["agent"], (data / "rulebook.md").read_text(encoding="utf-8"),
                                     data / "calendar.csv", make_fewshot(cfg))
        self.variants = cfg["agent"].get("variants") or agent_mod.DEFAULT_VARIANTS
        self.biz: dict = {}

    def get(self, r: dict, path: str) -> list[dict] | None:
        b = r["business_id"]
        if b not in self.biz:
            biz = self.loader.load_cached(self.data / "data" / b, self.cache)
            side = self.run / "logs" / f"contract_{b}.json"
            if side.exists():
                biz.extra_roles = json.loads(side.read_text(encoding="utf-8"))
            self.biz[b] = biz
        if path.startswith("base") and path[4:].isdigit():
            variant = self.variants[int(path[4:]) % len(self.variants)]
        elif path in self.agent_mod.ESCALATE_PATHS:
            variant = self.agent_mod.ESCALATE_PATHS[path]
        else:
            return None
        return self.agent._messages(r, self.biz[b], variant)


def render_text(t: dict, prompts=None, raw=None, full=False) -> str:
    out = [f"{'=' * 100}", f"{t['id']}  shop {t['business_id']}  hôm nay {t['as_of_date']}",
           f"Câu hỏi: {t['question']}",
           f"Quyết định: {t['decision']}" + (f" (dạng {t['template']})" if t["template"] else "")
           + f"  → {_short(t['final'])}" + (f"  [{MARK[t['mark']]}]" if t["mark"] is not None else ""),
           f"Gọi model {t['llm_calls']} lần, {t['elapsed_s']} giây"]
    if t["gold"]:
        out.append(f"Đáp án chuẩn: {_short(t['gold'])}")
    if t["error"]:
        out.append(f"Lỗi: {t['error']}")
    for k, v in t["extra"].items():
        out.append(f"{k}: {json.dumps(v, ensure_ascii=False)[:500]}")
    for name, p in t["paths"].items():
        mark = f"  [{MARK[p['mark']]}]" if p["mark"] is not None else ""
        out.append(f"\n--- phiếu {name}: {_short(p['answer'])}{mark}")
        msgs = prompts.get(raw, name) if prompts else None
        if msgs:
            body = msgs[-1]["content"]
            out.append(f"  [prompt người dùng, {len(body)} ký tự{'' if full else ', 1500 ký tự cuối'}]")
            out.append(_indent(body if full else body[-1500:]))
        for tr in p["turns"]:
            tag = "chạy được" if tr["ok"] else "lỗi"
            out.append(f"  lượt {tr['turn']}: {tag}" + (f", {tr['error_code']}" if tr["error_code"] else "")
                       + (f", loại {tr['repair_type']}" if tr["repair_type"] else ""))
            code = tr["code"] if tr["code"] is not None else tr["response"]
            if code and not full and len(code) > 2500:
                code = code[:2500] + "\n... (cắt, dùng --full)"
            out.append(_indent(code or "(không có phản hồi)"))
            if tr["ok"]:
                out.append(f"    → ANSWER {_short(tr['answer'])}")
            if tr["error"]:
                out.append(_indent(tr["error"] if full else tr["error"][:1200], "    ! "))
            if tr["stdout"].strip():
                out.append(_indent(tr["stdout"][-600:], "    > "))
    return "\n".join(out)


def _indent(s: str, pre: str = "    | ") -> str:
    return "\n".join(pre + line for line in str(s).splitlines())


CSS = """
:root{--bg:#fff;--fg:#1d2330;--mut:#5a6475;--line:#d8dee8;--card:#f6f8fb;--ok:#2f7d46;--bad:#b3261e;--warn:#9a6b00;--code:#f1f3f6}
@media (prefers-color-scheme:dark){:root{--bg:#15181e;--fg:#e4e7ec;--mut:#9aa3b2;--line:#2c323c;--card:#1c2027;--ok:#5cc27a;--bad:#ff7b72;--warn:#e0b341;--code:#22272f}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px}
h1{font-size:20px;margin:0 0 4px}.mut{color:var(--mut)}
.bar{display:flex;flex-wrap:wrap;gap:6px;margin:12px 0;position:sticky;top:0;background:var(--bg);padding:6px 0}
.bar button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:14px;padding:3px 10px;cursor:pointer}
.bar button.on{border-color:var(--fg);font-weight:600}.bar input{flex:1;min-width:160px;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
details.q{border:1px solid var(--line);border-radius:8px;margin:8px 0;background:var(--card)}
details.q>summary{padding:8px 10px;cursor:pointer;list-style:none}details.q[open]>summary{border-bottom:1px solid var(--line)}
.body{padding:8px 12px}.tag{display:inline-block;border-radius:4px;padding:0 6px;font-size:12px;border:1px solid var(--line);margin-right:4px}
.ok{color:var(--ok);border-color:var(--ok)}.bad{color:var(--bad);border-color:var(--bad)}.warn{color:var(--warn);border-color:var(--warn)}
.path{border-left:3px solid var(--line);margin:10px 0;padding-left:10px}.turn{margin:6px 0}
pre{background:var(--code);padding:8px;border-radius:6px;overflow-x:auto;white-space:pre-wrap;word-break:break-word;font:12px/1.4 Menlo,monospace;margin:4px 0}
pre.err{border-left:3px solid var(--bad)}details.sub>summary{cursor:pointer;color:var(--mut);font-size:13px}
"""

JS = """
const items=[...document.querySelectorAll('details.q')];let dec='all';
function apply(){const s=document.getElementById('s').value.toLowerCase();
 items.forEach(d=>{d.style.display=((dec==='all'||d.dataset.dec===dec||d.dataset.mark===dec)&&d.textContent.toLowerCase().includes(s))?'':'none'})}
document.querySelectorAll('.bar button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.bar button').forEach(x=>x.classList.remove('on'));b.classList.add('on');dec=b.dataset.k;apply()});
document.getElementById('s').oninput=apply;
"""


def _mark_tag(m) -> str:
    if m is None:
        return ""
    return f'<span class="tag {"ok" if m == 1 else "bad" if m == -1 else "warn"}">{MARK[m]}</span>'


def render_html(traces: list[dict], title: str, prompts=None, raws=None) -> str:
    e = html.escape
    counts: dict = {}
    for t in traces:
        counts[t["decision"]] = counts.get(t["decision"], 0) + 1
    marks = {k: sum(t["mark"] == v for t in traces) for k, v in (("ĐÚNG", 1), ("SAI", -1))}
    buttons = [f'<button class="on" data-k="all">tất cả {len(traces)}</button>']
    buttons += [f'<button data-k="{e(k)}">{e(k)} {n}</button>' for k, n in sorted(counts.items(), key=lambda x: -x[1])]
    buttons += [f'<button data-k="{k}">{k} {n}</button>' for k, n in marks.items() if n]
    parts = [f"<!doctype html><html lang='vi'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>Luồng trả lời</title><style>{CSS}</style></head><body><h1>Luồng trả lời</h1>"
             f"<div class='mut'>{e(title)}</div><div class='bar'>{''.join(buttons)}<input id='s' placeholder='tìm id, câu hỏi, lỗi…'></div>"]
    for t in traces:
        mark = {1: "ĐÚNG", -1: "SAI"}.get(t["mark"], "")
        parts.append(f"<details class='q' data-dec='{e(t['decision'])}' data-mark='{mark}'><summary>{_mark_tag(t['mark'])}"
                     f"<span class='tag'>{e(t['decision'])}</span><b>{e(t['id'])}</b> <span class='mut'>{e(str(t['business_id']))}</span> · "
                     f"{e(t['question'] or '')}</summary><div class='body'>")
        parts.append(f"<div>Hôm nay {e(str(t['as_of_date']))} · gọi model {t['llm_calls']} lần · {t['elapsed_s']} giây"
                     + (f" · dạng {e(t['template'])}" if t["template"] else "") + "</div>")
        parts.append(f"<div>Nộp: <code>{e(_short(t['final']))}</code></div>")
        if t["gold"]:
            parts.append(f"<div>Chuẩn: <code>{e(_short(t['gold']))}</code></div>")
        if t["error"]:
            parts.append(f"<pre class='err'>{e(t['error'])}</pre>")
        for k, v in t["extra"].items():
            parts.append(f"<div class='mut'>{e(k)}: <code>{e(json.dumps(v, ensure_ascii=False)[:800])}</code></div>")
        for name, p in t["paths"].items():
            parts.append(f"<div class='path'><b>phiếu {e(name)}</b> {_mark_tag(p['mark'])} <code>{e(_short(p['answer']))}</code>")
            msgs = prompts.get(raws[t["id"]], name) if prompts else None
            if msgs:
                parts.append(f"<details class='sub'><summary>prompt đã gửi ({len(msgs[-1]['content'])} ký tự)</summary>"
                             f"<pre>{e(msgs[-1]['content'])}</pre><details class='sub'><summary>system prompt</summary>"
                             f"<pre>{e(msgs[0]['content'])}</pre></details></details>")
            for tr in p["turns"]:
                head = ("chạy được" if tr["ok"] else "lỗi") + (f" · {tr['error_code']}" if tr["error_code"] else "") \
                    + (f" · loại {tr['repair_type']}" if tr["repair_type"] else "")
                parts.append(f"<div class='turn'><span class='tag {'ok' if tr['ok'] and not tr['error_code'] else 'bad'}'>lượt {tr['turn']}</span>{e(head)}")
                parts.append(f"<pre>{e(tr['code'] if tr['code'] is not None else tr['response'] or '(không có phản hồi)')}</pre>")
                if tr["code"] is not None and tr["response"].strip() != tr["code"].strip():
                    parts.append(f"<details class='sub'><summary>phản hồi đầy đủ của model</summary><pre>{e(tr['response'])}</pre></details>")
                if tr["ok"]:
                    parts.append(f"<div>→ ANSWER <code>{e(_short(tr['answer']))}</code></div>")
                if tr["error"]:
                    parts.append(f"<pre class='err'>{e(tr['error'])}</pre>")
                if tr["stdout"].strip():
                    parts.append(f"<details class='sub'><summary>stdout</summary><pre>{e(tr['stdout'])}</pre></details>")
                parts.append("</div>")
            parts.append("</div>")
        parts.append("</div></details>")
    parts.append(f"<script>{JS}</script></body></html>")
    return "".join(parts)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="thư mục --out của datamate.run")
    ap.add_argument("--id", nargs="*", default=[], help="các id cần xem")
    ap.add_argument("--ids", help="jsonl có trường id")
    ap.add_argument("--decision", nargs="*", default=[], help="vd no_consensus consensus template escalated")
    ap.add_argument("--gold", help="answers.jsonl để đánh dấu đúng sai từng phiếu")
    ap.add_argument("--wrong", action="store_true", help="chỉ câu nộp sai (cần --gold)")
    ap.add_argument("--model-only", action="store_true", help="bỏ câu lớp tính toán trả lời")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--data", help="bộ dữ liệu của lượt chạy, để dựng lại prompt")
    ap.add_argument("--cache", help="cache_dir của lượt chạy")
    ap.add_argument("--html", help="ghi ra một file HTML thay vì in")
    ap.add_argument("--full", action="store_true", help="in đủ prompt, code và lỗi")
    a = ap.parse_args(argv)

    run = Path(a.run)
    meta_p = run / "logs" / "run_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    cfg = meta.get("config") or {}
    seed0 = (cfg.get("agent") or {}).get("seed", 42)
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(a.gold, encoding="utf-8") if l.strip()} if a.gold else None
    want = set(a.id)
    if a.ids:
        want |= {json.loads(l)["id"] for l in open(a.ids, encoding="utf-8") if l.strip()}
    files = sorted(p for p in (run / "logs").glob("*.json") if p.name != "run_meta.json" and not p.name.startswith("contract_"))
    raws, traces = {}, []
    for p in files:
        if want and p.stem not in want:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if "id" not in r:
            continue
        t = build(r, gold, seed0)
        if a.decision and t["decision"] not in a.decision:
            continue
        if a.wrong and t["mark"] != -1:
            continue
        if a.model_only and t["decision"] in ("template", "template_rejected"):
            continue
        raws[t["id"]] = r
        traces.append(t)
        if a.limit and len(traces) >= a.limit:
            break
    if not traces:
        print("không có câu nào khớp bộ lọc")
        return 1

    prompts = None
    if a.data and a.cache:
        if not cfg:
            print("thiếu logs/run_meta.json nên không dựng được prompt", file=sys.stderr)
        else:
            from datamate import __version__
            if meta.get("code_version") != __version__:
                print(f"cảnh báo: lượt chạy dùng code {meta.get('code_version')}, code hiện tại {__version__}; prompt dựng lại có thể khác",
                      file=sys.stderr)
            prompts = Prompts(run, Path(a.data), Path(a.cache), cfg)

    if a.html:
        title = f"{run} · code {meta.get('code_version', '?')} · {len(traces)} câu"
        Path(a.html).write_text(render_html(traces, title, prompts, raws), encoding="utf-8")
        print(f"đã ghi {a.html} ({len(traces)} câu)")
    else:
        for t in traces:
            print(render_text(t, prompts, raws[t["id"]], a.full))
    return 0


if __name__ == "__main__":
    sys.exit(main())
