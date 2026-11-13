"""Điểm vào của pipeline.

Luồng: đọc questions.jsonl → loader (mỗi doanh nghiệp một lần, có cache) → agent (LLM viết code pandas
→ sandbox → sửa lỗi → bỏ phiếu) → formatter → ghi từng dòng submission.jsonl ngay khi xong.
Câu lỗi hoặc không đủ phiếu → abstain. Chạy lại cùng --out thì bỏ qua câu đã có log (chạy tiếp được).

    python -m datamate.run --data data/train --out outputs/train [--ids eval/dev.jsonl] [--limit 20]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

from . import __version__, contract, formatter, loader, shop as shop_mod
from .agent import Agent, budget_paths
from .fewshot import FewShot
from .llm import LLM, DummyLLM

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("datamate")


def _load_one(args):
    biz_dir, cache_dir = args
    t = time.time()
    loader.load_cached(biz_dir, cache_dir)
    return Path(biz_dir).name, time.time() - t


def cache_path(biz_dir: Path, cache_dir: Path) -> Path:
    return loader.cache_file(biz_dir, cache_dir)


def _sha256_file(path: Path) -> str | None:
    import hashlib
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def run_metadata(cfg: dict, data_dir: Path, questions: list, backend: str) -> dict:
    """Mọi thứ quyết định kết quả, để so hai lần chạy khi BTC kiểm tra tái lập (luật 8)."""
    import platform

    import numpy
    import pandas

    from .agent import PROMPT_DIR

    served = None
    if backend == "openai":
        try:
            import urllib.request
            with urllib.request.urlopen(cfg["model"].get("base_url", "").rstrip("/") + "/models", timeout=10) as r:
                served = json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            served = f"không đọc được: {e}"
    weights = os.environ.get("MODEL_DIR", "")
    return {
        "code_version": __version__,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_dir": str(data_dir),
        "n_questions": len(questions),
        "questions_sha256": _sha256_file(data_dir / "questions.jsonl"),
        "rulebook_sha256": _sha256_file(data_dir / "rulebook.md"),
        "backend": backend,
        "model": cfg.get("model", {}).get("name"),
        "served_models": served,
        "weights_sha256": _sha256_file(Path(weights + ".sha256")) if weights else None,
        "prompt_sha256": _sha256_file(PROMPT_DIR / "system.md"),
        "code_bank_sha256": _sha256_file(ROOT / (cfg.get("fewshot") or {}).get("code_bank", "-")),
        "config": cfg,
        "env": {k: os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT", "PYTHONHASHSEED", "MODEL_DIR", "NUM_GPUS")},
        "versions": {"python": platform.python_version(), "pandas": pandas.__version__, "numpy": numpy.__version__},
    }


def log_stats(logs_dir: Path, ids: set) -> dict:
    """Số lượt gọi LLM, thời gian mỗi câu và lý do quyết định: để ước lượng ngân sách 6 giờ."""
    import statistics
    from collections import Counter

    calls, secs, decisions, timed_out = [], [], Counter(), 0
    for p in logs_dir.glob("q_*.json"):
        if p.stem not in ids:
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        timed_out += "skipped" in r
        if "llm_calls" in r:
            calls.append(r["llm_calls"])
            secs.append(r.get("elapsed_s", 0.0))
            decisions[r.get("decision", "?")] += 1
    q = statistics.quantiles(secs, n=20, method="inclusive") if len(secs) >= 2 else (secs or [0.0]) * 19
    return {"timed_out_questions": timed_out,
            "llm_calls_mean": round(statistics.mean(calls), 2) if calls else 0,
            "question_seconds_p50": round(q[9], 1), "question_seconds_p95": round(q[18], 1),
            "decisions": dict(decisions)}


def make_fewshot(cfg: dict):
    """Bộ ví dụ train như lúc chạy; eval/show_trace.py dùng lại để dựng đúng prompt đã gửi."""
    fs_cfg = cfg.get("fewshot") or {}
    if not (fs_cfg.get("questions") and Path(ROOT / fs_cfg["questions"]).exists()):
        return None
    exclude = []
    if fs_cfg.get("exclude_businesses_file") and (ROOT / fs_cfg["exclude_businesses_file"]).exists():
        exclude = (ROOT / fs_cfg["exclude_businesses_file"]).read_text(encoding="utf-8").split()
    bank = fs_cfg.get("code_bank")
    return FewShot(ROOT / fs_cfg["questions"], ROOT / fs_cfg["answers"], exclude, ROOT / bank if bank else None)


def _check_versions() -> None:
    """Cảnh báo khi phiên bản pandas khác requirements.txt: loader sẽ đọc ra dữ liệu khác bản đã chấm (luật 8)."""
    import re
    import pandas
    req = (ROOT / "requirements.txt")
    m = re.search(r"^pandas==(\S+)", req.read_text(encoding="utf-8"), re.M) if req.exists() else None
    if m and pandas.__version__ != m.group(1):
        log.warning("pandas %s khác requirements.txt (%s): dữ liệu sau làm sạch có thể khác bản đã chấm. "
                    "Chạy: pip install -r requirements.txt", pandas.__version__, m.group(1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "datamate" / "config.yaml"))
    ap.add_argument("--data", help="thư mục bộ dữ liệu (có questions.jsonl, rulebook.md, calendar.csv, data/)")
    ap.add_argument("--out", help="thư mục kết quả: submission.jsonl + logs/")
    ap.add_argument("--ids", help="jsonl có trường id: chỉ chạy các câu này")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--backend", choices=["openai", "dummy"], default="openai")
    ap.add_argument("--set", action="append", default=[], help="đổi cấu hình khi so sánh, vd model.enable_thinking=true, agent.judge=false")
    a = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    for item in a.set:
        key, _, value = item.partition("=")
        node = cfg
        for part in key.split(".")[:-1]:
            node = node.setdefault(part, {})
        node[key.split(".")[-1]] = yaml.safe_load(value)
    data_dir = Path(a.data or cfg["data_dir"])
    out_dir = Path(a.out or cfg["output_dir"])
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cfg.get("cache_dir", "/tmp/datamate_cache"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(out_dir / "run.log")])
    t0 = time.time()

    questions = [json.loads(l) for l in (data_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()
                 if l.strip()]
    if a.ids:
        keep = {json.loads(l)["id"] for l in Path(a.ids).read_text(encoding="utf-8").splitlines() if l.strip()}
        questions = [q for q in questions if q["id"] in keep]
    if a.limit:
        questions = questions[: a.limit]
    # cùng doanh nghiệp đi liền nhau để vLLM dùng lại prefix cache
    questions.sort(key=lambda q: (q["business_id"], q["id"]))

    sub_path = out_dir / "submission.jsonl"
    done = {}
    if sub_path.exists():
        for l in sub_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                done[r["id"]] = r
    todo = [q for q in questions if q["id"] not in done]
    log.info("code %s", __version__)
    _check_versions()
    log.info("data=%s: %d câu, %d đã có, còn %d", data_dir, len(questions), len(done), len(todo))

    # 1) nạp dữ liệu song song, cache pickle
    biz_ids = sorted({q["business_id"] for q in todo})
    jobs = [(str(data_dir / "data" / b), str(cache_dir)) for b in biz_ids]
    with ProcessPoolExecutor(max_workers=min(len(jobs), cfg.get("load_workers", os.cpu_count() or 4)) or 1) as ex:
        for fut in as_completed([ex.submit(_load_one, j) for j in jobs]):
            try:
                b, sec = fut.result()
                log.info("nạp %s trong %.1fs", b, sec)
            except Exception as e:  # noqa: BLE001
                log.error("nạp dữ liệu lỗi: %s", e)

    # 2) agent
    llm = DummyLLM() if a.backend == "dummy" else LLM(cfg["model"])
    fewshot = make_fewshot(cfg)
    # model.second: model thứ hai cho lượt bỏ phiếu chẵn; tổng tham số các model phải ≤ 15B
    second = cfg["model"].get("second")
    vote_llms = [llm, LLM({**cfg["model"], **second})] if second and a.backend != "dummy" else None
    if cfg["agent"].get("budget") and not cfg["agent"].get("collect_all"):
        plan = budget_paths(len(questions), cfg["agent"])
        log.info("kế hoạch theo %d câu: %s", len(questions), plan)
        cfg["agent"].update(escalate_paths=plan["paths"], template_guard=plan["template_guard"])
    agent = Agent(llm, cfg["agent"], (data_dir / "rulebook.md").read_text(encoding="utf-8"),
                  data_dir / "calendar.csv", fewshot, vote_llms)

    meta = run_metadata(cfg, data_dir, questions, a.backend)
    (logs_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("metadata: %s", json.dumps({k: meta[k] for k in ("model", "weights_sha256", "prompt_sha256")}))

    lock = threading.Lock()
    biz_cache: dict = {}

    def get_biz(bid):
        with lock:
            if bid not in biz_cache:
                biz_cache[bid] = loader.load_cached(data_dir / "data" / bid, cache_dir)
            return biz_cache[bid]

    shop_cache: dict = {}

    def get_shop(bid, biz):
        with lock:
            if bid not in shop_cache:
                try:
                    pickle_path = cache_path(data_dir / "data" / bid, cache_dir)
                    if cfg["agent"].get("contract", False) and a.backend != "dummy":
                        contract.ensure(llm, biz, pickle_path, logs_dir)
                    else:
                        contract.sidecar(pickle_path).unlink(missing_ok=True)  # đừng để sandbox đọc hợp đồng của lượt khác
                    shop_cache[bid] = shop_mod.build(biz)
                except Exception as e:  # noqa: BLE001 — không dựng được khung thì để model trả lời
                    log.warning("không dựng được khung chung cho %s: %s", bid, e)
                    shop_cache[bid] = None
            return shop_cache[bid]

    # lập hợp đồng tuần tự trước khi các câu chạy song song: gọi xen với câu khác thì vLLM gộp batch khác nhau, hợp đồng lệch giữa hai lượt (luật 8)
    for bid in biz_ids:
        try:
            get_shop(bid, get_biz(bid))
        except Exception as e:  # noqa: BLE001 — câu của shop này tự ghi lỗi ở work()
            log.warning("không chuẩn bị được %s: %s", bid, e)

    deadline = t0 + cfg.get("time_budget_s", 5.5 * 3600)

    def work(q):
        if time.time() > deadline:
            return {"id": q["id"], "type": "abstain"}, {"id": q["id"], "skipped": "hết thời gian"}
        try:
            biz = get_biz(q["business_id"])
            return agent.answer(q, biz, cache_path(data_dir / "data" / q["business_id"], cache_dir), get_shop(q["business_id"], biz))
        except Exception as e:  # noqa: BLE001 — một câu lỗi không làm dừng cả lượt chạy
            log.exception("câu %s lỗi", q["id"])
            return {"id": q["id"], "type": "abstain"}, {"id": q["id"], "error": repr(e)}

    n_done = 0
    with sub_path.open("a", encoding="utf-8") as sub, ThreadPoolExecutor(cfg.get("workers", 16)) as ex:
        futs = {ex.submit(work, q): q for q in todo}
        for fut in as_completed(futs):
            final, qlog = fut.result()
            final = formatter.finalize(final["id"], final)
            with lock:
                sub.write(json.dumps(final, ensure_ascii=False) + "\n")
                sub.flush()
                (logs_dir / f"{final['id']}.json").write_text(
                    json.dumps(qlog, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
                n_done += 1
                if n_done % 20 == 0 or n_done == len(todo):
                    log.info("xong %d/%d (%.0fs)", n_done, len(todo), time.time() - t0)

    # câu thiếu (nếu có) → abstain, để file đủ dòng
    have = {json.loads(l)["id"] for l in sub_path.read_text(encoding="utf-8").splitlines() if l.strip()}
    with sub_path.open("a", encoding="utf-8") as sub:
        for q in questions:
            if q["id"] not in have:
                sub.write(json.dumps({"id": q["id"], "type": "abstain"}, ensure_ascii=False) + "\n")
    meta["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["elapsed_s"] = round(time.time() - t0, 1)
    meta.update(log_stats(logs_dir, {q["id"] for q in questions}))
    (logs_dir / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("hoàn tất trong %.0fs → %s", time.time() - t0, sub_path)


if __name__ == "__main__":
    main()
