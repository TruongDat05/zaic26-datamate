"""Đưa log chạy thật (data/real_logs/<lượt>/) vào nguồn sinh dữ liệu: data/synth_source/errors/<lượt>/.

    python eval/ingest_real_logs.py                 # mọi lượt trong data/real_logs
    python eval/ingest_real_logs.py --runs hard42_model

Mỗi lượt là một thư mục --out của datamate.run (logs/q_*.json, logs/run_meta.json, submission.jsonl, run.log).
- Lượt trên câu train (có đáp án trong data/train/answers.jsonl): chạy eval/mine_errors.py → questions, positives, negatives,
  summary; thêm by_shop.json (đúng/sai/bỏ, PA theo shop). Bộ dữ liệu của lượt lấy theo tên thư mục data_dir trong run_meta
  (vd /content/hard → data/real_logs/hard, cache data/real_logs/hard_cache).
- Lượt không có đáp án (public): chỉ thống kê tổng (cách chọn đáp án, mã lỗi, kiểu đáp án, thời gian). Không ghi câu hỏi, mã câu,
  hay số theo shop; không dùng để chọn/chỉnh dữ liệu sinh (luật 6).

Shop holdout/tune: câu và code chỉ để phân tích mẫu lỗi (usable_as_seed = false), không đưa vào dữ liệu train.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
REAL = ROOT / "data/real_logs"
SOURCE = ROOT / "data/synth_source"


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _resolve(path_str: str | None, real: Path) -> Path | None:
    if not path_str:
        return None
    p = Path(path_str)
    for c in (p if p.is_absolute() else ROOT / p, real / p.name):
        if c.exists():
            return c
    return None


def by_shop(questions: list[dict]) -> dict:
    out = collections.defaultdict(lambda: {"n": 0, "dung": 0, "sai": 0, "bo": 0, "nhom": collections.Counter()})
    for q in questions:
        s = out[q["business_id"]]
        s["n"] += 1
        s["dung"] += q["mark"] == 1
        s["sai"] += q["mark"] == -1
        s["bo"] += (q.get("final") or {}).get("type") == "abstain"
        s["nhom"][q["group"]] += 1
    return {b: {**{k: v for k, v in s.items() if k != "nhom"}, "PA": round((s["dung"] - s["sai"]) / s["n"], 3), "nhom": dict(s["nhom"])}
            for b, s in sorted(out.items())}


def stats_only(run: Path, meta: dict) -> dict:
    dec, codes, rtypes, types = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    calls, secs, n = [], [], 0
    for f in sorted((run / "logs").glob("q_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        n += 1
        dec[r.get("decision") or "?"] += 1
        types[(r.get("final") or {}).get("type", "?")] += 1
        calls.append(r.get("llm_calls", 0))
        if r.get("elapsed_s") is not None:
            secs.append(r["elapsed_s"])
        for st in r.get("steps", []):
            if st.get("error_code"):
                codes[st["error_code"]] += 1
            if st.get("repair_type"):
                rtypes[st["repair_type"]] += 1
    q = statistics.quantiles(secs, n=20) if len(secs) >= 20 else []
    return {"run": run.name, "chi_thong_ke": True,
            "ly_do": "không có đáp án (public); chỉ thống kê tổng, không chọn/chỉnh dữ liệu sinh theo lượt này (luật 6)",
            "code_version": meta.get("code_version"), "model": meta.get("model"), "n_questions": n, "decisions": dict(dec.most_common()),
            "final_types": dict(types.most_common()), "step_codes": dict(codes.most_common()), "repair_types": dict(rtypes.most_common()),
            "llm_calls_tb": round(statistics.mean(calls), 2) if calls else None, "llm_calls_max": max(calls) if calls else None,
            "giay_trung_vi": round(statistics.median(secs), 1) if secs else None, "giay_p95": round(q[18], 1) if q else None,
            "giay_max": round(max(secs), 1) if secs else None}


def ingest(run: Path, source: Path, real: Path) -> dict:
    meta_p = run / "logs/run_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    cfg = meta.get("config") or {}
    data = _resolve(meta.get("data_dir") or cfg.get("data_dir"), real)
    out = source / "errors" / run.name
    out.mkdir(parents=True, exist_ok=True)
    gold_ids = {g["id"] for g in _jsonl(ROOT / "data/train/answers.jsonl")}
    qfile = data / "questions.jsonl" if data else None
    qids = [q["id"] for q in _jsonl(qfile)] if qfile and qfile.exists() else []
    has_gold = bool(qids) and sum(i in gold_ids for i in qids) >= 0.95 * len(qids)
    if not has_gold:
        s = stats_only(run, meta)
        for f in ("questions.jsonl", "positives.jsonl", "negatives.jsonl", "by_shop.json"):
            (out / f).unlink(missing_ok=True)
        (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        return {"run": run.name, "loai": "chi_thong_ke", "n": s["n_questions"], "data": str(data.relative_to(ROOT)) if data else None}
    import mine_errors
    cache = _resolve(cfg.get("cache_dir"), real) or ROOT / "outputs/cache"
    mine_errors.main([str(run), "--name", run.name, "--data", str(data), "--cache", str(cache), "--source", str(source), "--out", str(out)])
    qs = _jsonl(out / "questions.jsonl")
    bs = by_shop(qs)
    (out / "by_shop.json").write_text(json.dumps(bs, ensure_ascii=False, indent=1), encoding="utf-8")
    roles = collections.Counter(q["role"] for q in qs)
    return {"run": run.name, "loai": "co_dap_an", "n": len(qs), "data": str(data.relative_to(ROOT)), "cache": str(cache.relative_to(ROOT)),
            "code_version": meta.get("code_version"), "roles": dict(roles), "seed_duoc": sum(q["usable_as_seed"] for q in qs)}


README = """# Lỗi từ log chạy

Mỗi lượt một thư mục. Dựng lại: `python eval/ingest_real_logs.py` (log gốc ở `data/real_logs/`), hoặc từng lượt mới:
`python eval/mine_errors.py OUT/<lượt> --name <lượt>`. Ý nghĩa nhóm E1–E6b: `taxonomy/error_groups.json`.

| Lượt | Loại | Số câu | Ghi chú |
|---|---|---|---|
{rows}

- `questions.jsonl`: mỗi câu một dòng (nhóm, nhãn, từng phiếu đúng/sai, mã lỗi). `positives.jsonl`: lượt trả lời đúng (kể cả sửa lỗi
  thành công) kèm phản hồi dựng lại nguyên văn. `negatives.jsonl`: lượt sai. `by_shop.json`: đúng/sai/bỏ, PA theo shop.
- **Chỉ dòng `usable_as_seed = true` (shop mining) được dùng làm seed hay mẫu train.** Lượt trên shop holdout/tune chỉ để phân tích
  mẫu lỗi (loại lỗi, cấu trúc shop khó); lượt public chỉ có thống kê tổng (luật 6).
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=str(REAL))
    ap.add_argument("--source", default=str(SOURCE))
    ap.add_argument("--runs", help="tên lượt, cách nhau dấu phẩy (mặc định: mọi lượt)")
    a = ap.parse_args(argv)
    real, source = Path(a.real), Path(a.source)
    runs = sorted(p for p in real.iterdir() if (p / "logs").is_dir())
    if a.runs:
        runs = [p for p in runs if p.name in a.runs.split(",")]
    res = [ingest(r, source, real) for r in runs]
    idx = source / "errors/index.json"
    old = {x["run"]: x for x in json.loads(idx.read_text(encoding="utf-8"))} if idx.exists() else {}
    old.update({x["run"]: x for x in res})
    idx.write_text(json.dumps(list(old.values()), ensure_ascii=False, indent=1), encoding="utf-8")
    rows = []
    for x in old.values():
        note = (f"{x.get('code_version')}; vai trò shop {x.get('roles')}; seed dùng được {x.get('seed_duoc')}" if x["loai"] == "co_dap_an"
                else "chỉ thống kê tổng, không có đáp án")
        rows.append(f"| `{x['run']}` | {x['loai']} | {x['n']} | {note} |")
    (source / "errors/README.md").write_text(README.format(rows="\n".join(rows)), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
