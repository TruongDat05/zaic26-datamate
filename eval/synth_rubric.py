"""Rubric chung đo chất lượng câu hỏi + đáp án, áp y hệt cho dữ liệu gốc (train BTC, chỉ đọc cục bộ) và dữ liệu sinh.

    python eval/synth_rubric.py baseline --out outputs/<run>/rubric      # đo dữ liệu gốc, ghi rubric_v1.json/.md (công bố trước khi tối ưu)
    python eval/synth_rubric.py score --rows <jsonl> --codes <json> --rubric outputs/<run>/rubric/rubric_v1.json

Chỉ in và ghi số liệu gộp; không in câu hỏi BTC. Rubric gồm 9 tiêu chí từng mẫu (đạt/không) và 6 tiêu chí mức tập (0–1).
R = trung bình 15 thành phần. Yêu cầu: R_sinh > 0,9 × R_gốc, và báo riêng thành phần nào dưới 0,9 × mốc gốc.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
VERSION = "rubric_v1"
TYPES = ("number", "percent", "list", "string", "date", "boolean", "table", "unanswerable")
STYLE = ("khong_dau", "co_viet_tat", "co_tu_dem", "go_telex", "chu_thuong_dau", "khong_dau_hoi")
# Nhận diện kỳ: synth_spec.PERIODS + vài cách nói tương đối mà PERIODS chưa có (định nghĩa trước khi đo, áp cho cả hai tập).
EXTRA_PERIODS = [("nam_nay", r"năm nay|nam nay"), ("quy_nay", r"quý này|quy nay"), ("dau_thang", r"đầu tháng|dau thang"),
                 ("ngay_qua", r"ngày qua|ngay qua|ngày gần|ngay gan"), ("tuan_truoc_nua", r"tuần trước nữa|tuan truoc nua")]
PER_SAMPLE = ("period_ok", "no_leak", "no_hint", "type_consistent", "gold_wellformed", "not_zero_suspect", "length_ok",
              "no_internal_code", "no_blocklist")
DATASET = ("style_fidelity", "length_fidelity", "lexical_diversity", "uniqueness", "answer_type_coverage", "period_coverage")


def period_tags(q: str) -> list[str]:
    from synth_spec import PERIODS
    t = q.lower()
    return [k for k, p in list(PERIODS) + EXTRA_PERIODS if re.search(p, t)]


def per_sample(q: str, gold: dict, codes: set, bounds: tuple[int, int]) -> dict[str, bool]:
    """Tiêu chí từng mẫu trên câu model thấy + đáp án chuẩn."""
    import synth_checks
    from datamate import formatter, verify
    from score import match
    g = {k: gold[k] for k in ("type", "answer") if k in gold}
    fin = formatter.finalize("x", g)
    wellformed = fin["type"] == g.get("type") and match(fin, {"alias": [], **g}) == 1
    typed = {k: v for k, v in fin.items() if k != "id"}
    leak = False
    if wellformed and typed["type"] in ("number", "percent", "date", "string", "list"):
        try:
            leak = synth_checks.lo_dap_an(q, typed)
        except Exception:  # noqa: BLE001 — đáp án lạ dạng: coi như không lộ
            leak = False
    n = len(q.split())
    return {"period_ok": bool(period_tags(q)) or synth_checks.ky_ro(q) or g.get("type") == "unanswerable",
            "no_leak": not leak, "no_hint": not synth_checks.goi_y(q),
            "type_consistent": not verify.check(q, typed) if wellformed else False, "gold_wellformed": wellformed,
            "not_zero_suspect": not verify.zero_suspect(q, typed) if wellformed else True,
            "length_ok": bounds[0] <= n <= bounds[1], "no_internal_code": not synth_checks.ma_noi_bo(q, codes),
            "no_blocklist": not synth_checks.blocklist_hits(q)}


def _fold(s: str) -> str:
    import synth_checks
    return synth_checks.fold(s)


def _shingles(q: str) -> set:
    q = re.sub(r"\b(?:cho biết|bao nhiêu|là|nhé)\b", " ", q, flags=re.I)
    w = _fold(q).strip(" ?!.,;:").split()
    return {" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


def _entropy(counter: collections.Counter, k: int) -> float:
    tot = sum(counter.values())
    if not tot or k <= 1:
        return 0.0
    h = -sum(c / tot * math.log(c / tot) for c in counter.values() if c)
    return round(h / math.log(k), 4)


def style_rates(questions: list[str], known=frozenset()) -> dict:
    from build_synth_source import style_of
    from stress_noise import VanPhong
    nets = [VanPhong.net(style_of(q, known)) for q in questions]
    return {t: round(sum(n[t] for n in nets) / max(1, len(nets)), 4) for t in STYLE}


def dataset_scores(rows: list[dict], profile: dict, seed: int = 2026) -> dict:
    """Tiêu chí mức tập; các số phụ thuộc cỡ tập được đo trên mẫu con cùng cỡ (bootstrap) để so được tập lớn/nhỏ."""
    qs = [r["question"] for r in rows]
    rates = style_rates(qs)
    gaps = {t: round(abs(rates[t] - profile["ti_le"][t]), 4) for t in STYLE}
    words = sorted(len(q.split()) for q in qs)
    med = words[len(words) // 2] if words else 0
    rng = random.Random(seed)
    div = []
    for _ in range(20):
        sub = rng.sample(qs, min(400, len(qs)))
        grams = [g for q in sub for g in _ngrams(q)]
        div.append(len(set(grams)) / max(1, len(grams)))
    by_shop = collections.defaultdict(list)
    for r in rows:
        by_shop[r["shop"]].append(r["question"])
    dups, tot = 0, 0
    for _ in range(5):
        for shop, sq in sorted(by_shop.items()):
            sub = rng.sample(sq, min(25, len(sq)))
            sh = [_shingles(q) for q in sub]
            for i, a in enumerate(sh):
                tot += 1
                dups += any(len(a & b) / max(1, len(a | b)) >= 0.7 for j, b in enumerate(sh) if j != i)
    types = collections.Counter(r["gold"]["type"] for r in rows)
    tags = collections.Counter(t for r in rows for t in (period_tags(r["question"]) or ["khong_ro_ky"]))
    from synth_spec import PERIODS
    return {"style_fidelity": round(1 - sum(gaps.values()) / len(gaps), 4), "style_rates": rates, "style_gaps": gaps,
            "length_fidelity": round(1 - min(1.0, abs(med - profile["so_tu"]["p50"]) / profile["so_tu"]["p50"]), 4), "median_words": med,
            "lexical_diversity": round(sum(div) / len(div), 4) if div else 0.0, "uniqueness": round(1 - dups / max(1, tot), 4),
            "answer_type_coverage": _entropy(types, len(TYPES)), "answer_types": dict(types),
            "period_coverage": _entropy(tags, len(PERIODS) + len(EXTRA_PERIODS) + 1), "period_tags": dict(tags)}


def _ngrams(q: str, n: int = 3) -> list[str]:
    w = re.sub(r"\d+", "0", _fold(q)).split()
    return [" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))]


def score(rows: list[dict], codes_by_shop: dict, rubric: dict) -> dict:
    """rows: {question, gold, shop}. Trả tỉ lệ đạt từng tiêu chí, điểm mức tập, R và số mẫu đạt mọi tiêu chí."""
    bounds = tuple(rubric["length_bounds"])
    flags = [per_sample(r["question"], r["gold"], set(codes_by_shop.get(r["shop"], ())), bounds) for r in rows]
    rates = {c: round(sum(f[c] for f in flags) / max(1, len(flags)), 4) for c in PER_SAMPLE}
    ds = dataset_scores(rows, rubric["style_profile"])
    comps = [rates[c] for c in PER_SAMPLE] + [ds[c] for c in DATASET]
    return {"n": len(rows), "per_sample": rates, "all_pass": round(sum(all(f.values()) for f in flags) / max(1, len(flags)), 4),
            "dataset": ds, "R": round(sum(comps) / len(comps), 4), "flags": flags}


def compare(syn: dict, orig: dict) -> dict:
    """Tỉ lệ so với mốc gốc theo từng thành phần và tổng R."""
    out = {}
    for c in PER_SAMPLE:
        out[c] = round(syn["per_sample"][c] / orig["per_sample"][c], 4) if orig["per_sample"][c] else None
    for c in DATASET:
        out[c] = round(syn["dataset"][c] / orig["dataset"][c], 4) if orig["dataset"][c] else None
    return {"ratio_R": round(syn["R"] / orig["R"], 4), "ratio_by_component": out,
            "below_0_9": sorted(k for k, v in out.items() if v is not None and v < 0.9), "pass": syn["R"] > 0.9 * orig["R"]}


# ---------------------------------------------------------------- dữ liệu gốc (chỉ cục bộ)
def original_rows() -> list[dict]:
    rows = [json.loads(line) for line in (ROOT / "data/synth_source/seeds/questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [{"question": r["question"], "gold": r["gold"], "shop": r["business_id"], "role": r["role"]} for r in rows]


def original_codes(shops) -> dict:
    import synth_checks
    from datamate import loader
    out = {}
    for b in sorted(shops):
        biz = loader.load_cached(ROOT / "data/train/data" / b, ROOT / "outputs/cache")
        out[b] = sorted(synth_checks.codes_of(biz.tables))
    return out


def baseline(out: Path) -> dict:
    """Đo dữ liệu gốc, cố định ngưỡng độ dài (p1–p99 số từ của câu gốc), ghi rubric_v1.json + .md. Chạy một lần trước khi sinh."""
    import pandas as pd
    out.mkdir(parents=True, exist_ok=True)
    rows = original_rows()
    words = sorted(len(r["question"].split()) for r in rows)
    bounds = (words[int(0.01 * (len(words) - 1))], words[int(0.99 * (len(words) - 1))])
    profile = json.loads((ROOT / "data/synth_source/style/style_profile.json").read_text(encoding="utf-8"))
    rubric = {"version": VERSION, "created": dt.datetime.now().isoformat(timespec="seconds"), "length_bounds": list(bounds),
              "style_profile": {"ti_le": profile["ti_le"], "so_tu": profile["so_tu"]},
              "per_sample": list(PER_SAMPLE), "dataset": list(DATASET), "pandas": pd.__version__,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "requirement": "R_syn > 0.9 * R_orig; meaning audit > 0.9; 100% kept rows pass schema/runtime/gold"}
    codes = original_codes({r["shop"] for r in rows})
    res = score(rows, codes, rubric)
    by_role = {role: score([r for r in rows if r["role"] == role], codes, rubric) for role in ("mining", "tune", "holdout")}
    rubric["original"] = {k: v for k, v in res.items() if k != "flags"}
    rubric["original_by_role"] = {k: {kk: vv for kk, vv in v.items() if kk != "flags"} for k, v in by_role.items()}
    failing = collections.Counter(c for f in res["flags"] for c, ok in f.items() if not ok)
    rubric["original_fail_counts"] = dict(failing)
    (out / f"{VERSION}.json").write_text(json.dumps(rubric, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / f"{VERSION}.md").write_text(render_md(rubric), encoding="utf-8")
    return rubric


def render_md(r: dict) -> str:
    o = r["original"]
    lines = [f"# Rubric chung {r['version']} (công bố {r['created']}, trước mọi vòng tối ưu)", "",
             "Áp y hệt cho dữ liệu gốc (3.000 câu train, chỉ đo cục bộ, không in câu) và dữ liệu sinh. Code: `eval/synth_rubric.py`, "
             f"SHA256 `{r['code_sha256'][:16]}…`.", "",
             "## Tiêu chí từng mẫu (đạt/không, trên câu model thấy + đáp án chuẩn)",
             "| Mã | Định nghĩa | Gốc |", "|---|---|---:|"]
    defs = {"period_ok": "câu có kỳ rõ hoặc cách nói kỳ tương đối nhận diện được (PERIODS + EXTRA_PERIODS), trừ câu không trả lời được",
            "no_leak": "câu không chứa sẵn đáp án (synth_checks.lo_dap_an)", "no_hint": "không viết sẵn cách giải (synth_checks.goi_y)",
            "type_consistent": "kiểu đáp án hợp cách hỏi (datamate.verify.check rỗng)", "gold_wellformed": "đáp án qua formatter giữ kiểu và tự khớp score.match",
            "not_zero_suspect": "không phải số 0 đáng ngờ (verify.zero_suspect)", "length_ok": f"số từ trong [{r['length_bounds'][0]}, {r['length_bounds'][1]}] (p1–p99 của câu gốc)",
            "no_internal_code": "không hỏi bằng mã nội bộ có trong bảng của shop (synth_checks.ma_noi_bo)", "no_blocklist": "không có từ LOCAL_BLOCKLIST"}
    lines += [f"| {c} | {defs[c]} | {o['per_sample'][c]:.4f} |" for c in PER_SAMPLE]
    lines += ["", "## Tiêu chí mức tập (0–1)", "| Mã | Định nghĩa | Gốc |", "|---|---|---:|"]
    ddefs = {"style_fidelity": "1 − trung bình |tỉ lệ nét văn phong − style_profile| trên 6 nét",
             "length_fidelity": "1 − |trung vị số từ − 12| / 12", "lexical_diversity": "tỉ lệ 3-gram khác nhau (bỏ dấu, che số) trên mẫu con 400 câu, trung bình 20 lần",
             "uniqueness": "1 − tỉ lệ câu có câu khác cùng shop Jaccard 3-gram ≥ 0,7 (mẫu con ≤25 câu/shop)",
             "answer_type_coverage": "entropy chuẩn hoá của 8 kiểu đáp án", "period_coverage": "entropy chuẩn hoá của nhãn kỳ"}
    lines += [f"| {c} | {ddefs[c]} | {o['dataset'][c]:.4f} |" for c in DATASET]
    lines += ["", f"**R gốc = {o['R']:.4f}** (n = {o['n']}); tỉ lệ đạt mọi tiêu chí từng mẫu: {o['all_pass']:.4f}.",
              f"Ngưỡng dữ liệu sinh: R_sinh > {0.9 * o['R']:.4f} (0,9 × mốc gốc). Thành phần nào < 0,9 × gốc được báo riêng.", "",
              "Theo vai trò shop (chỉ để tham khảo): " + ", ".join(f"{k} R={v['R']:.4f} (n={v['n']})" for k, v in r["original_by_role"].items()), "",
              "Giới hạn: rubric chỉ đo được những gì kiểm bằng code trên câu + đáp án; đúng nghĩa (câu ↔ đáp án) đo riêng bằng audit độc lập trên dữ liệu sinh, "
              "không đo được trên dữ liệu gốc vì không gửi câu BTC đi đâu và không đọc tay câu BTC."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--rows", type=Path, required=True)
    s.add_argument("--codes", type=Path, required=True)
    s.add_argument("--rubric", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "baseline":
        r = baseline(a.out)
        print(json.dumps({"R": r["original"]["R"], "per_sample": r["original"]["per_sample"],
                          "dataset": {k: r["original"]["dataset"][k] for k in DATASET}, "length_bounds": r["length_bounds"]}, ensure_ascii=False))
    else:
        rubric = json.loads(a.rubric.read_text(encoding="utf-8"))
        rows = [json.loads(x) for x in a.rows.read_text(encoding="utf-8").splitlines() if x.strip()]
        res = score(rows, json.loads(a.codes.read_text(encoding="utf-8")), rubric)
        print(json.dumps({k: v for k, v in res.items() if k != "flags"} | {"vs_original": compare(res, rubric["original"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
