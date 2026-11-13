"""EDA so sánh syn_sales_sql_v1 với câu MINING của BTC train (CHỈ SỐ GỘP; không in/chép câu BTC).

    python eval/sqlv1/eda.py stats   → outputs/synth_sql_v1/eda/eda_stats.json
    python eval/sqlv1/eda.py notebook → build/synthetic_sql_v1/eda_syn_sales_sql_v1.ipynb (eval/sqlv1/eda_notebook.py, tiếng Anh, đã chạy)
"""
from __future__ import annotations

import collections
import json
import re
import statistics
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

from sqlv1 import package as PK  # noqa: E402
from sqlv1.questions import WORK  # noqa: E402

ROOT = PK.ROOT
OUT = WORK / "eda"
PKG = controller_root() / "build/synthetic_sql_v1"
ABBR = re.compile(r"\b(dthu|bn|bnhiu|nhiu|dc|đc|hnay|hqua|vs|trc|sl|sp|ko|k|z|j|bill|revenue|total|order)\b", re.I)
FILL = re.compile(r"\b(với|cho hỏi|xem giúp|vậy|nhỉ|giùm|cho mình hỏi|ạ|ha|check giùm|hả|coi giùm|nha|cho hoi|xem giup|vay|nhi|gium|nhe)\b", re.I)
TELEX = re.compile(r"\b\w*(?:aa|ee|oo|dd|ow|aw|uw)\w*[sfrxj]?\b|\b\w+[sfrxj]\b")


def accentless(q):
    return not any(unicodedata.decomposition(c) or c in "đĐ" for c in q if c.isalpha() and ord(c) > 127)


def style(qs):
    n = len(qs)
    return {"khong_dau": sum(accentless(q) for q in qs) / n, "co_viet_tat": sum(bool(ABBR.search(q)) for q in qs) / n,
            "co_tu_dem": sum(bool(FILL.search(q)) for q in qs) / n, "chu_thuong_dau": sum(q[:1].islower() for q in qs) / n}


def dist(xs):
    c = collections.Counter(xs)
    n = sum(c.values())
    return {k: round(v / n, 4) for k, v in c.most_common()}


def words(qs):
    w = sorted(len(q.split()) for q in qs)
    return {"p10": w[len(w) // 10], "p50": statistics.median(w), "p90": w[9 * len(w) // 10], "mean": round(statistics.mean(w), 2)}


def stats():
    mining = [json.loads(x) for x in (ROOT / "data/synth_source/seeds/questions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    mining = [r for r in mining if r["role"] == "mining"]
    mq = [r["question"] for r in mining]
    msk = [r for r in mining if r.get("skill_family")]
    m = {"n": len(mining), "industry": dist(r["industry"] if r["industry"] in PK.WB.CORE else "khac" for r in mining),
         "skill": dist(r["skill_family"] for r in msk), "type": dist((r.get("gold") or {}).get("type") for r in mining),
         "period": dist(PK.ptag(q) for q in mq), "words": words(mq), "word_bins": dist(PK.wbin(q) for q in mq), "style": style(mq)}
    sel = [json.loads(x) for x in (WORK / "selected.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    # dòng SFT thật sự được exporter nhận (sft/), không phải ứng viên trước export
    sft = {s: [json.loads(x) for x in (PKG / "sft" / f"{s}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()] for s in ("train", "dev")}
    rows_by = collections.Counter(r["metadata"]["question_id"] for rs in sft.values() for r in rs)
    out = {"mining": m, "splits": {}, "source": "sft/{train,dev}.jsonl (exporter build " + json.loads((PKG / "sft" / "sft_manifest.json").read_text(encoding="utf-8"))["build_id"] + ")"}
    for split in ("train", "dev", "final"):
        xs = [x for x in sel if x["split"] == split and (split == "final" or rows_by[x["id"]])]
        if not xs:
            continue
        qs = [x["question"] for x in xs]
        f = [PK.features(x) for x in xs]
        rows = [rows_by[x["id"]] if split != "final" else 1 for x in xs]
        tgt = PK.mining_targets()
        out["splits"][split] = {
            "n_questions": len(xs), "n_rows": sum(rows), "n_shops": len({x["business_id"] for x in xs}),
            "industry": dist(x["industry"] for x in xs), "industry_core_row_share": round(sum(r for x, r in zip(xs, rows) if x["industry"] in PK.WB.CORE) / max(1, sum(rows)), 4),
            "skill": dist(ff["skill"] for ff in f), "type": dist(ff["type"] for ff in f), "period": dist(ff["period"] for ff in f),
            "word_bins": dist(ff["words"] for ff in f), "words": words(qs), "style": style(qs),
            "family": dist(x["family"] for x in xs), "origin": dist(x["origin"] for x in xs), "slot": dist(x["spec"]["slot"] for x in xs),
            "L1": {d: round(PK.l1(collections.Counter(ff[d] for ff in f), len(f), tgt[d]), 4) for d in PK.DIM_W},
            "exact_dup_question_share": round(1 - len({PK.fold(q) for q in qs}) / len(qs), 4),
        }
    allr = sft["train"] + sft["dev"]
    by_src = collections.Counter(r["metadata"]["solver"]["mode"] + ":" + str(r["metadata"]["solver"]["model_key"]) for r in sft["train"])
    out["solutions"] = {"train_rows_by_source": dict(by_src), "route": dist(r["metadata"]["route"] for r in allr),
                        "mutants_changed_any": round(sum(1 for r in allr if r["metadata"]["mutation"]["discriminating"]) / max(1, len(allr)), 4)}
    acc = collections.defaultdict(collections.Counter)
    for f in (WORK / "solutions").glob("pool_*.jsonl"):
        for x in map(json.loads, f.read_text(encoding="utf-8").splitlines()):
            acc[x["provider"] + ":" + x.get("attempt", "")]["ok" if x.get("ok") else (x.get("stage") or "?").split(":")[0]] += 1
    out["acceptance"] = {k: dict(v) for k, v in acc.items()}
    shops = json.loads((WORK / "build_report.json").read_text(encoding="utf-8"))["shops"]
    out["shops"] = {"n": len(shops), "view_ok": sum(s["view_ok"] for s in shops), "debt_layer": sum(s["debt_ok"] for s in shops),
                    "lines_p50": statistics.median(s["n_lines"] for s in shops if s["n_lines"]), "lines_max": max(s["n_lines"] for s in shops if s["n_lines"])}
    plan = json.loads((WORK / "plan.json").read_text(encoding="utf-8"))
    traps = collections.Counter(k for s in plan for k, v in s["style"].items() if v and k != "xlsx")
    out["shops"]["trap_share"] = {k: round(v / len(plan), 3) for k, v in traps.most_common()}
    btc = json.loads((ROOT / "data/synth_source/data_spec/shop_complexity.json").read_text(encoding="utf-8"))["theo_shop"]
    KEYMAP = {"ten_viet_khac_nhau": "name_variants", "gop_nhieu_sheet_file": "xlsx_sheets", "so_luong_kem_don_vi": "qty_unit", "dong_ten_bao_cao_dau_sheet": "title_rows",
              "tieu_de_hai_tang": "two_level", "dong_trung_cung_don": "dup_rows", "tach_file_theo_phan_hoac_nam": "split_files", "dong_tong_cong": "total_row",
              "o_trong_dong_sau_cua_don": "blank_cont"}
    bt = collections.Counter(KEYMAP.get(t, t) for x in btc.values() for t in x["bay"])
    out["shops"]["btc_trap_share"] = {k: round(v / len(btc), 3) for k, v in bt.most_common()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "eda_stats.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({s: {k: v for k, v in d.items() if k in ("n_questions", "n_rows", "L1", "industry_core_row_share", "words")} for s, d in out["splits"].items()}, ensure_ascii=False, indent=1))


def notebook():
    """The single EDA notebook (English, vs BTC train): eval/sqlv1/eda_notebook.py."""
    from sqlv1 import eda_notebook
    eda_notebook.main()


if __name__ == "__main__":
    {"stats": stats, "notebook": notebook}[sys.argv[1]]()
