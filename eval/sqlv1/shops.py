"""Kế hoạch shop (132 thế giới cũ + shop mới của 5 ngành lõi), dựng thư mục shop, nạp bằng ingest upstream, dựng view/nợ
đúng như runtime trình bày ctx (pipeline.setup_shop), lưu trạng thái để dùng lại.

    python eval/sqlv1/shops.py build --work outputs/synth_sql_v1 [--only syn_shop_0001,...] [--procs 10]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import hashlib
import json
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.up import SYNTH_ROOT, debts, ingest, upstream_commit, views  # noqa: E402
from sqlv1 import render as RD  # noqa: E402
from sqlv1 import world as WB  # noqa: E402
import synth_world as W  # noqa: E402

TRAIN_FAMILIES = ("joined", "flat", "vn_cols", "neg_lines")
CAL = SYNTH_ROOT / "data" / "train" / "calendar.csv"  # lịch âm dương 2024–2027 (chỉ cục bộ)
GEN = "eval/sqlv1/world.py+render.py"


def old_plan() -> list[dict]:
    import synth_corpus as C
    out = []
    for s in C.shop_plan():
        split = {"train": "train", "val": "dev"}.get(s["split"], "final")
        out.append({**s, "old_name": s["name"], "split": split, "old_split": s["split"], "origin": "world_v1"})
    return out


def new_plan() -> list[dict]:
    """Shop mới cho 5 ngành lõi (hạt mới, khoảng dữ liệu dài hơn như BTC: 9–16 tháng, có năm 2025–2026)."""
    rng = random.Random(20261111)
    out, k = [], 0
    counts = {"train": {"tap_hoa": 12, "quan_nhau": 10, "ca_phe": 10, "vlxd": 10, "nha_thuoc": 10}, "dev": {"tap_hoa": 1, "quan_nhau": 1, "ca_phe": 1, "vlxd": 1, "nha_thuoc": 1}}
    for split, per in counts.items():
        for ind, n in per.items():
            for j in range(n):
                fam = TRAIN_FAMILIES[(k + j) % len(TRAIN_FAMILIES)] if split == "train" else "events"
                y0 = rng.choice([2025, 2025, 2024])
                m0 = rng.randint(1, 12)
                start = dt.date(y0, m0, 1)
                span = rng.randint(270, 480)
                today = start + dt.timedelta(days=span)
                seed = 120000 + 61 * k
                out.append({"name": f"new_{split}_{ind}_{fam}_{j}", "industry": ind, "family": fam, "seed": seed, "split": split, "start": start.isoformat(),
                            "today": today.isoformat(), "after_days": rng.choice([0, 0, 0, 2]), "variant": True, "complex": rng.random() < 0.35,
                            "origin": "world_v2"})
                k += 1
    return out


def opts_for(s: dict) -> dict:
    r = random.Random(f"opts:{s['seed']}")
    fam, ind = s["family"], s["industry"]
    o = {"codes_on_orders": fam in ("joined", "vn_cols"), "service_fee": ind in ("quan_nhau", "ca_phe") and r.random() < 0.6,
         "line_discount": r.random() < 0.5 and fam != "neg_lines", "ship_fee": r.random() < 0.5 and fam != "neg_lines",
         "voucher": r.random() < 0.4 and fam != "neg_lines", "note_status": (fam == "monthly" and r.random() < 0.6) or (fam == "flat" and r.random() < 0.3),
         "same_name": fam in ("joined", "vn_cols") and r.random() < 0.3}
    return o


def plan() -> list[dict]:
    shops = old_plan() + new_plan()
    ids = sorted(range(len(shops)), key=lambda i: hashlib.sha256(f"bid:{shops[i]['seed']}:{shops[i]['industry']}".encode()).hexdigest())
    for n, i in enumerate(ids, 1):
        s = shops[i]
        s["business_id"] = f"syn_shop_{n:04d}"
        s["parent_example_id"] = f"latent_world_{s['seed']}"
        s["opts"] = opts_for(s)
        s["style"] = RD.style_for(s["seed"], s["family"], s["split"])
    return shops


def record(s: dict) -> dict:
    old = W.simulate(s["industry"], s["seed"], s["start"], s["today"], s["after_days"], variant=s.get("variant", False), complex_=s.get("complex", False))
    return WB.to_btc(old, s["opts"])


def title_of(s, rec) -> str:
    return f"{rec['shop_name']} ({W.INDUSTRIES[s['industry']]['label']}) — sổ bán hàng"


def load(root: Path, bid: str, written: dict, k_agree: tuple[int, int]):
    """Nạp shop như runtime; dựng view/nợ bằng SELECT tất định; trả (db, meta, layers)."""
    db, meta = ingest.load_business(root, bid, canon=True)
    db.execute("SET enable_external_access = false")
    db.execute("SET memory_limit = '3GB'")
    Nt = RD.resolve_tables(meta, written["files"])
    N = RD.relations(Nt)
    missing = [(k, t, c) for k, cs in written.get("tables", {}).items() if k in Nt for t in Nt[k] for c in cs if c not in set(_cols(db, t))]
    if missing or set(Nt) != set(written["files"]):
        raise ValueError(f"header/table detection: missing {missing[:4]} unresolved {sorted(set(written['files']) - set(Nt))}")
    R = written["roles"]
    layers = {"N": Nt, "view_sql": None, "view_ok": False, "view_err": None, "debt_sqls": None, "debt_ok": False, "debt_err": None}
    try:
        layers["view_sql"] = RD.view_sql(R, N, {k: sorted({c for t in Nt[k] for c in _cols(db, t)}) for k in Nt})
        views._raw.clear()  # cache raw_values của upstream khoá theo id(db) + danh sách cột: shop cùng họ trong một tiến trình trùng khoá
        chk = views.check(db, ["```sql\n" + layers["view_sql"] + "\n```"])[0]
        layers["view_ok"], layers["view_err"] = chk[0] is not None, (None if chk[0] else chk[1])
    except Exception as e:  # noqa: BLE001
        layers["view_err"] = f"{type(e).__name__}: {e}"[:300]
    view_text = ""
    if layers["view_ok"]:
        views.build(db, layers["view_sql"])
        n, k = k_agree
        view_text = views.describe(db, layers["view_sql"], n, k)
        meta["tables"] += [ingest.profile(db, v, "canonical view") for v in ("v_sales", "v_orders")]
    debt_text = ""
    ds = RD.debt_sqls(R, N)
    if ds:
        layers["debt_sqls"] = list(ds)
        try:
            c = debts.check(db, ["```sql\n" + ds[0] + "\n```\n```sql\n" + ds[1] + "\n```"])[0]
            layers["debt_ok"], layers["debt_err"] = c[0] not in (None, "none"), (None if c[0] not in (None, "none") else str(c[1])[:300])
        except Exception as e:  # noqa: BLE001
            layers["debt_err"] = str(e)[:300]
        if layers["debt_ok"]:
            debts.build(db, ds)
            debt_text = debts.describe(*k_agree)
    ctx = "README:\n" + meta["readme"] + "\n\nSCHEMA:\n" + ingest.schema_text(meta) + ("\n\nCANONICAL VIEWS:\n" + view_text if view_text else "") + ("\n\n" + debt_text if debt_text else "")
    layers["ctx"] = ctx
    return db, meta, layers


def _cols(db, name):
    return [r[0] for r in db.execute(f'DESCRIBE "{name}"').fetchall()]


def files_hash(folder: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(folder.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(folder).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def build_one(args) -> dict:
    try:
        return _build_one(args)
    except Exception as e:  # noqa: BLE001
        return {"business_id": args[0]["business_id"], "view_ok": False, "view_err": f"BUILD {type(e).__name__}: {e}"[:400], "debt_ok": False, "debt_err": None,
                "n_lines": None, "n_orders": None, "secs": None, "source_hash": None}


def _build_one(args) -> dict:
    s, work = args
    work = Path(work)
    t0 = time.time()
    root = work / "shops_root"
    bid = s["business_id"]
    rec = record(s)
    folder = root / "data" / bid
    if folder.exists():
        shutil.rmtree(folder)
    written = RD.write_shop(rec, s["family"], s["style"], folder, title_of(s, rec))
    r = random.Random(f"agree:{s['seed']}")
    k_agree = (r.choice([5, 6, 7, 8, 8, 8]), 8)
    db, meta, layers = load(root, bid, written, k_agree)
    rows = db.execute("SELECT count(*), count(DISTINCT order_id) FROM v_sales_all").fetchone() if layers["view_ok"] else (None, None)
    db.close()
    out = {"business_id": bid, "files": written["files"], "roles": written["roles"], "tables": written["tables"], "layers": {k: v for k, v in layers.items() if k != "ctx"},
           "ctx": layers["ctx"], "k_agree": k_agree, "source_hash": files_hash(folder), "n_lines": rows[0], "n_orders": rows[1], "secs": round(time.time() - t0, 1)}
    st = work / "shops" / bid
    st.mkdir(parents=True, exist_ok=True)
    with gzip.open(st / "rec.pkl.gz", "wb") as f:
        pickle.dump(rec, f)
    (st / "shop.json").write_text(json.dumps({**out, "plan": s}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return {k: out[k] for k in ("business_id", "n_lines", "n_orders", "secs", "source_hash")} | {"view_ok": layers["view_ok"], "view_err": layers["view_err"],
                                                                                                "debt_ok": layers["debt_ok"], "debt_err": layers["debt_err"]}


def load_shop(work: Path, bid: str):
    st = work / "shops" / bid
    info = json.loads((st / "shop.json").read_text(encoding="utf-8"))
    with gzip.open(st / "rec.pkl.gz", "rb") as f:
        rec = pickle.load(f)
    return info, rec


def open_db(work: Path, bid: str, info: dict, root: Path | None = None):
    """Nạp lại shop đã dựng (hoặc bản đột biến ở root khác) với cùng cách dựng view/nợ."""
    root = root or (work / "shops_root")
    return load(root, bid, {"files": info["files"], "roles": info["roles"], "tables": info.get("tables", {})}, tuple(info["k_agree"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "build"])
    ap.add_argument("--work", type=Path, default=SYNTH_ROOT / "outputs" / "synth_sql_v1")
    ap.add_argument("--only")
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    shops = plan()
    a.work.mkdir(parents=True, exist_ok=True)
    (a.work / "plan.json").write_text(json.dumps(shops, ensure_ascii=False, indent=1), encoding="utf-8")
    root = a.work / "shops_root"
    (root / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(CAL, root / "calendar.csv")
    if a.cmd == "plan":
        import collections
        print(len(shops), collections.Counter((s["split"], s["origin"]) for s in shops))
        return
    todo = [s for s in shops if not a.only or s["business_id"] in a.only.split(",")]
    res = []
    with cf.ProcessPoolExecutor(a.procs) as ex:
        for r in ex.map(build_one, [(s, str(a.work)) for s in todo]):
            res.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
    (a.work / "build_report.json").write_text(json.dumps({"generator": GEN, "upstream": upstream_commit(), "oracle": WB.ORACLE_VERSION, "shops": res},
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    ok = sum(r["view_ok"] for r in res)
    print(f"{len(res)} shop, view ok {ok}, debt ok {sum(r['debt_ok'] for r in res)}")


if __name__ == "__main__":
    main()
