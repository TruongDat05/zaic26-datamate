"""Sinh mẫu T4 trung gian (giai đoạn 2, việc 2.1 của synth_spec.MODEL_ROUTING) cho các shop đã dựng.

    python eval/gen_data.py --shops data/synth_shops/truth --batches 5 --out outputs/gen/<lượt>
    python eval/gen_data.py --shops data/synth_shops/truth --dry-run      # dựng prompt + kiểm chặn rò dữ liệu, không gọi API

Đầu vào: mỗi shop một thư mục con có tables.pkl (dict tên bảng → DataFrame của lớp sự thật, như outputs/cache; môi trường không có
pyarrow nên không đọc parquet) và rules.md (quy ước tính tự soạn, không chép rulebook BTC: prompt đi qua proxy bên thứ ba); shop thiếu
rules.md dùng --rules. Khoảng ngày lấy từ các cột ngày.
Mỗi lô 12 mẫu (model_bakeoff.t4_batch): teacher soạn → chạy code, kiểm kiểu (sửa một lượt kèm thông báo lỗi) → lớp văn phong cục bộ →
hai model mạnh khác họ + model yếu giải lại song song → giữ mẫu hợp lệ và qua kiểm nghĩa câu viết lại. Chỉ sinh bù số mẫu còn thiếu
với model sau trong danh sách teacher. Bỏ mẫu trùng mẫu đã giữ của cùng shop và câu giống câu tune/holdout (synth_tools.Kit.contaminated).
Ra (--out, mặc định outputs/gen/<giờ>): raw_samples.jsonl (mẫu T4 trung gian, chưa phải dữ liệu fine-tune), raw.jsonl, tong_hop.json.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import pickle
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "eval"), str(ROOT)]
import llm  # noqa: E402
import model_bakeoff as mb  # noqa: E402
from stress_noise import VanPhong  # noqa: E402
from synth_spec import MODEL_ROUTING  # noqa: E402

TEACHER = next(r["model"] for r in MODEL_ROUTING if r["viec"].startswith("2.1"))


def load_shop(d: Path, rules: str | None) -> dict:
    """Thư mục shop → bộ dữ liệu như model_bakeoff.bo(): bảng, quy ước, khoảng ngày."""
    quy_uoc = (d / "rules.md").read_text(encoding="utf-8") if (d / "rules.md").exists() else rules
    if not (d / "tables.pkl").exists() or not quy_uoc:
        raise ValueError(f"{d}: cần tables.pkl và rules.md (hoặc --rules)")
    T = pickle.load(open(d / "tables.pkl", "rb"))
    # Report transaction date columns individually; expiry and other future reference dates
    # must not extend the period in which transactions occurred.
    roles = ("ngày bán", "ngày đặt", "ngày giao dịch", "ngày hđ", "ngày hóa đơn", "ngày hoá đơn", "order_date", "transaction_date", "sale_date", "return_date", "ngày trả", "ngày thu")
    ranges = []
    for name, df in T.items():
        for col in df.columns:
            values = df[col]
            if str(col).casefold() in roles and pd.api.types.is_datetime64_any_dtype(values) and values.notna().any():
                ranges.append(f"{name}.{col}: {values.min():%d/%m/%Y} đến {values.max():%d/%m/%Y}")
    khoang = "Khoảng ngày giao dịch theo cột: " + "; ".join(ranges) + "." if ranges else "Không xác định được khoảng ngày giao dịch; xem mô tả từng cột."

    return {"T": T, "quy_uoc": quy_uoc, "khoang": khoang}


def keep(c: dict, kept: list[dict], kit) -> str | None:
    """Lý do bỏ một mẫu hợp lệ (None = giữ): câu viết lại lệch nghĩa, trùng mẫu đã giữ của shop, giống câu tune/holdout."""
    if not c.get("vp", {}).get("khop"):
        return "vp_lech_nghia"
    key = dict(c, _sh=mb._shingles(c["question_goc"]), _typed=mb.typed(c["raw"], c["loai"], c["question_goc"]))
    if any(mb._trung(key, o) for o in kept):
        return "trung_giua_lo"
    if kit.contaminated(c["question"])["hit"] or kit.contaminated(c["question_goc"])["hit"]:
        return "giong_cau_danh_gia"
    kept.append(key)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shops", required=True, help="thư mục chứa mỗi shop một thư mục con (tables.pkl + rules.md)")
    ap.add_argument("--rules", help="file quy ước dùng chung cho shop không có rules.md")
    ap.add_argument("--teacher", default=",".join(TEACHER), help="model soạn theo thứ tự dùng (model:mức suy nghĩ), mặc định theo MODEL_ROUTING 2.1")
    ap.add_argument("--batches", type=int, default=5, help="số lô 12 mẫu mỗi shop")
    ap.add_argument("--workers", type=int, default=4, help="số lô chạy song song mỗi model soạn")
    ap.add_argument("--out", default=str(ROOT / "outputs/gen" / f"{dt.datetime.now():%Y%m%d_%H%M%S}"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ledger", default=str(llm.DEFAULT_LEDGER), help="shared HTTP attempt ledger, cap 800")
    a = ap.parse_args(argv)
    llm.configure_run(a.ledger, limit=800, provider_limits={p: a.workers for p in llm.PROVIDERS})
    rules = Path(a.rules).read_text(encoding="utf-8") if a.rules else None
    mb.SHOPS.clear()
    mb.bo.cache_clear()
    mb._known.cache_clear()
    mb.SHOPS.update({d.name: load_shop(d, rules) for d in sorted(Path(a.shops).iterdir()) if d.is_dir()})
    teachers = a.teacher.split(",")
    jobs = [{"task": "T4", "sample": k, "key": f"{s}_lo{mb.N_CASES}", "bo": s, "prompt": mb.prompt_t4(mb.bo(s)), "pv": mb.prompt_version(mb.PROMPT8)}
            for s in mb.SHOPS for k in range(a.batches)]
    for j in jobs:
        llm.guard(j["prompt"])
    print(f"{len(mb.SHOPS)} shop × {a.batches} lô = {len(jobs)} lô ({len(jobs) * mb.N_CASES} mẫu yêu cầu, ~6 lần gọi API mỗi lô); "
          f"teacher {teachers}; prompt qua kiểm chặn rò dữ liệu", flush=True)
    if a.dry_run:
        print(jobs[0]["prompt"][:6000] if jobs else "")
        return 0
    from synth_tools import Kit
    kit, out = Kit(), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / "harness.py").write_text(mb.HARNESS, encoding="utf-8")
    for name, b in mb.SHOPS.items():
        pickle.dump(b["T"], open(out / f"{name}.pkl", "wb"))
    vp = VanPhong(seed=0, known=frozenset().union(*(mb._known(n) for n in mb.SHOPS)))
    sem = {t.split(":")[0]: threading.Semaphore(a.workers) for t in teachers}
    lock, kept, rows, bo_qua = threading.Lock(), collections.defaultdict(list), [], collections.Counter()
    raw, train = open(out / "raw.jsonl", "a", encoding="utf-8"), open(out / "raw_samples.jsonl", "a", encoding="utf-8")

    def one(j):
        n = 0
        for t in teachers:  # finite fallback; retain accepted samples from every attempt
            prov, think = t.split(":")
            remaining = mb.N_CASES - n
            attempt = dict(j, provider=prov, think=think, n_req=remaining, prompt=mb.prompt_t4(mb.bo(j["bo"]), n_cases=remaining))
            llm.guard(attempt["prompt"])
            with sem[prov]:
                rec = mb.ban_ghi_t4(mb.t4_batch(attempt, out, vp))
            with lock:
                for c in mb._hop_le(rec):
                    if n >= mb.N_CASES:
                        break
                    why = keep(c, kept[j["bo"]], kit)
                    if why:
                        bo_qua[why] += 1
                    else:
                        n += 1
                        train.write(json.dumps(mb.mau_finetune(c, rec, mb.bo(rec["bo"])), ensure_ascii=False, default=str) + "\n")
                rows.append(rec | {"run": out.name})
                raw.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                raw.flush(), train.flush()
            if n >= mb.N_CASES:
                break
        with lock:
            if n < mb.N_CASES:
                bo_qua["thieu_mau_sau_fallback"] += mb.N_CASES - n
            print(f"{j['bo']} lô {j['sample']}: giữ {n}/{mb.N_CASES}", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers * len(sem)) as ex:
        for f in as_completed([ex.submit(one, j) for j in jobs]):
            f.result()
    raw.close(), train.close()
    res = {"ngay": dt.date.today().isoformat(), "shops": sorted(mb.SHOPS), "teacher": teachers, "lo_moi_shop": a.batches,
           "mau_giu": sum(map(len, kept.values())), "bo_qua": dict(bo_qua), **mb.summarize(rows)}
    (out / "tong_hop.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"giữ {res['mau_giu']} mẫu → {out / 'raw_samples.jsonl'}; bỏ {dict(bo_qua)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
