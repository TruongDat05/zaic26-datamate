"""Sinh – lọc – sinh bù dữ liệu synthetic đa dạng trên shop tự tạo (synth_world). Phạm vi PILOT: data/synth_shops/ chưa có.

    python eval/synth_corpus.py --work outputs/synth_corpus_20261110 --ledger <ledger> plan          # dựng shop, fixture, kiểm, không gọi API
    python eval/synth_corpus.py --work ... --ledger ... run --phase dev --rounds 1 --shops a,b      # chạy thử
    python eval/synth_corpus.py --work ... --ledger ... run --phase main --stop-at 730               # sinh lặp tới khi đủ quota / chạm ngưỡng
    python eval/synth_corpus.py --work ... --ledger ... run --phase final                           # xác nhận một lần trên final test
    python eval/synth_corpus.py --work ... --ledger ... export                                      # dữ liệu sạch + schema + manifest

Vai trò model (mỗi lần gọi một shop, nhiều mẫu): teacher Gemini chỉ viết câu hỏi từ đặc tả; gold do oracle Python (synth_world.gold)
tính độc lập; verifier GPT (khác họ teacher) chỉ thấy câu đã qua lớp văn phong + mô tả dữ liệu, không thấy gold hay đặc tả;
coder Qwen viết code runtime (ANSWER = dm…) từ câu đã qua văn phong, không nhận gold; coder dự phòng Gemini, cũng không nhận gold.
Mọi lần gọi qua llm.chat (guard bật, ledger chung, trần llm.HARD_CAP cộng dồn). Không gửi dữ liệu BTC, few-shot, log hay Kit.prompt.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import pickle
import random
import re
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval")]
import llm  # noqa: E402
import synth_world as W  # noqa: E402

PROMPT_VERSION = "corpus_v3"  # v3: câu ngắn hơn (6–16 từ, như tin nhắn), slot shop phức tạp; v2: quy ước liệt kê đúng cách ghi trạng thái/khách lẻ  # v2: quy ước liệt kê đúng cách ghi trạng thái/khách lẻ của shop; cấm ngày cứng ở câu tương đối
N_BATCH = 16
EXEC_SEM = threading.BoundedSemaphore(int(os.environ.get("SYNTH_LOCAL_PROCS", "6")))
LOCK = threading.Lock()

# ---------------------------------------------------------------- shop và split (tách trước khi sinh biến thể)
TRAIN_FAMILIES = ("joined", "flat", "vn_cols", "neg_lines")
VAL_FAMILY, FINAL_FAMILY = "events", "monthly"


def shop_plan() -> list[dict]:
    """Danh sách shop cố định: train 4 họ schema × 13 ngành (26 shop), val họ events (6 shop), final họ monthly (6 shop, 2 ngành mới).
    Mỗi shop một hạt riêng → source_hash riêng; không shop/hạt/họ nào nằm ở hai split."""
    inds = list(W.ORIGINAL)  # thứ tự cố định: shop cũ không đổi khi thêm ngành
    shops = []
    rng = random.Random(20261110)
    for i, ind in enumerate(inds):  # mỗi ngành 2 shop train, họ schema xoay vòng
        for j in range(2):
            fam = TRAIN_FAMILIES[(i + j * 2) % len(TRAIN_FAMILIES)]
            shops.append(_shop(ind, fam, 41000 + 17 * len(shops), "train", rng))
    for ind in ("tiem_banh", "vlxd", "my_pham", "gas_nuoc", "nha_thuoc", "thoi_trang"):
        shops.append(_shop(ind, VAL_FAMILY, 52000 + 19 * len(shops), "val", rng))
    for ind in ("thu_cung", "trai_cay", "tap_hoa", "hoa_tuoi", "dien_gia_dung", "ca_phe"):
        shops.append(_shop(ind, FINAL_FAMILY, 93000 + 23 * len(shops), "final", rng))
    # mở rộng trước vòng chính (hạt riêng, không đổi tham số shop đã có): họ schema thứ ba cho mỗi ngành train + 2 shop val
    ext = random.Random(20261111)
    for i, ind in enumerate(inds):
        shops.append(_shop(ind, TRAIN_FAMILIES[(i + 1) % len(TRAIN_FAMILIES)], 61000 + 13 * i, "train", ext))
    for k, ind in enumerate(("quan_nhau", "nha_sach_vpp")):
        shops.append(_shop(ind, VAL_FAMILY, 72000 + 11 * k, "val", ext))
    # mở rộng lần 2 (sau final, mục tiêu ≥3.000 dòng train): CHỈ train, họ train, hạt mới; không đụng val/final
    ext2 = random.Random(20261112)
    for i, ind in enumerate(inds):  # họ schema thứ tư còn thiếu của mỗi ngành gốc
        shops.append(_shop(ind, TRAIN_FAMILIES[(i + 3) % len(TRAIN_FAMILIES)], 84000 + 29 * i, "train", ext2, months=(1, 4, 7)))
    for i, ind in enumerate(W.ADDED_TRAIN):  # 6 ngành mới × 4 họ train
        for j, fam in enumerate(TRAIN_FAMILIES):
            shops.append(_shop(ind, fam, 85000 + 31 * (4 * i + j), "train", ext2, months=(1, 4, 7)))
    for k, ind in enumerate(("tap_hoa", "ca_phe", "nha_thuoc", "vlxd", "quan_nhau", "my_pham", "gas_nuoc", "dien_gia_dung")):
        fam = TRAIN_FAMILIES[ext2.randrange(len(TRAIN_FAMILIES))]  # biến thể: tập con danh mục, nhịp bán khác, mùa khác
        shops.append(_shop(ind, fam, 86000 + 37 * k, "train", ext2, months=(7,), variant=True))
    # shop phức tạp (giá vốn, lịch sử giá, tồn đầu kỳ, nhân viên/ca, kê đơn): thiên về 5 ngành của mining theo tỉ lệ mining
    ext4 = random.Random(20261114)
    k = 0
    for ind, cnt in (("tap_hoa", 6), ("quan_nhau", 5), ("ca_phe", 5), ("vlxd", 3), ("nha_thuoc", 3)):
        for j in range(cnt):
            fam = TRAIN_FAMILIES[(k + j) % len(TRAIN_FAMILIES)]
            sh = _shop(ind, fam, 88000 + 43 * k, "train", ext4, months=(1, 4, 7))
            sh["name"] += f"_c{j}"
            sh["complex"] = True
            shops.append(sh)
            k += 1
    for ind in ("tap_hoa", "nha_thuoc"):
        sh = _shop(ind, VAL_FAMILY, 89000 + 47 * k, "val", ext4, months=(4, 7))
        sh["name"] += "_c"
        sh["complex"] = True
        shops.append(sh)
        k += 1
    # final test mới cho cấu hình B (verifier DeepSeek khi proxy GPT sập): hạt mới, họ monthly, chạy một lần sau khi đóng băng B
    ext3 = random.Random(20261113)
    for k, ind in enumerate(("thu_cung", "trai_cay", "nhac_cu", "cay_canh", "tap_hoa", "my_pham")):
        shops.append(_shop(ind, FINAL_FAMILY, 97000 + 41 * k, "final_b", ext3))
    # final test cho cấu hình C (shop phức tạp + slot mới): hạt mới, họ monthly, chạy một lần sau khi đóng băng C
    ext5 = random.Random(20261115)
    for k, ind in enumerate(("quan_nhau", "ca_phe", "trai_cay", "nha_thuoc")):
        sh = _shop(ind, FINAL_FAMILY, 98000 + 53 * k, "final_c", ext5, months=(4, 7))
        sh["name"] += "_c"
        sh["complex"] = True
        shops.append(sh)
    return shops


def _shop(ind, fam, seed, split, rng, months=(1, 4), variant=False):
    year = 2025
    start_m = rng.choice(list(months))
    today = dt.date(year, start_m + 4, rng.randint(8, 26))
    s = {"name": f"syn_{split}_{ind}_{fam}" + ("_b" if variant else ""), "industry": ind, "family": fam, "seed": seed, "split": split,
         "start": dt.date(year, start_m, 1).isoformat(), "today": today.isoformat(), "after_days": rng.choice([0, 0, 2])}
    if variant:
        s["variant"] = True
    return s


def shop_records(s: dict) -> dict:
    return W.simulate(s["industry"], s["seed"], s["start"], s["today"], s["after_days"], variant=s.get("variant", False), complex_=s.get("complex", False))


def digest(x) -> str:
    b = x if isinstance(x, bytes) else json.dumps(x, ensure_ascii=False, sort_keys=True, default=str).encode()
    return hashlib.sha256(b).hexdigest()


def write_json(p: Path, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


class Shop:
    """Bản ghi + fixture base đã nạp qua loader; fixture đối chứng dựng khi cần."""

    def __init__(self, work: Path, s: dict):
        self.s, self.work = s, work
        self.name, self.today = s["name"], s["today"]
        self.rec = shop_records(s)
        self.source_hash = digest(self.rec)
        self.root = work / "fixtures" / self.name
        self.fx = W.write_fixture(self.root, self.name, self.rec, s["family"])
        self.errors = W.integrity(self.rec, s["family"], self.fx["biz"].tables)
        self._mut, self._mut_lock = {}, threading.Lock()
        self._kit = None
        import synth_checks
        self.codes = synth_checks.codes_of(self.fx["biz"].tables)
        self.doc = W.render(self.rec, s["family"])[1] + "\n" + W.rules_text(self.rec, s["family"])
        words = " ".join([it["name"] + " " + it["group"] for it in self.rec["items"]] + [o["channel"] for o in self.rec["orders"][:200]]).lower()
        from synth_spec import ABBREVIATIONS
        self.known = frozenset(re.findall(r"[a-z]+", W._fold_ascii(words))) | {w for a in ABBREVIATIONS for w in a.split()}

    def kit(self):
        from synth_tools import Kit
        if self._kit is None:
            self._kit = Kit(data=self.root, cache=self.root / "cache")
        return self._kit

    def business_block(self) -> str:
        from datamate import agent
        return agent.business_block(self.fx["biz"])

    def mutated(self, kind: str) -> dict:
        """Fixture đối chứng (bản ghi đã sửa + nạp lại qua loader). Trả {rec, root, name, tables_pkl, kit}."""
        with self._mut_lock:
            if kind in self._mut:
                return self._mut[kind]
            rec = W.mutate(self.rec, kind)
            name = f"{self.name}__{kind}"
            root = self.work / "fixtures_mut" / self.name / kind
            fx = W.write_fixture(root, name, rec, self.s["family"])
            from synth_tools import Kit
            m = {"rec": rec, "root": root, "name": name, "tables_pkl": fx["tables_pkl"], "kit": Kit(data=root, cache=root / "cache")}
            self._mut[kind] = m
            return m


# ---------------------------------------------------------------- đặc tả + quota
# Kỳ: 60% tuyệt đối, 40% tương đối (gốc ~70/30; tăng tương đối vì pilot lỗi TODAY nhiều nhất ở nhóm này)
TIME_KINDS = {"month": 0.38, "day": 0.10, "range": 0.07, "quarter": 0.05, "today": 0.06, "yesterday": 0.05, "this_month": 0.07,
              "last_month": 0.06, "this_week": 0.03, "last_week": 0.05, "last_n": 0.05, "this_year": 0.03}
DIFF_SHARE = {1: 0.2, 2: 0.4, 3: 0.4}
VERIFIER = os.environ.get("SYNTH_VERIFIER", "gpt")  # verifier khác họ teacher (gemini) và không thấy gold; mặc định GPT như cấu hình đã đóng băng
CERT_STANDARD = "corpus_v3+m_target3+key+draft+until+quality"  # v2: thêm phiếu trả trong kỳ (bất biến) vào m_target; chuẩn chứng nhận: cổng v2 + đối chứng nhắm đặc tả (thêm 10/11 sau khi audit thấy mẫu đúng do may)
TRAIN_TARGET = 6400  # 10/11 tối: pool rộng hơn 5.000 để chọn tập bám phân bố mining (slot mới chỉ có ở shop phức tạp);  # 5.000 + dư ~4% cho phần loại ở lần chứng nhận cuối; 10/11: mục tiêu mới ≥3.000, hướng tới 5.000 dòng train sạch (trước đó 1.150)
EDGE_TARGET = {"hoan_cheo_ky": 60, "quy_doi_don_vi": 70, "khach_le_lon": 50, "moc_nua_dem": 120, "co_du_lieu_sau_today": 50,
               "trang_thai_nhieu_cach_ghi": 80}  # trên train; val lấy 20%
SLOT_SHARE = {"revenue": 6, "revenue_item": 4, "revenue_group": 4, "revenue_channel": 3, "revenue_net": 7, "refund_total": 3, "n_orders": 4,
              "n_cancelled": 3, "cancel_rate": 5, "avg_order": 4, "qty_base": 5, "top_items": 4, "best_item_rev": 3, "best_day": 3, "best_month": 2,
              "top_customer": 4, "top_customers": 3, "n_customers": 3, "exists_item_day": 3, "group_table": 3, "unanswerable": 4, "unsold_items": 3,
              "rev_change_pct": 5, "share_group": 4, "days_over": 2, "debt_total": 2, "top_debtor": 1.5, "purchase_money": 2, "lots_expiring": 2,
              # slot shop phức tạp (thêm 10/11 chiều): tỉ trọng tính trên cùng thang, nhắm nhóm lai_nhap_ton, gia_ban, ca, công nợ, nhiều kỳ, đếm có điều kiện
              "shift_rev": 1.4, "shift_share": 1.2, "staff_rev": 0.6, "staff_best": 0.4, "price_on_day": 2.4, "price_change_pct": 2.0, "avg_cost": 1.4,
              "profit_gross": 1.8, "stock_qty": 1.8, "debt_collected": 0.8, "debt_n_cust": 0.6, "debt_cust": 0.8, "g_rev_change": 1.4, "rev_list": 0.8,
              "min_month": 0.6, "avg_daily_rev": 0.8, "orders_over": 1.0, "items_ge": 0.8, "cust_ge_n": 0.8, "rx_rev": 0.8, "rx_share": 1.0}
# kỳ cho phép theo slot (kỳ gắn TODAY dùng TODAY của shop)
SLOT_KINDS = {
    "exists_item_day": ["day", "today", "yesterday"], "best_month": ["quarter", "this_year", "range"],
    "rev_change_pct": ["month", "this_month", "last_week", "last_n"], "debt_total": ["until", "until_today"], "top_debtor": ["until", "until_today"],
    "lots_expiring": ["lots"], "unanswerable": ["month", "last_month", "this_month"], "days_over": ["month", "last_month", "this_month", "range"],
    "best_day": ["month", "last_month", "this_month", "last_week", "range"], "unsold_items": ["day", "yesterday", "this_week", "last_week", "range", "month"],
    "price_on_day": ["day", "today", "yesterday"], "price_change_pct": ["price_range"], "stock_qty": ["until", "until_today"],
    "debt_n_cust": ["until", "until_today"], "debt_cust": ["until", "until_today"], "avg_cost": ["month", "quarter", "last_month", "range_months"],
    "g_rev_change": ["month", "this_month", "last_week", "last_n"], "rev_list": ["quarter", "range_months"], "min_month": ["quarter", "this_year", "range_months"],
    "avg_daily_rev": ["month", "last_month", "last_week", "last_n", "range"], "debt_collected": ["month", "last_month", "this_month", "quarter", "range"],
}
SKILL_FAMILY = {  # slot sinh → nhóm kỹ năng của taxonomy gốc (skill_family của seeds MINING) — dùng cho EDA và chọn mẫu theo phân bố
    "revenue": "doanh_thu", "revenue_item": "doanh_thu", "revenue_net": "doanh_thu", "refund_total": "doanh_thu", "rev_change_pct": "doanh_thu",
    "best_day": "doanh_thu", "best_month": "doanh_thu", "days_over": "doanh_thu", "rev_list": "doanh_thu", "min_month": "doanh_thu",
    "avg_daily_rev": "doanh_thu", "n_orders": "don", "n_cancelled": "don", "cancel_rate": "don", "avg_order": "don", "orders_over": "don",
    "qty_base": "so_luong_xep_hang", "top_items": "so_luong_xep_hang", "best_item_rev": "so_luong_xep_hang", "unsold_items": "so_luong_xep_hang",
    "exists_item_day": "so_luong_xep_hang", "items_ge": "so_luong_xep_hang", "revenue_group": "nhom_kenh_ca", "revenue_channel": "nhom_kenh_ca",
    "share_group": "nhom_kenh_ca", "group_table": "nhom_kenh_ca", "g_rev_change": "nhom_kenh_ca", "shift_rev": "nhom_kenh_ca",
    "shift_share": "nhom_kenh_ca", "purchase_money": "lai_nhap_ton", "lots_expiring": "lai_nhap_ton", "avg_cost": "lai_nhap_ton",
    "profit_gross": "lai_nhap_ton", "stock_qty": "lai_nhap_ton", "price_on_day": "gia_ban", "price_change_pct": "gia_ban",
    "top_customer": "khach", "top_customers": "khach", "n_customers": "khach", "cust_ge_n": "khach", "debt_total": "cong_no",
    "top_debtor": "cong_no", "debt_collected": "cong_no", "debt_n_cust": "cong_no", "debt_cust": "cong_no", "staff_rev": "nhan_vien",
    "staff_best": "nhan_vien", "rx_rev": "nganh", "rx_share": "nganh", "unanswerable": "khong_tra_loi"}
DEFAULT_KINDS = ["month", "day", "range", "quarter", "today", "yesterday", "this_month", "last_month", "this_week", "last_week", "last_n", "this_year"]


def edge_tags(rec: dict, spec: dict) -> list[str]:
    """Tình huống biên mà mẫu thực sự chạm (đo trên dữ liệu, không phải nhãn mong muốn)."""
    tags = []
    if not spec.get("period") or spec["slot"] == "unanswerable":
        return ["thieu_du_lieu"] if spec["slot"] == "unanswerable" else []
    s, e = W.resolve(spec["period"], spec["as_of"])
    if spec["period"]["kind"] in W.RELATIVE:
        tags.append("ky_tuong_doi")
    in_p = lambda t: s <= W._d(t) <= e  # noqa: E731
    orders = {o["order"]: o for o in rec["orders"]}
    if spec["slot"] in ("revenue_net", "refund_total"):
        if any(in_p(r["at"]) and not in_p(orders[r["order"]]["at"]) for r in rec["refunds"]) or \
                any(in_p(orders[r["order"]]["at"]) and not in_p(r["at"]) and W._d(r["at"]) <= W._d(spec["as_of"]) for r in rec["refunds"]):
            tags.append("hoan_cheo_ky")
    times = [W._ts(o["at"]).time() for o in rec["orders"] if W._d(o["at"]) in (s, e)]
    if any(t >= dt.time(23, 58) or t <= dt.time(0, 0, 10) for t in times):
        tags.append("moc_nua_dem")
    if e == W._d(spec["as_of"]) and W._d(rec["data_end"]) > e:
        tags.append("co_du_lieu_sau_today")
    if e == W._d(spec["as_of"]):
        tags.append("ky_den_hom_nay")
    if spec["slot"] in ("qty_base", "top_items"):
        fac = {(u["item"], u["unit"]): u["factor"] for u in rec["units"]}
        if any(fac[(ln["item"], ln["unit"])] > 1 and in_p(orders[ln["order"]]["at"]) for ln in rec["lines"]
               if spec["slot"] == "top_items" or ln["item"] == spec["params"].get("item")):
            tags.append("quy_doi_don_vi")
    if spec["slot"] in ("top_customer", "top_customers", "n_customers"):
        walk = sum(ln["amount"] for ln in rec["lines"] if orders[ln["order"]]["cust"] is None and orders[ln["order"]]["status"] == "done"
                   and in_p(orders[ln["order"]]["at"]))
        tags.append("khach_le_lon" if walk > 0 else "khach_le")
    if spec["slot"] in ("n_orders", "n_cancelled", "cancel_rate", "revenue", "avg_order"):
        st = {o["status_text"] for o in rec["orders"] if in_p(o["at"])}
        if len(st) >= 4:
            tags.append("trang_thai_nhieu_cach_ghi")
    return tags


def difficulty(spec: dict, family: str, tags: list[str]) -> int:
    d = W.SLOTS[spec["slot"]]["hard"]
    if d < 3 and (family in ("events", "neg_lines", "vn_cols", "monthly") or {"hoan_cheo_ky", "quy_doi_don_vi"} & set(tags)) and \
            spec["period"] and spec["period"]["kind"] in W.RELATIVE:
        d += 1
    return d


def sample_spec(rec: dict, shop: dict, slot: str, kind: str, rng: random.Random) -> dict | None:
    """Một đặc tả với tham số ngẫu nhiên; None khi không hợp (thiếu tính năng, kỳ ngoài dữ liệu, đáp án suy biến)."""
    today, start = W._d(shop["today"]), W._d(shop["start"])
    feats = rec["features"]
    need = W.SLOTS[slot]["needs"]
    for f in need:
        if f == "multi_unit":
            if not any(u["factor"] > 1 for u in rec["units"]):
                return None
        elif f == "lots":
            if not feats["lots"] or shop["after_days"]:
                return None
        elif not feats.get(f):
            return None
    period, base, params = None, None, {}
    full_months = []
    y, m = start.year, start.month
    while W._month_end(y, m) < today:
        full_months.append((y, m))
        y, m = W._month_add(y, m, 1)
    if kind == "month":
        if not full_months:
            return None
        y, m = rng.choice(full_months)
        period = {"kind": "month", "y": y, "m": m}
    elif kind == "day":
        d = start + dt.timedelta(days=rng.randint(0, (today - start).days - 2))
        period = {"kind": "day", "d": d.isoformat()}
    elif kind == "range":
        d1 = start + dt.timedelta(days=rng.randint(0, (today - start).days - 15))
        period = {"kind": "range", "d1": d1.isoformat(), "d2": (d1 + dt.timedelta(days=rng.randint(4, 13))).isoformat()}
        if slot == "best_month":
            d1 = dt.date(*full_months[0], 1) if full_months else start
            m2 = full_months[min(len(full_months) - 1, rng.randint(1, 2))] if len(full_months) > 1 else None
            if not m2:
                return None
            period = {"kind": "range", "d1": d1.isoformat(), "d2": W._month_end(*m2).isoformat()}
    elif kind == "quarter":
        qs = [(yy, q) for yy, q in {(y, (m - 1) // 3 + 1) for y, m in full_months} if all((yy, 3 * (q - 1) + k) in full_months for k in (1, 2, 3))]
        if not qs:
            return None
        yy, q = rng.choice(sorted(qs))
        period = {"kind": "quarter", "y": yy, "q": q}
    elif kind == "last_n":
        period = {"kind": "last_n", "n": rng.choice([3, 7, 10, 14, 30])}
    elif kind == "until":
        d = start + dt.timedelta(days=rng.randint(30, (today - start).days - 1))
        period = {"kind": "until", "d": d.isoformat()}
    elif kind == "lots":
        params["n"] = rng.choice([7, 15, 30, 45, 60])
        period = None
    elif kind == "range_months":  # 2–3 tháng trọn liên tiếp (nhiều kỳ)
        if len(full_months) < 2:
            return None
        i = rng.randint(0, len(full_months) - 2)
        j = min(len(full_months) - 1, i + rng.choice([1, 2]))
        period = {"kind": "range", "d1": dt.date(*full_months[i], 1).isoformat(), "d2": W._month_end(*full_months[j]).isoformat()}
    elif kind == "price_range":  # hai ngày quanh một lần đổi giá
        ch = [x for x in rec.get("price_history", []) if W._d(x["from"]) <= today - dt.timedelta(days=1)]
        if not ch:
            return None
        x = rng.choice(ch)
        c = W._d(x["from"])
        d1 = max(start, c - dt.timedelta(days=rng.randint(3, 25)))
        d2 = min(today - dt.timedelta(days=1), c + dt.timedelta(days=rng.randint(0, 20)))
        if not d1 < c <= d2:
            return None
        params["item"] = x["item"]
        period = {"kind": "range", "d1": d1.isoformat(), "d2": d2.isoformat()}
    else:
        period = {"kind": kind}
    if period and period["kind"] not in ("until", "until_today"):
        s, e = W.resolve(period, today)
        if s < start or e > today:
            return None
    items = rec["items"]
    if slot in ("revenue_item", "exists_item_day"):
        params["item"] = rng.choice(items)["item"]
    if slot == "qty_base":
        multi = sorted({u["item"] for u in rec["units"] if u["factor"] > 1})
        params["item"] = rng.choice(multi) if rng.random() < 0.85 else rng.choice(items)["item"]
    if slot in ("revenue_group", "share_group"):
        params["group"] = rng.choice(sorted({it["group"] for it in items}))
    if slot == "revenue_channel":
        params["channel"] = rng.choice(W.INDUSTRIES[rec["industry"]]["channels"])
    if slot in ("top_items", "top_customers"):
        params["k"] = rng.choice([2, 3, 3, 4, 5])
    if slot == "unanswerable":
        params["missing"] = rng.choice(W.INDUSTRIES[rec["industry"]]["missing"])
    if slot == "price_on_day":
        moved = sorted({x["item"] for x in rec.get("price_history", [])})
        params["item"] = rng.choice(moved) if moved and rng.random() < 0.8 else rng.choice(items)["item"]
    if slot in ("avg_cost", "stock_qty"):
        bought = sorted({x["item"] for x in rec["purchases"]})
        if not bought:
            return None
        params["item"] = rng.choice(bought)
    if slot in ("shift_rev", "shift_share"):
        params["shift"] = rng.choice(rec["shifts"])["shift"]
    if slot == "staff_rev":
        params["staff"] = rng.choice(rec["staff"])
    if slot == "g_rev_change":
        params["group"] = rng.choice(sorted({it["group"] for it in items}))
    if slot == "items_ge" and rng.random() < 0.5:
        params["group"] = rng.choice(sorted({it["group"] for it in items}))
    if slot == "cust_ge_n":
        params["n"] = rng.choice([2, 3, 3, 4])
    if slot == "debt_cust":
        e0 = W.resolve(period, today)[1]
        owing = sorted(n for n, x in _debts(rec, today, e0).items() if x > 0)
        if not owing:
            return None
        params["customer"] = rng.choice(owing)
    if slot in ("rev_change_pct", "g_rev_change"):
        base = {"month": None, "this_month": {"kind": "prev_same_days"}, "last_week": {"kind": "week_before_last"},
                "last_n": {"kind": "prev_n", "n": period.get("n") if period else None}}[kind]
        if kind == "month":
            py, pm = W._month_add(period["y"], period["m"], -1)
            if dt.date(py, pm, 1) < start:
                return None
            base = {"kind": "month", "y": py, "m": pm}
        bs, _ = W.resolve(base, today)
        if bs < start:
            return None
    spec = {"slot": slot, "period": period, "base_period": base, "params": params, "as_of": shop["today"]}
    if slot in ("orders_over", "items_ge"):
        s, e = W.resolve(period, today)
        v = W.View(rec, today)
        tot = collections.Counter()
        for o, ln in v.done_lines(s, e):
            tot[o["order"] if slot == "orders_over" else ln["item"]] += ln["amount"] if slot == "orders_over" else v.base_qty(ln)
        vals = sorted(tot.values())
        if len(vals) < 4:
            return None
        raw = vals[int(len(vals) * rng.uniform(0.5, 0.9))]
        mag = 10 ** max(0 if slot == "items_ge" else 3, len(str(raw)) - 2)
        val = max(1, int(round(raw / mag) * mag))
        params["threshold" if slot == "orders_over" else "n"] = val
    if slot == "days_over":
        s, e = W.resolve(period, today)
        v = W.View(rec, today)
        daily = collections.Counter()
        for o, ln in v.done_lines(s, e):
            daily[W._d(o["at"])] += ln["amount"]
        vals = sorted(daily.values())
        if len(vals) < 4:
            return None
        raw = vals[int(len(vals) * rng.uniform(0.35, 0.8))]
        mag = 10 ** max(3, len(str(raw)) - 2)
        params["threshold"] = int(round(raw / mag) * mag)
    try:
        g = W.gold(rec, spec)
    except W.Degenerate:
        return None
    a = g.get("answer")
    if g["type"] == "number" and a == 0 and slot not in ("unsold_items",):
        return None
    if slot == "rev_list" and any(x == 0 for x in a):
        return None
    if slot == "unsold_items" and a == 0 and rng.random() < 0.7:
        return None
    if slot == "days_over" and (a == 0 or a == len(daily)):
        return None
    spec["gold"] = g
    spec["tags"] = edge_tags(rec, spec)
    spec["difficulty"] = difficulty(spec, shop["family"], spec["tags"])
    spec["key"] = digest([shop["name"], slot, period, base, params])[:16]
    return spec


def _debts(rec, today, d):
    """Nợ còn lại theo tên khách đến hết ngày d (dùng chọn tham số; gold vẫn do synth_world.gold tính)."""
    spec = {"slot": "debt_total", "period": {"kind": "until", "d": W._d(d).isoformat()}, "params": {}, "as_of": W._d(today).isoformat()}
    v = W.View(rec, today)
    out = collections.Counter()
    for o in v.orders:
        if o["status"] == "done" and o["payment"] == "Ghi nợ" and W._d(o["at"]) <= W._d(d):
            out[v.cust[o["cust"]]["name"]] += sum(ln["amount"] for ln in v.lines[o["order"]])
    for x in rec["payments"]:
        if W._d(x["at"]) <= W._d(d):
            out[v.cust[x["cust"]]["name"]] -= x["amount"]
    return out


def time_kind(spec) -> str:
    return "lots" if spec["slot"] == "lots_expiring" else spec["period"]["kind"] if spec["period"] else "none"


def spec_text(rec: dict, spec: dict) -> dict:
    """Đặc tả đưa cho teacher: tên thật (không mã), cách nói kỳ, kiểu đáp án; không có đáp án."""
    it = {i["item"]: i for i in rec["items"]}
    p = dict(spec["params"])
    fill = {"item_name": it[p["item"]]["name"] if "item" in p else "", "base": it[p["item"]]["base"] if "item" in p else "",
            "group": p.get("group", ""), "channel": p.get("channel", ""), "k": p.get("k", ""), "missing": p.get("missing", ""),
            "n": p.get("n", ""), "threshold_text": f"{p['threshold']:,}".replace(",", ".") if "threshold" in p else "",
            "period": W.period_text(spec["period"], spec["as_of"]) if spec["period"] else "",
            "base_text": W.period_text(spec["base_period"], spec["as_of"]) if spec.get("base_period") else "",
            "shift": p.get("shift", ""), "staff": p.get("staff", ""), "customer": p.get("customer", ""),
            "group_text": f"thuộc nhóm {p['group']} " if "group" in p else "",
            "d1_text": "", "d2_text": ""}
    if spec["slot"] == "price_change_pct":
        f2 = lambda d: f"{W._d(d).day}/{W._d(d).month}/{W._d(d).year}"  # noqa: E731
        fill["d1_text"], fill["d2_text"] = f2(spec["period"]["d1"]), f2(spec["period"]["d2"])
    measure = W.SLOTS[spec["slot"]]["measure"].format(**fill)
    out = {"so_do": measure, "ky": fill["period"] if spec["period"] else ("trong " + str(p.get("n")) + " ngày tới tính từ hôm nay"),
           "kieu_dap_an": W.SLOTS[spec["slot"]]["type"]}
    if spec["slot"] == "best_month":
        out["kieu_dap_an"] = "date (tháng nào)"
    if spec["slot"] == "best_day":
        out["kieu_dap_an"] = "date (ngày nào)"
    if spec["slot"] == "unanswerable":
        out["kieu_dap_an"] = "unanswerable (hỏi tự nhiên, cửa hàng không có số liệu này)"
    if spec["slot"] in ("debt_total", "top_debtor"):
        out["ky"] = fill["period"]
    if spec["slot"] == "price_change_pct":
        out["ky"] = f"hai ngày {fill['d1_text']} và {fill['d2_text']} (nêu đủ cả hai ngày)"
    if spec["slot"] in ("min_month",):
        out["kieu_dap_an"] = "date (tháng nào)"
    if spec["slot"] == "rev_list":
        out["kieu_dap_an"] = "list (các số tiền theo thứ tự tháng)"
    return out


# ---------------------------------------------------------------- prompt (tự soạn; không chứa dữ liệu BTC, few-shot hay Kit.prompt)
PITFALLS = """LƯU Ý KỸ THUẬT (lỗi đã gặp, tránh ngay từ đầu):
- Code là đáp án mẫu để học: KHÔNG để biến hay khối tính không dùng (ví dụ tự tính phần trăm rồi lại trả dm.pct_change), không gán một biến hai lần, không chú thích phỏng đoán về dữ liệu ('Giả sử…', 'Trong dữ liệu này…'), không chú thích tiếng Anh ghi chú phụ; với kỳ tương đối không ghi cứng năm (lấy năm từ TODAY hoặc từ kỳ).
- Kỳ thời gian tính trọn ngày cuối: lọc `cot >= pd.Timestamp(ngay_dau)` và `cot < pd.Timestamp(ngay_cuoi) + pd.Timedelta(days=1)`. KHÔNG viết `cot <= pd.Timestamp(ngay_cuoi)` (mất các giờ trong ngày cuối).
- Câu nói 'hôm nay', 'hôm qua', 'tháng này', 'tháng trước', 'tuần này', 'tuần trước', 'N ngày qua', 'năm nay', 'đến nay' phải tính mốc từ biến TODAY (ví dụ dm.this_week(TODAY), dm.last_n_days(n, TODAY), TODAY.replace(day=1)); KHÔNG viết ngày cụ thể tự suy ra từ hôm nay (như dt.date(2025, 5, 12) hay '2025-05-01'), không chép ngày hôm nay thành chuỗi hay số. Bỏ mọi bản ghi sau hết ngày TODAY.
- Hoàn tiền / trả hàng: đọc README để biết số tiền hoàn ghi dương hay âm (đã âm thì CỘNG, dương thì TRỪ). Lấy tiền hoàn theo thời điểm hoàn nằm trong kỳ, KHÔNG lọc theo việc đơn gốc có thuộc kỳ hay không.
- Số 'tính đến' một ngày (công nợ, tồn kho): cộng dồn MỌI bản ghi từ đầu dữ liệu đến hết ngày đó; KHÔNG tự đặt ngày bắt đầu (dữ liệu có thể có bản ghi cũ hơn).
- Nối bảng bằng `merge` hoặc `Series.map` theo mã (index duy nhất). KHÔNG gán `df['x'] = bang_khac.set_index(k).loc[df[k], 'cot']` (index trùng/lệch thứ tự gây lỗi hoặc sai).
- Trạng thái và tên khách lẻ ghi nhiều kiểu: xem đủ mọi giá trị trong mô tả bảng trước khi lọc. Đếm đơn theo mã đơn duy nhất, không đếm dòng.
- Không dùng thuộc tính bắt đầu bằng '__' (như __round__), getattr, eval, open, đọc/ghi file."""


def teacher_prompt(sh: Shop, specs: list[dict]) -> str:
    rec = sh.rec
    t = W._d(sh.today)
    brief = W.shop_brief(rec)
    lines = [f"Bạn soạn câu hỏi tiếng Việt mà chủ một cửa hàng nhỏ ở Việt Nam hỏi về dữ liệu bán hàng của chính mình. Cửa hàng và dữ liệu đều là giả lập tự tạo.",
             f"Cửa hàng: {rec['shop_name']} ({W.INDUSTRIES[rec['industry']]['label']}). Hôm nay là {W.WEEKDAY_VN[t.weekday()]} ngày {t.day}/{t.month}/{t.year}.",
             "Mặt hàng (tên, nhóm, đơn vị gốc, đơn vị lớn): " + "; ".join(f"{i['ten']} ({i['nhom']}, {i['don_vi_goc']}" + (", " + "/".join(i['don_vi_lon']) if i['don_vi_lon'] else "") + ")" for i in brief["items"]),
             "Kênh bán: " + ", ".join(brief["kenh_ban"]) + ".", "",
             f"Với MỖI đặc tả dưới đây viết đúng MỘT câu hỏi (câu gốc). Yêu cầu:",
             "- Nghĩa đúng hệt đặc tả: cùng số đo, cùng đối tượng (giữ nguyên tên hàng/nhóm/kênh), cùng kỳ, cùng kiểu đáp án; không thêm hay bớt điều kiện.",
             "- Kỳ: dùng đúng cách nói trong 'ky' (có thể đổi trật tự từ). Kỳ có ngày/tháng/quý cụ thể phải ghi đủ năm. Kỳ tương đối (hôm nay, tháng này, tuần trước, N ngày qua…) giữ nguyên là tương đối, không đổi thành ngày cụ thể; bỏ phần giải thích trong ngoặc.",
             "- Viết tự nhiên như chủ tiệm nhắn hỏi nhân viên, tiếng Việt có dấu đầy đủ, ngắn gọn 6–16 từ (chỉ dài hơn khi bắt buộc nêu hai ngày); "
             "đa dạng cách mở đầu và cấu trúc giữa các câu: có câu bắt đầu bằng kỳ, có câu bắt đầu bằng số đo, có câu ra lệnh ngắn ('tính giúp…', 'cho xem…'), không lặp một khuôn.",
             "- Không nhắc lại quy ước tính (không viết 'không tính khách lẻ', 'chỉ tính đơn hoàn thành', 'quy đổi về đơn vị gốc', 'làm tròn', 'trừ đơn hủy'); không nêu tên cột, tên bảng, mã hàng, mã khách, mã đơn; không đưa con số đáp án vào câu.",
             "- Kiểu đáp án: number → hỏi 'bao nhiêu'; percent → hỏi 'bao nhiêu phần trăm' hoặc 'tỷ lệ … là bao nhiêu %'; date (ngày nào) → hỏi 'ngày nào'; date (tháng nào) → hỏi 'tháng nào';"
             " boolean → 'có … không'; string → '… nào' (một đối tượng); list → nêu rõ số lượng k ('top 3', '3 … nào'); table → yêu cầu liệt kê từng nhóm kèm số; unanswerable → hỏi tự nhiên về thông tin trong đặc tả.",
             "- Câu hỏi số lượng phải nói rõ đơn vị gốc (ví dụ 'bao nhiêu viên', 'bao nhiêu kg'), không cần nói cách quy đổi.",
             f"Trả về đúng một JSON: [{{\"i\": <số đặc tả>, \"question\": \"<câu hỏi>\"}}, ...] đủ {len(specs)} phần tử theo thứ tự, không giải thích.", "", "ĐẶC TẢ:"]
    lines.append(json.dumps([{"i": i, **spec_text(rec, s)} for i, s in enumerate(specs, 1)], ensure_ascii=False, indent=0))
    return "\n".join(lines)


def solver_prompt(sh: Shop, questions: list[str], runtime: bool, today: str | None = None) -> str:
    """Verifier (runtime=False, gán `answer`) và coder (runtime=True, gán ANSWER = dm.…). Chỉ câu hỏi + mô tả dữ liệu tự tạo + quy ước."""
    t = today or sh.today
    qs = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    if runtime:
        env = ("Môi trường chạy code có sẵn (không import lại, không đọc file): T (dict tên bảng → DataFrame; cột ngày đã là datetime64, cột số đã là số), "
               "TODAY (datetime.date, hôm nay của câu hỏi), pd, np, dt (module datetime), calendar, math, re, dm.\n"
               "dm có: kỳ thời gian dm.day(d), dm.yesterday(TODAY), dm.date_range(d1, d2), dm.month(y, m), dm.quarter(y, q), dm.this_month(TODAY), "
               "dm.last_month(TODAY), dm.this_week(TODAY), dm.last_week(TODAY), dm.last_n_days(n, TODAY) (n ngày kể cả hôm nay), dm.year_to_date(TODAY), "
               "dm.until(d), dm.previous(p), dm.prev_month_same_days(TODAY); mỗi kỳ p có p.start, p.end (datetime.date) và p.mask(cot_ngay) → Series bool lấy trọn ngày cuối. "
               "Làm sạch: dm.norm_status(cot) → 'hoan_thanh'|'huy'|'tra_hang'|'dang_giao'|'', dm.is_walkin(cot_ten_khach) → True với khách lẻ/vãng lai/để trống, "
               "dm.eq(cot, ten), dm.match_one(danh_sach_ten, ten_trong_cau). "
               "Tạo đáp án (bắt buộc dùng cho ANSWER): dm.money(x) tiền VND (làm tròn đồng, nửa lên), dm.count(x), dm.number(x), dm.percent(x) với x là số phần trăm, "
               "dm.pct_change(moi, cu), dm.share(phan, tong), dm.date(d) → 'YYYY-MM-DD', dm.month_answer(d) → 'YYYY-MM', dm.boolean(x), dm.string(ten), "
               "dm.list_([...]), dm.table(columns, rows), dm.unanswerable('lý do') khi dữ liệu không có thông tin câu hỏi cần.")
        out = (f"Viết đúng {len(questions)} khối ```python``` theo thứ tự câu; khối thứ k mở đầu bằng dòng `# cau k`. Mỗi khối là một chương trình độc lập "
               "(chạy riêng, không dùng biến hay import của khối khác), tự tính từ T, gán ANSWER = dm.<kiểu>(...) ở cấp ngoài cùng. Không ghi sẵn đáp án, "
               "không tạo dữ liệu giả, không gán lại T. Bảng dùng dm.table với cột đầu là tên đối tượng. Tối đa 40 dòng mỗi khối, comment ngắn.")
    else:
        env = ("Biến có sẵn: T (dict tên bảng → DataFrame; cột ngày đã là datetime64, cột số đã là số), pd, np, dt (module datetime), "
               "TODAY (datetime.date). Không đọc file, không in bảng.")
        out = (f"Viết đúng {len(questions)} khối ```python``` theo thứ tự câu; khối thứ k mở đầu bằng dòng `# cau k`; mỗi khối tự tính lại từ T, không dùng biến "
               "của khối khác, gán kết quả cuối vào biến `answer`: số tiền, số lượng, số đơn là số (tiền làm tròn đồng); phần trăm là số phần trăm (12.5 nghĩa "
               "là 12,5%); một tên là chuỗi; nhiều tên là list theo thứ tự câu hỏi yêu cầu; ngày là chuỗi 'YYYY-MM-DD', tháng là 'YYYY-MM'; câu có/không là "
               "True/False; bảng là dict {'columns': [...], 'rows': [[...], ...]} với cột đầu là tên đối tượng; câu không trả lời được từ dữ liệu thì answer = None.")
    return (f"Bạn viết code Python (pandas) trả lời câu hỏi của chủ cửa hàng về dữ liệu bán hàng của họ (dữ liệu giả lập tự tạo).\n{env}\n\n"
            f"{sh.business_block()}\n\n## QUY ƯỚC TÍNH CỦA CỬA HÀNG\n{W.rules_text(sh.rec, sh.s['family'])}\n{PITFALLS}\n\n"
            f"Hôm nay (TODAY) = {t}.\nCâu hỏi (có thể viết tắt, không dấu, gõ sai chính tả):\n{qs}\n\n{out}")


def repair_prompt(sh: Shop, items: list[dict]) -> str:
    """Sửa khối code runtime lỗi chạy: chỉ gửi câu hỏi, code cũ và thông báo lỗi; không gửi đáp án."""
    parts = []
    for k, it in enumerate(items, 1):
        parts.append(f"## Khối {k}\nCâu hỏi: {it['question']}\nCode cũ:\n```python\n{it['code']}\n```\nLỗi khi chạy:\n{it['error']}\n")
    return (solver_prompt(sh, [it["question"] for it in items], True).split("Câu hỏi (có thể viết tắt")[0]
            + "Các khối code dưới đây chạy bị lỗi. Viết lại từng khối cho đúng (giữ nghĩa câu hỏi), mỗi khối độc lập, gán ANSWER = dm.<kiểu>(...).\n\n"
            + "\n".join(parts) + f"\nTrả về đúng {len(items)} khối ```python```, khối thứ k mở đầu bằng dòng `# cau k` (k là số khối ở trên).")


# ---------------------------------------------------------------- gọi API: cache theo nội dung prompt, ledger chung
def ledger_count(path: Path) -> int:
    with sqlite3.connect(path) as db:
        return db.execute("select count(*) from requests").fetchone()[0]


BREAKER = collections.Counter()  # lỗi mạng/API liên tiếp theo nhà cung cấp trong vòng hiện tại
BREAKER_LIMIT = 2


def model_call(work: Path, role: str, shop: str, provider: str, prompt: str, *, think="off", max_tokens=20000) -> tuple[dict, bool]:
    settings = {"provider": provider, "model": llm.config(provider)["model"], "think": think, "max_tokens": max_tokens, "temperature": 0}
    key = digest({"prompt": prompt, "settings": settings})
    path = work / "api" / role / shop / f"{key[:20]}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["response"], True
    if BREAKER[provider] >= BREAKER_LIMIT:  # cầu dao: nhà cung cấp đang treo/lỗi → không gửi thêm, mẫu chờ vòng sau
        return {"content": "", "finish": "api_error", "error": "breaker_open", "usage": {}}, True
    try:
        r = llm.chat(provider, prompt, think=think, max_tokens=max_tokens, temperature=0, retries=1, timeout=420 if provider == "gemini" else 240,
                     tag=f"corpus:{role}:{shop}", check_leak=True)
    except (llm.LeakError, llm.BudgetExceeded):
        raise
    except Exception as e:  # noqa: BLE001 — lỗi mạng/API: ghi lại, mẫu chờ vòng sau (không cache lỗi)
        with LOCK:
            BREAKER[provider] += 1
        return {"content": "", "finish": "api_error", "error": str(e)[:300], "usage": {}}, False
    with LOCK:
        BREAKER[provider] = 0
    write_json(path, {"role": role, "shop": shop, "prompt": prompt, "prompt_sha256": digest(prompt), "settings": settings,
                      "response": r, "ts": dt.datetime.now().isoformat(timespec="seconds")})
    return r, False


# ---------------------------------------------------------------- chạy code cục bộ
VHARNESS = """import sys, json, pickle, runpy, datetime as dt
import numpy as np, pandas as pd
T = pickle.load(open(sys.argv[1], "rb"))
code = open(sys.argv[2], encoding="utf-8").read()
restrict = runpy.run_path(sys.argv[3])["restrict_process"]
env = {"T": T, "pd": pd, "np": np, "dt": dt, "datetime": dt, "TODAY": dt.date.fromisoformat(sys.argv[4])}
restrict()
exec(compile(code, "<code>", "exec"), env)
a = env.get("answer")
if isinstance(a, (pd.Series, pd.Index, np.ndarray)):
    a = list(a)
if isinstance(a, (pd.Timestamp, dt.datetime)):
    a = a.strftime("%Y-%m-%d")
elif isinstance(a, dt.date):
    a = a.isoformat()
if hasattr(a, "item"):
    a = a.item()
if isinstance(a, list):
    a = [x.item() if hasattr(x, "item") else x for x in a]
if isinstance(a, dict) and isinstance(a.get("rows"), list):
    a = {"columns": [str(c) for c in a.get("columns", [])], "rows": [[x.item() if hasattr(x, "item") else x for x in r] for r in a["rows"]]}
print(json.dumps({"answer": a}, ensure_ascii=False, default=str))
"""


def run_verifier(code: str | None, tables_pkl: str, as_of: str, work: Path) -> tuple[object, str | None]:
    from synth_execution import preflight, child_environment
    if not code:
        return None, "missing_code"
    if problem := preflight(code):
        return None, "unsafe_code: " + problem
    # thư mục tạm trên đĩa cục bộ: trên NFS, xoá file còn mở để lại .nfsXXXX làm rmdir hỏng
    base = Path(os.environ.get("SYNTH_EXEC_TMP") or tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v_", dir=base, ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)
        (tmp / "code.py").write_text(code, encoding="utf-8")
        (tmp / "h.py").write_text(VHARNESS, encoding="utf-8")
        try:
            with EXEC_SEM:
                p = subprocess.run([sys.executable, "-I", str(tmp / "h.py"), str(Path(tables_pkl).resolve()), str(tmp / "code.py"),
                                    str(ROOT / "eval/synth_execution.py"), as_of], capture_output=True, text=True, timeout=90, cwd=tmp,
                                   env=child_environment())
        except subprocess.TimeoutExpired:
            return None, "timeout"
        if p.returncode != 0:
            return None, "error: " + ((p.stderr.strip().splitlines() or ["?"])[-1])[:200]
        try:
            return json.loads(p.stdout.strip().splitlines()[-1])["answer"], None
        except (ValueError, KeyError, IndexError):
            return None, "unreadable_output"


def run_runtime(kit, name: str, code: str | None, as_of: str, question: str) -> dict:
    from synth_execution import preflight
    if not code:
        return {"ok": False, "error": "missing_code", "answer": None, "issues": [], "zero_suspect": False}
    if issue := preflight(code):
        return {"ok": False, "error": "unsafe_code: " + issue, "answer": None, "issues": [], "zero_suspect": False}
    with EXEC_SEM:
        return kit.execute(name, code, as_of, question=question, timeout=90)


def check_answer(ans: dict | None, gold: dict) -> bool:
    from synth_tools import Kit
    return bool(ans) and Kit.check(ans, gold) == 1


def verifier_pass(raw, err, gold: dict, question: str) -> tuple[bool, str]:
    import model_bakeoff as mb
    if err:
        return False, "crash"
    typed = mb.typed(raw, gold["type"], question)
    if typed is None:
        return False, "invalid"
    return check_answer(typed, gold), "ok"


# ---------------------------------------------------------------- kiểm tĩnh code + câu hỏi
def _date_literals(d: str) -> list[str]:
    x = W._d(d)
    return [x.isoformat(), f"{x.day:02d}/{x.month:02d}/{x.year}", f"{x.day}/{x.month}/{x.year}"]


DRAFT = re.compile(r"#\s*(?:sai\b|[^\n]*\bnhầm\b|[^\n]*placeholder|[^\n]*will fix|[^\n]*không phải số|[^\n]*cần (?:tính|lấy) tổng)", re.I)
PROVIDED_IMPORTS = {"import pandas as pd", "import numpy as np", "import datetime as dt", "import calendar", "import math", "import re", "import datetime"}


def strip_provided_imports(code: str) -> str:
    """Bỏ dòng import trùng biến runtime đã có sẵn (pd, np, dt, calendar, math, re): đích học gọn như prompt runtime dặn."""
    lines = code.strip().splitlines()
    keep = [ln for ln in lines if ln.strip() not in PROVIDED_IMPORTS or (ln.strip() == "import datetime" and "datetime." in code.replace("dt.datetime", ""))]
    return "\n".join(keep).strip()


def today_shift(spec: dict) -> str:
    """TODAY mới cho đối chứng m_today: lùi qua ranh giới kỳ (tháng/tuần) để code ghi cứng mốc kỳ bị lộ."""
    t = W._d(spec["as_of"])
    k = spec["period"]["kind"] if spec.get("period") else "lots"
    days = {"this_month": t.day + 2, "last_month": t.day + 2, "this_week": t.weekday() + 3, "last_week": t.weekday() + 3}.get(k, 9)
    return (t - dt.timedelta(days=days)).isoformat()


JUNK_COMMENT = re.compile(r"#\s*ponytail\b", re.I)
SPECULATIVE = re.compile(r"#[^\n]*(?:\bgiả sử\b|\bgiả định\b|(?<!có )\btrong dữ liệu(?: này)?\b|\bhoặc đơn giản hơn\b|\bsửa\s*:|\bkiểm tra xem\b"
                         r"|\bkiểm tra\s*:|\bcần kiểm tra\b|\btodo\b|\bfixme\b|\bassum)", re.I)


def _strip_comments(code: str) -> str:
    import io
    import tokenize
    try:
        return tokenize.untokenize(t for t in tokenize.generate_tokens(io.StringIO(code).readline) if t.type != tokenize.COMMENT)
    except (tokenize.TokenError, IndentationError):
        return code


def quality_issues(code: str, spec: dict) -> list[str]:
    """Đích học phải sạch (audit giai đoạn 2, 10/11/2026): không chú thích rác/suy đoán dữ liệu, không code thừa (biến gán không đọc,
    gán đè trước khi đọc, if/else hai nhánh trùng), không ghi cứng năm khi kỳ tương đối (sai khi TODAY sang năm khác)."""
    import ast
    out = []
    if JUNK_COMMENT.search(code):
        out.append("junk_comment")
    if SPECULATIVE.search(code):
        out.append("speculative_comment")
    tree = ast.parse(code)
    names = lambda node, ctx: {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ctx)}  # noqa: E731
    stores = {t.id for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)}
    dead = bool(stores - names(tree, ast.Load) - {"ANSWER", "_"})

    def blocks(body):
        yield body
        for n in body:
            for f in ("body", "orelse", "finalbody"):
                sub = getattr(n, f, None)
                if isinstance(sub, list) and sub and isinstance(sub[0], ast.stmt):
                    yield from blocks(sub)
    for blk in blocks(tree.body):
        last = {}
        for i, st in enumerate(blk):
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                nm = st.targets[0].id
                if nm in last and nm not in names(st.value, ast.Load) and not any(nm in names(b, ast.Load) for b in blk[last[nm] + 1:i]):
                    dead = True
                last[nm] = i
            else:
                for nm in names(st, ast.Load):
                    last.pop(nm, None)
                for nm in names(st, ast.Store):
                    last[nm] = i
    if any(isinstance(n, ast.If) and n.orelse and ast.dump(ast.Module(body=n.body, type_ignores=[])) == ast.dump(ast.Module(body=n.orelse, type_ignores=[]))
           for n in ast.walk(tree)):
        dead = True
    if dead:
        out.append("dead_code")
    if (spec.get("period") or {}).get("kind") in W.RELATIVE and re.search(r"\(\s*20\d\d\s*,|[=!<>]=?\s*20\d\d\b|\b20\d\d\s*[=!<>]=?", _strip_comments(code)):
        out.append("year_literal_rel")
    return out


def code_issues(code: str, spec: dict, question: str, codes: set | None = None, doc: str = "") -> list[str]:
    """Code runtime: hợp đồng (build_sft_data.code_problem), ghi cứng đáp án, ghi cứng TODAY ở câu tương đối, quá dài."""
    import ast
    from build_sft_data import code_problem
    gold = spec["gold"]
    out = []
    if p := code_problem(code, gold["type"]):
        return [p]
    tree = ast.parse(code)
    consts = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)]
    a = gold.get("answer")
    nums = [a] if gold["type"] in ("number", "percent") else [x for x in (a or []) if isinstance(x, (int, float))] if gold["type"] == "list" else []
    for v in nums:
        if isinstance(v, (int, float)) and not isinstance(v, bool) and (abs(v) >= 100 or v != int(v)):
            if any(isinstance(c, (int, float)) and not isinstance(c, bool) and abs(c - v) < 1e-9 for c in consts):
                out.append("hardcoded_answer")
                break
    strs = [a] if gold["type"] in ("string", "date") else [x for x in (a or []) if isinstance(x, str)] if gold["type"] == "list" else []
    fq = W._fold_ascii(question)
    for v in strs:
        if isinstance(v, str) and len(v) >= 3 and W._fold_ascii(v) not in fq and any(isinstance(c, str) and W._fold_ascii(v) in W._fold_ascii(c) for c in consts):
            out.append("hardcoded_answer")
            break
    rel = (spec["period"] and spec["period"]["kind"] in W.RELATIVE) or spec["slot"] == "lots_expiring"
    if rel:
        lits = _date_literals(spec["as_of"])
        t = W._d(spec["as_of"])
        if any(isinstance(c, str) and any(l in c for l in lits) for c in consts) or \
                re.search(rf"\(\s*{t.year}\s*,\s*0?{t.month}\s*,\s*0?{t.day}\s*\)", code):
            out.append("hardcoded_today")
        elif any(isinstance(c, str) and re.search(r"20\d\d-\d\d-\d\d|\d{1,2}/\d{1,2}/20\d\d", c) for c in consts) or \
                re.search(r"(?:date|datetime|Timestamp)\(\s*20\d\d\s*,\s*\d{1,2}\s*,\s*\d{1,2}", code):
            out.append("hardcoded_date_relative")  # mốc suy ra từ hôm nay mà ghi cứng: sai khi TODAY đổi
        if "TODAY" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}:
            out.append("missing_TODAY")
    if codes:  # mã nội bộ (mã hàng, mã khách…) ghi cứng mà câu hỏi/README/quy ước không nêu: học sinh không suy ra được từ đề
        ql, dl = question.lower(), doc.lower()
        if any(isinstance(c, str) and c.strip().lower() in codes and c.strip().lower() not in ql and c.strip().lower() not in dl for c in consts):
            out.append("hardcoded_key")
    if spec.get("period") and spec["period"]["kind"] in ("until", "until_today"):  # công nợ/tồn "tính đến": chỉ được có mốc cuối
        allowed = set(_date_literals(spec["period"].get("d") or spec["as_of"]))
        dd = W._d(spec["period"].get("d") or spec["as_of"]) + dt.timedelta(days=1)
        allowed |= set(_date_literals(dd.isoformat()))
        lits = {c for c in consts if isinstance(c, str) and re.search(r"20\d\d-\d\d-\d\d|\d{1,2}/\d{1,2}/20\d\d", c)}
        tup = {(int(a), int(b), int(c)) for a, b, c in re.findall(r"(?:date|datetime|Timestamp)\(\s*(20\d\d)\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})", code)}
        okt = {(x.year, x.month, x.day) for x in (W._d(spec["period"].get("d") or spec["as_of"]), dd)}
        if any(not any(a in l for a in allowed) for l in lits) or (tup - okt):
            out.append("fixed_date_until")
    if DRAFT.search(code):  # chú thích nháp tự sửa ("# Sai, cần…", "placeholder") trong đích học là rác
        out.append("draft_comment")
    out += quality_issues(code, spec)  # audit giai đoạn 2: code thừa, chú thích rác/suy đoán, năm ghi cứng ở kỳ tương đối
    if len(code) > 6000:
        out.append("too_long")
    return sorted(set(out))


REL_PHRASE = {"today": r"hom nay|hnay|bua nay", "yesterday": r"hom qua|hqua", "this_month": r"thang nay|dau thang", "last_month": r"thang truoc",
              "this_week": r"tuan nay", "last_week": r"tuan truoc|tuan roi", "last_n": r"\d+ ngay", "this_year": r"nam nay|dau nam",
              "until_today": r"den nay|den hom nay|hien tai|hien gio|bay gio|tinh den|den hnay|toi nay|cho toi nay|hien nay"}


def period_ok(canonical: str, spec: dict) -> bool:
    """Câu gốc nêu đúng kỳ của đặc tả (bắt teacher đổi kỳ)."""
    q = W._fold_ascii(canonical)
    p = spec["period"]
    if spec["slot"] == "lots_expiring":
        return bool(re.search(rf"\b{spec['params']['n']}\s*ngay", q))
    if p is None:
        return True
    k = p["kind"]

    def has_date(d):
        x = W._d(d)
        return bool(re.search(rf"(?<!\d)0?{x.day}\s*[/-]\s*0?{x.month}\s*[/-]\s*{x.year}(?!\d)|(?<!\d)0?{x.day}\s*thang\s*0?{x.month}\s*(nam\s*)?{x.year}", q))
    if k == "month":
        return bool(re.search(rf"thang\s*0?{p['m']}\s*(/|-|nam)\s*{p['y']}|(?<!\d)0?{p['m']}/{p['y']}", q))
    if k == "day":
        return has_date(p["d"])
    if k == "range":
        a, b = W._d(p["d1"]), W._d(p["d2"])
        if a.day == 1 and b == W._month_end(b.year, b.month) and (a.year, a.month) != (b.year, b.month):  # nhiều tháng trọn: cho nói theo tháng
            m1 = rf"thang\s*0?{a.month}(?!\d)(\s*(/|-|nam)\s*{a.year})?"
            m2 = rf"thang\s*0?{b.month}\s*(/|-|nam)\s*{b.year}|(?<!\d)0?{b.month}/{b.year}"
            if re.search(m1, q) and re.search(m2, q) and a.year == b.year:
                return True
        return has_date(p["d1"]) and has_date(p["d2"])
    if k == "quarter":
        return bool(re.search(rf"quy\s*{p['q']}\b", q)) and str(p["y"]) in q
    if k == "until":
        return has_date(p["d"])
    pat = REL_PHRASE.get(k)
    ok = bool(re.search(pat, q)) if pat else True
    if k == "last_n":
        ok = bool(re.search(rf"\b{p['n']}\s*ngay", q))
    if spec.get("base_period"):
        b = spec["base_period"]
        ok = ok and (b["kind"] != "month" or bool(re.search(rf"thang\s*0?{b['m']}\s*(/|-|nam)\s*{b['y']}|(?<!\d)0?{b['m']}/{b['y']}", q)))
    return ok


def question_issues(sh: Shop, spec: dict, canonical: str, styled: str, bounds) -> list[str]:
    import synth_checks
    from datamate import verify
    out = []
    if not canonical or not styled:
        return ["empty_question"]
    if not period_ok(canonical, spec):
        out.append("period_drift")
    if spec["slot"] != "unanswerable":
        typed = {k: v for k, v in spec["gold"].items() if k in ("type", "answer")}
        if typed["type"] in ("number", "percent", "date", "string", "list") and (synth_checks.lo_dap_an(canonical, typed) or synth_checks.lo_dap_an(styled, typed)):
            out.append("answer_leak")
        if verify.check(styled, typed):
            out.append("type_phrase_mismatch")
        if verify.zero_suspect(styled, typed):
            out.append("zero_suspect")
    if synth_checks.goi_y(canonical) or synth_checks.goi_y(styled):
        out.append("solving_hint")
    if synth_checks.ma_noi_bo(canonical, sh.codes) or synth_checks.ma_noi_bo(styled, sh.codes):
        out.append("internal_code")
    if synth_checks.blocklist_hits(canonical) or synth_checks.blocklist_hits(styled):
        out.append("blocklist")
    n = len(styled.split())
    if not bounds[0] <= n <= bounds[1]:
        out.append("length")
    p = spec["params"]
    it = {i["item"]: i for i in sh.rec["items"]}
    fq = W._fold_ascii(canonical)
    need = [it[p["item"]]["name"]] if "item" in p else []
    need += [p["group"]] if "group" in p else []
    need += [p["channel"]] if "channel" in p else []
    if any(W._fold_ascii(x) not in fq for x in need):
        out.append("filter_drift")
    if "k" in p and not re.search(rf"(?<!\d){p['k']}(?!\d)", fq):
        out.append("k_drift")
    return out


# ---------------------------------------------------------------- trạng thái mẫu
def new_sample(sh: Shop, spec: dict, rnd: int) -> dict:
    return {"id": f"{sh.name}:{spec['key']}", "shop": sh.name, "split": sh.s["split"], "family": sh.s["family"], "industry": sh.s["industry"],
            "source_hash": sh.source_hash, "seed": sh.s["seed"], "spec": {k: v for k, v in spec.items() if k not in ("tags",)},
            "slot": spec["slot"], "skill": W.SLOTS[spec["slot"]]["skill"], "answer_type": spec["gold"]["type"], "difficulty": spec["difficulty"],
            "time_kind": time_kind(spec), "relative": bool(spec["period"] and spec["period"]["kind"] in W.RELATIVE) or spec["slot"] == "lots_expiring",
            "edge_tags": spec["tags"], "as_of": sh.today, "round_created": rnd, "prompt_version": PROMPT_VERSION,
            "status": "planned", "reject": [], "history": [], "question_canonical": None, "question": None, "style_variant": 0,
            "verifier": {"status": "pending"}, "coder": {"status": "pending", "attempts": 0}, "mutations": []}


def hist(s: dict, event: str, **kw):
    s["history"].append({"event": event, **kw})


# ---------------------------------------------------------------- vòng xử lý một shop
class Runner:
    def __init__(self, work: Path, ledger: Path, phase: str, stop_at: int):
        self.work, self.ledger, self.phase, self.stop_at = work, ledger, phase, stop_at
        self.state_path = work / "state" / "samples.json"
        self.samples: dict[str, dict] = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        self.rubric = json.loads((work / "rubric" / "rubric_v1.json").read_text(encoding="utf-8"))
        self.bounds = tuple(self.rubric["length_bounds"])
        self.shops: dict[str, Shop] = {}
        self._feasible = {}
        fp = work / "focus.json"  # quota bù khoảng trống so với MINING (ghi file để truy vết); chỉ áp cho train, chỉ đếm mẫu từ vòng since_round
        self.focus = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() and phase in ("dev", "main") else None

    def shop(self, s: dict) -> Shop:
        if s["name"] not in self.shops:
            self.shops[s["name"]] = Shop(self.work, s)
            if self.shops[s["name"]].errors:
                raise RuntimeError(f"fixture integrity {s['name']}: {self.shops[s['name']].errors}")
        return self.shops[s["name"]]

    def of(self, shop: str, statuses=None) -> list[dict]:
        with LOCK:
            return [x for x in self.samples.values() if x["shop"] == shop and (statuses is None or x["status"] in statuses)]

    def save(self):
        with LOCK:
            write_json(self.state_path, self.samples)

    def budget_left(self) -> int:
        return self.stop_at - ledger_count(self.ledger)

    # ------------------------------------------------------------ quota
    def feasible(self, split: str) -> dict:
        """Số shop của split làm được từng slot (thiếu tính năng thì quota 0, ít shop thì trần 4 mẫu/shop)."""
        if split not in self._feasible:
            c = collections.Counter()
            for s in shop_plan():
                if s["split"] != split:
                    continue
                sh = self.shop(s)
                rng = random.Random(7)
                for slot in W.SLOTS:
                    if any(sample_spec(sh.rec, s, slot, rng.choice(SLOT_KINDS.get(slot, DEFAULT_KINDS)), rng) for _ in range(8)):
                        c[slot] += 1
            self._feasible[split] = c
        return self._feasible[split]

    def targets(self, split: str) -> dict:
        fc = self.focus
        if fc and split == "train":
            n = sum(fc["slot"].values())
            return {"slot": {k: fc["slot"].get(k, 0) for k in SLOT_SHARE}, "time": {k: round(n * v) for k, v in fc["time"].items()},
                    "diff": {k: round(n * v) for k, v in DIFF_SHARE.items()}, "edge": {k: round(v * n / 900) for k, v in EDGE_TARGET.items()}}
        n = {"train": TRAIN_TARGET, "val": 250, "final": 0}[split]
        tot = sum(SLOT_SHARE.values())
        fz = self.feasible(split) if n else collections.Counter()
        slot = {k: (min(max(4, round(n * v / tot)), 4 * fz[k]) if fz[k] else 0) for k, v in SLOT_SHARE.items()}
        scale = n / 900
        return {"slot": slot, "time": {k: round(n * v) for k, v in TIME_KINDS.items()}, "diff": {k: round(n * v) for k, v in DIFF_SHARE.items()},
                "edge": {k: round(v * scale) for k, v in EDGE_TARGET.items()}}

    def counts(self, split: str, statuses=("accepted",)) -> dict:
        c = {k: collections.Counter() for k in ("slot", "time", "shop", "shop_slot", "diff", "edge")}
        with LOCK:
            vals = list(self.samples.values())
        since = self.focus["since_round"] if self.focus and split == "train" else 0
        for s in vals:
            if s["split"] == split and s["status"] in statuses and (s["round_created"] >= since or s["status"] != "accepted"):
                c["slot"][s["slot"]] += 1
                c["time"][s["time_kind"]] += 1
                c["shop"][s["shop"]] += 1
                c["shop_slot"][(s["shop"], s["slot"])] += 1
                c["diff"][s["difficulty"]] += 1
                c["edge"].update(s["edge_tags"])
        return c

    def yield_by_slot(self) -> dict:
        done = collections.defaultdict(lambda: [0, 0])
        with LOCK:
            vals = list(self.samples.values())
        for s in vals:
            if s["status"] in ("accepted", "rejected") and s["split"] != "final":
                done[s["slot"]][0] += s["status"] == "accepted"
                done[s["slot"]][1] += 1
        return {k: min(1.0, max(0.25, (a + 2) / (n + 3))) for k, (a, n) in done.items()}

    def plan_batch(self, sh: Shop, n: int, rnd: int, planned: dict) -> list[dict]:
        """Chọn n đặc tả cho shop theo thiếu hụt quota (mẫu đang chờ tính theo tỉ lệ giữ ước lượng của slot)."""
        split = sh.s["split"]
        rng = random.Random(f"{sh.name}:{rnd}:{PROMPT_VERSION}")
        if split in ("final", "final_b", "final_c"):  # final: một lô cố định rải đều kỹ năng, không theo quota hay kết quả trước
            slots = list(W.SLOTS)
            rng.shuffle(slots)
            out = []
            for slot in slots:
                for _ in range(8):
                    spec = sample_spec(sh.rec, sh.s, slot, rng.choice(SLOT_KINDS.get(slot, DEFAULT_KINDS)), rng)
                    if spec:
                        out.append(spec)
                        break
                if len(out) == n:
                    break
            return out
        tg, have = self.targets(split), self.counts(split)
        pend = self.counts(split, statuses=("planned", "asked", "pending"))
        y = self.yield_by_slot()
        used = {s["spec"]["key"] for s in self.of(sh.name)}
        cands = []
        for slot in W.SLOTS:
            if not tg["slot"][slot]:
                continue
            kinds = SLOT_KINDS.get(slot, DEFAULT_KINDS)
            for _ in range(4):
                spec = sample_spec(sh.rec, sh.s, slot, rng.choice(kinds), rng)
                if spec and spec["key"] not in used and spec["key"] not in {c["key"] for c in cands}:
                    cands.append(spec)
        chosen, batch_slots, batch_time = [], collections.Counter(), collections.Counter()
        for k in ("diff", "edge"):
            planned.setdefault(k, collections.Counter())

        def deficit(dim, key, default=30, rate=0.6):
            t = tg[dim].get(key, default)
            exp = have[dim][key] + rate * (pend[dim][key] + planned[dim][key])
            return (t - exp) / max(1, t)
        for _ in range(n):
            best, best_score = None, -1e9
            for spec in cands:
                if spec in chosen:
                    continue
                slot, tk = spec["slot"], time_kind(spec)
                d_slot = deficit("slot", slot, rate=y.get(slot, 0.6))
                if d_slot <= 0:  # slot đã đủ quota (tính cả mẫu đang chờ theo tỉ lệ giữ): không giao thêm
                    continue
                d_time = deficit("time", tk) if tk in tg["time"] else 0.0  # kỳ bắt buộc của slot (công nợ, lô) không có quota riêng
                d_diff = deficit("diff", spec["difficulty"])
                edges = [deficit("edge", t) for t in spec["tags"] if t in tg["edge"]]
                d_edge = max(edges) if edges else 0.0
                score = 2.0 * d_slot + 1.5 * d_time + 0.8 * d_diff + 0.4 * d_edge - 1.5 * batch_slots[slot] - 0.5 * batch_time[tk] \
                    - 0.6 * have["shop_slot"][(sh.name, slot)] + rng.random() * 0.05
                if score > best_score:
                    best, best_score = spec, score
            if best is None:  # mọi slot khả thi của shop đã đủ quota
                break
            chosen.append(best)
            batch_slots[best["slot"]] += 1
            batch_time[time_kind(best)] += 1
            planned["slot"][best["slot"]] += 1
            planned["time"][time_kind(best)] += 1
            planned["diff"][best["difficulty"]] += 1
            planned["edge"].update(best["tags"])
        return chosen

    # ------------------------------------------------------------ một vòng của một shop
    def shop_round(self, sh: Shop, rnd: int, new_specs: list[dict]) -> dict:
        from stress_noise import VanPhong
        import model_bakeoff as mb
        work, calls = self.work, collections.Counter()
        news = [new_sample(sh, spec, rnd) for spec in new_specs]
        with LOCK:
            for s in news:
                self.samples[s["id"]] = s
        if news:
            r, hit = model_call(work, "teacher", sh.name, "gemini", teacher_prompt(sh, [s["spec"] for s in news]), think="high", max_tokens=24000)
            calls["teacher"] += not hit
            try:
                items = llm.extract_json(r["content"]) if r.get("finish") == "stop" else []
            except ValueError:
                items = []
            items = items if isinstance(items, list) else []
            got = {int(x["i"]): str(x["question"]).strip() for x in items if isinstance(x, dict) and str(x.get("i", "")).isdigit() and x.get("question")}
            vp = VanPhong(seed=int(digest([sh.name, rnd])[:8], 16), known=sh.known)
            for i, s in enumerate(news, 1):
                q = got.get(i)
                if not q:
                    s["status"], s["reject"] = "rejected", ["teacher_api_error" if r.get("finish") == "api_error" else "teacher_missing"]
                    hist(s, "teacher_fail", finish=r.get("finish"))
                    continue
                s["question_canonical"], s["question"], s["status"] = q, vp.render(q), "asked"
                hist(s, "teacher", question=q)
                bad = question_issues(sh, s["spec"], q, s["question"], self.bounds) + self.contamination(sh, q, s["question"])
                if bad:
                    s["status"], s["reject"] = "rejected", bad
                    hist(s, "question_reject", reasons=bad)
        active = self.of(sh.name, ("asked", "pending"))
        v_items = [s for s in active if s["verifier"]["status"] in ("pending", "restyled")]
        v_canon = [s for s in active if s["verifier"]["status"] == "recheck_canonical"]
        c_items = [s for s in active if s["coder"]["status"] in ("pending", "retry")]
        qlist = [s["question"] for s in v_items] + [s["question_canonical"] for s in v_canon]
        with cf.ThreadPoolExecutor(max_workers=2) as ex:
            fv = ex.submit(model_call, work, "verifier", sh.name, VERIFIER, solver_prompt(sh, qlist, False), think="off", max_tokens=24000) if qlist else None
            fc = ex.submit(model_call, work, "coder", sh.name, "qwen", solver_prompt(sh, [s["question"] for s in c_items], True), think="off",
                           max_tokens=24000) if c_items else None
            rv = fv.result() if fv else None
            rc = fc.result() if fc else None
        if rv:
            r, hit = rv
            calls["verifier"] += not hit
            codes = mb.tach_khoi(r["content"], len(qlist)) if r.get("finish") == "stop" else [None] * len(qlist)
            targets = [(s, False) for s in v_items] + [(s, True) for s in v_canon]
            with cf.ThreadPoolExecutor(max_workers=4) as ex:
                outs = list(ex.map(lambda c: run_verifier(c, sh.fx["tables_pkl"], sh.today, work), codes))
            for (s, canon), code, (raw, err) in zip(targets, codes, outs):
                self._verdict(sh, s, canon, code, raw, err, r.get("finish"))
        if rc:
            r, hit = rc
            calls["coder"] += not hit
            self._apply_codes(sh, c_items, r, "qwen")
        for _ in range(2):  # sửa lỗi chạy: chỉ gửi câu + code + lỗi, cùng model viết code đó
            fix = [s for s in self.of(sh.name, ("asked", "pending")) if s["coder"]["status"] == "crash" and s["coder"].get("repairs", 0) < 3]
            if not fix:
                break
            for model in sorted({s["coder"]["model"] for s in fix}):
                grp = [s for s in fix if s["coder"]["model"] == model]
                items = [{"question": s["question"], "code": s["coder"]["code"] or "", "error": s["coder"]["error"][-1500:]} for s in grp]
                r, hit = model_call(work, "repair", sh.name, model, repair_prompt(sh, items), think="high" if model == "gemini" else "off", max_tokens=20000)
                calls["repair"] += not hit
                for s in grp:
                    s["coder"]["repairs"] = s["coder"].get("repairs", 0) + 1
                self._apply_codes(sh, grp, r, model, repair=True)
        fb = [s for s in self.of(sh.name, ("asked", "pending")) if s["coder"]["status"] == "fallback"]
        if fb:
            r, hit = model_call(work, "coder_fallback", sh.name, "gemini", solver_prompt(sh, [s["question"] for s in fb], True), think="high", max_tokens=32000)
            calls["coder_fallback"] += not hit
            self._apply_codes(sh, fb, r, "gemini")
        for s in self.of(sh.name, ("asked", "pending")):
            self._advance(sh, s)
        with sh._mut_lock:
            sh._mut.clear()  # bản ghi đối chứng ~7 MB/kiểu: dựng lại khi cần, không giữ qua vòng
        return dict(calls)

    def contamination(self, sh: Shop, canonical: str, styled: str) -> list[str]:
        """Trùng câu tune/holdout (Kit.contaminated, cục bộ) hoặc gần trùng mẫu đã nhận của cùng shop."""
        kit = sh.kit()
        out = []
        if kit.contaminated(styled)["hit"] or kit.contaminated(canonical)["hit"]:
            out.append("contaminated_eval")
        a = _shingles(canonical)
        for o in self.of(sh.name, ("accepted", "asked", "pending")):
            if o.get("question_canonical") and o["question_canonical"] != canonical:
                b = _shingles(o["question_canonical"])
                if len(a & b) / max(1, len(a | b)) >= 0.7:
                    out.append("duplicate")
                    break
        return out

    def _verdict(self, sh: Shop, s: dict, canon: bool, code, raw, err, finish):
        q = s["question_canonical"] if canon else s["question"]
        ok, why = verifier_pass(raw, err, s["spec"]["gold"], q)
        hist(s, "verifier", form="canonical" if canon else "styled", variant=s["style_variant"], raw=raw, error=err, ok=ok, why=why, api=finish)
        if finish == "api_error":
            return
        if not canon:
            s["verifier"] = {"status": "pass", "code": code, "variant": s["style_variant"], "provider": VERIFIER} if ok else \
                {"status": "recheck_canonical", "first": why, "raw": raw, "error": err, "code": code, "provider": VERIFIER}
            return
        if ok:  # câu gốc đúng mà câu văn phong sai → lệch nghĩa do văn phong: viết lại kiểu khác, tối đa 2 lần
            if s["style_variant"] >= 2:
                s["status"], s["reject"] = "rejected", ["style_drift"]
                return
            from stress_noise import VanPhong
            s["style_variant"] += 1
            s["question"] = VanPhong(seed=int(digest([s["id"], s["style_variant"]])[:8], 16), known=sh.known).render(s["question_canonical"])
            bad = question_issues(sh, s["spec"], s["question_canonical"], s["question"], self.bounds)
            if bad:
                s["status"], s["reject"] = "rejected", ["style_drift"] + bad
                return
            s["verifier"] = {"status": "restyled"}
            s["coder"] = {"status": "pending", "attempts": s["coder"].get("attempts", 0), "restyled": True}
            return
        coder_ok = s["coder"]["status"] == "pass"
        s["status"] = "rejected"
        s["reject"] = ["verifier_disagree" if coder_ok else "ambiguous_or_wrong_meaning"] if why == "ok" else ["verifier_cannot_solve_" + why]

    def _apply_codes(self, sh: Shop, items: list[dict], r: dict, model: str, repair=False):
        import model_bakeoff as mb
        codes = mb.tach_khoi(r["content"], len(items)) if r.get("finish") == "stop" else [None] * len(items)
        codes = [strip_provided_imports(re.sub(r"^\s*#\s*c[aâ]u\s*\d+[^\n]*\n", "", c, count=1)) if c else None for c in codes]

        def run(pair):
            s, code = pair
            return run_runtime(sh.kit(), sh.name, code, sh.today, s["question"]) if code else None
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            exs = list(ex.map(run, zip(items, codes)))
        for s, code, ex in zip(items, codes, exs):
            if r.get("finish") == "api_error":
                hist(s, "coder_api_error", model=model)
                continue
            st = {"model": model, "code": code, "attempts": s["coder"].get("attempts", 0) + (0 if repair else 1), "repairs": s["coder"].get("repairs", 0)}
            if not code:
                st.update(status="crash", error="missing_code: không thấy khối code cho câu này")
            elif not (ex["ok"] and not ex["issues"] and not ex["zero_suspect"]):
                msg = ex.get("error") or ""
                if ex.get("issues"):
                    from datamate import verify
                    msg += "\n" + "; ".join(i.message for i in verify.check(s["question"], ex["answer"] or {}))
                if ex.get("zero_suspect"):
                    msg += "\nĐáp án bằng 0: kiểm tra lại bộ lọc tên, kỳ, trạng thái (bảng có dữ liệu)."
                if "duplicate labels" in msg or "reindex" in msg:
                    msg += "\nGợi ý: nối bằng merge hoặc Series.map theo mã thay vì gán theo .loc với index trùng."
                st.update(status="crash", error=msg.strip() or "error")
            else:
                ci = code_issues(code, s["spec"], s["question"], sh.codes, sh.doc)
                if ci:
                    st.update(status="static_fail", issues=ci, answer=ex["answer"])
                elif check_answer(ex["answer"], s["spec"]["gold"]):
                    st.update(status="pass", answer=ex["answer"])
                else:
                    st.update(status="wrong", answer=ex["answer"])
            if st["status"] == "crash" and st["repairs"] >= 3:
                st["status"] = "crash_final"
            hist(s, "coder", **{k: v for k, v in st.items() if k != "code"}, repair=repair)
            s["coder"] = st

    def _advance(self, sh: Shop, s: dict):
        """Sau vòng: Qwen sai/ghi cứng → thử lại một lần; vẫn hỏng → coder dự phòng Gemini; Gemini hỏng → loại. Hai bên qua → đối chứng."""
        c = s["coder"]
        if c["status"] in ("wrong", "static_fail", "crash_final"):
            if c.get("model") == "qwen" and c["status"] != "crash_final" and c.get("attempts", 1) < 2:
                s["coder"] = dict(c, status="retry")
            elif c.get("model") == "qwen":
                s["coder"] = dict(c, status="fallback", repairs=0, qwen_last=c["status"])
            else:
                reason = {"wrong": "coder_wrong", "crash_final": "code_error"}.get(c["status"]) or ("hardcode:" + ",".join(c.get("issues", [])))
                if c["status"] == "wrong" and s["verifier"]["status"] == "recheck_canonical":
                    reason = "both_disagree_gold"
                s["status"], s["reject"] = "rejected", [reason]
                return
        if s["verifier"]["status"] == "pass" and s["coder"]["status"] == "pass":
            self._mutations(sh, s)
            return
        s["status"] = "pending"

    def _mutations(self, sh: Shop, s: dict):
        """Đối chứng có chủ đích: chạy lại code runtime (và code verifier) trên dữ liệu đã sửa; đáp án mới do oracle tính."""
        spec = s["spec"]
        res, ok = [], True
        for kind in mutation_plan(sh, spec):
            if kind == "m_today":
                new_today = today_shift(spec)
                g = W.gold(sh.rec, dict(spec, as_of=new_today))
                ex = run_runtime(sh.kit(), sh.name, s["coder"]["code"], new_today, s["question"])
                r_ok = ex["ok"] and not ex["issues"] and check_answer(ex["answer"], g)
                res.append({"kind": kind, "gold": g, "gold_changed": True, "runtime_pass": r_ok, "runtime_answer": ex.get("answer"),
                            "runtime_error": ex.get("error"), "verifier_pass": None})
                ok &= r_ok
                continue
            m = sh.mutated(kind)
            g = W.gold(m["rec"], spec)
            ex = run_runtime(m["kit"], m["name"], s["coder"]["code"], sh.today, s["question"])
            r_ok = ex["ok"] and not ex["issues"] and check_answer(ex["answer"], g)
            raw, err = run_verifier(s["verifier"]["code"], m["tables_pkl"], sh.today, self.work)
            v_ok, _ = verifier_pass(raw, err, g, s["question"])
            res.append({"kind": kind, "gold": g, "gold_changed": g != spec["gold"], "runtime_pass": r_ok, "runtime_answer": ex.get("answer"),
                        "runtime_error": ex.get("error"), "verifier_pass": v_ok, "verifier_answer": raw, "verifier_error": err})
            ok &= r_ok and v_ok
        if ok and spec["slot"] != "unanswerable":  # đối chứng nhắm đúng kỳ/đối tượng của đặc tả: đáp án chắc chắn đổi
            r = target_check(sh, s, self.work)
            if r:
                res.append(r)
                ok &= r["runtime_pass"] and r["verifier_pass"]
        s["mutations"] = res
        if ok:
            s["status"] = "accepted"
            hist(s, "accepted", mutations=[r["kind"] for r in res])
        else:
            bad = [r["kind"] for r in res if not r["runtime_pass"] or r["verifier_pass"] is False]
            who = "runtime" if any(not r["runtime_pass"] for r in res) else "verifier"
            s["status"], s["reject"] = "rejected", [f"mutation_{who}:" + ",".join(bad)]
            hist(s, "mutation_fail", kinds=bad)


def _shingles(q: str) -> set:
    w = W._fold_ascii(q).strip(" ?!.,;:").split()
    return {" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}


def target_check(sh: Shop, s: dict, work: Path) -> dict | None:
    """Chạy code runtime và code verifier trên fixture tạm có sửa đổi nhắm đúng đặc tả (synth_world.target_mutation)."""
    from synth_tools import Kit
    spec = {k: v for k, v in s["spec"].items() if k != "gold"}
    tm = W.target_mutation(sh.rec, spec)
    if not tm:
        return None
    rec2, desc = tm
    g = W.gold(rec2, spec)
    base = Path(os.environ.get("SYNTH_EXEC_TMP") or tempfile.gettempdir())
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tgt_", dir=base, ignore_cleanup_errors=True) as tmp:
        name = f"{sh.name}__target"
        fx = W.write_fixture(Path(tmp), name, rec2, sh.s["family"])
        kit = Kit(data=Path(tmp), cache=Path(tmp) / "cache")
        ex = run_runtime(kit, name, s["coder"]["code"], sh.today, s["question"])
        raw, err = run_verifier(s["verifier"]["code"], fx["tables_pkl"], sh.today, work)
    r_ok = ex["ok"] and not ex["issues"] and check_answer(ex["answer"], g)
    v_ok = verifier_pass(raw, err, g, s["question"])[0]
    return {"kind": "m_target", "what": desc, "gold": g, "gold_changed": True, "runtime_pass": r_ok, "runtime_answer": ex.get("answer"),
            "runtime_error": ex.get("error"), "verifier_pass": v_ok, "verifier_answer": raw, "verifier_error": err}


def mutation_plan(sh: Shop, spec: dict) -> list[str]:
    """Chọn đối chứng: tối đa 2 kiểu làm đổi đáp án (độ nhạy, theo ưu tiên của slot) + kiểu bất biến liên quan + đổi TODAY với câu tương đối."""
    slot = spec["slot"]
    feats = sh.rec["features"]
    if slot == "unanswerable":
        return ["m_amount"]
    pri = {"revenue_net": ["m_refund", "m_boundary"], "refund_total": ["m_refund"], "qty_base": ["m_units", "m_boundary"], "top_items": ["m_units", "m_unsold"],
           "top_customer": ["m_walkin", "m_amount"], "top_customers": ["m_walkin", "m_amount"], "n_customers": ["m_walkin", "m_boundary"],
           "cancel_rate": ["m_cancel", "m_alias"], "n_cancelled": ["m_cancel", "m_alias"], "n_orders": ["m_cancel", "m_boundary"],
           "debt_total": ["m_debt"], "top_debtor": ["m_debt"], "lots_expiring": ["m_lots"], "purchase_money": ["m_purchase"],
           "unsold_items": ["m_unsold"], "exists_item_day": ["m_unsold", "m_cancel"], "best_day": ["m_boundary", "m_walkin"], "best_month": ["m_boundary", "m_walkin"]}
    order = pri.get(slot, []) + ["m_amount", "m_boundary", "m_cancel", "m_walkin", "m_unsold", "m_refund"]
    blocked = {"m_debt": not feats["debt"], "m_lots": not feats["lots"], "m_purchase": not feats["purchases"]}
    out = []
    for k in dict.fromkeys(order):
        if len(out) == 2 or blocked.get(k):
            continue
        try:
            if W.gold(sh.mutated(k)["rec"], spec) != spec["gold"]:
                out.append(k)
        except W.Degenerate:
            continue
    ends_today = bool(spec["period"]) and W.resolve(spec["period"], spec["as_of"])[1] == W._d(spec["as_of"])
    if ends_today and "m_after" not in out:
        out.append("m_after")
    if slot in ("n_orders", "n_cancelled", "cancel_rate", "revenue", "avg_order", "top_customer", "top_customers", "n_customers") and "m_alias" not in out:
        out.append("m_alias")
    if (spec["period"] and spec["period"]["kind"] in W.RELATIVE) or slot == "lots_expiring":
        try:
            if W.gold(sh.rec, dict(spec, as_of=today_shift(spec))) != spec["gold"]:
                out.append("m_today")
        except W.Degenerate:
            pass
    return out


# ---------------------------------------------------------------- vòng lặp chính
def run_phase(work: Path, ledger: Path, phase: str, rounds: int, stop_at: int, only=None, n_batch=N_BATCH, workers=6) -> dict:
    llm.configure_run(ledger, limit=llm.HARD_CAP,
                      provider_limits=json.loads(os.environ.get("SYNTH_PROVIDER_LIMITS") or '{"gemini": 4, "gpt": 4, "qwen": 4, "deepseek": 2}'))
    run = Runner(work, ledger, phase, stop_at)
    want = {"dev": ("train",), "main": ("train", "val"), "final": ("final",), "final_b": ("final_b",), "final_c": ("final_c",)}[phase]
    if phase in ("final", "final_b", "final_c"):  # final chỉ chạy một lần với cấu hình đã đóng băng
        fz = json.loads((work / {"final": "freeze.json", "final_b": "freeze_b.json", "final_c": "freeze_c.json"}[phase]).read_text(encoding="utf-8"))
        if fz["implementation"]["combined"] != implementation_hash()["combined"]:
            raise RuntimeError("code/prompt đã đổi sau khi đóng băng: không chạy final")
    shops = [s for s in shop_plan() if s["split"] in want and (not only or s["name"] in only)]
    log, pauses = [], 0
    for _ in range(rounds):
        tripped = sorted(p for p, n in BREAKER.items() if n >= BREAKER_LIMIT)
        BREAKER.clear()
        if tripped:  # vòng trước có nhà cung cấp treo: nghỉ rồi chỉ xử lý mẫu dở (không mở đặc tả mới)
            pauses += 1
            if pauses > 6:
                log.append({"stop": "provider_down", "providers": tripped})
                break
            time.sleep(600)
        rlog = work / "rounds.jsonl"
        r_id = 1 + (len(rlog.read_text(encoding="utf-8").splitlines()) if rlog.exists() else 0)
        left = run.budget_left()
        if left <= 0:
            log.append({"round": r_id, "stop": "budget"})
            break
        objs = [run.shop(s) for s in shops]
        random.Random(r_id).shuffle(objs)
        planned = collections.defaultdict(lambda: {k: collections.Counter() for k in ("slot", "time", "diff", "edge")})  # theo split
        jobs, need = [], 0
        for sh in objs:  # việc dở của shop luôn được làm tiếp; lô mới chỉ mở khi còn ngân sách cho cả vòng
            pending = run.of(sh.name, ("asked", "pending"))
            fresh = []
            if phase in ("final", "final_b", "final_c"):
                if not run.of(sh.name):
                    fresh = run.plan_batch(sh, n_batch, r_id, planned[sh.s["split"]])
            elif not tripped and need + (3 if pending else 0) + 4 <= left:
                fresh = run.plan_batch(sh, n_batch, r_id, planned[sh.s["split"]])
            if fresh or pending:
                jobs.append((sh, fresh))
                need += (1 if fresh else 0) + 3
        if not jobs:
            log.append({"round": r_id, "stop": "nothing_to_do"})
            break
        t0, calls = time.time(), collections.Counter()
        errors = {}
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run.shop_round, sh, r_id, fresh): sh.name for sh, fresh in jobs}
            for f in cf.as_completed(futs):
                try:
                    calls.update(f.result())
                except (llm.LeakError, llm.BudgetExceeded):
                    run.save()
                    raise
                except Exception as e:  # noqa: BLE001 — một shop lỗi không làm mất kết quả các shop khác; mẫu dở chờ vòng sau
                    import traceback
                    errors[futs[f]] = traceback.format_exc()[-1500:]
        run.save()
        if errors:
            write_json(work / "errors" / f"round_{r_id}.json", errors)
        with LOCK:
            st = collections.Counter(x["status"] for x in run.samples.values() if x["split"] in want)
        entry = {"round": r_id, "phase": phase, "shops": len(jobs), "new_specs": sum(len(f) for _, f in jobs), "calls": dict(calls), "shop_errors": len(errors),
                 "breaker": {p: n for p, n in BREAKER.items() if n >= BREAKER_LIMIT},
                 "status": dict(st), "ledger": ledger_count(ledger), "wall_s": round(time.time() - t0, 1), "prompt_version": PROMPT_VERSION}
        log.append(entry)
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        with open(rlog, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if phase == "main" and quota_met(run) and not any(run.of(sh.name, ("asked", "pending")) for sh in objs):
            log.append({"round": r_id, "stop": "quota_met"})
            break
    return {"log": log}


def quota_met(run: "Runner") -> bool:
    for split in ("train", "val"):
        tg, have = run.targets(split), run.counts(split)
        if any(have["slot"][k] < v for k, v in tg["slot"].items()):
            return False
    return True


def plan_only(work: Path) -> dict:
    """Dựng mọi fixture, kiểm toàn vẹn, thử dựng prompt + guard cục bộ cho một lô mỗi shop (không gọi API)."""
    out = []
    for s in shop_plan():
        sh = Shop(work, s)
        specs = []
        rng = random.Random(1)
        for slot in W.SLOTS:
            for _ in range(5):
                sp = sample_spec(sh.rec, s, slot, rng.choice(SLOT_KINDS.get(slot, DEFAULT_KINDS)), rng)
                if sp:
                    specs.append(sp)
                    break
        tp = teacher_prompt(sh, specs[:N_BATCH])
        vp = solver_prompt(sh, ["Doanh thu tháng 4/2025 là bao nhiêu?"], False)
        cp = solver_prompt(sh, ["Doanh thu tháng 4/2025 là bao nhiêu?"], True)
        for p in (tp, vp, cp):
            llm.guard(p)
        out.append({"shop": s["name"], "split": s["split"], "family": s["family"], "industry": s["industry"], "integrity": sh.errors,
                    "feasible_slots": len({x["slot"] for x in specs}), "orders": len(sh.rec["orders"]), "lines": len(sh.rec["lines"]),
                    "teacher_chars": len(tp), "coder_chars": len(cp)})
    write_json(work / "plan_check.json", out)
    return {"shops": len(out), "integrity_errors": sum(bool(x["integrity"]) for x in out), "max_coder_chars": max(x["coder_chars"] for x in out)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    sub.add_parser("revalidate")
    cp = sub.add_parser("certify")
    cp.add_argument("--recheck-all", action="store_true")
    cp.add_argument("--workers", type=int, default=6)
    fzp = sub.add_parser("freeze")
    fzp.add_argument("--name", default="freeze.json")
    sub.add_parser("audit")
    e = sub.add_parser("export")
    e.add_argument("--tokenizer-python", default="/mnt/data/dattq53/miniconda3/envs/smh/bin/python")
    e.add_argument("--tokenizer", type=Path, default=ROOT / "outputs/synth_verified_pilot_20261110/tokenizer")
    r = sub.add_parser("run")
    r.add_argument("--phase", choices=["dev", "main", "final", "final_b", "final_c"], required=True)
    r.add_argument("--rounds", type=int, default=1)
    r.add_argument("--stop-at", type=int, default=3700)
    r.add_argument("--shops", default="")
    r.add_argument("--batch", type=int, default=N_BATCH)
    r.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    if a.cmd == "plan":
        print(json.dumps(plan_only(a.work), ensure_ascii=False))
    elif a.cmd == "revalidate":
        print(json.dumps(revalidate(a.work, a.ledger), ensure_ascii=False))
    elif a.cmd == "freeze":
        print(json.dumps(freeze(a.work, a.ledger, a.name), ensure_ascii=False))
    elif a.cmd == "certify":
        print(json.dumps(certify(a.work, a.ledger, workers=a.workers, recheck_all=a.recheck_all), ensure_ascii=False))
    elif a.cmd == "audit":
        print(json.dumps(run_audit(a.work, a.ledger), ensure_ascii=False))
    elif a.cmd == "export":
        print(json.dumps(export(a.work, a.ledger, a.tokenizer_python, a.tokenizer), ensure_ascii=False))
    else:
        only = set(filter(None, a.shops.split(","))) or None
        print(json.dumps(run_phase(a.work, a.ledger, a.phase, a.rounds, a.stop_at, only, a.batch, a.workers), ensure_ascii=False)[:2000])



def revalidate(work: Path, ledger: Path) -> dict:
    """Áp chuẩn kiểm hiện tại cho mẫu đã xử lý bằng bản trước (không gọi API): kiểm tĩnh code + đối chứng lại; quy ước đã sửa thì verify lại câu.
    Mẫu trượt chuẩn mới không bị nhận; được đi lại đường thử lại/dự phòng như mẫu mới."""
    run = Runner(work, ledger, "main", llm.HARD_CAP)
    plan = {s["name"]: s for s in shop_plan()}
    out = collections.Counter()
    for s in list(run.samples.values()):
        if s["prompt_version"] == PROMPT_VERSION:
            continue
        sh = run.shop(plan[s["shop"]])
        if s["status"] == "pending" and s["verifier"]["status"] == "recheck_canonical":
            s["verifier"] = {"status": "pending"}  # quy ước bản trước liệt kê thiếu cách ghi trạng thái: verify lại câu văn phong
            hist(s, "reverify_after_rules_fix")
            out["reverify"] += 1
        unfair = s["status"] == "rejected" and s["reject"] and s["reject"][0].startswith("mutation_") and set(s["reject"][0].split(":")[1].split(",")) <= {"m_cancel"}
        if unfair:
            s["status"], s["reject"] = "pending", []
            hist(s, "remeasure_after_mutation_fix")
            out["remeasure"] += 1
        if s["status"] in ("accepted", "pending") and s["coder"].get("code"):
            code = strip_provided_imports(s["coder"]["code"])
            ci = code_issues(code, s["spec"], s["question"])
            s["coder"]["code"] = code
            if ci:
                s["coder"] = dict(s["coder"], status="static_fail", issues=ci)
                s["status"] = "pending"
                hist(s, "static_fail_new_standard", issues=ci)
                out["static_fail"] += 1
            elif s["status"] == "accepted" or (s["verifier"]["status"] == "pass" and s["coder"]["status"] == "pass"):
                run._mutations(sh, s)
                out["remutated_" + s["status"]] += 1
        s["revalidated"] = PROMPT_VERSION
    for s in run.samples.values():
        if s["status"] == "pending":
            run._advance(run.shops[s["shop"]], s) if s["coder"]["status"] in ("static_fail",) else None
    run.save()
    return dict(out)



# ---------------------------------------------------------------- chứng nhận lại từ đầu + xuất
META_FIELDS = {"schema_version": str, "split": str, "family": str, "industry": str, "source_id": str, "source_hash": str, "seed": int,
               "slot": str, "skill": str, "answer_type": str, "answer_json": str, "difficulty": int, "time_kind": str, "relative_time": bool,
               "edge_tags": list, "as_of_date": str, "question": str, "question_canonical": str, "teacher": str, "verifier": str, "coder": str,
               "coder_attempts": int, "mutations": list, "prompt_version": str, "round": int, "audit": str, "code_sha256": str,
               "prompt_sha256": str, "tokens": int, "cert_standard": str, "cert_version": str, "shop_complex": bool, "schema_features": list}


def certify_one(run: "Runner", sh: Shop, s: dict) -> list[str]:
    """Chạy lại mọi cổng trên bản ghi dựng lại từ hạt: nguồn, gold, câu, code tĩnh, runtime, verifier, đối chứng. Trả lỗi (rỗng = đạt)."""
    errs = []
    if sh.source_hash != s["source_hash"]:
        return ["source_changed"]
    spec = dict(s["spec"])
    try:
        g = W.gold(sh.rec, {k: v for k, v in spec.items() if k != "gold"})
    except W.Degenerate:
        return ["gold_degenerate"]
    if g != spec["gold"]:
        return ["gold_changed"]
    errs += question_issues(sh, spec, s["question_canonical"], s["question"], run.bounds)
    if sh.kit().contaminated(s["question"])["hit"] or sh.kit().contaminated(s["question_canonical"])["hit"]:
        errs.append("contaminated_eval")
    code = s["coder"]["code"]
    errs += code_issues(code, spec, s["question"], sh.codes, sh.doc)
    ex = run_runtime(sh.kit(), sh.name, code, sh.today, s["question"])
    if not (ex["ok"] and not ex["issues"] and not ex["zero_suspect"] and check_answer(ex["answer"], g)):
        errs.append("runtime_replay")
    raw, err = run_verifier(s["verifier"]["code"], sh.fx["tables_pkl"], sh.today, run.work)
    if not verifier_pass(raw, err, g, s["question"])[0]:
        errs.append("verifier_replay")
    if errs:
        return sorted(set(errs))
    probe = dict(s)
    run._mutations(sh, probe)
    if probe["status"] != "accepted":
        errs.append("mutation_replay:" + ",".join(probe["reject"]))
    return errs


CERT_FILES = ["eval/synth_corpus.py", "eval/synth_world.py", "eval/synth_rubric.py", "eval/synth_tools.py", "eval/synth_execution.py",
              "eval/synth_checks.py", "eval/build_sft_data.py", "eval/stress_noise.py", "datamate/sandbox.py", "datamate/dm.py",
              "datamate/timeutil.py", "datamate/loader.py", "datamate/clean.py", "datamate/formatter.py", "datamate/verify.py", "requirements.txt"]


def cert_version() -> str:
    """Phiên bản chứng nhận = chuẩn + hash code/dữ liệu ảnh hưởng tới cổng kiểm (sinh bản ghi, oracle, kiểm tĩnh, runtime, loader, pandas)."""
    import pandas as pd
    files = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in CERT_FILES}
    return digest([CERT_STANDARD, files, pd.__version__])


def cert_current(s: dict, ver: str | None = None) -> bool:
    """Mẫu có chứng nhận ĐẠT theo đúng chuẩn + phiên bản hiện hành, cho đúng code và nguồn đang lưu (sửa code sau chứng nhận → hết hiệu lực)."""
    c = s.get("certified") or {}
    return bool(c.get("ok")) and c.get("standard") == CERT_STANDARD and c.get("version") == (ver or cert_version()) \
        and c.get("code_sha256") == digest(s["coder"]["code"]) and c.get("source_hash") == s["source_hash"]


def load_manual_audits(work: Path) -> dict:
    """Gộp mọi audit/audit_manual*.json (phase A, final, phase B/C…). Mỗi mục áp cho đúng code đã đọc khi audit (code_sha256);
    mục cũ không ghi hash được gán từ bản sao trạng thái ngay trước lúc audit (audit_code_hash.json)."""
    out = {}
    hashes = {}
    hp = work / "audit" / "audit_code_hash.json"
    if hp.exists():
        hashes = json.loads(hp.read_text(encoding="utf-8"))
    for f in sorted((work / "audit").glob("audit_manual*.json")):
        for k, v in json.loads(f.read_text(encoding="utf-8")).items():
            v = dict(v, file=f.name)
            v.setdefault("code_sha256", hashes.get(k))
            if k in out and out[k].get("ts", "") > v.get("ts", ""):
                continue
            out[k] = v
    return out


def manual_verdict(s: dict, mv: dict | None) -> str:
    """'pass' | 'fail' | 'stale' (code đã sửa sau audit: phải audit lại) | 'none'."""
    if not mv:
        return "none"
    meaning_bad = not all(mv.get(k, True) for k in ("ro_rang", "dung_nghia", "giu_nghia"))
    if meaning_bad:  # lỗi nghĩa câu: câu không đổi thì sửa code cũng không cứu
        return "fail"
    if mv.get("code_sha256") and mv["code_sha256"] != digest(s["coder"]["code"]):
        return "stale"
    return "pass" if mv["pass"] else "fail"


def exclusion(s: dict, manual: dict, ver: str) -> str | None:
    """Lý do không xuất một mẫu (None = xuất được): chưa nhận, chứng nhận không đúng chuẩn/phiên bản/code hiện hành, kiểm tay trượt hoặc
    hết hiệu lực (code đã sửa sau audit), auditor LLM gắn cờ mà chưa có kiểm tay xác nhận đạt."""
    if s["status"] != "accepted":
        return "not_accepted"
    if not cert_current(s, ver):
        return "certify_fail" if (s.get("certified") or {}).get("ok") is False else "cert_not_current"
    verdict = manual_verdict(s, manual.get(s["id"]))
    if verdict in ("fail", "stale"):
        return "manual_audit_" + verdict
    a = s.get("audit_llm") or {}
    if a and not a.get("missing") and not all(a.get(k) for k in ("ro_rang", "dung_nghia", "giu_nghia", "code_ok") if k in a) and verdict != "pass":
        return "llm_audit_flag"
    return None


def certify(work: Path, ledger: Path, workers: int = 6, recheck_all: bool = False) -> dict:
    """Chứng nhận lại toàn bộ mẫu đã nhận bằng bản ghi dựng lại từ hạt (không gọi API). Ghi certified vào state."""
    run = Runner(work, ledger, "main", llm.HARD_CAP)
    plan = {s["name"]: s for s in shop_plan()}
    ver = cert_version()
    acc = [s for s in run.samples.values() if s["status"] == "accepted" and (recheck_all or not cert_current(s, ver))]
    by_shop = collections.defaultdict(list)
    for s in acc:
        by_shop[s["shop"]].append(s)
    out = collections.Counter()

    def do(shop):
        sh = run.shop(plan[shop])
        res = []
        for s in by_shop[shop]:
            res.append((s, certify_one(run, sh, s)))
        with sh._mut_lock:
            sh._mut.clear()
        return res
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(do, sorted(by_shop)):
            for s, errs in res:
                s["certified"] = {"ok": not errs, "errors": errs, "ts": dt.datetime.now().isoformat(timespec="seconds"), "standard": CERT_STANDARD,
                                  "version": ver, "code_sha256": digest(s["coder"]["code"]), "source_hash": s["source_hash"]}
                if errs:  # trượt chuẩn hiện hành → loại hẳn để quota sinh bù, giữ bằng chứng
                    s["status"], s["reject"] = "rejected", ["certify:" + errs[0]]
                    hist(s, "certify_fail", errors=errs)
                out["ok" if not errs else "fail"] += 1
                for e in errs:
                    out["err:" + e.split(":")[0]] += 1
    run.save()
    return dict(out)



# ---------------------------------------------------------------- audit độc lập (phân tầng) + xuất dữ liệu
def explain(rec: dict, spec: dict) -> str:
    """Diễn giải đặc tả bằng lời (do code sinh, không phải teacher): kỳ, số đo, bộ lọc — để người/LLM kiểm câu ↔ đáp án."""
    f = lambda d: f"{d.day:02d}/{d.month:02d}/{d.year}"  # noqa: E731
    p = spec["params"]
    it = {i["item"]: i for i in rec["items"]}
    parts = []
    if spec.get("period"):
        s, e = W.resolve(spec["period"], spec["as_of"])
        parts.append(f"kỳ {W.period_text(spec['period'], spec['as_of'])} = {'từ đầu dữ liệu' if s.year < 2000 else f(s)} → hết ngày {f(e)}")
    if spec.get("base_period"):
        s, e = W.resolve(spec["base_period"], spec["as_of"])
        parts.append(f"kỳ so = {f(s)} → hết ngày {f(e)}")
    defs = {"revenue": "tổng thành tiền dòng hàng của đơn hoàn thành bán trong kỳ", "revenue_item": "doanh thu bán của riêng mặt hàng",
            "revenue_group": "doanh thu bán của các mặt hàng thuộc nhóm", "revenue_channel": "doanh thu bán của đơn qua kênh",
            "revenue_net": "doanh thu bán của kỳ trừ tiền hoàn có thời điểm hoàn trong kỳ (kể cả đơn gốc kỳ khác)",
            "refund_total": "tổng tiền hoàn có thời điểm hoàn trong kỳ", "n_orders": "số mã đơn hoàn thành", "n_cancelled": "số mã đơn nhóm hủy",
            "cancel_rate": "số đơn hủy / mọi đơn đặt trong kỳ × 100", "avg_order": "doanh thu bán / số đơn hoàn thành, làm tròn đồng",
            "qty_base": "tổng số lượng × hệ số quy đổi về đơn vị gốc, đơn hoàn thành", "top_items": "xếp mặt hàng theo số lượng gốc giảm dần, bằng nhau theo mã",
            "best_item_rev": "mặt hàng doanh thu bán cao nhất", "best_day": "ngày có doanh thu bán cao nhất (bằng nhau lấy ngày sớm)",
            "best_month": "tháng có doanh thu bán cao nhất (YYYY-MM)", "top_customer": "khách có tên mua nhiều tiền nhất, bỏ khách lẻ",
            "top_customers": "xếp khách có tên theo tiền mua, bỏ khách lẻ", "n_customers": "số khách có tên có đơn hoàn thành, bỏ khách lẻ",
            "exists_item_day": "có dòng bán mặt hàng trong đơn hoàn thành của ngày đó hay không", "group_table": "bảng (nhóm hàng, doanh thu bán) cho nhóm có bán",
            "unanswerable": "cửa hàng không ghi thông tin này → không trả lời được", "unsold_items": "số mặt hàng trong danh mục không bán được đơn vị nào",
            "rev_change_pct": "(doanh thu kỳ này − kỳ so) / kỳ so × 100", "share_group": "doanh thu nhóm / tổng doanh thu bán × 100",
            "days_over": "số ngày có doanh thu bán ≥ ngưỡng", "debt_total": "tổng nợ còn lại (bán chịu − đã trả) của các khách đến hết ngày",
            "top_debtor": "khách còn nợ nhiều nhất đến hết ngày", "purchase_money": "tổng tiền các dòng phiếu nhập trong kỳ",
            "lots_expiring": "số lô còn tồn có hạn dùng từ hôm nay đến hết ngày TODAY + N",
            "shift_rev": "doanh thu bán của các đơn hoàn thành lập trong giờ của ca", "shift_share": "doanh thu bán của ca / tổng doanh thu bán × 100",
            "staff_rev": "doanh thu bán của đơn hoàn thành do nhân viên lập", "staff_best": "nhân viên có doanh thu bán cao nhất (bằng nhau theo tên)",
            "price_on_day": "giá một đơn vị gốc của lần đổi giá gần nhất có ngày áp dụng ≤ ngày hỏi (chưa đổi: giá danh mục)",
            "price_change_pct": "(giá ngày sau − giá ngày trước) / giá ngày trước × 100", "avg_cost": "tổng tiền nhập / tổng số lượng gốc nhập trong kỳ, làm tròn đồng",
            "profit_gross": "doanh thu bán − Σ(số lượng gốc bán × giá vốn) của đơn hoàn thành trong kỳ",
            "stock_qty": "tồn đầu kỳ + số lượng gốc nhập − số lượng gốc bán (đơn hoàn thành) đến hết ngày", "debt_collected": "tổng tiền khách trả nợ trong kỳ",
            "debt_n_cust": "số khách còn nợ > 0 đến hết ngày", "debt_cust": "nợ còn lại của khách đến hết ngày",
            "g_rev_change": "(doanh thu nhóm kỳ này − kỳ so) / kỳ so × 100", "rev_list": "doanh thu bán từng tháng theo thứ tự tháng",
            "min_month": "tháng có doanh thu bán thấp nhất (bằng nhau lấy tháng sớm, YYYY-MM)", "avg_daily_rev": "doanh thu bán / số ngày của kỳ, làm tròn đồng",
            "orders_over": "số đơn hoàn thành có tổng thành tiền ≥ ngưỡng", "items_ge": "số mặt hàng có số lượng gốc bán ≥ N",
            "cust_ge_n": "số khách có tên có ≥ N đơn hoàn thành", "rx_rev": "doanh thu bán của các mặt hàng kê đơn",
            "rx_share": "doanh thu thuốc kê đơn / tổng doanh thu bán × 100"}
    parts.append("số đo: " + defs[spec["slot"]])
    if "item" in p:
        parts.append(f"mặt hàng: {it[p['item']]['name']} (đơn vị gốc {it[p['item']]['base']})")
    for k, lab in (("group", "nhóm"), ("channel", "kênh"), ("k", "k"), ("threshold", "ngưỡng"), ("n", "N"), ("missing", "thông tin hỏi"),
                   ("shift", "ca"), ("staff", "nhân viên"), ("customer", "khách")):
        if k in p:
            parts.append(f"{lab}: {p[k]}")
    return "; ".join(parts)


def audit_sample(run: "Runner", n_per_slot=4, seed=2026) -> list[dict]:
    """Mẫu audit phân tầng theo slot (kỹ năng) trên train+val đã chứng nhận: tối đa n_per_slot mỗi slot, ưu tiên rải family/độ khó."""
    rng = random.Random(seed)
    pool = [s for s in run.samples.values() if s["status"] == "accepted" and s["split"] in ("train", "val") and s.get("certified", {}).get("ok")]
    by = collections.defaultdict(list)
    for s in pool:
        by[s["slot"]].append(s)
    out = []
    for slot in sorted(by):
        xs = sorted(by[slot], key=lambda s: s["id"])
        rng.shuffle(xs)
        picked, fams = [], collections.Counter()
        for s in sorted(xs, key=lambda s: (fams[s["family"]], rng.random())):
            if len(picked) == n_per_slot:
                break
            if fams[s["family"]] <= min(fams.values() or [0]) or len(picked) >= len(set(x["family"] for x in xs)):
                picked.append(s)
                fams[s["family"]] += 1
        picked += [s for s in xs if s not in picked][:n_per_slot - len(picked)]
        out += picked
    return out


AUDIT_RULES = ("Với MỖI mẫu, đánh giá độc lập (không cần viết code):\n"
               "- ro_rang: câu hỏi (dạng văn phong, như chủ tiệm gõ) có một cách hiểu hợp lý duy nhất theo quy ước của cửa hàng không.\n"
               "- dung_nghia: diễn giải phép tính và đáp án chuẩn có đúng là thứ câu hỏi hỏi không (đúng số đo, kỳ, đối tượng, kiểu đáp án, quy ước).\n"
               "- giu_nghia: câu văn phong có giữ nguyên nghĩa của câu gốc không (viết tắt, không dấu, lỗi gõ không làm đổi nghĩa).\n"
               "Đáp án chuẩn do chương trình tính từ dữ liệu, bạn không cần tính lại con số; chỉ xét câu hỏi có khớp phép tính đã diễn giải không.\n"
               "Trả về đúng một JSON: [{\"id\": \"<id mẫu>\", \"ro_rang\": true/false, \"dung_nghia\": true/false, \"giu_nghia\": true/false, \"ly_do\": \"<ngắn, nêu lỗi nếu có>\"}, ...]")


def audit_prompt(groups: list[tuple[Shop, list[dict]]]) -> str:
    parts = ["Bạn là người kiểm định độc lập dữ liệu huấn luyện: câu hỏi của chủ cửa hàng nhỏ về dữ liệu bán hàng (dữ liệu giả lập tự tạo) kèm đáp án chuẩn.", AUDIT_RULES]
    for sh, items in groups:
        parts.append(f"\n## CỬA HÀNG {sh.rec['shop_name']} — hôm nay (TODAY) = {sh.today}\n{W.render(sh.rec, sh.s['family'])[1]}\n{W.rules_text(sh.rec, sh.s['family'])}")
        for s in items:
            g = s["spec"]["gold"]
            parts.append(f"- id: {s['id']}\n  câu văn phong: {s['question']}\n  câu gốc: {s['question_canonical']}\n"
                         f"  diễn giải: {explain(sh.rec, s['spec'])}\n  đáp án chuẩn: {json.dumps({k: g[k] for k in g if k in ('type', 'answer')}, ensure_ascii=False)}")
    return "\n".join(parts)


def run_audit(work: Path, ledger: Path, shops_per_call=5) -> dict:
    """Audit LLM độc lập (DeepSeek: khác họ teacher Gemini, verifier GPT, coder Qwen) trên mẫu phân tầng train/val + toàn bộ final đã chứng nhận."""
    llm.configure_run(ledger, limit=llm.HARD_CAP, provider_limits={"gemini": 4, "gpt": 4, "qwen": 4, "deepseek": 2})
    run = Runner(work, ledger, "main", llm.HARD_CAP)
    plan = {s["name"]: s for s in shop_plan()}
    chosen = audit_sample(run) + [s for s in run.samples.values() if s["split"] == "final" and s["status"] == "accepted" and s.get("certified", {}).get("ok")]
    write_json(work / "audit" / "audit_sample.json", [{"id": s["id"], "split": s["split"], "slot": s["slot"], "family": s["family"]} for s in chosen])
    by = collections.defaultdict(list)
    for s in chosen:
        by[s["shop"]].append(s)
    shops = sorted(by)
    verdicts = {}
    calls = 0
    for i in range(0, len(shops), shops_per_call):
        grp = [(run.shop(plan[name]), by[name]) for name in shops[i:i + shops_per_call]]
        r, hit = model_call(work, "audit", "+".join(shops[i:i + shops_per_call])[:120], "deepseek", audit_prompt(grp), think="off", max_tokens=12000)
        calls += not hit
        try:
            items = llm.extract_json(r["content"]) if r.get("finish") == "stop" else []
        except ValueError:
            items = []
        for x in items if isinstance(items, list) else []:
            if isinstance(x, dict) and x.get("id") in {s["id"] for _, ss in grp for s in ss}:
                verdicts[x["id"]] = {k: x.get(k) for k in ("ro_rang", "dung_nghia", "giu_nghia", "ly_do")}
    for s in chosen:
        v = verdicts.get(s["id"])
        s["audit_llm"] = v or {"missing": True}
    run.save()
    write_json(work / "audit" / "audit_deepseek.json", {s["id"]: s["audit_llm"] for s in chosen})
    return {"audited": len(chosen), "answered": len(verdicts), "calls": calls}



def validate_rows(path: Path) -> int:
    """Khuôn dòng SFT: đúng 5 khoá, 3 vai, meta đủ trường đúng kiểu cố định, answer_json parse được, code qua hợp đồng."""
    from build_sft_data import code_problem
    from datamate.llm import extract_code
    seen, n = set(), 0
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if set(r) != {"id", "kind", "shop", "messages", "meta"} or r["kind"] != "code_synth" or r["id"] in seen:
            raise ValueError("outer contract / duplicate id: " + str(r.get("id")))
        seen.add(r["id"])
        if [m["role"] for m in r["messages"]] != ["system", "user", "assistant"] or any(set(m) != {"role", "content"} or not m["content"] for m in r["messages"]):
            raise ValueError("messages: " + r["id"])
        meta = r["meta"]
        if set(meta) != set(META_FIELDS) or any(type(meta[k]) is not t for k, t in META_FIELDS.items()):
            raise ValueError("meta schema: " + r["id"])
        if any(not isinstance(x, str) for x in meta["edge_tags"] + meta["mutations"]):
            raise ValueError("meta list item type: " + r["id"])
        json.loads(meta["answer_json"])
        if code_problem(extract_code(r["messages"][-1]["content"]), meta["answer_type"]):
            raise ValueError("assistant code contract: " + r["id"])
        n += 1
    return n


def export(work: Path, ledger: Path, tokenizer_python: str, tokenizer: Path) -> dict:
    """Ghi clean/ (train/val/test_final SFT + samples_clean + schema + manifest + README), raw/, rejected/. Chỉ mẫu đã chứng nhận lại,
    không trượt audit thủ công. Messages dựng cục bộ bằng Kit.prompt(base0) — có few-shot của repo — không bao giờ gửi API."""
    run = Runner(work, ledger, "main", llm.HARD_CAP)
    plan = {s["name"]: s for s in shop_plan()}
    manual = load_manual_audits(work)
    ver = cert_version()
    out_dir, tmp = work / "clean", work / "clean_tmp"
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    rows, flat, excluded = [], [], collections.Counter()
    for s in sorted(run.samples.values(), key=lambda x: x["id"]):
        if s["status"] != "accepted":
            continue
        why = exclusion(s, manual, ver)
        if why:
            excluded[why] += 1
            continue
        mv = manual.get(s["id"])
        verdict = manual_verdict(s, mv)
        llm_a = s.get("audit_llm") or {}
        sh = run.shop(plan[s["shop"]])
        q = {"id": s["id"], "business_id": sh.name, "question": s["question"], "as_of_date": sh.today}
        messages = sh.kit().prompt(q, "base0")  # CỤC BỘ
        code = s["coder"]["code"]
        audit = "manual_pass" if verdict == "pass" else ("llm_pass" if llm_a and not llm_a.get("missing") else "not_audited")
        g = s["spec"]["gold"]
        meta = {"schema_version": "corpus_v1", "split": s["split"], "family": s["family"], "industry": s["industry"], "source_id": sh.name,
                "source_hash": s["source_hash"], "seed": int(s["seed"]), "slot": s["slot"], "skill": s["skill"], "answer_type": g["type"],
                "answer_json": json.dumps(g.get("answer"), ensure_ascii=False), "difficulty": int(s["difficulty"]), "time_kind": s["time_kind"],
                "relative_time": bool(s["relative"]), "edge_tags": list(s["edge_tags"]), "as_of_date": sh.today, "question": s["question"],
                "question_canonical": s["question_canonical"], "teacher": "gemini:high", "verifier": {"gpt": "gpt:low"}.get(s["verifier"].get("provider", "gpt"), s["verifier"].get("provider", "gpt") + ":off"), "coder": s["coder"]["model"] + (":high" if s["coder"]["model"] == "gemini" else ":off"),
                "coder_attempts": int(s["coder"].get("attempts", 1)), "mutations": [m["kind"] for m in s["mutations"]], "prompt_version": s["prompt_version"],
                "round": int(s["round_created"]), "audit": audit, "code_sha256": digest(code), "prompt_sha256": digest(messages), "tokens": 0,
                "cert_standard": s["certified"]["standard"], "cert_version": s["certified"]["version"], "shop_complex": bool(plan[s["shop"]].get("complex")),
                "schema_features": sorted(k for k, x in sh.rec["features"].items() if x)}
        rows.append({"id": s["id"], "kind": "code_synth", "shop": sh.name, "messages": messages + [{"role": "assistant", "content": "```python\n" + code + "\n```"}], "meta": meta})
        flat.append({"id": s["id"], **{k: meta[k] for k in ("split", "family", "industry", "source_id", "slot", "skill", "answer_type", "difficulty",
                                                             "time_kind", "relative_time", "edge_tags", "as_of_date", "question", "question_canonical",
                                                             "coder", "coder_attempts", "mutations", "prompt_version", "round", "audit",
                                                             "verifier", "source_hash", "seed", "cert_version", "shop_complex", "schema_features")},
                     "gold": g, "spec_explained": explain(sh.rec, s["spec"]), "code": code, "verifier_code": s["verifier"]["code"]})
    # đếm token thật (tokenizer Qwen3-14B đã ghim SHA, template enable_thinking=False, mask tiền tố)
    pend = tmp / "pending_rows.jsonl"
    pend.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    subprocess.run([tokenizer_python, str(ROOT / "eval/check_sft_tokens.py"), "--input", str(pend), "--tokenizer", str(tokenizer), "--out", str(tmp / "tokens.json")],
                   check=True, capture_output=True, text=True)
    tok = json.loads((tmp / "tokens.json").read_text(encoding="utf-8"))
    counts = {x["id"]: x for x in tok["rows"]}
    for r, f in zip(rows, flat):
        r["meta"]["tokens"] = counts[r["id"]]["tokens"]
        f["tokens"], f["prompt_tokens"], f["assistant_tokens"] = counts[r["id"]]["tokens"], counts[r["id"]]["prompt_tokens"], counts[r["id"]]["assistant_tokens"]
    pend.unlink()
    files = {"train": "train.jsonl", "val": "val.jsonl", "final": "test_final.jsonl", "final_b": "test_final_b.jsonl", "final_c": "test_final_c.jsonl"}
    sel_path = work / "selection_train.json"  # tập train chính chọn bám phân bố MINING; phần train sạch còn lại → train_extra.jsonl
    sel = set(json.loads(sel_path.read_text(encoding="utf-8"))["ids"]) if sel_path.exists() else None
    if sel is not None:
        files["train_extra"] = "train_extra.jsonl"
    for split, fn in files.items():
        if split == "train_extra":
            part = [r for r in rows if r["meta"]["split"] == "train" and r["id"] not in sel]
        else:
            part = [r for r in rows if r["meta"]["split"] == split and (split != "train" or sel is None or r["id"] in sel)]
        if split == "train":
            random.Random(2026).shuffle(part)
        (tmp / fn).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
        validate_rows(tmp / fn)
    (tmp / "samples_clean.jsonl").write_text("".join(json.dumps(f, ensure_ascii=False) + "\n" for f in flat), encoding="utf-8")
    types = {str: "string", int: "integer", bool: "boolean", list: "array"}
    schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "title": "DataMate synthetic code_synth row (pilot)", "type": "object",
              "additionalProperties": False, "required": ["id", "kind", "shop", "messages", "meta"],
              "properties": {"id": {"type": "string", "minLength": 1}, "kind": {"const": "code_synth"}, "shop": {"type": "string"},
                             "messages": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "object", "additionalProperties": False,
                                          "required": ["role", "content"], "properties": {"role": {"enum": ["system", "user", "assistant"]}, "content": {"type": "string", "minLength": 1}}}},
                             "meta": {"type": "object", "additionalProperties": False, "required": list(META_FIELDS),
                                      "properties": {k: {"type": types[t], **({"items": {"type": "string"}} if t is list else {})} for k, t in META_FIELDS.items()}}}}
    write_json(tmp / "schema.json", schema)
    raw_dir, rej_dir = work / "raw", work / "rejected"
    raw_dir.mkdir(exist_ok=True)
    rej_dir.mkdir(exist_ok=True)
    (raw_dir / "raw_samples.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False, default=str) + "\n" for s in run.samples.values()), encoding="utf-8")
    (rej_dir / "rejected.jsonl").write_text("".join(json.dumps({"id": s["id"], "split": s["split"], "shop": s["shop"], "slot": s["slot"], "difficulty": s["difficulty"],
                                                                 "answer_type": s["answer_type"], "time_kind": s["time_kind"], "reasons": s["reject"],
                                                                 "question_canonical": s["question_canonical"], "question": s["question"],
                                                                 "gold": s["spec"]["gold"], "round": s["round_created"]}, ensure_ascii=False) + "\n"
                                                     for s in run.samples.values() if s["status"] == "rejected"), encoding="utf-8")
    if out_dir.exists():
        import shutil
        shutil.move(str(out_dir), str(work / f"clean_prev_{dt.datetime.now():%H%M%S}"))
    tmp.rename(out_dir)
    return {"rows": len(rows), "by_split": dict(collections.Counter(r["meta"]["split"] for r in rows)), "excluded": dict(excluded),
            "train_selected": len(sel & {r["id"] for r in rows}) if sel is not None else None,
            "tokens_max": max(x["tokens"] for x in tok["rows"]) if rows else 0}



FROZEN_FILES = ["eval/synth_corpus.py", "eval/synth_world.py", "eval/synth_rubric.py", "eval/llm.py", "eval/model_bakeoff.py", "eval/synth_tools.py",
                "eval/synth_execution.py", "eval/stress_noise.py", "eval/synth_checks.py", "eval/build_sft_data.py", "eval/score.py",
                "eval/build_synth_source.py", "datamate/sandbox.py", "datamate/dm.py", "datamate/timeutil.py", "datamate/loader.py",
                "datamate/clean.py", "datamate/formatter.py", "datamate/verify.py", "datamate/agent.py", "datamate/prompts/system.md", "requirements.txt"]


def implementation_hash() -> dict:
    files = {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FROZEN_FILES}
    import numpy as np
    import pandas as pd
    return {"files": files, "combined": digest([files, VERIFIER]), "verifier": VERIFIER, "prompt_version": PROMPT_VERSION, "python": sys.version.split()[0],
            "pandas": pd.__version__, "numpy": np.__version__, "models": {p: llm.config(p)["model"] for p in llm.PROVIDERS}}


def freeze(work: Path, ledger: Path, name: str = "freeze.json") -> dict:
    """Đóng băng cấu hình sau vòng chính, trước final: final chỉ chạy khi code/prompt trùng hash này (không chỉnh theo final)."""
    path = work / name
    if path.exists():
        raise FileExistsError("đã đóng băng; không ghi đè")
    imp = implementation_hash()
    run = Runner(work, ledger, "main", llm.HARD_CAP)
    st = collections.Counter((s["split"], s["status"]) for s in run.samples.values())
    rec = {"ts": dt.datetime.now().isoformat(timespec="seconds"), "ledger": ledger_count(ledger), "implementation": imp,
           "samples": {f"{a}:{b}": n for (a, b), n in sorted(st.items())}}
    write_json(path, rec)
    return {"combined": imp["combined"], "ledger": rec["ledger"]}


if __name__ == "__main__":
    main()
