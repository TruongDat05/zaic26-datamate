"""Kiểm đầu ra của teacher cho giai đoạn 1 (dùng trong eval/model_bakeoff.py và bộ sinh dữ liệu shop).

Mỗi hàm trả danh sách lỗi (rỗng = đạt). Tên lỗi ngắn, cố định, để đếm được theo model/ngành.
"""
from __future__ import annotations

import re
import unicodedata

from synth_spec import DIACRITIC, LOCAL_BLOCKLIST

HAS_KEYS = ("customers", "staff", "debt", "purchase", "stock", "lots", "deliveries", "expenses")


def fold(s: str) -> str:
    """Bỏ dấu, chữ thường, gộp khoảng trắng — để bắt tên chỉ khác dấu."""
    s = unicodedata.normalize("NFD", str(s).replace("đ", "d").replace("Đ", "D"))
    return re.sub(r"\s+", " ", "".join(c for c in s if unicodedata.category(c) != "Mn")).strip().lower()


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _pair(x, lo, hi, strict=False) -> bool:
    return (isinstance(x, list) and len(x) == 2 and all(_num(v) for v in x) and lo <= x[0] <= x[1] <= hi
            and (x[0] < x[1] if strict else True))


INDUSTRY_CONTEXT = r"(?:cửa hàng|cua hang|tiệm|tiem|shop|ngành|nganh|quầy|đại lý|dai ly|bán|ban|kinh doanh|chuyên|chuyen)\s+(?:\w+\s+){0,2}"


def blocklist_hits(text: str, strict: bool = False) -> list[str]:
    """Từ trong LOCAL_BLOCKLIST có trong text. strict=True: khớp chuỗi con (tên hàng, nhóm hàng, luật).
    strict=False (văn bản tự do): tên ngành chỉ tính khi đứng sau từ chỉ ngành ('cửa hàng điện thoại', 'tiệm sửa xe'),
    để 'số điện thoại', 'đặt qua điện thoại' không bị bắt nhầm; khái niệm vẫn khớp chuỗi con."""
    low = text.lower()
    hits = [t for t in LOCAL_BLOCKLIST["khai_niem"] if t in low]
    for t in LOCAL_BLOCKLIST["nganh"]:
        if (t in low) if strict else re.search(INDUSTRY_CONTEXT + re.escape(t), low):
            hits.append(t)
    return hits


# Lỗi "nhẹ": không làm hỏng bộ mô phỏng (sửa tay/bổ sung được). Còn lại là lỗi "nặng": đầu ra không dùng được.
SOFT = {"so_nhom", "pho_bien_khong_lech", "luat_nganh", "service_lines", "mo_ta_dai"}


def hard_errors(errs: list[str]) -> list[str]:
    return [e for e in errs if e not in SOFT]


def _to_num(x):
    if isinstance(x, str):
        s = x.strip().replace("đ", "").replace("VND", "").replace(" ", "")
        if re.fullmatch(r"-?\d{1,3}([.,]\d{3})+", s):  # 85.000 / 85,000
            return int(re.sub(r"[.,]", "", s))
        try:
            v = float(s.replace(",", "."))
            return int(v) if v == int(v) else v
        except ValueError:
            return x
    return x


def _to_hour(x):
    if isinstance(x, str):
        m = re.fullmatch(r"\s*(\d{1,2})(?:[:hg](\d{2}))?\s*", x)
        if m:
            return int(m.group(1)) + (int(m.group(2)) / 60 if m.group(2) else 0)
    return _to_num(x)


def _to_pair(x, conv=_to_num):
    if isinstance(x, str):
        parts = re.split(r"\s*(?:-|–|,|đến|to)\s*", x.strip().strip("[]()"))
        if len(parts) == 2:
            return [conv(p) for p in parts]
        v = conv(x)
        return [v, v] if not isinstance(v, str) else x
    if isinstance(x, dict) and len(x) == 2:
        return [conv(v) for v in x.values()]
    if isinstance(x, list):
        return [conv(v) for v in x]
    if isinstance(x, (int, float)):
        return [x, x]
    return x


WEEKDAY_KEYS = [("t2", "thu2", "thứ 2", "monday", "mon"), ("t3", "thu3", "thứ 3", "tuesday", "tue"), ("t4", "thu4", "thứ 4", "wednesday", "wed"),
                ("t5", "thu5", "thứ 5", "thursday", "thu"), ("t6", "thu6", "thứ 6", "friday", "fri"), ("t7", "thu7", "thứ 7", "saturday", "sat"),
                ("cn", "chu nhat", "chủ nhật", "sunday", "sun")]


def normalize_catalog(c):
    """Sửa kiểu dữ liệu hay gặp ở đầu ra teacher (số trong ngoặc kép, '[a, b]' dạng chuỗi, giờ '07:00', hệ số thứ dạng chuỗi/dict,
    dòng tiêu đề trong bảng luật). Trả (bản đã chuẩn hoá, danh sách việc đã sửa). Không bịa giá trị thiếu."""
    import copy
    if not isinstance(c, dict):
        return c, []
    c, fixed = copy.deepcopy(c), set()

    def put(obj, key, new, tag):
        if obj.get(key) != new:
            obj[key] = new
            fixed.add(tag)

    for it in c.get("items") or []:
        if not isinstance(it, dict):
            continue
        put(it, "price_vnd", _to_pair(it.get("price_vnd")), "gia")
        for k in ("cost_ratio", "popularity"):
            put(it, k, _to_num(it.get(k)), k)
        bu = it.get("big_units")
        if isinstance(bu, list):
            nb = [dict(u, factor=_to_num(u.get("factor"))) for u in bu if isinstance(u, dict)]
            nb = [u for u in nb if not (isinstance(u["factor"], (int, float)) and u["factor"] <= 1)]
            put(it, "big_units", nb, "quy_doi")
        elif bu is None:
            put(it, "big_units", [], "quy_doi")
    put(c, "open_hours", _to_pair(c.get("open_hours"), _to_hour), "gio_mo")
    for k in ("orders_per_day", "lines_per_order"):
        put(c, k, _to_pair(c.get(k)), k)
    s = c.get("seasonality")
    if isinstance(s, dict):
        put(s, "tet_multiplier", _to_num(s.get("tet_multiplier")), "he_so_tet")
        wd = s.get("weekday")
        if isinstance(wd, str):
            wd = [_to_num(x) for x in re.split(r"[,;\s]+", wd.strip("[] ")) if x]
        elif isinstance(wd, dict):
            low = {fold(k).replace("_", " "): v for k, v in wd.items()}
            vals = [next((low[k] for k in keys if k in low), None) for keys in WEEKDAY_KEYS]
            wd = [_to_num(v) for v in vals] if None not in vals else list(wd.values())
        elif isinstance(wd, list):
            wd = [_to_num(x) for x in wd]
        put(s, "weekday", wd, "he_so_thu")
        if isinstance(s.get("events"), list):
            put(s, "events", [dict(e, he_so=_to_num(e.get("he_so"))) if isinstance(e, dict) else e for e in s["events"]], "dip_le")
    r = c.get("rules_section_6")
    if isinstance(r, list):
        rows = []
        for x in r:
            if not isinstance(x, str) or re.fullmatch(r"\s*\|?\s*:?-{3,}.*", x) or re.search(r"Luật\s*\|\s*Mặc định", x):
                continue
            x = x.strip()
            if x.count("|") == 1 or (x.count("|") >= 1 and not x.startswith("|")):
                x = "| " + x.strip("| ") + " |"
            rows.append(x)
        put(c, "rules_section_6", rows, "luat_nganh")
    return c, sorted(fixed)


def check_catalog(c, n_items: int) -> list[str]:
    """Danh mục ngành mới theo synth_spec.CATALOG_SCHEMA (prompts/7_danh_muc_nganh_moi.md)."""
    if not isinstance(c, dict):
        return ["khong_phai_object"]
    err = []
    if not re.fullmatch(r"[a-z][a-z0-9_]*", str(c.get("industry", ""))):
        err.append("ma_nganh")
    items = c.get("items")
    if not isinstance(items, list) or not items:
        return err + ["thieu_items"]
    if len(items) < 0.9 * n_items:
        err.append("thieu_mat_hang")
    bad = {"truong_mat_hang": 0, "gia": 0, "gia_von": 0, "pho_bien": 0, "quy_doi": 0}
    for it in items:
        if not isinstance(it, dict) or not all(isinstance(it.get(k), str) and it.get(k).strip() for k in ("name", "group", "base_unit")):
            bad["truong_mat_hang"] += 1
            continue
        if not _pair(it.get("price_vnd"), 100, 500_000_000):
            bad["gia"] += 1
        if not (_num(it.get("cost_ratio")) and 0.3 <= it["cost_ratio"] <= 0.95):
            bad["gia_von"] += 1
        if not (_num(it.get("popularity")) and it["popularity"] == int(it["popularity"]) and 1 <= it["popularity"] <= 5):
            bad["pho_bien"] += 1
        bu = it.get("big_units", [])
        if not isinstance(bu, list) or any(not isinstance(u, dict) or not isinstance(u.get("unit"), str) or not _num(u.get("factor"))
                                            or u["factor"] != int(u["factor"]) or u["factor"] <= 1 for u in bu):
            bad["quy_doi"] += 1
    err += [k for k, v in bad.items() if v]
    names = [fold(it.get("name", "")) for it in items if isinstance(it, dict)]
    if len(set(names)) < len(names):
        err.append("trung_ten")
    groups = {fold(it.get("group", "")) for it in items if isinstance(it, dict)}
    if not 5 <= len(groups) <= 12:
        err.append("so_nhom")
    pops = [it.get("popularity") for it in items if isinstance(it, dict) and _num(it.get("popularity"))]
    if pops and sum(p >= 4 for p in pops) > 0.5 * len(pops):
        err.append("pho_bien_khong_lech")
    for k in ("customer_groups", "channels", "pay_methods"):
        if not (isinstance(c.get(k), list) and c[k] and all(isinstance(x, str) for x in c[k])):
            err.append(k)
    if not isinstance(c.get("service_lines", []), list):
        err.append("service_lines")
    if not _pair(c.get("open_hours"), 0, 24, strict=True):
        err.append("gio_mo")
    if not _pair(c.get("orders_per_day"), 1, 2000):
        err.append("don_ngay")
    if not _pair(c.get("lines_per_order"), 1, 60):
        err.append("dong_don")
    s = c.get("seasonality")
    if not isinstance(s, dict):
        err.append("mua_vu")
    else:
        if not (_num(s.get("tet_multiplier")) and 0.1 <= s["tet_multiplier"] <= 5):
            err.append("he_so_tet")
        wd = s.get("weekday")
        if not (isinstance(wd, list) and len(wd) == 7 and all(_num(x) and 0 < x <= 5 for x in wd)):
            err.append("he_so_thu")
        ev = s.get("events", [])
        if not isinstance(ev, list) or any(not isinstance(e, dict) or not re.fullmatch(r"(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])", str(e.get("thang_ngay", "")))
                                           or not (_num(e.get("he_so")) and 0.1 <= e["he_so"] <= 10) for e in ev):
            err.append("dip_le")
    has = c.get("has")
    if not (isinstance(has, dict) and all(isinstance(has.get(k), bool) for k in HAS_KEYS)):
        err.append("has")
    rules = c.get("rules_section_6")
    if not (isinstance(rules, list) and 2 <= len(rules) <= 4 and all(isinstance(r, str) and re.fullmatch(r"\s*\|[^|]+\|[^|]+\|\s*", r) for r in rules)):
        err.append("luat_nganh")
    import json
    strict_part = json.dumps([[it.get("name"), it.get("group")] for it in items if isinstance(it, dict)]
                             + [c.get("ten"), c.get("service_lines"), c.get("rules_section_6")], ensure_ascii=False)
    if blocklist_hits(strict_part, strict=True) or blocklist_hits(json.dumps(c, ensure_ascii=False)):
        err.append("blocklist")
    return err


PROFILE_KEYS = ("ten_shop", "tinh_thanh", "quan_huyen", "chu_tiem", "mo_ta", "gio_mo", "kenh_ban", "so_nhan_vien", "quy_mo")


def check_profiles(p, n: int) -> list[str]:
    """Hồ sơ shop (việc dễ 1.1)."""
    if not isinstance(p, list):
        return ["khong_phai_list"]
    err = []
    if len(p) != n:
        err.append("so_ho_so")
    bad = {"thieu_khoa": 0, "chu_tiem": 0, "gio_mo": 0, "nhan_vien": 0, "quy_mo": 0, "kenh_ban": 0, "mo_ta_dai": 0}
    for x in p:
        if not isinstance(x, dict) or any(k not in x for k in PROFILE_KEYS):
            bad["thieu_khoa"] += 1
            continue
        if not (isinstance(x["chu_tiem"], str) and len(x["chu_tiem"].split()) >= 2 and DIACRITIC.search(x["chu_tiem"].lower())):
            bad["chu_tiem"] += 1
        if not (_pair(x["gio_mo"], 0, 24, strict=True) and all(v == int(v) for v in x["gio_mo"])):
            bad["gio_mo"] += 1
        if not (_num(x["so_nhan_vien"]) and x["so_nhan_vien"] == int(x["so_nhan_vien"]) and 0 <= x["so_nhan_vien"] <= 20):
            bad["nhan_vien"] += 1
        if x["quy_mo"] not in ("nhỏ", "vừa"):
            bad["quy_mo"] += 1
        if not (isinstance(x["kenh_ban"], list) and x["kenh_ban"]):
            bad["kenh_ban"] += 1
        if not isinstance(x["mo_ta"], str) or len(x["mo_ta"].split()) > 40:
            bad["mo_ta_dai"] += 1
    err += [k for k, v in bad.items() if v]
    ok = [x for x in p if isinstance(x, dict)]
    for k in ("ten_shop", "chu_tiem"):
        vals = [fold(x.get(k, "")) for x in ok]
        if len(set(vals)) < len(vals):
            err.append(f"trung_{k}")
    import json
    if blocklist_hits(json.dumps(p, ensure_ascii=False)):
        err.append("blocklist")
    return err


# ---------------------------------------------------------------- mẫu dữ liệu train (T4): kiểm câu hỏi
# Chạy trên câu gốc (đủ dấu) của teacher; so khớp trên chữ đã bỏ dấu (fold).
_NGAY = r"\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}(?:-\d{2})?\b|\b\d{1,2}/\d{4}\b|\b(?:thang|quy|t|q)\s*\d{1,2}\s*(?:/|-|nam)\s*\d{4}\b|\bnam\s+\d{4}\b"
_GOI_Y = re.compile(
    r"\b(khong tinh|khong ke|bo qua|bo|tru|loai|ngoai)\s+(ra\s+)?(cac\s+)?(don\s+(cua\s+)?)?(khach le|khach vang lai|vang lai)\b"
    r"|\bchi\s+(tinh|lay|xet)\s+(cac\s+)?(don|hoa don)\s+(da\s+)?(thanh cong|hoan thanh|hoan tat|da giao|giao xong|khong huy)\b"
    r"|\b(khong tinh|bo|loai|tru)\s+(cac\s+)?(don|hoa don)\s+(bi\s+)?huy\b"
    r"|\bquy\s+(doi\s+)?(ve|ra)\s+(theo\s+)?(dvt|don vi)\b|\btheo\s+(dvt|don vi)\s+(goc|co ban|nho nhat)\b"
    r"|\blam tron\b|\bcot\b|\btrung\s+(nhau\s+)?thi\b")


def ky_ro(question: str) -> bool:
    """Câu nêu rõ kỳ hoặc mốc ngày (tháng/quý kèm năm, năm, ngày đủ năm): không phụ thuộc 'hôm nay'."""
    return bool(re.search(_NGAY, fold(question)))


def lo_dap_an(question: str, ans: dict) -> bool:
    """Câu hỏi chứa sẵn đáp án (ans đã qua formatter): số ≥ 100 hoặc có phần lẻ viết trong câu (bỏ số thuộc ngày tháng); ngày/tháng
    đáp án có trong câu; chuỗi (≥ 3 ký tự) hoặc mọi phần tử chữ của list có trong câu."""
    t, v, s = ans["type"], ans["answer"], fold(question)
    if t in ("number", "percent"):
        if isinstance(v, bool) or (abs(v) < 100 and v == int(v)):
            return False
        s = re.sub(r"(?<=\d)[.,\s](?=\d{3}(?!\d))", "", re.sub(_NGAY, " ", s))  # bỏ ngày tháng, dấu ngăn nghìn
        return any(abs(float(x.replace(",", ".")) - v) < 1e-9 for x in re.findall(r"-?\d+(?:[.,]\d+)?", s))
    if t == "date":
        y, m, *d = v.split("-")
        pats = [v] + ([f"{int(d[0])}/{int(m)}/{y}", f"{d[0]}/{m}/{y}"] if d else [f"{int(m)}/{y}", f"{m}/{y}", f"thang {int(m)} nam {y}"])
        return any(re.search(rf"(?<![\d/]){re.escape(p)}(?![\d/])", s) for p in pats)
    items = [fold(x) for x in ([v] if t == "string" else v if t == "list" else []) if isinstance(x, str)]
    return bool(items) and all(len(x) >= 3 and x in s for x in items)


def codes_of(T: dict) -> set[str]:
    """Mã nội bộ có trong bảng: giá trị chữ + số (KH01, HD000123, DB-00012) hoặc chữ in hoa 2–4 ký tự (KL, HT) — chủ tiệm không hỏi bằng mã."""
    pat = re.compile(r"[A-Za-z]{1,5}[-_]?\d{2,}|[A-Z]{2,4}")
    return {v.strip().lower() for df in T.values() for c in df.columns for v in df[c].dropna().unique()
            if isinstance(v, str) and pat.fullmatch(v.strip())}


def ma_noi_bo(question: str, codes: set[str]) -> list[str]:
    return sorted(set(re.findall(r"[\w-]+", question.lower())) & codes)


def goi_y(question: str) -> list[str]:
    """Gợi ý cách giải viết sẵn trong câu (không tính khách lẻ, chỉ tính đơn hoàn thành, quy về đơn vị gốc, làm tròn, tên cột, cách xử lý
    bằng nhau): quy ước đã nói, câu hỏi thật không nhắc lại — mẫu có gợi ý dạy model dựa vào gợi ý."""
    return [m.group(0) for m in _GOI_Y.finditer(fold(question))]
