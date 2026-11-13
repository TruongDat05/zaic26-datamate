"""Trả lời trực tiếp các dạng câu cơ bản bằng lớp tính toán (shop.py), không qua model: một đường độc lập với code model viết.

Chỉ trả lời khi nhận chắc dạng câu; có từ ngoài khung thì trả None để model xử lý. Độ chính xác đo bằng eval/template_eval.py.
"""
from __future__ import annotations

import datetime as dt
import re

from . import timeutil as tu
from .clean import norm_key
from .normq import _MONTH_WORD, has_unread_typo, normalize, shop_words
from .shop import round2

_MONTH = re.compile(r"\b(?:(?:thang|th|thg|t)\s*)?(\d{1,2})\s*(?:/|-|\s*nam\s*|\s)\s*(20\d{2}|2[4-7])\b")
# câu public viết tắt nhiều hơn train
_SLANG = [('\\bhqua\\b', 'hom qua'), ('\\bhnay\\b|\\bhum nay\\b|\\bhom ni\\b', 'hom nay'), ('\\btrc\\b', 'truoc'), ('\\bbnhieu\\b|\\bbao nhiu\\b|\\bbnh\\b', 'bao nhieu'), ('\\bdc\\b', 'duoc'), ('\\bko\\b|\\bk\\b', 'khong'), ('\\bdon hang\\b', 'don'), ('\\b(?:tinh )?tu(?: ngay)? 1[ /]1 (?:toi|den)(?: het)? hom nay\\b', 'tu dau nam den het hom nay'), ('\\bnam nay\\b', ''), ('\\bdthu\\b', 'doanh thu')]
_QTY_WORD = r"(may|bao nhieu|bao nhiu|bnhiu|bn|so)"


def find_name(q: str, names: list[str]):
    """Tên dài nhất xuất hiện nguyên văn (bỏ dấu, hoa thường) trong câu hỏi; hai tên cùng độ dài thì không đoán."""
    hits = []
    for n in names:
        k = norm_key(n)
        # tên một từ ngắn ('Trà', 'Cá') trùng từ thường ('trả bằng', 'ca sáng'): phải có chữ 'nhóm', 'món' đứng trước
        pre = r"(?:nhom|nhom hang|mon|mat hang|san pham|hang) " if " " not in k and len(k) <= 4 else r"(?:^|\s)"
        if k and re.search(pre + re.escape(k) + r"($|\s)", q):
            hits.append((len(k), n))
    if not hits:
        return None
    hits.sort(reverse=True)
    if len(hits) > 1 and hits[0][0] == hits[1][0]:
        return None
    return hits[0][1]


def _clip(p, today):
    return tu.date_range(p.start, today) if p.end > today else p


def _ans(t, v, name):
    return {"type": t, "answer": v, "template": name}


# chữ đệm: sau khi bỏ tên, kỳ và từ khoá của khung mà còn chữ ngoài danh sách này thì câu có ý khác, không đoán
_FILL = set("""cho minh toi em anh chi ban hoi xem giup gium dum voi nhe nhi nha a ah ha vay z ta day do roi the ma nao oi coi check tra cuu
    tinh biet duoc khong ko hok la bao nhieu nhiu bn bnhiu may so co trong cua tai o vao tong cong toan bo het ca hang don vi dong vnd
    tien gi su dung thang nam ky luc nay nhung chot len tat don oi gium ne thoi cac mik z zi ha hen e ua rua wan kua sop luon nhat nhanh ti 1
    giup dum thi va ve em mot nhu mi kho tong rieng ee ad ban liet ke mat ra vay minh tat ca voi
    hi he ho ru dzay dza nghe nheng chu cha cai bit xiu nho chi ban tiem dat""".split())


_FILL -= set('ad bit chot cuu dum dza dzay kua mik nhanh nheng nhu sop su toan vnd wan zi'.split())  # không có trong câu hỏi train; giữ tiểu từ thông dụng của tiếng Việt (em, ơi, nhé, nè, hen, ừa)


def _only_fill(rest: str, *keywords: str) -> bool:
    words = set(rest.split())
    for k in keywords:
        words -= set(k.split())
    return not (words - _FILL)


_D = r"(\d{1,2})[ /-](\d{1,2})[ /-](20\d{2})"
_TODAY_WORDS = r"(?:hom nay|hnay|bua nay|hom ni|nay)"
_PERIODS = [
    (re.compile(r"\b(?:giai doan |khoang )?tu(?: ngay)? " + _D + r" (?:den|toi)(?: het)?(?: ngay)? " + _D),
     lambda m, t: tu.date_range(dt.date(int(m[3]), int(m[2]), int(m[1])), dt.date(int(m[6]), int(m[5]), int(m[4])))),
    (re.compile(r"\btu dau nam (?:den|toi)(?: het)? " + _TODAY_WORDS + r"\b|\btu dau nam (?:toi nay|nay)?\b"), lambda m, t: tu.year_to_date(t)),
    (re.compile(r"\btu dau thang (?:nay )?(?:den|toi)(?: het)? " + _TODAY_WORDS + r"\b"), lambda m, t: tu.this_month(t)),
    (re.compile(r"\bquy (\d)(?: nam | )(20\d{2})\b"), lambda m, t: tu.quarter(int(m[2]), int(m[1]))),
    (re.compile(r"\b(\d+) ngay (?:qua|gan day|gan nhat|vua roi)\b"), lambda m, t: tu.last_n_days(int(m[1]), t)),
    (re.compile(r"\b(?:ngay )?" + _D + r"\b"), lambda m, t: tu.day(dt.date(int(m[3]), int(m[2]), int(m[1])))),
    (_MONTH, lambda m, t: tu.month(_year(m[2]), int(m[1]))),
    (re.compile(r"\b(?:dip |mua )?tet(?: am lich| nguyen dan)?(?: nam)? (20\d{2})\b"), lambda m, t: _cal_tet(int(m[1]))),
    (re.compile(r"\bnam (20\d{2})\b"), lambda m, t: tu.year(int(m[1]))),
    (re.compile(r"\bthang nay\b"), lambda m, t: tu.this_month(t)),
    (re.compile(r"\bthang (?:truoc|roi)\b"), lambda m, t: tu.last_month(t)),
    (re.compile(r"\btuan nay\b"), lambda m, t: tu.this_week(t)),
    (re.compile(r"\btuan (?:truoc|roi)\b"), lambda m, t: tu.last_week(t)),
    (re.compile(r"\b(?:tinh rieng )?(?:ngay )?(?:hom nay|hnay|bua nay|hom ni)\b"), lambda m, t: tu.day(t)),
    (re.compile(r"\bhom qua\b"), lambda m, t: tu.yesterday(t)),
    (re.compile(r"\b(?:thang|th|t) (1[0-2]|[1-9])\b"), lambda m, t: tu.month_no_year(int(m[1]), t)),
    (re.compile(r"\bthang (mot|gieng|hai|ba|tu|nam|sau|bay|tam|chin|muoi mot|muoi hai|muoi|chap)\b"), lambda m, t: tu.month_no_year(_MONTH_WORD[m[1]], t)),
]
_COMPARE = re.compile(r"\b(?:so(?: voi| vs)?|hon hay kem|cao hay thap hon|tang hay giam so voi)\s+(?:cung ky\s+)?(?:thang|th|tuan|quy|ky)?\s*(?:lien )?(?:truoc|roi)(?: do)?\b")


def _cal_tet(y: int):
    from . import dm
    if dm.cal is None:
        raise ValueError("chưa nạp lịch âm")
    return dm.tet(y)


def _year(y: str) -> int:
    return int(y) if len(y) == 4 else 2000 + int(y)


def _parse_period(q: str, today):
    """(Period, câu đã bỏ phần kỳ); câu nêu từ hai kỳ trở lên hoặc không nêu kỳ thì None."""
    found, rest = [], q
    for pat, make in _PERIODS:
        for m in pat.finditer(rest):
            try:
                found.append(make(m, today))
            except ValueError:
                return None
        rest = pat.sub(" ", rest)
    return (found[0], rest) if len(found) == 1 else None


_CHANNELS = [("pay", r"tien mat|cash", r"\btien mat\b|\btm\b|\btien tuoi\b"), ("pay", r"chuyen khoan|transfer|bank|\bck\b", r"\bchuyen khoan\b|\bck\b"),
             ("pay", r"\bvi\b|wallet|momo", r"\bvi dien tu\b"), ("pay", r"\bcod\b|thu ho", r"\bcod\b|\bxi o\b"),
             ("channel", r"store|tai quay|offline|tai cua hang", r"\btai quay\b"), ("channel", r"phone|dien thoai", r"\bdien thoai\b"),
             ("channel", r"zalo", r"\bzalo(?: oa)?\b")]


def _channel_values(rest: str) -> list:
    """(cột, mẫu giá trị, cụm trong câu), theo thứ tự câu hỏi nêu."""
    found = []
    for col, pat, phrase in _CHANNELS:
        m = re.search(phrase, rest)
        if m:
            found.append((m.start(), col, pat, m.group(0)))
    return [(c, p, ph) for _, c, p, ph in sorted(found)]


def _out_of_data(shop, p) -> bool:
    return p.end < shop.lines["date"].min().date() or p.start > shop.lines["date"].max().date()


def answer(question: str, as_of: str, shop) -> dict | None:
    """Đáp án {'type', 'answer', 'template'} hoặc None. Chỉ trả khi dòng luật của dạng câu còn trong rulebook của bộ đang chạy."""
    from .templates_ext import rule_ok
    r = _answer(question, as_of, shop)
    return r if r is not None and rule_ok(r["template"]) else None


def _answer(question: str, as_of: str, shop) -> dict | None:
    if not shop.reliable or has_unread_typo(normalize(question, _keep(shop)), _keep(shop)):
        return None
    r = _answer_base(question, as_of, shop)
    if r is not None:
        return r
    from .templates_ext import answer_ext
    return answer_ext(question, as_of, shop)


def _keep(shop) -> set:
    if "_keep" not in shop.__dict__:
        names = [shop.lines[c].dropna().astype(str).unique() for c in ("cust", "staff") if c in shop.lines]
        shop.__dict__["_keep"] = shop_words(shop.items, shop.groups, *names)
    return shop.__dict__["_keep"]


def _answer_base(question: str, as_of: str, shop) -> dict | None:
    q = normalize(question, _keep(shop))
    for pat, rep in _SLANG:
        q = re.sub(pat, rep, q)
    today = dt.date.fromisoformat(as_of)
    is_rate = "%" in question or bool(re.search(r"phan tram|\bti le\b|\bty le\b|ti trong|ty trong", q))
    is_rev = bool(re.search(r"\b(doanh thu|dthu|dt|thu ve bao nhieu|tien ban|tien hang|(bao nhieu|bao nhiu|bnhiu|bn|nhieu) tien)\b", q))

    # --- doanh thu N tháng liền nhau, mỗi tháng
    m = re.search(r"(\d+) thang lien (?:nhau|tiep) ket thuc (?:o|tai|vao)? ?thang (\d{1,2})(?:/| nam | )(20\d{2})", q)
    if m and is_rev and not is_rate:
        ps = tu.months_ending(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        if ps[-1].start > today or any(_out_of_data(shop, x) for x in ps):
            return None
        return _ans("list", [shop.revenue(_clip(x, today)) for x in ps], "rev_months")

    compare = bool(_COMPARE.search(q))
    parsed = _parse_period(_COMPARE.sub(" ", q), today)
    if parsed is None or parsed[0].start > today:
        return None
    p, rest = _clip(parsed[0], today), parsed[1]
    if _out_of_data(shop, p):
        return None
    item = find_name(q, shop.items)
    group = find_name(q, shop.groups) if item is None else None
    if re.search(r"\bnhom\b", q) and group is None and find_name(q, shop.groups):  # 'nhóm sữa' và có mặt hàng tên 'Sữa'
        item, group = None, find_name(q, shop.groups)
    for name in filter(None, [item, group]):
        rest = rest.replace(norm_key(name), " ")
    if re.search(r"\bnhom\b", q) and group is None:
        return None
    rest = re.sub(r"\bnhom( hang)?\b", " ", rest)
    rest = re.sub(r"\b(cua hang|cuwar hangf|tiem|quan|shop|nha thuoc|hieu thuoc)\b", " ", rest)

    # --- doanh thu so với kỳ trước
    if compare:
        if is_rev and item is None and group is None and _only_fill(rest, "doanh thu dthu tang giam thay doi phan tram ti le ty hay hon kem chenh lech len xuong"):
            prev = tu.previous(p)
            if _out_of_data(shop, prev):
                return None
            new, old = shop.revenue(p), shop.revenue(prev)
            if is_rate:
                return _ans("percent", round2((new - old) / old * 100), "rev_change_pct") if old else None
            return _ans("number", new - old, "rev_change")
        return None

    # --- ca sáng, ca chiều
    ca = re.findall(r"\bca (sang|chieu)\b", rest)
    if ca and item is None and group is None and shop.lines["hour"].notna().any():
        r2 = re.sub(r"\bca (sang|chieu)\b", " ", rest)
        kw = "doanh thu dthu ban tien tang giam hay hon kem chenh lech thay doi phan tram ti le ty voi vs nhieu it cao thap dem"
        if not _only_fill(r2, kw):
            return None
        is_ord = bool(re.search(r"\b(don|hoa don)\b", r2)) and not is_rev
        if not (is_rev or is_ord):
            return None
        m_ = (lambda d: d["order"].nunique()) if is_ord else (lambda d: d["amount"].sum())
        if len(set(ca)) == 2:
            if not is_rate:
                return None
            base = m_(shop.by_shift(p, "sang"))
            return _ans("percent", round2((m_(shop.by_shift(p, "chieu")) - base) / base * 100), "shift_change") if base else None
        if is_rate:
            return None
        v = m_(shop.by_shift(p, ca[0]))
        return _ans("number", int(v), "shift")

    # --- kênh bán, hình thức thanh toán
    vals = _channel_values(rest)
    if vals and item is None and group is None:
        r2 = rest
        for _, _, phrase in vals:
            r2 = r2.replace(phrase, " ")
        if not _only_fill(r2, "doanh thu dthu ban tra bang qua tai chiem phan tram ti le ty tong loai ben moi lan luot so voi rieng"):
            return None
        out = []
        for col, pat, _ in vals:
            if col not in shop.lines:
                return None
            x = shop.counted(p)
            hit = x[col].map(lambda v: bool(re.search(pat, norm_key(v))))
            if not hit.any():
                return None
            out.append(x.loc[hit, "amount"].sum())
        if is_rate:
            tot = shop.counted(p)["amount"].sum()
            return _ans("percent", round2(out[0] / tot * 100), "channel_share") if len(out) == 1 and tot else None
        return _ans("number", int(out[0]), "channel_rev") if len(out) == 1 else _ans("list", [int(v) for v in out], "channel_list")

    # --- thuốc kê đơn
    if re.search(r"\bke don\b", rest) and "rx" not in shop.lines:
        return None
    if re.search(r"\bke don\b", rest) and item is None and group is None:
        neg = bool(re.search(r"\b(khong|hong|hok)\b", rest))
        r2 = re.sub(r"\b(khong|hong|hok|thuoc|hang|ke don)\b", " ", rest)
        if not _only_fill(r2, "doanh thu dthu ban chiem phan tram ti le ty bang tien"):
            return None
        x = shop.counted(p)
        v = x.loc[x["rx"] != neg, "amount"].sum()
        if is_rate:
            return _ans("percent", round2(v / x["amount"].sum() * 100), "rx_share") if x["amount"].sum() else None
        return _ans("number", int(v), "rx_rev")

    # --- khách
    m_g = re.search(r"\bkhach (?:hang )?(?:nhom )?(si|le|than thiet|vip|moi|thuong)\b", rest)
    if m_g and "cust_group" in shop.lines and item is None and group is None:
        names = [g for g in shop.lines["cust_group"].dropna().astype(str).unique() if norm_key(g) == m_g.group(1)]
        r2 = rest.replace(m_g.group(0), " ")
        if not names or not _only_fill(r2, "doanh thu dthu mang dong gop chiem phan tram ti le ty mua tong cac"):
            return None
        x = shop.counted(p)
        v = x.loc[x["cust_group"] == names[0], "amount"].sum()
        if is_rate:
            return _ans("percent", round2(v / x["amount"].sum() * 100), "cust_group_share") if x["amount"].sum() else None
        return _ans("number", int(v), "cust_group_rev") if is_rev else None
    if re.search(r"\b(khach|ai)\b", rest) and item is None and group is None and not is_rate and shop.has_orders:
        m_n = re.search(r"\b(?:tu|it nhat|tren)\s*(\d+) don\b", rest)
        if m_n and re.search(r"\b(may|bao nhieu|bn|bnhiu|so)\b", rest):
            r2 = rest.replace(m_n.group(0), " ")
            if _only_fill(r2, "khach mua tro len nguoi hang"):
                d = shop.customers(p, "orders")
                if "cust" not in shop.lines and "cust_code" not in shop.lines:
                    return None
                return _ans("number", int((d["v"] >= int(m_n.group(1))).sum()), "cust_ge_n")
            return None
        by = "money" if re.search(r"\b(tien|chi tieu|chi)\b", rest) else "orders" if re.search(r"\b(don|lan)\b", rest) else None
        m_k = re.search(r"\b(?:top )?(\d+) khach\b", rest)
        if by and re.search(r"\bnhat\b", rest) and _only_fill(re.sub(r"\d+", " ", rest), "khach ai mua nhieu tien chi tieu don lan dem theo nao hang nhat cao la"):
            d = shop.customers(p, by)
            if d.empty:
                return None
            if m_k:
                return _ans("list", d["name"].astype(str).head(int(m_k.group(1))).tolist(), "top_customers")
            return _ans("string", str(d["name"].iloc[0]), "top_customer")
        return None

    # --- mặt hàng bán chạy nhất, top k
    m_k = re.search(r"\btop ?(\d+)\b|\b(\d+) (?:mat hang|mon|san pham|sp)\b", rest)
    if item is None and not is_rev and not is_rate and shop.items and re.search(r"\b(ban chay|ban nhieu|ban duoc nhieu|hot)\b", rest) and re.search(r"\bnhat\b", rest):
        if _only_fill(re.sub(r"\d+", " ", rest), "mat hang mon san pham sp ban chay nhieu duoc hot nhat top ra xep cao thap nao ly chai hop goi cai"):
            r_ = shop.rank_items(p, group)
            if not r_:
                return None
            if m_k:
                return _ans("list", r_[:int(m_k.group(1) or m_k.group(2))], "top_items")
            return _ans("string", r_[0], "top_item")
        return None

    # --- số lượng cả nhóm hàng
    if group is not None and not is_rev and not is_rate and re.search(r"\bban\b", rest):
        if _only_fill(rest, "ban ra duoc ly chai hop goi cai cay vien thung kg bao lon tui suat phan dia to mon cuon lo"):
            v = shop.qty(p, None, group)
            return _ans("number", int(v) if float(v).is_integer() else v, "qty_group") if v else None
        return None

    # --- mặt hàng có mặt trong ít nhất N đơn
    m_n = re.search(r"\b(?:tu|it nhat)\s*(\d+) don\b", rest)
    if m_n and re.search(r"\b(mat hang|mon|san pham|sp)\b", rest) and item is None:
        if _only_fill(rest.replace(m_n.group(0), " "), "mat hang mon san pham sp co mat gop trong khac nhau tro len"):
            x = shop.counted(p)
            return _ans("number", int((x.groupby("item")["order"].nunique() >= int(m_n.group(1))).sum()), "items_ge_n")
        return None

    # --- đơn có tiền hàng từ ngưỡng
    th_ = re.search(r"\d{1,3}(?:[.,]\d{3})+", question)
    if th_ and re.search(r"\b(don|hoa don)\b", rest) and re.search(r"tro len", rest) and item is None and group is None:
        if _only_fill(re.sub(r"\d[\d.,]*", " ", rest), "don hoa tien hang doanh thu dthu dat tu tro len"):
            x = shop.counted(p)
            return _ans("number", int((x.groupby("order")["amount"].sum() >= float(re.sub(r"[.,]", "", th_.group(0)))).sum()), "orders_over")
        return None

    # --- ngày đắt khách nhất
    if re.search(r"\bngay nao\b.*\bdat khach nhat\b", rest) and _only_fill(rest, "ngay nao dat khach"):
        x = shop.counted(p)
        d = x.groupby(x["date"].dt.date)["order"].nunique()
        return _ans("date", d.idxmax().strftime("%Y-%m-%d"), "busiest_day") if len(d) and (d == d.max()).sum() == 1 else None

    # --- ngày doanh thu cao nhất
    if (is_rev or re.search(r"\bban dat\b|\bnhieu tien\b|\bthu ve\b", rest)) and item is None and group is None and re.search(r"\bngay\b.*\b(ban dat nhat|cao nhat|nhieu nhat|nhieu tien nhat)\b|\b(cao nhat|nhieu nhat|nhieu tien nhat)\b.*\bngay\b", rest):
        if _only_fill(rest, "xet theo doanh thu dthu ngay ban dat nhat cao nhieu tien thu ve duoc dat"):
            x = shop.counted(p)
            d = x.groupby(x["date"].dt.date)["amount"].sum()
            if len(d) == 0 or (d == d.max()).sum() > 1:
                return None
            return _ans("date", d.idxmax().strftime("%Y-%m-%d"), "best_day")
        return None

    # --- số ngày doanh thu đạt ngưỡng
    th = re.search(r"\d{1,3}(?:[.,]\d{3})+", question)
    if is_rev and th and item is None and group is None and re.search(r"\bngay\b", rest):
        v = float(re.sub(r"[.,]", "", th.group(0)))
        if _only_fill(re.sub(r"\d[\d.,]*", " ", rest), "ngay dat doanh thu dthu tu tro len dem ca"):
            x = shop.counted(p)
            return _ans("number", int((x.groupby(x["date"].dt.date)["amount"].sum() >= v).sum()), "days_over")
        return None

    if not shop.has_orders and re.search(r"\b(huy|don|hoa don|bill)\b", rest):
        return None

    # --- huỷ
    if re.search(r"\bhuy\b", rest) and item is None and group is None:
        if is_rate and _only_fill(rest, "huy bi ti le ty phan tram da"):
            return _ans("percent", shop.cancel_rate(p), "cancel_rate")
        if not is_rate and _only_fill(rest, "huy bi dem da trang thai"):
            return _ans("number", shop.n_cancelled(p), "n_cancelled")
        return None

    # --- tỉ trọng nhóm trong doanh thu
    if group is not None and is_rate and is_rev:
        if _only_fill(rest, "doanh thu dthu chiem dong gop ti trong ty phan tram bang"):
            return _ans("percent", shop.share_group(p, group), "share_group")
        return None
    if is_rate:
        return None

    if re.search(r"\b(binh quan|trung binh|tb)\b", rest):
        if item is None and group is None and re.search(r"\b(don|hoa don)\b", rest) and _only_fill(rest, "doanh thu dthu binh quan trung tb moi hoa gia tri mot tren mang ve"):
            return _ans("number", shop.avg_order(p), "avg_order")
        return None
    if re.search(r"\b(mat hang|mon|san pham|hang|khach|ai) nao\b|\bnao\b.*\bnhat\b", rest):  # hỏi đối tượng, không hỏi số tiền
        return None
    if is_rev and _only_fill(rest, "doanh thu dthu dt ban thu ve tien ra duoc"):
        if item is not None and shop.n_orders(p, item) == 0:
            return None
        v = shop.revenue(p, item, group)
        return _ans("number", v, "revenue") if v else None  # 0 gần như luôn do lọc rỗng
    if not is_rev and re.search(r"\b" + _QTY_WORD + r"\s+(don|hoa don|bill)\b", rest) and _only_fill(rest, "don hoa bill ban ra dem"):
        return _ans("number", shop.n_orders(p, item, group), "n_orders")
    if item is not None and not is_rev and re.search(r"\bban\b", rest) and _only_fill(rest, "ban ra duoc ly chai hop goi cai cay vien thung kg bao lon tui suat phan dia to mon cuon lo"):
        v = shop.qty(p, item)
        if v == 0:  # đáp án 0 gần như luôn do thiếu cột số lượng hoặc lọc rỗng (dev_003: hỏi mấy cái, ra 0)
            return None
        return _ans("number", int(v) if float(v).is_integer() else v, "qty")
    return None
