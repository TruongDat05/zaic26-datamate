"""Model đọc câu hỏi thành JSON (số đo, kỳ, bộ lọc); code dựng lại câu chuẩn và đưa qua lớp tính toán đã đo trên train.

Model không tự tính số. Tên hàng, nhóm phải nằm trong danh sách của shop, số đo ngoài danh sách thì trả None để đi đường pandas.
"""
from __future__ import annotations

from .clean import norm_key

MEASURES = ["revenue", "n_orders", "n_cancelled", "cancel_rate", "avg_order", "qty", "share_group", "top_items", "top_customers_money", "top_customers_orders", "best_day",
            "revenue_change_previous", "revenue_change_last_year", "purchase_money", "profit", "debt_total", "unsupported"]
PERIODS = ["month", "month_no_year", "quarter", "year", "day", "range", "this_month", "last_month", "this_week", "last_week", "today", "yesterday", "last_n_days", "year_to_date", "month_to_date", "none"]
SCHEMA = {"type": "object", "properties": {
    "measure": {"enum": MEASURES},
    "period": {"type": "object", "properties": {"kind": {"enum": PERIODS}, "year": {"type": "integer"}, "month": {"type": "integer"}, "quarter": {"type": "integer"}, "day": {"type": "integer"},
                                                "n": {"type": "integer"}, "start": {"type": "string"}, "end": {"type": "string"}}, "required": ["kind"]},
    "item": {"type": "string"}, "group": {"type": "string"}, "shift": {"enum": ["", "sang", "chieu"]}, "top_k": {"type": "integer"}},
    "required": ["measure", "period"]}
SYSTEM = """Bạn đọc câu hỏi tiếng Việt của chủ cửa hàng (có thể viết tắt, không dấu, sai chính tả) và ghi lại ý định dạng JSON. Không tính số.
- measure: revenue doanh thu, tiền bán; n_orders số đơn; n_cancelled số đơn huỷ; cancel_rate tỉ lệ đơn huỷ; avg_order doanh thu bình quân mỗi đơn; qty số lượng bán của một mặt hàng hoặc nhóm;
  share_group tỉ trọng doanh thu của nhóm; top_items mặt hàng bán chạy; top_customers_money khách mua nhiều tiền nhất; top_customers_orders khách nhiều đơn nhất; best_day ngày doanh thu cao nhất;
  revenue_change_previous phần trăm doanh thu so với kỳ trước; revenue_change_last_year so với cùng kỳ năm trước; purchase_money tiền nhập hàng; profit lãi gộp; debt_total tổng tiền khách còn nợ.
  Câu có điều kiện khác hoặc hỏi điều khác: unsupported.
- period.kind: month (year, month), month_no_year (month), quarter (year, quarter), year, day (year, month, day), range (start, end dạng YYYY-MM-DD), this_month, last_month, this_week, last_week, today, yesterday, last_n_days (n), year_to_date, month_to_date, none.
- item, group: chép đúng một tên trong danh sách được cho; không có thì để trống. shift: sang, chieu hoặc để trống. top_k: số đối tượng khi hỏi top.
Ví dụ:
"dthu t5/25 bnhiu" → {"measure":"revenue","period":{"kind":"month","year":2025,"month":5}}
"tuan roi ca chieu dc may don" → {"measure":"n_orders","period":{"kind":"last_week"},"shift":"chieu"}
"3 mon chay nhat thang nay" → {"measure":"top_items","period":{"kind":"this_month"},"top_k":3}
"hom qua khach nao hay gop y" → {"measure":"unsupported","period":{"kind":"yesterday"}}"""


def candidates(question: str, names: list, k: int = 15) -> list:
    """Tên trong dữ liệu gần với câu hỏi nhất: model chỉ được chọn trong đó."""
    q = set(norm_key(question).split())
    scored = []
    for n in names:
        w = set(norm_key(n).split())
        hit = len(w & q)
        if hit:
            scored.append((-hit / max(len(w), 1), -hit, str(n)))
    return [x[2] for x in sorted(scored)[:k]]


def render_period(p: dict) -> str | None:
    k = p.get("kind")
    try:
        if k == "month":
            return f"thang {int(p['month'])} {int(p['year'])}"
        if k == "month_no_year":
            return f"thang {int(p['month'])}"
        if k == "quarter":
            return f"quy {int(p['quarter'])} {int(p['year'])}"
        if k == "year":
            return f"nam {int(p['year'])}"
        if k == "day":
            return f"ngay {int(p['day']):02d} {int(p['month']):02d} {int(p['year'])}"
        if k == "range":
            a, b = [str(p[x]).split("-") for x in ("start", "end")]
            return f"tu ngay {a[2]} {a[1]} {a[0]} den ngay {b[2]} {b[1]} {b[0]}"
        if k == "last_n_days":
            return f"{int(p['n'])} ngay gan nhat"
    except (KeyError, ValueError, TypeError, IndexError):
        return None
    return {"this_month": "thang nay", "last_month": "thang truoc", "this_week": "tuan nay", "last_week": "tuan truoc", "today": "hom nay", "yesterday": "hom qua",
            "year_to_date": "tu dau nam den hom nay", "month_to_date": "tu dau thang nay den hom nay"}.get(k)


def render(slots: dict, shop) -> str | None:
    """Câu chuẩn cho lớp tính toán; None nếu không dựng được."""
    if not isinstance(slots, dict) or slots.get("measure") in (None, "unsupported"):
        return None
    period = render_period(slots.get("period") or {})
    item, group = slots.get("item") or None, slots.get("group") or None
    if (item and item not in shop.items) or (group and group not in shop.groups) or (item and group) or period is None:
        return None
    m = slots["measure"]
    target = f"mat hang {item}" if item else f"nhom {group}" if group else ""
    shift = f"ca {slots['shift']}" if slots.get("shift") else ""
    k = slots.get("top_k")
    if shift and (target or m not in ("revenue", "n_orders")):
        return None
    if m == "revenue":
        return f"doanh thu {shift} {target} {period} la bao nhieu"
    if m == "n_orders":
        return f"{period} {shift} {target} co bao nhieu don"
    if target and m not in ("qty", "share_group", "top_items"):
        return None
    if m == "n_cancelled":
        return f"{period} co bao nhieu don bi huy"
    if m == "cancel_rate":
        return f"ti le huy don {period} la bao nhieu phan tram"
    if m == "avg_order":
        return f"doanh thu binh quan moi don {period} la bao nhieu"
    if m == "qty" and target:
        return f"{period} {target} ban duoc bao nhieu"
    if m == "share_group" and group:
        return f"{target} chiem bao nhieu phan tram doanh thu {period}"
    if m == "top_items" and not item:
        return f"{target} {'top ' + str(int(k)) if k else ''} mat hang ban chay nhat {period}"
    if m == "top_customers_money":
        return f"{str(int(k)) + ' khach' if k else 'khach nao'} mua nhieu tien nhat {period}"
    if m == "top_customers_orders":
        return f"{str(int(k)) + ' khach' if k else 'khach nao'} mua nhieu don nhat {period}"
    if m == "best_day":
        return f"{period} ngay nao doanh thu cao nhat"
    if m == "revenue_change_previous":
        return f"doanh thu {period} so voi {'tuan' if 'tuan' in period else 'thang'} truoc tang giam bao nhieu phan tram"
    if m == "revenue_change_last_year":
        return f"doanh thu {period} so voi cung ky nam truoc tang giam bao nhieu phan tram"
    if m == "purchase_money":
        return f"{period} nhap hang het bao nhieu tien"
    if m == "profit":
        return f"lai gop {period} la bao nhieu"
    if m == "debt_total" and slots["period"].get("kind") in ("today", "day"):
        return f"tinh den het {period} khach con no tong cong bao nhieu tien"
    return None


def parse(llm, question: str, as_of: str, shop, seed: int = 42):
    names = candidates(question, shop.items)
    groups = candidates(question, shop.groups)
    user = f"Hôm nay: {as_of}\nTên hàng có thể liên quan: {names}\nNhóm hàng có thể liên quan: {groups}\nCâu hỏi: {question}"
    return llm.chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], SCHEMA, seed=seed)
