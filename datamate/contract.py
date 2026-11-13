"""Hợp đồng dữ liệu: model đọc README, tên cột và giá trị mẫu để gán vai trò cột; code kiểm tra lại trước khi dùng.

Chỉ lấy vai trò qua được phép kiểm tra trên giá trị thật và chỉ điền chỗ shop.py chưa nhận ra, nên model không thể làm hỏng
vai trò đã biết. Kết quả ghi cạnh file cache để tiến trình sandbox dựng cùng khung S.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from .clean import norm_status

log = logging.getLogger("datamate")

ROLES = {
    "order": "mã đơn, hoá đơn, phiếu bán", "date": "ngày đặt, ngày bán", "status": "trạng thái đơn", "item": "tên hàng", "item_code": "mã hàng",
    "amount": "thành tiền của dòng, sau giảm trên dòng", "qty": "số lượng ghi theo đơn vị của dòng", "qty_base": "số lượng đã đổi ra đơn vị cơ bản", "price": "đơn giá",
    "cost": "giá vốn một đơn vị bán", "group": "nhóm hàng", "unit": "đơn vị tính", "big_unit": "đơn vị lớn trong bảng quy đổi", "factor": "hệ số quy đổi",
    "cust": "tên khách", "cust_code": "mã khách", "cust_group": "nhóm khách", "staff": "nhân viên bán", "channel": "kênh bán", "pay": "hình thức thanh toán", "time": "giờ tạo đơn", "rx": "thuốc kê đơn",
}
SCHEMA = {"type": "object", "properties": {"tables": {"type": "array", "items": {"type": "object", "properties": {
    "table": {"type": "string"}, "roles": {"type": "object", "properties": {r: {"type": "string"} for r in ROLES}, "additionalProperties": False}},
    "required": ["table", "roles"]}}}, "required": ["tables"]}
SYSTEM = ("Bạn gán vai trò cho các cột trong dữ liệu bán hàng của cửa hàng nhỏ. Đọc README và mô tả bảng, chỉ dùng đúng tên bảng và tên cột có sẵn. "
          "Không chắc thì bỏ trống vai trò đó, không đoán. Trả về JSON.\nVai trò:\n" + "\n".join(f"- {k}: {v}" for k, v in ROLES.items()))


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _text_col(s) -> bool:
    return not str(s.dtype).startswith(("datetime", "Int", "int", "float"))


def _check(role: str, s: pd.Series) -> bool:
    """Giá trị của cột có đúng vai trò được gán không."""
    v = s.dropna()
    if v.empty:
        return False
    if role == "date":
        return str(s.dtype).startswith("datetime") or pd.to_datetime(v.head(500), errors="coerce").notna().mean() > 0.9
    if role in ("amount", "qty", "qty_base", "price", "cost", "factor"):
        return _num(v).notna().mean() > 0.9 or (role == "factor" and _num(v).notna().mean() > 0.5)
    if role == "status":
        return v.map(norm_status).ne("").mean() > 0.6
    if role == "order":
        return len(v) / len(s) > 0.9
    if role in ("item", "group", "cust", "cust_group", "channel", "pay", "unit", "big_unit"):
        return _text_col(s) and v.nunique() >= 2
    return True


def validate(biz, raw) -> dict:
    """{bảng: {vai trò: cột}}, đã bỏ tên lạ và cột không qua kiểm tra. Số lượng × đơn giá phải khớp thành tiền nếu có đủ ba cột."""
    out = {}
    for row in (raw or {}).get("tables", []) if isinstance(raw, dict) else []:
        df = biz.tables.get(row.get("table")) if isinstance(row, dict) else None
        if df is None or not isinstance(row.get("roles"), dict):
            continue
        ok = {r: c for r, c in row["roles"].items() if r in ROLES and c in df.columns and _check(r, df[c])}
        if {"qty", "price", "amount"} <= set(ok):
            q, p, a = _num(df[ok["qty"]]), _num(df[ok["price"]]), _num(df[ok["amount"]])
            if ((q * p - a).abs() <= a.abs() * 0.5 + 1).mean() < 0.7:
                for r in ("qty", "price", "amount"):
                    ok.pop(r)
        if len(set(ok.values())) < len(ok):  # một cột không giữ hai vai trò
            seen, keep = set(), {}
            for r, c in ok.items():
                if c not in seen:
                    seen.add(c)
                    keep[r] = c
            ok = keep
        if ok:
            out[row["table"]] = ok
    return out


def user_prompt(biz) -> str:
    return f"# README\n{biz.readme.strip()[:6000]}\n\n# Các bảng\n{biz.profile()[:12000]}\n\nTrả về JSON: tables[].table, tables[].roles (vai trò → tên cột)."


def propose(llm, biz, seed: int = 42):
    return llm.chat_json([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(biz)}], SCHEMA, seed=seed)


def sidecar(biz_pickle) -> Path:
    return Path(str(biz_pickle) + ".contract.json")


def ensure(llm, biz, biz_pickle, logs_dir) -> dict:
    """Lấy hợp đồng và gắn vào biz. Lưu theo lượt chạy (chạy tiếp thì dùng lại, đổi model thì lập lại), chép ra sidecar cho sandbox.

    Lỗi thì coi như không có hợp đồng.
    """
    saved = Path(logs_dir) / f"contract_{biz.business_id}.json"
    if saved.exists():
        roles = json.loads(saved.read_text(encoding="utf-8"))
    else:
        try:
            roles = validate(biz, propose(llm, biz))
        except Exception as e:  # noqa: BLE001 — không có hợp đồng thì đi đường cũ
            log.warning("không lập được hợp đồng dữ liệu cho %s: %s", biz.business_id, e)
            roles = {}
        saved.write_text(json.dumps(roles, ensure_ascii=False, indent=1), encoding="utf-8")
    biz.extra_roles = roles
    sidecar(biz_pickle).write_text(json.dumps(roles, ensure_ascii=False, indent=1), encoding="utf-8")
    return roles
