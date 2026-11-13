"""Chạy code do LLM sinh trong một tiến trình con, có giới hạn thời gian.

Gọi: python -m datamate.sandbox <business.pkl> <calendar.csv> <as_of_date>   (code đọc từ stdin)
In ra một dòng JSON cuối cùng: {"ok": true, "answer": {...}, "stdout": "..."} hoặc {"ok": false, "error": "..."}.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import traceback
from pathlib import Path

MARK = "@@DATAMATE_RESULT@@"


def run_code(code: str, biz_pickle: Path, calendar_csv: Path, as_of_date: str, timeout: float, *, isolation_hook: Path | None = None, child_env: dict | None = None) -> dict:
    try:
        p = subprocess.run(
            [sys.executable, "-m", "datamate.sandbox", str(biz_pickle), str(calendar_csv), as_of_date] + ([str(isolation_hook)] if isolation_hook else []),
            input=code, capture_output=True, text=True, timeout=timeout,
            cwd=str(Path(__file__).resolve().parent.parent),
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "timeout": True, "error": f"Hết thời gian chạy ({timeout:.0f}s). Viết code đơn giản, vector hoá hơn."}
    for line in reversed(p.stdout.splitlines()):
        if line.startswith(MARK):
            return json.loads(line[len(MARK):])
    return {"ok": False, "error": (p.stderr or p.stdout)[-3000:] or f"tiến trình thoát mã {p.returncode}"}


def _main():
    biz_pickle, calendar_csv, as_of = sys.argv[1], sys.argv[2], sys.argv[3]
    code = sys.stdin.read()
    if len(sys.argv) > 4:
        import runpy
        restrict = runpy.run_path(sys.argv[4])["restrict_process"]
    else:
        restrict = None

    import calendar
    import datetime as dt
    import math
    import re
    import warnings

    import numpy as np
    import pandas as pd

    from . import dm, shop as shop_mod
    from .timeutil import LunarCalendar

    warnings.filterwarnings("ignore")
    with open(biz_pickle, "rb") as fh:
        biz = pickle.load(fh)
    dm.cal = LunarCalendar(calendar_csv)
    side = biz_pickle + ".contract.json"  # hợp đồng dữ liệu của shop: S trong sandbox phải giống S của agent
    if os.path.exists(side):
        with open(side, encoding="utf-8") as fh:
            biz.extra_roles = json.load(fh)
    # model hay viết `import dm` / `from dm import ...` dù prompt dặn dm có sẵn
    sys.modules["dm"] = dm
    _date_safe_compare(pd, dt)
    try:
        S = shop_mod.build(biz)
    except Exception:  # noqa: BLE001 — shop lạ: code model vẫn dùng được T
        S = None
    env = {
        "S": S,
        "__name__": "__main__",  # code bọc trong `if __name__ == "__main__":` vẫn chạy
        "T": biz.tables,  # tiến trình con dùng một lần, sửa bảng tại chỗ cũng không sao
        "TODAY": dt.date.fromisoformat(as_of),
        "pd": pd, "np": np, "dm": dm, "dt": dt, "datetime": dt, "calendar": calendar, "math": math, "re": re,
        "ANSWER": None,
    }
    env["sql"] = _sql_helper(biz.tables, pd)
    buf = io.StringIO()
    result: dict
    try:
        if restrict:
            restrict()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<code>", "exec"), env)
        ans = env.get("ANSWER")
        if isinstance(ans, (dt.date, pd.Timestamp)):  # ANSWER = dm.date(y, m, d) / một ngày trần: chỉ có thể là kiểu date
            ans = dm.date(ans)
        if isinstance(ans, dict) and "type" in ans:
            result = {"ok": True, "answer": _jsonable(dict(ans))}
        elif ans is None:
            result = {"ok": False, "error": "Code chạy xong nhưng ANSWER vẫn là None. Gán ANSWER = dm.<kiểu>(...) ở cấp "
                                            "ngoài cùng của code, không đặt trong hàm hay khối if không chạy tới."}
        else:
            result = {"ok": False, "error": f"ANSWER đang là {type(ans).__name__} {str(ans)[:200]!r}, chưa qua hàm tạo "
                                            "đáp án. Gán ANSWER = dm.money(...) / dm.count(...) / dm.percent(...) / "
                                            "dm.string(...) / dm.list_(...)… đúng kiểu câu hỏi."}
    except BaseException as e:  # noqa: BLE001 — cả SystemExit khi code gọi exit()
        tb = traceback.format_exc().splitlines()
        tb = [ln for ln in tb if 'File "<code>"' in ln or not ln.startswith("  File")]
        err = f"{type(e).__name__}: {e}\n" + "\n".join(tb[-12:])
        hint = _hint(e, env)
        result = {"ok": False, "error": err + (f"\nGợi ý: {hint}" if hint else "")}
    out = buf.getvalue()
    result["stdout"] = out[-2000:]
    print(MARK + json.dumps(result, ensure_ascii=False, default=str))


def _sql_helper(tables: dict, pd):
    """sql(câu lệnh) → DataFrame, chạy trên bản sao của T trong bộ nhớ, tạo khi gọi lần đầu."""
    conn = None

    def sql(query: str):
        nonlocal conn
        if conn is None:
            import sqlite3
            conn = sqlite3.connect(":memory:")
            for name, df in tables.items():
                x = df.loc[:, ~df.columns.duplicated()].copy()
                for c in x.columns:
                    if pd.api.types.is_datetime64_any_dtype(x[c]):
                        x[c] = x[c].dt.strftime("%Y-%m-%d %H:%M:%S")
                    elif x[c].dtype == object:
                        x[c] = x[c].map(lambda v: v if v is None or isinstance(v, (str, int, float)) else str(v))
                x.to_sql(str(name), conn, index=False)
        return pd.read_sql_query(query, conn)
    return sql


def _date_safe_compare(pd, dt) -> None:
    """So cột datetime64 với một ngày (TODAY, dt.date) theo cả ngày, thay vì pandas báo lỗi hoặc trả toàn False.

    `cot <= TODAY` nghĩa là đến hết ngày TODAY (kể cả các giờ trong ngày), `cot > d` là từ ngày sau d, `cot == d` là cùng ngày.
    Lỗi này chiếm nhiều nhất trong log holdout sau bản v2.2."""
    day = pd.Timedelta(days=1)
    rules = {
        "__lt__": lambda s, t: s < t, "__ge__": lambda s, t: s >= t,
        "__le__": lambda s, t: s < t + day, "__gt__": lambda s, t: s >= t + day,
        "__eq__": lambda s, t: s.dt.normalize() == t, "__ne__": lambda s, t: s.dt.normalize() != t,
    }
    for name, rule in rules.items():
        orig = getattr(pd.Series, name)

        def op(self, other, _orig=orig, _rule=rule):
            if (isinstance(other, dt.date) and not isinstance(other, dt.datetime)
                    and pd.api.types.is_datetime64_any_dtype(self.dtype)):
                return _rule(self, pd.Timestamp(other))
            return _orig(self, other)
        setattr(pd.Series, name, op)


def _hint(e: BaseException, env: dict) -> str:
    """Gợi ý sửa cho các lỗi model hay mắc (đo trên log chạy holdout), kèm thông tin thật của dữ liệu."""
    msg = str(e)
    if isinstance(e, KeyError):
        import difflib
        key = str(e.args[0]) if e.args else msg
        allc = {f"T[{name!r}][{c!r}]": str(c) for name, df in env["T"].items() for c in df.columns}
        allc.update({f"T[{name!r}]": str(name) for name in env["T"]})
        near = difflib.get_close_matches(key, list(allc.values()), n=5, cutoff=0.4)
        near = [ref for ref, c in allc.items() if c in near][:5]
        cols = "; ".join(f"T[{name!r}]: {list(df.columns)}" for name, df in env["T"].items())
        return (f"không có cột/khóa {msg}." + (f" Gần nhất: {', '.join(near)}." if near else "")
                + f" Các bảng và cột hiện có — {cols}")[:3000]
    if isinstance(e, SystemExit):
        return "không gọi exit() / sys.exit(); chỉ cần gán ANSWER."
    if "Invalid comparison between dtype=datetime64" in msg or (
            isinstance(e, TypeError) and "datetime.date" in msg and "datetime.datetime" in msg):
        return ("cột ngày là datetime64, TODAY là datetime.date. Lọc theo kỳ bằng p.mask(cot_ngay) "
                "(p = dm.last_week(TODAY), dm.month(y, m)…), hoặc so với pd.Timestamp(TODAY).")
    if "monthrange" in msg:
        return "dùng calendar.monthrange(y, m) (module calendar có sẵn) hoặc dm.month(y, m) cho cả tháng."
    if "'Period' object has no attribute" in msg:
        return "Period chỉ có .start, .end (datetime.date), .days, .mask(cot_ngay), .dates()."
    if "module 'dm' has no attribute" in msg or "'dm' has no attribute" in msg:
        from . import dm
        names = sorted(n for n in dir(dm) if not n.startswith("_") and callable(getattr(dm, n)))
        return "dm không có hàm đó. Các hàm có: " + ", ".join(names)
    if "Answer" in msg and ("Timestamp" in msg or "date" in msg):
        return ("dm.date(...) / dm.money(...)… chỉ để tạo ĐÁP ÁN cuối. Tạo một ngày để lọc: dt.date(y, m, d), "
                "pd.Timestamp('YYYY-MM-DD'), hoặc dùng kỳ của dm (dm.month(y, m), dm.this_month(TODAY)…).")
    if "Cannot compare Timestamp with datetime.date" in msg:
        return "đổi về cùng kiểu: pd.Timestamp(d) hoặc ts.date(); lọc kỳ thì dùng p.mask(cot_ngay)."
    if "Lengths must match" in msg or "Unalignable boolean Series" in msg:
        return ("điều kiện lọc phải tạo từ chính DataFrame đang lọc (cùng index); đừng dùng điều kiện của bảng khác "
                "hay của bảng trước khi lọc / merge.")
    if "you passed a `method`" in msg:
        return "gọi hàm có ngoặc, vd p.dates() chứ không phải p.dates."
    if "cần một số, nhận Series" in msg or "cần một số, nhận DataFrame" in msg:
        return "dm.money / dm.count / dm.number cần MỘT số: cộng .sum(), lấy .iloc[0] hoặc len(...) trước."
    if "bad operand type for unary ~" in msg:
        return "~ chỉ dùng cho Series đúng/sai; với một giá trị đơn dùng `not`."
    if "passed to isin()" in msg and "Period" in msg:
        return "Period không phải danh sách; lọc kỳ bằng p.mask(cot_ngay), hoặc p.dates() nếu cần list ngày."
    if "Can only use .dt accessor" in msg:
        return "cột này chưa phải datetime; đổi bằng dm.to_datetime(cot) trước khi dùng .dt, hoặc chọn đúng cột ngày."
    if "day is out of range for month" in msg:
        return "đừng tự cộng trừ ngày tháng; dùng dm.month, dm.last_month, dm.previous(p), dm.prev_month_same_days."
    if "truth value of a Series is ambiguous" in msg or "unsupported operand type(s) for &" in msg:
        return "ghép điều kiện Series bằng & | ~ và bọc từng vế trong ngoặc: (a == 1) & (b > 2); không dùng and/or."
    return ""


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


if __name__ == "__main__":
    _main()
