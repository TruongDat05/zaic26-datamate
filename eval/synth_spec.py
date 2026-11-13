"""Phần soạn tay cho bộ nguồn sinh dữ liệu (data/synth_source): phân loại kỹ năng, nhóm lỗi, kiểu sổ, mẫu lỗi đã biết,
biến thể luật. eval/build_synth_source.py đọc file này và kiểm tra lại với rulebook, dạng câu, mã lỗi trong code.

Nguồn: rulebook train, datamate/prompts/system.md, docs/pipeline_baocao.md (phần phân tích lỗi), templates_ext.RULE_OF.
Không lấy gì từ public (luật 6).
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------- kỹ năng (nhãn K, theo mục rulebook)
SKILLS = {
    "K1": {"ten": "Kỳ thời gian", "muc": "§2", "mo_ta": "tuần (thứ Hai), tháng này so với cùng số ngày tháng trước, N ngày kể cả hôm nay, "
           "quý/tháng không ghi năm, Tết (23 Chạp → mùng 10), đến hết ngày X, bản ghi sau hôm nay"},
    "K2": {"ten": "Đơn tính doanh thu", "muc": "§1", "mo_ta": "bỏ đơn huỷ và cả đơn có trả hàng (kể cả trả một phần), giữ đơn đang giao; "
           "doanh thu = thành tiền dòng, không trừ giảm giá cả đơn, không cộng phí giao; theo hình thức thanh toán ghi trên đơn"},
    "K3": {"ten": "Đếm đơn, khách", "muc": "§1, §3", "mo_ta": "đếm mã đơn chứ không đếm dòng; khách đã mua; khách lẻ không phải khách; "
           "xác định khách theo mã hoặc tên; ngưỡng 'quá N' không gồm N, 'từ N' gồm N"},
    "K4": {"ten": "Đơn vị, số lượng", "muc": "§3, §10", "mo_ta": "quy về đơn vị cơ bản, đơn vị lớn nhiều cấp, 'bán được bao nhiêu <đơn vị>', tồn theo đơn vị lớn"},
    "K5": {"ten": "Khớp tên", "muc": "§8", "mo_ta": "viết tắt, không dấu, nhiều cách viết cùng một tên, cột __ten/__loai, theo mã khi có"},
    "K6": {"ten": "Xếp hạng, bằng nhau", "muc": "§3, §9", "mo_ta": "bán chạy theo số lượng cơ bản, bằng nhau xếp theo mã tăng dần, phí phục vụ không là mặt hàng, top k"},
    "K7": {"ten": "Tỉ lệ, phần trăm", "muc": "§1, §9", "mo_ta": "mẫu số đúng (tỉ lệ huỷ: mọi đơn đặt trong kỳ), % thay đổi so với kỳ trước, giảm ra số âm, làm tròn 2 chữ số"},
    "K8": {"ten": "Công nợ", "muc": "§4", "mo_ta": "nợ theo tổng đơn gồm phí giao, đơn trả vẫn tính, COD/sàn không phải nợ, tuổi nợ từ ngày đặt, nợ thêm/thu nợ trong kỳ"},
    "K9": {"ten": "Kho, nhập, giá vốn, lãi", "muc": "§5", "mo_ta": "tiền nhập theo ngày phiếu, giá nhập bình quân theo số lượng, tồn hết ngày, giá vốn ghi trên dòng bán, lãi gộp, lãi ròng"},
    "K10": {"ten": "Luật ngành", "muc": "§6", "mo_ta": "cà phê (size M/L), nhà thuốc (kê đơn, hoạt chất, lô, hạn dùng), VLXD (đợt giao)"},
    "K11": {"ten": "Không trả lời được", "muc": "§7", "mo_ta": "kỳ hoàn toàn ngoài dữ liệu, shop không ghi loại thông tin, câu hỏi nhận xét; không có dữ liệu khác với 0"},
    "K12": {"ten": "Đọc schema", "muc": "§8", "mo_ta": "chọn đúng bảng/cột, nối bảng qua mã, sổ không có mã đơn, sổ chỉ có hoá đơn, sheet theo tháng"},
    "K13": {"ten": "Format đáp án", "muc": "§9", "mo_ta": "đúng kiểu (number/percent/date/list/table…), tiền nguyên đồng, 'tháng nào' ra YYYY-MM, list có thứ tự"},
    "K14": {"ten": "Hiểu ý câu hỏi", "muc": "§10", "mo_ta": "đọc đúng số đo, đối tượng, điều kiện trong câu viết tắt, sai chính tả, đảo trật tự"},
}

# ---------------------------------------------------------------- dạng câu của lớp tính toán → nhóm kỹ năng
FAMILIES = {
    "doanh_thu": "Doanh thu và so sánh doanh thu",
    "don": "Số đơn, đơn huỷ, giá trị trung bình đơn",
    "so_luong_xep_hang": "Số lượng bán, mặt hàng bán chạy, mặt hàng ế",
    "gia_ban": "Giá bán theo ngày, hiệu ứng tăng giá",
    "nhom_kenh_ca": "Nhóm hàng, kênh bán, thanh toán, ca",
    "khach": "Khách hàng",
    "nhan_vien": "Nhân viên",
    "lai_nhap_ton": "Lãi, nhập hàng, tồn kho",
    "cong_no": "Công nợ",
    "nganh": "Luật ngành",
    "khong_tra_loi": "Không trả lời được",
}

TEMPLATES = {
    "revenue": ("doanh_thu", ["K2"], "Doanh thu trong kỳ, có thể lọc mặt hàng, kênh, khách"),
    "g_rev": ("doanh_thu", ["K2"], "Doanh thu của nhóm hàng"),
    "rev_months": ("doanh_thu", ["K1", "K2"], "Doanh thu nhiều tháng"),
    "rev_list": ("doanh_thu", ["K1", "K2", "K13"], "Doanh thu từng tháng trong dãy tháng (list)"),
    "rev_trend": ("doanh_thu", ["K1", "K2", "K13"], "Doanh thu có tăng liên tục không (boolean)"),
    "best_day": ("doanh_thu", ["K1", "K6"], "Ngày doanh thu cao nhất"),
    "best_month": ("doanh_thu", ["K1", "K6"], "Tháng doanh thu cao nhất trong dãy tháng"),
    "days_over": ("doanh_thu", ["K1", "K3"], "Số ngày doanh thu vượt ngưỡng"),
    "g_days_over": ("doanh_thu", ["K1", "K3"], "Số ngày doanh thu nhóm hàng vượt ngưỡng"),
    "rev_change_pct": ("doanh_thu", ["K1", "K7"], "% thay đổi doanh thu so với kỳ trước"),
    "g_rev_change": ("doanh_thu", ["K1", "K7"], "% thay đổi doanh thu nhóm hàng"),
    "rev_change_2m": ("doanh_thu", ["K1", "K7"], "% thay đổi giữa hai tháng nêu tên"),
    "rev_yoy": ("doanh_thu", ["K1", "K7"], "% thay đổi so với cùng kỳ năm trước (cả Tết)"),
    "n_orders": ("don", ["K3"], "Số đơn trong kỳ"),
    "n_cancelled": ("don", ["K2", "K3"], "Số đơn huỷ"),
    "cancel_rate": ("don", ["K2", "K7"], "Tỉ lệ đơn huỷ"),
    "avg_order": ("don", ["K2", "K13"], "Giá trị trung bình một đơn"),
    "g_avg": ("don", ["K2", "K13"], "Giá trị trung bình đơn có nhóm hàng"),
    "orders_over": ("don", ["K3"], "Số đơn từ ngưỡng tiền"),
    "g_orders_over": ("don", ["K3"], "Số đơn có nhóm hàng từ ngưỡng tiền"),
    "items_ge_n": ("so_luong_xep_hang", ["K3"], "Số mặt hàng có mặt trong từ N đơn"),
    "g_items_ge": ("so_luong_xep_hang", ["K3"], "Số mặt hàng của nhóm có mặt trong từ N đơn"),
    "qty": ("so_luong_xep_hang", ["K4", "K5"], "Số lượng bán của mặt hàng"),
    "g_qty": ("so_luong_xep_hang", ["K4"], "Số lượng bán của nhóm hàng"),
    "unit_total": ("so_luong_xep_hang", ["K4"], "Bán được bao nhiêu <đơn vị>"),
    "top_item": ("so_luong_xep_hang", ["K4", "K6"], "Mặt hàng bán chạy nhất"),
    "top_items": ("so_luong_xep_hang", ["K4", "K6", "K13"], "Top k mặt hàng bán chạy"),
    "stale_count": ("so_luong_xep_hang", ["K1", "K6"], "Số mặt hàng ế trong kỳ"),
    "price_on_day": ("gia_ban", ["K4"], "Giá bán một mặt hàng vào ngày D"),
    "price_effect": ("gia_ban", ["K1", "K4", "K7"], "So số lượng N ngày sau và trước lần tăng giá"),
    "share_group": ("nhom_kenh_ca", ["K7"], "Tỉ trọng doanh thu của nhóm hàng"),
    "g_share": ("nhom_kenh_ca", ["K7"], "Tỉ trọng nhóm hàng (dạng khác)"),
    "shift": ("nhom_kenh_ca", ["K1"], "Doanh thu, số đơn theo ca"),
    "shift_change": ("nhom_kenh_ca", ["K1", "K7"], "So ca chiều với ca sáng"),
    "channel_rev": ("nhom_kenh_ca", ["K2"], "Doanh thu theo kênh hoặc thanh toán"),
    "channel_list": ("nhom_kenh_ca", ["K2", "K13"], "Doanh thu nhiều kênh (list)"),
    "channel_share": ("nhom_kenh_ca", ["K2", "K7"], "Tỉ trọng kênh hoặc thanh toán"),
    "cust_ge_n": ("khach", ["K3"], "Số khách mua từ N đơn"),
    "cust_share_ge": ("khach", ["K3", "K7"], "% khách mua từ N đơn"),
    "top_customer": ("khach", ["K3", "K6"], "Khách mua nhiều nhất"),
    "top_customers": ("khach", ["K3", "K6", "K13"], "Top k khách"),
    "top_customers2": ("khach", ["K3", "K6", "K13"], "Top k khách kèm số đo (table)"),
    "staff_best": ("nhan_vien", ["K6"], "Nhân viên bán giỏi nhất"),
    "staff_rev": ("nhan_vien", ["K2"], "Doanh thu của một nhân viên"),
    "profit": ("lai_nhap_ton", ["K9"], "Lãi gộp"),
    "profit_best": ("lai_nhap_ton", ["K9", "K6"], "Hàng lời nhất"),
    "net_profit": ("lai_nhap_ton", ["K9"], "Lãi ròng tháng"),
    "purchase_money": ("lai_nhap_ton", ["K9"], "Tiền nhập hàng trong kỳ"),
    "top_supplier": ("lai_nhap_ton", ["K9", "K6"], "Nhà cung cấp nhập nhiều nhất"),
    "avg_purchase": ("lai_nhap_ton", ["K9", "K4"], "Giá nhập bình quân trong tháng"),
    "in_minus_out": ("lai_nhap_ton", ["K9", "K4"], "Nhập trừ bán"),
    "stock": ("lai_nhap_ton", ["K9", "K1"], "Tồn kho đến hết ngày"),
    "stock_big": ("lai_nhap_ton", ["K9", "K4", "K13"], "Tồn theo đơn vị lớn (list 2 số)"),
    "stock_value": ("lai_nhap_ton", ["K9"], "Giá trị hàng tồn kho"),
    "debt_total": ("cong_no", ["K8"], "Tổng khách còn nợ"),
    "debt_top": ("cong_no", ["K8", "K6", "K13"], "Khách nợ nhiều nhất"),
    "debt_n_cust": ("cong_no", ["K8", "K3"], "Số khách còn nợ"),
    "debt_bands": ("cong_no", ["K8", "K1"], "Tuổi nợ theo nhóm ngày"),
    "debt_collected": ("cong_no", ["K8", "K1"], "Thu nợ trong kỳ"),
    "debt_cust_period": ("cong_no", ["K8", "K1"], "Nợ thêm, trả nợ của khách trong kỳ"),
    "rx_rev": ("nganh", ["K10"], "Doanh thu thuốc kê đơn/không kê đơn"),
    "rx_share": ("nganh", ["K10", "K7"], "Tỉ lệ doanh thu kê đơn"),
    "active_rev": ("nganh", ["K10", "K5"], "Doanh thu theo hoạt chất"),
    "lot_nearest": ("nganh", ["K10", "K9"], "Lô gần hết hạn nhất"),
    "size_qty": ("nganh", ["K10", "K4"], "Số ly size M/L"),
    "deliveries": ("nganh", ["K10", "K1"], "Số đợt giao trong tháng"),
    "un_opinion": ("khong_tra_loi", ["K11"], "Câu nhận xét, file không ghi"),
    "un_period": ("khong_tra_loi", ["K11", "K1"], "Kỳ nằm ngoài dữ liệu"),
}

# Gắn nhãn kỹ năng cho câu lớp tính toán không nhận (nhãn gợi ý theo từ khoá, chưa kiểm tay)
KEYWORD_SKILLS = [
    ("K1", r"tuần|tuan|tết|tet|quý|quy |cùng kỳ|cung ky|\d+ ngày|\d+ ngay|đến hết|den het|tháng này|thang nay|từ đầu năm|tu dau nam"),
    ("K2", r"tiền mặt|tien mat|chuyển khoản|thanh toán|thanh toan|kênh|kenh|online|điện thoại|dien thoai|huỷ|hủy|huy|trả hàng|tra hang"),
    ("K3", r"bao nhiêu đơn|bnhiu don|mấy đơn|may don|số đơn|so don|khách|khach|≥|từ \d+ đơn|tu \d+ don"),
    ("K4", r"hộp|hop|vỉ|vi |viên|vien|thùng|thung|lốc|loc|chai|lon|gói|goi|chẵn|chan|lẻ|le "),
    ("K6", r"nhất|nhat|top|best seller|hot|xếp hạng"),
    ("K7", r"%|phần trăm|phan tram|tỉ lệ|ti le|tỷ lệ|chiếm|chiem|đóng góp|dong gop"),
    ("K8", r"nợ|no |công nợ|cong no"),
    ("K9", r"tồn|ton kho|kho|nhập|nhap|vốn|von|lãi|lai|lợi nhuận|loi nhuan|lô|hạn dùng|han dung|hết hạn|het han"),
    ("K10", r"kê đơn|ke don|bác sĩ|bac si|hoạt chất|hoat chat|size|đợt giao|dot giao|lô|hạn dùng|het han"),
]

# ---------------------------------------------------------------- ngành và kiểu sổ
INDUSTRY_NAMES = {"tap_hoa": "Tạp hoá", "ca_phe": "Cà phê", "vlxd": "Vật liệu xây dựng", "nha_thuoc": "Nhà thuốc",
                  "quan_nhau": "Quán nhậu", "khong_ro": "Không rõ (sổ chỉ có hoá đơn)"}
INDUSTRY_RULE_SECTION = {"ca_phe": "Cà phê", "nha_thuoc": "Nhà thuốc", "vlxd": "Vật liệu xây dựng"}


def industry_of(items: list) -> str:
    """Đoán ngành theo danh mục hàng. Mỗi ngành trong train dùng chung một danh mục cho 6 shop."""
    names = [str(x) for x in items]
    if not names:
        return "khong_ro"
    text = " ".join(names).lower()
    n = len(names)
    if sum(x.strip().endswith(("(M)", "(L)")) for x in names) >= 3:
        return "ca_phe"
    if len(re.findall(r"\bviên\b|\bvỉ\b|\bmg\b|tuýp", text)) >= n * 0.3:
        return "nha_thuoc"
    if re.search(r"xi măng|gạch|bay thợ|thép|bồn nước|bóng led", text):
        return "vlxd"
    if re.search(r"chiên|xào|lẩu|gỏi|nướng", text):
        return "quan_nhau"
    return "tap_hoa"


# (mã, mẫu nhận trong dòng mô tả đầu README, mô tả). Thứ tự quan trọng: mẫu cụ thể trước.
ARCHETYPES = [
    ("K_nhieu_nguon", r"ba nguồn|nhiều nguồn", "Dữ liệu ghép từ nhiều nguồn (Excel phần mềm, CSV, sổ tay)"),
    ("G_chi_hoa_don", r"Danh sách hoá đơn|Danh sách hóa đơn", "Mỗi dòng một hoá đơn, không có dòng hàng"),
    ("J_bao_cao_mot_file", r"mỗi báo cáo, danh sách một file", "Excel phần mềm: mỗi báo cáo, danh sách một file hoặc sheet"),
    ("I_so_chi_tiet", r"Sổ chi tiết bán hàng", "Excel phần mềm: sổ chi tiết bán hàng, mỗi dòng một dòng hàng"),
    ("B_xuat_nhieu_file", r"mỗi danh sách một file", "Excel phần mềm: mỗi danh sách một file, danh sách dài tách nhiều file"),
    ("C_don_moi_dong_sp", r"File đơn hàng mỗi dòng một sản phẩm", "Excel phần mềm: file đơn hàng mỗi dòng một sản phẩm"),
    ("D_so_tay_theo_thang", r"ghi tay chuyển sang Excel", "Sổ ghi tay chuyển sang Excel, mỗi tháng một sheet"),
    ("E_may_pos", r"máy POS", "File máy POS xuất (Excel nhiều bảng hoặc SQLite)"),
    ("F_so_chu_tiem", r"chủ tiệm tự ghi", "Sổ chủ tiệm tự ghi một sheet, thường không có mã đơn"),
    ("H_nhieu_file_noi_ma", r"nối với nhau qua mã", "Nhiều file CSV nối với nhau qua mã"),
    ("A_file_phang", r"Một file phẳng", "Một file CSV phẳng, mỗi dòng một dòng hàng"),
]


def archetype_of(readme: str) -> str:
    head = " ".join(readme.splitlines()[:4])
    for code, pat, _ in ARCHETYPES:
        if re.search(pat, head):
            return code
    return "?"


# ---------------------------------------------------------------- nhóm kết quả trong log (eval/mine_errors.py)
ERROR_GROUPS = {
    "T_ok": "Lớp tính toán trả lời đúng (không qua model)",
    "T_wrong": "Lớp tính toán trả lời sai: lỗi oracle, không dùng làm nhãn",
    "E1": "Đã nộp đáp án của model (đồng thuận hoặc đường bổ sung) nhưng sai: bị −1, ưu tiên cao nhất",
    "E2": "Bỏ trống nhưng có ít nhất một phiếu đúng: model làm được, chưa ổn định",
    "E3": "Bỏ trống, có phiếu ra đáp án nhưng không phiếu nào đúng",
    "E4": "Không phiếu nào ra đáp án (lỗi chạy, sai format, ra 0)",
    "E6": "Nộp đúng và mọi phiếu ra đáp án đều đúng",
    "E6b": "Nộp đúng nhưng có phiếu sai",
    "no_gold": "Không có đáp án chuẩn (lượt chạy public): chỉ thống kê",
}
ERROR_TAGS = {
    "E5_format": "Có lượt bị chặn vì sai kiểu hoặc sai format (TYPE_MISMATCH, TOPK_TOO_LONG, FORMAT_INVALID)",
    "E7_answerability": "Nhầm giữa có và không trả lời được (đáp án chuẩn unanswerable mà phiếu ra số, hoặc ngược lại)",
    "dropped_type": "Đáp án đồng thuận thuộc kiểu luôn bỏ trống (drop_types)",
}
STEP_CODES = {
    "NO_CODE_BLOCK": "Không có khối ```python``` hoàn chỉnh (thường do comment dài, bị cắt)",
    "EXECUTION_FAILED": "Code lỗi khi chạy; xem repair_type",
    "TIMEOUT": "Chạy quá thời gian",
    "FORMAT_INVALID": "ANSWER không qua được formatter",
    "TYPE_MISMATCH": "Sai kiểu đáp án so với cách hỏi (verify.py)",
    "TOPK_TOO_LONG": "Top k nhưng trả nhiều hơn k phần tử (verify.py)",
    "ZERO_RESULT": "Ra 0 ở câu không thể bằng 0 (verify.zero_suspect)",
}

# ---------------------------------------------------------------- mẫu lỗi đã biết
KNOWN_ERRORS = [
    ("dem_dong_thay_don", "K3", "Đếm số dòng (len(df), shape[0]) thay vì số mã đơn khác nhau", "dm.n_orders(df, cot_ma_don) hoặc df[cot_ma].nunique()", "báo cáo v2.7: 92 → 60; 1203 → 549"),
    ("cung_ky_thang_truoc", "K1", "'Tháng này so với tháng trước' lấy cả tháng trước", "So dm.this_month(TODAY) với dm.prev_month_same_days(TODAY)", "báo cáo v2.7: −12,96 → 19,85; rulebook §2"),
    ("khong_lam_tron_tien", "K13", "Trung bình tiền để số lẻ", "dm.money(x): làm tròn về đồng, nửa đồng lên", "báo cáo v2.7: 70873,95 → 70874; rulebook §9"),
    ("loc_ten_bang_bang", "K5", "So == với tên chép từ câu hỏi, sót cách viết khác", "dm.match_one / dm.eq / cột X__ten", "báo cáo v2.7: 2.352.000 → 2.640.000"),
    ("unanswerable_nham", "K11", "Kết luận không trả lời được khi thông tin nằm trong tên hàng (size M trong 'Trà sữa (M)')", "Tìm bằng dm.match(..., n=5, cutoff=0.4) trước khi dùng dm.unanswerable", "báo cáo v2.7: Size M unanswerable → 3584"),
    ("ra_so_0", "K5", "Bộ lọc rỗng nên ra 0", "Kiểm tra tên, kỳ, cột; 0 chỉ hợp lý với câu hàng ế, tồn, hết hàng", "báo cáo: 55 đáp án 0 ở lượt v2.5"),
    ("so_sanh_ngay", "K12", "So cột datetime64 trực tiếp với datetime.date", "p.mask(cot_ngay) hoặc pd.Timestamp(TODAY)", "system.md; repair.py SO_SANH_NGAY"),
    ("tra_mot_phan", "K2", "Chỉ bỏ dòng bị trả, giữ phần còn lại của đơn", "Đơn có trả hàng (kể cả một phần) bỏ cả đơn", "rulebook §1"),
    ("mau_so_ty_le_huy", "K7", "Mẫu số tỉ lệ huỷ chỉ gồm đơn hoàn thành", "Mẫu số là mọi đơn đặt trong kỳ, mọi trạng thái", "rulebook §1"),
    ("khach_le_la_khach", "K3", "Đếm hoặc xếp hạng cả 'Khách lẻ'", "Loại bằng dm.is_walkin", "rulebook §3"),
    ("ban_chay_theo_tien", "K6", "Bán chạy theo số tiền hoặc số dòng", "Theo số lượng quy về đơn vị cơ bản, bỏ đơn huỷ, trả", "rulebook §3"),
    ("cong_lan_don_vi", "K4", "Cộng lẫn số thùng với số lon", "Cột số lượng quy đổi (qty_base) hoặc hệ số trong bảng quy đổi", "rulebook §10"),
    ("tuan_truoc_7_ngay", "K1", "'Tuần trước' hiểu là 7 ngày gần nhất", "Cả tuần liền trước, thứ Hai → Chủ nhật: dm.last_week(TODAY)", "rulebook §2"),
    ("n_ngay_bo_hom_nay", "K1", "'N ngày gần nhất' không tính hôm nay", "N ngày kể cả hôm nay: dm.last_n_days(n, TODAY)", "rulebook §2"),
    ("tinh_ca_sau_hom_nay", "K1", "Tính cả bản ghi có ngày sau TODAY", "Bản ghi sau hôm nay coi như chưa xảy ra", "rulebook đầu file"),
    ("tet_mung_1", "K1", "Dịp Tết hiểu là quanh mùng 1", "23 tháng Chạp → hết mùng 10 tháng Giêng: dm.tet(Y), dm.before_tet(Y)", "rulebook §2"),
    ("hoa_tuy_y", "K6", "Bằng nhau thì lấy tuỳ ý", "Bằng nhau xếp theo mã tăng dần: dm.rank", "rulebook §9"),
    ("giam_ra_so_duong", "K7", "% giảm ghi số dương", "Giảm ra số âm: dm.pct_change(moi, cu)", "rulebook §1"),
    ("code_bi_cat", "K13", "Comment dài, câu trả lời bị cắt mất code", "Code ngắn, mỗi bước một dòng comment", "system.md; mã NO_CODE_BLOCK"),
    ("gan_answer_trong_ham", "K13", "Gán ANSWER trong def main() hoặc khối không chạy tới", "Gán ANSWER ở cấp ngoài cùng bằng hàm dm", "system.md; repair.py CHUA_GAN"),
]

# ---------------------------------------------------------------- biến thể luật (rulebook phản thực)
# kind: replace (thay dòng old bằng new) | add (chèn new sau dòng anchor).
# trigger: regex trên câu hỏi (chữ thường). Khi biến thể đang bật, eval/synth_tools.Kit.oracle KHÔNG trả lời câu khớp
# trigger hoặc thuộc affected_templates: lớp tính toán vẫn tính theo luật mặc định, nhãn sẽ sai.
# Các khái niệm thêm mới soạn từ khái niệm bán lẻ phổ biến, cố ý tránh các khái niệm đã biết là có trong rulebook public.
COUNTERFACTUAL = [
    {"id": "cf_dang_giao_khong_tinh", "kind": "replace", "section": "1. Đơn hàng và doanh thu", "skills": ["K2"],
     "old": "| Đơn tính vào doanh thu | đơn **không huỷ** và **không trả hàng**. Đơn có trả hàng, **kể cả chỉ trả một phần**, là đơn trả hàng: bỏ **cả đơn**. Đơn đang giao vẫn tính |",
     "new": "| Đơn tính vào doanh thu | chỉ đơn **hoàn thành**. Đơn huỷ, đơn trả hàng (**kể cả chỉ trả một phần**, bỏ **cả đơn**) và đơn **đang giao** đều **không** tính |",
     "effect": "Đơn đang giao không còn tính vào doanh thu, số đơn, xếp hạng", "trigger": r".", "affected_templates": "*"},
    {"id": "cf_ty_le_huy_bo_don_tra", "kind": "replace", "section": "1. Đơn hàng và doanh thu", "skills": ["K7", "K2"],
     "old": "| Tỉ lệ đơn huỷ trong kỳ | phần trăm đơn huỷ trong **mọi đơn đặt trong kỳ** (theo ngày đặt, **mọi trạng thái**: cả đơn huỷ, đơn trả, đơn đang giao). Làm tròn 2 chữ số thập phân. |",
     "new": "| Tỉ lệ đơn huỷ trong kỳ | phần trăm đơn huỷ trong các đơn đặt trong kỳ **không tính đơn trả hàng** (mẫu số gồm đơn hoàn thành, đơn đang giao, đơn huỷ). Làm tròn 2 chữ số thập phân. |",
     "effect": "Mẫu số tỉ lệ huỷ bỏ đơn trả hàng", "trigger": r"huỷ|hủy|huy", "affected_templates": ["cancel_rate"]},
    {"id": "cf_tuan_chu_nhat", "kind": "replace", "section": "2. Thời gian", "skills": ["K1"],
     "old": "| Tuần | bắt đầu **thứ Hai**. **Tuần này**: từ thứ Hai của tuần chứa hôm nay đến hết hôm nay. **Tuần trước, tuần rồi**: **cả** tuần liền trước, thứ Hai đến Chủ nhật |",
     "new": "| Tuần | bắt đầu **Chủ nhật**. **Tuần này**: từ Chủ nhật của tuần chứa hôm nay đến hết hôm nay. **Tuần trước, tuần rồi**: **cả** tuần liền trước, Chủ nhật đến thứ Bảy |",
     "effect": "Ranh giới tuần lệch một ngày", "trigger": r"tuần|tuan", "affected_templates": []},
    {"id": "cf_n_ngay_khong_gom_hom_nay", "kind": "replace", "section": "2. Thời gian", "skills": ["K1"],
     "old": "| N ngày gần nhất | N ngày liền nhau **kể cả hôm nay**, ngày cuối là hôm nay |",
     "new": "| N ngày gần nhất | N ngày liền nhau **không gồm hôm nay**, ngày cuối là **hôm qua** |",
     "effect": "Cửa sổ N ngày lùi một ngày", "trigger": r"\d+\s*(ngày|ngay)", "affected_templates": []},
    {"id": "cf_qua_n_gom_n", "kind": "replace", "section": "2. Thời gian", "skills": ["K3"],
     "old": "| Quá N, hơn N, trên N, dưới N | **không** gồm N. **Từ N trở lên** thì gồm N |",
     "new": "| Quá N, hơn N, trên N, dưới N | theo quy ước của shop **gồm cả** N. **Từ N trở lên** cũng gồm N |",
     "effect": "Ngưỡng 'quá/trên/dưới N' tính cả N", "trigger": r"quá|qua |hơn|hon |trên|tren |dưới|duoi|vượt|vuot", "affected_templates": []},
    {"id": "cf_thang_nay_ca_thang_truoc", "kind": "replace", "section": "2. Thời gian", "skills": ["K1", "K7"],
     "old": "| Tháng này | từ ngày 1 đến **hết hôm nay**. So với tháng trước thì lấy **cùng số ngày** của tháng trước: từ ngày 1 đến hết ngày trùng số với hôm nay; tháng trước không có ngày đó thì đến hết tháng trước |",
     "new": "| Tháng này | từ ngày 1 đến **hết hôm nay**. So với tháng trước thì lấy **cả tháng trước**, từ ngày 1 đến hết ngày cuối tháng |",
     "effect": "Kỳ so sánh của 'tháng này' là cả tháng trước", "trigger": r"tháng này|thang nay|tháng trước|thang truoc|cùng kỳ|cung ky", "affected_templates": []},
    {"id": "cf_quy_tu_thang_4", "kind": "replace", "section": "2. Thời gian", "skills": ["K1"],
     "old": "| Tháng M, quý Q không ghi năm | của **năm nay** nếu không sau hôm nay; sau hôm nay thì của **năm trước**. Quý theo dương lịch, mỗi quý 3 tháng tính từ tháng 1 |",
     "new": "| Tháng M, quý Q không ghi năm | của **năm nay** nếu không sau hôm nay; sau hôm nay thì của **năm trước**. Quý theo **năm tài chính của shop**: quý 1 là tháng 4–6, quý 2 tháng 7–9, quý 3 tháng 10–12, quý 4 tháng 1–3 năm sau |",
     "effect": "Quý lệch 3 tháng", "trigger": r"quý|quy \d|q\d", "affected_templates": []},
    {"id": "cf_tet_20_chap_mung_6", "kind": "replace", "section": "2. Thời gian", "skills": ["K1"],
     "old": "| Tết năm Y, dịp Tết, mùa Tết | từ ngày dương của **23 tháng Chạp** (năm âm Y − 1) đến **hết** ngày dương của **mùng 10 tháng Giêng** (năm âm Y). **So với trước Tết** = cùng số ngày ngay trước 23 tháng Chạp. Tra lịch âm trong `calendar.csv` cùng thư mục. |",
     "new": "| Tết năm Y, dịp Tết, mùa Tết | từ ngày dương của **20 tháng Chạp** (năm âm Y − 1) đến **hết** ngày dương của **mùng 6 tháng Giêng** (năm âm Y). **So với trước Tết** = cùng số ngày ngay trước 20 tháng Chạp. Tra lịch âm trong `calendar.csv` cùng thư mục. |",
     "effect": "Cửa sổ Tết ngắn hơn và sớm hơn", "trigger": r"tết|tet", "affected_templates": []},
    {"id": "cf_ca_ba_ca", "kind": "replace", "section": "2. Thời gian", "skills": ["K1"],
     "old": "| Ca sáng, ca chiều | theo giờ tạo đơn: ca sáng 06:00–13:59, ca chiều 14:00–21:59. So ca chiều với ca sáng: mốc là **ca sáng** |",
     "new": "| Ca sáng, ca chiều, ca tối | theo giờ tạo đơn: ca sáng 06:00–11:59, ca chiều 12:00–17:59, ca tối 18:00–22:59. So hai ca: mốc là **ca sớm hơn** |",
     "effect": "Giờ ca đổi, thêm ca tối", "trigger": r"\bca\b", "affected_templates": ["shift", "shift_change"], "needs_roles": ["time"]},
    {"id": "cf_ban_chay_theo_doanh_thu", "kind": "replace", "section": "3. Xếp hạng", "skills": ["K6"],
     "old": "| Bán chạy | theo **số lượng quy về đơn vị cơ bản**, không theo số dòng hay số tiền; bỏ đơn huỷ, trả. Mặt hàng khác đơn vị cơ bản vẫn xếp chung theo số đó. **Bán nhiều nhất**, **bán được nhiều nhất**, **hot nhất** cùng nghĩa với **bán chạy nhất** |",
     "new": "| Bán chạy | theo **doanh thu** (§1) của mặt hàng, không theo số lượng hay số dòng; bỏ đơn huỷ, trả. **Bán nhiều nhất**, **bán được nhiều nhất**, **hot nhất** cùng nghĩa với **bán chạy nhất** |",
     "effect": "Xếp hạng bán chạy theo tiền", "trigger": r"chạy|chay|bán nhiều|ban nhieu|hot|best", "affected_templates": ["top_item", "top_items"]},
    {"id": "cf_nhan_vien_theo_so_don", "kind": "replace", "section": "3. Xếp hạng", "skills": ["K6", "K3"],
     "old": "| Nhân viên bán giỏi nhất | theo **doanh thu** (§1) của nhân viên trong kỳ, không theo số đơn |",
     "new": "| Nhân viên bán giỏi nhất | theo **số đơn** (§1) của nhân viên trong kỳ; bằng số đơn thì theo doanh thu |",
     "effect": "Nhân viên giỏi nhất theo số đơn", "trigger": r"nhân viên|nhan vien|\bnv\b", "affected_templates": ["staff_best"], "needs_roles": ["staff"]},
    {"id": "cf_tb_don_lam_tron_nghin", "kind": "replace", "section": "3. Xếp hạng", "skills": ["K13"],
     "old": "| Giá trị trung bình một đơn | làm tròn về đồng, nửa đồng làm tròn lên |",
     "new": "| Giá trị trung bình một đơn | làm tròn đến **nghìn đồng**, nửa nghìn làm tròn lên |",
     "effect": "Làm tròn trung bình đơn đến nghìn", "trigger": r"trung bình|trung binh|bình quân|binh quan|\btb\b", "affected_templates": ["avg_order", "g_avg"]},
    {"id": "cf_tuoi_no_tinh_ca_ngay_dat", "kind": "replace", "section": "4. Công nợ", "skills": ["K8"],
     "old": "| Tuổi nợ | số ngày tính từ **ngày đặt**, không từ hạn thanh toán. Đơn đặt hôm nay có tuổi nợ **0** ngày, đơn đặt hôm qua **1** ngày |",
     "new": "| Tuổi nợ | số ngày tính từ **ngày đặt**, **tính cả ngày đặt**. Đơn đặt hôm nay có tuổi nợ **1** ngày, đơn đặt hôm qua **2** ngày |",
     "effect": "Tuổi nợ cộng thêm một ngày", "trigger": r"tuổi nợ|tuoi no|nợ quá|no qua|nợ lâu|no lau", "affected_templates": ["debt_bands"], "needs_tables": ["cong_no"]},
    {"id": "cf_gia_nhap_trung_binh_cong", "kind": "replace", "section": "5. Kho, nhập hàng, giá vốn", "skills": ["K9"],
     "old": "| Giá nhập bình quân một mặt hàng trong tháng | bình quân **theo số lượng nhập**, **không** phải trung bình cộng các mức giá nhập; theo đơn vị nhập, các phiếu có ngày phiếu trong tháng, bỏ phiếu huỷ |",
     "new": "| Giá nhập bình quân một mặt hàng trong tháng | **trung bình cộng các mức giá nhập** của các phiếu có ngày phiếu trong tháng, mỗi phiếu tính một lần, không theo số lượng; theo đơn vị nhập, bỏ phiếu huỷ |",
     "effect": "Giá nhập bình quân không trọng số", "trigger": r"giá nhập|gia nhap", "affected_templates": ["avg_purchase"], "needs_tables": ["nhap_hang"]},
    {"id": "cf_ke_don_theo_so_luong", "kind": "replace", "section": "Nhà thuốc", "skills": ["K10"],
     "old": "| Thuốc kê đơn | **Thuốc kê đơn** = sản phẩm ghi kê đơn trong danh mục thuốc; mọi sản phẩm còn lại là **hàng không kê đơn**. Tỉ lệ kê đơn tính theo doanh thu |",
     "new": "| Thuốc kê đơn | **Thuốc kê đơn** = sản phẩm ghi kê đơn trong danh mục thuốc; mọi sản phẩm còn lại là **hàng không kê đơn**. Tỉ lệ kê đơn tính theo **số lượng quy về đơn vị cơ bản** |",
     "effect": "Tỉ lệ kê đơn theo số lượng", "trigger": r"kê đơn|ke don|bác sĩ|bac si", "affected_templates": ["rx_share"], "needs_roles": ["rx"]},
    {"id": "new_don_lon", "kind": "add", "section": "3. Xếp hạng", "skills": ["K3", "K2"],
     "anchor": "| Giá trị trung bình một đơn | làm tròn về đồng, nửa đồng làm tròn lên |",
     "new": "| Đơn lớn | đơn (§1) có doanh thu từ **1.000.000 đồng** trở lên |",
     "effect": "Khái niệm mới: đơn lớn", "trigger": r"đơn lớn|don lon", "affected_templates": []},
    {"id": "new_khach_moi", "kind": "add", "section": "3. Xếp hạng", "skills": ["K3"],
     "anchor": "| Giá trị trung bình một đơn | làm tròn về đồng, nửa đồng làm tròn lên |",
     "new": "| Khách mới trong kỳ | khách (§3) có **đơn đầu tiên** (§1) của mình nằm trong kỳ; khách lẻ không tính |",
     "effect": "Khái niệm mới: khách mới", "trigger": r"khách mới|khach moi", "affected_templates": [], "needs_roles": ["cust"]},
    {"id": "new_don_nhieu_mon", "kind": "add", "section": "3. Xếp hạng", "skills": ["K3", "K12"],
     "anchor": "| Giá trị trung bình một đơn | làm tròn về đồng, nửa đồng làm tròn lên |",
     "new": "| Đơn nhiều món | đơn (§1) có từ **5 mặt hàng khác nhau** trở lên |",
     "effect": "Khái niệm mới: đơn nhiều món", "trigger": r"nhiều món|nhieu mon", "affected_templates": []},
    {"id": "new_ty_le_tra_hang", "kind": "add", "section": "1. Đơn hàng và doanh thu", "skills": ["K7", "K2"],
     "anchor": "| Phần trăm thay đổi, tăng trưởng | so với **kỳ trước**; giảm thì ra **số âm** |",
     "new": "| Tỉ lệ đơn trả hàng trong kỳ | phần trăm đơn trả hàng trong **mọi đơn đặt trong kỳ** (mọi trạng thái). Làm tròn 2 chữ số thập phân |",
     "effect": "Khái niệm mới: tỉ lệ trả hàng", "trigger": r"trả hàng|tra hang|trả lại|tra lai", "affected_templates": []},
    {"id": "new_ngay_thap_diem", "kind": "add", "section": "2. Thời gian", "skills": ["K1", "K6"],
     "anchor": "| Tháng M/Y | cả tháng, từ ngày 1 đến hết ngày cuối tháng |",
     "new": "| Ngày thấp điểm | ngày **có ít nhất một đơn** (§1) trong kỳ mà doanh thu thấp nhất; bằng nhau lấy ngày sớm hơn |",
     "effect": "Khái niệm mới: ngày thấp điểm", "trigger": r"thấp điểm|thap diem|ế nhất|e nhat", "affected_templates": []},
]

# ---------------------------------------------------------------- phong cách câu hỏi (đo trên câu train)
ABBREVIATIONS = ["dthu", "dt", "bn", "bnh", "bnhiu", "nhiu", "sp", "dc", "đc", "z", "j", "ko", "hok", "k", "hqua", "hnay", "trc", "vs", "kh",
                 "nv", "sl", "tb", "ncc", "dvt", "đvt", "ddown", "hum", "bữa", "nhứt", "nhut", "order", "bill", "revenue", "total", "best seller"]
FILLERS = ["cho hỏi", "cho mình hỏi", "xem giúp", "xem giùm", "coi giùm", "check giùm", "check giúp", "tính giúp", "ạ", "nhỉ", "nhé", "nha",
           "vậy", "z", "ha", "hả", "đó", "thế", "anh ơi", "chị ơi", "em ơi", "giùm", "với"]
PERIODS = [("hom_nay", r"hôm nay|hom nay|bữa nay|hnay"), ("hom_qua", r"hôm qua|hqua|hom qua"), ("tuan_nay", r"tuần này|tuan nay"),
           ("tuan_truoc", r"tuần trước|tuần rồi|tuan truoc|tuan roi"), ("thang_nay", r"tháng này|thang nay"), ("thang_truoc", r"tháng trước|thang truoc"),
           ("thang_cu_the", r"tháng \d+|thang \d+|t\d+/\d{4}"), ("quy", r"quý|quy \d"), ("nam", r"năm \d{4}|nam \d{4}|từ đầu năm|tu dau nam"),
           ("n_ngay", r"\d+ ngày|\d+ ngay"), ("tet", r"tết|tet"), ("ngay_cu_the", r"ngày \d+/\d+|ngay \d+/\d+|\d+/\d+/\d{4}"),
           ("cung_ky", r"cùng kỳ|cung ky"), ("den_het", r"đến hết|tính đến|den het|tinh den")]
DIACRITIC = re.compile("[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]")
# Gõ telex sót ('tieenf', 'soos', 'dduwowcj', 'vaayj'): chữ Việt không dấu không có cặp telex hay tận cùng bằng phím dấu thanh.
# known = từ tiếng Anh/thương hiệu đã gặp (tên hàng, viết tắt) để không bắt nhầm. ~7,6% câu mining có lỗi này.
TELEX_PAIR = re.compile(r"dd|aa|ee|oo|aw|ow|uw")
TELEX_TONE_END = re.compile(r"[aeiouy](?:ng|nh|ch|[a-z])?[fjsrx]$")  # phụ âm cuối 2 chữ: thangs, nhanhf


def telex_words(text: str, known=frozenset()) -> list[str]:
    return [w for w in re.findall(r"(?<![\w])[a-z]+(?![\w])", text.lower())
            if len(w) >= 2 and w not in known and (TELEX_PAIR.search(w) or TELEX_TONE_END.search(w))]


# ================================================================ GIAI ĐOẠN 1: SINH DỮ LIỆU SHOP
# Lớp "sự thật": bảng sạch do bộ mô phỏng sinh. Đáp án giai đoạn 2 tính trực tiếp trên các bảng này, không qua loader.
CANONICAL_SCHEMA = {
    "shop": {"mo_ta": "Một dòng: thông tin shop", "cot": {
        "business_id": "str", "industry": "str — mã ngành", "archetype": "str — kiểu sổ khi xuất", "date_start": "date", "date_end": "date — ngày cuối có dữ liệu",
        "open_hours": "[giờ mở, giờ đóng]", "seed": "int — hạt ngẫu nhiên", "has": "dict — shop có ghi: customers, staff, channel, pay, debt, purchase, stock, lots, deliveries, expenses, price_history"}},
    "items": {"mo_ta": "Danh mục hàng (cả dịch vụ, phí phục vụ)", "cot": {
        "item_code": "str — mã hàng, duy nhất", "name": "str — tên đầy đủ, duy nhất", "group": "str|null — nhóm hàng", "base_unit": "str — đơn vị cơ bản",
        "base_price": "int — giá bán một đơn vị cơ bản lúc đầu", "cost_ratio": "float — giá vốn / giá bán", "is_service": "bool — dịch vụ, phí phục vụ (không phải mặt hàng)",
        "attrs": "dict — thuộc tính theo ngành: rx, active_ingredient, size, warranty_months, brand, shelf_life_days…"},
        "rang_buoc": ["item_code, name duy nhất", "base_price > 0", "0 < cost_ratio < 1"]},
    "units": {"mo_ta": "Quy đổi đơn vị lớn", "cot": {"item_code": "str", "unit": "str — đơn vị lớn", "factor": "int — số đơn vị cơ bản trong 1 đơn vị lớn"},
              "rang_buoc": ["factor > 1", "nhiều cấp: factor tính thẳng ra đơn vị cơ bản (hộp = 10 vỉ = 100 viên → hộp 100)"]},
    "price_history": {"mo_ta": "Bảng giá theo ngày áp dụng (giá một đơn vị cơ bản)", "cot": {"item_code": "str", "effective_date": "date", "price": "int", "old_price": "int|null"},
                      "rang_buoc": ["giá trên dòng bán = giá đang áp dụng ngày bán × factor (trừ chiết khấu dòng)"]},
    "customers": {"mo_ta": "Khách có lưu (khách lẻ không có dòng)", "cot": {
        "cust_code": "str", "name": "str — có thể trùng tên giữa hai mã", "phone": "str", "group": "str|null — nhóm khách", "created_date": "date"}},
    "staff": {"mo_ta": "Nhân viên bán", "cot": {"staff_code": "str", "name": "str"}},
    "orders": {"mo_ta": "Đầu đơn", "cot": {
        "order_id": "str", "created_at": "datetime — ngày đặt + giờ", "cust_code": "str|null — null = khách lẻ", "status": "hoan_thanh|huy|tra_hang|dang_giao",
        "channel": "str|null", "pay_method": "str|null — tiền mặt, chuyển khoản, COD, ghi nợ, sàn…", "staff_code": "str|null",
        "order_discount": "int — giảm giá cả đơn (không trừ vào doanh thu §1)", "shipping_fee": "int — phí giao (không vào doanh thu §1)", "note": "str|null"},
        "rang_buoc": ["đang giao chỉ ở các ngày cuối dữ liệu", "đơn huỷ không có thanh toán, không giao"]},
    "order_lines": {"mo_ta": "Dòng hàng", "cot": {
        "order_id": "str", "line_no": "int", "item_code": "str", "unit": "str — đơn vị ghi trên dòng", "qty": "number — theo đơn vị dòng",
        "qty_base": "number — quy ra đơn vị cơ bản", "unit_price": "int — giá một đơn vị dòng, trước chiết khấu", "line_discount": "int",
        "amount": "int — thành tiền sau chiết khấu dòng", "unit_cost": "int|null — giá vốn một đơn vị dòng", "lot_id": "str|null"},
        "rang_buoc": ["amount = qty × unit_price − line_discount", "qty_base = qty × factor(unit)", "mỗi (order_id, line_no) duy nhất"]},
    "returns": {"mo_ta": "Trả hàng (kể cả trả một phần)", "cot": {"order_id": "str", "line_no": "int", "return_date": "date", "qty": "number", "refund": "int"},
                "rang_buoc": ["đơn có dòng trả thì status = tra_hang", "return_date ≥ ngày đặt", "qty trả ≤ qty bán"]},
    "payments": {"mo_ta": "Các khoản trả của đơn (trả lúc mua và trả nợ về sau)", "cot": {"order_id": "str", "pay_date": "date", "amount": "int", "method": "str"},
                 "rang_buoc": ["tổng trả ≤ tổng đơn (đã trừ giảm giá cả đơn, cộng phí giao)", "đơn COD, sàn không ghi nợ"]},
    "debt_terms": {"mo_ta": "Hạn nợ của đơn ghi nợ", "cot": {"order_id": "str", "days_allowed": "int", "due_date": "date"}},
    "deliveries": {"mo_ta": "Giao nhiều đợt (VLXD và ngành có giao)", "cot": {
        "order_id": "str", "batch_no": "int", "delivery_date": "date", "item_code": "str", "qty": "number", "vehicle": "str", "driver": "str"},
        "rang_buoc": ["tổng giao mỗi dòng ≤ qty bán", "delivery_date ≥ ngày đặt"]},
    "purchases": {"mo_ta": "Phiếu nhập (đầu phiếu + dòng)", "cot": {
        "receipt_id": "str", "date": "date", "supplier": "str", "status": "hoan_thanh|huy", "note": "str — có 'nhập đầu kỳ'", "item_code": "str", "unit": "str",
        "qty": "number", "unit_cost": "int", "amount": "int", "lot_id": "str|null", "expiry_date": "date|null"}},
    "lots": {"mo_ta": "Lô hàng (nhà thuốc, thực phẩm)", "cot": {"lot_id": "str", "item_code": "str", "import_date": "date", "expiry_date": "date", "qty_in": "number", "unit_cost": "int"},
             "rang_buoc": ["tồn lô không âm theo thời gian", "lô bán ra theo hạn gần nhất trước (FEFO)"]},
    "stock_snapshot": {"mo_ta": "Tồn cuối ngày cuối dữ liệu (suy ra, để xuất báo cáo tồn)", "cot": {"item_code": "str", "date": "date", "qty_base": "number"}},
    "expenses": {"mo_ta": "Sổ chi phí", "cot": {"date": "date", "category": "str", "amount": "int", "note": "str"}},
}

# Danh mục ngành mới do teacher sinh (prompts/7_danh_muc_nganh_moi.md); script kiểm theo khuôn này
CATALOG_SCHEMA = {
    "industry": "mã ngành không dấu, vd 'nha_sach'", "ten": "tên ngành tiếng Việt",
    "items": [{"name": "tên đầy đủ như trên phần mềm bán hàng", "group": "nhóm hàng", "base_unit": "đơn vị cơ bản",
               "big_units": [{"unit": "đơn vị lớn", "factor": "số đơn vị cơ bản"}], "price_vnd": "[giá thấp, giá cao] một đơn vị cơ bản",
               "cost_ratio": "giá vốn / giá bán, 0.4–0.9", "popularity": "1–5", "attrs": {}}],
    "service_lines": ["dòng không phải mặt hàng: phí phục vụ, phí giao, công lắp đặt…"],
    "customer_groups": ["nhóm khách"], "channels": ["kênh bán"], "pay_methods": ["hình thức thanh toán"],
    "open_hours": "[giờ mở, giờ đóng]", "orders_per_day": "[thấp, cao]", "lines_per_order": "[p50, p90]",
    "seasonality": {"tet_multiplier": "hệ số số đơn dịp Tết so với trước Tết", "events": [{"ten": "dịp", "thang_ngay": "MM-DD", "he_so": "hệ số"}],
                    "weekday": "hệ số thứ 2 → CN"},
    "has": {"customers": True, "staff": False, "debt": False, "purchase": True, "stock": True, "lots": False, "deliveries": False, "expenses": False},
    "rules_section_6": ["2–4 dòng luật ngành theo đúng khuôn bảng của rulebook: '| Luật | Mặc định |'"],
}

# Ví dụ đúng kiểu cho prompt 7 (ngành khác hẳn, tự soạn): cho teacher thấy số là số, khoảng là [thấp, cao], luật là định nghĩa tính được.
CATALOG_EXAMPLE = {
    "industry": "do_cau_ca", "ten": "đồ câu cá",
    "items": [{"name": "Cần câu tay carbon 5m4 Shimano", "group": "Cần câu", "base_unit": "cây", "big_units": [], "price_vnd": [350000, 420000],
               "cost_ratio": 0.68, "popularity": 4, "attrs": {"brand": "Shimano", "chuyen_nghiep": True}},
              {"name": "Lưỡi câu Owner số 7 (vỉ 10 lưỡi)", "group": "Lưỡi câu", "base_unit": "vỉ", "big_units": [{"unit": "hộp", "factor": 20}],
               "price_vnd": [25000, 30000], "cost_ratio": 0.55, "popularity": 5, "attrs": {"chuyen_nghiep": False}}],
    "service_lines": ["Công thay dây cước"], "customer_groups": ["khách lẻ", "hội câu", "đại lý nhỏ"], "channels": ["tại cửa hàng", "Zalo"],
    "pay_methods": ["tiền mặt", "chuyển khoản", "ghi nợ"], "open_hours": [7, 21], "orders_per_day": [15, 40], "lines_per_order": [2, 5],
    "seasonality": {"tet_multiplier": 0.6, "events": [{"ten": "mở mùa câu biển", "thang_ngay": "04-15", "he_so": 1.4}],
                    "weekday": [0.8, 0.8, 0.9, 0.9, 1.1, 1.4, 1.5]},
    "has": {"customers": True, "staff": False, "debt": True, "purchase": True, "stock": True, "lots": False, "deliveries": False, "expenses": True},
    "rules_section_6": ["| Đồ câu chuyên nghiệp | mặt hàng có attrs.chuyen_nghiep = true; doanh thu tính như doanh thu chung |",
                        "| Số lưỡi câu bán ra | tổng số lượng nhóm Lưỡi câu quy về đơn vị cơ bản (vỉ) theo hệ số quy đổi, bỏ đơn huỷ |"],
}

# Công thức xuất: kiểu sổ → họ file → nguồn bảng chuẩn. Tên file/sheet, tên cột, định dạng, bẫy lấy theo data_spec.
RENDER_RECIPES = {
    "A_file_phang": [{"file": "ban_hang.csv", "nguon": "order_lines ⋈ orders ⋈ items (+units)", "chia": None}],
    "B_xuat_nhieu_file": [{"file": "DanhSachHoaDon.xlsx", "nguon": "orders (+payments)", "chia": "sheet theo tháng hoặc file _1,_2"},
                          {"file": "DanhSachHoaDonChiTiet.xlsx", "nguon": "order_lines ⋈ items", "chia": "file _1,_2"},
                          {"file": "DanhSachSanPham.xlsx", "nguon": "items"}, {"file": "DanhSachKhachHang.xlsx", "nguon": "customers"},
                          {"file": "DanhSachQuyDoi.xlsx", "nguon": "units"}, {"file": "DanhSachNhapHang.xlsx", "nguon": "purchases (đầu phiếu)"},
                          {"file": "DanhSachChiTietNhapHang.xlsx", "nguon": "purchases (dòng)"}, {"file": "BaoCaoTonKho.xlsx", "nguon": "stock_snapshot"}],
    "C_don_moi_dong_sp": [{"file": "DonHang_ChiTiet.xlsx", "nguon": "order_lines ⋈ orders, ô đầu đơn để trống ở dòng sau", "chia": "sheet theo tháng"},
                          {"file": "DanhSachQuyDoi.xlsx", "nguon": "units"}, {"file": "DanhSachSanPham.xlsx", "nguon": "items"}],
    "D_so_tay_theo_thang": [{"file": "SoBanHang.xlsx", "nguon": "order_lines ⋈ orders, tiêu đề hai tầng, dòng tổng, tiền ghi tắt", "chia": "sheet mỗi tháng, file theo năm"},
                            {"file": "BangGia.xlsx", "nguon": "price_history"}, {"file": "DanhMuc.xlsx", "nguon": "items ⋈ units"},
                            {"file": "SoNhapHang.xlsx", "nguon": "purchases"}, {"file": "TonKho.xlsx", "nguon": "stock_snapshot"}],
    "E_may_pos": [{"file": "pos.sqlite hoặc DuLieuBanHang.xlsx", "nguon": "mọi bảng chuẩn, mỗi bảng một bảng/sheet, nối qua mã"}],
    "F_so_chu_tiem": [{"file": "SoBanHang.xlsx", "nguon": "order_lines ⋈ orders, KHÔNG có mã đơn, số lượng kèm đơn vị", "chia": None}],
    "G_chi_hoa_don": [{"file": "danh_sach_hoa_don.csv", "nguon": "orders + tổng tiền + đã trả/còn nợ, không có dòng hàng, có dòng cộng theo ngày"}],
    "H_nhieu_file_noi_ma": [{"file": "don_hang.csv", "nguon": "orders"}, {"file": "chi_tiet_don.csv", "nguon": "order_lines"}, {"file": "san_pham.csv", "nguon": "items"},
                            {"file": "khach_hang.csv", "nguon": "customers"}, {"file": "thanh_toan.csv", "nguon": "payments"}],
    "I_so_chi_tiet": [{"file": "So_chi_tiet_ban_hang.xlsx", "nguon": "order_lines ⋈ orders, dòng tên báo cáo + kỳ báo cáo", "chia": "sheet theo tháng"},
                      {"file": "Danh sách hàng hóa.xlsx", "nguon": "items (nhiều cột thừa)"}],
    "J_bao_cao_mot_file": [{"file": "Bao_cao_ban_hang.xlsx", "nguon": "mỗi báo cáo một sheet: sổ chi tiết, hàng hoá, phiếu giao, công nợ theo hạn"}],
    "K_nhieu_nguon": [{"file": "BanHang_*.xlsx", "nguon": "phần mềm: hoá đơn, chi tiết, hàng hoá, khách, lô, phiếu thu, lịch sử giá"},
                      {"file": "SoTay_*.xlsx", "nguon": "sổ tay: nhập hàng, chi phí"}, {"file": "ton_kho.csv", "nguon": "stock_snapshot"}],
}

# Lọc CỤC BỘ đầu ra của teacher (không gửi danh sách này ra API): ngành và khái niệm đã biết là chỉ có trong rulebook public
# (theo docs nội bộ của nhóm). Danh mục/luật sinh ra trùng các mục này thì bỏ, để không bị xem là chỉnh theo public (luật 6).
LOCAL_BLOCKLIST = {
    "nganh": ["mẹ và bé", "me va be", "sửa xe", "sua xe", "điện thoại", "dien thoai"],
    "khai_niem": ["khách sộp", "giờ cao điểm", "nhà cung cấp chính", "khách trung thành", "khách hay đổi ý", "mặt hàng ưa chuộng", "thợ chăm chỉ"],
}

# ---------------------------------------------------------------- ngành mới (đề xuất cho giai đoạn 1)
# Chọn theo 3 tiêu chí: (1) teacher biết rõ thị trường VN (tên hàng, đơn vị, giá) → JSON đạt cao; (2) buộc dữ liệu có cấu trúc model
# đang yếu (nhiều bảng, quy đổi đơn vị, lô/hạn, công nợ, giao nhiều đợt, trả hàng — xem data_spec/shop_complexity.json);
# (3) có luật ngành riêng tính được từ dữ liệu (nhóm lỗi "nganh" đứng thứ 4 trong log hard42). Không chọn ngành trong LOCAL_BLOCKLIST.
# 'cau_truc' dùng cùng nhãn với shop_complexity: quy_doi, lo, cong_no, giao, nhap, ton, tra, thanh_toan (+ so_le, bien_the, coc, dich_vu).
NEW_INDUSTRIES = [
    {"code": "nha_sach_vpp", "ten": "nhà sách – văn phòng phẩm", "nhom": "chinh",
     "cau_truc": ["quy_doi", "cong_no", "nhap", "ton", "tra"], "kieu_so": ["B", "E", "J", "A", "I"],
     "dac_trung": "quy đổi nhiều cấp (thùng/hộp/cây/ram); khách sỉ là trường, công ty ghi nợ; mùa tựu trường tháng 8–9"},
    {"code": "hoa_tuoi", "ten": "tiệm hoa tươi", "nhom": "chinh",
     "cau_truc": ["lo", "giao", "tra", "coc", "thanh_toan"], "kieu_so": ["F", "H", "K", "D", "B"],
     "dac_trung": "hoa nhập theo lô, hạn 3–7 ngày, hao hụt; đặt trước có cọc, giao tận nơi; đỉnh 14/2, 8/3, 20/10, 20/11, rằm, Tết"},
    {"code": "my_pham", "ten": "mỹ phẩm", "nhom": "chinh",
     "cau_truc": ["lo", "tra", "thanh_toan", "nhap", "ton"], "kieu_so": ["E", "C", "H", "B", "A"],
     "dac_trung": "lô/hạn dùng; kênh tại quầy, website, sàn, livestream; đổi trả; combo khuyến mãi; ngày đôi (9/9, 11/11, 12/12)"},
    {"code": "trai_cay", "ten": "trái cây – nông sản", "nhom": "du_phong",  # 9/11: danh mục đạt 4/9 (thấp nhất) → dự phòng
     "cau_truc": ["so_le", "lo", "tra", "cong_no"], "kieu_so": ["F", "D", "A", "I", "K"],
     "dac_trung": "bán theo kg, số lượng thập phân; giá sỉ/lẻ; hao hụt; giỏ quà Tết, rằm tháng 7, Trung thu"},
    {"code": "tiem_banh", "ten": "tiệm bánh", "nhom": "chinh",
     "cau_truc": ["coc", "thanh_toan", "bien_the", "giao"], "kieu_so": ["H", "E", "F", "C", "J"],
     "dac_trung": "bánh kem đặt trước có cọc, trả phần còn lại khi nhận; size/biến thể; ca sáng/chiều; Trung thu, Noel"},
    {"code": "dien_gia_dung", "ten": "điện gia dụng", "nhom": "chinh",
     "cau_truc": ["cong_no", "giao", "dich_vu", "tra", "nhap", "ton"], "kieu_so": ["J", "B", "I", "K", "D"],
     "dac_trung": "đơn giá cao, ít đơn; trả góp/ghi nợ có hạn; giao + công lắp đặt; bảo hành đổi máy; mùa nóng (quạt, máy lạnh)"},
    {"code": "thoi_trang", "ten": "thời trang – giày dép", "nhom": "chinh",
     "cau_truc": ["bien_the", "tra", "thanh_toan", "nhap", "ton"], "kieu_so": ["C", "E", "H", "B", "A"],
     "dac_trung": "biến thể size × màu (nhiều mã hàng); đổi size/trả hàng nhiều; kênh online; xả hàng cuối mùa"},
    {"code": "vat_tu_nong_nghiep", "ten": "vật tư nông nghiệp", "nhom": "chinh",
     "cau_truc": ["cong_no", "quy_doi", "lo", "nhap", "ton", "giao", "so_le"], "kieu_so": ["D", "I", "J", "K", "F"],
     "dac_trung": "phân bón, giống, thuốc bảo vệ thực vật; bán chịu theo mùa vụ, thu nợ sau thu hoạch; bao/kg, phân lẻ cân ký (số lượng thập phân); hạn thuốc theo lô"},
    {"code": "gas_nuoc", "ten": "gas – nước uống giao tận nơi", "nhom": "chinh",  # 9/11: danh mục đạt 8/9 → lên nhóm chính
     "cau_truc": ["giao", "coc", "cong_no", "thanh_toan"], "kieu_so": ["H", "F", "G", "B", "I"],
     "dac_trung": "cọc vỏ bình; giao theo tuyến; khách quen đặt định kỳ; ghi nợ theo tháng"},
    {"code": "thu_cung", "ten": "cửa hàng thú cưng", "nhom": "du_phong",
     "cau_truc": ["dich_vu", "quy_doi", "khach", "tra"], "kieu_so": ["E", "H", "A"],
     "dac_trung": "hàng (thức ăn bao/kg) + dịch vụ (tắm, cắt tỉa, lưu chuồng); khách có mã"},
]

# Khối lượng giai đoạn 1: đợt thử → đợt chính → (tuỳ) đợt mở rộng. Tầng độ phức tạp đo bằng số bảng thô (data_spec/shop_complexity.json):
# đơn giản ≤ 2 bảng, vừa 3–9, phức tạp ≥ 10 hoặc có lô/quy đổi/công nợ từ nhiều nguồn. Train: 8 / 8 / 14 shop.
VOLUME_PLAN = {
    "thang_moi_shop": [12, 18],
    "ti_le_tang": {"don_gian": 0.20, "vua": 0.30, "phuc_tap": 0.50},
    "sinh_du": 0.15,
    "dot": [
        {"ten": "0_thu", "shop": 13, "cach": "1 shop/ngành (5 ngành có sẵn + 8 ngành mới), kiểu sổ phức tạp; sửa bộ sinh tới khi G1–G4 đạt 100%"},
        {"ten": "A_nganh_co_san", "shop": 30, "cach": "5 ngành × 6 kiểu sổ ngành đó CHƯA có trong train; nửa số shop đổi tên hàng/khách một-một"},
        {"ten": "B_nganh_moi", "shop": 40, "cach": "8 ngành chính × 5 kiểu sổ (1 đơn giản, 1–2 vừa, 2–3 phức tạp)"},
        {"ten": "C_mo_rong", "shop": 48, "cach": "chỉ chạy nếu đo giai đoạn 2 còn tăng: 2 ngành dự phòng + thêm kiểu sổ phức tạp cho ngành yếu"},
    ],
    "cau_hoi_moi_shop_giai_doan_2": [50, 60],
}

# Chia việc cho 4 model theo độ khó × độ phức tạp, theo số đo 9/11 (eval/model_bakeoff.py → data_spec/model_bakeoff.json).
# Ký hiệu "model:mức suy nghĩ"; danh sách là thứ tự dùng: model đầu làm chính, hỏng (sau một lần sửa kèm thông báo lỗi) thì chuyển model sau.
# Quy tắc rút ra: (1) DeepSeek/Qwen bật suy nghĩ với đầu ra dài thì nghĩ không dừng (T2: 0/10, hết 24K token) → luôn tắt;
# (2) DeepSeek viết code hay áp nhầm luật (T3 9/12: trừ trả hàng vào doanh thu thường) → không dùng cho code/đáp án;
# (3) Gemini (high) tốt nhất ở việc cần chính xác: danh mục 10/10 không lỗi nhẹ, code 12/12; sinh mẫu dữ liệu trên bộ đồ chơi khó (T4, prompt 8 v5)
#     72/72 trên dev + holdout, khớp cả hai model giải 0,92 (Qwen tắt: 59/72, 0,79);
# (4) API Qwen và Qwen3-14B mục tiêu khác model: chỉ chọn coder bằng replay + oracle + đối chứng, không suy từ cùng họ;
# (5) GPT, Gemini qua proxy bên thứ ba: chỉ gửi nội dung tự sinh. "chua_do" = chưa đo trực tiếp, chọn theo việc gần nhất;
# (6) giải lại để kiểm mẫu: cả lô một lần gọi, chỉ thấy câu hỏi; gpt:off quyết định hợp lệ, model mạnh thứ hai khác họ model soạn (qwen:off khi
#     Gemini soạn, gemini:high khi Qwen soạn) để đo khớp cả hai, deepseek:off là model yếu đánh dấu "khó thật". qwen:off khi giải hay lọc thiếu
#     cách ghi trạng thái và tự bỏ dấu tên cột nếu không được nhắc chép nguyên văn → khớp cả hai dao động 0,53–0,97.
MODEL_ROUTING = [
    {"viec": "1.1 hồ sơ shop, tên nhà cung cấp, ghi chú đơn", "giai_doan": 1, "do_kho": "dễ", "so_luong": "nhiều",
     "model": ["deepseek:off", "qwen:off"], "kiem": "check_profiles", "so_do": "T1: cả 4 model 4/4; deepseek 6 s/lần, rẻ nhất"},
    {"viec": "1.2 danh mục + tham số mùa vụ ngành mới", "giai_doan": 1, "do_kho": "vừa", "so_luong": "ít (10 ngành × 2 bản)",
     "model": ["gemini:high", "qwen:off", "gpt:off"], "kiem": "normalize_catalog + check_catalog; bản 1 Gemini, bản 2 Qwen; tham số số lấy trung vị",
     "so_do": "T2 prompt v2: gemini 10/10 (0 lỗi nhẹ), qwen:off 9/10, gpt:off 8/10, deepseek:off 7/10; deepseek/qwen bật suy nghĩ 0/10"},
    {"viec": "1.3 luật riêng của ngành (§6 rulebook tổng hợp)", "giai_doan": 1, "do_kho": "khó", "so_luong": "rất ít (2–4 luật/ngành)",
     "model": ["gemini:high", "gpt:high"], "kiem": "model kia đọc ngược ra đặc tả (prompt 3); tính được trên lớp sự thật",
     "so_do": "chua_do; luật trong T2 của Gemini không lỗi khuôn"},
    {"viec": "1.4 cách gọi cột/trạng thái/đơn vị mới", "giai_doan": 1, "do_kho": "dễ", "so_luong": "ít",
     "model": ["deepseek:off", "qwen:off"], "kiem": "không trùng tên train; kiểm tay", "so_do": "chua_do; gần T1"},
    {"viec": "1.5 README văn phong chủ tiệm (từ dữ kiện bộ xuất)", "giai_doan": 1, "do_kho": "vừa", "so_luong": "1/shop (~95)",
     "model": ["qwen:off", "deepseek:off"], "kiem": "mọi cột/file xuất có mặt nguyên văn; không bịa cột", "so_do": "chua_do; gần T1/T2"},
    {"viec": "1.6 soát shop mẫu (cổng G7)", "giai_doan": 1, "do_kho": "khó", "so_luong": "10% số shop",
     "model": ["gemini:high", "gpt:high"], "kiem": "người xem lại các lỗi model nêu", "so_do": "chua_do"},
    {"viec": "2.1 mẫu dữ liệu train: câu hỏi + đặc tả + code (số lượng lớn)", "giai_doan": 2, "do_kho": "vừa", "so_luong": "nhiều (~5.000 câu)",
     "model": ["gemini:high", "qwen:off"],
     "kiem": "eval/gen_data.py, prompt 8 v5: code chạy, ra đúng kiểu qua formatter, kiểu hợp cách hỏi (verify.check), không lộ đáp án số, không trùng; "
             "gpt:off giải lại độc lập ra cùng đáp án (score.match); lớp văn phong cục bộ + gpt:off kiểm nghĩa câu viết lại; bỏ trùng giữa lô, câu giống tune/holdout",
     "so_do": "T4 bộ khó, prompt v5, dev + holdout (6 lô × 12): gemini 72/72 hợp lệ, khớp cả hai 0,92, khó thật 0,17, 15 s/mẫu hợp lệ; "
              "qwen:off 59/72, 0,79, 0,18, 22–27 s/mẫu; bản v1 trên bảng tạp hoá chấm lại theo formatter: 13/16 mỗi model, deepseek:off 12/16"},
    {"viec": "2.2 câu khó, nhiều bước, biến thể luật", "giai_doan": 2, "do_kho": "khó", "so_luong": "vừa",
     "model": ["gemini:high", "gpt:high"], "kiem": "như 2.1 + câu đối chứng", "so_do": "T4: gpt 16/16, phủ 9 kỹ năng/lô"},
    {"viec": "2.3 lời giải code (mẫu cho model học)", "giai_doan": 2, "do_kho": "vừa–khó", "so_luong": "nhiều",
     "model": ["qwen:off", "gemini:high", "gpt:off"], "kiem": "chạy code khớp đáp án lớp sự thật và đối chứng; không ưu tiên theo họ model",
     "so_do": "T3: gemini 12/12, gpt 12/12, qwen:off 11/12 (3 s), qwen:high 10/12, deepseek:off 9/12, deepseek:high 6/12"},
    {"viec": "2.4 giải lại độc lập để kiểm mẫu dữ liệu", "giai_doan": 2, "do_kho": "vừa", "so_luong": "1–2 lần/câu",
     "model": ["gpt:off", "qwen:off", "gemini:high"],
     "kiem": "hai model đầu khác họ model soạn: model đầu quyết định hợp lệ, model sau đo khớp cả hai; thêm deepseek:off làm model yếu (khó thật = câu khó "
             "model yếu sai, hai model mạnh khớp); giải cả lô một lần gọi",
     "so_do": "T3: gpt:off 12/12, 5 s/câu; T4: khớp cả hai 0,53–1,0 tuỳ lô (qwen:off giải hay lọc thiếu trạng thái); qwen:off lỗi tên cột 12/12 nếu không nhắc chép nguyên văn"},
    {"viec": "2.5 chẩn đoán lỗi từ log (prompt 5)", "giai_doan": 2, "do_kho": "khó", "so_luong": "vài trăm",
     "model": ["gemini:high", "gpt:high"], "kiem": "code sửa chạy ra đúng đáp án", "so_do": "chua_do; gần T3"},
]
