Rulebook của cửa hàng vừa đổi một dòng luật. Soạn câu hỏi để kiểm tra người trả lời có đọc và áp đúng luật MỚI không.

Dòng luật cũ: {old}
Dòng luật mới: {new}
Ảnh hưởng: {effect}

Cửa hàng: ngành {industry}; có {capabilities}; dữ liệu từ {date_min} đến {date_max}; hôm nay {as_of_date}.
Mặt hàng (một phần): {items}. Nhóm hàng: {groups}.

Viết:
- {n_cham} câu `cham_luat`: đáp án theo luật mới KHÁC đáp án theo luật cũ (chọn kỳ, ngưỡng, mặt hàng sao cho có khác biệt).
- {n_doi_chung} câu `doi_chung`: cùng shop, KHÔNG liên quan dòng luật đã đổi (đáp án như cũ), để người trả lời không áp luật mới tràn lan.
Mỗi câu một cách hiểu, kỳ trong khoảng dữ liệu, không sau hôm nay.

Trả về đúng một mảng JSON: [{"loai": "cham_luat|doi_chung", "cau_hoi": "...", "dac_ta": {...}, "vi_sao_khac": "..."}]
