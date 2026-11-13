Bạn soạn câu hỏi dữ liệu bán hàng cho chủ cửa hàng nhỏ ở Việt Nam. Đáp án sẽ do chương trình tính, bạn KHÔNG tính số.

# Cửa hàng
Ngành: {industry}. Kiểu sổ: {archetype}. Dữ liệu từ {date_min} đến {date_max}. Hôm nay (as_of_date): {as_of_date}.
Shop có: {capabilities}
Mặt hàng (một phần): {items}
Nhóm hàng: {groups}

# Câu gốc (seed) và dạng câu
{seed_questions}

# Việc cần làm
Viết {n} câu hỏi MỚI theo phép biến đổi `{operator}`:
- `doi_cach_hoi`: cùng ý câu gốc nhưng cách nói khác (đáp án giữ nguyên).
- `doi_dieu_kien`: đổi kỳ, mặt hàng, nhóm, ngưỡng hoặc top k sang giá trị khác CÓ trong dữ liệu (đáp án đổi).
- `kho_hon`: ghép 2–3 điều kiện (kỳ + nhóm + ca/kênh/khách), so hai kỳ, hỏi xu hướng hoặc top k trong nhóm.
- `ky_hiem`: dùng kỳ ít gặp: tuần này, tuần trước, tháng này so với cùng kỳ tháng trước, quý không ghi năm, N ngày gần nhất, dịp Tết, đến hết ngày X.

Yêu cầu:
1. Mỗi câu có đúng một cách hiểu theo rulebook; không hỏi ý kiến, không hỏi thứ shop không ghi (trừ khi được yêu cầu câu không trả lời được).
2. Chỉ dùng mặt hàng, nhóm có trong danh sách trên; kỳ phải nằm trong khoảng dữ liệu và không sau hôm nay.
3. Viết `cau_chuan` bằng tiếng Việt đầy đủ dấu, rõ ràng, gần cách hỏi của câu gốc (để chương trình nhận dạng được).
4. Ghi `dac_ta` để đối chiếu.

Trả về đúng một mảng JSON:
```json
[{"cau_chuan": "...", "dac_ta": {"so_do": "...", "ky": "...", "mat_hang": null, "nhom": null, "khach": null, "kenh_ca_thanh_toan": null,
  "nguong": null, "top_k": null, "kieu_dap_an": "number|percent|date|string|list|table|boolean|unanswerable"}, "phep_bien_doi": "{operator}"}]
```
