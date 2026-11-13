Một model đã viết code pandas trả lời sai câu hỏi dưới đây. Tìm nguyên nhân và sửa ÍT NHẤT có thể.

# Câu hỏi
Hôm nay: {as_of_date}. Câu hỏi: {question}

# Dòng rulebook liên quan
{rule_lines}

# Dữ liệu cửa hàng (như model thấy)
{shop_block}

# Code của model
```python
{code}
```
Đáp án model: {answer}
Đáp án đúng: {gold}
Nhóm kết quả: {group}; mã lỗi trong các lượt: {codes}

# Nhãn kỹ năng (chọn 1–3)
{skills}

Trả về đúng một JSON, rồi một khối code đã sửa (chỉ sửa chỗ sai, giữ phần còn lại):
```json
{"k_tags": ["K1"], "nguyen_nhan": "một câu", "dong_code_sai": "...", "dong_luat_bi_vi_pham": "...", "loai": "hieu_sai_cau|sai_luat|sai_schema|sai_format|khac"}
```
```python
# code đã sửa, gán ANSWER bằng hàm dm
```
Nhãn chỉ được nhận khi code đã sửa chạy ra đúng đáp án đúng.
