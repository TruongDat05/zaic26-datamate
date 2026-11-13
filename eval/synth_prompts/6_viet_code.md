# Teacher viết code

Dùng NGUYÊN VĂN prompt lúc chạy: `messages = Kit().prompt(q, path="base0")` (system = system.md + rulebook, user = dữ liệu cửa hàng
+ ví dụ + gợi ý luật + câu hỏi). Như vậy code của teacher có cùng phân bố với code model phải viết, và mẫu train dựng lại được.

Teacher trả về đúng một khối ```python```, gán `ANSWER` bằng hàm `dm`, comment ngắn. Không dùng thinking trong đầu ra
(lúc chạy thinking tắt, max_tokens 3072).

## Câu model tự giải sai (gợi ý kiểu STaR)
Chỉ khi teacher hoặc model giải sai nhiều lần, thêm vào cuối tin nhắn user (KHÔNG lưu đoạn này vào mẫu train):

```
Gợi ý kiểm tra (không chép vào code): đáp án đúng có kiểu {gold_type}; dòng luật cần áp: {rule_lines}.
```
Mẫu train lưu prompt gốc (không gợi ý) + code đã chạy ra đúng đáp án.

## Kiểm
`Kit().execute(business_id, code, as_of_date)` → `answer`; nhận khi `Kit().check(answer, gold) == 1` (gold từ oracle hoặc đáp án chuẩn).
