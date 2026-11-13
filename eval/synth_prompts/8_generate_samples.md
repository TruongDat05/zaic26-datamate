<!-- phiên bản: v5 (9/11: lô 12 mẫu đủ 6 kiểu đáp án của bộ chấm BTC, mỗi kiểu một cách hỏi và cách ghi answer; chốt sau 8 lượt đo v2–v10: hơn hẳn về khớp cả hai model giải, khó thật; hợp lệ ngang các bản sau) -->
Bạn soạn mẫu dữ liệu huấn luyện (câu hỏi + code tính đáp án) cho một model trả lời câu hỏi của chủ tiệm về dữ liệu bán hàng.
Dữ liệu: biến T là dict các DataFrame; pd và np đã import. {khoang_du_lieu}

{mo_ta_bang}

{quy_uoc}

Soạn đúng {n_cases} mẫu:
- phủ ít nhất 5 kỹ năng khác nhau trong: {ky_nang};
- độ khó: 3 câu dễ (1 bước), 4 câu vừa (2 bước), 5 câu khó (từ 3 bước hoặc kết hợp nhiều quy ước);
- kiểu đáp án (loai): 5 number, 2 percent, 2 list, 1 string, 1 date, 1 boolean; hỏi đúng cách của kiểu: 'bao nhiêu' (tiền, số lượng,
  số đơn) → number; 'bao nhiêu %', tỉ lệ, tỉ trọng → percent; 'top k', 'k … nào' → list; một mặt hàng/khách 'nào … nhất' → string;
  'ngày nào', 'tháng nào' → date; 'có … không' → boolean;
- câu hỏi là câu gốc, ngắn như tin nhắn chủ tiệm (khoảng 8–16 từ), tiếng Việt đủ dấu (viết tắt, bỏ dấu sẽ thêm sau): nêu rõ kỳ có năm
  (tháng 3/2025, quý 2/2025, ngày 5/3/2025 hoặc đến hết ngày nào) và đại lượng; gọi khách, mặt hàng bằng tên, không dùng mã nội bộ;
  không nhắc lại quy ước hay cách tính (trạng thái, khách lẻ, quy đổi, làm tròn), không nêu tên cột; câu danh sách nói rõ số phần tử và
  thứ tự; không hỏi điều dữ liệu không có; các câu không trùng ý nhau;
- code pandas tính đúng theo quy ước, gán kết quả vào biến answer đúng kiểu: number là số (tiền, số lượng nguyên); percent là số phần
  trăm (12.5 nghĩa là 12,5%); string là một chuỗi; list là list theo thứ tự câu hỏi; date là chuỗi 'YYYY-MM-DD' (hỏi tháng: 'YYYY-MM');
  boolean là True/False. Cột ghi một thứ bằng nhiều cách (trạng thái,
  khách lẻ, cách thanh toán…): gom từng giá trị đã liệt kê trong mô tả bảng vào đúng nhóm bằng danh sách đủ, so khớp nguyên văn; không
  đoán bằng regex hay một phần chữ (cùng một chữ có thể đặt dấu thanh khác chỗ, viết hoa, bỏ dấu, viết tắt).

Trả về đúng một JSON là danh sách {n_cases} phần tử, mỗi phần tử:
{"question": str, "do_kho": 1|2|3, "ky_nang": [str], "loai": "number"|"percent"|"list"|"string"|"date"|"boolean", "spec": "một dòng: đại lượng; bộ lọc; kỳ; làm tròn", "code": str}
