# QUY ƯỚC TÍNH (bản diễn đạt của đội, dùng cho dữ liệu giả lập)

T là ngày "hôm nay" của câu hỏi. Khoảng [a, b] tính trọn cả ngày a và ngày b. Câu hỏi nêu rõ cách tính khác thì theo câu hỏi.

## A. Đơn và doanh thu
- Mỗi đơn thuộc đúng một lớp: xong (hoàn thành), huỷ (mọi cách ghi hủy/huỷ, cả "không giao được"), trả (có bất kỳ dòng hoặc phiếu trả nào gắn với đơn, dù chỉ trả một phần, dù phiếu trả ghi ngày sau T), đang giao (tính như đơn xong).
- Đơn hợp lệ = xong hoặc đang giao. Đơn huỷ và đơn trả bị loại NGUYÊN ĐƠN khỏi doanh thu, số đơn, số lượng bán, xếp hạng. Đơn trả không phải đơn huỷ.
- "Trả bằng / trả qua / thanh toán bằng ..." là cách khách thanh toán, không phải trả hàng.
- Sổ không có cột trạng thái thì đọc cột ghi chú; ghi chú không nói về trạng thái nghĩa là đơn xong.
- Doanh thu (cũng là tiền hàng, tiền bán, bán được bao nhiêu tiền) = tổng thành tiền các dòng của đơn hợp lệ, thành tiền đã trừ chiết khấu từng dòng. Không trừ giảm giá/voucher trên cả đơn, không cộng phí giao hàng, không cộng VAT. Dòng phí phục vụ được tính vào doanh thu nhưng không phải là một mặt hàng.
- Đơn thuộc kỳ theo ngày đặt. Bản ghi có ngày sau T coi như chưa xảy ra.
- Số đơn = số mã đơn hợp lệ khác nhau (không đếm dòng). Số đơn huỷ = số mã đơn lớp huỷ.
- Tỷ lệ huỷ = số đơn huỷ chia cho số mọi đơn đặt trong kỳ (mọi lớp: xong, huỷ, trả, đang giao) × 100.
- Giá trị trung bình một đơn = doanh thu / số đơn hợp lệ, làm tròn đến đồng.
- Doanh thu theo kênh bán / hình thức thanh toán lấy theo giá trị ghi trên đơn; "Ghi nợ" và "COD" là hình thức riêng.

## B. Khách
- Đơn khách lẻ (ghi khách lẻ, kl, lẻ, vãng lai, để trống, mã KL) không thuộc khách nào: không đếm, không xếp hạng.
- Sổ ghi mã khách trên đơn: mỗi mã là một khách (hai mã cùng tên vẫn là hai khách). Không có mã: cùng tên chỉ khác hoa/thường, khoảng trắng, dấu câu là một khách.
- Khách đã mua trong kỳ = khách có ít nhất một đơn hợp lệ trong kỳ. Khách mua nhiều nhất xếp theo doanh thu của khách (trừ khi câu hỏi nói theo số đơn).

## C. Số lượng, đơn vị
- Số lượng theo đơn vị cơ bản = số lượng dòng × hệ số của đúng mặt hàng và đơn vị của dòng (bảng quy đổi; nhiều bậc thì nhân dồn; đơn vị không có trong bảng quy đổi có hệ số 1). Không lấy số trong tên hàng.
- Bán chạy / bán nhiều nhất = theo tổng số lượng đơn vị cơ bản của đơn hợp lệ, không theo số dòng hay số tiền.

## D. Kỳ thời gian
- Hôm nay = [T, T]; hôm qua = đúng một ngày T−1; ngày X = [X, X].
- Một tuần chạy từ thứ Hai tới Chủ nhật. "Tuần này" = các ngày của tuần có T, tính tới T. "Tuần trước" = đủ bảy ngày (thứ Hai tới Chủ nhật) của tuần ngay trước đó.
- Tháng M/Y = từ ngày 1 đến ngày cuối tháng. Tháng này = từ ngày 1 đến T. Khi so tháng này với tháng trước: tháng trước chỉ lấy từ ngày 1 đến ngày có cùng số với T (tháng trước ngắn hơn thì đến cuối tháng đó).
- Quý Q = các tháng 3Q−2, 3Q−1, 3Q. Năm Y = cả năm. Từ đầu năm = 1/1 của năm T đến T. N ngày qua / gần nhất = N ngày liền nhau kết thúc ở T.
- Tháng hoặc quý không ghi năm: lấy năm của T nếu kỳ đó bắt đầu không sau T, ngược lại lấy năm trước.
- "Đến hết ngày X", "tính đến hôm nay": tình trạng lúc hết ngày đó, cộng dồn từ đầu dữ liệu.
- Ca theo giờ tạo đơn: ca sáng gồm các giờ 6 tới 13 (6h00 đến trước 14h00), ca chiều gồm các giờ 14 tới 21 (14h00 đến trước 22h00).
- "Trên, hơn, quá, dưới N" không gồm N; "từ N trở lên" gồm N.

## E. Làm tròn, định dạng đáp án
- Tiền và trung bình tiền: làm tròn đến đồng (nửa đồng làm tròn lên). Phần trăm và các trung bình khác: 2 chữ số thập phân.
- Phần trăm thay đổi = (kỳ hỏi − kỳ so) / kỳ so × 100, giảm thì âm. Chênh lệch dạng số = kỳ hỏi − kỳ so.
- Ngày trả về dạng YYYY-MM-DD; hỏi tháng nào thì YYYY-MM. Tên trả đúng như ghi trong dữ liệu.

## F. Xếp hạng và liệt kê
- Bằng nhau ở tiêu chí: mã nhỏ hơn đứng trước (mặt hàng theo mã hàng, khách theo mã khách); không có mã thì theo tên; ngày/tháng: sớm hơn đứng trước.
- Hỏi một đối tượng thì lấy đối tượng đầu theo thứ tự trên; top k là k đối tượng đầu, giảm dần.
- Kết quả dạng bảng: cột thứ nhất ghi tên (mặt hàng, khách, nhóm...), tiếp theo là từng số đo đúng trình tự câu hỏi nêu.

## G. Công nợ
- Chỉ đơn ghi nợ không huỷ mới có thể nợ; đơn trả hàng vẫn còn nợ nếu chưa trả đủ. Đơn COD và đơn bán qua sàn TMĐT không tính là nợ.
- Tổng phải trả của một đơn = tổng thành tiền dòng − giảm giá cả đơn + phí giao.
- Nợ của đơn tại ngày X = tổng phải trả − các khoản đã trả đến hết ngày X (chỉ với đơn đặt đến hết X). Còn nợ = cộng các đơn còn nợ dương.
- Nợ theo khách xác định khách như mục B; đơn khách lẻ vẫn cộng vào tổng nợ nhưng không có dòng khách.
- Tuổi nợ = số ngày từ ngày đặt đến ngày X (đặt hôm nay: 0 ngày).
- Nợ mới trong kỳ = phần chưa trả ngay trong ngày mua của các đơn ghi nợ đặt trong kỳ. Thu nợ trong kỳ = tổng tiền khách trả vào những ngày thuộc kỳ, chỉ tính lần trả diễn ra sau ngày mua (trả ngay trong ngày mua thuộc nợ mới).

## H. Nhập hàng, giá, kho, lãi
- Tiền nhập của kỳ = tổng tiền dòng phiếu nhập có ngày phiếu trong kỳ, bỏ phiếu đã huỷ. Giá nhập bình quân = tổng tiền nhập / tổng số lượng nhập.
- Giá bán của mặt hàng ngày D = đơn giá trên các dòng bán mặt hàng đó của đơn đặt ngày D (mọi lớp đơn), quy về một đơn vị cơ bản, trước chiết khấu dòng.
- Lãi gộp = doanh thu − tổng (số lượng dòng × giá vốn ghi trên dòng bán, giá vốn cho một đơn vị bán).
- Tồn đến hết ngày X (đơn vị cơ bản) = tồn đầu kỳ + nhập (không huỷ) đến X − bán của đơn hợp lệ đến X.
- Mặt hàng ế trong kỳ = mặt hàng đã có trong đơn hợp lệ trước kỳ mà trong kỳ không có đơn hợp lệ nào.
- Lô sắp hết hạn trong N ngày = lô còn tồn có hạn dùng từ T đến T+N. Thuốc kê đơn theo cờ kê đơn trong danh mục.

## I. Không trả lời được
- Khi dữ liệu không ghi loại thông tin câu hỏi cần, hoặc kỳ hỏi nằm hoàn toàn ngoài khoảng ngày có dữ liệu. Kỳ có dữ liệu mà không có dòng khớp thì đáp án là 0 hoặc rỗng.

## J. Cách ghi trong file
- Hai dòng giống hệt nhau của cùng một đơn là lỗi xuất file, chỉ tính một lần.
- Ô trống ở các dòng sau của cùng một đơn lấy giá trị dòng đầu của đơn.
