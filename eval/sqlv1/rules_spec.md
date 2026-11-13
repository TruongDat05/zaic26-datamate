# Đặc tả quy ước tính (`rules_spec`, bản `btc_v1`)

Viết lại bằng lời của đội từ rulebook train + `prompts.SYSTEM`, dưới dạng phép tính. Oracle (`eval/sqlv1/world.py`),
SQL mẫu (`eval/sqlv1/sqlgen.py`) và `rulebook_api.md` đều theo file này. Ký hiệu: `T` = ngày "hôm nay" (`as_of_date`),
`[a, b]` = khoảng ngày kể cả hai đầu, `D(o)` = ngày đặt của đơn `o`.

## R1. Trạng thái đơn
Mỗi đơn có đúng một trong bốn lớp, xác định từ cột trạng thái, cột ghi chú, hoặc phiếu trả gắn mã đơn:

| Lớp | Nhận diện (sau `bỏ dấu + chữ thường`) | Ý nghĩa tính toán |
|---|---|---|
| `done` hoàn thành | còn lại | đơn hợp lệ |
| `cancel` hủy | chứa `huy`, `cancel`, hoặc `khong giao duoc` | không hợp lệ; được đếm là đơn hủy |
| `returned` trả hàng | chứa `tra hang`, `tra lai`, `doi tra`, `hoan tra`, `hang tra ve`, `khach tra`; hoặc có dòng/phiếu trả gắn mã đơn, **kể cả trả một phần**, **kể cả phiếu trả ghi ngày sau T** | không hợp lệ; **không** là đơn hủy |
| `transit` đang giao | `dang giao`, `dang van chuyen`, `cho giao` | **hợp lệ như đơn hoàn thành** |

- `valid(o) = lớp(o) ∈ {done, transit}`. Một dòng của đơn bị hủy/trả làm **cả đơn** không hợp lệ.
- "trả bằng / trả qua / thanh toán bằng" là hình thức thanh toán, không phải trả hàng.
- Shop không có cột trạng thái: đọc cột ghi chú; ghi chú không nói gì về trạng thái → `done`.
- Ô trạng thái/ghi chú/khách để trống ở dòng sau của cùng đơn lấy giá trị dòng đầu đơn.

## R2. Bản ghi sau hôm nay
- Bỏ mọi đơn có `D(o) > T`, mọi khoản trả nợ/phiếu nhập có ngày `> T`.
- Ngoại lệ: lớp hủy/trả của đơn lấy theo ghi nhận trên đơn, kể cả khi phiếu trả có ngày `> T`.
- Kỳ có tên (tháng/quý/năm) phủ quá T trong khi dữ liệu có bản ghi sau T: không sinh câu (mơ hồ).

## R3. Doanh thu, số đơn, khách
- `rev(dòng) = thành tiền dòng` (đã trừ chiết khấu dòng; = SL × đơn giá − chiết khấu dòng nếu không ghi sẵn).
- Dòng **phí phục vụ** là dòng dịch vụ: tính vào doanh thu, không phải mặt hàng.
- Dòng/cột **voucher, giảm giá cả đơn, phí giao hàng, VAT**: không tính vào doanh thu.
- `DT(kỳ) = Σ rev(dòng)` trên các đơn `valid` có `D(o) ∈ kỳ`. "Tiền hàng", "tiền bán", "bán được bao nhiêu tiền" = DT.
- `SoDon(kỳ) = |{mã đơn valid, D(o) ∈ kỳ}|` (đếm theo mã đơn, không theo dòng). Số đơn có mặt hàng X / của khách K cũng chỉ đếm đơn valid.
- `SoDonHuy(kỳ) = |{mã đơn lớp cancel, D(o) ∈ kỳ}|`.
- `TyLeHuy(kỳ) = round2(100 × SoDonHuy / |{mọi mã đơn, D(o) ∈ kỳ, mọi lớp}|)`.
- `GiaTriTB(kỳ) = round0(DT / SoDon)`.
- Doanh thu theo kênh/hình thức thanh toán: theo giá trị ghi trên đơn. "Ghi nợ" và "COD" là hình thức riêng, không thuộc tiền mặt/chuyển khoản.
- **Khách**: đơn khách lẻ (`khách lẻ`, `kl`, `lẻ`, `vãng lai`, ô trống, mã KL) không thuộc khách nào.
  Shop có mã khách trên đơn → khách = mã khách (hai mã trùng tên là hai khách). Không có mã → khách = tên sau khi bỏ khác biệt hoa/thường, khoảng trắng, dấu câu.
- `SoKhach(kỳ)` = số khách có ≥ 1 đơn valid trong kỳ.
- Doanh thu theo nhóm khách: đơn khách lẻ không thuộc nhóm nào; tỉ trọng chia cho DT toàn shop.

## R4. Số lượng và đơn vị
- `SLgoc(dòng) = SL(dòng) × hệ số(mặt hàng, đơn vị của dòng)`; hệ số đọc ở bảng quy đổi/README; đơn vị không có dòng quy đổi là đơn vị cơ bản (hệ số 1). Quy đổi nhiều bậc thì nhân dồn.
- Không lấy số trong tên hàng làm hệ số.
- "Bán được bao nhiêu <đơn vị>": cộng SLgoc của các mặt hàng có đơn vị cơ bản là <đơn vị>.

## R5. Kỳ thời gian (đầu–cuối kể cả hai đầu)
| Cách nói | Khoảng |
|---|---|
| ngày X, hôm nay, hôm qua | `[X, X]`, `[T, T]`, `[T−1, T−1]` |
| tuần này / tuần trước | `[thứ Hai của T, T]` / `[thứ Hai tuần trước, Chủ nhật tuần trước]` |
| tháng M/Y | `[1/M/Y, ngày cuối tháng]` |
| tháng này / tháng trước | `[1/tháng(T), T]` / cả tháng liền trước |
| tháng này **so với** tháng trước | tháng trước lấy `[1, min(ngày(T), ngày cuối)]` |
| quý Q/Y | tháng `3Q−2 .. 3Q` |
| năm Y / từ đầu năm | `[1/1/Y, 31/12/Y]` / `[1/1/năm(T), T]` |
| N ngày qua / gần nhất | `[T−N+1, T]` |
| tháng M, quý Q không ghi năm | năm của T nếu ngày đầu kỳ ≤ T, ngược lại năm trước |
| đến hết ngày X, tính đến hôm nay | trạng thái lúc hết ngày X, cộng mọi phát sinh từ đầu dữ liệu |
| ca sáng / ca chiều | giờ tạo đơn `06:00–13:59` / `14:00–21:59` |
| quá/hơn/trên/dưới N | so sánh nghiêm ngặt; "từ N trở lên" gồm N |

## R6. Làm tròn và định dạng
- Tiền (kể cả trung bình tiền): `round0` nửa lên (half up). Phần trăm và trung bình khác: 2 số lẻ, nửa lên.
- `%thay đổi = round2(100 × (mới − cũ) / cũ)`, âm khi giảm; "chênh lệch / hơn kém bao nhiêu" (số) = kỳ hỏi − kỳ so.
- Ngày: `%Y-%m-%d`; câu hỏi tháng nào: `%Y-%m`. Boolean: true/false. Không đơn vị, không dấu nghìn.
- Tên trả đúng như ghi trong dữ liệu.

## R7. Xếp hạng, đồng hạng, liệt kê
- Bán chạy = theo `Σ SLgoc` (không theo số dòng, số tiền), bỏ đơn không valid, bỏ dòng phí phục vụ.
- Khách mua nhiều tiền nhất = theo DT của khách. Nhân viên giỏi nhất = theo DT.
- Đồng hạng: mã tăng dần (mặt hàng theo mã hàng, khách theo mã khách); không có mã thì theo tên; ngày/tháng: sớm hơn trước.
- Câu hỏi một đối tượng: lấy phần tử đầu theo thứ tự trên. Top k: k phần tử đầu, giảm dần theo số đo.
- Liệt kê có số đo: số đo giảm dần, đồng thì theo mã; liệt kê chỉ lọc: theo mã tăng dần.
- Bảng: cột đầu là tên đối tượng, các cột sau là số đo theo thứ tự câu hỏi nêu; thứ tự các DÒNG của bảng không ảnh hưởng chấm (rulebook BTC §9).
- "top N / N … nào / là gì" hỏi tên → list (một phần tử thì string); bảng chỉ khi câu hỏi đòi số đo kèm từng đối tượng.

## R8. Công nợ
- Đơn có thể nợ: đơn ghi nợ, không hủy (đơn trả hàng **vẫn** tính). Đơn COD và đơn qua sàn TMĐT không bao giờ là nợ.
- `TongDon(o) = Σ thành tiền dòng − giảm giá cả đơn + phí giao`.
- `No(o, X) = TongDon(o) − Σ khoản trả của o có ngày ≤ X`, chỉ với `D(o) ≤ X`. Còn nợ tại X = `Σ No(o, X)` trên các đơn `No > 0`.
- Nợ theo khách: khách như R3; đơn khách lẻ vào **tổng** nợ nhưng không có dòng khách.
- Tuổi nợ = `X − D(o)` (ngày). Đơn đặt hôm nay tuổi 0.
- Nợ thêm trong kỳ = `Σ (TongDon − trả trong ngày mua)` của đơn ghi nợ đặt trong kỳ.
- Thu nợ trong kỳ = `Σ` khoản trả **sau ngày mua** có ngày trả trong kỳ, của mọi đơn ghi nợ không hủy.

## R9. Nhập hàng, giá, kho, lãi
- Tiền nhập kỳ = `Σ` tiền các dòng phiếu nhập có ngày phiếu trong kỳ, bỏ phiếu hủy.
- Giá nhập bình quân = `Σ tiền nhập / Σ SL nhập` (theo số lượng, theo đơn vị câu hỏi nêu, mặc định đơn vị nhập).
- Giá bán X ngày D = đơn giá trên dòng bán X của các đơn đặt ngày D, quy về một đơn vị cơ bản, trước chiết khấu dòng, **kể cả** đơn hủy/trả.
- Lãi gộp = DT − `Σ SL(dòng) × giá vốn ghi trên dòng` (giá vốn cho một đơn vị bán).
- Tồn đến hết ngày X (đơn vị cơ bản) = tồn đầu + `Σ SLgoc nhập ≤ X` − `Σ SLgoc bán` của đơn valid `≤ X`.
- Hàng ế trong kỳ = mặt hàng đã có trong đơn valid **trước** kỳ nhưng không có trong đơn valid nào của kỳ.
- Lô sắp hết hạn trong N ngày = lô còn tồn (> 0) có hạn dùng trong `[T, T+N]`.
- Thuốc kê đơn = mặt hàng có cờ kê đơn trong danh mục; tỉ lệ kê đơn theo DT.

## R10. Không trả lời được
- Dữ liệu không ghi loại thông tin câu hỏi cần (không suy từ cột khác), hoặc kỳ nằm **hoàn toàn** ngoài khoảng ngày dữ liệu.
- Kỳ có một phần trong dữ liệu: tính trên phần có dữ liệu. Kỳ trong dữ liệu mà không có dòng khớp → 0 / rỗng, không phải unanswerable.

## R11. Cách ghi trong file
- Cùng tên chỉ khác hoa/thường, khoảng trắng → cùng một thứ; có mã thì theo mã.
- Hai dòng giống hệt nhau của cùng một đơn = lỗi xuất file, tính một lần; bảng không theo đơn thì tính cả hai.
- Ngày không ghi năm lấy năm của sheet.
