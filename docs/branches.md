# Làm việc với hai nhánh DataMate

`main` dành cho inference, train LoRA, đánh giá và Docker. `synthetic_data` dành cho sinh và kiểm chứng dữ liệu. Giữ hai lịch sử độc lập; không merge hoặc rebase `synthetic_data` vào `main`.

## Checkout riêng

Nếu đang làm việc trên `main` và có chỉnh sửa chưa commit, tạo worktree cho nhánh sinh dữ liệu để giữ nguyên các tệp đang sửa:

```bash
git fetch origin
git worktree add ../datamate-synthetic synthetic_data
cd ../datamate-synthetic
```

Nếu chưa có nhánh local, dùng `git worktree add -b synthetic_data ../datamate-synthetic origin/synthetic_data`. Một nhánh chỉ được checkout trong một worktree tại một thời điểm; dùng `git worktree list` để tìm checkout đã có.

Mỗi checkout dùng môi trường `.venv` riêng. Cả hai nhánh có package `datamate`; không thêm đồng thời hai checkout vào `PYTHONPATH` hoặc chạy test của hai nhánh trong cùng tiến trình. Các module `eval/sqlv1` dùng `DATAMATE_UPSTREAM` để nạp runtime SQL ở tiến trình riêng.

## Commit và pull request

Tạo nhánh sửa từ `synthetic_data`, rồi mở PR về chính nhánh đó:

```bash
git switch synthetic_data
git pull --ff-only origin synthetic_data
git switch -c fix/synthetic-topic
# Sửa code, cập nhật tài liệu và chạy kiểm thử.
.venv/bin/python -m pytest -q -rs
git add <cac-tep-da-kiem-tra>
git commit -m "Fix synthetic data topic"
git push -u origin fix/synthetic-topic
gh pr create --base synthetic_data --head fix/synthetic-topic
```

Thay đổi inference thực hiện tương tự từ `main` và mở PR về `main`. Không mở PR giữa `main` và `synthetic_data`: hai nhánh có code, package và lịch sử khác nhau.

## Chuyển artifact sang train

Sau khi exporter của controller tạo `train.jsonl`, `dev.jsonl`, `sft_manifest.json`, chuyển các artifact đã kiểm chứng sang đường dẫn dữ liệu cục bộ mà trainer nhánh `main` yêu cầu. Giữ manifest, nguồn gốc runtime, hash và split; không chuyển code bằng merge nhánh. Dữ liệu thi, SFT, khóa API, model và output vẫn nằm ngoài Git.

Controller SFT v2 là phụ thuộc riêng, không có trong `main` hiện tại. `eval/check_setup.py --stage export` liệt kê tệp thiếu và phát hiện exporter không có tùy chọn `--offline-candidates`; lệnh này không tự tải controller hoặc chạy API.
