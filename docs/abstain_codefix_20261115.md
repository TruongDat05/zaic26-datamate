# Thay đổi code giảm abstain — 2026-11-15

Tất cả tên file dưới đây là đường dẫn tương đối từ thư mục gốc dự án; không phụ thuộc vị trí đặt repo.
Tài liệu ghi đúng các thay đổi đã triển khai trong lần sửa này. Commit nền để đối chiếu: `8ee8fce8f32ab6563e6b636e95073043ccba5517`.

> Các số liệu và môi trường ở mục 9 là báo cáo của lần sửa gốc do người cung cấp bản cập nhật ghi nhận. Việc áp dụng bản cập nhật tại repo này được kiểm chứng riêng ở cuối mục 9; repo này không có run nguồn để tái xác nhận số liệu replay trên 1.000 câu.

## 1. Danh sách file sửa và file mới

| File | Thao tác |
|---|---|
| `datamate/voting.py` | Sửa |
| `datamate/pipeline.py` | Sửa |
| `datamate/prompts.py` | Sửa |
| `tests/test_voting.py` | Sửa |
| `analysis/revote_run.py` | Thêm mới |
| `tests/test_pipeline_sampling.py` | Thêm mới |
| `tests/test_revote_run.py` | Thêm mới |
| `docs/abstain_codefix_20261115.md` | Thêm mới: tài liệu này |

Để update code chạy inference, cần cập nhật đủ ba file `datamate/voting.py`, `datamate/pipeline.py`, `datamate/prompts.py` cùng nhau.
Các file test và công cụ replay dùng để kiểm chứng và tái hiện kết quả. Phần diff cuối tài liệu chứa đủ nội dung tất cả 7 file code/test, gồm cả toàn bộ nội dung 3 file mới.

Những sửa đổi có sẵn trước lần này trong `training/train_lora_bf16.py`, `tests/test_train_lora_bf16.py` và các file untracked cũ không nằm trong phạm vi tài liệu/diff này.

## 2. File `datamate/voting.py` — sửa

### Thêm phiên bản chính sách

```python
VOTE_POLICY = "eligible_adaptive_v1"
```

Dùng trong log để biết chính xác chính sách nào tạo quyết định.

### Thêm `empty_answer(ans, change_q=False)`

Tách quy tắc guard cũ thành một hàm dùng chung cho lấy mẫu, bỏ phiếu và quyết định cuối:

- Đánh dấu kết quả số 0, answer None, list rỗng, list toàn số 0, hoặc table không có dòng là kết quả cần bị guard chặn.
- Boolean `False` vẫn hợp lệ.
- `percent = 0` vẫn hợp lệ khi câu hỏi là tăng/giảm phần trăm, giống ngoại lệ đã có trước sửa.
- Đây không phải việc mở guard để chấp nhận tất cả đáp án 0.

### Sửa `vote()`

API mới:

```python
def vote(preds, k, *, base_k=None, change_q=False):
```

- Không cho kết quả mà guard sẽ chặn tranh vị trí đáp án thắng. Nhờ vậy kết quả 0/rỗng lặp lại không che mất một đáp án khác được hỗ trợ và không làm sampling tưởng rằng câu đã đủ tự tin để bỏ qua retry.
- `unanswerable` vẫn là một nhóm phiếu. Cờ `--answer-unanswerable` và ngưỡng vẫn được kiểm tra ở quyết định cuối.
- Giữ quy tắc gộp number/percent cùng giá trị như trước.
- Với mẫu adaptive bổ sung, dùng ngân sách mẫu thường ban đầu làm mức sàn cho mẫu số. Không chỉ chia cho các mẫu chạy được, vì một mẫu thành công giữa nhiều mẫu lỗi không đủ bằng chứng.
- Nếu các nhóm dẫn đầu hòa phiếu, giữ nguyên mẫu số đầy đủ `k`; việc bỏ mẫu lỗi không giúp một trường hợp hòa phiếu mới vượt ngưỡng.
- Khi không có nhóm hợp lệ, trả `abstain` với confidence 0.
- Thêm kiểm tra ngân sách phải dương khi có phiếu.

Công thức chính xác:

```text
N = k = số mẫu trong voting pool.
B = base_k = số mẫu thường ban đầu = 2 * (args.k // 2).
V = tổng số mẫu có đáp án được phép tranh phiếu.
W = số mẫu thuộc nhóm dẫn đầu.

Nếu không có đáp án được phép tranh phiếu: abstain, confidence = 0.
Nếu có base_k và nhóm dẫn đầu duy nhất:  D = max(min(B, N), V).
Nếu không truyền base_k hoặc hòa phiếu: D = N.
confidence = W / D.
```

Ngưỡng trả lời mặc định vẫn **0.5**. Với B=8:

| Bằng chứng | Trước sửa | Sau sửa |
|---|---|---|
| 5 phiếu cùng đáp án + 3 phiếu khác + 4 mẫu lỗi bổ sung | 5/12, abstain | 5/8, đủ ngưỡng |
| 1 đáp án hợp lệ + 11 mẫu lỗi | 1/12 | 1/8, vẫn abstain |
| Hai nhóm mỗi nhóm 4 phiếu + 4 mẫu lỗi | 4/12 | 4/12, vẫn abstain |
| 8 kết quả 0 bị guard chặn + 4 mẫu retry cùng đáp án khác 0 | 0 thắng rồi bị chặn | Đáp án khác 0 có 4/8 phiếu |

Lời gọi cũ `vote(preds, k)` vẫn chạy được; mẫu số vẫn là `k` nhưng có áp dụng quy tắc loại kết quả guard chặn.
Nếu code gốc có lời gọi trực tiếp `vote()` ngoài `Pipeline`, cần truyền `base_k` và `change_q` tương ứng nếu muốn tái hiện chính xác chính sách adaptive mới.

### Sửa `decide()`

Thay điều kiện zero/empty lặp lại bằng lời gọi `empty_answer()`, giữ ngưỡng và ngoại lệ False/zero percent change thống nhất với voting.

## 3. File `datamate/pipeline.py` — sửa

### `_repair()`

- Thêm hàm nội bộ `repair_context()` đọc phần trả lời/repair cuối cùng.
- Nếu thinking có `<think>` nhưng chưa có `</think>`, thay suy nghĩ dang dở trong prompt retry bằng thông báo ngắn. Không gửi lại đoạn suy nghĩ bị cắt rất dài làm đầy context.
- Gửi lỗi riêng `thinking budget exhausted before final TYPE and SQL` để lấy đúng hint repair.
- Retry dùng chế độ không thinking, `temperature=0`, giới hạn `--max-tokens` và tối đa `--repairs` lần như cấu hình sẵn. Không tạo vòng retry vô hạn.
- Sau repair thành công, đặt `errs[i][j] = None`. Các lần thực thi lỗi cũ vẫn giữ trong `executions`, cùng lịch sử model calls.

### `sample()`

- Khi tìm các câu cần thinking, gọi `vote(..., change_q=...)` để dùng cùng guard với quyết định cuối.
- Vì kết quả 0/rỗng bị guard chặn không còn được tính là đồng thuận hợp lệ, một câu có 8 mẫu đều trả 0 vẫn được chọn chạy thinking khi `--think-k > 0` và chưa bị deadline chặn.
- Bỏ điều kiện chỉ repair thinking có `</think>`. Mọi mẫu thinking lỗi, kể cả bị cắt trước SQL, đều được repair có giới hạn.
- `--think-k 0`, `--repairs 0` và logic deadline vẫn được tôn trọng.

### `decide()`

- Tính `change_q` một lần cho mỗi câu.
- Gọi `vote(..., base_k=2 * self.half, change_q=change_q)`.
- Dùng cùng `change_q` cho `final_answer()`.

### `run()` và log

- Import `VOTE_POLICY`.
- Trace mỗi câu thêm `vote_policy`.
- `decision.inputs` trong log thêm `vote_policy`, `base_k`, `change_q`.
- Giữ nguyên đáp án từng mẫu và chỉ số voting pool để có đủ bằng chứng tái tính quyết định.

## 4. File `datamate/prompts.py` — sửa

Thêm nhánh đầu hàm `repair_hint()`:

```python
if err.startswith("thinking budget exhausted"):
    return "Finish with the final TYPE line and one read-only SQL block using the listed schema. Do not start another <think> section.\n"
```

Chỉ thêm hint cho repair; nội dung system prompt `SYSTEM` dùng trong SFT không đổi. Test khớp prompt train/inference đã qua.

## 5. File `tests/test_voting.py` — sửa

Thêm regression tests cho:

- Mẫu retry lỗi không làm loãng một đáp án dẫn đầu duy nhất đã được hỗ trợ.
- Mẫu hợp lệ trái đáp án vẫn được tính và có thể đổi đáp án dẫn đầu.
- Một mẫu hợp lệ giữa nhiều mẫu lỗi không tự đạt confidence 100%.
- Voting pool nhỏ hơn ngân sách ban đầu vẫn giữ kích thước riêng.
- Hòa phiếu không được lợi từ việc giảm mẫu số.
- Kết quả 0/rỗng không che một đáp án khác được hỗ trợ.
- False, zero percent change và cờ unanswerable hoạt động đúng.

## 6. File `tests/test_pipeline_sampling.py` — thêm mới

Tạo test tích hợp dùng DuckDB thật và model stub xác định:

- Thinking bị cắt được retry không thinking và có thể cho đáp án thành công.
- Prompt retry không mang theo khối suy nghĩ dang dở rất dài.
- Đạt giới hạn repair thì dừng.
- 8 mẫu ban đầu đều trả 0 vẫn được thinking; khi 4 mẫu retry thống nhất đáp án khác 0, có thể chọn đáp án đó.
- False và zero percent change hợp lệ không kích hoạt retry sai.
- Lỗi cuối cùng được xóa khi sửa thành công, lịch sử thực thi lỗi vẫn tồn tại.

## 7. File `analysis/revote_run.py` — thêm mới

Công cụ replay chỉ áp dụng voting mới lên các output SQL đã lưu:

```bash
python analysis/revote_run.py runs/public_h100_v2_20261115 \
  --questions data/public_test/questions.jsonl \
  --output runs/public_h100_v2_20261115_codefix_revote
```

Tên thư mục output phải chưa tồn tại; có thể đổi tên theo máy của bạn.

Nội dung được thêm:

- Đọc ngân sách lấy mẫu từ `shard*/run.json`.
- Kiểm tra đủ câu, ID/thứ tự, nội dung câu hỏi, submission, trace, log và voting pool phải khớp nhau.
- Từ chối run thiếu câu, trùng ID, bằng chứng không khớp, hoặc output đã tồn tại.
- Giữ nguyên submission nguồn; ghi checksum và kiểm tra định dạng candidate.
- Ghi `submission.jsonl`, `decisions.jsonl`, `revote_report.json`, `submission_check.log`, `README.md` vào output mới.
- Liên kết từng quyết định tới log nguồn; lưu đáp án, vote và confidence trước/sau.
- Ghi rõ đây là replay không gọi model và không chạy SQL lại. Các fix retry cần inference mới để đo tác động.

## 8. File `tests/test_revote_run.py` — thêm mới

Test replay tạo output riêng đúng định dạng, giữ nguyên file nguồn, ghi đủ audit, từ chối ghi đè và từ chối bằng chứng không nhất quán.

## 9. Kết quả đã kiểm chứng

- **168 tests passed in 12.11s.**
- Kiểm tra whitespace bằng `git diff --check`: đạt.
- Đã tạo patch từ đúng 7 file code/test, áp dụng thử vào bản nguồn trước sửa lấy từ commit nền trong thư mục tạm, rồi so sánh từng byte: cả 7 file khớp hoàn toàn với code hiện tại.
- Các hash source trong báo cáo replay khớp code hiện tại.
- Trên 1.000 câu đã lưu: abstain **125 → 89**, thêm **36 câu có trả lời**, giữ nguyên toàn bộ **875 đáp án đã trả lời trước đó**.
- Submission nguồn và candidate đều có checksum khớp báo cáo; nguồn không bị thay đổi.
- Đây là kết quả offline replay về coverage, chưa chạy lại model sau sửa và chưa chứng minh điểm PA tăng. Không dùng nhãn public để sửa đáp án.
- Bộ lọc guard vẫn có tính bảo thủ: số 0 thật sự hợp lệ ngoài ngoại lệ zero percent change vẫn có thể bị abstain. Lần sửa này không mở guard cho mọi giá trị 0.

Chạy lại bộ test, từ thư mục gốc dự án và môi trường đã có dependencies inference/dev:

```bash
python -m pytest tests/test_voting.py tests/test_pipeline_sampling.py tests/test_revote_run.py tests/test_predict_offline.py tests/test_sqlexec.py tests/test_debt_macro_call.py tests/test_sft_prompt_match.py tests/test_harness_fixes.py tests/test_view_checks.py tests/test_list_order.py tests/test_hints.py tests/test_ingest.py tests/test_submission.py -q
```

Trên instance hiện tại, test dùng Python của `.venv-infer`; pytest được nạp từ `.venv-h100` ở cuối `sys.path`, tắt autoload plugin ngoài để giữ đúng dependency inference.

Các file kết quả nằm trong thư mục tương đối `runs/public_h100_v2_20261115_codefix_revote/`. Chi tiết mọi câu ở `decisions.jsonl`; danh sách 36 câu được thêm đáp án ở trường `changed` của `revote_report.json`.

### Kiểm chứng khi áp dụng tại repo nhận bản cập nhật — 2026-11-16

- Repo bắt đầu ở đúng commit nền, worktree sạch; chỉ cập nhật 8 file liệt kê ở mục 1.
- Cả 7 file code/test khớp từng byte với các SHA256 ở mục 10.
- Chạy đúng bộ test trên: **168 passed in 6.56s**, với Python 3.12.3, DuckDB 2.0.0, pandas 3.0.3 và `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`.
- Môi trường có sẵn `.venv-rt` được dùng để kiểm thử; repo này không có `.venv-infer` hoặc `.venv-h100`. Lượt thử với pandas 2.2.3 gặp lỗi `Cannot set non-string value '0' into a StringArray` ở phần nạp ngày trong `datamate/ingest.py`, file không thay đổi trong bản cập nhật này.
- Patch ở mục 11 được áp dụng thử vào nguồn của commit nền trong thư mục tạm; cả 7 file tái tạo khớp từng byte. `git diff --check` đạt.
- Không có `runs/public_h100_v2_20261115/` hoặc `data/public_test/questions.jsonl` tại repo này, nên chưa tái chạy replay 1.000 câu và chưa xác minh lại số liệu **125 → 89** tại đây. Không chạy model inference.

## 10. SHA256 code sau sửa

Đối chiếu nội dung khi cập nhật ở máy khác:

| File | SHA256 |
|---|---|
| `datamate/voting.py` | `f23827b22b65e66c5dd07834ef4e6eddf158f64c1075a620e5f9aacbdeb8ceac` |
| `datamate/pipeline.py` | `d24cdd4c478de03be9a9354ad994ac9c6043d6e997834d6dfa568086b21e0970` |
| `datamate/prompts.py` | `f6869d9694aba1289eda39d3ceb3f325e1c556bfcd44b88cd9622bc27f3b884b` |
| `tests/test_voting.py` | `aa6665f627df3828363c5197a5a7dca7c7980dff3307f85ebf0f0351b71e0009` |
| `analysis/revote_run.py` | `1eb4356766c255da08c341e0ffb8cc3ba48a09c0875e23768176b0ccd5891937` |
| `tests/test_pipeline_sampling.py` | `1ce5cabd41c63ef19e63250ea879f24c2c3fc8ad4c175429380df62c8b35cbda` |
| `tests/test_revote_run.py` | `b3e8ef0fc3fbae609258776386d8ee1b781944bf674824c818d98b817efa2143` |

## 11. Diff đầy đủ của lần sửa này

Không rút gọn. Hunk `new file mode` chứa toàn bộ nội dung file mới. Chỉ gồm các thay đổi của lần sửa abstain này, không gộp các sửa đổi training có sẵn.

Nếu muốn dùng `git apply`, lưu nguyên nội dung khối diff dưới đây thành một file `.patch`, chạy `git apply --check ten_file.patch` rồi `git apply ten_file.patch` từ thư mục gốc của bản code chưa sửa. Hoặc cập nhật thủ công theo từng file.

```diff
diff --git a/datamate/pipeline.py b/datamate/pipeline.py
index 68a9c3b..eebea71 100644
--- a/datamate/pipeline.py
+++ b/datamate/pipeline.py
@@ -22,7 +22,7 @@ from .ingest import entity_hits, load_business, norm, profile, schema_text
 from .prompts import NOTES, SYSTEM, SYSTEM_ALT, chat, repair_hint
 from .sqlexec import CHANGE, execute, parse
 from .submission import merge_shards, write_submission
-from .voting import debt_pool, final_answer, top_votes, vote
+from .voting import VOTE_POLICY, debt_pool, final_answer, top_votes, vote


 def load_questions(a):
@@ -248,7 +248,16 @@ class Pipeline:
     def _repair(self, shop, todo, chats, texts, preds, errs, pairs, prompt_of):
         """Greedy repair turns for failed samples (i, j): the error plus a hint for its kind, up to --repairs turns."""
         a, eng = self.a, self.engine
-        conv = {(i, j): chats[i][prompt_of(j)] + [{"role": "assistant", "content": texts[i][j].split("</think>")[-1].strip()}]
+
+        def repair_context(text):
+            text = text.split("\n#REPAIR\n")[-1]
+            if "<think>" in text and "</think>" not in text:
+                # There is no final query to fix. Do not feed the unfinished thought back: it can fill the context
+                # or reopen thinking in what should be a bounded, non-thinking SQL retry.
+                return "(Reasoning reached its token budget before producing a final TYPE and SQL query.)"
+            return text.split("</think>")[-1].strip()
+
+        conv = {(i, j): chats[i][prompt_of(j)] + [{"role": "assistant", "content": repair_context(texts[i][j])}]
                 for i, j in pairs}
         turns = 0
         for _ in range(a.repairs):
@@ -256,7 +265,9 @@ class Pipeline:
             if not pairs:
                 break
             for i, j in pairs:
-                e = errs[i][j].split(" | ")[-1]
+                last = texts[i][j].split("\n#REPAIR\n")[-1]
+                e = ("thinking budget exhausted before final TYPE and SQL" if "<think>" in last and "</think>" not in last
+                     else errs[i][j].split(" | ")[-1])
                 conv[i, j].append({"role": "user", "content": f"ERROR: {e}\n{repair_hint(e)}Fix the query. Answer again in the same format."})
             routs = eng.chat([conv[ij] for ij in pairs], eng.params(n=1, temperature=0, max_tokens=a.max_tokens), lora=True)
             for (i, j), o in zip(pairs, routs):
@@ -265,7 +276,7 @@ class Pipeline:
                 texts[i][j] += "\n#REPAIR\n" + out
                 conv[i, j].append({"role": "assistant", "content": out})
                 try:
-                    preds[i][j] = self._exec(shop, i, j, out, todo[i]["question"], todo[i].get("as_of_date"))
+                    preds[i][j], errs[i][j] = self._exec(shop, i, j, out, todo[i]["question"], todo[i].get("as_of_date")), None
                 except Exception as e:
                     errs[i][j] += " | " + str(e)[:400]
             turns += len(pairs)
@@ -321,7 +332,7 @@ class Pipeline:
         behind = is_behind()
         debt_q = [bool(shop.debt) and a.debt_route and bool(DEBT_WORDS.search(norm(q["question"]))) for q in todo]
         if a.think_k and not behind:  # adaptive compute: low-agreement questions, and debt questions (thinking samples are the ones calling the macros), get thinking-mode samples
-            hard = [i for i, ps in enumerate(preds) if texts[i] and (debt_q[i] or vote(ps, len(ps))[1] < a.think_below)]
+            hard = [i for i, ps in enumerate(preds) if texts[i] and (debt_q[i] or vote(ps, len(ps), change_q=bool(CHANGE.search(norm(todo[i]["question"]))))[1] < a.think_below)]
             if hard:
                 # with examples available all thinking samples get them: right 86% vs 71.5% on 873 hard train questions
                 hchats = [chats[i][1 if bank else s % len(systems)] for i in hard for s in range(a.think_k)]
@@ -337,9 +348,9 @@ class Pipeline:
                         preds[i].append(self._exec(shop, i, len(texts[i]) - 1, t, todo[i]["question"], todo[i].get("as_of_date"))); errs[i].append(None)
                     except Exception as e:
                         preds[i].append(None); errs[i].append(str(e)[:400])
-                # failed thinking samples get the same repair turns (non-thinking). A cut-off thought has no query to fix
+                # All failed thinking samples get bounded non-thinking retries, including thoughts cut off before SQL.
                 n_repair += self._repair(shop, todo, chats, texts, preds, errs,
-                                         [(i, j) for i in hard for j in range(2 * half, len(texts[i])) if preds[i][j] is None and "</think>" in texts[i][j]], prompt_idx)
+                                         [(i, j) for i in hard for j in range(2 * half, len(texts[i])) if preds[i][j] is None], prompt_idx)
         pools = [debt_pool(ts, 2 * half, dq) if a.debt_route else list(range(len(ts))) for ts, dq in zip(texts, debt_q)]
         return texts, preds, errs, pools, len(repair), n_repair, behind

@@ -348,8 +359,9 @@ class Pipeline:
         out = []
         for q, ps, pool in zip(todo, preds, pools):
             voters = [ps[j] for j in pool]
-            ans, conf = vote(voters, len(voters))
-            out.append((final_answer(ans, conf, self.a.threshold, q["_oor"], self.a.answer_unanswerable, bool(CHANGE.search(norm(q["question"])))), ans, conf))
+            change_q = bool(CHANGE.search(norm(q["question"])))
+            ans, conf = vote(voters, len(voters), base_k=2 * self.half, change_q=change_q)
+            out.append((final_answer(ans, conf, self.a.threshold, q["_oor"], self.a.answer_unanswerable, change_q), ans, conf))
         return out

     # ---- the run -----------------------------------------------------------------------------------------------------
@@ -383,13 +395,16 @@ class Pipeline:
             for n, (q, (r, ans, conf), ts, ps, es, cs, pool) in enumerate(zip(todo, decisions, texts, preds, errs, chats, pools)):
                 results[q["id"]] = r
                 rec = {"id": q["id"], "business_id": b, "question": q["question"], "final": r, "vote": ans, "conf": conf,
-                       "elapsed_seconds": round(per_q, 2), "samples": [{"text": t, "pred": pr, "error": e} for t, pr, e in zip(ts, ps, es)]}
+                       "elapsed_seconds": round(per_q, 2), "vote_policy": VOTE_POLICY,
+                       "samples": [{"text": t, "pred": pr, "error": e} for t, pr, e in zip(ts, ps, es)]}
                 if a.save_prompts:  # the system message is SYSTEM / SYSTEM_ALT + rulebook (samples 0..K/2-1 / the rest), the user message is shared
                     rec["prompt_user"] = cs[0][1]["content"]
                 tr.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                 log = question_log(q, r, elapsed_ms, self._calls[n], self._runs[n],
                                    {"code": "src/datamate/voting.py::vote, final_answer",
                                     "inputs": {"answers": [ps[j] for j in pool], "samples": pool, "threshold": a.threshold,
+                                               "vote_policy": VOTE_POLICY, "base_k": 2 * self.half,
+                                               "change_q": bool(CHANGE.search(norm(q["question"]))),
                                                "out_of_range": q["_oor"], "answer_unanswerable": a.answer_unanswerable},
                                     "vote": ans, "share": conf}, layers)
                 (self.log_dir / f"{q['id']}.json").write_text(json.dumps(log, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
diff --git a/datamate/prompts.py b/datamate/prompts.py
index 95dd48e..a6f521d 100644
--- a/datamate/prompts.py
+++ b/datamate/prompts.py
@@ -84,6 +84,8 @@ def chat(system, biz_ctx, q, shots=""):
 def repair_hint(err):
     """One generic line for the commonest failure kinds (public trace: 1,933 of 9,912 samples still failed after one repair,
     mostly an empty result or a result of the wrong shape)."""
+    if err.startswith("thinking budget exhausted"):
+        return "Finish with the final TYPE line and one read-only SQL block using the listed schema. Do not start another <think> section.\n"
     if err.startswith(DEBT_LINT):
         # the recipes of the DEBTS block, next to the error: with the old one-line error a solver kept rebuilding the macros from
         # the raw tables over both repair turns (SEARCH: 3 of 63 such samples reached a macro call, the other solver 24 of 24)
diff --git a/datamate/voting.py b/datamate/voting.py
index 73fd190..f93bdb0 100644
--- a/datamate/voting.py
+++ b/datamate/voting.py
@@ -6,6 +6,8 @@ from collections import Counter, defaultdict

 from .sqlexec import calls_debt_macro, parse

+VOTE_POLICY = "eligible_adaptive_v1"
+

 def top_votes(checked):
     """Size of the largest group of drafts with one fingerprint ('none' included). 0 when no draft passed."""
@@ -49,22 +51,49 @@ def debt_pool(texts, start, debt_q=False):
     return list(range(len(texts)))


-def vote(preds, k):
-    """Majority over executed answers. Confidence = share of all K samples agreeing. A number and a percent of the same
-    value are one answer: Qwen3.5 labels a % change either way (public: 38 abstains were such a split), so the group
-    takes its commoner label, percent on a tie, and that label's commonest value."""
+def empty_answer(ans, change_q=False):
+    """Results the final guard rejects. Keep this same rule in sampling, voting and the final decision.
+
+    False is a real boolean answer. A computed zero percent change is also allowed; zero/empty matches otherwise
+    still require another query rather than being accepted just because several samples repeated them.
+    """
+    kind, v = ans["type"], ans.get("answer")
+    if kind == "boolean":
+        return False
+    if change_q and kind == "percent" and isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
+        return False
+    zeros = isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == 0 for x in v)
+    return bool(v in (0, [], None) or zeros or kind == "table" and not v["rows"])
+
+
+def vote(preds, k, *, base_k=None, change_q=False):
+    """Vote over usable answers; guarded zero/empty results cannot hide a supported usable answer or skip retries.
+
+    With adaptive samples, keep at least the original sample budget (`base_k`, capped at this pool's size) in the
+    denominator, and count usable answers beyond it. Failed/guarded extra samples add no evidence against the leader.
+    This never turns one successful retry among many failures into 100% confidence. Tied leaders keep the full K
+    denominator, so discounting failed retries cannot newly accept a tie. Without base_k the denominator stays K.
+
+    A number and a percent of the same value are one answer; the commoner label wins, percent on a label tie.
+    """
     dump = lambda p: json.dumps(p, ensure_ascii=False, default=str, sort_keys=True)
     groups = defaultdict(list)
     for p in preds:
-        if p:
+        if p and p["type"] != "abstain" and (p["type"] == "unanswerable" or not empty_answer(p, change_q)):
             groups[answer_key(p)].append(p)
     if not groups:
         return {"type": "abstain"}, 0.0
+    if k <= 0 or base_k is not None and base_k <= 0:
+        raise ValueError("vote sample budgets must be positive")
     best = max(groups.values(), key=len)  # first seen on a tie, as Counter.most_common
+    sizes = [len(g) for g in groups.values()]
+    denominator = k
+    if base_k is not None and sizes.count(len(best)) == 1:
+        denominator = max(min(base_k, k), sum(sizes))
     types = Counter(p["type"] for p in best)
     kind = "percent" if types["percent"] and types["percent"] >= types["number"] else max(types, key=types.get)
     pick = Counter(dump(p) for p in best if p["type"] == kind).most_common(1)[0][0]
-    return json.loads(pick), len(best) / k
+    return json.loads(pick), len(best) / denominator


 def decide(ans, conf, threshold, change_q=False):
@@ -73,12 +102,7 @@ def decide(ans, conf, threshold, change_q=False):
     Exception (`change_q`, a question asking how much a value rose/fell between periods): a 0 % change is computed, not a sign that
     nothing matched. ROUND(100.0 * (new - old) / old, 2) is 0 only when both periods have the same nonzero total (an empty base
     gives NULL, which execution rejects), so the 0/empty reading does not apply to it."""
-    v = ans.get("answer")
-    zeros = isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) and x == 0 for x in v)  # [0, 0]: train gold has none in 3000
-    nil = v in (0, [], None) or zeros or ans["type"] == "table" and not ans["answer"]["rows"]
-    if change_q and ans["type"] == "percent" and isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
-        nil = False
-    return conf >= threshold and ans["type"] not in ("unanswerable", "abstain") and not (nil and ans["type"] != "boolean")
+    return conf >= threshold and ans["type"] not in ("unanswerable", "abstain") and not empty_answer(ans, change_q)


 def final_answer(ans, conf, threshold, out_of_range=False, answer_unanswerable=False, change_q=False):
diff --git a/tests/test_voting.py b/tests/test_voting.py
index 89295cf..2b30366 100644
--- a/tests/test_voting.py
+++ b/tests/test_voting.py
@@ -23,6 +23,46 @@ def test_vote_with_no_executed_sample_abstains():
     assert vote([None, None], 2) == ({"type": "abstain"}, 0.0)


+def test_failed_adaptive_samples_keep_the_initial_support_budget():
+    plain = [N(7)] * 5 + [N(9)] * 3
+    assert vote(plain, 8, base_k=8) == (N(7), 5 / 8)
+    assert vote(plain + [None] * 4, 12, base_k=8) == (N(7), 5 / 8)
+    assert vote(plain + [N(9), None, None, None], 12, base_k=8) == (N(7), 5 / 9)
+    # Usable contrary evidence still counts: it can replace the initial leader.
+    assert vote(plain + [N(9)] * 4, 12, base_k=8) == (N(9), 7 / 12)
+
+
+def test_a_single_usable_retry_is_not_full_confidence():
+    ans, conf = vote([N(7)] + [None] * 11, 12, base_k=8)
+    assert ans == N(7) and conf == 1 / 8
+    assert final_answer(ans, conf, 0.5) == {"type": "abstain"}
+    # A debt-macro pool smaller than the initial budget keeps its own attempted size.
+    assert vote([N(7), None, None, None], 4, base_k=8) == (N(7), 1 / 4)
+
+
+def test_discarding_failed_retries_does_not_newly_accept_a_tie():
+    ans, conf = vote([N(7)] * 4 + [N(9)] * 4 + [None] * 4, 12, base_k=8)
+    assert conf == 4 / 12 and final_answer(ans, conf, 0.5) == {"type": "abstain"}
+
+
+def test_guarded_results_cannot_skip_retry_or_hide_a_usable_answer():
+    assert vote([N(0)] * 8, 8) == ({"type": "abstain"}, 0.0)
+    assert vote([N(0)] * 8 + [N(7)] * 4, 12, base_k=8) == (N(7), 0.5)
+    empty = [{"type": "list", "answer": []}, {"type": "list", "answer": [0, 0]},
+             {"type": "table", "answer": {"columns": ["name"], "rows": []}}]
+    assert vote(empty + [N(7)], 4) == (N(7), 0.25)
+    assert vote([{"type": "boolean", "answer": False}] * 8, 8) == ({"type": "boolean", "answer": False}, 1.0)
+    assert vote([{"type": "percent", "answer": 0}] * 8, 8, change_q=True) == ({"type": "percent", "answer": 0}, 1.0)
+    assert vote([{"type": "percent", "answer": 0}] * 8, 8) == ({"type": "abstain"}, 0.0)
+
+
+def test_unanswerable_votes_still_need_the_flag_and_support():
+    ans, conf = vote([{"type": "unanswerable"}] * 4 + [None] * 8, 12, base_k=8)
+    assert conf == 0.5
+    assert final_answer(ans, conf, 0.5) == {"type": "abstain"}
+    assert final_answer(ans, conf, 0.5, answer_unanswerable=True) == {"type": "unanswerable"}
+
+
 def test_decide_threshold_zero_and_unanswerable():
     assert decide(N(5), 0.5, 0.5)
     assert not decide(N(5), 0.49, 0.5)
diff --git a/analysis/revote_run.py b/analysis/revote_run.py
new file mode 100644
index 0000000..5de1562
--- /dev/null
+++ b/analysis/revote_run.py
@@ -0,0 +1,129 @@
+"""Re-vote a completed run's saved SQL outputs without generating or executing anything again.
+
+    python analysis/revote_run.py RUN --questions DATA/questions.jsonl --output NEW_DIRECTORY
+
+The source submission, traces and per-question logs must agree and cover every question. The output directory must
+be new. This measures selection/coverage changes only; sampling/repair fixes require a fresh inference run.
+"""
+import argparse
+import contextlib
+import hashlib
+import io
+import json
+import sys
+from collections import Counter
+from datetime import datetime, timezone
+from pathlib import Path
+
+ROOT = Path(__file__).resolve().parents[1]
+sys.path.insert(0, str(ROOT))
+from datamate.ingest import norm  # noqa: E402
+from datamate.sqlexec import CHANGE  # noqa: E402
+from datamate.submission import check, write_submission  # noqa: E402
+from datamate.voting import VOTE_POLICY, empty_answer, final_answer, vote  # noqa: E402
+
+
+def sha(path):
+    return hashlib.sha256(path.read_bytes()).hexdigest()
+
+
+def jsonl(path):
+    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
+
+
+def revote(run, questions, output):
+    run, questions, output = Path(run).resolve(), Path(questions).resolve(), Path(output).resolve()
+    if output.exists():
+        raise FileExistsError(f"Refusing to overwrite {output}")
+    source = run / "submission.jsonl"
+    source_sha = sha(source)
+    qs, old_rows = jsonl(questions), jsonl(source)
+    ids = [q["id"] for q in qs]
+    if len(set(ids)) != len(ids) or ids != [row["id"] for row in old_rows]:
+        raise ValueError("Source submission must match the exact question IDs and order")
+    records = {}
+    for trace in sorted(run.glob("shard*/trace.jsonl")) or [run / "trace.jsonl"]:
+        settings = json.loads((trace.parent / "run.json").read_text())["args"]
+        base_k = 2 * (int(settings["k"]) // 2)
+        if base_k <= 0:
+            raise ValueError("The source run has no initial sampling budget")
+        for rec in jsonl(trace):
+            qid = rec["id"]
+            if qid in records:
+                raise ValueError(f"Duplicate trace ID {qid}")
+            records[qid] = (rec, base_k, trace.parent)
+    if set(records) != set(ids):
+        raise ValueError("Trace coverage must match every question; partial runs cannot be re-voted")
+
+    results, decisions, changed = {}, [], []
+    for q, old in zip(qs, old_rows):
+        qid = q["id"]
+        rec, base_k, shard = records[qid]
+        log_path = run / "logs" / f"{qid}.json"
+        if not log_path.exists():
+            log_path = shard / "logs" / f"{qid}.json"
+        log = json.loads(log_path.read_text(encoding="utf-8"))
+        if old != {"id": qid, **rec["final"]} or log["answer"] != rec["final"]:
+            raise ValueError(f"Submission, trace and question log disagree: {qid}")
+        if any(log[k] != q[k] for k in ("question", "business_id", "as_of_date")):
+            raise ValueError(f"Question contents differ from the source run: {qid}")
+        inp = log["decision"]["inputs"]
+        voters = [rec["samples"][j]["pred"] for j in inp["samples"]]
+        if voters != inp["answers"]:
+            raise ValueError(f"Logged voting pool disagrees with trace samples: {qid}")
+        change_q = bool(CHANGE.search(norm(q["question"])))
+        ans, conf = vote(voters, len(voters), base_k=base_k, change_q=change_q)
+        final = final_answer(ans, conf, inp["threshold"], inp["out_of_range"], inp["answer_unanswerable"], change_q)
+        results[qid] = final
+        decision = {"id": qid, "business_id": q["business_id"], "question": q["question"], "source_log": str(log_path),
+                    "old_final": rec["final"], "final": final, "old_vote": rec["vote"], "vote": ans,
+                    "old_share": rec["conf"], "share": conf, "threshold": inp["threshold"], "base_k": base_k,
+                    "samples": inp["samples"], "attempted": len(voters),
+                    "usable": sum(bool(p) and p["type"] != "abstain" and (p["type"] == "unanswerable" or not empty_answer(p, change_q)) for p in voters),
+                    "vote_policy": VOTE_POLICY}
+        decisions.append(decision)
+        if final != rec["final"]:
+            changed.append(decision)
+
+    if sha(source) != source_sha:
+        raise RuntimeError("Source submission changed during replay")
+    output.mkdir(parents=True)
+    write_submission(output / "submission.jsonl", qs, results)
+    captured = io.StringIO()
+    with contextlib.redirect_stdout(captured):
+        errors = check(output / "submission.jsonl", questions)
+    (output / "submission_check.log").write_text(captured.getvalue() + ("format OK\n" if not errors else str(errors)), encoding="utf-8")
+    if errors:
+        raise ValueError(f"Re-voted submission failed schema checks: {errors}")
+    (output / "decisions.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in decisions), encoding="utf-8")
+    changed_answers = [d["id"] for d in changed if d["old_final"]["type"] != "abstain"]
+    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "mode": "offline_revote_no_new_generation_or_sql",
+              "source_run": str(run), "source_submission_sha256": source_sha, "questions_sha256": sha(questions),
+              "submission": str(output / "submission.jsonl"), "submission_sha256": sha(output / "submission.jsonl"),
+              "questions": len(qs), "vote_policy": VOTE_POLICY,
+              "before_types": dict(Counter(row["type"] for row in old_rows)), "after_types": dict(Counter(r["type"] for r in results.values())),
+              "rescued_abstains": sum(d["old_final"]["type"] == "abstain" and d["final"]["type"] != "abstain" for d in changed),
+              "changed_existing_answers": changed_answers, "changed": changed,
+              "not_replayed": ["retry of truncated thinking", "new thinking samples for guarded zero/empty results"],
+              "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "datamate/voting.py", ROOT / "datamate/pipeline.py", ROOT / "datamate/prompts.py", Path(__file__).resolve()]}}
+    (output / "revote_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
+    (output / "README.md").write_text(
+        "# Offline re-vote\n\nThis candidate reuses the source run's saved executed outputs. No model generation or SQL execution was repeated. "
+        "The original submission is preserved. `decisions.jsonl` links every decision to its original model/SQL log.\n\n"
+        "Coverage changes do not establish answer correctness or competition score. The truncated-thinking and guarded-result retry fixes "
+        "take effect on fresh inference; they cannot be measured by this replay.\n", encoding="utf-8")
+    return report
+
+
+def main():
+    ap = argparse.ArgumentParser(description=__doc__)
+    ap.add_argument("run", type=Path)
+    ap.add_argument("--questions", type=Path, required=True)
+    ap.add_argument("--output", type=Path, required=True)
+    args = ap.parse_args()
+    report = revote(args.run, args.questions, args.output)
+    print(json.dumps({k: v for k, v in report.items() if k not in ("changed", "source_hashes")}, ensure_ascii=False, indent=2))
+
+
+if __name__ == "__main__":
+    main()
diff --git a/tests/test_pipeline_sampling.py b/tests/test_pipeline_sampling.py
new file mode 100644
index 0000000..8894be8
--- /dev/null
+++ b/tests/test_pipeline_sampling.py
@@ -0,0 +1,95 @@
+"""Exercise adaptive retries with real DuckDB execution and a deterministic model stub."""
+from types import SimpleNamespace
+
+import duckdb
+import pytest
+
+from datamate.pipeline import Pipeline, Shop
+from datamate.predict import parse_args
+
+SUM = 'TYPE: number\n```sql\nSELECT SUM(amount) FROM sales\n```'
+MAX = 'TYPE: number\n```sql\nSELECT MAX(amount) FROM sales\n```'
+ZERO = 'TYPE: number\n```sql\nSELECT COUNT(*) FROM sales WHERE amount < 0\n```'
+
+
+class RetryEngine:
+    def __init__(self, initial=(SUM, MAX), thinking=None, repaired=SUM):
+        self.initial, self.thinking, self.repaired = initial, thinking, repaired
+        self.thinking_calls, self.repairs = [], []
+
+    def params(self, **kwargs):
+        return SimpleNamespace(**kwargs)
+
+    def n_tokens(self, chat):
+        return 100
+
+    def chat(self, chats, params, **kwargs):
+        if kwargs.get("thinking"):
+            self.thinking_calls.extend([dict(m) for m in c] for c in chats)
+            texts = [self.thinking] * len(chats)
+        elif chats and chats[0][-1]["content"].startswith("ERROR:"):
+            self.repairs.extend([dict(m) for m in c] for c in chats)
+            texts = [self.repaired] * len(chats)
+        else:
+            texts = list(self.initial)
+        return [SimpleNamespace(outputs=[SimpleNamespace(text=t) for _ in range(params.n)]) for t in texts]
+
+
+def sample(tmp_path, engine, question="Tổng tiền là bao nhiêu?", repairs=1):
+    (tmp_path / "rulebook.md").write_text("Tiền bằng tổng các dòng.", encoding="utf-8")
+    args = parse_args(["--data", str(tmp_path), "--output", str(tmp_path / "out"), "--model", "unused",
+                       "--views", "none", "--k", "8", "--think-k", "4", "--repairs", str(repairs)])
+    pipe = Pipeline(args, engine)
+    db = duckdb.connect()
+    db.execute("CREATE TABLE sales AS SELECT * FROM (VALUES (10), (20)) t(amount)")
+    shop = Shop("shop", db, {}, {})
+    todo = [{"question": question, "as_of_date": "2026-01-01", "_oor": False}]
+    chat = [{"role": "system", "content": "Use DuckDB."}, {"role": "user", "content": question}]
+    try:
+        result = pipe.sample(shop, todo, [[chat, chat]], lambda: False)
+        texts, preds, errs, pools, _, n_repair, _ = result
+        decision, = pipe.decide(todo, preds, pools)
+        return texts[0], preds[0], errs[0], n_repair, decision, pipe
+    finally:
+        db.close()
+
+
+def test_cutoff_thoughts_get_bounded_non_thinking_retries(tmp_path):
+    unfinished = "<think>\n" + "unfinished reasoning " * 10000
+    engine = RetryEngine(thinking=unfinished)
+    texts, preds, errs, n_repair, (final, _, conf), pipe = sample(tmp_path, engine)
+    assert len(engine.thinking_calls) == 4 and len(engine.repairs) == n_repair == 4
+    assert final == {"type": "number", "answer": 30} and conf == 8 / 12
+    assert all(p == final for p in preds[8:]) and errs[8:] == [None] * 4
+    assert all("#REPAIR" in t for t in texts[8:])
+    assert all("unfinished reasoning" not in m["content"] for c in engine.repairs for m in c)
+    assert all("Do not start another <think>" in c[-1]["content"] for c in engine.repairs)
+    assert sum(c["kind"] == "repair" and not c["thinking_mode"] for c in pipe._calls[0]) == 4
+    # The audit keeps the failed execution before the successful retry.
+    assert sum(e["error"] is not None for e in pipe._runs[0]) == 4
+
+
+def test_cutoff_retries_respect_the_repair_limit(tmp_path):
+    engine = RetryEngine(thinking="<think>unfinished", repaired='TYPE: number\n```sql\nSELECT missing FROM sales\n```')
+    _, preds, errs, n_repair, (final, _, _), _ = sample(tmp_path, engine, repairs=2)
+    assert len(engine.repairs) == n_repair == 8
+    assert preds[8:] == [None] * 4 and all("Binder Error" in e for e in errs[8:])
+    assert final == {"type": "abstain"}
+
+
+def test_consensus_on_a_guarded_zero_still_gets_thinking(tmp_path):
+    engine = RetryEngine(initial=(ZERO, ZERO), thinking="<think>Check the filters.</think>\n" + SUM)
+    _, _, _, _, (final, _, conf), _ = sample(tmp_path, engine)
+    assert len(engine.thinking_calls) == 4
+    assert final == {"type": "number", "answer": 30} and conf == 0.5
+
+
+@pytest.mark.parametrize("text,question,answer", [
+    ("TYPE: boolean\n```sql\nSELECT SUM(amount) < 0 FROM sales\n```", "Tổng tiền có âm không?", {"type": "boolean", "answer": False}),
+    ("TYPE: percent\n```sql\nSELECT 100.0 * (SUM(amount) - SUM(amount)) / SUM(amount) FROM sales\n```",
+     "Doanh thu tăng bao nhiêu %?", {"type": "percent", "answer": 0}),
+])
+def test_real_false_and_zero_change_do_not_trigger_a_retry(tmp_path, text, question, answer):
+    engine = RetryEngine(initial=(text, text))
+    _, _, _, _, (final, _, conf), _ = sample(tmp_path, engine, question)
+    assert not engine.thinking_calls and final == answer and conf == 1.0
diff --git a/tests/test_revote_run.py b/tests/test_revote_run.py
new file mode 100644
index 0000000..a8e2fa4
--- /dev/null
+++ b/tests/test_revote_run.py
@@ -0,0 +1,61 @@
+"""A replay candidate must preserve its source, question coverage and per-question voting evidence."""
+import json
+
+import pytest
+
+from analysis.revote_run import revote
+
+
+def make_run(tmp_path):
+    run = tmp_path / "source"
+    shard = run / "shard0"
+    shard.mkdir(parents=True)
+    logs = run / "logs"
+    logs.mkdir()
+    (shard / "run.json").write_text(json.dumps({"args": {"k": "8"}}))
+    qs, rows, trace = [], [], []
+    for i, ps in enumerate([
+        [{"type": "number", "answer": 7}] * 5 + [{"type": "unanswerable"}] * 2 + [None] * 5,
+        [{"type": "number", "answer": 42}] * 8,
+        [None] * 12,
+    ]):
+        q = {"id": f"q_{i}", "business_id": "shop", "question": "Tổng tiền là bao nhiêu?", "as_of_date": "2026-01-01"}
+        vote = ps[0] or {"type": "abstain"}
+        final = vote if i == 1 else {"type": "abstain"}
+        share = [5 / 12, 1.0, 0.0][i]
+        qs.append(q)
+        rows.append({"id": q["id"], **final})
+        trace.append({**q, "final": final, "vote": vote, "conf": share, "samples": [{"pred": p} for p in ps]})
+        log = {**q, "answer": final, "decision": {"inputs": {"answers": ps, "samples": list(range(len(ps))),
+               "threshold": 0.5, "out_of_range": False, "answer_unanswerable": True}}}
+        (logs / f"{q['id']}.json").write_text(json.dumps(log))
+    for path, items in [(run / "submission.jsonl", rows), (shard / "trace.jsonl", trace), (tmp_path / "questions.jsonl", qs)]:
+        path.write_text("".join(json.dumps(r) + "\n" for r in items))
+    return run, tmp_path / "questions.jsonl"
+
+
+def test_revote_writes_a_separate_valid_candidate_and_decision_audit(tmp_path):
+    run, qs = make_run(tmp_path)
+    before = (run / "submission.jsonl").read_bytes()
+    output = tmp_path / "candidate"
+    report = revote(run, qs, output)
+    assert report["before_types"]["abstain"] == 2 and report["after_types"]["abstain"] == 1
+    assert report["rescued_abstains"] == 1 and report["changed_existing_answers"] == []
+    assert (run / "submission.jsonl").read_bytes() == before
+    assert "format OK" in (output / "submission_check.log").read_text()
+    decisions = [json.loads(line) for line in (output / "decisions.jsonl").read_text().splitlines()]
+    assert len(decisions) == 3 and decisions[0]["share"] == 5 / 8
+    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
+        revote(run, qs, output)
+
+
+def test_revote_rejects_inconsistent_source_evidence(tmp_path):
+    run, qs = make_run(tmp_path)
+    p = run / "logs/q_0.json"
+    log = json.loads(p.read_text())
+    log["decision"]["inputs"]["answers"][0] = {"type": "number", "answer": 99}
+    p.write_text(json.dumps(log))
+    output = tmp_path / "candidate"
+    with pytest.raises(ValueError, match="voting pool disagrees"):
+        revote(run, qs, output)
+    assert not output.exists()
```
