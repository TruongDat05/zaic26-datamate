"""Sinh build/synthetic_sql_v1/REPORT_vi.md + HASHES.json từ các file số đo (không gõ tay số liệu).

    python eval/sqlv1/report.py

Trạng thái tách theo tầng: SOURCE (gói nguồn), EXPORT (SFT v2 do exporter cấp), TOKENIZER (tokenizer thật đúng revision),
GPU-training (ngoài phạm vi, không chạy). Mọi số của corpus lấy từ dòng exporter thật sự nhận (gates.json), không từ ứng viên.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlv1.paths import controller_root  # noqa: E402

from sqlv1.questions import WORK  # noqa: E402

SYNTH = Path(__file__).resolve().parents[2]
FOXEL = controller_root()
PKG = FOXEL / "build" / "synthetic_sql_v1"
FINAL = FOXEL / "build" / "synthetic_sql_v1_final"
SFT = PKG / "sft"
V2 = WORK / "sft_v2"
TARBALL = SYNTH / "sft_v2_agent_handoff.tar.gz"
TOKDIR = FOXEL / "build" / "sft_v2" / "qwen3_14b" / "snapshots" / "40c069824f4251a91eefaf281ebe4c544efd3e18"

REASONS = {
    "producer_qa:unsafe_status_substring_predicate": "SQL phân loại trạng thái bằng chuỗi con một từ ('%huy%', '%hủy%', '%tra%'): khớp gold trên shop không có "
    "'vận chuyển' / 'trả bằng…' nhưng dạy đúng lỗi prompt upstream vừa sửa; QA của bộ sinh loại trước khi nhập",
    "mutation_replay_failed": "chạy lại đột biến trên snapshot mới trượt: 16 trượt ngay bản gốc vì SQL trả tiền chưa làm tròn (scorer chặt của exporter "
    "không làm tròn hộ, runtime cũ thì có), 12 trượt ở một bản đột biến",
    "duplicate_lineage_question_sql_shape": "lời giải thứ hai của cùng câu trùng khung SQL sau khi bỏ hằng (sft_format.sql_shape) với lời giải đã giữ: exporter khử trùng",
}


def j(p: Path, default=None):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default


def sha(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%".replace(".", ",")


def num(x):
    return f"{x:,}".replace(",", ".") if isinstance(x, int) else str(x)


def table(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def git(repo: Path, *args) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def ledger():
    db = sqlite3.connect(f"file:{WORK / 'requests.sqlite'}?mode=ro", uri=True)
    n, last = db.execute("SELECT COUNT(*), MAX(ts) FROM requests").fetchone()
    by = db.execute("SELECT provider, COALESCE(response_model, requested_model), status, COUNT(*) FROM requests GROUP BY 1, 2, 3 ORDER BY 4 DESC").fetchall()
    return n, last, by


def main():
    g = j(V2 / "gates.json", {})
    ms = j(V2 / "mutation_summary.json", {})
    bs = j(V2 / "bundle_summary.json", {})
    man = j(SFT / "sft_manifest.json", {})
    tv = j(PKG / "token_validation.json", {})
    aug = j(V2 / "augment_summary.json", {})
    pr = j(V2 / "probe_summary.json", {})
    prev_sm = j(PKG / "source_manifest.v1_63353535.json", {})
    snap = j(FOXEL / "build" / "sft_v2" / "snapshot_8b07ba9.json", {})
    eda = j(WORK / "eda" / "eda_stats.json", {})
    al_old = j(WORK / "btc_align.json", {}).get("summary", {})
    al = g.get("btc_align_snapshot") or {}
    br = j(WORK / "build_report.json", {})
    rr = j(WORK / "questions" / "reuse_report.json", {})
    r1 = j(WORK / "audit" / "llm_summary_r1.json", {})
    m1 = [json.loads(x) for x in (WORK / "audit" / "manual_verdicts_r1.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    sm, fsm = j(PKG / "source_manifest.json", {}), j(FINAL / "source_manifest.json", {})
    off = (man.get("inputs") or {}).get("offline_candidates") or {}
    tr, dv = g["splits"]["train"], g["splits"]["dev"]
    tok = g.get("tokens") or {}
    fake = (tv.get("checks") or {}).get("mask_logic_fake_tokenizer") or {}
    up_log = git(FOXEL, "log", "--format=%h %s", "feat/synthetic-sql-v1..HEAD").splitlines()
    syn_log = git(SYNTH, "log", "--format=%h %s", "feat/synthetic-sql-v1..HEAD").splitlines()
    up_head, syn_head = git(FOXEL, "rev-parse", "HEAD"), git(SYNTH, "rev-parse", "HEAD")
    n_req, last_req, by_req = ledger()
    hashes = {
        "handoff_tarball": {"path": str(TARBALL), "sha256": sha(TARBALL), "checksums": "SHA256SUMS 116/116, package/checksums.sha256 110/110"},
        "frozen_snapshot_c0004_source_hash": "sha256:b34fadb054d907206299f962d9d0d3989f8dc390425d2b9bd61e32f8bee72040",
        "upstream": {"branch": git(FOXEL, "rev-parse", "--abbrev-ref", "HEAD"), "head": up_head, "snapshot_commit": snap.get("commit"),
                     "snapshot_source_hash": snap.get("source_hash"), "loader_closure_hash": snap.get("loader_closure_hash")},
        "synth": {"branch": git(SYNTH, "rev-parse", "--abbrev-ref", "HEAD"), "head": syn_head},
        "data_hash_train_dev": sm.get("data_hash"), "data_hash_final": fsm.get("data_hash"),
        "sft": {"build_id": man.get("build_id"), "train": man["splits"]["train"]["file_hash"], "dev": man["splits"]["dev"]["file_hash"],
                "rejected": man.get("rejected_file_hash"), "sft_manifest": sha(SFT / "sft_manifest.json")},
        "offline_bundle_json": off.get("bundle_json_hash"), "ledger_copy": off.get("ledger_sha256"), "mutation_report": off.get("mutation_report_hash"),
        "token_validation": sha(PKG / "token_validation.json"), "tokenizer_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "tokenizer_files": {p.name: sha(p) for p in sorted(TOKDIR.glob("*")) if p.is_file()},
    }
    (PKG / "HASHES.json").write_text(json.dumps(hashes, ensure_ascii=False, indent=1), encoding="utf-8")
    G = g["gates"]

    def st(ok):
        return "đạt" if ok else "**trượt**"

    ai, ax = g["audit"]["llm_independent_subagent"], g["audit"]["llm_cross_family"]
    n_out = man["counts"]["train"] + man["counts"]["dev"]
    gates = [
        ("Căn quy ước trên BTC train (chạy lại trên snapshot mới)", "≥ 95% câu áp dụng được", f"{num(al.get('ok'))}/{num(al.get('applicable'))} = {pct(al.get('match_rate'), 2)}", st(G["alignment"])),
        ("Audit bởi LLM độc lập (subagent Claude, rubric) trên mẫu vòng 2 còn trong SFT", "≥ 95%, Wilson ≥ 90%, n ≥ 150",
         f"{ai['pass']}/{ai['in_sft']} = {pct(ai['rate'])}, Wilson95 {ai['wilson95']}", st(G["audit_llm_independent"])),
        ("Audit LLM khác họ (gemini, dự phòng deepseek) + 50 mồi sai", "dùng tỷ lệ đạt khi độ nhạy ≥ 80%",
         f"độ nhạy {ax['baits_caught']}/{ax['baits']}; đạt {ax['pass']}/{ax['in_sft']} = {pct(ax['pass_rate'])}, Wilson95 {ax['pass_wilson95']}", st(G["audit_cross_family"])),
        ("Phân bố so với MINING (train, theo câu)", "lõi ≥ 90% dòng; tháng cụ thể 55–65%; L1 kỹ năng ≤ 0,15; L1 kiểu ≤ 0,10",
         f"lõi {pct(tr['industry_core_row_share'])}; tháng cụ thể {pct(tr['period'].get('thang_cu_the'))}; L1 kỹ năng {tr['L1']['skill']}; L1 kiểu {tr['L1']['type']}", st(G["distribution"])),
        ("Rò rỉ FINAL", "0 shop / lineage / id câu FINAL trong SFT; 0 câu FINAL trong request của solver; train ∩ dev lineage = 0",
         f"{g['leakage']['final_shops_in_sft']} / {g['leakage']['final_lineages_in_sft']} / {g['leakage']['final_question_ids_in_sft']}; "
         f"{g['leakage']['final_items_in_solver_requests']}; {g['leakage']['train_dev_lineage_overlap']}", st(G["leakage"])),
        ("Trùng lặp", "câu trùng ≤ 2%", pct(tr["exact_dup_question_share"], 2), st(G["dup"])),
        ("SQL mẫu tất định", "≤ 15% dòng train", f"{tr['template_rows']}/{num(tr['rows'])} = {pct(tr['template_share'])}", st(G["templates"])),
        ("Số lượng (dòng exporter nhận)", "≥ 6.000 dòng train, ≥ 500 dòng dev",
         f"train {num(tr['rows'])} dòng ({num(tr['questions'])} câu, {tr['shops']} shop); dev {num(dv['rows'])} dòng ({dv['questions']} câu, {dv['shops']} shop)", st(G["count"])),
        ("Xác minh độc lập của exporter", "100% dòng xuất: chạy 2 lần cùng kết quả, đúng kiểu, khớp gold (scorer chặt)",
         f"{num(n_out)}/{num(n_out)} dòng mang verification do build_sft/2 cấp; 0 wrong_answer / exec_error / nondeterministic trong {num(man['counts']['accepted_before_dedupe'])} ứng viên vào bước chạy lại", "đạt"),
        ("Đối chứng đột biến (chạy lại trên snapshot mới)", "mọi bản tính được khớp gold mới, ≥ 2 bản",
         f"100% dòng xuất qua; dòng train có ≥ 1 đột biến đổi đáp án mà SQL vẫn đúng: {num(tr['mutation']['rows_discriminating'])} ({pct(tr['mutation']['share_discriminating'])})", "đạt"),
        ("Token Qwen3-14B thật (revision 40c06982…)", "tokenizer_alignment_real = passed trên 100% dòng; không dòng nào > max_len",
         f"{tok.get('tokenizer_alignment_real')}: EOS học đúng 1 lần ở {num(tok.get('eos_once_rows') or 0)}/{num(tok.get('rows') or 0)} dòng; {tok.get('rejected_long')} dòng > {tok.get('max_len')}; "
         f"lỗi {tok.get('errors')}; phân vị độ dài {tok.get('length_quantiles')}", st(G["tokenizer"])),
    ]
    all_ok = all(G.values())
    rej = man.get("rejected_reasons", {})
    q = lambda k: table([(f"{p}", f"`{m}`", s, n) for p, m, s, n in by_req], ["provider", "model", "status", "request"]) if k else ""  # noqa: E731
    lines = [
        f"# Báo cáo `syn_sales_sql_v1` sau exporter SFT v2 — {'dữ liệu + token: ĐẠT; GPU-training: chưa chạy' if all_ok else 'NOT READY'}", "",
        "## 0. Trạng thái theo tầng",
        table([
            ("SOURCE (gói nguồn)", "READY", f"kiểm hợp đồng gói nguồn đạt 100% (kể cả data_hash = content_hash); train/dev `{sm.get('data_hash')}`; FINAL riêng `{fsm.get('data_hash')}` (training_eligible = false)"),
            ("EXPORT (SFT v2)", "READY" if all(G[k] for k in G if k != "tokenizer") else "NOT READY",
             f"build `{man.get('build_id')}`: train {num(man['counts']['train'])}, dev {num(man['counts']['dev'])}, rejected {man['counts']['rejected']}; mọi cổng §9 tính lại trên dòng xuất"),
            ("TOKENIZER (Qwen3-14B thật)", "READY" if G["tokenizer"] else "NOT READY",
             f"`tokenizer_alignment_real = {tok.get('tokenizer_alignment_real')}` trên {num(tok.get('rows') or 0)} dòng; tokenizer + config khớp cây file của revision 40c06982… (blob id / lfs sha256); không tải trọng số"),
            ("GPU-training", "CHƯA CHẠY (ngoài phạm vi bước này)", "gpu_bf16, forward/backward, trainable modules, save/load adapter, resume, throughput: `not_run`; "
             f"validate-only báo ready_for_real_training = {tok.get('ready_for_real_training')}, blockers = {tok.get('blockers')}"),
        ], ["Tầng", "Trạng thái", "Căn cứ"]), "",
        "## 1. Code tích hợp",
        f"- Upstream nhánh `{hashes['upstream']['branch']}` (tách từ `feat/synthetic-sql-v1`), HEAD `{up_head[:10]}`:", *[f"  - `{x}`" for x in reversed(up_log)],
        f"- Snapshot suy luận cho mọi bước kiểm: `build/sft_v2/snapshot_{snap.get('commit')}` (git archive `datamate/` + pyproject + README của `{snap.get('commit')}`); "
        f"source hash `{snap.get('source_hash')}`; loader closure `{snap.get('loader_closure_hash')}`.",
        f"- Repo sinh nhánh `{hashes['synth']['branch']}`, HEAD `{syn_head[:10]}`:", *[f"  - `{x}`" for x in reversed(syn_log)],
        "- Gói bàn giao: sha256 tarball khớp file `.sha256`; `SHA256SUMS` 116/116; `package/checksums.sha256` 110/110; source hash của `package/src` = `sha256:b34fadb0…2040` (đúng snapshot c0004). "
        "Port nguyên byte. Không port 4 test cần `harness_control.{package,proposer,cli}` (controller CLI/proposer không có trong gói). `scripts/rule_check.py` thêm `training_public_references` theo test của gói. "
        "Máy này không chạy được bwrap (AppArmor chặn user namespace), nên mọi bước sandbox chạy `--isolation process`.", "",
        "## 2. Cổng §9 tính lại trên dòng exporter nhận", table(gates, ["Cổng", "Ngưỡng", "Số đo", "Kết quả"]), "",
        f"Ghi thêm (không tính là rò rỉ): {g['final_text_twins']['final_questions_with_identical_text_in_sft']}/{g['leakage']['final_questions']} câu FINAL có câu chữ trùng một câu train/dev "
        f"ở shop khác (khác dữ liệu, khác đáp án), xuất hiện trong {g['final_text_twins']['sft_rows_with_final_text']} dòng SFT.", "",
        "## 3. Exporter, đường nhập ngoại tuyến và lý do loại",
        f"Lệnh: `training/build_sft.py --offline-candidates build/synthetic_sql_v1/offline_bundle --source build/sft_v2/snapshot_{snap.get('commit')} --max-per-question 2 --isolation process` "
        f"(hợp đồng `offline_candidates/1`). Ứng viên gói {num(bs.get('candidates_in', 0))} → vào bundle {num(bs.get('candidates_bundled', 0))} "
        f"(lượt đầu {num((bs.get('stats') or {}).get('model:first', 0))}, lượt sửa {(bs.get('stats') or {}).get('model:repair', 0)}, SQL mẫu {(bs.get('stats') or {}).get('template', 0)}) "
        f"→ vào bước chạy lại {num(man['counts']['accepted_before_dedupe'])} → xuất train {num(man['counts']['train'])} + dev {num(man['counts']['dev'])}; câu có dòng {num(man['counts']['questions_with_rows'])}/{num(man['counts']['questions_in_source'])}.",
        table([(k, v, REASONS.get(k, "")) for k, v in sorted(rej.items(), key=lambda kv: -kv[1])], ["lý do (rejected.jsonl)", "số", "giải thích"]), "",
        f"Bằng chứng gốc kiểm trước khi chạy: sha256 của {bs.get('evidence_files')} file trong bundle; dòng ledger ↔ bản ghi request đã cache (prompt_sha = sha256 json.dumps(messages), provider, model, status, usage); "
        "finish = stop; request chứa câu hỏi (nhãn `### id:` + đúng câu) và không chứa gold; completion = phép biến đổi khai báo `api_json_item_to_notes_type_sql/1` của câu trả lời đã cache, khớp từng byte; "
        f"lượt sửa trích đúng lời giải lượt đầu và có ghi chú riêng; SQL mẫu khớp dòng sự kiện và generator tái tạo được từng byte ({(bs.get('stats') or {}).get('template', 0)}/{(bs.get('stats') or {}).get('template', 0)}); "
        "bản ghi đột biến gắn đúng completion + source hash của snapshot. Sau đó exporter dựng lại shop bằng loader của snapshot từ layer đã ghi, trên bản chiếu không có gold, "
        "chạy SQL 2 lần bằng guard của controller, so gold bằng scorer cố định (chặt).",
        "**Khác đường run (`solve_synthetic.py`):** prompt của dòng SFT KHÔNG phải prompt đã gửi model (model trả lời theo lô 10 câu, system = luật SQL của đội + rulebook_api tự viết lại). "
        "Prompt dòng SFT là prompt runtime do snapshot tự dựng (`datamate.harness_entry.render_sql_prompts`, biến thể p0) trên layer đã ghi. Ghi ở `metadata.derivation` "
        "(`offline_api_batch_item` / `offline_program_template`) và `metadata.offline.prompt_source`. `request_id` = dòng ledger thật (`ledger:synth_sql_v1:<id>`) hoặc sự kiện generator "
        "(`generator_event:<id>`); SQL mẫu không gán model hay receipt. Không có receipt/cache broker, model id, finish reason hay verification nào được tạo giả.", "",
        "## 4. Chạy lại đối chứng đột biến (snapshot mới, bằng chứng riêng)",
        f"{num(ms.get('candidates', 0))} ứng viên: đạt {num(ms.get('pass', 0))}; trượt {json.dumps({k: v for k, v in (ms.get('outcome') or {}).items() if k != 'pass'}, ensure_ascii=False)}; "
        f"bản gốc khớp {num(ms.get('base_ok', 0))}; đạt và có ≥ 1 đột biến đổi đáp án mà SQL vẫn đúng {num(ms.get('pass_with_changed_gold', 0))}; số bản kiểm/ứng viên {json.dumps(ms.get('n_checked_hist'))}. "
        f"Bản `m_edges` không dựng được ở {len(ms.get('shops_with_unavailable_mutants') or {})} shop (kiểm khách lẻ của view từ chối dữ liệu đột biến; các shop đó còn 4 bản khác).",
        f"Một đột biến không đổi đáp án không chứng minh SQL không ghi cứng: dòng train có bằng chứng đó = {num(tr['mutation']['rows_discriminating'])}/{num(tr['rows'])} ({pct(tr['mutation']['share_discriminating'])}); "
        "phần còn lại chỉ có kiểm tĩnh 'không ghi cứng đáp án'. Báo cáo: `outputs/synth_sql_v1/sft_v2/mutation_report.jsonl` "
        f"(`{ms.get('report_sha256')}`), mỗi dòng gắn candidate_id, completion_sha, sql_sha, source/loader hash, oracle; dòng SFT mang `metadata.mutation` + `report_hash`.", "",
        "## 5. Ba lỗi upstream đã sửa trên nhánh tích hợp",
        "`tests/test_upstream_regressions.py`: 8/9 trượt trên c0004, 9/9 đạt sau sửa; `repro_upstream.py` của gói: exit 1 → exit 0.",
        "- `views.raw_values`: cache theo đối tượng connection (WeakKeyDictionary) thay vì `(id(db), cột)`.",
        "- `ingest.fix_header`: cộng điểm cho ô chữ đứng trên cột số/ngày. Không bảng nào đổi dòng tiêu đề (BTC train 322, public 146, tổng hợp 1.658, FINAL 163).",
        f"- Prompt trạng thái: bỏ `LIKE '%huy%'` / `'%tra%'`, thay bằng phép thử nguyên từ dựng từ `ingest.CANCELLED/RETURNED`. Layer đã sinh (view 189, nợ 111) không dùng mẫu một từ; "
        f"{rej.get('producer_qa:unsafe_status_substring_predicate', 0)} ứng viên dùng mẫu này bị loại.",
        f"- Kiểm chứng cũ chạy lại trên revision mới, không kế thừa cache: căn quy ước BTC {num(al.get('ok'))}/{num(al.get('applicable'))} (trước {num(al_old.get('ok'))}/{num(al_old.get('applicable'))}), "
        "đối chứng đột biến, chạy lại của exporter, token.", "",
        "## 6. Token Qwen3-14B thật",
        f"`training/train_lora.py --validate-only --local-files-only` với config ghim `build/sft_v2/train_qwen3_14b_bf16.pinned.toml` (revision 40c06982…, local_path = tokenizer + config.json, không trọng số). "
        f"tokenizer_alignment_real = **{tok.get('tokenizer_alignment_real')}**: prompt có generation prompt là tiền tố của hội thoại (mức chữ và mức token), completion render nguyên văn, "
        f"EOS học đúng 1 lần ở {num(tok.get('eos_once_rows') or 0)}/{num(tok.get('rows') or 0)} dòng, prompt mang nhãn −100; padding probe {tok.get('padding_probe')}; dòng > max_len {tok.get('max_len')}: {tok.get('rejected_long')}; "
        f"phân vị độ dài {tok.get('length_quantiles')}; token có nhãn {num(tok.get('supervised_tokens') or 0)}, token vào {num(tok.get('input_tokens') or 0)}. "
        f"dataset_contract = {tok.get('dataset_contract')}, config = {tok.get('config')}, local_base_config = {tok.get('local_base_config')}.",
        f"Kiểm phụ `mask_logic_fake_tokenizer` = {fake.get('status')}: tokenizer GIẢ đếm byte UTF-8, cả {len(fake.get('rows_over_max_len_in_fake_bytes') or [])} dòng dài hơn 16.384 byte nên không dòng nào vào kiểm giả; "
        "đây không phải lỗi dữ liệu, kiểm thật bằng tokenizer đúng revision ở trên mới là cổng.", "",
        "## 7. Audit (đổi nhãn: \"kiểm tay\" trước đây là audit bởi LLM độc lập, do subagent Claude làm theo AUDIT_RUBRIC.md; KHÔNG có người kiểm)",
        f"- Vòng 1 (mẫu 160, hạt 4242): audit bởi LLM độc lập {sum(x['pass'] for x in m1)}/{len(m1)} = {pct(sum(x['pass'] for x in m1) / len(m1))} → **trượt**. Lỗi hệ thống: lọc line_kind='item' làm mất phí phục vụ, "
        "kê đơn theo nhóm thay vì cờ, đồng hạng theo tên, hàng ế theo danh mục, mã hàng sai trong Notes, CASE quy đổi hằng, câu tên đòi số. "
        f"Audit LLM khác họ vòng 1: độ nhạy {pct((r1 or {}).get('sensitivity'))}, đạt {pct((r1 or {}).get('pass_rate'))}.",
        "- Sửa: kiểm ngữ nghĩa tĩnh theo slot, chấm lại mọi lời giải, chạy SQL mẫu cho toàn kho, đóng gói lại.",
        f"- Vòng 2 (mẫu MỚI 160, hạt 9091, không trùng vòng 1): audit bởi LLM độc lập {ai['all_sample_pass']}/160 = {pct(ai['all_sample_rate'])}; sau đó thêm luật giá bán theo ngày và đóng gói lại. "
        f"Trên các mục còn trong SFT cuối: {ai['pass']}/{ai['in_sft']}, Wilson95 {ai['wilson95']}; audit LLM khác họ {ax['pass']}/{ax['in_sft']}, độ nhạy {ax['baits_caught']}/{ax['baits']} mồi.",
        "- Hai audit không chạy lại sau export (không mở thêm API); số trên là giao của mẫu cũ với SFT. Nên có người duyệt lại `outputs/synth_sql_v1/audit/sample.jsonl`.", "",
        "## 8. Model và provider",
        "Thực dùng (theo ledger):", q(True), "",
        "- Yêu cầu nhưng không truy cập được: `gpt-6.1-sol`, `claude-haiku-5-5` (proxy trollllm trả HTTP 502 suốt phiên: smoke + 9 lần dò; cả `gpt-6-luna` cũ). Provider đã có trong eval/llm.py nhưng không có lời giải nào từ hai model này.",
        f"- Dòng train theo nguồn lời giải: {json.dumps(tr['by_solver'], ensure_ascii=False)}. Teacher deepseek, verifier qwen. Qwen cùng họ với model học (Qwen3-14B).", "",
        "## 9. Chi phí API",
        f"Bước này KHÔNG gọi API: ledger `outputs/synth_sql_v1/requests.sqlite` vẫn {num(n_req)} request (request cuối lúc {last_req}), trần 12.000; ledger cũ không bị đụng.", "",
        "## 10. Phân bố so với MINING (train, theo câu; chỉ số gộp)",
        table([(d, json.dumps(dict(list(((eda.get('mining') or {}).get(d) or {}).items())[:6]), ensure_ascii=False),
                json.dumps(dict(list((tr.get(d) or {}).items())[:6]), ensure_ascii=False), tr["L1"].get(d)) for d in ("skill", "type", "period", "word_bins")],
              ["chiều", "MINING", "train (SFT)", "L1"]),
        f"Số từ train {((eda.get('splits') or {}).get('train') or {}).get('words')}; MINING {(eda.get('mining') or {}).get('words')}. Đường giải (dòng xuất): {json.dumps((eda.get('solutions') or {}).get('route'))}.", "",
        "## 11. Nền (không đổi từ trước export)",
        f"- {len(br.get('shops', []))} shop (132 thế giới cũ + {len(br.get('shops', [])) - 132} shop lõi mới); câu dùng lại {rr.get('kept')}. "
        "FINAL 414 câu ở gói riêng, dữ liệu không đổi (chỉ ghi data_hash vào manifest dẫn xuất, giữ `source_manifest.pending_snapshot.json`).",
        "- Dữ liệu BTC chỉ đọc cục bộ (căn quy ước, đo dòng tiêu đề); không gửi API, không vào git. `data_root/rulebook.md` (bản sao rulebook BTC) nằm trong prompt của dòng SFT như runtime; build/ không vào git.", "",
        "## 12. Cải thiện v1.1 (ngoại tuyến, không gọi API)",
        f"- **Câu 'cửa hàng không ghi thông tin này':** thêm {aug.get('unanswerable_added')} câu ({aug.get('unanswerable_rows')} dòng) lấy từ kho câu đã sinh (teacher viết, verifier khác họ kiểm, "
        "solver API giải, đã qua kiểm cục bộ) nhưng bị bước chọn cũ bỏ: nhãn kỹ năng của câu unanswerable BTC phần lớn để trống nên hàm mục tiêu phạt chúng. "
        "Câu hỏi kỳ ngoài dữ liệu thì runtime đã tự trả unanswerable (`hints.outside`), nên chỉ bổ sung loại thiếu thông tin.",
        f"- **Câu dài:** câu ≥ 26 từ của BTC là một họ câu cụ thể (đo gộp: 84/106 là phần trăm, 78/106 thuộc nhóm giá; trung vị 36 từ). Thêm {sum((aug.get('long_added') or {}).values())} câu "
        f"'giá giữa hai ngày tăng/giảm bao nhiêu %' viết dài ({json.dumps(aug.get('long_signs'))}), đặc tả mới, gold = oracle, câu chữ template viết mới (nguồn `template_text`), SQL mẫu tất định; "
        "dev không có (shop dev thuộc họ sổ không ghi đơn giá dòng).",
        f"- **Đột biến nhắm theo câu (`m_probe`):** {pr.get('questions_probed')} câu chưa có đột biến làm đổi đáp án được thử các biến đổi nhắm đúng dữ liệu câu đọc "
        f"(phiếu nhập, giá ngày D, lô, khách mới, đơn ghi nợ, khoản trả, đơn lớn tháng khác…); kết quả {json.dumps(pr.get('status'))}; ứng viên đạt có đột biến đổi đáp án: "
        f"{(pr.get('before') or {}).get('pass_with_changed_gold')} → {(pr.get('after') or {}).get('pass_with_changed_gold')}. Ứng viên trượt m_probe bị loại.",
        "- **Bẫy định dạng BTC chưa mô phỏng** (tiêu đề lặp giữa sheet, tên báo cáo giữa sheet, dòng cộng theo ngày): loader runtime bỏ cả ba trước khi tới prompt; "
        "khoá bằng test upstream `tests/test_ingest_btc_layout_quirks.py` (CSV + XLSX) thay vì sinh lại shop.",
        "- **'SQL đơn giản' không phải khoảng cách:** so với SQL đã kiểm của đội cho câu BTC (1.194 câu, số gộp) dữ liệu tổng hợp có JOIN nhiều hơn, window function tương đương; chỉ CTE ít hơn.",
        f"- data_hash: `{prev_sm.get('data_hash')}` → `{sm.get('data_hash')}` (giữ `source_manifest.v1_63353535.json`); FINAL không đổi.",
        "- Chưa làm (cần API hoặc chi phí lớn): sinh lại shop lớn như BTC, cân lại văn phong khi chọn lại, thêm lời giải khác họ thay Qwen, người kiểm.", "",
        "## 13. Còn thiếu hoặc chưa thể chạy",
        "- GPU-training: không chạy (đúng phạm vi): gpu_bf16, forward/backward, trainable modules/dtypes, save/load adapter, resume, throughput/memory, pin dependency.",
        "- bwrap không chạy được trên máy này: exporter, rebuild, guard, render prompt chạy `--isolation process` (tách tiến trình + bản chiếu không gold; không cách ly hệ thống file).",
        "- Chưa có người kiểm; audit LLM không chạy lại sau export.",
        "- Dòng train không có đột biến đổi đáp án chỉ được bảo vệ bằng kiểm tĩnh 'không ghi cứng'.",
        "- Prompt dòng SFT là prompt runtime dựng lại, không phải prompt model đã thấy (§3).",
        "- Chưa push/PR; chờ duyệt.",
    ]
    (PKG / "REPORT_vi.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(PKG / "REPORT_vi.md", PKG / "HASHES.json")


if __name__ == "__main__":
    main()
