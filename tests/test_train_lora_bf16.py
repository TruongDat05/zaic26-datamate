"""training/train_lora_bf16.py.

Data checks (normalize / prepare / length report / adapter check) run with a fake tokenizer, without torch. The training tests need the
BF16 stack (torch, transformers, peft: training/requirements_bf16.txt) and the tokenizer of adapters/sql_rft_v3 (Git LFS); they build a
4-layer random Qwen3.5 with the real tokenizer and run on CPU:
    .venv-train/bin/python -m pytest tests/test_train_lora_bf16.py
"""
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from training import train_lora_bf16 as T

REPO = Path(__file__).resolve().parents[1]
TOKENIZER = REPO / "adapters/sql_rft_v3"


class FakeTok:
    """Character-level stand-in with the Qwen3.5 chat template layout."""
    chat_template = "fake"
    pad_token_id = 0

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        body = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return body + "<|im_start|>assistant\n" + T.TAIL[bool(enable_thinking)]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


def row(i, reply="Notes:\n- doanh thu tháng này\n```sql\nSELECT SUM(amount) FROM v_sales\n```", thinking=False, user=None):
    return {"id": f"q_{i}", "system": "SYSTEM", "user": user or f"SCHEMA ...\n\nQUESTION: câu hỏi {i}", "reply": reply, "thinking": thinking, "_src": f"x:{i}"}


def test_normalize_accepts_plain_and_thinking_rows():
    assert T.normalize(row(1))[0]["reply"].startswith("Notes:")
    r, why = T.normalize(row(2, reply="<think>\nnghĩ\n</think>\n\n```sql\nSELECT 1\n```", thinking=True))
    assert why is None and r["reply"].startswith("nghĩ")          # the generation prompt already opened the thought


@pytest.mark.parametrize("reply, thinking, reason", [
    ("SELECT 1", False, "no ```sql"),
    ("<think>x</think>```sql\nSELECT 1\n```", False, "holds a thought"),
    ("x\n```sql\nSELECT 1\n```", True, "exactly one </think>"),
    ("```sql\nSELECT 1\n```<|im_end|>", False, "special token"),
])
def test_normalize_rejects_broken_rows(reply, thinking, reason):
    r, why = T.normalize(row(1, reply=reply, thinking=thinking))
    assert r is None and reason in why


def test_prepare_masks_prompt_and_rejects_long_and_duplicate_rows():
    rows = [row(1), row(1), row(2, user="x" * 500), row(3, reply="no sql")]
    ex, rej, lengths = T.prepare(rows, FakeTok(), max_len=300)
    assert len(ex) == 1 and {x["reason"].split(" ")[0] for x in rej} >= {"duplicate", "no"}
    assert any("> --max-len 300" in x["reason"] for x in rej)
    e = ex[0]
    prompt = FakeTok().apply_chat_template([{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": rows[0]["user"]}])
    assert e["n_prompt"] == len(prompt) and "".join(map(chr, e["input_ids"][e["n_prompt"]:])) == rows[0]["reply"] + T.END
    rep = T.length_report(lengths, ex, rej)
    assert rep["rows_kept"] == 1 and 0 < rep["label_share"] < 1 and rep["share_kept_at_max_len"][8192] == 1.0


def exporter_row(i, shop="syn_shop_0001", thinking=False, **kw):
    """A row as the SFT exporter writes it (data/data/train.jsonl)."""
    sql = "```sql\nSELECT COUNT(*) FROM v_orders\n```"
    completion = f"<think>\nđếm đơn hợp lệ\n</think>\n\nNotes:\n- số đơn\nTYPE: number\n{sql}" if thinking else f"Notes:\n- số đơn\nTYPE: number\n{sql}"
    r = {"schema_version": 2, "id": f"syn:q{i:06d}#{i:04x}", "business_id": shop,
         "prompt_messages": [{"role": "system", "content": "SYSTEM\n\nRULEBOOK:\n..."}, {"role": "user", "content": f"SCHEMA ...\n\nQUESTION: câu {i}"}],
         "completion": completion, "thinking_mode": thinking,
         "metadata": {"question_id": f"syn:q{i:06d}", "route": "view", "training_eligible": True}}
    r.update(kw)
    return r


def test_exporter_rows_read_as_trainer_rows(tmp_path):
    rows = [exporter_row(1), exporter_row(2, thinking=True), exporter_row(3, prompt_messages=[{"role": "user", "content": "x"}]),
            exporter_row(4, metadata={"question_id": "syn:q000004", "training_eligible": False})]
    path = tmp_path / "train.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    read = T.read_rows([path])
    assert read[0]["system"].startswith("SYSTEM") and read[0]["question_id"] == "syn:q000001" and read[0]["shop"] == "syn_shop_0001"
    ex, rej, lengths = T.prepare(read, FakeTok(), max_len=10_000)
    assert [e["thinking"] for e in ex] == [False, True]
    assert "".join(map(chr, ex[1]["input_ids"][ex[1]["n_prompt"]:])).startswith("đếm đơn")   # <think>\n comes from the generation prompt
    assert sorted(x["reason"] for x in rej) == ["metadata.training_eligible is false", "prompt_messages roles ['user'], expected [system, user]"]
    assert {x["route"] for x in lengths} == {"view"}


def test_split_overlap_finds_shared_questions_and_shops():
    train = [T.from_exporter(dict(exporter_row(i, shop="a"), _src="t")) for i in range(3)]
    assert T.split_overlap(train, [T.from_exporter(dict(exporter_row(9, shop="b"), _src="d"))]) == ([], [])
    assert T.split_overlap(train, [T.from_exporter(dict(exporter_row(9, shop="a"), _src="d"))]) == ([], ["a"])
    assert T.split_overlap(train, [T.from_exporter(dict(exporter_row(1, shop="b"), _src="d"))]) == (["syn:q000001"], [])
    assert T.split_overlap([row(1)], [row(1)])[0] == ["q_1"]   # build_sft rows: the id is the question id


def test_manifest_must_match_the_file(tmp_path):
    f = tmp_path / "train.jsonl"
    f.write_text(json.dumps(exporter_row(1)) + "\n", encoding="utf-8")
    assert T.check_manifest(f) is None                         # no manifest: nothing to check
    man = {"build_id": "sha256:b", "splits": {"train": {"file": "train.jsonl", "file_hash": "sha256:" + T.sha256_file(f), "rows": 1}}}
    (tmp_path / "sft_manifest.json").write_text(json.dumps(man), encoding="utf-8")
    assert T.check_manifest(f) == {"file": "train.jsonl", "split": "train", "rows": 1, "build_id": "sha256:b"}
    f.write_text(json.dumps(exporter_row(2)) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="does not match"):
        T.check_manifest(f)


def write_adapter(d, n_modules, r=16, alpha=32, rename=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}), encoding="utf-8")
    names = [f"base_model.model.model.language_model.layers.{i}.mlp.{p}.lora_{ab}.weight"
             for i in range(n_modules // 3) for p in ("gate_proj", "up_proj", "down_proj") for ab in "AB"]
    if rename:
        names[0] = rename
    header = {n: {"dtype": "F32", "shape": [1], "data_offsets": [4 * k, 4 * k + 4]} for k, n in enumerate(names)}
    raw = json.dumps(header).encode()
    (d / "adapter_model.safetensors").write_bytes(len(raw).to_bytes(8, "little") + raw + b"\0" * 4 * len(names))


def test_check_adapter(tmp_path):
    write_adapter(tmp_path / "ok", 6)
    assert T.check_adapter(tmp_path / "ok", 6, 16, 32)["tensors"] == 12
    write_adapter(tmp_path / "bad", 6, rename="base_model.model.lm_head.lora_A.weight")
    with pytest.raises(SystemExit, match="outside the language-model"):
        T.check_adapter(tmp_path / "bad", 6, 16, 32)
    with pytest.raises(SystemExit, match="r=16"):
        T.check_adapter(tmp_path / "ok", 6, 8, 16)


# ---- training on a tiny random Qwen3.5 (CPU) ------------------------------------------------------------------------------------
LAYERS = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    if not (TOKENIZER / "tokenizer.json").exists() or (TOKENIZER / "tokenizer.json").stat().st_size < 1_000_000:
        pytest.skip("adapters/sql_rft_v3 tokenizer missing or a Git LFS pointer")
    cfg = transformers.Qwen3_5Config(
        text_config=dict(hidden_size=64, intermediate_size=128, num_hidden_layers=4, layer_types=LAYERS, num_attention_heads=4,
                         num_key_value_heads=2, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
                         linear_value_head_dim=16, vocab_size=248320, max_position_embeddings=8192),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64))
    cfg.architectures = [T.ARCH]
    torch.manual_seed(0)
    d = tmp_path_factory.mktemp("tiny") / "Qwen3.5-9B"   # the base name scripts/use_adapter.py compares with the submitted adapter's
    transformers.Qwen3_5ForConditionalGeneration(cfg).to(torch.bfloat16).save_pretrained(d)
    transformers.AutoTokenizer.from_pretrained(TOKENIZER).save_pretrained(d)
    return d


def sft_rows(n, shop):
    out = []
    for i in range(n):
        thinking = i % 3 == 0
        sql = f"```sql\nSELECT SUM(amount) FROM v_sales WHERE day >= DATE '2026-0{1 + i % 9}-01'\n```"
        reply = (f"Khách hỏi doanh thu, lọc theo ngày.\n</think>\n\n{sql}" if thinking else f"Notes:\n- doanh thu, đơn hoàn tất\n{sql}")
        out.append({"id": f"q_{shop}_{i}", "system": "Bạn viết SQL DuckDB.\n\nRULEBOOK:\n- doanh thu = tổng amount của đơn hoàn tất",
                    "user": f"SCHEMA:\nv_sales(day DATE, amount DOUBLE)\n\nQUESTION: ({i}) doanh thu từ tháng {1 + i % 9} của shop {shop}?",
                    "reply": reply, "thinking": thinking, "route": "view"})
    return out


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def test_chunked_head_matches_the_plain_head(tiny_model):
    torch = pytest.importorskip("torch")
    tok = T.load_tokenizer(str(tiny_model))
    ex, _, _ = T.prepare([dict(r, _src="x") for r in sft_rows(2, "a")], tok, 4096)
    a = Namespace(model=str(tiny_model), attn="sdpa", r=8, alpha=16, dropout=0.0, init_adapter=None, ce_chunk=5, out=tiny_model / "eq",
                  micro_batch=2, grad_accum=1, epochs=1, max_steps=-1, lr=1e-3, scheduler="constant", warmup_ratio=0.0, weight_decay=0.0,
                  max_grad_norm=1.0, logging_steps=1, eval_steps=1, save_steps=1, save_total_limit=1, eval_on_start=False, report_to="none",
                  seed=0, longest_first=False, early_stop=0)
    model, cfg = T.load_model(a, 0)
    model = T.add_lora(a, model, cfg, print)
    torch.manual_seed(1)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.02)                                        # a trained-looking adapter: lora_B starts at zero
    trainer = T.make_trainer(a, model, tok, ex, [], 1, print, head=None)
    batch = {k: v.to(next(model.parameters()).device) for k, v in trainer.data_collator(ex).items()}

    def loss_and_grads(tr):
        model.zero_grad()
        loss = tr.compute_loss(model, batch)
        loss.backward()
        return loss.item(), {n: p.grad.float().clone() for n, p in model.named_parameters() if p.requires_grad}

    ref_loss, ref_grads = loss_and_grads(trainer)
    head = T.install_hidden_head(model.get_base_model())
    chunked_loss, chunked_grads = loss_and_grads(T.make_trainer(a, model, tok, ex, [], 1, print, head=head))
    assert chunked_loss == pytest.approx(ref_loss, rel=1e-3)
    worst = max(((chunked_grads[n] - g).norm() / g.norm().clamp_min(1e-12)).item() for n, g in ref_grads.items())
    assert worst < 2e-2


def test_training_run_keeps_best_dev_checkpoint_and_a_servable_adapter(tiny_model, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("DATAMATE_TRAIN_ALLOW_CPU", "1")
    train = write_jsonl(tmp_path / "train.jsonl", sft_rows(12, "a") + sft_rows(12, "b"))
    dev = write_jsonl(tmp_path / "dev.jsonl", sft_rows(4, "c"))
    out = tmp_path / "run"
    T.main(["--train", str(train), "--dev", str(dev), "--model", str(tiny_model), "--out", str(out), "--max-len", "4096", "--r", "8",
            "--alpha", "16", "--lr", "2e-3", "--epochs", "2", "--grad-accum", "4", "--eval-steps", "2", "--save-steps", "4",
            "--early-stop", "3", "--ce-chunk", "16", "--no-eval-on-start", "--dev-max-rows", "3"])
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["adapter_check"]["modules"] == 3 * 8 + 7 and manifest["adapter_check"]["r"] == 8
    assert manifest["best_dev_loss"] is not None and manifest["best_checkpoint"]
    hist = json.loads((out / "trainer_state.json").read_text())["log_history"]
    assert any("eval_loss" in h for h in hist) and any("loss" in h for h in hist)
    report = json.loads((out / "data_report.json").read_text())
    assert report["train"]["rows_kept"] == 24 and report["dev"]["rows_kept"] == 4
    # the saved config carries v3's target_modules string, so vLLM and scripts/use_adapter.py see the same setup as v3
    cfg = json.loads((out / "adapter/adapter_config.json").read_text())
    assert cfg["target_modules"] == T.PEFT_TARGETS == json.loads((TOKENIZER / "adapter_config.json").read_text())["target_modules"]
    args = json.loads((out / "train_args.json").read_text())
    assert args["lr"] == 2e-3 and args["early_stop"] == 3 and not {"train", "dev", "model", "out"} & set(args)
    assert args["dev_max_rows"] == 3 and json.loads((out / "run_identity.json").read_text())["rows_dev"] == 3   # dev loss on 3 of the 4 dev rows

    # train_args.json reruns the run: --config gives every setting, the command line only the files (and here a 1-step cap)
    T.main(["--config", str(out / "train_args.json"), "--train", str(train), "--dev", str(dev), "--model", str(tiny_model),
            "--out", str(tmp_path / "rerun"), "--max-steps", "1"])
    again = json.loads((tmp_path / "rerun/train_args.json").read_text())
    assert {k: v for k, v in again.items() if k != "max_steps"} == {k: v for k, v in args.items() if k != "max_steps"} and again["max_steps"] == 1

    # --resume with the same data and settings continues the run (here: already at its last step), other settings are refused
    T.main(["--train", str(train), "--dev", str(dev), "--model", str(tiny_model), "--out", str(out), "--max-len", "4096", "--r", "8",
            "--alpha", "16", "--lr", "2e-3", "--epochs", "2", "--grad-accum", "4", "--eval-steps", "2", "--save-steps", "4",
            "--early-stop", "3", "--ce-chunk", "16", "--no-eval-on-start", "--dev-max-rows", "3", "--resume"])
    assert json.loads((out / "run_manifest.json").read_text())["steps"] == manifest["steps"]
    with pytest.raises(SystemExit, match="another run"):
        T.main(["--train", str(train), "--dev", str(dev), "--model", str(tiny_model), "--out", str(out), "--max-len", "4096", "--r", "8",
                "--alpha", "16", "--lr", "1e-3", "--epochs", "2", "--grad-accum", "4", "--eval-steps", "2", "--save-steps", "4", "--resume"])
    T.main(["--train", str(train), "--model", str(tiny_model), "--out", str(tmp_path / "preflight"), "--max-len", "4096", "--r", "8",
            "--alpha", "16", "--max-steps", "1", "--longest-first", "--grad-accum", "2"])
    assert json.loads((tmp_path / "preflight/run_manifest.json").read_text())["steps"] == 1


def test_bf16_adapter_can_become_the_submitted_one(tiny_model, tmp_path, monkeypatch):
    """scripts/use_adapter.py --dry-run accepts an adapter of this trainer (same rank, alpha, target_modules and base model as v3)."""
    pytest.importorskip("torch")
    monkeypatch.setenv("DATAMATE_TRAIN_ALLOW_CPU", "1")
    train = write_jsonl(tmp_path / "train.jsonl", sft_rows(4, "a"))
    T.main(["--train", str(train), "--model", str(tiny_model), "--out", str(tmp_path / "run"), "--max-len", "4096", "--max-steps", "1", "--grad-accum", "2"])
    root = tmp_path / "repo"
    for rel in ("docker/Dockerfile", ".dockerignore", "scripts/run_public.sh", "scripts/run_holdout.sh", "docker/model_manifest.json",
                "adapters/sql_rft_v3/adapter_config.json"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, root / rel)
    sys.path.insert(0, str(REPO / "scripts"))
    import use_adapter
    use_adapter.main([str(tmp_path / "run/adapter"), "--name", "sql_bf16_test", "--root", str(root), "--dry-run"])
    assert not (root / "adapters/sql_bf16_test").exists()


def test_same_settings_give_the_same_adapter(tiny_model, tmp_path, monkeypatch):
    """lora_A starts random: two runs with one seed must start, and so end, the same (train.sh rebuilds the submitted adapter)."""
    torch = pytest.importorskip("torch")
    from safetensors.torch import load_file
    monkeypatch.setenv("DATAMATE_TRAIN_ALLOW_CPU", "1")
    train = write_jsonl(tmp_path / "train.jsonl", sft_rows(4, "a"))
    for k in (1, 2):
        T.main(["--train", str(train), "--model", str(tiny_model), "--out", str(tmp_path / f"run{k}"), "--max-len", "4096", "--max-steps", "1",
                "--grad-accum", "2", "--seed", "7"])
    a, b = (load_file(tmp_path / f"run{k}/adapter/adapter_model.safetensors") for k in (1, 2))
    assert set(a) == set(b) and all(torch.allclose(a[k], b[k], atol=1e-5) for k in a)


def test_config_with_an_unknown_setting_is_refused(tmp_path):
    (tmp_path / "c.json").write_text(json.dumps({"lr": 1e-4, "lerning_rate": 1}), encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown settings"):
        T.main(["--config", str(tmp_path / "c.json"), "--train", "x.jsonl", "--model", "m", "--out", str(tmp_path / "o")])


def test_validate_only_reads_the_tokenizer_alone(tmp_path):
    pytest.importorskip("transformers")
    if not (TOKENIZER / "tokenizer.json").exists() or (TOKENIZER / "tokenizer.json").stat().st_size < 1_000_000:
        pytest.skip("adapters/sql_rft_v3 tokenizer missing or a Git LFS pointer")
    train = write_jsonl(tmp_path / "train.jsonl", sft_rows(6, "a") + [dict(sft_rows(1, "z")[0], reply="no sql here", thinking=False)])
    rep = T.main(["--train", str(train), "--model", "models/absent", "--tokenizer", str(TOKENIZER), "--out", str(tmp_path / "check"),
                  "--validate-only"])
    assert rep["train"]["rows_kept"] == 6 and rep["train"]["rejected"] == {"no ```sql block in the answer": 1}
    assert (tmp_path / "check/rejected_rows.jsonl").read_text(encoding="utf-8").count("\n") == 1
