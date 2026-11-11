"""SFT LoRA in BF16 on Qwen3.5-9B: the base model loads unquantized (bfloat16) and stays frozen, LoRA trains on every projection of
the language model, the loss is on the reply only. notebooks/05_train_lora.ipynb drives this script.

    # 1. data check with the tokenizer only (no GPU, no weights): format, template, masks, lengths, rows over --max-len
    python training/train_lora_bf16.py --train SFT/train.jsonl --dev SFT/dev.jsonl --model models/Qwen3.5-9B --tokenizer adapters/sql_rft_v3 \
        --out runs/lora/x/data_check --validate-only
    # 2. training on one GPU; N GPUs (data parallel, one bf16 copy each): torchrun --standalone --nproc_per_node N training/train_lora_bf16.py ...
    python training/train_lora_bf16.py --train SFT/train.jsonl --dev SFT/dev.jsonl --model models/Qwen3.5-9B --out runs/lora/x [--resume]

What is trained
    LoRA (r 16, alpha 32 by default) on the 248 projections that adapters/sql_rft_v3 covers: full-attention q/k/v/o (8 layers),
    Gated DeltaNet in_proj_qkv / in_proj_z / in_proj_b / in_proj_a / out_proj (24 layers), MLP gate / up / down (all 32 layers).
    Same module names and the same target_modules string in adapter_config.json, so datamate.predict (vLLM --lora) serves the result
    as it serves v3, scripts/use_adapter.py can make it the submitted adapter, and --init-adapter continues v3.
    The vision tower is not touched. The base stays bf16: the adapter is fitted to the weights vLLM serves, not to a 4-bit copy of
    them (training/train_lora.py, Unsloth QLoRA, stays as the recipe of the submitted adapters v1-v3).

Rows: {"id", "system", "user", "reply", "thinking"} (training/build_sft.py, add_notes.py, split_sft.py), or rows of the SFT exporter
    (data/data/{train,dev}.jsonl: {"id", "business_id", "prompt_messages": [system, user], "completion", "thinking_mode", "metadata"}),
    read as the same fields. When the exporter's sft_manifest.json sits next to a file, the file must match it (SHA-256). Train and dev
    must not share a question or a shop.
    The prompt is rendered with the model's chat template, as datamate.predict sends it to vLLM: system + user, generation prompt,
    enable_thinking from the row. Qwen3.5 ends a non-thinking prompt with "<think>\n\n</think>\n\n" and a thinking one with "<think>\n",
    so a thinking reply holds the thought, "</think>" and the answer. Prompt and reply are tokenized apart (the prompt as vLLM
    tokenizes it, the reply + "<|im_end|>" as the model writes it). Labels are -100 on the prompt. Rows longer than --max-len are
    rejected and listed, never cut (a cut row loses the reply at its end). Duplicate rows and rows that break the format are rejected.

Loss
    Token-mean cross-entropy over the reply of each row; the Trainer averages rows over gradient accumulation (Qwen3.5 takes no loss
    kwargs). The LM head runs only from the last prompt position on, and by chunks of --ce-chunk positions under activation
    checkpointing: the same loss and gradients as the plain head, without a [reply tokens x 248k vocabulary] float32 logit tensor
    (several GB for a long thinking reply). --ce-chunk 0 runs the plain head on the reply positions.

Out dir: data_report.json, rejected_rows.jsonl, run_identity.json (what --resume must match), train_args.json (every setting without
file paths: `--config train_args.json` reruns it, as docker/train.sh does for a submitted BF16 adapter), checkpoint-*/ (resumable),
adapter/ (the final adapter: the checkpoint with the lowest dev loss when --dev is given), trainer_state.json, run_manifest.json
(settings, data report, losses, peak memory, adapter check).
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import os
import random
import re
import struct
import sys
import time
from collections import Counter
from pathlib import Path

ARCH = "Qwen3_5ForConditionalGeneration"
# The LoRA modules of adapters/sql_rft_v3 under their Qwen3_5ForConditionalGeneration names. The vision tower is not touched.
TARGET_MODULES = (r"model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
                  r"|linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|mlp\.(?:gate_proj|up_proj|down_proj))")
# The target_modules string of adapters/sql_rft_v3/adapter_config.json, verbatim: PEFT selects with it, so the saved config is the one
# vLLM already serves and scripts/use_adapter.py accepts. On Qwen3_5ForConditionalGeneration it selects exactly the TARGET_MODULES set
# (checked in add_lora; the explicit pattern above is what the checks read).
PEFT_TARGETS = (r"(?:.*?(?:language|text).*?(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer).*?\.(?:qkv|proj|linear_fc1|linear_fc2|"
                r"out_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj))|(?:\bmodel\.layers\.[\d]{1,}"
                r"(?!.*\.experts\.\d)\..*?(?:self_attn|attention|attn|mixer|mlp|feed_forward|ffn|dense|mixer).*?\.(?:(?:qkv|proj|linear_fc1|linear_fc2|"
                r"out_proj|in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)))")
MODULES_PER_LAYER = {"full_attention": 4 + 3, "linear_attention": 5 + 3}   # attention or Gated DeltaNet projections + MLP
END = "<|im_end|>"                                                          # Qwen closes the model turn with it
TAIL = {False: "<think>\n\n</think>\n\n", True: "<think>\n"}                # end of the generation prompt, enable_thinking off / on
SPECIAL = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")
LENGTHS = (8192, 12288, 16384, 20480, 24576, 32768, 40960)
IDENTITY_FIELDS = ("r", "alpha", "dropout", "lr", "epochs", "grad_accum", "micro_batch", "max_len", "warmup_ratio", "scheduler",
                   "weight_decay", "max_grad_norm", "seed", "limit_rows", "max_steps", "longest_first", "dev_max_rows")
# Arguments that name files or steer one session only: never part of train_args.json, never read from --config
NOT_CONFIG = ("train", "dev", "model", "tokenizer", "out", "validate_only", "resume", "config", "report_to")


class TemplateError(SystemExit):
    pass


# ---- data (no torch: --validate-only and the tests run without the training stack) ----------------------------------------------
def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def from_exporter(r):
    """A row of the SFT exporter (prompt_messages + completion + thinking_mode) as the fields this trainer reads: system, user, reply,
    thinking, plus question_id / shop for the split check and route for the report. Rows of build_sft.py pass through unchanged."""
    if "prompt_messages" not in r:
        return r
    meta = r.get("metadata") or {}
    msgs = [m for m in r.get("prompt_messages") or [] if isinstance(m, dict)]
    out = {"id": r.get("id"), "question_id": meta.get("question_id"), "shop": r.get("business_id"), "route": meta.get("route"),
           "reply": r.get("completion"), "_src": r["_src"]}
    if "thinking_mode" in r:
        out["thinking"] = bool(r["thinking_mode"])
    roles = [m.get("role") for m in msgs]
    if roles == ["system", "user"]:
        out["system"], out["user"] = msgs[0].get("content"), msgs[1].get("content")
    else:
        out["_bad"] = f"prompt_messages roles {roles}, expected [system, user]"
    if meta.get("training_eligible") is False:
        out["_bad"] = "metadata.training_eligible is false"
    return out


def read_rows(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if line.strip():
                    r = json.loads(line)
                    r["_src"] = f"{Path(p).name}:{n}"
                    rows.append(from_exporter(r))
    return rows


def check_manifest(path):
    """The exporter writes sft_manifest.json next to train.jsonl / dev.jsonl with the SHA-256 of each file. When it is there, the file must
    be the one it lists: a truncated copy or a hand-edited row is refused before hours of training. Returns what the run records, or None."""
    path = Path(path)
    m = path.parent / "sft_manifest.json"
    if not m.exists():
        return None
    man = json.loads(m.read_text(encoding="utf-8"))
    split = next((k for k, v in (man.get("splits") or {}).items() if isinstance(v, dict) and v.get("file") == path.name), None)
    if split is None:
        return None
    want, got = man["splits"][split].get("file_hash"), "sha256:" + sha256_file(path)
    if got != want:
        raise SystemExit(f"{path} does not match {m} (split {split}: {want}, file {got}): the file was changed or copied incompletely")
    return {"file": path.name, "split": split, "rows": man["splits"][split].get("rows"), "build_id": man.get("build_id")}


def split_overlap(train_rows, dev_rows):
    """Question ids and shops found in both splits: the dev loss must measure new questions of new shops."""
    def qids(rows):
        return {str(q) for r in rows if (q := r.get("question_id") or r.get("id")) is not None}

    def shops(rows):
        return {r["shop"] for r in rows if r.get("shop")}
    return sorted(qids(train_rows) & qids(dev_rows)), sorted(shops(train_rows) & shops(dev_rows))


def normalize(r, default_thinking=False):
    """(row ready to encode, None) or (None, reason): the format checks a row must pass, so the adapter never learns a broken turn."""
    if r.get("_bad"):
        return None, r["_bad"]
    if not all(isinstance(r.get(k), str) and r[k].strip() for k in ("system", "user", "reply")):
        return None, "missing system / user / reply"
    thinking = bool(r.get("thinking", default_thinking))
    reply = r["reply"].strip()
    if thinking and reply.startswith("<think>"):  # the generation prompt already opened the thought
        reply = reply[len("<think>"):].lstrip("\n")
    if any(s in reply for s in SPECIAL):
        return None, "special token in the reply"
    if thinking:
        if reply.count("</think>") != 1 or "<think>" in reply:
            return None, "thinking reply without exactly one </think>"
        answer = reply.split("</think>")[1]
    else:
        if "<think>" in reply or "</think>" in reply:
            return None, "non-thinking reply holds a thought"
        answer = reply
    if "```sql" not in answer:
        return None, "no ```sql block in the answer"
    return {**r, "reply": reply, "thinking": thinking}, None


def render_prompt(tok, r):
    """The prompt exactly as inference renders it (datamate/llm.py: chat_template_kwargs enable_thinking, add_generation_prompt)."""
    prompt = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=r["thinking"])
    if not prompt.endswith(TAIL[r["thinking"]]):
        raise TemplateError(f"the chat template does not end a {'thinking' if r['thinking'] else 'non-thinking'} generation prompt with "
                            f"{TAIL[r['thinking']]!r} (got ...{prompt[-40:]!r}): this is not the Qwen3.5 template the rows were built for")
    return prompt


def encode(tok, r):
    """(prompt ids, reply ids). Each part is tokenized on its own: the prompt as vLLM tokenizes it, the reply + END as the model writes it."""
    prompt = render_prompt(tok, r)
    return tok(prompt, add_special_tokens=False)["input_ids"], tok(r["reply"] + END, add_special_tokens=False)["input_ids"], prompt


def prepare(rows, tok, max_len, default_thinking=False, roundtrip=16):
    """Examples {"id", "input_ids", "n_prompt"}, rejected rows with their reason, and per-row token counts of the rows that fit the format."""
    examples, rejected, lengths, seen = [], [], [], set()
    for r in rows:
        rid = str(r.get("id", r["_src"]))
        row, why = normalize(r, default_thinking)
        if row is None:
            rejected.append({"id": rid, "src": r["_src"], "reason": why})
            continue
        key = hashlib.sha1("\0".join((row["system"], row["user"], row["reply"])).encode()).hexdigest()
        if key in seen:
            rejected.append({"id": rid, "src": r["_src"], "reason": "duplicate row"})
            continue
        seen.add(key)
        p, c, prompt = encode(tok, row)
        if len(examples) < roundtrip and tok.decode(p + c, skip_special_tokens=False) != prompt + row["reply"] + END:
            raise TemplateError(f"tokenization does not round-trip on row {rid}: the tokenizer and the rows disagree")
        n = len(p) + len(c)
        lengths.append({"id": rid, "tokens": n, "reply_tokens": len(c), "thinking": row["thinking"], "route": row.get("route")})
        if n > max_len:
            rejected.append({"id": rid, "src": r["_src"], "reason": f"{n} tokens > --max-len {max_len}", "tokens": n})
            continue
        # int32 array, not a list: 4 bytes a token instead of ~36 (the 7.4k rows of data/data, ~96M tokens: ~0.4 GB of RAM, not ~3.5 GB)
        examples.append({"id": rid, "input_ids": array.array("i", p + c), "n_prompt": len(p), "thinking": row["thinking"]})
    return examples, rejected, lengths


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else 0


def length_report(lengths, examples, rejected):
    tok = [x["tokens"] for x in lengths]
    lab = sum(len(e["input_ids"]) - e["n_prompt"] for e in examples)
    total = sum(len(e["input_ids"]) for e in examples)
    return {"rows_read": len(lengths) + sum(1 for x in rejected if "tokens" not in x), "rows_kept": len(examples),
            "rejected": dict(Counter(re.sub(r"^\d+ tokens > ", "longer than ", x["reason"]) for x in rejected)),
            "thinking_rows_kept": sum(e["thinking"] for e in examples),
            "tokens_kept": total, "label_tokens_kept": lab, "label_share": round(lab / max(total, 1), 4),
            "row_tokens": {"p50": quantile(tok, .5), "p90": quantile(tok, .9), "p99": quantile(tok, .99), "max": max(tok, default=0)},
            "reply_tokens": {"p50": quantile([x["reply_tokens"] for x in lengths], .5), "max": max((x["reply_tokens"] for x in lengths), default=0)},
            "share_kept_at_max_len": {L: round(sum(t <= L for t in tok) / max(len(tok), 1), 4) for L in LENGTHS},
            "routes": dict(Counter(x["route"] for x in lengths))}


def load_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    if not tok.chat_template:  # a multimodal checkpoint may keep the template with its processor only
        from transformers import AutoProcessor
        tok.chat_template = AutoProcessor.from_pretrained(path).chat_template
    if not tok.chat_template:
        raise TemplateError(f"no chat template in {path}")
    ids = tok(END, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise TemplateError(f"{END} is not a single token of this tokenizer: not a Qwen3.5 tokenizer")
    if tok.pad_token_id is None:
        tok.pad_token = "<|endoftext|>"
    return tok


def identity(a, tok, train_files, dev_files, n_train, n_dev, world):
    cfg = Path(a.model) / "config.json"
    weights = sorted((p.name, p.stat().st_size) for p in Path(a.model).glob("*.safetensors"))
    init = Path(a.init_adapter) / "adapter_model.safetensors" if a.init_adapter else None
    return {"model_config_sha256": sha256_file(cfg) if cfg.exists() else None, "model_weight_files": weights,
            "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
            "train": [[str(p), sha256_file(p)] for p in train_files], "dev": [[str(p), sha256_file(p)] for p in dev_files],
            "rows_train": n_train, "rows_dev": n_dev, "world_size": world,
            "init_adapter_sha256": sha256_file(init) if init else None, **{k: getattr(a, k) for k in IDENTITY_FIELDS}}


def safetensors_header(path):
    """Tensor name -> {dtype, shape, data_offsets} of a .safetensors file, read from its JSON header (no torch, no safetensors)."""
    with open(path, "rb") as f:
        head = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    head.pop("__metadata__", None)
    return head


def check_adapter(path, n_modules, r, alpha):
    """Structural check of a saved adapter: r / alpha, one lora_A and one lora_B per target module, every name a projection of the
    language model as vLLM expects it (the names of adapters/sql_rft_v3). Raises on a mismatch, returns a summary."""
    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    head = safetensors_header(path / "adapter_model.safetensors")
    name = re.compile(rf"base_model\.model\.({TARGET_MODULES})\.lora_[AB]\.weight")
    problems = []
    if (cfg.get("r"), cfg.get("lora_alpha")) != (r, alpha):
        problems.append(f"r={cfg.get('r')} alpha={cfg.get('lora_alpha')}, expected r={r} alpha={alpha}")
    if len(head) != 2 * n_modules:
        problems.append(f"{len(head)} tensors, expected {2 * n_modules} (lora_A + lora_B of {n_modules} modules)")
    odd = sorted(k for k in head if not name.fullmatch(k))
    if odd:
        problems.append(f"{len(odd)} tensor names outside the language-model projections, e.g. {odd[:2]}")
    if problems:
        raise SystemExit(f"adapter {path}: " + "; ".join(problems))
    return {"tensors": len(head), "modules": n_modules, "r": cfg["r"], "alpha": cfg["lora_alpha"],
            "dtypes": sorted({h["dtype"] for h in head.values()}), "mb": round((path / "adapter_model.safetensors").stat().st_size / 2**20, 1)}


# ---- training (heavy imports here only) ---------------------------------------------------------------------------------------
def expected_modules(text_config):
    types = getattr(text_config, "layer_types", None) or ["full_attention"] * text_config.num_hidden_layers
    return sum(MODULES_PER_LAYER[t] for t in types)


def chunked_ce(hidden, target, weight, chunk):
    """Sum of the token cross-entropies of `target` given the final hidden states that predict it, the LM head applied `chunk`
    positions at a time. Each chunk runs under non-reentrant checkpointing, so its logits are rebuilt in the backward pass instead of
    being kept: peak memory ~ chunk x vocabulary x 10 bytes, whatever the reply length. bf16 head, float32 softmax, as the Hugging Face loss."""
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    def part(h, y):   # accelerate hands the forward outputs back in float32: the head runs in its own dtype, as under autocast
        logits = F.linear(h.to(weight.dtype), weight).float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), ignore_index=-100, reduction="sum")

    total = hidden.sum() * 0.0   # keeps the graph when there is no target
    for i in range(0, hidden.shape[1], chunk):
        total = total + checkpoint(part, hidden[:, i:i + chunk], target[:, i:i + chunk], use_reentrant=False)
    return total.float()


def install_hidden_head(model):
    """Swap lm_head for a pass-through that holds the real head: the forward then returns the final hidden states of the kept
    positions as "logits", and the loss applies the head by chunks (chunked_ce). Returns the real head."""
    import torch

    class HiddenOut(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.head = head

        def forward(self, x):
            return x

    head = model.lm_head
    model.lm_head = HiddenOut(head)
    return head


def gdn_kernel():
    """'fla' or 'torch': the chunked gated delta rule the 24 Gated DeltaNet layers will run. When fla does not import (not installed,
    or a missing dependency such as einops) transformers falls back to its PyTorch reference with only a logger warning, and that
    path is ~90x slower per layer: the run log says which one is in use."""
    import inspect
    from transformers.models.qwen3_5 import modeling_qwen3_5 as M
    if hasattr(M, "chunk_gated_delta_rule"):   # transformers <= 5.13: fla imported at module level, None without it
        f = M.chunk_gated_delta_rule
    else:                                      # later versions: a dispatch wrapper around the torch function holds the implementation
        try:
            f = inspect.getclosurevars(M.torch_chunk_gated_delta_rule).nonlocals.get("implementation")
        except (TypeError, ValueError):
            return "unknown"
    return "fla" if f is not None and (getattr(f, "__module__", "") or "").startswith("fla") else "torch"


def load_model(a, local_rank):
    import torch
    import transformers
    cfg = transformers.AutoConfig.from_pretrained(a.model)
    arch = (cfg.architectures or ["?"])[0]
    if arch != ARCH:
        raise SystemExit(f"{a.model} is a {arch}, expected {ARCH} (Qwen3.5-9B)")
    if getattr(cfg, "quantization_config", None):
        raise SystemExit(f"{a.model} is a quantized checkpoint: BF16 LoRA trains on the bf16 base")
    model = getattr(transformers, arch).from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=a.attn,
                                                       device_map={"": local_rank} if torch.cuda.is_available() else None)
    model.config.use_cache = False
    model.config.text_config.use_cache = False
    bad = {str(m.weight.dtype) for n, m in model.named_modules() if re.fullmatch(TARGET_MODULES, n) and m.weight.dtype != torch.bfloat16}
    if bad:
        raise SystemExit(f"target weights load as {bad}, not bfloat16")
    return model, cfg


def add_lora(a, model, cfg, log):
    import torch
    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from peft.tuners.lora import LoraLayer
    from transformers import set_seed
    lcfg = LoraConfig(r=a.r, lora_alpha=a.alpha, lora_dropout=a.dropout, bias="none", target_modules=PEFT_TARGETS, task_type="CAUSAL_LM")
    set_seed(a.seed)   # lora_A starts random: seed it here, the Trainer seeds only after the adapter exists (a rerun must start the same)
    model = get_peft_model(model, lcfg)   # base frozen; adapter weights in float32 (peft autocast_adapter_dtype), compute in bf16
    wrapped = [k.removeprefix("base_model.model.") for k, m in model.named_modules() if isinstance(m, LoraLayer)]
    n, want = len(wrapped), expected_modules(cfg.text_config)
    odd = [k for k in wrapped if not re.fullmatch(TARGET_MODULES, k)]
    if n != want or odd:
        raise SystemExit(f"LoRA on {n} modules, expected {want} (attention / Gated DeltaNet / MLP projections of every layer)"
                         + (f"; outside them: {odd[:3]}" if odd else ""))
    if a.init_adapter:
        init = Path(a.init_adapter)
        icfg = json.loads((init / "adapter_config.json").read_text(encoding="utf-8"))
        if (icfg["r"], icfg["lora_alpha"]) != (a.r, a.alpha):
            raise SystemExit(f"{init}: r={icfg['r']} alpha={icfg['lora_alpha']}, this run r={a.r} alpha={a.alpha}: continue only with the same r and alpha")
        from safetensors.torch import load_file
        sd = load_file(init / "adapter_model.safetensors")
        own = {k.replace(".default.", "."): k for k, _ in model.named_parameters() if "lora_" in k}
        if set(sd) != set(own):
            raise SystemExit(f"{init} does not cover the same modules: {len(set(own) - set(sd))} missing, {len(set(sd) - set(own))} unexpected, "
                             f"e.g. {sorted(set(own) ^ set(sd))[:2]}")
        res = set_peft_model_state_dict(model, sd)
        if res.unexpected_keys:
            raise SystemExit(f"{init}: {len(res.unexpected_keys)} keys not loaded")
        log(f"continuing from {init}: {len(sd)} tensors loaded")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log(f"LoRA r={a.r} alpha={a.alpha} on {n} modules: {trainable:,} trainable of {total:,} parameters ({trainable / total:.2%}); "
        f"trainable dtype {sorted({str(p.dtype) for p in model.parameters() if p.requires_grad})}, base dtype {torch.bfloat16}")
    return model


def make_trainer(a, model, tok, train_ex, dev_ex, world, log, head=None):
    import torch
    from transformers import EarlyStoppingCallback, Trainer, TrainerCallback, TrainingArguments

    class Rows(torch.utils.data.Dataset):
        def __init__(self, ex):
            self.ex = ex

        def __len__(self):
            return len(self.ex)

        def __getitem__(self, i):
            return self.ex[i]

    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        lab = torch.full((len(batch), n), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            t = torch.tensor(b["input_ids"], dtype=torch.long)   # list or int32 array
            ids[i, :len(t)], att[i, :len(t)] = t, 1
            lab[i, b["n_prompt"]:len(t)] = t[b["n_prompt"]:]
        return {"input_ids": ids, "attention_mask": att, "labels": lab}

    class CompletionOnlyTrainer(Trainer):
        """The causal-LM loss on the reply, with the LM head run only from the last prompt position on: position (first label - 1)
        predicts the first reply token, so keep that many positions. With a chunked head (head given) the model returns hidden states
        there and chunked_ce applies the head; without, the model's own loss runs on the kept logits and the matching label slice."""

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs["labels"]
            cols = (labels != -100).any(0).nonzero()
            keep = labels.shape[1] - int(cols[0]) + 1 if len(cols) else 1
            if head is None:
                inputs = {**inputs, "labels": labels[:, -keep:].contiguous(), "logits_to_keep": keep}
                return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
            out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], logits_to_keep=keep)
            target = labels[:, labels.shape[1] - keep + 1:]               # the tokens that kept positions 0 .. keep-2 predict
            total = chunked_ce(out.logits[:, :-1], target, head.weight, a.ce_chunk)
            if num_items_in_batch is not None and self.model_accepts_loss_kwargs:   # token-sum convention of the Trainer (not Qwen3.5)
                loss = total / num_items_in_batch.to(total.device) if torch.is_tensor(num_items_in_batch) else total / num_items_in_batch
                if self.args.average_tokens_across_devices:
                    loss = loss * self.accelerator.num_processes
            else:                                                            # Qwen3.5: token mean of each micro-batch
                loss = total / (target != -100).sum().clamp_min(1)
            return (loss, out) if return_outputs else loss

        def _get_train_sampler(self, train_dataset=None):
            if a.longest_first and world == 1:  # memory check: the longest rows first, in order
                return torch.utils.data.SequentialSampler(train_dataset or self.train_dataset)
            return super()._get_train_sampler(train_dataset)

    class Log(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()

        def on_log(self, args, state, control, logs=None, **kw):
            if not state.is_world_process_zero or not logs:
                return
            el = time.time() - self.t0
            eta = el / max(state.global_step, 1) * (state.max_steps - state.global_step)
            if "loss" in logs:
                log(f"[train] step {state.global_step}/{state.max_steps} loss {logs['loss']:.4f} grad_norm {logs.get('grad_norm', float('nan')):.3f} "
                    f"lr {logs.get('learning_rate', 0):.2e} epoch {logs.get('epoch', 0):.3f} | {el / 60:.1f} min, ETA {eta / 60:.1f} min")
            if "eval_loss" in logs:
                log(f"[eval] step {state.global_step} dev_loss {logs['eval_loss']:.4f}")

    cuda = torch.cuda.is_available()
    dev = bool(dev_ex)
    if dev and a.save_steps % a.eval_steps:
        raise SystemExit("--save-steps must be a multiple of --eval-steps (the best checkpoint is picked on the dev loss)")
    targs = TrainingArguments(
        output_dir=str(a.out), per_device_train_batch_size=a.micro_batch, per_device_eval_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, max_steps=a.max_steps, learning_rate=a.lr, lr_scheduler_type=a.scheduler, warmup_steps=a.warmup_ratio,
        weight_decay=a.weight_decay, max_grad_norm=a.max_grad_norm, optim="adamw_torch_fused" if cuda else "adamw_torch",
        bf16=True, tf32=cuda or None, gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=a.logging_steps, logging_first_step=True, disable_tqdm=True,
        eval_strategy="steps" if dev else "no", eval_steps=a.eval_steps if dev else None, eval_on_start=dev and a.eval_on_start,
        save_strategy="steps", save_steps=a.save_steps, save_total_limit=a.save_total_limit,
        load_best_model_at_end=dev, metric_for_best_model="eval_loss" if dev else None, greater_is_better=False if dev else None,
        prediction_loss_only=True, report_to=a.report_to, seed=a.seed, data_seed=a.seed, remove_unused_columns=False,
        dataloader_pin_memory=cuda, ddp_find_unused_parameters=False if world > 1 else None, use_cpu=not cuda)
    model.enable_input_require_grads()
    callbacks = [Log()] + ([EarlyStoppingCallback(early_stopping_patience=a.early_stop)] if dev and a.early_stop > 0 else [])
    return CompletionOnlyTrainer(model=model, args=targs, train_dataset=Rows(train_ex), eval_dataset=Rows(dev_ex) if dev else None,
                                 data_collator=collate, callbacks=callbacks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, help="JSON of settings (the train_args.json of an earlier run): defaults that the command line overrides; "
                                                "docker/train.sh retrains the submitted BF16 adapter this way")
    ap.add_argument("--train", type=Path, nargs="+", required=True, help="SFT JSONL file(s)")
    ap.add_argument("--dev", type=Path, nargs="*", default=[], help="dev JSONL (other shops than --train): dev loss, best checkpoint")
    ap.add_argument("--model", required=True, help="local Qwen3.5-9B checkpoint (bf16 safetensors, tokenizer, chat template)")
    ap.add_argument("--tokenizer", help="tokenizer + chat template source (default --model); e.g. adapters/sql_rft_v3 for --validate-only without the weights")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--validate-only", action="store_true", help="tokenizer only: format, masks, lengths; writes data_report.json and exits")
    ap.add_argument("--max-len", type=int, default=24576, help="tokens per row; longer rows are rejected (data_report.json lists the share kept per length)")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--lr", type=float, default=1e-4, help="1e-4 for a new adapter, 5e-5 to continue one (--init-adapter)")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8, help="rows per optimizer step = micro-batch x grad-accum x GPUs")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--scheduler", default="cosine")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--init-adapter", help="continue from this adapter (same r, alpha and modules), e.g. adapters/sql_rft_v3")
    ap.add_argument("--thinking-default", action="store_true", help="enable_thinking for rows without a 'thinking' field (build_sft rows have one)")
    ap.add_argument("--attn", default="sdpa", help="attention of the 8 full-attention layers: sdpa or flash_attention_2")
    ap.add_argument("--limit-rows", type=int, help="train on N rows drawn at random (a short run to learn or check the setup)")
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after N optimizer steps (memory / speed check)")
    ap.add_argument("--longest-first", action="store_true", help="the longest rows first (with --max-steps: peak memory at --max-len)")
    ap.add_argument("--logging-steps", type=int, default=1)
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--dev-max-rows", type=int, default=0, help="dev loss on a fixed random subset of N dev rows (same seed, same rows at every "
                                                               "evaluation; 0 = every dev row). Rows are ~13k tokens: a full dev pass can cost "
                                                               "as much as a third of the training between two evaluations")
    ap.add_argument("--early-stop", type=int, default=0, help="with --dev: stop after N evaluations without a lower dev loss (0 = off)")
    ap.add_argument("--ce-chunk", type=int, default=1024, help="LM head + cross-entropy by chunks of N reply positions (0 = plain head)")
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--no-eval-on-start", dest="eval_on_start", action="store_false", help="skip the dev loss of the start point")
    ap.add_argument("--report-to", default="none", help="none, tensorboard, wandb ...")
    ap.add_argument("--resume", action="store_true", help="continue from the newest checkpoint in --out (same data and settings only)")
    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        cfg = json.loads(pre.config.read_text(encoding="utf-8"))
        known = {x.dest for x in ap._actions}
        unknown = sorted(k for k in cfg if k not in known)
        if unknown:
            raise SystemExit(f"{pre.config}: unknown settings {unknown}")
        ap.set_defaults(**{k: v for k, v in cfg.items() if k not in NOT_CONFIG})
    a = ap.parse_args(argv)
    if not 0 <= a.warmup_ratio < 1:
        raise SystemExit("--warmup-ratio must be in [0, 1)")
    if a.ce_chunk < 0 or a.early_stop < 0 or a.dev_max_rows < 0:
        raise SystemExit("--ce-chunk, --early-stop and --dev-max-rows must be >= 0")

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    log = (lambda *x: print(*x, flush=True)) if rank == 0 else (lambda *x: None)
    a.out.mkdir(parents=True, exist_ok=True)

    manifests = [m for m in (check_manifest(p) for p in [*a.train, *a.dev]) if m]
    tok = load_tokenizer(a.tokenizer or a.model)
    train_rows, dev_rows = read_rows(a.train), read_rows(a.dev)
    same_q, same_shop = split_overlap(train_rows, dev_rows)
    if same_q or same_shop:
        raise SystemExit(f"--train and --dev share {len(same_q)} question ids (e.g. {same_q[:3]}) and {len(same_shop)} shops (e.g. {same_shop[:3]}): "
                         "split by shop (04_training_data)")
    train_ex, train_rej, train_len = prepare(train_rows, tok, a.max_len, a.thinking_default)
    dev_ex, dev_rej, dev_len = prepare(dev_rows, tok, a.max_len, a.thinking_default)
    del train_rows, dev_rows   # the prompt strings (~40k characters a row): the examples keep token ids only
    report = {"model": a.model, "max_len": a.max_len, "sft_manifest": manifests or None, "train": length_report(train_len, train_ex, train_rej),
              "dev": length_report(dev_len, dev_ex, dev_rej) if a.dev else None}
    if rank == 0:
        (a.out / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        with (a.out / "rejected_rows.jsonl").open("w", encoding="utf-8") as f:
            for x in [dict(x, split="train") for x in train_rej] + [dict(x, split="dev") for x in dev_rej]:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    t = report["train"]
    log(f"train: {t['rows_kept']} rows kept of {t['rows_read']}, rejected {t['rejected'] or 'none'}; tokens p50 {t['row_tokens']['p50']} "
        f"p99 {t['row_tokens']['p99']} max {t['row_tokens']['max']}; trained tokens {t['label_share']:.1%} of {t['tokens_kept']:,}")
    log("share of rows kept at max_len: " + ", ".join(f"{k}: {v:.1%}" for k, v in t["share_kept_at_max_len"].items()))
    if a.dev:
        log(f"dev: {report['dev']['rows_kept']} rows kept of {report['dev']['rows_read']}")
    for m in manifests:
        log(f"{m['file']}: matches sft_manifest.json ({m['split']}, {m['rows']} rows, build {str(m['build_id'])[:19]})")
    if not train_ex:
        raise SystemExit(f"no train row passes the checks at --max-len {a.max_len}: see {a.out / 'rejected_rows.jsonl'}")
    if a.validate_only:
        log(f"validate-only: data_report.json and rejected_rows.jsonl in {a.out}")
        return report

    if a.dev_max_rows and len(dev_ex) > a.dev_max_rows:   # a fixed subset: every evaluation scores the same rows
        keep = set(random.Random(a.seed).sample(range(len(dev_ex)), a.dev_max_rows))
        dev_ex = [e for i, e in enumerate(dev_ex) if i in keep]
        log(f"dev loss on {len(dev_ex)} of {report['dev']['rows_kept']} dev rows (--dev-max-rows)")
    random.Random(a.seed).shuffle(train_ex)
    if a.limit_rows:
        train_ex = train_ex[:a.limit_rows]
    if a.longest_first:
        train_ex.sort(key=lambda e: -len(e["input_ids"]))

    import torch
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        if not torch.cuda.is_bf16_supported():
            raise SystemExit(f"{torch.cuda.get_device_name()} has no bf16 (Ampere or newer needed): BF16 LoRA cannot run here")
    elif os.environ.get("DATAMATE_TRAIN_ALLOW_CPU") != "1":
        raise SystemExit("no CUDA GPU: BF16 LoRA of Qwen3.5-9B needs a bf16 GPU (DATAMATE_TRAIN_ALLOW_CPU=1 is for the tiny-model tests only)")

    ident = json.loads(json.dumps(identity(a, tok, a.train, a.dev, len(train_ex), len(dev_ex), world)))   # as stored: tuples become lists
    ckpts = sorted(a.out.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    if ckpts and not a.resume:
        raise SystemExit(f"{a.out} already holds checkpoints: --resume continues that run, a new --out starts another one")
    last = ckpts[-1] if ckpts else None
    idf = a.out / "run_identity.json"
    if last:
        old = json.loads(idf.read_text(encoding="utf-8")) if idf.exists() else {}
        if old != ident:
            diff = sorted(k for k in set(ident) | set(old) if old.get(k) != ident.get(k))
            raise SystemExit(f"{a.out} holds checkpoints of another run (it differs in {diff}): resume needs the same data and settings, or use a new --out")
    if rank == 0:
        idf.write_text(json.dumps(ident, indent=1), encoding="utf-8")
        # every setting of this run, without file paths: --config of a rerun (docker/train.sh) reproduces it on the packaged data
        settings = {k: v for k, v in vars(a).items() if k not in NOT_CONFIG}
        settings["init_adapter"] = Path(a.init_adapter).name if a.init_adapter else None
        (a.out / "train_args.json").write_text(json.dumps(settings, indent=1) + "\n", encoding="utf-8")

    model, cfg = load_model(a, local_rank)
    kernel = gdn_kernel()
    log(f"Gated DeltaNet kernel: {kernel}" + ("" if kernel != "torch" or not torch.cuda.is_available() else
        "  WARNING: fla does not import here (training/check_fla.py says why): the PyTorch reference path is correct but ~90x slower per layer"))
    head = install_hidden_head(model) if a.ce_chunk else None
    model = add_lora(a, model, cfg, log)
    trainer = make_trainer(a, model, tok, train_ex, dev_ex, world, log, head)
    rows_per_step = a.micro_batch * a.grad_accum * world
    steps = a.max_steps if a.max_steps > 0 else math.ceil(math.ceil(len(train_ex) / (a.micro_batch * world)) / a.grad_accum * a.epochs)
    log(f"{len(train_ex)} train rows, {len(dev_ex)} dev rows; {rows_per_step} rows per optimizer step on {world} GPU(s); ~{steps} steps; "
        f"lr {a.lr} {a.scheduler}, warmup {a.warmup_ratio:.0%}; bf16 base, gradient checkpointing, completion-only loss"
        + (f", LM head by chunks of {a.ce_chunk}" if a.ce_chunk else "") + (f", early stop after {a.early_stop} evaluations" if dev_ex and a.early_stop else ""))
    if last:
        log(f"resuming from {last}")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=str(last) if last else None)
    dt = time.time() - t0
    if head is not None:
        model.get_base_model().lm_head = head     # the real head back in place before the model is saved or reused
    trainer.save_model(str(a.out / "adapter"))   # the best dev-loss checkpoint when --dev is given (load_best_model_at_end)
    if rank == 0:
        tok.save_pretrained(str(a.out / "adapter"))
        trainer.state.save_to_json(str(a.out / "trainer_state.json"))
        import peft
        import transformers
        done = trainer.state.global_step * rows_per_step / max(len(train_ex), 1)
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
        reserved = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None
        total_mem = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else None
        adapter_check = check_adapter(a.out / "adapter", expected_modules(cfg.text_config), a.r, a.alpha)
        logged = [h["loss"] for h in trainer.state.log_history if "loss" in h]   # every session of a resumed run, not only this one
        train_loss = sum(logged) / len(logged) if logged else result.training_loss
        manifest = {"identity": ident, "data_report": report, "steps": trainer.state.global_step, "seconds": round(dt),
                    "train_loss": train_loss, "best_dev_loss": trainer.state.best_metric, "best_checkpoint": trainer.state.best_model_checkpoint,
                    "peak_gpu_gib": round(peak, 1) if peak else None, "peak_gpu_reserved_gib": round(reserved, 1) if reserved else None,
                    "gpu_total_gib": round(total_mem, 1) if total_mem else None, "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
                    "stopped_early": bool(a.early_stop and trainer.state.global_step < trainer.state.max_steps), "adapter_check": adapter_check, "gdn_kernel": kernel,
                    "tokens_per_second": round(report["train"]["tokens_kept"] * done / max(dt, 1)) if a.limit_rows is None else None,
                    "versions": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
                    "adapter": str(a.out / "adapter")}
        (a.out / "run_manifest.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
        log(f"{trainer.state.global_step} steps in {dt / 60:.1f} min; train loss {train_loss:.4f}"
            + (f"; best dev loss {trainer.state.best_metric:.4f} ({trainer.state.best_model_checkpoint})" if trainer.state.best_metric is not None else "")
            + (f"; peak GPU {peak:.1f} GiB allocated, {reserved:.1f} GiB reserved of {total_mem:.1f}" if peak else ""))
        log(f"adapter check: {adapter_check['tensors']} tensors on {adapter_check['modules']} modules, r={adapter_check['r']} alpha={adapter_check['alpha']}, "
            f"{adapter_check['mb']} MB")
        log(f"adapter -> {a.out / 'adapter'}")
    return trainer


if __name__ == "__main__":
    main()
