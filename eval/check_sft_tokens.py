"""Exact Qwen3-14B template/token check, run in an existing tokenizer environment.

No new dependency is installed by the data pipeline. Missing transformers/torch
is a failed gate, never replaced with a character estimate. This implements the
documented prefix-masked collator; it does not claim TRL's automatic mask works.
"""
import argparse
import hashlib
import json
from pathlib import Path

# Official Qwen/Qwen3-14B revision 40c069824f4251a91eefaf281ebe4c544efd3e18.
PINNED_FILES = {"tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
                "config.json": "e73c3664ca09b10a673fef0c22e8a6b456201d49bd4713c9691f775720e8857a"}


def verify_identity(path):
    for name, expected in PINNED_FILES.items():
        p = Path(path) / name
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise ValueError("Qwen3-14B tokenizer identity mismatch: " + name)


def template_tokens(tok, messages, generation):
    # Transformers 5 can return BatchEncoding by default; never compare its keys
    # or len(mapping) as though they were token IDs.
    value = tok.apply_chat_template(messages, tokenize=True, return_dict=False,
                                    add_generation_prompt=generation, enable_thinking=False)
    return value["input_ids"] if hasattr(value, "keys") else value


def check(rows, tokenizer_path, max_length=32768):
    verify_identity(tokenizer_path)
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    sequences, result = [], []
    for r in rows:
        full = template_tokens(tok, r["messages"], False)
        prefix = template_tokens(tok, r["messages"][:-1], True)
        if full[:len(prefix)] != prefix:
            raise ValueError("Qwen template is not prefix aligned")
        if len(full) > max_length or len(prefix) >= len(full):
            raise ValueError("sequence too long or missing assistant completion")
        labels = [-100] * len(prefix) + full[len(prefix):]
        sequences.append((full, labels, len(prefix)))
        result.append({"id": r["id"], "tokens": len(full), "prompt_tokens": len(prefix), "assistant_tokens": len(full) - len(prefix)})
    # Check real padded batches including variable length examples, no truncation.
    for start in range(0, len(sequences), 8):
        batch = sequences[start:start + 8]
        length = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), length), tok.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        for i, (tokens, loss, n) in enumerate(batch):
            ids[i, :len(tokens)] = torch.tensor(tokens)
            labels[i, :len(loss)] = torch.tensor(loss)
            assert (labels[i, :n] == -100).all()
            assert (labels[i, n:len(tokens)] == ids[i, n:len(tokens)]).all()
            assert (labels[i, len(tokens):] == -100).all()
            assert (labels[i] != -100).any()
    return {"tokenizer": "Qwen/Qwen3-14B", "enable_thinking": False, "max_length": max_length,
            "revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
            "mask": "prefix labels -100; assistant completion and EOS labels equal input IDs; padding -100",
            "automatic_assistant_only_loss": "not assumed: official template has no generation mask blocks",
            "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(tokenizer_path).glob("*.json")},
            "rows": result, "passed": True}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.input.read_text().splitlines() if l.strip()]
    a.out.write_text(json.dumps(check(rows, a.tokenizer), ensure_ascii=False, indent=2))
