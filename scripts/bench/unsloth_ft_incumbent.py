#!/usr/bin/env python3
"""Incumbent (Unsloth) side of beat-unsloth-finetune-throughput-v1 (0.72 T2, row R5).

Called by unsloth_ft_incumbent_side.sh, which unsloth_finetune_throughput.sh
runs under gpu-q. Trains the canonical task once and writes ONE receipt with
the fields unsloth_ft_verdict.py requires. It never decides anything.

  unsloth_ft_incumbent.py --side incumbent --model M --gpu N --run K --receipt PATH
                          [--planted-no-fla | --planted-incumbent-adamw8bit]

The data is APR_FT_DATA (JSONL, one {"text": ...} per line), the same file the
apr side reads; its sha256 goes in the receipt, so different data on the two
sides is a SAME-WORK FAIL, not a silent ratio.

Planted falsifiers (contract 1.1.0):
  --planted-no-fla               fla is made unimportable BEFORE transformers is
                                 imported, so HF GDN really takes its torch
                                 fallback; versions.fla is recorded empty and the
                                 verdict refuses INCUMBENT_SLOW_PATH.
  --planted-incumbent-adamw8bit  bitsandbytes AdamW8bit; optimizer=adamw_8bit,
                                 and the verdict says SAME-WORK FAIL.

Token count (both sides count the same way): per sequence, the non-pad tokens
that carry a loss after the one-token shift, i.e. max(nonpad - 1, 0).

The pure parts (argument rules, token count, receipt) are tested on a CPU box
by scripts/tests/unsloth_ft_incumbent_test.py. The GPU part is [U] until it
runs on lambda at train-idle: the Unsloth calls follow its documented API
(FastLanguageModel.from_pretrained / get_peft_model), not a run.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TASK = dict(rank=16, alpha=32, seq_len=512, batch=4, grad_accum=1,
            warmup_steps=50, timed_steps=200, lr=2e-4)
# The distribution name each receipt key is read from.
VERSION_DISTS = dict(unsloth="unsloth", unsloth_zoo="unsloth_zoo", torch="torch",
                     transformers="transformers", fla="flash-linear-attention",
                     causal_conv1d="causal-conv1d")


def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--side", required=True, choices=["incumbent"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", required=True, type=int)
    ap.add_argument("--run", required=True, type=int)
    ap.add_argument("--receipt", required=True)
    ap.add_argument("--planted-no-fla", action="store_true")
    ap.add_argument("--planted-incumbent-adamw8bit", action="store_true")
    a = ap.parse_args(argv)
    if a.planted_no_fla and a.planted_incumbent_adamw8bit:
        ap.error("one planted falsifier per run")
    return a


def block_fla():
    """Make fla unimportable for this process (planted-no-fla). Must run before
    transformers is imported, or HF has already chosen the fla path."""
    for name in ("fla", "causal_conv1d"):
        sys.modules[name] = None


def versions(blocked):
    """Installed versions; a blocked or absent package records ""."""
    out = {}
    for key, dist in VERSION_DISTS.items():
        try:
            out[key] = "" if key in blocked else importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[key] = ""
    return out


def label_tokens(rows, pad_id):
    """Loss-carrying tokens in a batch of token-id rows: max(nonpad - 1, 0) each."""
    return sum(max(sum(1 for t in row if t != pad_id) - 1, 0) for row in rows)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def optimizer_name(args):
    return "adamw_8bit" if args.planted_incumbent_adamw8bit else "adamw_fp32"


DTYPE_NAMES = {"torch.bfloat16": "bf16", "torch.float32": "fp32", "torch.float16": "fp16"}


def precision_name(dtype):
    """bf16/fp32/fp16 for a torch dtype's str(); anything else verbatim, so it cannot pin."""
    return DTYPE_NAMES.get(str(dtype), str(dtype))


def build_receipt(args, data_sha, gpu, measured, vers):
    """The receipt the verdict reads. gpu: name, uuid, trace; measured:
    precision, trainable_params, label_tokens_timed, timed_seconds, timed_after_compile."""
    return dict(
        side="incumbent", model=args.model, data_sha256=data_sha,
        gpu_name=gpu["name"], gpu_uuid=gpu["uuid"], device_trace_line=gpu["trace"],
        precision=measured["precision"], rank=TASK["rank"], alpha=TASK["alpha"],
        targets=list(TARGETS),
        trainable_params=measured["trainable_params"], optimizer=optimizer_name(args),
        grad_checkpointing=False, packing=False, seq_len=TASK["seq_len"],
        batch=TASK["batch"], grad_accum=TASK["grad_accum"],
        warmup_steps=TASK["warmup_steps"], timed_steps=TASK["timed_steps"],
        timed_after_compile=measured["timed_after_compile"],
        label_tokens_timed=measured["label_tokens_timed"],
        timed_seconds=measured["timed_seconds"], versions=vers,
    )


# ---- GPU part ([U] until it runs on lambda) ----------------------------------

def load_batches(path, tokenizer):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line)["text"])
    enc = tokenizer(texts, max_length=TASK["seq_len"], truncation=True,
                    padding="max_length", return_tensors="pt")
    b = TASK["batch"]
    return [(enc["input_ids"][i:i + b], enc["attention_mask"][i:i + b])
            for i in range(0, len(texts) - b + 1, b)]


def gpu_identity(torch, index):
    props = torch.cuda.get_device_properties(index)
    uuid = "GPU-" + str(getattr(props, "uuid", "unknown"))
    trace = "[TRACE] device=cuda:%d name=%s uuid=%s" % (index, props.name, uuid)
    return dict(name=props.name, uuid=uuid, trace=trace)


def make_optimizer(torch, params, args):
    if args.planted_incumbent_adamw8bit:
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(params, lr=TASK["lr"])
    return torch.optim.AdamW(params, lr=TASK["lr"])


def step(torch, model, opt, ids, mask, pad_id):
    # Count on the CPU copy before the move, so counting adds no GPU sync.
    tokens = label_tokens(ids.tolist(), pad_id)
    ids, mask = ids.cuda(), mask.cuda()
    labels = ids.masked_fill(ids == pad_id, -100)
    model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return tokens


def compile_count(torch):
    try:
        return int(torch._dynamo.utils.counters["stats"]["unique_graphs"])
    except (AttributeError, KeyError, TypeError):
        return -1


def train(args):
    from unsloth import FastLanguageModel  # unsloth first: it patches transformers
    import torch
    model, tok = FastLanguageModel.from_pretrained(
        args.model, max_seq_length=TASK["seq_len"], dtype=torch.bfloat16, load_in_4bit=False)
    model = FastLanguageModel.get_peft_model(
        model, r=TASK["rank"], lora_alpha=TASK["alpha"], target_modules=TARGETS,
        lora_dropout=0, bias="none", use_gradient_checkpointing=False)
    params = [p for p in model.parameters() if p.requires_grad]
    frozen = next(p for p in model.parameters() if not p.requires_grad)
    opt = make_optimizer(torch, params, args)
    batches = load_batches(os.environ["APR_FT_DATA"], tok)
    pad = tok.pad_token_id
    n = TASK["warmup_steps"] + TASK["timed_steps"]
    seq = [batches[i % len(batches)] for i in range(n)]
    for ids, mask in seq[:TASK["warmup_steps"]]:
        step(torch, model, opt, ids, mask, pad)
    torch.cuda.synchronize()
    graphs0, t0, tokens = compile_count(torch), time.perf_counter(), 0
    for ids, mask in seq[TASK["warmup_steps"]:]:
        tokens += step(torch, model, opt, ids, mask, pad)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    # Timed after compile = no new graph compiled inside the timed window.
    after = graphs0 >= 0 and compile_count(torch) == graphs0
    measured = dict(precision=precision_name(frozen.dtype),
                    trainable_params=sum(p.numel() for p in params), label_tokens_timed=tokens,
                    timed_seconds=seconds, timed_after_compile=after)
    return measured, gpu_identity(torch, torch.cuda.current_device())


def main(argv):
    args = parse(argv)
    blocked = ("fla", "causal_conv1d") if args.planted_no_fla else ()
    if blocked:
        block_fla()
    data = os.environ.get("APR_FT_DATA", "")
    if not data or not os.path.isfile(data):
        print("APR_FT_DATA is not a file: %r" % data, file=sys.stderr)
        return 3
    measured, gpu = train(args)
    receipt = build_receipt(args, sha256_file(data), gpu, measured, versions(blocked))
    with open(args.receipt, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
