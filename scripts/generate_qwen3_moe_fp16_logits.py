#!/usr/bin/env python3
"""
M32d.1 — Generate HF FP16 reference logits for FALSIFY-QW3-MOE-FORWARD-004.

Loads HuggingFace Qwen3-Coder-30B-A3B-Instruct at FP16 (BF16 fallback if
FP16 is not exposed by the published config), runs ONE greedy decode step
on the canonical M32d prompt "What is 2+2?", and dumps the full 151936-dim
logit vector at position 0 (i.e. the logits-for-next-token-after-prompt
distribution) to a JSON fixture.

Usage:
    uv run --with torch --with transformers --with accelerate \\
        scripts/generate_qwen3_moe_fp16_logits.py \\
        --output crates/aprender-serve/tests/fixtures/qwen3_moe_fp16_logits_pos0.json

Resource requirements (operator-confirm before running):
    - Disk: ~60 GB for FP16 weights at HF cache (~/.cache/huggingface).
    - VRAM: 24 GB on RTX 4090 is INSUFFICIENT for a single-device
      FP16 30B-A3B load — the script uses `device_map="auto"` so
      `accelerate` will split params across GPU + CPU + disk.
      Expect ~10-30 minutes per forward pass with offload.
    - One-time cost: this fixture is captured once and committed
      verbatim; downstream tests (M32d.2) read the JSON.

Output schema (JSON):
    {
        "model_name":   "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "model_dtype":  "torch.float16" | "torch.bfloat16",
        "prompt":       "What is 2+2?",
        "tokens":       [<input_ids: int32>, ...],
        "vocab_size":   151936,
        "position":     <usize: position-after-prompt where logits are sampled>,
        "logits":       [<f32 array of length vocab_size>],
        "argmax_token": <int32: argmax of logits>,
        "argmax_text":  <str: tokenizer.decode([argmax_token])>,
        "git_sha":      <str: aprender HEAD SHA at fixture-generation time>,
        "transformers_version": <str>,
        "torch_version": <str>,
        "generated_utc": <ISO-8601 UTC timestamp>
    }

This script is a one-time fixture-generator. It does NOT validate
parity itself — that is M32d.2 (`qwen3_moe_parity.rs` integration test
loads this JSON and computes cosine similarity vs apr's CPU forward).
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
PROMPT = "What is 2+2?"
EXPECTED_VOCAB_SIZE = 151936


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("crates/aprender-serve/tests/fixtures/qwen3_moe_fp16_logits_pos0.json"),
        help="Path to write the JSON fixture (relative to repo root).",
    )
    p.add_argument(
        "--prompt",
        type=str,
        default=PROMPT,
        help="Prompt to encode + run forward on (default: %(default)r).",
    )
    p.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="HuggingFace model id (default: %(default)r).",
    )
    p.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="torch dtype to load weights at (default: %(default)s).",
    )
    return p.parse_args()


def get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent, text=True
        ).strip()
    except Exception as e:
        return f"<git rev-parse failed: {e}>"


def main() -> int:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print("=== M32d.1 fixture generation ===")
    print(f"  model:  {args.model}")
    print(f"  prompt: {args.prompt!r}")
    print(f"  dtype:  {dtype}")
    print(f"  output: {args.output}")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print("Loading model (this may take 10-30 minutes for the 30B-A3B FP16 load)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    print("Tokenizing prompt...")
    input_ids = tokenizer.encode(args.prompt, add_special_tokens=True, return_tensors="pt")
    tokens_list = input_ids[0].tolist()
    print(f"  tokens ({len(tokens_list)}): {tokens_list}")

    print("Running forward pass (greedy, no_grad)...")
    with torch.no_grad():
        outputs = model(input_ids=input_ids.to(model.device), use_cache=False)

    # logits shape: [batch, seq_len, vocab_size]; we want position seq_len-1 (predicts next token)
    logits_tensor = outputs.logits[0, -1, :].float().cpu()
    logits = logits_tensor.numpy().astype(np.float32)

    if logits.shape[0] != EXPECTED_VOCAB_SIZE:
        print(
            f"WARNING: vocab_size={logits.shape[0]} != expected {EXPECTED_VOCAB_SIZE}",
            file=sys.stderr,
        )

    argmax_token = int(np.argmax(logits))
    argmax_text = tokenizer.decode([argmax_token])
    l2 = float(np.linalg.norm(logits))

    print(f"  vocab_size:    {logits.shape[0]}")
    print(f"  argmax_token:  {argmax_token}")
    print(f"  argmax_text:   {argmax_text!r}")
    print(f"  ||logits||_2:  {l2:.4f}")
    print(f"  logits[:5]:    {logits[:5].tolist()}")
    print()

    fixture = {
        "model_name": args.model,
        "model_dtype": str(dtype),
        "prompt": args.prompt,
        "tokens": tokens_list,
        "vocab_size": int(logits.shape[0]),
        "position": len(tokens_list) - 1,
        "logits": logits.tolist(),
        "argmax_token": argmax_token,
        "argmax_text": argmax_text,
        "git_sha": get_git_sha(),
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(fixture, f)

    size_mib = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mib:.2f} MiB)")
    print("=== M32d.1 fixture generation complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
