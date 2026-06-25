#!/usr/bin/env -S uv run --with numpy --with safetensors python3
"""Bridge: remap an entrenar InstructTrainer LoRA checkpoint (model.safetensors)
into the tensor-naming + metadata that `apr finetune --merge` expects.

The trainer writes tensors named:
    lora.{layer}.q_proj.lora_a / .lora_b
    lora.{layer}.v_proj.lora_a / .lora_b
but `apr finetune --merge` looks for `{base_tensor_name}.lora_a/.lora_b` where the
base (Qwen2 q4k APR / GGUF) tensor names are:
    blk.{layer}.attn_q.weight
    blk.{layer}.attn_v.weight
It also reads lora_rank/lora_alpha from the safetensors header metadata (the
trainer omits them, so merge silently defaults to rank=64/alpha=16). This script
rewrites both so the merge applies the *actually trained* delta at the right rank.

Usage:
    remap_adapter.py IN.safetensors OUT.safetensors --rank 16 --alpha 32
"""
import argparse
import json
import sys

import numpy as np
from safetensors.numpy import load_file, save_file


def remap_name(name: str) -> str | None:
    # lora.{L}.{q|v}_proj.lora_{a|b} -> blk.{L}.attn_{q|v}.weight.lora_{a|b}
    if not name.startswith("lora."):
        return None
    parts = name.split(".")
    # ["lora", "{L}", "{q|v}_proj", "lora_{a|b}"]
    if len(parts) != 4:
        return None
    _, layer, proj, ab = parts
    proj_short = proj.replace("_proj", "")  # q / v
    if proj_short not in ("q", "v"):
        return None
    return f"blk.{layer}.attn_{proj_short}.weight.{ab}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--alpha", type=float, required=True)
    args = ap.parse_args()

    tensors = load_file(args.inp)
    out = {}
    remapped = 0
    for name, arr in tensors.items():
        new = remap_name(name)
        if new is None:
            print(f"  skip (unmapped): {name}", file=sys.stderr)
            continue
        out[new] = np.ascontiguousarray(arr.astype(np.float32))
        remapped += 1

    if not out:
        print("ERROR: no tensors remapped — wrong input format?", file=sys.stderr)
        return 1

    meta = {"lora_rank": str(args.rank), "lora_alpha": str(args.alpha)}
    save_file(out, args.out, metadata=meta)
    print(
        f"remapped {remapped} tensors -> {args.out} (rank={args.rank}, alpha={args.alpha})",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
