#!/usr/bin/env python3
"""Generate the W2 ragged workload corpus (APR-PERF-GATE-001 v2.2 §4.3.2).

W1 is a single shape at every band. Real serving is variable-length, and the two
properties that matter most are invisible at a fixed shape: static batching pads
to the longest sequence, and a long prefill injected into an active batch stalls
every concurrent decode.

THE MIXTURE IS A SHAPE PARAMETER, NOT A THRESHOLD. The spec marks the weights
`[U]` -- chosen to span the range, not derived. They describe the input, so no
verdict may be computed from them and none is.

Deterministic: seeded PRNG, fixed ordering, no wall-clock. Re-running reproduces
the file byte-for-byte, which is what lets a receipt name the corpus by sha256.
"""
import argparse
import hashlib
import json
import random

# §4.3.2: 40% at 128, 30% at 512, 20% at 2048, 10% at 8192 tokens.
PROMPT_MIX = ((128, 40), (512, 30), (2048, 20), (8192, 10))
# §4.3.2: 40% max_tokens=16, 40% 128, 20% 512.
GEN_MIX = ((16, 40), (128, 40), (512, 20))
CONTEXT = 16384
SEED = 0
# One-token filler. The corpus records a TARGET length; the harness records the
# ACTUAL tokenized length in the receipt. Claiming an exact count here without a
# tokenizer in the loop would be a number with the shape of a measurement.
FILLER = "code "


def _expand(mix, total):
    out = []
    for value, pct in mix:
        out.extend([value] * (total * pct // 100))
    while len(out) < total:
        out.append(mix[0][0])
    return out[:total]


def build(count):
    rng = random.Random(SEED)
    prompts = _expand(PROMPT_MIX, count)
    gens = _expand(GEN_MIX, count)
    rng.shuffle(prompts)
    rng.shuffle(gens)
    rows = []
    for i, (ptok, gtok) in enumerate(zip(prompts, gens)):
        rows.append({
            "id": i,
            "target_prompt_tokens": ptok,
            "max_tokens": gtok,
            "temperature": 0.0,
            "seed": SEED,
            "prompt": ("// qwen-coder W2 ragged corpus\n" + FILLER * ptok).rstrip(),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--out", default="crates/aprender-serve/benchmarks/qwen-coder/prompts-w2.jsonl")
    ap.add_argument("--print-sha", action="store_true")
    args = ap.parse_args()
    rows = build(args.count)
    body = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body)
    dist = {}
    for r in rows:
        dist[r["target_prompt_tokens"]] = dist.get(r["target_prompt_tokens"], 0) + 1
    print(f"wrote {args.out}: {len(rows)} rows, context={CONTEXT}")
    print("prompt-length distribution:", dict(sorted(dist.items())))
    if args.print_sha:
        print("sha256:", hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
