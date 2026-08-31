#!/usr/bin/env python3
"""Generate the W1 homogeneous workload corpus (APR-PERF-GATE-001 v2.2 §4.3.1).

W1 is the blocking workload: one prompt shape at every band, mirroring
`APR-BENCH-RFC-001`'s `pp512`/`tg128` so the canonical benchmark and this gate
cross-validate.

WHAT §4.3.1 DETERMINES, and is encoded here:

  target_prompt_tokens = 512 (tolerance +/-8)   max_tokens = 128
  temperature = 0.0 (greedy)                    seed = 0
  context = 4096                                model = qwen2.5-coder-7b Q4_K_M

WHAT §4.3.1 DOES NOT DETERMINE, and is chosen here (each choice is recorded in
the `_meta` header record of the corpus itself, not only in this docstring):

  1. HOW MANY prompts. Chosen: 256. §4.4.2 requires `max(30, 8*c)` sampled
     requests plus `2*c` warmups, and §4.5's widest band is c=16, so a band
     consumes at most 128 + 32 = 160 requests. 256 lets every request in every
     band draw a distinct prompt with headroom for a future c=32.

  2. WHETHER THE PROMPTS ARE DISTINCT. Chosen: distinct. "Fixed corpus" in
     §4.3.1 constrains the corpus to be pinned, not to be one prompt repeated.
     A corpus of N identical prompts lets a server with prefix caching serve
     bands 2..c from cache and report a scaling_efficiency that measures the
     cache, not the scheduler -- Arm A's numerator would rise with c for a
     reason that has nothing to do with batching. That is the gate-that-cannot-
     fail class this epic exists to remove, so the prompts differ from their
     third token onward and share only the `// w1-` marker.

  3. THE CONTENT. Chosen: shuffled filler drawn from a fixed pool of short
     lowercase ASCII words. The model under test is a CODE model, so the pool
     is code-flavoured, but nothing downstream reads the text -- W1 measures
     throughput, never output quality, and no verdict is computed from what
     the prompts say.

  4. WHETHER 512 IS COUNTED BEFORE OR AFTER THE CHAT TEMPLATE. NOT decided
     here, because it is not this file's to decide: the corpus stores raw
     prompt text and the harness applies the template. Recorded in `_meta` as
     an open question so it is settled in the receipt (§4.4.6 `tokenization`),
     which is where token counting is already required to be declared.

THE 512 IS A TARGET, NOT A MEASUREMENT. There is no tokenizer in this script,
so `target_prompt_tokens` is exactly that -- a target. The +/-8 band of §4.3.1
is an assertion the HARNESS makes against the real tokenizer at measurement
time, and `_meta.token_count_verified` is `false` in this file to say so out
loud. Emitting an exact count from a word count here would be a number with the
shape of a measurement, which is the defect this epic is named after.

Deterministic: seeded PRNG, fixed ordering, no wall-clock. Re-running
reproduces the file byte-for-byte, which is what lets a receipt name the
corpus by sha256.
"""
import argparse
import hashlib
import json
import random

# §4.3.1 -- these four are the spec's, not this script's.
TARGET_PROMPT_TOKENS = 512
TOLERANCE_TOKENS = 8
MAX_TOKENS = 128
CONTEXT = 4096
SEED = 0

# Chosen (see docstring 1): 256 >= 8*16 sampled + 2*16 warmup, with headroom.
DEFAULT_COUNT = 256

# Chosen (see docstring 3). Short lowercase ASCII words, each of which a
# byte-level BPE of the Qwen2.5 family is overwhelmingly likely to emit as a
# single " word" token. "Overwhelmingly likely" is not "verified" -- see
# `token_count_verified` below.
WORD_POOL = (
    "let", "fn", "mut", "ref", "impl", "trait", "type", "enum", "match",
    "loop", "next", "iter", "map", "fold", "push", "pop", "len", "size",
    "node", "edge", "list", "tree", "hash", "key", "value", "index", "slot",
    "cache", "queue", "stack", "heap", "block", "chunk", "batch", "token",
    "byte", "word", "line", "file", "path", "name", "field", "table", "row",
    "col", "cell", "code", "data", "read", "write", "open", "close", "send",
    "recv", "wait", "lock", "free", "init", "start", "stop", "count", "sum",
)

# Chosen: 503 words of body, and this number is now MEASURED rather than
# reasoned. The header is `// w1-NNNN` plus a newline, which a byte-level BPE
# renders in 9 tokens for a four-digit id; every word in the pool above encodes
# as exactly one token, so the count is `body_words + 9` with no variance at
# all -- min == median == max across all 256 records.
#
# WHY IT MOVED FROM 496. 496 was reasoned, not measured ("roughly 8 tokens...
# lands near 504-512"), and the first thing able to measure it -- the section
# 4.4.6 client_tokenizer counter -- put it at 505 on every one of the 256
# prompts. 505 is INSIDE 512 +/- 8, by one token: the band's floor is 504, so
# the blocking workload sat 0.198% above the value at which it fails. It had
# been invisible because under `server_usage --stream` neither server emits a
# usage block, so every observation was `prompt_tokens = 0` and the band
# assertion failed as an instrumentation gap before it ever compared 505 to 504.
#
# 503 puts the count at exactly 512 -- the target, dead centre, 8 tokens of
# margin on each side instead of 1 and 15. Measured with the canonical Qwen2.5-
# Coder tokenizer.json, sha256 c0382117ea329cdf097041132f6d735924b697924d6f6fc
# 3945713e96ce87539, 7,031,645 bytes; pinned by
# `the_w1_corpus_counts_512_tokens_per_prompt`, which reads THIS file.
#
# The knob stays a knob. A tokenizer re-serialization that changes any word's
# encoding moves the count, and the remedy is unchanged: retune --body-words and
# regenerate. What changed is that a one-token perturbation no longer fails the
# gate.
DEFAULT_BODY_WORDS = 503


def build(count, body_words):
    """Build `count` distinct prompt records of `body_words` filler words each."""
    rng = random.Random(SEED)
    rows = []
    for i in range(count):
        words = [rng.choice(WORD_POOL) for _ in range(body_words)]
        rows.append({
            "id": i,
            "max_tokens": MAX_TOKENS,
            "prompt": f"// w1-{i:04d}\n" + " ".join(words),
            "seed": SEED,
            "target_prompt_tokens": TARGET_PROMPT_TOKENS,
            "temperature": 0.0,
        })
    return rows


def meta(count, body_words):
    """The `_meta` header record -- provenance, in the file, not beside it."""
    return {"_meta": {
        "corpus": "W1",
        "spec": "APR-PERF-GATE-001 v2.2 4.3.1",
        "generator": "scripts/gen_prompts_w1.py",
        "regenerate": (
            f"python3 scripts/gen_prompts_w1.py --count {count} "
            f"--body-words {body_words}"
        ),
        "provenance": (
            "SYNTHETIC. Generated by the script named above from a seeded PRNG "
            "over a fixed word pool. Not sampled from real traffic, not human "
            "written, not drawn from any dataset. Nothing downstream reads the "
            "prompt text -- W1 measures throughput only."
        ),
        "count": count,
        "body_words": body_words,
        "target_prompt_tokens": TARGET_PROMPT_TOKENS,
        "tolerance_tokens": TOLERANCE_TOKENS,
        "token_count_verified": False,
        "token_count_note": (
            "target_prompt_tokens is a TARGET. No tokenizer ran in this "
            "generator. The 512 +/-8 of 4.3.1 is asserted by the harness "
            "against the model's own tokenizer at measurement time; if it "
            "fails, retune --body-words and regenerate. body_words = 503 IS "
            "the outcome of doing exactly that: the first measurement, by the "
            "4.4.6 client_tokenizer counter against the canonical Qwen2.5-"
            "Coder tokenizer (sha256 c0382117...), put body_words = 496 at 505 "
            "tokens -- inside 512 +/- 8 by ONE token, 0.198% above the floor. "
            "503 measures 512 exactly on all 256 records, 8 tokens of margin "
            "each side. Pinned by "
            "the_w1_corpus_counts_512_tokens_per_prompt, which reads this file."
        ),
        "template_boundary_open": (
            "4.3.1 does not say whether prompt_tokens = 512 is counted before "
            "or after the chat template wrapper. This corpus stores RAW prompt "
            "text; the harness applies the template. The receipt's 4.4.6 "
            "tokenization block must declare which side of that boundary the "
            "count was taken on."
        ),
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "seed": SEED,
        "context": CONTEXT,
        "prompts_distinct": True,
        "distinctness_rationale": (
            "Identical prompts would let prefix caching, not the scheduler, "
            "drive Arm A's scaling_efficiency. 4.3.1 does not require "
            "distinctness; it is chosen here."
        ),
    }}


def render(rows, header):
    body = json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n"
    body += "".join(
        json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows
    )
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--body-words", type=int, default=DEFAULT_BODY_WORDS)
    ap.add_argument(
        "--out",
        default="crates/aprender-serve/benchmarks/qwen-coder/prompts-w1.jsonl",
    )
    ap.add_argument("--print-sha", action="store_true")
    args = ap.parse_args()

    rows = build(args.count, args.body_words)
    body = render(rows, meta(args.count, args.body_words))
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(body)

    distinct = len({r["prompt"] for r in rows})
    print(f"wrote {args.out}: {len(rows)} rows + 1 _meta header, context={CONTEXT}")
    print(f"distinct prompts: {distinct}/{len(rows)}")
    print(f"target_prompt_tokens: {TARGET_PROMPT_TOKENS} +/-{TOLERANCE_TOKENS} (UNVERIFIED)")
    if args.print_sha:
        print("sha256:", hashlib.sha256(body.encode()).hexdigest())


if __name__ == "__main__":
    main()
