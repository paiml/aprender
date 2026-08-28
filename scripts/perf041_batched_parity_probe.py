#!/usr/bin/env python3
"""FALSIFY-CB-006, made runnable: does batched decode produce the same output?

contracts/continuous-batching-v1.yaml states the obligation

    correctness_under_batching: |output_batched(r, c) - output_single(r, 1)| < eps
    invariants: "No garbage or empty outputs", "Token count matches"

and FALSIFY-CB-006 names the test as "probador correctness at c=1 and c=4,
compare token-level output" -- a sentence, not a command. Nothing executes it.

Exact whole-output equality is NOT a usable gate here: the solo path is not
byte-reproducible across requests (measured 286 vs 319 completion_tokens for
the same greedy prompt on consecutive calls, because the prefix cache changes
what prefill returns). What IS reproducible is the PREFIX, and this probe
establishes that in the same run rather than assuming it:

  1. two solo (m=1) references. If their prefixes disagree, the reference is
     not stable and the probe exits 2 -- UNMEASURABLE, not FAIL. A guard that
     reports a code defect for a box or model it cannot evaluate is the
     failure mode this repo has hit three times in one day.
  2. concurrent requests, with the server log inspected to PROVE a batch of
     m > 1 actually formed. Intent is not evidence: firing N requests does not
     mean a batch of N formed, and two staggered arrivals produce two m=1
     batches that never exercise the batched path at all.
  3. every batched output's prefix must equal the reference prefix.

Exit codes: 0 PASS, 1 FAIL (code defect), 2 UNMEASURABLE (env/model/harness).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import urllib.request

PREFIX_CHARS = 40
UNMEASURABLE = 2


def complete(url: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": "q",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        payload = json.load(resp)
    choice = payload["choices"][0]
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason"),
        "tokens": payload["usage"]["completion_tokens"],
    }


def fire_concurrent(url: str, prompt: str, max_tokens: int, n: int) -> list[dict]:
    out: dict[int, dict] = {}

    def go(i: int) -> None:
        out[i] = complete(url, prompt, max_tokens)

    threads = [threading.Thread(target=go, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [out[i] for i in sorted(out)]


def max_batch_formed(server_log: str) -> int:
    """Largest m in `[PMAT-044] Batch m=N done` -- proof the batched path ran."""
    try:
        with open(server_log, errors="replace") as fh:
            sizes = [int(m) for m in re.findall(r"Batch m=(\d+) done", fh.read())]
    except OSError:
        return 0
    return max(sizes) if sizes else 0


def check_reference(url: str, prompt: str, max_tokens: int) -> tuple[str, dict]:
    """Two solo calls; their shared prefix is the reference. Exits 2 if unstable."""
    a = complete(url, prompt, max_tokens)
    b = complete(url, prompt, max_tokens)
    pa, pb = a["text"][:PREFIX_CHARS], b["text"][:PREFIX_CHARS]
    print(f"  ref#1 finish={a['finish_reason']} tokens={a['tokens']} prefix={pa!r}")
    print(f"  ref#2 finish={b['finish_reason']} tokens={b['tokens']} prefix={pb!r}")
    if pa != pb:
        print("UNMEASURABLE: the m=1 reference is not prefix-stable on this "
              "box/model, so a batched difference cannot be attributed to "
              "batching. Not a code verdict.")
        sys.exit(UNMEASURABLE)
    if not pa.strip():
        print("UNMEASURABLE: the m=1 reference is empty.")
        sys.exit(UNMEASURABLE)
    return pa, a


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--server-log", required=True)
    ap.add_argument("--prompt", default="Write an essay on compilers.")
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[2, 4])
    args = ap.parse_args()

    print("== m=1 reference (positive control) ==")
    ref_prefix, ref = check_reference(args.url, args.prompt, args.max_tokens)

    failures = 0
    for c in args.concurrency:
        print(f"== concurrency {c} ==")
        results = fire_concurrent(args.url, args.prompt, args.max_tokens, c)
        formed = max_batch_formed(args.server_log)
        if formed < 2:
            print(f"  UNMEASURABLE at c={c}: no batch with m>1 ever formed "
                  f"(max m={formed}); the batched path was not exercised.")
            continue
        print(f"  batched path PROVEN engaged: max batch m={formed}")
        for i, r in enumerate(results):
            got = r["text"][:PREFIX_CHARS]
            ok = got == ref_prefix
            print(f"  [{i}] finish={r['finish_reason']} tokens={r['tokens']} "
                  f"{'OK ' if ok else 'DIVERGED'} prefix={got!r}")
            if not ok:
                failures += 1

    print()
    if failures:
        print(f"FALSIFY-CB-006 RED: {failures} batched output(s) diverge from the "
              f"m=1 reference on identical greedy input.")
        print(f"  reference tokens={ref['tokens']} finish={ref['finish_reason']}")
        return 1
    print("FALSIFY-CB-006 GREEN: every batched output matches the m=1 reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
