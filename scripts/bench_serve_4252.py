#!/usr/bin/env python3
"""#4252 3-way serve bench: one running `apr serve` endpoint, n reps, non-streaming.

`/v1/completions` stream=true is buffered (#4272): every chunk arrives after the
generation ends, so a streaming TTFT is the full wall. This client uses the two-request
method of the #4221 spike instead:

  * TTFT     = wall of a completion with max_tokens=1; prefill tok/s = prompt_tokens / TTFT
  * decode   = (completion_tokens(N) - 1) / (T(N) - T(1)) tok/s

Every response must carry `used_gpu`; the value is recorded per row, and `--expect-gpu`
refuses a row that says false (#4089: `--backend cuda` alone ran on CPU and said nothing).
The server's /health, /v1/effective-config and /v1/gpu/status are captured once as the
provenance envelope.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request


def get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read())
    except Exception as e:  # recorded, not fatal: a missing route is itself a finding
        return {"error": repr(e)}


def complete(base, prompt, max_tokens, timeout):
    body = {"model": "default", "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.0}
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    dt = time.monotonic() - t0
    u = d.get("usage") or {}
    return dt, d["choices"][0]["text"], u.get("prompt_tokens"), u.get("completion_tokens"), \
        d.get("used_gpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--prompt-file", action="append", required=True)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--label", required=True)
    ap.add_argument("--bin-sha256", default="", help="sha256 of the serving binary")
    ap.add_argument("--expect-gpu", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = a.url.rstrip("/")
    rec = {"label": a.label, "host": os.uname().nodename, "url": base,
           "bin_sha256": a.bin_sha256, "max_tokens": a.max_tokens, "reps": a.reps,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "health": get(base + "/health"),
           "effective_config": get(base + "/v1/effective-config"),
           "gpu_status": get(base + "/v1/gpu/status"),
           "rows": [], "summary": {}}
    complete(base, "Hello", 4, a.timeout)  # warm-up, not recorded
    for pf in a.prompt_file:
        prompt = open(pf).read()
        psha = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        for rep in range(a.reps):
            t1, _, ptok, _, g1 = complete(base, prompt, 1, a.timeout)
            tn, text, _, ctok, gn = complete(base, prompt, a.max_tokens, a.timeout)
            row = {"prompt": os.path.basename(pf), "prompt_sha16": psha, "rep": rep,
                   "prompt_tokens": ptok, "ttft_s": t1,
                   "prefill_tok_s": ptok / t1 if ptok else None,
                   "tN_s": tn, "completion_tokens": ctok,
                   "decode_tok_s": ((ctok or 1) - 1) / (tn - t1) if tn > t1 else None,
                   "used_gpu": [g1, gn], "text": text}
            if a.expect_gpu and not (g1 is True and gn is True):
                row["void"] = f"--expect-gpu but used_gpu={[g1, gn]}"
            rec["rows"].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)
            json.dump(rec, open(a.out, "w"), indent=1)
        good = [r for r in rec["rows"] if r["prompt"] == os.path.basename(pf)
                and "void" not in r]
        if good:
            rec["summary"][os.path.basename(pf)] = {
                k: statistics.median(r[k] for r in good if r[k] is not None)
                for k in ("ttft_s", "prefill_tok_s", "decode_tok_s")}
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rec, open(a.out, "w"), indent=1)
    print(json.dumps(rec["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
