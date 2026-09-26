#!/usr/bin/env python3
"""SRV-TIM-001 live falsifier: FALSIFY-SRV-TIM-001/002/003/005/006.

Starts `apr serve run <model> --timings-log <out>/timings.jsonl`, sends chat and
completions requests (stream and non-stream, 1 and 32 tokens), then checks
every response against contracts/apr-serve-timings-v1.yaml:

  F1  the response (or the SSE terminal chunk) carries a non-null `timings`
  F2  timings.prompt_n / predicted_n == usage.prompt_tokens / completion_tokens
  F3  prompt_ms > 0, predicted_ms > 0, prompt_ms + predicted_ms <= wall_ms + 1
  F5  the server's `[request] {json}` stderr line for that request id carries
      the same prefill_ms / decode_ms / prompt_n / predicted_n
  F6  the --timings-log JSONL line for that request id carries the same values,
      plus `build` and `host`

Exit 0 iff every cell is GREEN. A cell is never skipped: a missing line is RED.

    python3 scripts/serve_timings_falsify.py --apr "$APR" --model M.gguf \
        --out /tmp/srvtim -- --gpu-layers all --context-length 36864
    # GPU hosts: --prefix "flock /tmp/apr-gpu.priority flock /tmp/apr-gpu.lock"
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request

LOG_PREFIX = "[request] "
CHAT_PROMPT = "Explain in two sentences why the sky is blue."
COMPLETION_PROMPT = "The sky is blue because"


def post(port, route, body):
    """POST `body`; return (raw text, wall ms)."""
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{route}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        raw = resp.read().decode()
    return raw, (time.monotonic() - t0) * 1000.0


def parse(raw):
    """(id, timings, usage) from a JSON body or an SSE stream's chunks."""
    if raw.lstrip().startswith("{"):
        o = json.loads(raw)
        return o.get("id"), o.get("timings"), o.get("usage")
    rid = timings = usage = None
    for line in raw.splitlines():
        if not line.startswith("data: ") or line[6:].strip() == "[DONE]":
            continue
        o = json.loads(line[6:])
        rid = rid or o.get("id")
        timings = o.get("timings") or timings
        usage = o.get("usage") or usage
    return rid, timings, usage


def records(lines, strip_prefix):
    """request_id -> record, from `[request] ` stderr lines or JSONL lines."""
    out = {}
    for line in lines:
        if strip_prefix:
            if not line.startswith(LOG_PREFIX):
                continue
            line = line[len(LOG_PREFIX):]
        line = line.strip()
        if line:
            rec = json.loads(line)
            out[rec.get("request_id")] = rec
    return out


def same(rec, t):
    return rec is not None and t is not None and (
        rec.get("prefill_ms") == t["prompt_ms"]
        and rec.get("decode_ms") == t["predicted_ms"]
        and rec.get("prompt_n") == t["prompt_n"]
        and rec.get("predicted_n") == t["predicted_n"]
    )


def cases():
    for tokens in (1, 32):
        for stream in (False, True):
            tag = f"{'s' if stream else 'ns'}{tokens}"
            yield f"chat-{tag}", "/v1/chat/completions", {
                "model": "default",
                "messages": [{"role": "user", "content": CHAT_PROMPT}],
                "temperature": 0,
                "seed": 4354,
                "max_tokens": tokens,
                "stream": stream,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            yield f"completions-{tag}", "/v1/completions", {
                "model": "default",
                "prompt": COMPLETION_PROMPT,
                "temperature": 0,
                "max_tokens": tokens,
                "stream": stream,
            }


def wait_healthy(port, proc, seconds):
    for _ in range(seconds):
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--apr", required=True, help="pinned apr binary ($APR)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="directory for logs and bodies")
    ap.add_argument("--port", type=int, default=18431)
    ap.add_argument("--prefix", default="", help="command prefix, e.g. flock ...")
    ap.add_argument("--health-seconds", type=int, default=600)
    ap.add_argument("serve_args", nargs="*", help="extra `apr serve run` args (after --)")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    tlog = os.path.join(a.out, "timings.jsonl")
    if os.path.exists(tlog):
        os.remove(tlog)
    slog = os.path.join(a.out, "serve.log")
    cmd = shlex.split(a.prefix) + [
        a.apr, "serve", "run", a.model, "--port", str(a.port),
        "--timings-log", tlog, *a.serve_args,
    ]
    with open(slog, "w") as err:
        proc = subprocess.Popen(cmd, stdout=err, stderr=subprocess.STDOUT,
                                start_new_session=True)
    try:
        if not wait_healthy(a.port, proc, a.health_seconds):
            print(f"RED: server never became healthy; see {slog}")
            return 1
        results = []
        for name, route, body in cases():
            raw, wall = post(a.port, route, body)
            with open(os.path.join(a.out, f"{name}.resp"), "w") as f:
                f.write(raw)
            results.append((name, wall) + parse(raw))
        time.sleep(0.5)  # the terminal-chunk emit precedes the client's EOF
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=60)

    with open(slog) as f:
        logged = records(f, strip_prefix=True)
    with open(tlog) if os.path.exists(tlog) else open(os.devnull) as f:
        filed = records(f, strip_prefix=False)

    red = 0
    print(f"{'case':<20} {'F1':<5} {'F2':<5} {'F3':<5} {'F5':<5} {'F6':<5} prompt_ms/predicted_ms/wall_ms")
    for name, wall, rid, t, u in results:
        f1 = t is not None
        f2 = f1 and u is not None and (t["prompt_n"], t["predicted_n"]) == (
            u["prompt_tokens"], u["completion_tokens"])
        f3 = f1 and t["prompt_ms"] > 0 and t["predicted_ms"] > 0 and (
            t["prompt_ms"] + t["predicted_ms"] <= wall + 1)
        f5 = same(logged.get(rid), t)
        frec = filed.get(rid)
        f6 = same(frec, t) and bool(frec.get("build")) and bool(frec.get("host"))
        cells = [f1, f2, f3, f5, f6]
        red += cells.count(False)
        ms = "-" if not f1 else f"{t['prompt_ms']:.1f}/{t['predicted_ms']:.1f}/{wall:.1f}"
        print(f"{name:<20} " + " ".join(f"{'ok' if c else 'RED':<5}" for c in cells) + f" {ms}")
    print(f"{'GREEN' if red == 0 else 'RED'}: {red} red cell(s) over {len(results)} cases")
    return 0 if red == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
