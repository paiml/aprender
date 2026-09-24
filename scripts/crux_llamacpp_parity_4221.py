#!/usr/bin/env python3
"""#4221 CRUX spike: apr serve vs llama.cpp llama-server on the SAME GGUF, same host.

One engine x one model per invocation (one output file per engine per model). Run it
under gpu-q so the whole session holds /tmp/apr-gpu.lock:

    gpu-q -- python3 scripts/crux_llamacpp_parity_4221.py --engine apr --bin <apr> \
        --model ~/models/Qwen3.5-4B-Q4_K_M.gguf --out <dir>

Method (identical client for both engines):
  * TTFT = wall time of a non-streaming completion with max_tokens=1; prefill tok/s =
    prompt_tokens / TTFT (so it includes one sampled token and HTTP overhead).
  * decode tok/s = 127 / (T(max_tokens=128) - T(max_tokens=1)) on the 850-token prompt.
  * temperature 0, prompt caching OFF (llama-server `cache_prompt: false`; apr has none,
    #4214) so every repetition pays its full prefill.
  * correctness: the 128-token greedy text is saved per repetition; the comparison across
    engines is done by the caller from the two output files.
"""

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

PROMPTS = os.path.expanduser("~/.cache/lqw/prompts")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def post(url, body, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return time.monotonic() - t0, data


def foreign_gpu_apps():
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def start_server(args, log):
    ctx = str(args.ctx)
    if args.engine == "apr":
        cmd = [args.bin, "serve", "run", args.model, "--gpu", "--port", str(args.port),
               "--context-length", ctx, "--trace"]
    else:
        cmd = [args.bin, "-m", args.model, "--port", str(args.port), "-ngl", "999",
               "-c", ctx, "--temp", "0", "--parallel", "1", "--no-warmup"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited rc={proc.returncode} before ready: {cmd}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                if r.status == 200:
                    return proc, cmd
        except Exception:
            pass
        time.sleep(1)
    raise SystemExit("server not ready in 600 s")


def complete(args, prompt, max_tokens, timeout):
    base = f"http://127.0.0.1:{args.port}"
    if args.engine == "apr":
        body = {"model": "default", "prompt": prompt, "max_tokens": max_tokens,
                "temperature": 0.0}
        dt, d = post(base + "/v1/completions", body, timeout)
        text = d["choices"][0]["text"]
        usage = d.get("usage", {})
        return dt, text, usage.get("prompt_tokens"), usage.get("completion_tokens"), None
    body = {"prompt": prompt, "n_predict": max_tokens, "temperature": 0.0,
            "cache_prompt": False, "top_k": 1}
    dt, d = post(base + "/completion", body, timeout)
    return dt, d["content"], d.get("tokens_evaluated"), d.get("tokens_predicted"), d.get("timings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["apr", "llama"], required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--port", type=int, default=18421)
    ap.add_argument("--ctx", type=int, default=34816)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--sizes", default="850,4k,32k")
    ap.add_argument("--reps-32k", type=int, default=None,
                    help="repetitions at 32k (apr's 1-token prefill makes n=3 cost ~20 min/model)")
    ap.add_argument("--timeout", type=int, default=1500)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    stem = f"{args.engine}-{os.path.basename(args.model).removesuffix('.gguf')}"
    res_path = os.path.join(args.out, stem + ".json")
    log_path = os.path.join(args.out, stem + ".server.log")

    foreign = foreign_gpu_apps()
    rec = {"engine": args.engine, "label": args.label, "bin": os.path.realpath(args.bin),
           "bin_sha256": sha256(args.bin), "model": os.path.realpath(args.model),
           "gguf_sha256": sha256(args.model),
           "gpu": subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                                  "--format=csv,noheader"], capture_output=True,
                                 text=True).stdout.strip(),
           "host": os.uname().nodename, "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "foreign_gpu_apps_at_start": foreign, "rows": [], "gen": []}
    ver = subprocess.run([args.bin, "--version"], capture_output=True, text=True)
    rec["version"] = (ver.stdout + ver.stderr).strip().splitlines()[-2:]
    if foreign:
        rec["void"] = "foreign GPU process present at start; not measured"
        json.dump(rec, open(res_path, "w"), indent=1)
        print(f"VOID {stem}: {foreign}", file=sys.stderr)
        return 3

    with open(log_path, "w") as log:
        proc, cmd = start_server(args, log)
        rec["server_cmd"] = cmd
        try:
            complete(args, "Hello", 8, 300)  # warm-up, not recorded
            prompts = {s: open(os.path.join(PROMPTS, f"p{s}.txt")).read()
                       for s in args.sizes.split(",")}
            for size, prompt in prompts.items():
                reps = args.reps_32k if (size == "32k" and args.reps_32k) else args.reps
                for rep in range(reps):
                    dt1, _, ptok, _, tim = complete(args, prompt, 1, args.timeout)
                    row = {"size": size, "rep": rep, "ttft_s": dt1, "prompt_tokens": ptok,
                           "prefill_tok_s": (ptok / dt1) if ptok else None,
                           "server_timings": tim}
                    if size == "850":
                        dt128, text, _, ctok, tim128 = complete(args, prompt, 128, args.timeout)
                        row.update({"t128_s": dt128, "completion_tokens": ctok,
                                    "decode_tok_s": ((ctok or 128) - 1) / (dt128 - dt1),
                                    "server_timings_128": tim128})
                        rec["gen"].append(text)
                    rec["rows"].append(row)
                    print(json.dumps({k: row[k] for k in row if not k.startswith("server")}),
                          flush=True)
                    json.dump(rec, open(res_path, "w"), indent=1)
        finally:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    json.dump(rec, open(res_path, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
