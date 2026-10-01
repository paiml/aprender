"""Forced cold starts on a Lambda: bump an env nonce (new environments), invoke cold, then warm.

  uv run --with boto3 python tools/lambda_cold.py kev-spike-lambda-s3 --rounds 3 --out results/lambda-s3.jsonl
"""
import argparse, json, subprocess, sys, time
import boto3

ap = argparse.ArgumentParser()
ap.add_argument("function"); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--out", required=True)
ap.add_argument("--row", type=int, default=1); ap.add_argument("--region", default="us-east-1")
ap.add_argument("--no-bump", action="store_true", help="skip the env bump on the first round (already cold)")
a = ap.parse_args()
lam = boto3.client("lambda", region_name=a.region)

def bump():
    cfg = lam.get_function_configuration(FunctionName=a.function)
    env = cfg.get("Environment", {}).get("Variables", {})
    env["COLD_NONCE"] = str(time.time())
    lam.update_function_configuration(FunctionName=a.function, Environment={"Variables": env})
    lam.get_waiter("function_updated_v2").wait(FunctionName=a.function)

def probe(label, warm):
    out = subprocess.run([sys.executable, "tools/probe.py", "lambda", a.function, "--row", str(a.row), "--warm", str(warm),
                          "--region", a.region, "--label", label], capture_output=True, text=True)
    if out.returncode:
        print(out.stderr, file=sys.stderr)
    return json.loads(out.stdout)

with open(a.out, "a") as f:
    for r in range(a.rounds):
        if r or not a.no_bump:
            bump()
        for label, warm in [(f"cold-{r}", 0), (f"warm-{r}", 5)]:
            rec = probe(label, warm); rec["function"] = a.function
            f.write(json.dumps(rec) + "\n"); f.flush()
            s = rec.get("server", {})
            print(label, "wall", rec["client_wall_ms"], "init", rec.get("init_ms"), "dur", rec.get("duration_ms"), "mem", rec.get("max_mem_mb"),
                  "decision", s.get("decision_ms"), "warm_p50", s.get("warm_p50_ms"), "dp", s.get("max_abs_dp"),
                  "cpu", s.get("host", {}).get("cpu_guess"), "err", rec.get("function_error") or s.get("tool_error") or s.get("mcp_error"), flush=True)
