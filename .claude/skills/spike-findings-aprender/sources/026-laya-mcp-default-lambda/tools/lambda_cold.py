"""Forced cold starts on a Laya Lambda (env nonce -> fresh environment), then warm calls; one JSON line per call.
  uv run --with boto3 python tools/lambda_cold.py laya-spike-10g --rounds 3 --out results/lambda-10g.jsonl
"""
import argparse, json, subprocess, sys, time
import boto3
ap = argparse.ArgumentParser(); ap.add_argument("function"); ap.add_argument("--rounds", type=int, default=3); ap.add_argument("--out", required=True)
a = ap.parse_args()
lam = boto3.client("lambda", region_name="us-east-1")
def bump():
    env = lam.get_function_configuration(FunctionName=a.function)["Environment"]["Variables"]; env["COLD_NONCE"] = str(time.time())
    lam.update_function_configuration(FunctionName=a.function, Environment={"Variables": env})
    lam.get_waiter("function_updated_v2").wait(FunctionName=a.function)
def probe(label, records, warm):
    out = subprocess.run([sys.executable, "tools/probe.py", "lambda", a.function, "--records", records, "--warm", str(warm), "--label", label], capture_output=True, text=True)
    if out.returncode: print(out.stderr[-2000:], file=sys.stderr)
    return [json.loads(l) for l in out.stdout.splitlines() if l.strip()]
with open(a.out, "a") as f:
    for r in range(a.rounds):
        bump()
        for label, recs, warm in [(f"cold-{r}", "1", 0), (f"warm-{r}", "1", 5), (f"warm-ticket-{r}", "0", 3), (f"warm-512-{r}", "11", 2)]:
            for rec in probe(label, recs, warm):
                rec["function"] = a.function; f.write(json.dumps(rec) + "\n"); f.flush()
                s = rec["server"]; tl = {x["step"]: x for x in s.get("load_timeline", [])}
                print(label, "wall", rec["client_wall_ms"], "init", rec.get("init_ms"), "mem", rec.get("max_mem_mb"), "req", s.get("request_ms") and round(s["request_ms"]),
                      "warm_p50", s.get("warm_p50_ms") and round(s["warm_p50_ms"]), "dp", rec.get("max_abs_dp"), s.get("host", {}).get("cpu_guess"),
                      "dl", (tl.get("s3 download") or {}).get("detail"), "ready_ms", (tl.get("engine ready") or {}).get("t_ms") and round(tl["engine ready"]["t_ms"]),
                      "err", s.get("function_error") or s.get("tool_error") or s.get("mcp_error"), flush=True)
