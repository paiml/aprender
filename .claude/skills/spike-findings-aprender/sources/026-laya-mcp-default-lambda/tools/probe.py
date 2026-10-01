"""MCP client for the Laya `decide` server: sends spike-025 fixture requests (state + typed questions) through the real
text API and checks the probabilities against Laya's own torch fp32 path. One JSON line per call.

  probe.py http   http://127.0.0.1:8792/   --records all --warm 3
  probe.py lambda laya-spike-lambda-s3     --records 1   --warm 5      # default Lambda (API-GW v2 event)
"""
import argparse, base64, json, re, sys, time, urllib.request
from pathlib import Path

FX = json.loads((Path(__file__).resolve().parents[2] / "025-laya-rust-forward-parity/fixtures/laya-en_fixture.json").read_text())
HEADERS = {"content-type": "application/json", "accept": "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18"}


def call(mode, target, args, region):
    payload = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "decide", "arguments": args}}
    rep = {}
    t = time.perf_counter()
    if mode == "http":
        req = urllib.request.Request(target, data=json.dumps(payload).encode(), headers=HEADERS, method="POST")
        with urllib.request.urlopen(req, timeout=900) as r:
            body = r.read().decode()
        wall = (time.perf_counter() - t) * 1e3
    else:
        import boto3
        from botocore.config import Config
        lam = boto3.client("lambda", region_name=region, config=Config(read_timeout=900, retries={"max_attempts": 0}))
        ev = {"version": "2.0", "routeKey": "$default", "rawPath": "/", "rawQueryString": "", "headers": HEADERS,
              "isBase64Encoded": False, "body": json.dumps(payload),
              "requestContext": {"http": {"method": "POST", "path": "/", "protocol": "HTTP/1.1", "sourceIp": "127.0.0.1", "userAgent": "probe"},
                                 "routeKey": "$default", "stage": "$default", "requestId": "probe", "accountId": "x", "apiId": "x",
                                 "domainName": "x", "domainPrefix": "x", "time": "x", "timeEpoch": 0}}
        out = lam.invoke(FunctionName=target, Payload=json.dumps(ev).encode(), LogType="Tail")
        wall = (time.perf_counter() - t) * 1e3
        log = base64.b64decode(out.get("LogResult", "")).decode(errors="replace")
        for k, pat in [("init_ms", r"Init Duration: ([\d.]+) ms"), ("duration_ms", r"\tDuration: ([\d.]+) ms"),
                       ("max_mem_mb", r"Max Memory Used: (\d+) MB")]:
            m = re.search(pat, log)
            if m: rep[k] = float(m.group(1))
        po = json.loads(out["Payload"].read())
        if out.get("FunctionError"):
            return wall, {"function_error": po, "log": log[-1200:]}, rep
        body = base64.b64decode(po["body"]).decode() if po.get("isBase64Encoded") else po.get("body", "")
    msg = json.loads(body)
    if "error" in msg:
        return wall, {"mcp_error": msg["error"]}, rep
    res = msg["result"]
    if res.get("isError"):
        return wall, {"tool_error": res["content"]}, rep
    return wall, json.loads(res["content"][0]["text"]), rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["http", "lambda"]); ap.add_argument("target")
    ap.add_argument("--records", default="1", help='"all" or comma list of fixture record indices')
    ap.add_argument("--warm", type=int, default=0); ap.add_argument("--region", default="us-east-1"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    idx = range(len(FX["records"])) if a.records == "all" else [int(x) for x in a.records.split(",")]
    for i in idx:
        rec = FX["records"][i]
        wall, s, rep = call(a.mode, a.target, {"state": rec["state"], "questions": rec["questions"], "warm_rounds": a.warm}, a.region)
        out = {"label": a.label, "record": i, "client_wall_ms": round(wall, 1), **rep}
        if "answers" in s:
            want = {r["qid"]: r["probs"] for r in rec["rows"]}
            dps = [max(abs(x - y) for x, y in zip(ans["probabilities"], want[ans["id"]])) for ans in s["answers"]]
            same = [max(range(len(ans["probabilities"])), key=ans["probabilities"].__getitem__) ==
                    max(range(len(want[ans["id"]])), key=want[ans["id"]].__getitem__) for ans in s["answers"]]
            out.update({"max_abs_dp": max(dps), "argmax_agree": f"{sum(same)}/{len(same)}", "tokens": [ans["tokens"] for ans in s["answers"]]})
        out["server"] = s
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
