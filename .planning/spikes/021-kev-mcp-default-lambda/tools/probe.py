"""MCP client probe for the Kev `decide` server on any host; prints one JSON line per call.

  probe.py http  <url>        --row 0 --warm 5      # local / Fargate: streamable-HTTP, stateless JSON
  probe.py lambda <function>  --row 0 --warm 5      # default Lambda / LMI: Invoke with an API-GW v2 event

Client-side wall time is measured around the whole call. For Lambda, the REPORT log tail supplies
Init Duration (present only on a cold environment), Duration, Billed and Max Memory Used.
"""
import argparse, base64, json, re, sys, time, urllib.request

def rpc(method, params, id_=1):
    return {"jsonrpc": "2.0", "id": id_, "method": method, "params": params}

CALL = lambda row, warm: rpc("tools/call", {"name": "decide", "arguments": {"row": row, "warm_rounds": warm}})
HEADERS = {"content-type": "application/json", "accept": "application/json, text/event-stream",
           "mcp-protocol-version": "2025-06-18"}

def parse_mcp(body: str):
    msg = json.loads(body)
    if "error" in msg:
        return {"mcp_error": msg["error"]}
    res = msg["result"]
    if res.get("isError"):
        return {"tool_error": res["content"]}
    text = res["content"][0]["text"]
    return json.loads(text)

def via_http(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=HEADERS, method="POST")
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        body = r.read().decode()
    return (time.perf_counter() - t) * 1e3, body, {}

def via_lambda(fn, payload, region):
    import boto3
    from botocore.config import Config
    lam = boto3.client("lambda", region_name=region, config=Config(read_timeout=900, retries={"max_attempts": 0}))
    event = {"version": "2.0", "routeKey": "$default", "rawPath": "/", "rawQueryString": "",
             "headers": HEADERS, "isBase64Encoded": False, "body": json.dumps(payload),
             "requestContext": {"http": {"method": "POST", "path": "/", "protocol": "HTTP/1.1", "sourceIp": "127.0.0.1",
                                         "userAgent": "probe"}, "routeKey": "$default", "stage": "$default",
                                "requestId": "probe", "accountId": "x", "apiId": "x", "domainName": "x", "domainPrefix": "x",
                                "time": "x", "timeEpoch": 0}}
    t = time.perf_counter()
    kw = {} if ":" in fn and fn.startswith("kev-spike-lmi") else {"LogType": "Tail"}  # LMI: "Tail logs are not supported"
    out = lam.invoke(FunctionName=fn, Payload=json.dumps(event).encode(), **kw)
    wall = (time.perf_counter() - t) * 1e3
    log = base64.b64decode(out.get("LogResult", "")).decode(errors="replace")
    rep = {}
    for k, pat in [("init_ms", r"Init Duration: ([\d.]+) ms"), ("duration_ms", r"\tDuration: ([\d.]+) ms"),
                   ("billed_ms", r"Billed Duration: ([\d.]+) ms"), ("max_mem_mb", r"Max Memory Used: (\d+) MB")]:
        m = re.search(pat, log)
        if m: rep[k] = float(m.group(1))
    payload_out = json.loads(out["Payload"].read())
    if out.get("FunctionError"):
        return wall, None, {**rep, "function_error": payload_out, "log_tail": log[-1500:]}
    body = payload_out.get("body", "")
    if payload_out.get("isBase64Encoded"):
        body = base64.b64decode(body).decode()
    return wall, body, rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["http", "lambda"]); ap.add_argument("target")
    ap.add_argument("--row", type=int, default=0); ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--region", default="us-east-1"); ap.add_argument("--label", default="")
    a = ap.parse_args()
    payload = CALL(a.row, a.warm)
    wall, body, rep = via_http(a.target, payload) if a.mode == "http" else via_lambda(a.target, payload, a.region)
    rec = {"label": a.label, "client_wall_ms": round(wall, 1), **rep, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if body is not None:
        rec["server"] = parse_mcp(body)
    print(json.dumps(rec))

if __name__ == "__main__":
    main()
