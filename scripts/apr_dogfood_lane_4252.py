#!/usr/bin/env python3
"""#4252 mandatory apr dogfood: the advisory quorum lane's serve fleet.

PRIMARY  gx10, `apr serve` built with cuda, Qwen3.5-4B.
FALLBACK lambda, the CPU release asset, inside the capped user slice
         apr-dogfood.slice (Nice 19, CPUQuota 800%, MemoryMax 12G).

Operator, verbatim (via the cop, 2026-09-24): "perfect, if box is needed, it pauses and
passes work to lambda cpu". So the gx10 server YIELDS to any need for the box: a gpu-q
hold or queued ticket, a foreign GPU process (a CI GPU job), another session's claim file,
or a disk/memory alarm. Its work, including a request in flight, passes to lambda CPU.

Subcommands
  watch   (runs ON gx10)  every --period s: if the box is needed, stop the server by its
                          recorded pid; if free, (re)start it. State: STATE_DIR/state.json.
  ask     (runs on lambda) send one chat request: gx10 if its state says serving and
                          /health answers, else lambda CPU; a gx10 failure mid-request is
                          re-sent to lambda. Prints ONE JSON receipt.
  need    (runs ON gx10)  print the need signals (for receipts and the mutation proof).

The lane is ADVISORY: a late or missing answer never blocks a quorum (the rail's side).
Speed is not a gate here; the receipt records it.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

STATE_DIR = os.path.expanduser("~/.local/state/apr-dogfood-4252")
CLAIM = "/tmp/apr-gx10-claim"          # another session's claim: any content, any owner
LOCK = "/tmp/apr-gpu.lock"
QUEUE = "/tmp/apr-gpu-queue"
DF_FLOOR_GB = 60                        # yield above the 50G alarm so a build never races it
MEM_FLOOR_GB = 16
GX10_PORT = 18253
LAMBDA_URL = "http://127.0.0.1:18252"


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def load_state():
    try:
        with open(os.path.join(STATE_DIR, "state.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(st):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = os.path.join(STATE_DIR, "state.json.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(STATE_DIR, "state.json"))


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # the pid must still be OUR server, not a recycled number
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return b"serve" in f.read()
    except OSError:
        return False


def need_signals(own_pid):
    """Every reason the box is needed. Empty list = free. Fails CLOSED: a probe that
    cannot answer is itself a reason."""
    why = []
    # 1. gpu-q: a held lock or a queued ticket
    r = sh(["flock", "-n", LOCK, "true"])
    if r.returncode != 0:
        why.append(f"gpu lock held ({LOCK})")
    try:
        q = [n for n in os.listdir(QUEUE) if not n.startswith(".")]
        if q:
            why.append(f"gpu-q tickets queued: {sorted(q)[:3]}")
    except FileNotFoundError:
        pass
    # 2. any GPU process that is not our server (CI GPU jobs, other sessions)
    r = sh(["nvidia-smi", "--query-compute-apps=pid,process_name",
            "--format=csv,noheader"])
    if r.returncode != 0:
        why.append(f"nvidia-smi failed rc={r.returncode}")
    else:
        for line in r.stdout.splitlines():
            pid = line.split(",")[0].strip()
            if pid and pid != str(own_pid or ""):
                why.append(f"foreign GPU process: {line.strip()}")
    # 3. another session's claim
    if os.path.exists(CLAIM):
        try:
            why.append(f"claim file {CLAIM}: {open(CLAIM).read().strip()[:120]}")
        except OSError:
            why.append(f"claim file {CLAIM}")
    # 4. disk and memory alarms (gx10's RAID is its root disk)
    free_gb = shutil.disk_usage("/").free / 1e9
    if free_gb < DF_FLOOR_GB:
        why.append(f"disk / free {free_gb:.0f}G < {DF_FLOOR_GB}G")
    try:
        with open("/proc/meminfo") as f:
            avail = next(int(l.split()[1]) for l in f if l.startswith("MemAvailable"))
        if avail / 1e6 < MEM_FLOOR_GB:
            why.append(f"MemAvailable {avail / 1e6:.0f}G < {MEM_FLOOR_GB}G")
    except (OSError, StopIteration, ValueError):
        why.append("meminfo unreadable")
    return why


def stop_server(st, reason):
    pid = st.get("pid")
    if pid_alive(pid):
        os.killpg(pid, signal.SIGTERM)
        for _ in range(30):
            if not pid_alive(pid):
                break
            time.sleep(1)
        else:
            os.killpg(pid, signal.SIGKILL)
    st.update({"serving": False, "pid": None, "stopped_at": time.time(),
               "stop_reason": reason})
    save_state(st)


def start_server(st, args):
    os.makedirs(STATE_DIR, exist_ok=True)
    log = open(os.path.join(STATE_DIR, "serve.log"), "a")
    # #4089: on v0.69.1 `--backend cuda` alone silently runs on the CPU. The lane asks for
    # the layers as a QUANTITY and proves the offload per request (`used_gpu`, below).
    cmd = [args.apr, "serve", "run", args.model, "--backend", "cuda", "--gpu-layers", "all",
           "--port", str(GX10_PORT), "--context-length", str(args.ctx), "--trace"]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True, env={**os.environ, "TMPDIR": args.tmpdir})
    st.update({"serving": False, "pid": proc.pid, "cmd": cmd, "started_at": time.time()})
    save_state(st)
    for _ in range(300):
        if proc.poll() is not None:
            st.update({"pid": None, "last_error": f"exited rc={proc.returncode}"})
            save_state(st)
            return
        if health(f"http://127.0.0.1:{GX10_PORT}"):
            st["serving"] = True
            save_state(st)
            return
        time.sleep(1)


def health(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def cmd_watch(args):
    while True:
        st = load_state()
        why = need_signals(st.get("pid"))
        running = pid_alive(st.get("pid"))
        if why and running:
            stop_server(st, why)
            print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} YIELD {why}", flush=True)
        elif not why and not running:
            print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} START", flush=True)
            start_server(st, args)
        elif not running and st.get("serving"):
            st["serving"] = False
            save_state(st)
        if args.once:
            return 0
        time.sleep(args.period)


def post_chat(url, messages, max_tokens, timeout):
    body = {"model": "default", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return time.monotonic() - t0, d


def gx10_ask(messages, max_tokens, timeout):
    """Run the request ON gx10 against its loopback server (no tunnel, no LAN bind)."""
    payload = json.dumps({"model": "default", "messages": messages,
                          "max_tokens": max_tokens, "temperature": 0.0})
    # state.json is written indented; the protocol is line-based, so emit it on ONE line.
    remote = (f"S={STATE_DIR}; python3 -c 'import json,sys; "
              f"print(json.dumps(json.load(open(sys.argv[1]))))' $S/state.json; "
              f"L=$(wc -l < $S/serve.log); "
              f"curl -sf -m {timeout} http://127.0.0.1:{GX10_PORT}/v1/chat/completions "
              f"-H 'Content-Type: application/json' -d @- ; rc=$?; echo; echo CURL_RC=$rc; "
              # The qwen35 chat response carries no used_gpu (#4146); a 1-token
              # /v1/completions probe on the SAME process right after it does.
              f"echo PROBE=$(curl -sf -m 60 http://127.0.0.1:{GX10_PORT}/v1/completions "
              f"-H 'Content-Type: application/json' "
              f"-d '{{\"model\":\"default\",\"prompt\":\"Hi\",\"max_tokens\":1,\"temperature\":0}}' | "
              f"python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"used_gpu\"))'); "
              f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; "
              f"echo TRACE_BEGIN; grep -iE 'gpu.layers|offload' $S/serve.log | tail -2; "
              f"tail -n +$((L+1)) $S/serve.log | grep -iE 'cuda|gpu|kernel' | head -5")
    t0 = time.monotonic()
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "gx10", remote], input=payload,
                       capture_output=True, text=True, timeout=timeout + 30)
    dt = time.monotonic() - t0
    lines = r.stdout.splitlines()
    st = json.loads(lines[0]) if lines and lines[0].startswith("{") else {}
    rc_line = next((l for l in lines if l.startswith("CURL_RC=")), "CURL_RC=?")
    if rc_line != "CURL_RC=0":
        raise RuntimeError(f"gx10 request failed ({rc_line}); state={st}")
    resp = json.loads(lines[1])
    i = lines.index(rc_line)
    j = lines.index("TRACE_BEGIN")
    probe = next((l for l in lines if l.startswith("PROBE=")), "PROBE=")
    apps = [l for l in lines[i + 1:j] if not l.startswith("PROBE=")]
    trace = lines[j + 1:]
    pid = str(st.get("pid"))
    return dt, resp, {"host": "gx10", "server_pid": st.get("pid"),
                      "gpu_apps": apps,
                      "server_holds_gpu": any(a.split(",")[0].strip() == pid for a in apps),
                      "used_gpu_probe": probe.removeprefix("PROBE=") == "True",
                      "trace_lines": trace}


def cmd_ask(args):
    brief = open(args.brief).read() if args.brief else sys.stdin.read()
    messages = [{"role": "user", "content": brief}]
    rec = {"lane": "apr-dogfood-4252", "advisory": True, "counts": False,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "attempts": []}
    if not args.force_lambda:
        try:
            dt, resp, prov = gx10_ask(messages, args.max_tokens, args.timeout)
            # The device flag is intent; `used_gpu` on the response is the mechanism. A
            # gx10 answer without it is labelled CPU, never CUDA (#4089).
            used = resp.get("used_gpu")
            if used is None:  # #4146: the chat route omits it; take the probe's answer
                used = prov["used_gpu_probe"]
                prov["used_gpu_source"] = "completions probe (chat omits used_gpu, #4146)"
            on_gpu = used is True and prov["server_holds_gpu"]
            rec["attempts"].append({"host": "gx10", "ok": True, "wall_s": dt,
                                    "used_gpu": resp.get("used_gpu")})
            rec.update({"served_by": "gx10-cuda" if on_gpu else "gx10-cpu-UNPROVEN-GPU",
                        "provenance": prov, "response": resp})
        except Exception as e:  # yield, stop mid-request, ssh loss: all pass to lambda
            rec["attempts"].append({"host": "gx10", "ok": False, "error": str(e)[:400]})
    if "served_by" not in rec:
        h = health(LAMBDA_URL)
        try:
            dt, resp = post_chat(LAMBDA_URL, messages, args.max_tokens, args.timeout)
            rec["attempts"].append({"host": "lambda", "ok": True, "wall_s": dt})
            rec.update({"served_by": "lambda-cpu", "response": resp,
                        "provenance": {"host": "lambda", "health": h,
                                       "unit": "apr-dogfood-4252.service (apr-dogfood.slice)"}})
        except Exception as e:  # advisory lane: no verdict is a receipt, never a crash
            rec["attempts"].append({"host": "lambda", "ok": False, "error": str(e)[:400]})
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if "served_by" not in rec:
        rec.update({"served_by": None, "verdict": "NO-VERDICT"})
        print(json.dumps(rec, indent=1))
        return 2
    rec["text"] = rec["response"]["choices"][0]["message"]["content"]
    rec["usage"] = rec["response"].get("usage")
    print(json.dumps(rec, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch")
    w.add_argument("--apr", required=True)
    w.add_argument("--model", required=True)
    w.add_argument("--ctx", type=int, default=8192)
    w.add_argument("--period", type=int, default=5)
    w.add_argument("--tmpdir", default="/mnt/nvme-raid0/tmp")
    w.add_argument("--once", action="store_true")
    a = sub.add_parser("ask")
    a.add_argument("--brief")
    a.add_argument("--max-tokens", type=int, default=512)
    a.add_argument("--timeout", type=int, default=1800)
    a.add_argument("--force-lambda", action="store_true")
    sub.add_parser("need")
    args = ap.parse_args()
    if args.cmd == "watch":
        return cmd_watch(args)
    if args.cmd == "ask":
        return cmd_ask(args)
    print(json.dumps(need_signals(load_state().get("pid")), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
