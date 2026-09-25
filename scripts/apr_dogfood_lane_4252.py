#!/usr/bin/env python3
"""#4252 mandatory apr dogfood: the advisory quorum lane's serve fleet.

PRIMARY  gx10, `apr serve` built with cuda, Qwen3.5-4B.
FALLBACK lambda, the CPU release asset, inside the capped user slice
         apr-dogfood.slice (Nice 19, CPUQuota 800%, MemoryMax 12G). OFF BY DEFAULT: operator,
         via the cop (2026-09-25): "lambda fans are loud … Keep lambda CPU runs only when
         load1 < 24 (infra#1087 row 5). Do not start new lambda-CPU apr runs." So `ask`
         tries lambda only with --allow-lambda (or --force-lambda) AND load1 < 24; otherwise
         the lambda attempt is recorded as not tried, with the reason.

Operator, verbatim (via the cop, 2026-09-24): "perfect, if box is needed, it pauses and
passes work to lambda cpu". So the gx10 server YIELDS to any need for the box: a gpu-q
hold or queued ticket, a foreign GPU process (a CI GPU job), another session's claim file,
or a disk/memory alarm. Its work, including a request in flight, passes to lambda CPU.

Subcommands
  watch   (runs ON gx10)  every --period s: if the box is needed, stop the server by its
                          recorded pid; if free, (re)start it. State: STATE_DIR/state.json.
  ask     (runs on lambda) send one chat request to gx10's loopback server over ssh; any
                          gx10 failure (yielded, stopped mid-request, ssh loss) re-sends it
                          to lambda CPU. One hard budget (--timeout, default 120 s) covers
                          both; on expiry the verdict is "unavailable". Prints ONE JSON receipt.
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

STATE_REL = ".local/state/apr-dogfood-4252"   # under $HOME on the host that runs `watch`
STATE_DIR = os.path.join(os.path.expanduser("~"), STATE_REL)
CLAIM = "/tmp/apr-gx10-claim"          # another session's claim: any content, any owner
LOCK = "/tmp/apr-gpu.lock"
QUEUE = "/tmp/apr-gpu-queue"
DF_FLOOR_GB = 60                        # yield above the 50G alarm so a build never races it
MEM_FLOOR_GB = 16
GX10_PORT = 18253
LAMBDA_URL = "http://127.0.0.1:18252"
LAMBDA_MAX_LOAD1 = 24.0  # infra#1087 row 5


def load1():
    return os.getloadavg()[0]


def lambda_refusal(args):
    """None when the lambda leg may run; else why it may not (recorded, never raised)."""
    if not (args.force_lambda or getattr(args, "allow_lambda", False)):
        return "not tried: lambda CPU fallback is off by default (operator 2026-09-25, fans); pass --allow-lambda"
    l1 = load1()
    if l1 >= LAMBDA_MAX_LOAD1:
        return f"not tried: lambda load1 {l1:.1f} >= {LAMBDA_MAX_LOAD1:g} (infra#1087 row 5)"
    return None
PROBE_TIMEOUT = 15                      # a hung probe (nvidia-smi) is a reason to yield


def sh(cmd, timeout=None, **kw):
    """A probe that does not answer in time reads as failed (rc 124), never as a hang:
    a hung nvidia-smi would otherwise freeze the watcher and silence every yield."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=timeout or PROBE_TIMEOUT, **kw)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"timed out: {cmd[0]}")


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
    # the pid must still be OUR server, not a recycled number: `serve` and our own port as
    # whole argv tokens, never a substring of some other process's path
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().split(b"\0")
    except OSError:
        return False
    return b"serve" in argv and str(GX10_PORT).encode() in argv


def need_signals(own_pid):
    """Every reason the box is needed. Empty list = free. Fails CLOSED: a probe that
    cannot answer is itself a reason."""
    why = []
    # 1. gpu-q: a held lock or a queued ticket
    r = sh(["flock", "-n", LOCK, "true"])
    if r.returncode != 0:
        why.append(f"gpu lock held ({LOCK})" if r.returncode != 124
                   else f"gpu lock probe timed out ({LOCK})")
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


def reap_children():
    """Reap every exited child: a server killed on yield (or dead on its own) would
    otherwise stay a zombie in the long-lived watcher, one per yield cycle."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def stop_server(st, reason):
    pid = st.get("pid")
    # the server may exit between pid_alive and the kill; that is "already gone", never a
    # crash of the watch loop (which would leave the lane down after the need clears)
    try:
        if pid_alive(pid):
            os.killpg(pid, signal.SIGTERM)
            for _ in range(30):
                if not pid_alive(pid):
                    break
                time.sleep(1)
            else:
                os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    reap_children()
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
        # a need that appears while the model loads is honoured now, not after the load
        why = need_signals(proc.pid)
        if why:
            stop_server(st, why)
            print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} YIELD (during start) {why}",
                  flush=True)
            return
        if proc.poll() is not None:
            st.update({"pid": None, "last_error": f"exited rc={proc.returncode}"})
            save_state(st)
            return
        if health(f"http://127.0.0.1:{GX10_PORT}"):
            st["serving"] = True
            save_state(st)
            return
        time.sleep(1)
    # never healthy: stop it and say so, rather than leave a live pid marked not-serving
    stop_server(st, ["startup timeout: /health never answered in 300 s"])


class LaneBudgetSpent(BaseException):
    """The ask's one hard budget ran out (raised by the SIGALRM handler)."""


def health(url):
    try:
        with urllib.request.urlopen(url + "/health", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def cmd_watch(args):
    while True:
        reap_children()
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
    # $HOME is expanded by gx10's shell: the state dir is the WATCHER's, not this host's.
    remote = (f"S=$HOME/{STATE_REL}; python3 -c 'import json,sys; "
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
              # the served binary's version, so the receipt names what answered (#4252)
              f"echo HEALTH=$(curl -sf -m 3 http://127.0.0.1:{GX10_PORT}/health | tr -d '\\n'); "
              f"nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; "
              f"echo TRACE_BEGIN; grep -iE 'gpu.layers|offload' $S/serve.log | tail -2; "
              f"tail -n +$((L+1)) $S/serve.log | grep -iE 'cublas|prefill|cuda|gpu|kernel' | head -5")
    t0 = time.monotonic()
    # ssh ends 10 s after the remote curl's -m, i.e. 20 s BEFORE the caller's deadline: a
    # hung gx10 raises TimeoutExpired here and never spends the alarm (quorum R4).
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=5", "gx10", remote], input=payload,
                       capture_output=True, text=True, timeout=timeout + 10)
    dt = time.monotonic() - t0
    resp, prov = parse_gx10(r.stdout)
    return dt, resp, prov


def parse_gx10(stdout):
    """Split gx10_ask's line protocol into (response, provenance). Pure: tested on a fixture."""
    lines = stdout.splitlines()
    st = json.loads(lines[0]) if lines and lines[0].startswith("{") else {}
    rc_line = next((l for l in lines if l.startswith("CURL_RC=")), "CURL_RC=?")
    if rc_line != "CURL_RC=0":
        raise RuntimeError(f"gx10 request failed ({rc_line}); state={st}")
    resp = json.loads(lines[1])
    i = lines.index(rc_line)
    j = lines.index("TRACE_BEGIN")
    probe = next((l for l in lines if l.startswith("PROBE=")), "PROBE=")
    hline = next((l for l in lines if l.startswith("HEALTH=")), "HEALTH=")
    try:
        h = json.loads(hline.removeprefix("HEALTH="))
    except ValueError:
        h = None
    apps = [l for l in lines[i + 1:j] if not l.startswith(("PROBE=", "HEALTH="))]
    trace = lines[j + 1:]
    pid = str(st.get("pid"))
    return resp, {"host": "gx10", "server_pid": st.get("pid"),
                  "health": h,
                  "apr": h.get("version") if isinstance(h, dict) else None,
                  "gpu_apps": apps,
                  "server_holds_gpu": any(a.split(",")[0].strip() == pid for a in apps),
                  "used_gpu_probe": probe.removeprefix("PROBE=") == "True",
                  "trace_lines": trace}


def cmd_ask(args):
    brief = open(args.brief).read() if args.brief else sys.stdin.read()
    messages = [{"role": "user", "content": brief}]
    # signal.alarm(0) CANCELS the alarm: a zero or negative budget would mean no budget
    args.timeout = max(1, args.timeout)
    rec = {"lane": "apr-dogfood-4252", "advisory": True, "counts": False,
           "timeout_s": args.timeout,
           "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "attempts": []}
    # ONE budget for the whole ask, both hosts (cop 2026-09-24: the lane never hangs the
    # quorum). A SIGSTOPped server (lambda's load watchdog) blocks a read forever, so the
    # per-host timeouts are backed by an alarm on the whole call.
    deadline = time.monotonic() + args.timeout
    remaining = lambda: max(1, int(deadline - time.monotonic()))

    def expired(*_):
        # NOT an Exception: every `except Exception` on the way (health(), the per-leg
        # handlers) would swallow it and leave the next call unbounded (quorum R4/R5).
        raise LaneBudgetSpent(f"lane budget {args.timeout}s spent")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(args.timeout)
    try:
        _ask_legs(args, messages, rec, deadline, remaining)
    except LaneBudgetSpent as e:
        rec["budget_spent"] = str(e)
        hosts = [a["host"] for a in rec["attempts"]]
        for h in ([] if args.force_lambda else ["gx10"]) + ["lambda"]:
            if h not in hosts:
                rec["attempts"].append({"host": h, "ok": False, "error": str(e)})
    signal.alarm(0)
    rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    if "served_by" in rec:
        try:
            rec["text"] = rec["response"]["choices"][0]["message"]["content"]
            rec["usage"] = rec["response"].get("usage")
        except (KeyError, IndexError, TypeError) as e:
            # a 200 without the OpenAI shape is no answer; the lane still prints a receipt
            rec["attempts"][-1].update({"ok": False, "error": f"malformed response: {e!r}"})
            del rec["served_by"]
    if "served_by" not in rec:
        rec.update({"served_by": None, "verdict": "unavailable"})
        print(json.dumps(rec, indent=1))
        return 2
    print(json.dumps(rec, indent=1))
    return 0


def _ask_legs(args, messages, rec, deadline, remaining):
    """gx10 first, lambda on any gx10 failure; each leg records one attempt."""
    if not args.force_lambda:
        try:
            dt, resp, prov = gx10_ask(messages, args.max_tokens, max(1, remaining() - 30))
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
    if "served_by" not in rec and deadline - time.monotonic() < 1:
        # gx10 failed with under a second left: starting lambda now could only be cut off
        # by the alarm mid-request. (The alarm itself can no longer be swallowed: it
        # raises LaneBudgetSpent, a BaseException — quorum R4/R5.)
        rec["attempts"].append({"host": "lambda", "ok": False,
                                "error": f"not tried: < 1 s of the {args.timeout}s lane budget left"})
    elif "served_by" not in rec and (why := lambda_refusal(args)) is not None:
        rec["attempts"].append({"host": "lambda", "ok": False, "error": why})
    elif "served_by" not in rec:
        try:
            h = health(LAMBDA_URL)
            dt, resp = post_chat(LAMBDA_URL, messages, args.max_tokens, remaining())
            rec["attempts"].append({"host": "lambda", "ok": True, "wall_s": dt})
            rec.update({"served_by": "lambda-cpu", "response": resp,
                        "provenance": {"host": "lambda", "health": h,
                                       "apr": h.get("version") if isinstance(h, dict) else None,
                                       "unit": "apr-dogfood-4252.service (apr-dogfood.slice)"}})
        except Exception as e:  # advisory lane: no verdict is a receipt, never a crash
            rec["attempts"].append({"host": "lambda", "ok": False, "error": str(e)[:400]})


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
    a.add_argument("--timeout", type=int, default=120,
                   help="hard budget for the whole ask; on expiry the lane is 'unavailable'")
    a.add_argument("--force-lambda", action="store_true")
    a.add_argument("--allow-lambda", action="store_true",
                   help="let a gx10 failure fall back to lambda CPU (still refused at load1 >= 24)")
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
