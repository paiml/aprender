#!/usr/bin/env python3
"""#4252: the watcher yields. Drives the REAL need_signals / start_server / stop_server /
cmd_watch against a fake `apr` (a loopback /health server) in a temp state dir.

Rows: free -> START; a real flock on the lock -> YIELD and the pid is gone; released ->
START again; a need raised while the model is still loading -> YIELD during start; a hung
nvidia-smi -> a yield reason, not a hang; a foreign GPU pid -> a yield reason.
Run: python3 scripts/test_apr_dogfood_lane_4252_watch.py
"""

import argparse
import contextlib
import fcntl
import io
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apr_dogfood_lane_4252 as lane  # noqa: E402

FAKE_APR = r'''#!/usr/bin/env python3
import http.server, sys, time
port = int(sys.argv[sys.argv.index("--port") + 1])
time.sleep(float(__import__("os").environ.get("FAKE_LOAD_S", "0")))
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')
    def log_message(self, *a): pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def main():
    tmp = tempfile.mkdtemp(prefix="lane4252-")
    fake = os.path.join(tmp, "apr")
    open(fake, "w").write(FAKE_APR)
    os.chmod(fake, 0o755)
    lane.STATE_DIR = os.path.join(tmp, "state")
    lane.LOCK = os.path.join(tmp, "gpu.lock")
    lane.QUEUE = os.path.join(tmp, "queue")
    lane.CLAIM = os.path.join(tmp, "claim")
    lane.GX10_PORT = free_port()
    lane.DF_FLOOR_GB = lane.MEM_FLOOR_GB = 0
    lane.PROBE_TIMEOUT = 2
    smi = {"out": ""}                      # the fake nvidia-smi's answer; "HANG" sleeps
    real_sh = lane.sh

    def sh(cmd, timeout=None, **kw):
        if cmd[0] == "nvidia-smi":
            if smi["out"] == "HANG":
                return real_sh(["sleep", "30"], timeout=timeout)
            return subprocess.CompletedProcess(cmd, 0, smi["out"], "")
        return real_sh(cmd, timeout=timeout, **kw)
    lane.sh = sh
    args = argparse.Namespace(apr=fake, model="m.gguf", ctx=8, period=1, tmpdir=tmp, once=True)
    rows = []

    def tick():
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            lane.cmd_watch(args)
        return out.getvalue().strip()

    def row(name, ok, detail):
        rows.append((name, ok))
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    t = tick(); st = lane.load_state(); pid1 = st.get("pid")
    row("free -> START and serving", "START" in t and st.get("serving") and lane.pid_alive(pid1),
        f"{t!r} pid={pid1}")
    # held in-process: `flock LOCK sleep` would leave the sleep child holding the fd after
    # the flock pid is killed, and the watcher (rightly) would keep yielding
    hold = open(lane.LOCK, "w")
    fcntl.flock(hold, fcntl.LOCK_EX)
    t = tick(); st = lane.load_state()
    row("gpu lock held -> YIELD, pid gone", "YIELD" in t and not lane.pid_alive(pid1)
        and st.get("serving") is False, f"{t!r} stop_reason={st.get('stop_reason')}")
    hold.close()
    t = tick(); st = lane.load_state(); pid2 = st.get("pid")
    row("released -> START again", "START" in t and lane.pid_alive(pid2), f"{t!r} pid={pid2}")
    lane.stop_server(st, ["test teardown"])
    # a need that appears while the model is loading
    os.environ["FAKE_LOAD_S"] = "6"
    threading.Timer(2.0, lambda: open(lane.CLAIM, "w").write("another session")).start()
    t0 = time.monotonic()
    t = tick(); st = lane.load_state()
    row("need during load -> YIELD during start", "YIELD (during start)" in t
        and st.get("pid") is None and time.monotonic() - t0 < 5,
        f"{t!r} after {time.monotonic() - t0:.1f}s")
    if os.path.exists(lane.CLAIM):
        os.unlink(lane.CLAIM)
    os.environ["FAKE_LOAD_S"] = "0"
    smi["out"] = "HANG"
    t0 = time.monotonic(); why = lane.need_signals(None)
    row("hung nvidia-smi -> reason, bounded", any("nvidia-smi failed rc=124" in w for w in why)
        and time.monotonic() - t0 < 5, f"{why} in {time.monotonic() - t0:.1f}s")
    smi["out"] = "99999, python3\n"
    why = lane.need_signals(12345)
    row("foreign GPU pid -> reason", any("foreign GPU process" in w for w in why), str(why))
    smi["out"] = ""
    row("all free -> no reason", lane.need_signals(None) == [], str(lane.need_signals(None)))
    st = lane.load_state()               # a mutant can leave a started server: stop it
    if lane.pid_alive(st.get("pid")):
        lane.stop_server(st, ["test teardown"])
    # a server that exits between pid_alive and killpg is "already gone", not a crash
    real_alive = lane.pid_alive
    lane.pid_alive = lambda pid: True
    try:
        lane.stop_server({"pid": 2 ** 22 + 7}, ["race"])   # no such process group
        row("killpg on a vanished pid -> no crash", True, "ProcessLookupError absorbed")
    except ProcessLookupError as e:
        row("killpg on a vanished pid -> no crash", False, repr(e))
    lane.pid_alive = real_alive
    bad = [n for n, ok in rows if not ok]
    print("RESULT", "PASS" if not bad else f"FAIL {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
