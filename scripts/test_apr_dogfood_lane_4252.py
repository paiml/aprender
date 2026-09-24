#!/usr/bin/env python3
"""#4252: the lane's hard budget holds on the gx10 -> lambda path, not only --force-lambda.

gx10 fails (the yield case) and lambda accepts the request and never answers (the
SIGSTOP case). `ask` must return verdict "unavailable", rc 2, inside the budget.
Run: python3 scripts/test_apr_dogfood_lane_4252.py [budget_s]   (default 6)
"""

import argparse
import contextlib
import io
import json
import os
import socket
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import apr_dogfood_lane_4252 as lane  # noqa: E402


def hung_server():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)
    held = []

    def accept():
        while True:
            conn, _ = s.accept()
            held.append(conn)  # read nothing, answer nothing
    threading.Thread(target=accept, daemon=True).start()
    return f"http://127.0.0.1:{s.getsockname()[1]}"


def trickle_server():
    """Answers headers, then one body byte every 0.3 s forever: a per-read socket timeout
    never fires, so only the lane's alarm can end the ask."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(8)

    def serve(conn):
        try:
            conn.recv(65536)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                         b"Content-Length: 100000000\r\n\r\n")
            while True:
                conn.sendall(b" ")
                time.sleep(0.3)
        except OSError:
            pass

    def accept():
        while True:
            conn, _ = s.accept()
            threading.Thread(target=serve, args=(conn,), daemon=True).start()
    threading.Thread(target=accept, daemon=True).start()
    return f"http://127.0.0.1:{s.getsockname()[1]}"


def main():
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    url = hung_server()
    lane.LAMBDA_URL = url
    lane.health = lambda u: {"stub": u}          # /health would hang too; the POST is the case

    def gx10_yielded(*_a, **_k):
        raise RuntimeError("gx10 CURL_RC=7 (yielded)")
    lane.gx10_ask = gx10_yielded
    brief = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"brief-{os.getpid()}.txt")
    open(brief, "w").write("Reply PASS.")
    args = argparse.Namespace(brief=brief, max_tokens=8, timeout=budget, force_lambda=False)
    out = io.StringIO()
    t0 = time.monotonic()
    with contextlib.redirect_stdout(out):
        rc = lane.cmd_ask(args)
    wall = time.monotonic() - t0
    os.unlink(brief)
    rec = json.loads(out.getvalue())
    hosts = [a["host"] for a in rec["attempts"]]
    ok = (rc == 2 and rec.get("verdict") == "unavailable" and rec["served_by"] is None
          and hosts == ["gx10", "lambda"] and wall <= budget + 2)
    print(json.dumps({"rc": rc, "verdict": rec.get("verdict"), "attempts": hosts,
                      "wall_s": round(wall, 1), "budget_s": budget,
                      "result": "PASS" if ok else "FAIL"}))
    if not ok:
        return 1
    # --timeout 0 must still be a budget: signal.alarm(0) would cancel it. The trickle
    # server defeats urlopen's per-read timeout, so this row sees the alarm alone.
    lane.LAMBDA_URL = trickle_server()
    open(brief, "w").write("Reply PASS.")
    args = argparse.Namespace(brief=brief, max_tokens=8, timeout=0, force_lambda=False)
    t0 = time.monotonic()
    with contextlib.redirect_stdout(io.StringIO()):
        rc0 = lane.cmd_ask(args)
    wall0 = time.monotonic() - t0
    os.unlink(brief)
    ok0 = rc0 == 2 and wall0 <= 4
    print(json.dumps({"case": "timeout 0", "rc": rc0, "wall_s": round(wall0, 1),
                      "result": "PASS" if ok0 else "FAIL"}))
    return 0 if ok0 else 1


if __name__ == "__main__":
    sys.exit(main())
