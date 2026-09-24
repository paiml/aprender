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
    # A case that hangs is a FAIL, not a stuck test: hard exit well past every budget.
    guard = threading.Timer(60, lambda: (print("FAIL: a case hung past 60 s", file=sys.__stderr__, flush=True),
                                 os._exit(3)))
    guard.daemon = True
    guard.start()
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    url = hung_server()
    lane.LAMBDA_URL = url
    real_health = lane.health
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
    # gx10 hangs in Python (no socket timeout applies) and lambda trickles one byte per
    # 0.3 s (defeats urlopen's per-read timeout): from here on only the alarm ends an ask.
    def gx10_hangs(*_a, **_k):
        time.sleep(3600)
    lane.gx10_ask = gx10_hangs
    lane.LAMBDA_URL = trickle_server()
    # --timeout 0 must still be a budget: signal.alarm(0) would cancel it.
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
    if not ok0:
        return 1
    # The alarm fires INSIDE the gx10 leg and the except swallows it; a lambda attempt
    # made after that would trickle unbounded. The ask must still end within the budget.
    open(brief, "w").write("Reply PASS.")
    args = argparse.Namespace(brief=brief, max_tokens=8, timeout=budget, force_lambda=False)
    out = io.StringIO()
    t0 = time.monotonic()
    with contextlib.redirect_stdout(out):
        rcg = lane.cmd_ask(args)
    wallg = time.monotonic() - t0
    os.unlink(brief)
    recg = json.loads(out.getvalue())
    okg = (rcg == 2 and wallg <= budget + 2
           and [a["host"] for a in recg["attempts"]] == ["gx10", "lambda"])
    print(json.dumps({"case": "alarm spent in the gx10 leg", "rc": rcg,
                      "wall_s": round(wallg, 1), "lambda": recg["attempts"][-1].get("error"),
                      "result": "PASS" if okg else "FAIL"}))
    if not okg:
        return 1
    # The alarm fires inside health() — the REAL one, whose `except Exception` swallowed
    # a TimeoutError — against the trickle server; post_chat after it would be unbounded.
    lane.health = real_health
    open(brief, "w").write("Reply PASS.")
    args = argparse.Namespace(brief=brief, max_tokens=8, timeout=budget, force_lambda=True)
    out = io.StringIO()
    t0 = time.monotonic()
    with contextlib.redirect_stdout(out):
        rch = lane.cmd_ask(args)
    wallh = time.monotonic() - t0
    os.unlink(brief)
    rech = json.loads(out.getvalue())
    okh = rch == 2 and wallh <= budget + 2 and "budget_spent" in rech
    print(json.dumps({"case": "alarm fires inside health()", "rc": rch,
                      "wall_s": round(wallh, 1), "budget_spent": rech.get("budget_spent"),
                      "result": "PASS" if okh else "FAIL"}))
    if not okh:
        return 1
    # A 200 that is not the OpenAI shape: no answer, but a receipt, never a crash.
    lane.post_chat = lambda *_a, **_k: (0.1, {"error": "not a completion"})
    lane.health = lambda u: {"stub": u}
    open(brief, "w").write("Reply PASS.")
    args = argparse.Namespace(brief=brief, max_tokens=8, timeout=budget, force_lambda=True)
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            rcm = lane.cmd_ask(args)
        recm = json.loads(out.getvalue())
        okm = rcm == 2 and recm["verdict"] == "unavailable" and \
            "malformed response" in recm["attempts"][-1].get("error", "")
        detail = recm["attempts"][-1].get("error")
    except Exception as e:
        rcm, okm, detail = None, False, repr(e)
    os.unlink(brief)
    print(json.dumps({"case": "malformed 200 -> unavailable receipt", "rc": rcm,
                      "detail": detail, "result": "PASS" if okm else "FAIL"}))
    return 0 if okm else 1


if __name__ == "__main__":
    sys.exit(main())
