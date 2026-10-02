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


def gx10_yielded(*_a, **_k):
    raise RuntimeError("gx10 CURL_RC=7 (yielded)")


def gx10_hangs(*_a, **_k):
    time.sleep(3600)


def ask(ctx, **kw):
    """One `lane.cmd_ask` on a fresh brief: (rc, its stdout, wall seconds)."""
    open(ctx["brief"], "w").write("Reply PASS.")
    args = argparse.Namespace(brief=ctx["brief"], max_tokens=8, **kw)
    out = io.StringIO()
    t0 = time.monotonic()
    try:
        with contextlib.redirect_stdout(out):
            rc = lane.cmd_ask(args)
    finally:
        wall = time.monotonic() - t0
        os.unlink(ctx["brief"])
    return rc, out.getvalue(), wall


def verdict(row, ok):
    """Print one case's row with its result last; the case passed iff `ok`."""
    row["result"] = "PASS" if ok else "FAIL"
    print(json.dumps(row))
    return ok


def case_both_legs_unavailable(ctx):
    budget = ctx["budget"]
    rc, out, wall = ask(ctx, timeout=budget, force_lambda=False, allow_lambda=True)
    rec = json.loads(out)
    hosts = [a["host"] for a in rec["attempts"]]
    ok = (rc == 2 and rec.get("verdict") == "unavailable" and rec["served_by"] is None
          and hosts == ["gx10", "lambda"] and wall <= budget + 2)
    return verdict({"rc": rc, "verdict": rec.get("verdict"), "attempts": hosts,
                    "wall_s": round(wall, 1), "budget_s": budget}, ok)


def case_timeout_zero(ctx):
    # gx10 hangs in Python (no socket timeout applies) and lambda trickles one byte per
    # 0.3 s (defeats urlopen's per-read timeout): from here on only the alarm ends an ask.
    lane.gx10_ask = gx10_hangs
    lane.LAMBDA_URL = trickle_server()
    # --timeout 0 must still be a budget: signal.alarm(0) would cancel it.
    rc0, _out, wall0 = ask(ctx, timeout=0, force_lambda=False, allow_lambda=True)
    return verdict({"case": "timeout 0", "rc": rc0, "wall_s": round(wall0, 1)}, rc0 == 2 and wall0 <= 4)


def case_alarm_in_gx10_leg(ctx):
    # The alarm fires INSIDE the gx10 leg and the except swallows it; a lambda attempt
    # made after that would trickle unbounded. The ask must still end within the budget.
    rcg, out, wallg = ask(ctx, timeout=ctx["budget"], force_lambda=False, allow_lambda=True)
    recg = json.loads(out)
    okg = (rcg == 2 and wallg <= ctx["budget"] + 2
           and [a["host"] for a in recg["attempts"]] == ["gx10", "lambda"])
    return verdict({"case": "alarm spent in the gx10 leg", "rc": rcg,
                    "wall_s": round(wallg, 1), "lambda": recg["attempts"][-1].get("error")}, okg)


def case_alarm_in_health(ctx):
    # The alarm fires inside health() — the REAL one, whose `except Exception` swallowed
    # a TimeoutError — against the trickle server; post_chat after it would be unbounded.
    lane.health = ctx["real_health"]
    rch, out, wallh = ask(ctx, timeout=ctx["budget"], force_lambda=True)
    rech = json.loads(out)
    okh = rch == 2 and wallh <= ctx["budget"] + 2 and "budget_spent" in rech
    return verdict({"case": "alarm fires inside health()", "rc": rch,
                    "wall_s": round(wallh, 1), "budget_spent": rech.get("budget_spent")}, okh)


def case_malformed_200(ctx):
    # A 200 that is not the OpenAI shape: no answer, but a receipt, never a crash.
    lane.post_chat = lambda *_a, **_k: (0.1, {"error": "not a completion"})
    lane.health = lambda u: {"stub": u}
    try:
        rcm, out, _wall = ask(ctx, timeout=ctx["budget"], force_lambda=True)
        recm = json.loads(out)
        okm = rcm == 2 and recm["verdict"] == "unavailable" and \
            "malformed response" in recm["attempts"][-1].get("error", "")
        detail = recm["attempts"][-1].get("error")
    except Exception as e:
        rcm, okm, detail = None, False, repr(e)
    return verdict({"case": "malformed 200 -> unavailable receipt", "rc": rcm, "detail": detail}, okm)


def case_lambda_leg_refused(ctx, name, kw, load, why):
    lane.load1 = lambda load=load: load
    rcg, out, _wall = ask(ctx, timeout=ctx["budget"], **kw)
    recg = json.loads(out)
    last = recg["attempts"][-1]
    okg = (rcg == 2 and recg["served_by"] is None and not ctx["posts"] and last["host"] == "lambda"
           and why in last.get("error", ""))
    return verdict({"case": name, "rc": rcg, "detail": last.get("error")}, okg)


def case_lambda_gate(ctx):
    # The lambda fallback is OFF unless asked for, and refused at load1 >= 24 even when asked
    # (operator 2026-09-25, fans; infra#1087 row 5). A leg that must not run must not POST.
    lane.gx10_ask = gx10_yielded                 # an earlier case left gx10 hanging
    posts = ctx["posts"]
    lane.post_chat = lambda *_a, **_k: (posts.append(1), (0.1, {"choices": [{"message": {"content": "PASS"}}]}))[1]
    cases = [("default: lambda off", dict(force_lambda=False), 0.0, "off by default"),
             ("--allow-lambda at load1 30", dict(force_lambda=False, allow_lambda=True), 30.0, "load1 30.0"),
             ("--force-lambda at load1 24", dict(force_lambda=True), 24.0, "load1 24.0")]
    return all(case_lambda_leg_refused(ctx, *c) for c in cases)


def case_lambda_positive_control(ctx):
    # The positive control: asked for and under the load line, the lambda leg does run.
    lane.load1 = lambda: 23.9
    rcp, out, _wall = ask(ctx, timeout=ctx["budget"], force_lambda=False, allow_lambda=True)
    recp = json.loads(out)
    okp = rcp == 0 and recp["served_by"] == "lambda-cpu" and len(ctx["posts"]) == 1
    return verdict({"case": "--allow-lambda at load1 23.9 runs", "rc": rcp,
                    "served_by": recp["served_by"]}, okp)


# In order: each case leaves the lane stubs as the next one expects. The first failure stops the run.
CASES = [case_both_legs_unavailable, case_timeout_zero, case_alarm_in_gx10_leg, case_alarm_in_health,
         case_malformed_200, case_lambda_gate, case_lambda_positive_control]


def main():
    # A case that hangs is a FAIL, not a stuck test: hard exit well past every budget.
    guard = threading.Timer(60, lambda: (print("FAIL: a case hung past 60 s", file=sys.__stderr__, flush=True),
                                 os._exit(3)))
    guard.daemon = True
    guard.start()
    ctx = {"budget": int(sys.argv[1]) if len(sys.argv) > 1 else 6, "real_health": lane.health, "posts": [],
           "brief": os.path.join(os.environ.get("TMPDIR", "/tmp"), f"brief-{os.getpid()}.txt")}
    lane.LAMBDA_URL = hung_server()
    lane.health = lambda u: {"stub": u}          # /health would hang too; the POST is the case
    lane.load1 = lambda: 0.0                     # the lambda leg's load gate is its own case below
    lane.gx10_ask = gx10_yielded
    return 0 if all(case(ctx) for case in CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
