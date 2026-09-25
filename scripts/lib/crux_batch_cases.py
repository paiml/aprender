"""Batch-mode case table shared by every CRUX source-weight driver (#3952; cop 2026-09-23: one engine load per
batch). Stdlib only: `run(engine)` monkeypatches the driver's `load_inproc` / `serve_session` with counting fakes
and returns the number of FAILED cases. The property under test is the reason batch mode exists: N items, ONE
load, exactly one row-contract-v1 row per item, in order.
"""

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path

LOADS = {"inproc": 0, "serve": 0}
CALLS = []


def clip_head(s, n):
    """`s` cut to its first `n` chars, SAYING how many were cut (#4046: a silent cut reads as the whole text)."""
    return s if len(s) <= n else f"{s[:n]} … and {len(s) - n} more chars"


def clip_tail(s, n):
    """`s` cut to its last `n` chars, SAYING how many were dropped (#4046)."""
    return s if len(s) <= n else f"[{len(s) - n} earlier chars dropped] {s[-n:]}"


def batch_env():
    w = Path(tempfile.mkdtemp())
    os.environ["CRUX_MANIFEST"], os.environ["CRUX_WORK"] = str(w / "manifest.jsonl"), str(w)
    return w


def msgs(w: Path, name: str, turns: list) -> str:
    p = w / f"{name}.json"
    p.write_text(json.dumps({"messages": [{"role": "user", "content": t} for t in turns]}))
    return str(p)


def model_args(**kw):
    base = dict(model="x", model_sha256="f" * 64, backend="gpu", host="t", seed=1, temperature=0.0, context=512,
                source_repo="r/r", source_revision="abc", dtype="bfloat16")
    base.update(kw)
    return argparse.Namespace(**base)


def fake_inproc(fail=None, boom_on=None, interface="inproc-fake"):
    def loader(_a):
        LOADS["inproc"] += 1
        if fail:
            raise RuntimeError(fail)

        def respond(convo, thinking, max_tokens):
            CALLS.append((convo[-1]["content"], thinking, max_tokens))
            if boom_on and convo[-1]["content"] == boom_on:
                raise ValueError("item-level failure")
            return f"ans:{convo[-1]['content']}", 7, 2
        return respond, interface, "cuda:0 fake"
    return loader


def fake_serve_for(interface):
    @contextlib.contextmanager
    def fake_serve(_a, _log):
        LOADS["serve"] += 1
        def respond(convo, thinking, max_tokens, resp_path=None, stream=False):
            CALLS.append(("serve", stream))
            return "s", 1, 1
        yield respond, interface, "cuda:0 fake (serve)"
    return fake_serve


def rows(w):
    return [json.loads(line) for line in open(w / "manifest.jsonl")]


def run(engine, serve_interface: str) -> int:
    """serve_interface: the name the driver's item_doc uses to pass resp_path ("vllm serve",
    "transformers serve")."""
    engine.serve_session = fake_serve_for(serve_interface)
    failed = 0

    def item(i, verb="run", turns=None, thinking="off", max_tokens=5, w=None):
        return {"prompt_id": f"p{i}", "verb": verb, "messages": msgs(w, f"m{i}", turns or [f"q{i}"]),
                "thinking": thinking, "max_tokens": max_tokens}

    def one_load():
        w = batch_env()
        engine.load_inproc = fake_inproc()
        engine.run_batch(model_args(), [item(i, max_tokens=10 + i, w=w) for i in range(5)])
        r = rows(w)
        good = (LOADS["inproc"] == 1 and [x["prompt_id"] for x in r] == [f"p{i}" for i in range(5)]
                and all(x["rc"] == 0 for x in r) and len({x["batch"]["id"] for x in r}) == 1
                and r[0]["batch"]["size"] == 5 and [c[2] for c in CALLS] == [10, 11, 12, 13, 14])
        return True if good else (dict(LOADS), [(x["prompt_id"], x["rc"]) for x in r], CALLS)

    def load_fails():
        w = batch_env()
        engine.load_inproc = fake_inproc(fail="engine init exploded")
        engine.run_batch(model_args(), [item(i, w=w) for i in range(3)])
        r = rows(w)
        return True if (len(r) == 3 and all(x["rc"] is None and "engine init exploded" in (x["refused"] or "")
                                            for x in r) and LOADS["inproc"] == 1) else r

    def isolation():
        w = batch_env()
        engine.load_inproc = fake_inproc(boom_on="q1")
        engine.run_batch(model_args(), [item(i, thinking="on", w=w) for i in range(3)])
        r = rows(w)
        return True if ([x["rc"] for x in r] == [0, None, 0] and "item-level failure" in r[1]["refused"]) else r

    def mixed():
        w = batch_env()
        engine.load_inproc = fake_inproc()
        engine.run_batch(model_args(), [item("a", w=w), item("b", verb="serve run", w=w),
                                        item("c", verb="chat", turns=["q1", "q2"], w=w)])
        r = {x["prompt_id"]: x for x in rows(w)}
        good = (LOADS == {"inproc": 1, "serve": 0} and r["pa"]["rc"] == 0 and r["pc"]["rc"] == 0
                and r["pb"]["rc"] is None and "on the same card" in (r["pb"]["refused"] or ""))
        return True if good else (dict(LOADS), {k: (v["rc"], v["refused"]) for k, v in r.items()})

    def chat_turns():
        w = batch_env()
        engine.load_inproc = fake_inproc()
        engine.run_batch(model_args(), [item("c", verb="chat", turns=["q1", "q2", "q3"], w=w)])
        doc = json.load(open(rows(w)[0]["stdout"]))
        return True if (doc["turns"] == ["ans:q1", "ans:q2", "ans:q3"] and len(CALLS) == 3) else (doc, CALLS)

    def invalid():
        w = batch_env()
        engine.load_inproc = fake_inproc()
        bad = item("bad", w=w)
        del bad["max_tokens"]
        engine.run_batch(model_args(), [item("ok", w=w), bad])
        r = {x["prompt_id"]: x for x in rows(w)}
        return True if (r["pok"]["rc"] == 0 and "no 'max_tokens'" in (r["pbad"]["refused"] or "")) else r

    def serve_one_server():
        w = batch_env()
        engine.run_batch(model_args(), [item(i, verb="serve run", w=w) for i in range(4)])
        r = rows(w)
        return True if (LOADS["serve"] == 1 and all(x["rc"] == 0 for x in r) and len(r) == 4) else (dict(LOADS), r)

    def gen_one():
        w = batch_env()
        engine.load_inproc = fake_inproc()
        engine.gen(model_args(prompt_id="g", verb="run", messages=msgs(w, "g", ["q"]), thinking="off",
                              max_tokens=9))
        r = rows(w)
        return True if (len(r) == 1 and r[0]["batch"]["size"] == 1 and LOADS["inproc"] == 1
                        and CALLS[0][2] == 9) else r

    def serve_stream():
        w = batch_env()
        engine.run_batch(model_args(), [item("n", verb="serve run", w=w), item("s", verb="serve stream", w=w)])
        r = {x["prompt_id"]: x for x in rows(w)}
        docs = {k: json.load(open(v["stdout"])) for k, v in r.items() if v["stdout"]}
        good = (LOADS["serve"] == 1 and CALLS == [("serve", False), ("serve", True)]
                and docs["ps"]["reported"]["interface"].endswith("(stream)")
                and not docs["pn"]["reported"]["interface"].endswith("(stream)"))
        return True if good else (dict(LOADS), CALLS, {k: v["reported"]["interface"] for k, v in docs.items()})

    def prefilled(text, want_text, want_reasoning):
        def case():
            w = batch_env()

            def loader(_a):
                LOADS["inproc"] += 1
                return (lambda convo, thinking, max_tokens: (text, 7, 2, True)), "inproc-fake", "cuda:0 fake"
            engine.load_inproc = loader
            engine.run_batch(model_args(), [item("t", thinking="on", w=w)])
            doc = json.load(open(rows(w)[0]["stdout"]))
            good = (doc["text"] == want_text and doc.get("reasoning", "") == want_reasoning
                    and doc["reported"]["prompt_opens_think"] is True and doc["raw_text"].startswith("<think>"))
            return True if good else doc
        return case

    def not_prefilled():
        w = batch_env()
        engine.load_inproc = fake_inproc()  # a 3-tuple respond: no prefill signal
        engine.run_batch(model_args(), [item("n", w=w)])
        doc = json.load(open(rows(w)[0]["stdout"]))
        return True if (doc["reported"]["prompt_opens_think"] is False and doc["raw_text"] == doc["text"]) else doc

    for name, fn in [("a prompt that OPENED the think block (#3990): the closed block splits into answer + reasoning",
                      prefilled("add them</think>4", "4", "add them")),
                     ("...and a block that never closes is NO answer, not the reasoning handed back as one",
                      prefilled("still adding", "", "still adding")),
                     ("without a prefill signal nothing is prepended", not_prefilled),
                     ("serve run and serve stream share ONE server; only the stream item streams", serve_stream),
                     ("5 items, ONE engine load, rows in order, per-item max_tokens", one_load),
                     ("a failed load refuses EVERY item with the load's reason", load_fails),
                     ("one item failing does not take the others down", isolation),
                     ("mixed interfaces: serve refused by name, only one engine loaded", mixed),
                     ("a chat item drives every user turn on the shared engine", chat_turns),
                     ("an invalid item is refused by name; the rest run", invalid),
                     ("4 serve items share ONE server", serve_one_server),
                     ("gen is a batch of one through the same path", gen_one)]:
        LOADS.update(inproc=0, serve=0)
        CALLS.clear()
        try:
            detail = fn()
            ok = detail is True
        except Exception as e:  # a case that raises is a broken case, not a pass
            ok, detail = False, f"{type(e).__name__}: {e}"
        print(f"{'ok  ' if ok else 'FAIL'} [batch] {name}" + ("" if ok else f"\n     got: {clip_head(str(detail), 300)}"))
        failed += not ok
    return failed


CASE_COUNT = 12


def run_sse() -> int:
    """The SSE reader both drivers use for `serve stream`: a truncated stream is an error, never an answer."""
    import crux_sse

    def ev(*chunks, done=True):
        lines = ["data: " + json.dumps(c) for c in chunks]
        return "\n\n".join(lines + (["data: [DONE]"] if done else [])) + "\n\n"

    d = lambda **kw: {"choices": [{"delta": kw}]}  # noqa: E731
    cases = [
        ("content deltas concatenate", lambda: crux_sse.parse_sse(ev(d(content="2 + "), d(content="2 = 4"))),
         ("2 + 2 = 4", {}, 2)),
        ("usage on the last chunk is read", lambda: crux_sse.parse_sse(ev(d(content="4"),
                                                                           {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 1}})),
         ("4", {"prompt_tokens": 9, "completion_tokens": 1}, 1)),
        ("reasoning deltas become a leading <think> block",
         lambda: crux_sse.parse_sse(ev(d(reasoning_content="add"), d(content="4"))), ("<think>add</think>4", {}, 1)),
        ("a stream with no [DONE] is TRUNCATED, refused", lambda: crux_sse.parse_sse(ev(d(content="4"), done=False)),
         crux_sse.TruncatedStream),
        ("an empty body is truncated too, not an empty answer", lambda: crux_sse.parse_sse(""),
         crux_sse.TruncatedStream),
        ("finish_reason terminal (transformers serve): the stop chunk ends the stream",
         lambda: crux_sse.parse_sse(ev(d(content="4"), {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                                       done=False), terminal="finish_reason"), ("4", {}, 1)),
        ("finish_reason terminal: no stop chunk is truncated",
         lambda: crux_sse.parse_sse(ev(d(content="4"), done=False), terminal="finish_reason"),
         crux_sse.TruncatedStream),
        ("done terminal (vLLM): a finish_reason does NOT excuse a missing [DONE]",
         lambda: crux_sse.parse_sse(ev(d(content="4"), {"choices": [{"delta": {}, "finish_reason": "stop"}]},
                                       done=False), terminal="done"), crux_sse.TruncatedStream),
    ]
    failed = 0
    for name, fn, want in cases:
        try:
            got = fn()
            ok = got == want
        except Exception as e:  # noqa: BLE001
            got, ok = f"{type(e).__name__}: {e}", isinstance(want, type) and isinstance(e, want)
        print(f"{'ok  ' if ok else 'FAIL'} [sse] {name}" + ("" if ok else f"\n     got: {clip_head(str(got), 200)}"))
        failed += not ok
    return failed


SSE_CASE_COUNT = 8


def run_proc() -> int:
    """crux_proc: SIGTERM to a driver takes EVERY engine child with it (aprender-dd measured VLLM::EngineCore left on
    gx10's card holding 58928 MiB). A child process installs the handler and spawns a plain child, a child in its
    OWN session (as `vllm serve` is started), and a grandchild behind a shell; SIGTERM to it must kill all three.
    Survivors are found by a per-case marker on the planted sleeps (pgrep -f from this view), which works across a
    pid namespace, where the pids the child prints would be the namespace's."""
    import random
    import signal
    import subprocess
    import sys

    lib = str(Path(__file__).resolve().parent)
    import importlib
    sys.path.insert(0, lib)
    crux_proc = importlib.import_module("crux_proc")

    def sig(proc_pid, signum):
        """Signal a process /proc numbers `proc_pid`. THIS test process may itself sit in a pid namespace whose
        /proc is the host's (quorum lanes 2 and 3 ran exactly there: ProcessLookupError), so the pid is translated
        through the same NSpid mapping the handler uses. A process not visible from here is skipped."""
        t = crux_proc.signal_pid(proc_pid)
        if t is None:
            return
        try:
            os.kill(t, signum)
        except ProcessLookupError:
            pass

    def prog(marker):
        return (
            "import os, sys, subprocess, time; sys.path.insert(0, %r); import crux_proc\n"
            "if os.environ.get('CRUX_PROC_INSTALL', '1') == '1': crux_proc.install()\n"
            "subprocess.Popen(['sleep', %r])\n"
            "subprocess.Popen(['sleep', %r], start_new_session=True)\n"
            "subprocess.Popen(['bash', '-c', 'sleep %s & wait'])\n"
            "print(os.readlink('/proc/self'), flush=True)\n"
            "time.sleep(120)\n" % (lib, marker, marker, marker))

    def survivors(marker):
        r = subprocess.run(["pgrep", "-f", "^sleep %s$" % marker], capture_output=True, text=True)
        return [int(x) for x in r.stdout.split()]

    def case(install, wrap=()):
        marker = "300.%06d" % random.randrange(10 ** 6)
        env = dict(os.environ, CRUX_PROC_INSTALL="1" if install else "0")
        parent = subprocess.Popen(list(wrap) + [sys.executable, "-c", prog(marker)], stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True, env=env)
        line = parent.stdout.readline().strip()
        if not line.isdigit():
            parent.kill()
            return "ENV", clip_head(parent.stderr.read().strip(), 160) or "no pid printed"
        deadline = time.time() + 5
        while time.time() < deadline and len(survivors(marker)) < 3:
            time.sleep(0.05)
        target = int(line)  # the pid as /proc numbers it
        sig(target, signal.SIGTERM)
        if wrap:
            # the namespace's init (bash) outlives the target by design: wait for the TARGET to go, then look —
            # before init exits and the kernel tears the namespace (and every survivor) down
            deadline = time.time() + 20
            while time.time() < deadline and os.path.exists("/proc/%d" % target):
                time.sleep(0.05)
            time.sleep(0.3)
            left = survivors(marker)
            rc = int(parent.stderr.readline().strip().split("=")[1]) if not os.path.exists("/proc/%d" % target) else None
            parent.kill()
            parent.wait()
            for k in left:
                sig(k, signal.SIGKILL)
            return rc, left
        try:
            rc = parent.wait(timeout=20)
        except subprocess.TimeoutExpired:
            parent.kill()
            rc = None
        time.sleep(0.3)
        left = survivors(marker)
        for k in left:  # never leave the planted sleeps behind
            sig(k, signal.SIGKILL)
        return rc, left

    import time

    failed = 0
    rc, left = case(install=True)
    ok = rc == 143 and not left
    print(f"{'ok  ' if ok else 'FAIL'} [proc] SIGTERM takes a plain child, a new-session child and a grandchild; exit 143"
          + ("" if ok else f"\n     got: rc={rc} survivors={left}"))
    failed += not ok
    rc, left = case(install=False)
    ok = len(left) == 3
    print(f"{'ok  ' if ok else 'FAIL'} [proc] control: WITHOUT the handler all three survive (the leak is real)"
          + ("" if ok else f"\n     got: rc={rc} survivors={left}"))
    failed += not ok
    # the lane's sandbox: a pid namespace whose /proc is the host's (getpid() != /proc's pid). The target must NOT
    # be the namespace's init: init exiting makes the kernel kill the whole namespace, which would pass this case
    # whatever the handler did (it did: a pre-fix mutant survived until init became a separate bash)
    rc, left = case(install=True, wrap=("unshare", "-Urpf", "bash", "-c", '"$@"; echo "rc=$?" >&2; sleep 8', "_"))
    if rc == "ENV":
        # Not measurable HERE (unprivileged containers refuse `unshare -p`): named, and counted as ENV so the suite
        # exits 2, never as a pass. Where this test process ALREADY runs in such a namespace, the two cases above
        # exercise the same translation.
        print(f"ENV  [proc] ...inside a pid namespace with the host's /proc: `unshare -Urpf` refused here ({left})")
        global ENV_CASES
        ENV_CASES += 1
        return failed
    ok = rc == 143 and not left
    print(f"{'ok  ' if ok else 'FAIL'} [proc] ...and inside a pid namespace with the host's /proc (unshare -Urpf)"
          + ("" if ok else f"\n     got: rc={rc} survivors={left}"))
    failed += not ok
    return failed


PROC_CASE_COUNT = 3
#: cases that could not run in this environment (named when they happen): a suite with any exits 2, never 0
ENV_CASES = 0
