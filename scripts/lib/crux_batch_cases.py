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
        yield (lambda convo, thinking, max_tokens, resp_path=None: ("s", 1, 1)), interface, "cuda:0 fake (serve)"
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

    for name, fn in [("5 items, ONE engine load, rows in order, per-item max_tokens", one_load),
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
        print(f"{'ok  ' if ok else 'FAIL'} [batch] {name}" + ("" if ok else f"\n     got: {str(detail)[:300]}"))
        failed += not ok
    return failed


CASE_COUNT = 8
