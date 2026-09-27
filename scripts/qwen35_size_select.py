#!/usr/bin/env python3
"""Qwen3.5 Pareto size selection, derived from committed evidence only (#3558).

A size is ADMISSIBLE for (host, consumer) when the committed evidence shows both:
  * capability -- the model ladder receipt's rung is `green` on that host, and the
    rung is not a sha-matched `known_red` of contracts/model-capability-ladder-v1.yaml;
  * context    -- a committed run on that host ingested >= the consumer's declared
    `max_prompt_tokens` (evidence/release/context-rungs.json) on the GPU, rc 0,
    not refused.
It is SELECTED when no smaller admissible size exists ("smaller" = model bytes from
the receipt inventory, which is the cost axis the committed evidence carries).

Absent evidence is never a pass. A size with no receipt row, no byte count or no
context run at the budget is `unmeasured` and not admissible, and the table says
which input is missing -- that is the list of measurements that could move the
frontier down. Nothing admissible selects NOTHING; it never falls back to the
largest size.

What "context met" means, stated so it is not over-read: the context runs are
prefill runs (`max_tokens: 1`). They prove the size ingests the budget on that GPU;
they do not prove it answers correctly at that length. The table records this as
`context.basis = "ingested"`.

Usage:
  qwen35_size_select.py [--version V] [--out PATH]     derive; print or write
  qwen35_size_select.py --check PATH                    re-derive, byte-compare
  qwen35_size_select.py --query CONSUMER --host HOST    selected size + receipts
  qwen35_size_select.py --selftest                      planted case table
Exit: 0 ok | 1 --check mismatch or --query with nothing admissible | 2 bad input.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile

import yaml

SCHEMA = "apr-qwen35-size-selection/v1"
FAMILY = "qwen3.5"
LADDER_CONTRACT = "contracts/model-capability-ladder-v1.yaml"
CONTEXT_RUNGS = "evidence/release/context-rungs.json"
RECEIPT_DIR = "evidence/dogfood/models"
CONTEXT_RUNS = "evidence/pmat-3596-{host}/ladder.json"


class InputError(Exception):
    pass


def _load(root, rel, used):
    path = os.path.join(root, rel)
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise InputError(f"{rel}: {e.strerror}") from e
    used[rel] = hashlib.sha256(raw).hexdigest()
    return yaml.safe_load(raw) if rel.endswith(".yaml") else json.loads(raw)


def _vkey(v):
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"[.-]", v))


def newest_version(root):
    d = os.path.join(root, RECEIPT_DIR)
    vs = [v for v in os.listdir(d) if re.fullmatch(r"\d+\.\d+\.\d+", v)]
    if not vs:
        raise InputError(f"{RECEIPT_DIR}: no version directory")
    return max(vs, key=_vkey)


def _capability(rung, row, known_red):
    if row is None:
        return {"ok": False, "why": "no receipt row for this rung"}
    for kr in known_red:
        if kr.get("rung") == rung["id"] and kr.get("sha256") == row.get("sha256"):
            return {"ok": False, "why": f"known_red {kr.get('ticket', '?')}"}
    if row.get("green") is True:
        return {"ok": True, "why": "ladder rung green"}
    why = []
    if row.get("gates_failed"):
        why.append("gates_failed: " + ",".join(row["gates_failed"]))
    for b, v in sorted((row.get("backends") or {}).items()):
        sv = (v.get("verbs") or {}).get("serve") or {}
        if not sv.get("probed", True) and sv.get("why"):
            why.append(f"{b} serve: {re.sub(r' [(]log: .*[)]$', '', sv['why'])}")
    return {"ok": False, "why": "; ".join(why) or "ladder rung not green"}


def _context_rows(root, host, used):
    rel = CONTEXT_RUNS.format(host=host)
    if not os.path.exists(os.path.join(root, rel)):
        return rel, []
    rows = []
    for r in _load(root, rel, used):
        run_rel = os.path.join(os.path.dirname(rel), f"{r.get('tag')}.json")
        try:
            run = _load(root, run_rel, used)
        except (InputError, ValueError):
            continue
        rows.append((r, run, run_rel))
    return rel, rows


def _context(gguf, budget, rows):
    best, seen = None, 0
    for r, run, run_rel in rows:
        if os.path.basename(str(run.get("model", ""))) != gguf:
            continue
        n = r.get("prompt_tokens")
        if not isinstance(n, int) or str(r.get("rc")) != "0" or r.get("refused"):
            continue
        if run.get("used_gpu") is not True:
            continue
        seen = max(seen, n)
        if n >= budget and (best is None or n < best[0]):
            best = (n, r.get("tag"), r.get("version"), run_rel)
    if best is None:
        return {"met": False, "basis": "unmeasured",
                "why": f"no GPU run ingested >= {budget} tokens (max seen {seen})"}
    return {"met": True, "basis": "ingested", "prompt_tokens": best[0],
            "tag": best[1], "apr": best[2], "receipt": best[3]}


def derive(root, version=None):
    used = {}
    version = version or newest_version(root)
    contract = _load(root, LADDER_CONTRACT, used)["ladder"]
    rungs = [r for r in contract["rungs"] if r.get("family") == FAMILY]
    if not rungs:
        raise InputError(f"{LADDER_CONTRACT}: no {FAMILY} rungs")
    known_red = contract.get("known_red") or []
    consumers = _load(root, CONTEXT_RUNGS, used)["consumers"]
    hosts = {}
    for host in sorted(h if isinstance(h, str) else h["id"] for h in contract["hosts"]):
        rrel = f"{RECEIPT_DIR}/{version}/{host}.json"
        receipt = _load(root, rrel, used)
        by_id = {r["id"]: r for r in receipt.get("rungs", [])}
        inv = {i["file"]: i.get("bytes") for i in receipt.get("inventory", [])}
        ctx_rel, ctx_rows = _context_rows(root, host, used)
        cands = []
        for rung in rungs:
            row = by_id.get(rung["id"])
            b = inv.get(rung["gguf"])
            cands.append({"rung": rung["id"], "gguf": rung["gguf"],
                          "bytes": b if isinstance(b, int) else None,
                          "capability": _capability(rung, row, known_red)})
        # Cost order; a size with no byte count cannot be placed on the frontier.
        cands.sort(key=lambda c: (c["bytes"] is None, c["bytes"] or 0, c["rung"]))
        per = {}
        for c in consumers:
            budget = c.get("max_prompt_tokens")
            if not isinstance(budget, int):
                raise InputError(f"{CONTEXT_RUNGS}: consumer {c.get('consumer')} has no budget")
            rows = []
            for cand in cands:
                ctx = _context(cand["gguf"], budget, ctx_rows)
                ok = cand["bytes"] is not None and cand["capability"]["ok"] and ctx["met"]
                rows.append({**cand, "context": ctx, "admissible": ok})
            sel = next((r["rung"] for r in rows if r["admissible"]), None)
            per[c["consumer"]] = {"max_prompt_tokens": budget, "selected": sel,
                                  "admissible": [r["rung"] for r in rows if r["admissible"]],
                                  "candidates": rows}
        hosts[host] = {"receipt": rrel, "apr": receipt.get("apr_version"),
                       "context_runs": ctx_rel, "consumers": per}
    return {"schema": SCHEMA, "family": FAMILY, "ladder_version": version,
            "context_basis_note": "ingested = prefill run (max_tokens 1) completed on GPU "
                                  "at >= budget; correctness at that length is not measured",
            "derived_from": [{"path": p, "sha256": s} for p, s in sorted(used.items())],
            "hosts": hosts}


def render(table):
    return json.dumps(table, indent=2, sort_keys=True) + "\n"


# --- selftest -----------------------------------------------------------------

def _fixture(root, *, known_red=(), ctx=None, green=None, inv_missing=()):
    sizes = [("0.8b", "Qwen3.5-0.8B-Q4_K_M.gguf", 500), ("2b", "Qwen3.5-2B-Q4_K_M.gguf", 1200),
             ("9b", "Qwen3.5-9B-Q4_K_M.gguf", 5600)]
    green = green or {}
    rungs = [{"id": f"qwen35-{s}-q4km", "family": FAMILY, "gguf": g, "sha256": f"sha-{s}"}
             for s, g, _ in sizes]
    os.makedirs(os.path.join(root, "contracts"))
    with open(os.path.join(root, LADDER_CONTRACT), "w") as f:
        yaml.safe_dump({"ladder": {"hosts": ["lambda"], "rungs": rungs,
                                   "known_red": list(known_red)}}, f)
    os.makedirs(os.path.join(root, "evidence/release"))
    with open(os.path.join(root, CONTEXT_RUNGS), "w") as f:
        json.dump({"consumers": [{"consumer": "arb", "max_prompt_tokens": 7000}]}, f)
    os.makedirs(os.path.join(root, RECEIPT_DIR, "0.1.0"))
    rec = {"apr_version": "apr 0.1.0 (x)",
           "inventory": [{"file": g, "bytes": b} for s, g, b in sizes if s not in inv_missing],
           "rungs": [{"id": f"qwen35-{s}-q4km", "sha256": f"sha-{s}",
                      "green": green.get(s, True), "gates_failed": []} for s, _, _ in sizes]}
    with open(os.path.join(root, RECEIPT_DIR, "0.1.0", "lambda.json"), "w") as f:
        json.dump(rec, f)
    d = os.path.join(root, "evidence/pmat-3596-lambda")
    os.makedirs(d)
    ctx = ctx if ctx is not None else {s: dict(n=8000) for s, _, _ in sizes}
    ladder = []
    for s, g, _ in sizes:
        if s not in ctx:
            continue
        c = ctx[s]
        tag = f"{s}-p8k"
        ladder.append({"tag": tag, "version": "apr 0.1.0 (x)", "prompt_tokens": c.get("n"),
                       "rc": c.get("rc", "0"), "refused": c.get("refused")})
        with open(os.path.join(d, f"{tag}.json"), "w") as f:
            json.dump({"model": f"$HOME/models/{g}", "used_gpu": c.get("gpu", True)}, f)
    with open(os.path.join(d, "ladder.json"), "w") as f:
        json.dump(ladder, f)


CASES = [
    # (name, fixture kwargs, expected selected rung for lambda/arb)
    ("baseline: smallest green+ingested size wins", {}, "qwen35-0.8b-q4km"),
    ("known_red sha-match demotes 0.8b",
     {"known_red": [{"rung": "qwen35-0.8b-q4km", "sha256": "sha-0.8b", "ticket": "#1"}]},
     "qwen35-2b-q4km"),
    ("known_red on a different sha does not apply",
     {"known_red": [{"rung": "qwen35-0.8b-q4km", "sha256": "other", "ticket": "#1"}]},
     "qwen35-0.8b-q4km"),
    ("red ladder rung is not admissible", {"green": {"0.8b": False}}, "qwen35-2b-q4km"),
    ("no context run for 0.8b = unmeasured, not a pass",
     {"ctx": {"2b": dict(n=8000), "9b": dict(n=8000)}}, "qwen35-2b-q4km"),
    ("context run below budget does not meet it",
     {"ctx": {"0.8b": dict(n=6000), "9b": dict(n=8000)}}, "qwen35-9b-q4km"),
    ("refused run does not meet the budget",
     {"ctx": {"0.8b": dict(n=8000, refused="OOM"), "9b": dict(n=8000)}}, "qwen35-9b-q4km"),
    ("rc != 0 does not meet the budget",
     {"ctx": {"0.8b": dict(n=8000, rc="8"), "9b": dict(n=8000)}}, "qwen35-9b-q4km"),
    ("used_gpu false does not meet the budget",
     {"ctx": {"0.8b": dict(n=8000, gpu=False), "9b": dict(n=8000)}}, "qwen35-9b-q4km"),
    ("no byte count = cannot be placed, not admissible",
     {"inv_missing": ("0.8b",)}, "qwen35-2b-q4km"),
    ("nothing admissible selects nothing, never the largest",
     {"green": {"0.8b": False, "2b": False, "9b": False}}, None),
]


def selftest():
    fails = 0
    for name, kw, want in CASES:
        with tempfile.TemporaryDirectory() as root:
            _fixture(root, **kw)
            a = render(derive(root))
            b = render(derive(root))
            got = json.loads(a)["hosts"]["lambda"]["consumers"]["arb"]["selected"]
            ok = got == want and a == b
            fails += not ok
            print(f"{'PASS' if ok else 'FAIL'}  {name}: selected={got} want={want}"
                  + ("" if a == b else " (NON-DETERMINISTIC)"))
    print(f"selftest: {len(CASES) - fails}/{len(CASES)} pass")
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--version")
    ap.add_argument("--out")
    ap.add_argument("--check")
    ap.add_argument("--query")
    ap.add_argument("--host")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    try:
        table = derive(a.root, a.version)
    except (InputError, KeyError, ValueError) as e:
        print(f"qwen35_size_select: bad input: {e}", file=sys.stderr)
        return 2
    text = render(table)
    if a.check:
        with open(a.check) as f:
            if f.read() != text:
                print(f"{a.check}: STALE — re-derive with --out {a.check}", file=sys.stderr)
                return 1
        print(f"{a.check}: re-derived byte-identical")
        return 0
    if a.query:
        h = table["hosts"].get(a.host or "")
        c = h and h["consumers"].get(a.query)
        if not c:
            print(f"unknown host/consumer: {a.host}/{a.query}", file=sys.stderr)
            return 2
        sel = next((r for r in c["candidates"] if r["rung"] == c["selected"]), None)
        print(f"{a.host}/{a.query} (budget {c['max_prompt_tokens']}): selected {c['selected']}")
        if sel:
            print(f"  capability: {h['receipt']} ({sel['capability']['why']})")
            print(f"  context:    {sel['context']['receipt']} "
                  f"({sel['context']['prompt_tokens']} tokens, {sel['context']['basis']})")
        return 0 if sel else 1
    if a.out:
        with open(a.out, "w") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
