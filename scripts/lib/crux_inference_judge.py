#!/usr/bin/env python3
"""crux_inference_judge.py: the judge of the CRUX inference dogfood (#3739).

scripts/crux_inference_dogfood.sh DRIVES the engines and writes one manifest
line per invocation, pointing at that invocation's raw stdout/stderr. This file
READS those files and nothing else: it starts no engine and opens no socket.
The split is deliberate. The judge is the part the case table
(scripts/check_crux_inference_judge.sh) can drive with fixtures, with no model,
GPU or comparator, in milliseconds.

THE ONE RULE (issue #3739 done_when 3). A cell is keyed by (model sha256, host,
verb, thinking, rung, prompt). It is RED when llama.cpp OR ollama answered it
correctly and apr did not. "apr did not" includes every way of not answering:
a non-zero exit, a backend that fell back to one the lane did not ask for, a
refusal, output the judge cannot parse, and an apr row that is simply MISSING.
Absence is a violation here, never a skip.

The other outcomes are reported and never scored as a pass:
  GREEN      apr correct, and at least one comparator ANSWERED (right or wrong)
  UNJUDGED   no comparator answered, so there is no external oracle for the cell
  ALL_WRONG  the comparators answered, none correctly, and apr was wrong too
A run with no GREEN cell declines (exit 2): a receipt that compared apr to
nothing, or only to wrong answers, is not evidence that apr is right.

PERFORMANCE IS TRANSCRIBED, NEVER COMPUTED. Token counts and rates are copied
from what each engine prints about itself, labelled with the engine that
reported them, and never ratioed, averaged or judged (the withdrawn-headline
rule, docs/BEATS.md). This file derives no rate from a clock, and no field it
writes is an input to the verdict.

Usage:
  crux_inference_judge.py collect --manifest M --prompts P --meta META \\
      --out-json R.json --out-md R.md
Exit: 0 no RED and at least one GREEN cell; 1 any RED; 2 decline.
"""

import argparse
import datetime
import json
import re
import sys

ENGINES = ("apr", "llama.cpp", "ollama")
COMPARATORS = ("llama.cpp", "ollama")
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07|\r")


def read_text(path):
    if not path:
        return ""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def norm(text):
    return " ".join(text.split())


# ---------------------------------------------------------------- parsers --
# Each parser returns {"answer": str|None, "why": str|None, "reported": {...},
# plus engine-specific fields}. answer=None means "did not answer" and `why`
# says what the judge saw instead.


def parse_apr(stdout, stderr):
    """apr's `run` verb with `--format json -v`: the answer is stdout JSON; -v
    puts the prompt apr actually built, and its token ids, on stderr."""
    out = {"answer": None, "why": None, "reported": {}, "backend": None,
           "prompt_ids": None, "prompt_token_count": None, "rendered_prompt": None}
    m = re.search(r'formatted_prompt="((?:[^"\\]|\\.)*)"', stderr)
    if m:
        out["rendered_prompt"] = m.group(1)
    m = re.search(r"encoded (\d+) tokens: \[([0-9, ]*)\]", stderr)
    if m:
        out["prompt_token_count"] = int(m.group(1))
        out["prompt_ids"] = [int(x) for x in m.group(2).replace(" ", "").split(",") if x]
    start = stdout.find("{")
    if start < 0:
        out["why"] = "no JSON object on stdout"
        return out
    try:
        doc, _ = json.JSONDecoder().raw_decode(stdout[start:])
    except ValueError as exc:
        out["why"] = "stdout JSON unparseable: %s" % exc
        return out
    out["backend"] = doc.get("backend")
    out["reported"] = {
        "reported_by": "apr",
        "prompt_tokens": out["prompt_token_count"],
        "completion_tokens": doc.get("tokens_generated"),
        "decode_rate_tokens_per_second": doc.get("tok_per_sec"),
        "inference_ms": doc.get("inference_time_ms"),
    }
    text = doc.get("text")
    if not isinstance(text, str):
        out["why"] = "stdout JSON has no text field"
        return out
    out["answer"] = text
    return out


RATE_LINE = re.compile(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]")


def parse_llamacpp_cli(stdout, prompt_text):
    """The pinned llama.cpp chat CLI echoes `> <prompt>`, streams the answer,
    then prints `[ Prompt: X t/s | Generation: Y t/s ]`. The answer is the text
    between the echoed prompt and that line, and nothing else is guessed."""
    out = {"answer": None, "why": None, "reported": {}}
    text = ANSI.sub("", stdout)
    echo = "> " + prompt_text
    at = text.rfind(echo)
    if at < 0:
        out["why"] = "the echoed prompt was not found in stdout"
        return out
    rest = text[at + len(echo):]
    m = RATE_LINE.search(rest)
    if not m:
        out["why"] = "no end-of-turn timing line after the echoed prompt"
        return out
    out["answer"] = rest[:m.start()].strip()
    out["reported"] = {
        "reported_by": "llama.cpp",
        "prompt_rate_tokens_per_second": float(m.group(1)),
        "decode_rate_tokens_per_second": float(m.group(2)),
        "prompt_tokens": None,
        "completion_tokens": None,
    }
    return out


OLLAMA_STAT = re.compile(r"^(total duration|load duration|prompt eval count|prompt eval duration|"
                         r"prompt eval rate|eval count|eval duration|eval rate):\s*(.+?)\s*$", re.M)


def parse_ollama(stdout, stderr):
    out = {"answer": None, "why": None, "reported": {}}
    stats = {k: v for k, v in OLLAMA_STAT.findall(ANSI.sub("", stderr))}

    def num(key):
        m = re.match(r"([0-9.]+)", stats.get(key, ""))
        return float(m.group(1)) if m else None

    pe, ev = num("prompt eval count"), num("eval count")
    out["reported"] = {
        "reported_by": "ollama",
        "prompt_tokens": int(pe) if pe is not None else None,
        "completion_tokens": int(ev) if ev is not None else None,
        "prompt_rate_tokens_per_second": num("prompt eval rate"),
        "decode_rate_tokens_per_second": num("eval rate"),
        "load_duration": stats.get("load duration"),
        "total_duration": stats.get("total duration"),
    }
    answer = ANSI.sub("", stdout).strip()
    if not answer:
        out["why"] = "empty stdout"
        return out
    out["answer"] = answer
    return out


# ------------------------------------------------------------------ judge --


def correct(entry, expect_any):
    return entry.get("answered", False) and any(p in (entry.get("answer") or "") for p in expect_any)


def engine_entry(row, prompt):
    """One engine's answer to one cell, from its manifest row."""
    e = {"answered": False, "rc": row.get("rc"), "why": None, "answer": None, "reported": {}}
    if row.get("refused"):
        e["why"] = "refused: " + row["refused"]
        return e
    stdout, stderr = read_text(row.get("stdout")), read_text(row.get("stderr"))
    content = prompt["messages"][-1]["content"]
    engine = row["engine"]
    if engine == "apr":
        p = parse_apr(stdout, stderr)
        e["backend"] = p["backend"]
        e["prompt_ids"] = p["prompt_ids"]
        e["prompt_token_count"] = p["prompt_token_count"]
        e["rendered_prompt"] = p["rendered_prompt"]
    elif engine == "llama.cpp":
        p = parse_llamacpp_cli(stdout, content)
    elif engine == "ollama":
        p = parse_ollama(stdout, stderr)
    else:
        e["why"] = "unknown engine %r" % engine
        return e
    e["reported"] = p["reported"]
    e["answer"] = p["answer"]
    if row.get("rc") != 0:
        e["why"] = "exit %s" % row.get("rc")
        return e
    if p["answer"] is None:
        e["why"] = p["why"]
        return e
    be = p.get("backend") if engine == "apr" else None
    if be and (be.get("fell_back") or (row.get("backend") and be.get("ran") != row.get("backend"))):
        e["why"] = "backend: asked %s, ran %s (fell_back=%s)" % (row.get("backend"), be.get("ran"), be.get("fell_back"))
        return e
    e["answered"] = True
    return e


def token_parity(apr_entry, tok_row):
    """apr's prompt ids (from -v) against llama.cpp's ids for ITS rendering of
    the same messages with the GGUF's own template. apr prints at most a prefix
    of its ids, so the comparison is the count plus that prefix, and says so."""
    if tok_row is None:
        return {"measured": False, "why": "no llama.cpp tokenization row"}
    try:
        with open(tok_row["ids"], encoding="utf-8") as fh:
            ref = json.load(fh).get("tokens")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"measured": False, "why": "llama.cpp ids unreadable: %s" % exc}
    if not isinstance(ref, list):
        return {"measured": False, "why": "llama.cpp ids missing"}
    ids, n = apr_entry.get("prompt_ids"), apr_entry.get("prompt_token_count")
    if ids is None or n is None:
        return {"measured": False, "why": "apr printed no prompt ids", "llama_cpp_count": len(ref)}
    first = next((i for i, (a, b) in enumerate(zip(ids, ref)) if a != b), None)
    if first is None and len(ids) > len(ref):
        first = len(ref)
    return {
        "measured": True,
        "apr_count": n,
        "llama_cpp_count": len(ref),
        "apr_ids_compared": len(ids),
        "apr_ids_complete": len(ids) == n,
        "first_divergence": first,
        "parity": n == len(ref) and first is None,
    }


def judge_cell(entries, expect_any):
    ok = {k: correct(v, expect_any) for k, v in entries.items()}
    answered = {k: v.get("answered", False) for k, v in entries.items()}
    if any(ok.get(c) for c in COMPARATORS) and not ok.get("apr"):
        return "RED", ok
    if not any(answered.get(c) for c in COMPARATORS):
        return "UNJUDGED", ok
    if ok.get("apr"):
        return "GREEN", ok
    return "ALL_WRONG", ok


def collect(args):
    with open(args.prompts, encoding="utf-8") as fh:
        prompts = {p["id"]: p for p in json.load(fh)["prompts"]}
    with open(args.meta, encoding="utf-8") as fh:
        meta = json.load(fh)
    rows = []
    with open(args.manifest, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    gens = [r for r in rows if r.get("kind") == "gen"]
    toks = {(r["model_sha256"], r["prompt_id"]): r for r in rows if r.get("kind") == "tok"}
    requested = meta.get("engines", list(ENGINES))

    keys = []
    by_key = {}
    for r in gens:
        k = (r["model_sha256"], r["host"], r["verb"], r["thinking"], prompts[r["prompt_id"]]["rung"], r["prompt_id"])
        if k not in by_key:
            keys.append(k)
            by_key[k] = {}
        by_key[k][r["engine"]] = r

    cells = []
    for k in keys:
        prompt = prompts[k[5]]
        entries = {}
        for eng in ENGINES:
            row = by_key[k].get(eng)
            if row is None:
                entries[eng] = {"answered": False, "missing": True,
                                "why": "missing: no row for this engine" if eng in requested else "not requested"}
            else:
                entries[eng] = engine_entry(row, prompt)
        verdict, ok = judge_cell(entries, prompt["expect_any"])
        for eng in ENGINES:
            entries[eng]["correct"] = ok[eng]
        said = {e: norm(v["answer"]) for e, v in entries.items() if v.get("answered")}
        cells.append({
            "key": dict(zip(("model_sha256", "host", "verb", "thinking", "rung", "prompt_id"), k)),
            "verdict": verdict,
            "expect_any": prompt["expect_any"],
            "engines": entries,
            "agreement": {
                "answered": sorted(said),
                "all_identical": len(said) >= 2 and len(set(said.values())) == 1,
                "apr_matches": sorted(e for e in said if e != "apr" and "apr" in said and said[e] == said["apr"]),
            },
            "token_parity": token_parity(entries["apr"], toks.get((k[0], k[5]))),
        })

    counts = {v: sum(1 for c in cells if c["verdict"] == v) for v in ("RED", "GREEN", "UNJUDGED", "ALL_WRONG")}
    judged = counts["RED"] + counts["GREEN"] + counts["ALL_WRONG"]
    if counts["RED"]:
        verdict, rc = "RED", 1
    elif counts["GREEN"] == 0:
        # Nothing shows apr right where a competitor answered: every cell was
        # unjudged or all-wrong. That is a broken run or prompt set, not a pass.
        verdict, rc = "DECLINE", 2
    else:
        verdict, rc = "PASS", 0
    receipt = dict(meta)
    receipt.update({
        "schema": "crux-inference-receipt/v1",
        "judged_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "cells": cells,
        "summary": dict(counts, cells=len(cells), judged=judged, verdict=verdict),
    })
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(receipt, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    md = render_md(receipt)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write(md)
    sys.stdout.write(md)
    return rc


def short(text, n=48):
    t = norm(text or "")
    return (t[:n - 1] + "…") if len(t) > n else t


def render_md(r):
    s = r["summary"]
    lines = [
        "# CRUX inference dogfood: %s on %s (%s lane)" % (r.get("version"), r.get("host"), r.get("backend")),
        "",
        "apr `%s` · llama.cpp `%s` · ollama `%s` · judged %s" % (
            r.get("apr", {}).get("version_line"), r.get("llama_cpp", {}).get("build"),
            r.get("ollama", {}).get("server_version"), r["judged_at"]),
        "",
        "**%s**: %d cells, %d RED, %d GREEN, %d ALL_WRONG, %d UNJUDGED." % (
            s["verdict"], s["cells"], s["RED"], s["GREEN"], s["ALL_WRONG"], s["UNJUDGED"]),
        "",
        "| model | verb | thinking | prompt | verdict | apr | llama.cpp | ollama | token parity |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    names = {m["sha256"]: m.get("name", m["sha256"][:12]) for m in r.get("models", [])}
    for c in r["cells"]:
        k = c["key"]
        cols = []
        for e in ENGINES:
            v = c["engines"][e]
            mark = "✅" if v["correct"] else ("❌" if v.get("answered") else "⛔")
            cols.append("%s %s" % (mark, short(v["answer"]) if v.get("answered") else short(v.get("why"), 60)))
        tp = c["token_parity"]
        if tp.get("measured"):
            tps = "%s (apr %d vs %d; first diff %s)" % ("=" if tp["parity"] else "≠", tp["apr_count"],
                                                         tp["llama_cpp_count"], tp["first_divergence"])
        else:
            tps = "unmeasured: " + tp.get("why", "")
        lines.append("| %s | %s | %s | %s | **%s** | %s | %s |" % (
            names.get(k["model_sha256"], k["model_sha256"][:12]), k["verb"], k["thinking"], k["prompt_id"],
            c["verdict"], " | ".join(cols), tps))
    lines += ["", "✅ correct · ❌ answered, wrong · ⛔ did not answer (reason shown).",
              "Rates and token counts are in the JSON, as each engine reported them, and are not judged.", ""]
    nc = r.get("not_covered")
    if nc:
        lines += ["Not covered by this run (the issue requires them): " + "; ".join(nc) + ".", ""]
    return "\n".join(lines)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    for flag in ("--manifest", "--prompts", "--meta", "--out-json", "--out-md"):
        c.add_argument(flag, required=True)
    args = ap.parse_args(argv)
    try:
        return collect(args)
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write("decline: %s: %s\n" % (type(exc).__name__, exc))
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
