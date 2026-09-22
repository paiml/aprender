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

The other outcomes:
  GREEN      apr correct, and at least one comparator ANSWERED (right or wrong)
  UNJUDGED   no comparator answered, so there is no external oracle for the cell
  ALL_WRONG  the comparators answered, none correctly, and apr was wrong too
ANY UNJUDGED cell declines the run (exit 2): it was never compared, and the
amended scope makes a missing cell a NO-GO.

ALL_WRONG is NAMED, never a violation and never a pass (the cop's ruling, #3739:
the bar is "apr right wherever a competitor is right"). Two things keep it from
becoming the escape: the receipt counts ALL_WRONG per model, and the prompt set
must declare a POSITIVE CONTROL (`"control": true`), a prompt the smallest model
answers right on every engine. Every model must have a measured control cell.
A control cell that comes back ALL_WRONG is a broken harness, not a model
limitation, and declines the run; so does a model with no control cell, and a
prompt set that declares no control at all.

PERFORMANCE IS TRANSCRIBED, NEVER COMPUTED. Token counts and rates are copied
from what each engine prints about itself, labelled with the engine that
reported them, and never ratioed, averaged or judged (the withdrawn-headline
rule, docs/BEATS.md). This file derives no rate from a clock, and no field it
writes is an input to the verdict.

Usage:
  crux_inference_judge.py collect --manifest M --prompts P --meta META \\
      --out-json R.json --out-md R.md
Exit: 0 no RED, no UNJUDGED, every model's control measured and not ALL_WRONG;
1 any RED; 2 decline.
"""

import argparse
import ast
import datetime
import json
import math
import re
import struct
import sys

COMPARATORS = ("llama.cpp", "ollama", "hf", "llamafile")
ENGINES = ("apr",) + COMPARATORS
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
        "decode_rate": doc.get("tok_per_sec"),
        "rate_unit": "tokens per second, as the engine reported it",
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
        "prompt_rate": float(m.group(1)),
        "decode_rate": float(m.group(2)),
        "rate_unit": "tokens per second, as the engine reported it",
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
        "prompt_rate": num("prompt eval rate"),
        "decode_rate": num("eval rate"),
        "rate_unit": "tokens per second, as the engine reported it",
        "load_duration": stats.get("load duration"),
        "total_duration": stats.get("total duration"),
    }
    answer = ANSI.sub("", stdout).strip()
    if not answer:
        out["why"] = "empty stdout"
        return out
    out["answer"] = answer
    return out


def parse_apr_chat(stdout):
    """apr's `chat` verb, fed one user turn per stdin line: each reply follows an
    `Assistant: ` prefix and runs to the next `You:` prompt. The final reply is
    the answer; every reply is recorded."""
    out = {"answer": None, "why": None, "reported": {"reported_by": "apr"}, "turns": []}
    text = ANSI.sub("", stdout)
    for seg in re.split(r"(?m)^Assistant: ", text)[1:]:
        cut = re.search(r"(?m)^You:", seg)
        out["turns"].append((seg[:cut.start()] if cut else seg).strip())
    if not out["turns"]:
        out["why"] = "no `Assistant:` turn in the transcript"
        return out
    out["answer"] = out["turns"][-1]
    return out


def parse_engine_json(stdout):
    """hf and llamafile rows (row contract v1, #3739 issuecomment-5765991210):
    the engine driver writes `{"text": <the answer only>, "reported": {...}}`."""
    out = {"answer": None, "why": None, "reported": {}}
    try:
        doc = json.loads(stdout)
    except ValueError as exc:
        out["why"] = "stdout is not the contract's JSON: %s" % exc
        return out
    if not isinstance(doc, dict) or not isinstance(doc.get("text"), str):
        out["why"] = "stdout JSON has no text field"
        return out
    rep = doc.get("reported")
    out["reported"] = rep if isinstance(rep, dict) else {}
    out["answer"] = doc["text"]
    if isinstance(doc.get("turns"), list):
        out["turns"] = doc["turns"]
    return out


# ------------------------------------------------------------------ judge --


def degenerate(text):
    """A degenerate answer is NO answer, from any engine. Measured on #3774's
    integration run: a broken HF load emitted token id 0 ("!") 64 times, and
    the golden greeting's own pattern "!" (golden_output.rs) scored that CORRECT.
    apr's #3726 failure mode is the same shape. Threshold: at least 8 non-space
    characters, 90% or more of them one character, which no real answer to the
    golden prompts ("4", "The capital of France is Paris.") comes near."""
    chars = [ch for ch in (text or "") if not ch.isspace()]
    if len(chars) < 8:
        return False
    top = max(chars.count(ch) for ch in set(chars))
    return top >= 0.9 * len(chars)


def correct(entry, expect_any):
    return entry.get("answered", False) and any(p in (entry.get("answer") or "") for p in expect_any)


def engine_entry(row, prompt):
    """One engine's answer to one cell, from its manifest row."""
    e = {"answered": False, "rc": row.get("rc"), "why": None, "answer": None, "reported": {}}
    if "ollama_unloaded" in row:
        # Did ollama's model leave VRAM before the cell dropped the GPU lock?
        # Recorded, never judged: it is a property of the harness, not an answer.
        e["vram_released"] = row["ollama_unloaded"]
    if row.get("refused"):
        e["why"] = "refused: " + row["refused"]
        return e
    stdout, stderr = read_text(row.get("stdout")), read_text(row.get("stderr"))
    content = prompt["messages"][-1]["content"]
    engine = row["engine"]
    if row.get("verb") == "serve run":
        # serve (#3739 slice 4): every server, apr's included, is read through the ONE
        # OpenAI client's contract JSON (scripts/lib/crux_openai_client.py).
        p = parse_engine_json(stdout)
        if engine == "apr":
            # apr serve's responses report no backend: recorded, not scored as verified.
            e["backend_verified"] = False
        elif engine not in COMPARATORS:
            e["why"] = "unknown engine %r" % engine
            return e
    elif row.get("verb") == "chat":
        # chat (#3739 slice 3): apr's own transcript; every other engine writes the
        # row-contract JSON (the pty helper for llama.cpp/ollama, the plugin drivers).
        if engine == "apr":
            p = parse_apr_chat(stdout)
            # apr chat reports no backend (#3794): recorded, not scored as verified.
            e["backend_verified"] = False
        elif engine in COMPARATORS:
            p = parse_engine_json(stdout)
        else:
            e["why"] = "unknown engine %r" % engine
            return e
        e["turns"] = p.get("turns") or []
    elif engine == "apr":
        p = parse_apr(stdout, stderr)
        e["backend"] = p["backend"]
        e["prompt_ids"] = p["prompt_ids"]
        e["prompt_token_count"] = p["prompt_token_count"]
        e["rendered_prompt"] = p["rendered_prompt"]
    elif engine == "llama.cpp":
        p = parse_llamacpp_cli(stdout, content)
    elif engine == "ollama":
        p = parse_ollama(stdout, stderr)
    elif engine in ("hf", "llamafile"):
        p = parse_engine_json(stdout)
        if row.get("source"):
            e["source"] = row["source"]
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
    if degenerate(p["answer"]):
        e["why"] = "degenerate output (one character is >=90%% of it): %r" % p["answer"][:24]
        return e
    be = p.get("backend") if engine == "apr" else None
    if be and (be.get("fell_back") or (row.get("backend") and be.get("ran") != row.get("backend"))):
        e["why"] = "backend: asked %s, ran %s (fell_back=%s)" % (row.get("backend"), be.get("ran"), be.get("fell_back"))
        return e
    if engine in ("hf", "llamafile"):
        # A plugin engine is held to the lane as apr is. The integration run's
        # cpu lane got `!` x64 from an HF load that went to CUDA anyway (and
        # outside the GPU lock). The device must be REPORTED: an unverifiable
        # lane cannot vouch for or against apr.
        dev = str((p.get("reported") or {}).get("device") or "")
        lane = row.get("backend")
        if not dev:
            e["why"] = "no reported.device: the %s lane cannot be verified" % lane
            return e
        if (lane == "cpu") != dev.lower().startswith("cpu"):
            e["why"] = "device %r is not the %s lane" % (dev, lane)
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


# ------------------------------------------- deterministic rows (row contract v1) --
# tok and tmpl are BYTE-EQUAL or RED (#3739, 19:03Z): an engine that produced
# ids / a rendering and an apr that produced a different one, or none, is RED.
# greedy is REPORTED: the judge holds no tokenizer, so it cannot decide from ids
# alone whether a divergence changed the answer; that verdict stays with `gen`.


def _load_json(path, field):
    with open(path, encoding="utf-8") as fh:
        v = json.load(fh).get(field)
    if not isinstance(v, list):
        raise ValueError("%s has no list %r" % (path, field))
    return v


def _load_bytes(path):
    with open(path, "rb") as fh:
        return fh.read()


def _first_diff(a, b):
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if i is None and len(a) != len(b):
        i = min(len(a), len(b))
    return i


def _det_side(row, loader, field):
    if row is None:
        return None, "missing: no row for this engine"
    if row.get("refused"):
        return None, "refused: " + row["refused"]
    try:
        return (loader(row[field]) if field else None), None
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, "unreadable: %s" % exc


def judge_deterministic(rows, kind):
    """kind 'tok': rows carrying `input` (raw-text ids). kind 'tmpl': chat renderings."""
    if kind == "tok":
        rows = [r for r in rows if r.get("kind") == "tok" and "input" in r]
        keyf = lambda r: (r["model_sha256"], r["host"], r["prompt_id"])
        names = ("model_sha256", "host", "prompt_id")
        load = lambda r: _det_side(r, lambda p: _load_json(p, "tokens"), "ids")
    else:
        rows = [r for r in rows if r.get("kind") == "tmpl"]
        keyf = lambda r: (r["model_sha256"], r["host"], r["prompt_id"], r.get("thinking", "unset"))
        names = ("model_sha256", "host", "prompt_id", "thinking")
        load = lambda r: _det_side(r, _load_bytes, "rendered")
    groups = {}
    for r in rows:
        groups.setdefault(keyf(r), {})[r["engine"]] = r
    out = []
    for key in sorted(groups):
        by = groups[key]
        apr_val, apr_why = load(by.get("apr"))
        refs = {}
        for eng, r in sorted(by.items()):
            if eng == "apr":
                continue
            val, why = load(r)
            refs[eng] = {"produced": val is not None, "why": why}
            if val is not None and apr_val is not None:
                d = _first_diff(apr_val, val)
                refs[eng]["equal"] = d is None
                refs[eng]["first_difference"] = d
        produced = [e for e, v in refs.items() if v["produced"]]
        if not produced:
            verdict = "UNJUDGED"
        elif apr_val is None or any(not refs[e]["equal"] for e in produced):
            verdict = "RED"
        else:
            verdict = "GREEN"
        out.append({"kind": kind, "key": dict(zip(names, key)), "verdict": verdict,
                    "apr": {"produced": apr_val is not None, "why": apr_why}, "references": refs})
    return out


def _npy_rows(path):
    """A float32 C-order 2-D .npy, read with the stdlib (runners have no numpy)."""
    data = _load_bytes(path)
    if data[:6] != b"\x93NUMPY":
        raise ValueError("not an .npy file")
    major = data[6]
    hlen = struct.unpack("<H" if major == 1 else "<I", data[8:10] if major == 1 else data[8:12])[0]
    start = (10 if major == 1 else 12) + hlen
    hdr = ast.literal_eval(data[(10 if major == 1 else 12):start].decode("latin1"))
    if hdr.get("descr") != "<f4" or hdr.get("fortran_order") or len(hdr.get("shape", ())) != 2:
        raise ValueError("expected little-endian float32 C-order 2-D, got %r" % (hdr,))
    n, m = hdr["shape"]
    flat = struct.unpack("<%df" % (n * m), data[start:start + 4 * n * m])
    return [flat[i * m:(i + 1) * m] for i in range(n)]


def _cosine(a, b):
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else None


def report_greedy(rows):
    rows = [r for r in rows if r.get("kind") == "greedy"]
    groups = {}
    for r in rows:
        groups.setdefault((r["model_sha256"], r["host"], r["prompt_id"]), {})[r["engine"]] = r
    out = []
    for key in sorted(groups):
        by = groups[key]
        rep = {"key": dict(zip(("model_sha256", "host", "prompt_id"), key)), "engines": sorted(by)}
        apr = by.get("apr")
        for eng, r in sorted(by.items()):
            if eng == "apr":
                continue
            try:
                if apr is None or apr.get("refused") or r.get("refused"):
                    raise ValueError("apr or %s has no greedy row" % eng)
                a = _load_json(apr["tokens"], "generated_ids")
                b = _load_json(r["tokens"], "generated_ids")
                d = _first_diff(a, b)
                item = {"first_divergence": d, "steps_compared": min(len(a), len(b))}
                if d is not None and apr.get("logits") and r.get("logits"):
                    la, lb = _npy_rows(apr["logits"]), _npy_rows(r["logits"])
                    if d < len(la) and d < len(lb):
                        item["logit_cosine_at_divergence"] = _cosine(la[d], lb[d])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                item = {"why": "not compared: %s" % exc}
            rep[eng] = item
        out.append(rep)
    return out


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
    # llama.cpp's template-level ids feed the REPORTED token_parity field; raw-text
    # `tok` rows (they carry `input`) are the byte-equal deterministic rows below.
    toks = {(r["model_sha256"], r["prompt_id"]): r for r in rows
            if r.get("kind") == "tok" and r.get("engine") == "llama.cpp" and "input" not in r}
    det = judge_deterministic(rows, "tok") + judge_deterministic(rows, "tmpl")
    greedy = report_greedy(rows)
    requested = meta.get("engines", list(ENGINES))

    keys = []
    by_key = {}
    for r in gens:
        # serve cells come in two modes (nonstream, stream): an additive key part,
        # absent for run and chat. Plugin serve rows are non-streaming.
        mode = r.get("mode") or ("nonstream" if r["verb"] == "serve run" else "")
        k = (r["model_sha256"], r["host"], r["verb"], r["thinking"], prompts[r["prompt_id"]]["rung"], r["prompt_id"], mode)
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
            "key": dict(zip(("model_sha256", "host", "verb", "thinking", "rung", "prompt_id"), k[:6]),
                        **({"mode": k[6]} if k[6] else {})),
            "verdict": verdict,
            # additive (aprender-97, #3715): pv reads the control by this flag, never by a prompt name
            "positive_control": bool(prompt.get("control")),
            "expect_any": prompt["expect_any"],
            "engines": entries,
            "agreement": {
                "answered": sorted(said),
                "all_identical": len(said) >= 2 and len(set(said.values())) == 1,
                "apr_matches": sorted(e for e in said if e != "apr" and "apr" in said and said[e] == said["apr"]),
            },
            "token_parity": (token_parity(entries["apr"], toks.get((k[0], k[5]))) if k[2] == "run"
                             else {"measured": False, "why": "not measured for the %s verb" % k[2]}),
        })

    counts = {v: sum(1 for c in cells if c["verdict"] == v) for v in ("RED", "GREEN", "UNJUDGED", "ALL_WRONG")}
    judged = counts["RED"] + counts["GREEN"] + counts["ALL_WRONG"]
    all_wrong_by_model = {}
    for c in cells:
        if c["verdict"] == "ALL_WRONG":
            k = c["key"]["model_sha256"]
            all_wrong_by_model[k] = all_wrong_by_model.get(k, 0) + 1
    controls = [pid for pid, p in prompts.items() if p.get("control")]
    broken = sorted({"%s/%s" % (c["key"]["model_sha256"][:12], c["key"]["prompt_id"])
                     for c in cells if c["key"]["prompt_id"] in controls and c["verdict"] == "ALL_WRONG"})
    models = sorted({c["key"]["model_sha256"] for c in cells})
    uncontrolled = [m[:12] for m in models
                    if not any(c["key"]["model_sha256"] == m and c["key"]["prompt_id"] in controls for c in cells)]
    declined_because = None
    if not controls:
        declined_because = "the prompt set declares no positive control (\"control\": true)"
    elif not cells:
        declined_because = "no cell was measured"
    elif uncontrolled:
        declined_because = "no positive-control cell was measured for model(s) " + ", ".join(uncontrolled)
    elif broken:
        declined_because = "positive control came back ALL_WRONG (a broken harness, not a model limit): " + ", ".join(broken)
    det_counts = {v: sum(1 for d in det if d["verdict"] == v) for v in ("RED", "GREEN", "UNJUDGED")}
    if counts["RED"] or det_counts["RED"]:
        verdict, rc = "RED", 1
    elif declined_because:
        verdict, rc = "DECLINE", 2
    elif counts["UNJUDGED"] or det_counts["UNJUDGED"]:
        # An UNJUDGED cell was never compared to anything: under the amended
        # scope (#3739, 17:19Z) "a missing cell is a NO-GO", and an unmeasured
        # cell is a missing one. (A run with no GREEN cannot reach PASS: every
        # model has a control cell, and a control that is not RED, UNJUDGED or
        # ALL_WRONG is GREEN.)
        verdict, rc = "DECLINE", 2
    else:
        verdict, rc = "PASS", 0
    receipt = dict(meta)
    receipt.update({
        "schema": "crux-inference-receipt/v1",
        "judged_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "cells": cells,
        "deterministic": det,
        "greedy": greedy,
        "summary": dict(counts, cells=len(cells), judged=judged, verdict=verdict, deterministic=det_counts,
                        all_wrong_by_model=all_wrong_by_model, controls=controls,
                        declined_because=declined_because if verdict == "DECLINE" else None),
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
        "apr `%s` · llama.cpp `%s` · ollama `%s` · hf `%s` · llamafile `%s` · judged %s" % (
            r.get("apr", {}).get("version_line"), r.get("llama_cpp", {}).get("build"),
            r.get("ollama", {}).get("server_version"), r.get("hf", {}).get("probe"),
            r.get("llamafile", {}).get("probe"), r["judged_at"]),
        "",
        "**%s**: %d cells, %d RED, %d GREEN, %d ALL_WRONG, %d UNJUDGED." % (
            s["verdict"], s["cells"], s["RED"], s["GREEN"], s["ALL_WRONG"], s["UNJUDGED"]),
        "",
        "| model | verb | thinking | prompt | verdict | %s | token parity |" % " | ".join(ENGINES),
        "|---|---|---|---|---|%s---|" % ("---|" * len(ENGINES)),
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
            names.get(k["model_sha256"], k["model_sha256"][:12]), k["verb"], k["thinking"],
            k["prompt_id"] + ("@" + k["mode"] if k.get("mode") else ""),
            c["verdict"], " | ".join(cols), tps))
    if r.get("deterministic"):
        lines += ["", "Deterministic rows (byte-equal or RED): %s" % s.get("deterministic"), "",
                  "| kind | model | prompt | thinking | verdict | apr | references |", "|---|---|---|---|---|---|---|"]
        for d in r["deterministic"]:
            k = d["key"]
            refs = "; ".join("%s: %s" % (e, ("= " if v.get("equal") else "≠ at %s" % v.get("first_difference"))
                                          if v["produced"] else short(v.get("why"), 50))
                             for e, v in d["references"].items())
            lines.append("| %s | %s | %s | %s | **%s** | %s | %s |" % (
                d["kind"], names.get(k["model_sha256"], k["model_sha256"][:12]), k["prompt_id"], k.get("thinking", ""),
                d["verdict"], "produced" if d["apr"]["produced"] else short(d["apr"]["why"], 50), refs))
    if r.get("greedy"):
        lines += ["", "Greedy divergence (REPORTED, not judged): %s" % json.dumps(r["greedy"])[:600]]
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
