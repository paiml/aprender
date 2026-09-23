#!/usr/bin/env python3
"""crux_inference_judge.py: the judge of the CRUX inference dogfood (#3739).

scripts/crux_inference_dogfood.sh DRIVES the engines and writes one manifest
line per invocation, pointing at that invocation's raw stdout/stderr. This file
READS those files and nothing else: it starts no engine and opens no socket.
The split is deliberate. The judge is the part the case table
(scripts/check_crux_inference_judge.sh) can drive with fixtures, with no model,
GPU or comparator, in milliseconds.

THE RULE (#3957 F6, the quorum-revised CRUX oracle). A cell is keyed by (model sha256, host,
verb, thinking, rung, prompt) and is GREEN or RED -- there is no third state (operator 2026-09-23,
"no defer"). It is GREEN only when ALL of these hold, and RED naming every one that does not:

  1. apr answered, and its answer is CORRECT under the prompt's oracle (scripts/lib/crux_oracles.py:
     the constrained <answer>X</answer>, think blocks stripped, an unclosed think block RED). A v1
     `expect_any` prompt is a substring oracle and is RED ("not Paris" contains "Paris").
  2. SAME-REPRESENTATION: the engines that read the IDENTICAL weights -- the ggml family
     (llama.cpp, ollama, llamafile: ONE vote, they share llama.cpp's code) for a GGUF, hf/vLLM for
     SafeTensors -- answered, agree with each other, and agree with apr on the EXTRACTED answer.
     A .apr has no such engine; it is proven only through its chain (check_model_ladder F8).
  3. GROUND TRUTH: a bf16 control (hf or vLLM) answered, and every one that did is correct. The
     control proves the prompt is answerable; it never votes against a quantized answer.

"apr did not answer" includes a non-zero exit, a backend fallback, a refusal, unparseable
output, a DEGENERATE completion (one character >= 90%, or a multi-character token loop,
#3971) and a MISSING row. ALL_WRONG is recorded as a flag on a RED cell, never its own verdict.

CONTROLS. Every (model, host, verb, thinking) must have a POSITIVE control cell (`"control": true`);
one missing declines the run (exit 2). Per verb, the judge plants a wrong apr answer (the prompt's
`negative`) into a GREEN control cell and must see RED; a lane that cannot see a wrong answer
declines the run. A v2 prompt set is used only with its certification receipt (#3962 J2,
`--certification`); an uncertified set declines. Exit 2 is never green (check_model_ladder
refuses a DECLINED receipt).

PERFORMANCE IS TRANSCRIBED, NEVER COMPUTED. Token counts and rates are copied
from what each engine prints about itself, labelled with the engine that
reported them, and never ratioed, averaged or judged (the withdrawn-headline
rule, docs/BEATS.md). This file derives no rate from a clock, and no field it
writes is an input to the verdict.

Usage:
  crux_inference_judge.py collect --manifest M --prompts P --meta META \\
      --out-json R.json --out-md R.md
Exit: 0 every cell GREEN, every lane controlled both ways, the prompt set certified;
1 any RED; 2 decline.
"""

import argparse
import ast
import datetime
import json
import math
import os
import re
import struct
import sys

COMPARATORS = ("llama.cpp", "ollama", "hf", "llamafile", "vllm")
# Plugin engines write row contract v1 through their own driver (scripts/crux_engine_<e>.*).
PLUGIN_ENGINES = ("hf", "llamafile", "vllm")
ENGINES = ("apr",) + COMPARATORS

#: Why an engine produced nothing (#3832). These are NOT interchangeable and the
#: receipt must not collapse them:
#:
#:   not_on_PATH              a working install the harness could not NAME.
#:   binary_not_found_at_path the install is genuinely absent.
#:
#: Measured 2026-09-22: `ssh lambda-labs llama-cli --version` says
#: `command not found` while `~/.local/bin/llama-cli --version` prints the pinned
#: `0.4.1-dev (build 10987, commit d1d3c3396)`. lambda's non-interactive PATH has
#: no `~/.local/bin`; gx10's does. CRUX runs cross-host over ssh, so the harness
#: would report an ABSENCE about a healthy comparator — and under the quorum floor
#: that silently drops the cell to one engine while the receipt blames the
#: comparator. One of these is a comparator gap; the other is a HARNESS DEFECT,
#: and only the first is a fact about the fleet.
NOT_RAN_REASONS = (
    "not_installed",
    "binary_not_found_at_path",
    "not_on_PATH",
    "model_not_pulled",
    "refused",
    "crashed",
    "timed_out",
    "not_requested",
    "unclassified",
)

#: Text an engine's `why` may carry, mapped to the enum above. Ordered: the first
#: match wins, and the PATH forms are tested before the generic "not found" ones,
#: because `command not found` is a PATH answer wearing an absence's words.
#: EVERY NEEDLE IS LOWERCASE because the haystack is lowercased before matching.
#: Two of them shipped with "PATH" capitalised and could never fire; the case
#: table below caught it on its first run, which is the whole reason a guard
#: ships one (CLAUDE.md: re-run the table, do not re-read the pattern).
_NOT_RAN_PATTERNS = (
    ("not requested", "not_requested"),
    ("command not found", "not_on_PATH"),
    ("not found on path", "not_on_PATH"),
    ("not on path", "not_on_PATH"),
    ("no such file or directory", "binary_not_found_at_path"),
    ("does not exist", "binary_not_found_at_path"),
    ("not installed", "not_installed"),
    ("model not pulled", "model_not_pulled"),
    ("no such model", "model_not_pulled"),
    ("refused", "refused"),
    ("timed out", "timed_out"),
    ("timeout", "timed_out"),
    ("killed", "crashed"),
    ("crashed", "crashed"),
    ("exited", "crashed"),
)


def classify_not_ran(why):
    """Map an engine's free-text `why` onto NOT_RAN_REASONS.

    Unrecognised text is `unclassified`, never a guess: a wrong reason is worse
    than an unknown one, because it reads as a fact someone measured.
    """
    if not why:
        return "unclassified"
    low = str(why).lower()
    for needle, reason in _NOT_RAN_PATTERNS:
        if needle in low:
            return reason
    return "unclassified"


def engine_versions(meta):
    """Per-engine version strings, pulled from the producer's meta block.

    The receipt carried these ONLY at the top level, so a cell stated which
    engines ran but never at which versions — and a comparator version is not
    stable across a receipt's lifetime: ollama 0.34.1 changed GGUF creation from
    safetensors, which is exactly the `Modelfile FROM <path>` mechanism CRUX
    depends on. A cell that names a verdict without naming what produced it makes
    the same class of claim as a cell judged on one engine.
    """
    ol = meta.get("ollama") or {}
    lc = meta.get("llama_cpp") or {}
    out = {
        "apr": (meta.get("apr") or {}).get("version_line"),
        # The SERVER's version, not the client's: `ollama --version` reports the
        # daemon and names the client only in a mismatch warning (paiml/infra#911).
        "ollama": ol.get("server_version"),
        "llama.cpp": lc.get("build"),
    }
    # A plugin engine's version IS its probe line (the producer writes `probe`, never `version`): this read
    # `version` alone, so every plugin cell carried version None and no receipt could say which hf, llamafile
    # or vllm had vouched (#3952).
    for eng in PLUGIN_ENGINES:
        m = meta.get(eng) or {}
        out[eng] = m.get("version") or m.get("probe")
    return {k: v for k, v in out.items()}


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
    """Plugin-engine rows — hf, llamafile, vllm (row contract v1, #3739 issuecomment-5765991210):
    the engine driver writes `{"text": <the answer only>, "reported": {...}}`."""
    out = {"answer": None, "why": None, "reported": {}}
    try:
        doc = json.loads(stdout)
    except ValueError as exc:
        out["why"] = "stdout is not the contract's JSON: %s" % exc
        return out
    if isinstance(doc, dict) and doc.get("protocol_fault"):
        # #3962 R2 (aprender-19): the wire broke. Named, never read as a missing text field.
        out["why"] = "protocol fault: %s" % doc["protocol_fault"]
        return out
    if isinstance(doc, dict) and isinstance(doc.get("reasoning"), str) and doc.get("reasoning"):
        # #3962 (aprender-dd): hf/vLLM split the think block off before writing `text`; an UNCLOSED
        # block arrives as `reasoning` + "". Rebuild the raw reply (crux_prompt_certify.driver_raw's
        # shape) so the oracle sees an unclosed think as unclosed, not as a missing <answer> tag.
        t = doc.get("text") or ""
        doc = dict(doc, text="<think>" + doc["reasoning"] + ("</think>" + t if t else ""))
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
    return top >= 0.9 * len(chars) or token_loop(text) is not None


def token_loop(text):
    """#3957 F6 / #3971: a MULTI-character loop is no answer either. Measured live: vLLM on a
    poisoned HF cache answered "NavControllerNavController..." (64 tokens) in every cell, and the
    one-character rule above scored it ANSWERED, so it corroborated apr 4/4. The signal is
    output_verification.rs's repeated-fragment rule, which scripts/model_ladder.sh mirrors: a
    fragment of 4..16 bytes repeated three times back to back. A fragment that is all whitespace
    or one repeated character is left to the rule above -- indentation is not a loop.
    -> the fragment, or None."""
    b = (text or "").encode("utf-8", "replace")
    if len(b) < 12:
        return None
    for fl in range(4, min(16, len(b) // 3) + 1):
        for i in range(0, len(b) - fl * 3 + 1):
            f = b[i:i + fl]
            if len(set(f)) < 2 or not f.strip():
                continue
            if b[i + fl:i + 2 * fl] == f and b[i + 2 * fl:i + 3 * fl] == f:
                return f.decode("utf-8", "replace")
    return None


# ------------------------------------------------------------ the oracle (#3957 F6) --
# Operator 2026-09-23: "CRUX is critical here, as we need quorum our style on them, as we could
# test 'the happy path'". Quorum review of that design: do-not-implement-as-written, 2/2; the
# revision (#3957 comment 5790953724) and the cop's corrections are what this implements:
#   Q1  no UNJUDGED -- a cell is GREEN or RED. A split, a missing oracle, a failed control: RED.
#   Q2  two oracles. SAME-REPRESENTATION: apr against the engines that read the IDENTICAL weights
#       (GGUF: the ggml family; SafeTensors: hf/vLLM), compared on the EXTRACTED answer.
#       GROUND TRUTH: the prompt's verifiable answer; hf/vLLM at bf16 are the control that the
#       prompt is answerable, never a vote against a quantized token stream.
#   Q3  the answer is the constrained <answer>X</answer> (scripts/lib/crux_oracles.py, #3962),
#       never a substring: "not Paris" contains "Paris". A v1 `expect_any` prompt is RED.
#   Q5  a think block is stripped before judging; an UNCLOSED one is RED (budget exhausted).
# ggml family = llama.cpp + ollama + llamafile: they share llama.cpp's code, so they are ONE vote.
import crux_oracles  # noqa: E402  (scripts/lib is this file's own directory)

FAMILY = {"llama.cpp": "ggml", "ollama": "ggml", "llamafile": "ggml", "hf": "bf16", "vllm": "bf16"}
SAME_REP = {"gguf": "ggml", "safetensors": "bf16"}
V1_WHY = ("v1 prompt: `expect_any` is a substring oracle, not a constrained <answer> -- \"not Paris\" "
          "contains \"Paris\" (#3957 Q3); certify a v2 prompt (#3962)")


def model_format(name):
    f = (name or "").lower()
    if f.endswith(".gguf"):
        return "gguf"
    if f.endswith(".apr"):
        return "apr"
    if f.endswith(".safetensors") or "safetensors" in f:
        return "safetensors"
    return None


def spoke(entry):
    """Answered -- or produced a DEGENERATE completion, which is garbage said, not silence."""
    return bool(entry.get("answered")) or str(entry.get("why") or "").startswith("degenerate")


def oracle_eval(prompt, entry):
    """-> {"correct", "extracted", "why"} for one engine's entry, through the one oracle."""
    if not entry.get("answered"):
        return {"correct": False, "extracted": None, "why": entry.get("why") or "did not answer"}
    if "oracle" not in prompt:
        return {"correct": False, "extracted": None, "why": V1_WHY}
    turns = entry.get("turns") or None
    text = entry.get("answer")
    v = crux_oracles.evaluate(prompt, text, turns)
    judged = turns[-1] if (turns and prompt["oracle"].get("type") == "state_recall") else text
    return {"correct": bool(v.get("correct")), "why": v.get("why"),
            "extracted": crux_oracles.extract(prompt, judged)}


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
    elif engine in PLUGIN_ENGINES:
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
    if engine in PLUGIN_ENGINES:
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
        # THE FLOOR HERE IS KEYED ON THE FIELD, NOT ON THE ENGINE (#3832).
        #
        # A parity cell compares SEQUENCES, so "did this engine answer?" is the
        # wrong question: slice 1's defect was llama.cpp answering in TEXT while
        # producing no `prompt_ids` at all. An engine-keyed floor counts that cell
        # as two-engine and then compares one side. `produced` therefore counts
        # engines that produced THIS FIELD, which is what the comparison needs.
        #
        # The asymmetry from judge_cell holds for the same reason it holds there:
        # a DIVERGENCE against a single reference is a complete finding — the
        # reference produced ids, apr differs, and the difference is a fact about
        # apr.
        #
        # A two-reference floor for GREEN was proposed and TESTED here, on the
        # theory that apr and one tokenizer reading the same GGUF vocab might
        # agree structurally rather than evidentially. The fixtures killed it:
        # `tok equal` and `tmpl equal` both went UNJUDGED, because CRUX's parity
        # design IS apr against llama.cpp's tokenizer — one reference by
        # construction (#3739 done_when 2, "token-id parity (apr vs
        # llama-tokenize)"). Requiring a second would make every parity cell
        # UNJUDGED and decline every run. The theory was reasonable and wrong,
        # and it is recorded here so nobody re-derives it.
        #
        # What DID need fixing is the coverage record below: the cell reported a
        # verdict without reporting how many engines produced the field it
        # compared.
        produced = [e for e, v in refs.items() if v["produced"]]
        coverage = {"field": "ids" if kind == "tok" else "rendered",
                    "produced_by": sorted(produced),
                    "references_producing": len(produced),
                    "apr_produced": apr_val is not None}
        if not produced:
            # #3957 Q1: no UNJUDGED. A parity cell no reference produced is non-corroboration: RED.
            verdict = "RED"
        elif apr_val is None or any(not refs[e]["equal"] for e in produced):
            verdict = "RED"
        else:
            verdict = "GREEN"
        out.append({"kind": kind, "key": dict(zip(names, key)), "verdict": verdict,
                    "coverage": coverage,
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


def _greedy_raw(r):
    """#3957 F9: an engine's RAW greedy record, verbatim, for the ladder judge to compare itself.
    -> {"raw": {...}} | {"refused": why} | {"why": unreadable}."""
    if r.get("refused"):
        return {"refused": r["refused"]}
    try:
        with open(r["tokens"], encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"why": "raw greedy record unreadable: %s" % exc}
    return {"raw": raw} if isinstance(raw, dict) else {"why": "raw greedy record is not an object"}


def report_greedy(rows):
    rows = [r for r in rows if r.get("kind") == "greedy"]
    groups = {}
    for r in rows:
        # #3957 F9: thinking is part of the key -- an ON and an OFF greedy row are different cells.
        # #3990: llama.cpp run on the model's OWN template is a second row for the same engine, kept apart
        # from the parity row (apr's ids) under "llama.cpp@official".
        eng = r["engine"] + ("@official" if r.get("prompt_source") == "official" else "")
        groups.setdefault((r["model_sha256"], r["host"], r["prompt_id"], r.get("thinking", "unset")), {})[eng] = r
    out = []
    for key in sorted(groups):
        by = groups[key]
        rep = {"key": dict(zip(("model_sha256", "host", "prompt_id", "thinking"), key)), "engines": sorted(by)}
        for eng, r in sorted(by.items()):
            rep.setdefault(eng, {}).update(_greedy_raw(r))
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
            rep[eng].update(item)
        out.append(rep)
    return out


#: The corroboration floor (#3832). CRUX exists to be an EXTERNAL oracle: apr
#: judged against other engines on the same GGUF. This is the number of engines a
#: cell needs before its verdict is CORROBORATED — recorded on every cell as
#: `quorum.met` so a reader can see the coverage behind a verdict instead of
#: inferring it. It is NOT applied to RED: see judge_cell for why that would
#: suppress the finding CRUX exists for.
CELL_QUORUM_FLOOR = 2


def cell_quorum(entries):
    """Who actually answered this cell, and did that clear the floor?"""
    names = sorted(e for e in ENGINES if entries.get(e, {}).get("answered"))
    return {
        "floor": CELL_QUORUM_FLOOR,
        "engines_answered": len(names),
        "answered": names,
        "comparators_answered": sorted(e for e in names if e in COMPARATORS),
        "met": len(names) >= CELL_QUORUM_FLOOR,
    }


def judge_cell(entries, prompt, fmt):
    """-> (verdict GREEN|RED, {engine: correct}, [reasons], {engine: extracted}). GREEN only when no
    reason stands: apr correct, the same-representation family agreeing with apr on the extracted
    answer, and a ground-truth control that answered correctly. Everything else is RED, named."""
    ev = {e: oracle_eval(prompt, v) for e, v in entries.items()}
    ok = {e: ev[e]["correct"] for e in entries}
    ext = {e: ev[e]["extracted"] for e in entries if entries[e].get("answered")}
    why = []
    a = entries["apr"]
    if not a.get("answered"):
        why.append("apr did not answer: %s" % (a.get("why") or "no row"))
    elif not ok["apr"]:
        why.append("apr is wrong: %s" % ev["apr"]["why"])
    fam = SAME_REP.get(fmt)
    if fam is None:
        why.append("no same-representation oracle: no engine but apr reads a %s file -- a .apr is proven only "
                   "through its chain to its source (#3957 F8)" % (fmt or "format-unknown"))
    else:
        same = [e for e in COMPARATORS if FAMILY.get(e) == fam and spoke(entries[e])]
        vals = {e: (ext.get(e) if entries[e].get("answered") else "<degenerate>") for e in same}
        if not same:
            why.append("no same-representation oracle: no %s-family engine answered on the identical weights "
                       "(#3957 Q2)" % fam)
        elif len(set(vals.values())) > 1 or None in vals.values():
            why.append("same-representation SPLIT %s -- a split is RED, never a prompt swap (#3957 Q1)"
                       % json.dumps(vals, ensure_ascii=False))
        elif a.get("answered") and ext.get("apr") != next(iter(vals.values())):
            why.append("apr differs from the %s family on the identical weights: apr %r vs %r (#3957 Q2)"
                       % (fam, ext.get("apr"), next(iter(vals.values()))))
    ctl = [e for e in COMPARATORS if FAMILY.get(e) == "bf16" and spoke(entries[e])]
    if not ctl:
        why.append("no ground-truth control: neither hf nor vllm answered, so nothing shows this prompt is "
                   "answerable (#3957 Q2)")
    else:
        bad = [e for e in ctl if not ok[e]]
        if bad:
            why.append("ground-truth control FAILED (%s) -- a control that cannot answer vouches for nothing; "
                       "the prompt or the control engine is broken (#3957 Q2, #3971)"
                       % "; ".join("%s: %s" % (e, ev[e]["why"]) for e in bad))
    return ("RED" if why else "GREEN"), ok, why, ext


def certification_ok(prompts_path, receipt_path):
    """#3962 J2 through the certifier's own `check` -> True, or the reason it refused."""
    if not receipt_path:
        return "no --certification receipt was given"
    import contextlib
    import io
    import crux_prompt_certify as mod   # through sys.path, as crux_oracles is -- never by this file's location
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = mod.check(argparse.Namespace(prompts=prompts_path, receipt=receipt_path))
    except (OSError, ValueError, KeyError) as exc:
        return "certification unreadable: %s" % exc
    return True if rc == 0 else (buf.getvalue().strip() or "refused (rc %s)" % rc)


def collect(args):
    with open(args.prompts, encoding="utf-8") as fh:
        pdoc = json.load(fh)
    prompts = {p["id"]: p for p in pdoc["prompts"]}
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
    # #3832: resolved once, stamped on every cell.
    versions = engine_versions(meta)
    # model sha256 -> development note, for models the operator has declared are
    # under active development. Keyed by sha so a rename cannot silently drop it.
    subject_dev = {m.get("sha256"): m.get("under_development")
                   for m in (meta.get("models") or []) if m.get("under_development")}
    # #3957 Q2: the FORMAT picks the same-representation oracle. From the row, else the model's name.
    fmt_of_model = {m.get("sha256"): (m.get("format") or model_format(m.get("name")))
                    for m in (meta.get("models") or [])}

    keys = []
    by_key = {}
    for r in gens:
        # serve cells come in two modes (nonstream, stream): an additive key part,
        # absent for run and chat. Plugin serve rows are non-streaming.
        mode = r.get("mode") or ("nonstream" if r["verb"] == "serve run" else "")
        pr = prompts[r["prompt_id"]]
        # #3962 R1 (aprender-19): each serve ROUTE is its own cell, an additive key part like `mode`.
        k = (r["model_sha256"], r["host"], r["verb"], r["thinking"], pr.get("rung") or pr.get("tier") or "v2",
             r["prompt_id"], mode, r.get("route") or "")
        if k not in by_key:
            keys.append(k)
            by_key[k] = {}
        by_key[k][r["engine"]] = r

    # #3962 J2 (per cell): the certification admits prompts PER MODEL (quant sha). A prompt it did not
    # admit for this model is RED on that cell, however right the answer -- it was never shown answerable.
    admitted = None
    if pdoc.get("schema") == "crux-inference-prompts/v2" and getattr(args, "certification", None):
        try:
            with open(args.certification, encoding="utf-8") as fh:
                admitted = json.load(fh).get("admitted_by_sha")
        except (OSError, ValueError):
            admitted = None
        if not isinstance(admitted, dict):
            admitted = {}   # a receipt with no per-model admission admits nothing
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
                # #3952: a comparator the receipt cannot name a version for cannot vouch — for apr or against
                # it. Its answer is kept on the record; it is not an oracle.
                if eng in COMPARATORS and entries[eng].get("answered") and not versions.get(eng):
                    entries[eng]["answered"] = False
                    entries[eng]["why"] = ("unpinned: the run's meta records no version for %s, so a verdict it "
                                           "vouched for could not name what produced it" % eng)
        fmt = next((by_key[k][e].get("format") for e in by_key[k] if by_key[k][e].get("format")), None) \
            or fmt_of_model.get(k[0])
        verdict, ok, reasons, extracted = judge_cell(entries, prompt, fmt)
        if admitted is not None and k[5] not in admitted.get(k[0], ()):
            reasons = reasons + ["prompt %s is not admitted for this model by the certification (admitted_by_sha) -- "
                                 "never shown answerable here (#3962 J2)" % k[5]]
            verdict = "RED"
        for eng in ENGINES:
            entries[eng]["correct"] = ok[eng]
            # #3832: one indivisible record per engine — WHICH engine, at WHICH
            # version, and if it did not run, WHY, in a classified form. Split
            # across fields a reader can see a count without provenance.
            entries[eng]["version"] = versions.get(eng)
            if not entries[eng].get("answered"):
                entries[eng]["not_ran_reason"] = classify_not_ran(entries[eng].get("why"))
        said = {e: norm(v["answer"]) for e, v in entries.items() if v.get("answered")}
        cells.append({
            "key": dict(zip(("model_sha256", "host", "verb", "thinking", "rung", "prompt_id"), k[:6]),
                        **({"mode": k[6]} if k[6] else {}), **({"route": k[7]} if k[7] else {})),
            "verdict": verdict,
            # #3957: why a cell is RED, every reason, and what each engine's answer extracted to.
            "reasons": reasons,
            "format": fmt,
            "extracted": extracted,
            "all_wrong": verdict == "RED" and not any(ok.get(e) for e in ENGINES)
                         and any(v.get("answered") for v in entries.values()),
            # #3832: the cell states its own coverage, so a reader never has to
            # infer how many engines produced the verdict they are reading.
            "quorum": cell_quorum(entries),
            # #3832: the SUBJECT of the comparison, stamped like the comparators.
            # Without it `engines[]` documents the comparators rigorously and
            # leaves what is being compared unqualified — the same asymmetry as a
            # parity column carrying prompt_ids for only one side. `under_development`
            # is not licence to ignore a red; it is what stops a red being read as
            # a REGRESSION when it is development state (operator 2026-09-22:
            # "qwen3.5 on our box is dicey as we are developing and testing").
            "subject": {
                "model_sha256": k[0],
                "apr_version": versions.get("apr"),
                "under_development": bool(subject_dev.get(k[0])),
                "development_note": subject_dev.get(k[0]) or None,
            },
            # additive (aprender-97, #3715): pv reads the control by this flag, never by a prompt name
            "positive_control": bool(prompt.get("control")),
            "oracle": prompt.get("oracle"),
            **({"expect_any": prompt["expect_any"]} if "expect_any" in prompt else {}),
            "engines": entries,
            "agreement": {
                "answered": sorted(said),
                "all_identical": len(said) >= 2 and len(set(said.values())) == 1,
                "apr_matches": sorted(e for e in said if e != "apr" and "apr" in said and said[e] == said["apr"]),
            },
            "token_parity": (token_parity(entries["apr"], toks.get((k[0], k[5]))) if k[2] == "run"
                             else {"measured": False, "why": "not measured for the %s verb" % k[2]}),
        })

    counts = {v: sum(1 for c in cells if c["verdict"] == v) for v in ("RED", "GREEN")}
    counts["ALL_WRONG"] = sum(1 for c in cells if c.get("all_wrong"))   # a SUBSET of RED (#3957 F6), never its own state
    judged = counts["RED"] + counts["GREEN"]
    all_wrong_by_model = {}
    for c in cells:
        if c.get("all_wrong"):
            k = c["key"]["model_sha256"]
            all_wrong_by_model[k] = all_wrong_by_model.get(k, 0) + 1
    controls = [pid for pid, p in prompts.items() if p.get("control")]
    # #3957 F6: ONE positive control per (model, host, verb, thinking) -- a control measured on
    # `run` says nothing about whether the `chat` lane can see a right answer.
    lanes = sorted({(c["key"]["model_sha256"], c["key"]["host"], c["key"]["verb"], c["key"]["thinking"]) for c in cells})
    uncontrolled = ["%s/%s/%s/%s" % (m[:12], h, v, t) for (m, h, v, t) in lanes
                    if not any((c["key"]["model_sha256"], c["key"]["host"], c["key"]["verb"], c["key"]["thinking"])
                               == (m, h, v, t) and c["positive_control"] for c in cells)]
    # #3957 F6 / J3: a NEGATIVE control per verb. The judge plants a wrong apr answer into a GREEN
    # control cell of that verb and must see RED; a lane that cannot see a wrong answer vouches
    # for nothing. The planted text is the prompt's own `negative` (#3962), else a fixed non-answer.
    negative = {}
    for verb in sorted({c["key"]["verb"] for c in cells}):
        ctl = next((c for c in cells if c["key"]["verb"] == verb and c["positive_control"] and c["verdict"] == "GREEN"), None)
        if ctl is None:
            continue   # no GREEN control for this verb: the run is RED or uncontrolled already
        planted = prompts[ctl["key"]["prompt_id"]].get("negative") or "<answer>__crux_negative_control__</answer>"
        ents = {e: dict(v) for e, v in ctl["engines"].items()}
        ents["apr"] = dict(ents["apr"], answered=True, answer=planted, why=None,
                           turns=(ents["apr"].get("turns") or [])[:-1] + [planted] if ents["apr"].get("turns") else None)
        nv = judge_cell(ents, prompts[ctl["key"]["prompt_id"]], ctl["format"])[0]
        negative[verb] = {"prompt_id": ctl["key"]["prompt_id"], "planted": planted, "verdict": nv}
    blind = sorted(v for v, r in negative.items() if r["verdict"] != "RED")
    # #3962 J2: a v2 prompt set is used only under the certification receipt that covers its bytes.
    certified = None
    if pdoc.get("schema") == "crux-inference-prompts/v2":
        certified = certification_ok(args.prompts, getattr(args, "certification", None))
    declined_because = None
    if not controls:
        declined_because = "the prompt set declares no positive control (\"control\": true)"
    elif not cells:
        declined_because = "no cell was measured"
    elif uncontrolled:
        declined_because = "no positive-control cell for (model/host/verb/thinking) " + ", ".join(uncontrolled)
    elif blind:
        declined_because = ("negative control: a planted wrong apr answer was NOT judged RED for verb(s) %s -- "
                            "the lane cannot see a wrong answer" % ", ".join(blind))
    elif certified is not None and certified is not True:
        declined_because = "the v2 prompt set is not certified: %s (#3962 J2)" % certified
    det_counts = {v: sum(1 for d in det if d["verdict"] == v) for v in ("RED", "GREEN")}
    if counts["RED"] or det_counts["RED"]:
        verdict, rc = "RED", 1
    elif declined_because:
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
                        all_wrong_by_model=all_wrong_by_model, controls=controls, negative_controls=negative,
                        certified=certified,
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
        "apr `%s` · llama.cpp `%s` · ollama `%s` · hf `%s` · llamafile `%s` · vllm `%s` · judged %s" % (
            r.get("apr", {}).get("version_line"), r.get("llama_cpp", {}).get("build"),
            r.get("ollama", {}).get("server_version"), r.get("hf", {}).get("probe"),
            r.get("llamafile", {}).get("probe"), r.get("vllm", {}).get("probe"), r["judged_at"]),
        "",
        "**%s**: %d cells, %d RED (%d of them ALL_WRONG), %d GREEN." % (
            s["verdict"], s["cells"], s["RED"], s["ALL_WRONG"], s["GREEN"]),
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
    c.add_argument("--certification", default=None, help="#3962 J2: the prompt-certification receipt for a v2 set")
    args = ap.parse_args(argv)
    try:
        return collect(args)
    except (OSError, ValueError, KeyError) as exc:
        sys.stderr.write("decline: %s: %s\n" % (type(exc).__name__, exc))
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
