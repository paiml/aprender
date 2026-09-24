#!/usr/bin/env python3
"""crux_oracles: the answer oracles for CRUX prompt-set v2 (#3962, quorum revision Q3).

A prompt's `oracle` names HOW its answer is verified, and this module is the only place that decides it.
The judge (crux_inference_judge.py, #3957) and the offline certifier (crux_prompt_certify.py) both
import `evaluate`, so a prompt is certified and gated by the same code.

Four oracle types, each judging ONLY what the model was told to produce:

  answer        the LAST <answer>...</answer> in the reply, normalized, compared for equality.
                No tag is `no_answer_tag`, never a substring fallback: "not Paris" contains
                "Paris", and that is the defect Q3 names.
  state_recall  `answer` applied to the reply to the FINAL user turn (turns[-1]).
  code_tests    the LAST fenced python block, run with the prompt's asserts in a sandbox:
                no network (`unshare -rn`), CPU/memory rlimits, a fresh tmpdir, `python3 -I`.
                No sandbox is `sandbox_unavailable`: an unjudgeable cell is never correct.
  structure     long output judged by invariants: the lines of the <answer> block must equal
                an exact list, so order, count and content are all checked.

A think block is never judged: closed <think>...</think> blocks are stripped here, so an <answer> the
model drafted while reasoning cannot count. An UNCLOSED think block (`<think>` with no `</think>`)
means the budget ran out mid-reasoning; that is `unclosed think (budget exhausted)`, never "empty
answer" (quorum Q5).

API (agreed with the #3957 judge, aprender-6c [8b6b78]):
  evaluate(prompt, text, turns=None) -> {"correct": bool, "extracted": str|None, "why": str|None}
      `text` is the raw engine answer, the final turn's for a multi-turn cell; `turns`, when the engine
      reports them, must number the prompt's user turns. `why` is None exactly when correct.
  extract(prompt, text) -> str|None
      the unit a SAME-REPRESENTATION comparison (apr vs ggml on one GGUF) compares, without judging it.

CLI: crux_oracles.py eval <prompt.json> <reply.json>   prints one JSON verdict line
                                                       (reply.json: the driver's {"text", "turns"?})
     crux_oracles.py lint <prompts.json>                schema errors for a v2 prompt set
     exit 0 correct/valid · 1 wrong/invalid · 2 usage/ENV
"""

from __future__ import annotations

import json
import os
import re
import resource
import shutil
import subprocess
import sys
import tempfile

SCHEMA = "crux-inference-prompts/v2"
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)
ORACLE_TYPES = ("answer", "state_recall", "code_tests", "structure")
NORMALIZERS = ("int", "casefold_strip", "exact")
VERBS = ("run", "chat", "code", "serve run", "serve stream")


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
UNCLOSED = "unclosed think (budget exhausted)"


def clip_head(s, n):
    """`s` cut to its first `n` chars, SAYING how many were cut (#4046: a silent cut reads as the whole text)."""
    return s if len(s) <= n else f"{s[:n]} … and {len(s) - n} more chars"


def clip_tail(s, n):
    """`s` cut to its last `n` chars, SAYING how many were dropped (#4046)."""
    return s if len(s) <= n else f"[{len(s) - n} earlier chars dropped] {s[-n:]}"


def verdict(correct: bool, why, extracted=None) -> dict:
    return {"correct": correct, "extracted": extracted, "why": None if correct else why}


def strip_think(text: str):
    """The text after every closed think block, or None when a think block never closed.

    A `</think>` with no opening tag is a reasoning block whose `<think>` the chat template prefilled
    (it is in the prompt, not the reply): everything up to the LAST `</think>` is reasoning."""
    rest = THINK_RE.sub("", text)
    if re.search(r"<think>", rest, re.I):
        return None
    closes = [m.end() for m in re.finditer(r"</think>", rest, re.I)]
    return rest[closes[-1]:] if closes else rest


def extract_answer(text: str):
    found = ANSWER_RE.findall(text)
    return found[-1] if found else None


def normalize(value: str, how: str):
    if how == "exact":
        return value
    if how == "casefold_strip":
        return " ".join(value.split()).casefold().rstrip(".")
    if how == "int":
        s = value.strip().replace(",", "").rstrip(".")
        return int(s) if re.fullmatch(r"[+-]?\d+", s) else None
    raise ValueError(f"unknown normalize {how!r} (one of {NORMALIZERS})")


def judge_answer(text: str, oracle: dict) -> dict:
    got = extract_answer(text)
    if got is None:
        return verdict(False, "no_answer_tag")
    how = oracle.get("normalize", "casefold_strip")
    have, want = normalize(got, how), normalize(str(oracle["expect"]), how)
    if have is None:
        return verdict(False, "answer_not_" + how, got)
    return verdict(have == want, "match" if have == want else "mismatch", got)


def sandbox_argv():
    """`unshare -rn` gives a network namespace with no interfaces. Probed, never assumed."""
    if not shutil.which("unshare"):
        return None
    probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True, check=False)
    return ["unshare", "-rn"] if probe.returncode == 0 else None


def _limits(timeout_s: int):
    def apply():
        resource.setrlimit(resource.RLIMIT_CPU, (timeout_s, timeout_s))
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 24, 1 << 24))
    return apply


def judge_code(text: str, oracle: dict) -> dict:
    if oracle.get("lang", "python") != "python":
        return verdict(False, "lang_unsupported")
    blocks = FENCE_RE.findall(text)
    if not blocks:
        return verdict(False, "no_code_block")
    code = blocks[-1]
    if not re.search(rf"^\s*def\s+{re.escape(oracle['entry'])}\s*\(", code, re.M):
        return verdict(False, "entry_not_defined", code)
    wrap = sandbox_argv()
    if wrap is None:
        return verdict(False, "sandbox_unavailable", code)
    timeout_s = int(oracle.get("timeout_s", 10))
    with tempfile.TemporaryDirectory(prefix="crux-code-") as d:
        path = os.path.join(d, "cell.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code + "\n\n" + oracle["tests"] + "\nprint('CRUX_TESTS_PASSED')\n")
        try:
            p = subprocess.run(wrap + [sys.executable, "-I", path], cwd=d, capture_output=True, text=True,
                               timeout=timeout_s + 5, preexec_fn=_limits(timeout_s),
                               env={"PATH": "/usr/bin:/bin"}, check=False)
        except subprocess.TimeoutExpired:
            return verdict(False, "tests_timeout", code)
    # The sentinel prints after the last assert; rc 0 alone would accept a reply that calls sys.exit(0).
    passed = p.returncode == 0 and p.stdout.rstrip().endswith("CRUX_TESTS_PASSED")
    tail = (p.stderr.strip().splitlines() or [f"rc={p.returncode}"])[-1]
    return verdict(passed, "tests_passed" if passed else clip_head(f"tests_failed: {tail}", 200), code)


def judge_structure(text: str, oracle: dict) -> dict:
    got = extract_answer(text)
    if got is None:
        return verdict(False, "no_answer_tag")
    lines = [ln.strip() for ln in got.strip().splitlines() if ln.strip()]
    want = [str(x) for x in oracle["lines"]]
    if lines == want:
        return verdict(True, "match", lines)
    first = next((i for i, (a, b) in enumerate(zip(lines, want)) if a != b), min(len(lines), len(want)))
    return verdict(False, f"lines_differ_at_{first}: got {len(lines)} want {len(want)}", lines)


def evaluate(prompt: dict, text, turns=None) -> dict:
    oracle = prompt["oracle"]
    kind = oracle.get("type")
    if turns is not None:
        n_user = sum(1 for m in prompt["messages"] if m["role"] == "user")
        if len(turns) != n_user:
            return verdict(False, f"turns_missing: got {len(turns)} want {n_user}")
        if kind == "state_recall":
            text = turns[-1]
    if not isinstance(text, str):
        return verdict(False, "no_text")
    text = strip_think(text)
    if text is None:
        return verdict(False, UNCLOSED)
    if kind in ("answer", "state_recall"):
        return judge_answer(text, oracle)
    if kind == "code_tests":
        return judge_code(text, oracle)
    if kind == "structure":
        return judge_structure(text, oracle)
    return verdict(False, f"unknown_oracle {kind!r}")


def extract(prompt: dict, text):
    """What a same-representation comparison compares, normalized as the oracle would; None if absent."""
    if not isinstance(text, str):
        return None
    text = strip_think(text)
    if text is None:
        return None
    oracle = prompt["oracle"]
    if oracle.get("type") == "code_tests":
        blocks = FENCE_RE.findall(text)
        return blocks[-1].strip() if blocks else None
    got = extract_answer(text)
    if got is None:
        return None
    if oracle.get("type") == "structure":
        return "\n".join(ln.strip() for ln in got.strip().splitlines() if ln.strip())
    v = normalize(got, oracle.get("normalize", "casefold_strip"))
    return None if v is None else str(v)


def validate_prompt(p: dict) -> list:
    """Schema errors for one v2 prompt entry; [] is valid."""
    errs = []
    pid = p.get("id", "?")
    o = p.get("oracle") or {}
    kind = o.get("type")
    if kind not in ORACLE_TYPES:
        errs.append(f"{pid}: oracle.type must be one of {ORACLE_TYPES}")
    verbs = p.get("verb")
    if not verbs or any(v not in VERBS for v in verbs):
        errs.append(f"{pid}: verb must be a non-empty list drawn from {VERBS}")
    if kind in ("answer", "state_recall"):
        if "expect" not in o:
            errs.append(f"{pid}: answer oracle needs `expect`")
        if o.get("normalize", "casefold_strip") not in NORMALIZERS:
            errs.append(f"{pid}: normalize must be one of {NORMALIZERS}")
    if kind == "code_tests" and not (o.get("entry") and o.get("tests")):
        errs.append(f"{pid}: code_tests needs `entry` and `tests`")
    if kind == "structure" and not o.get("lines"):
        errs.append(f"{pid}: structure needs `lines`")
    asks = " ".join(m.get("content", "") for m in p.get("messages", []))
    if kind in ("answer", "state_recall", "structure") and "<answer>" not in asks:
        errs.append(f"{pid}: the prompt never asks for <answer></answer>, so no reply can satisfy its oracle")
    if kind == "code_tests" and "```" not in asks and "code block" not in asks:
        errs.append(f"{pid}: the prompt never asks for a fenced code block")
    if kind == "state_recall" and sum(m.get("role") == "user" for m in p.get("messages", [])) < 2:
        errs.append(f"{pid}: state_recall needs at least 2 user turns")
    if p.get("control") and not isinstance(p.get("negative"), str):
        errs.append(f"{pid}: a positive control needs `negative`, the planted-wrong reply the judge's negative control injects")
    if isinstance(p.get("negative"), str) and o.get("type") in ORACLE_TYPES and evaluate(p, p["negative"])["correct"]:
        errs.append(f"{pid}: `negative` passes the prompt's own oracle, so it cannot prove the gate goes RED")
    mt = p.get("max_tokens") or {}
    if not (isinstance(mt.get("off"), int) and isinstance(mt.get("on"), int)):
        errs.append(f"{pid}: max_tokens must give both `off` and `on`")
    return errs


def validate_set(doc: dict) -> list:
    """Schema errors for a whole v2 prompt set, including the per-verb positive-control rule."""
    if doc.get("schema") != SCHEMA:
        return [f"schema is {doc.get('schema')!r}, want {SCHEMA!r}"]
    prompts = doc.get("prompts") or []
    if not prompts:
        return ["the prompt set is empty"]
    errs = [e for p in prompts for e in validate_prompt(p)]
    ids = [p.get("id") for p in prompts]
    errs += [f"duplicate id {i!r}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    for verb in VERBS:
        if not any(p.get("control") and verb in (p.get("verb") or []) for p in prompts):
            errs.append(f"no positive control (\"control\": true) serves verb {verb!r}")
    return errs


def main(argv: list) -> int:
    if len(argv) == 3 and argv[0] == "eval":
        with open(argv[1], encoding="utf-8") as f:
            prompt = json.load(f)
        with open(argv[2], encoding="utf-8") as f:
            reply = json.load(f)
        v = evaluate(prompt, reply.get("text"), reply.get("turns"))
        print(json.dumps(v, ensure_ascii=False))
        return 0 if v["correct"] else 1
    if len(argv) == 2 and argv[0] == "lint":
        with open(argv[1], encoding="utf-8") as f:
            errs = validate_set(json.load(f))
        for e in errs:
            print(e)
        return 1 if errs else 0
    print(__doc__.split("CLI:")[1].strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
