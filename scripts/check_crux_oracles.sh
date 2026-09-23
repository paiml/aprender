#!/usr/bin/env bash
# check_crux_oracles.sh: the case table for the CRUX answer oracles
# (scripts/lib/crux_oracles.py, #3962). Hermetic: every reply is a fixture, so it
# needs no model, GPU or engine.
#
# WHY A TABLE. An oracle is only a gate if it has been seen to say WRONG. Every
# must-RED row below is a way a wrong or unjudgeable reply could be scored
# correct: the substring fallback the quorum refuted ("not Paris"), a think
# block that never closed, a reply that exits 0 before the asserts run, code
# that reaches the network, code that never returns, a recall judged on the
# wrong turn, a list in the wrong order. The must-GREEN rows prove the same
# oracle is not simply always wrong.
#
# The certifier rows prove admission needs EVERY leg (ggml@bf16, hf@bf16,
# vllm@bf16, ggml@quant) correct: a missing row, a wrong quant, an unpinned
# source each reject, and `check` refuses a prompt file edited after certifying.
#
# It also lints the committed prompt set (scripts/crux_inference_prompts.v2.json)
# against schema v2, including one positive control per verb.
#
# Exit: 0 every row behaved · 1 a row broke or the prompt set is invalid · 2 ENV.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
PROG=check_crux_oracles
command -v python3 >/dev/null 2>&1 || { printf '%s: ENV - python3 is missing\n' "$PROG" >&2; exit 2; }
ORACLES="$ROOT/scripts/lib/crux_oracles.py"
PROMPTS="$ROOT/scripts/crux_inference_prompts.v2.json"
for f in "$ORACLES" "$PROMPTS"; do
  [ -f "$f" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$f" >&2; exit 2; }
done

python3 - "$ORACLES" <<'PY'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("crux_oracles", sys.argv[1])
o = importlib.util.module_from_spec(spec); spec.loader.exec_module(o)

ASK = "Reply with the final answer inside <answer></answer>."
def ans(expect, norm="casefold_strip"):
    return {"id": "t", "verb": ["run"], "messages": [{"role": "user", "content": "q " + ASK}],
            "max_tokens": {"off": 64, "on": 512}, "oracle": {"type": "answer", "expect": expect, "normalize": norm}}
RECALL = {"id": "r", "verb": ["chat"], "messages": [{"role": "user", "content": "My number is 7. " + ASK},
          {"role": "user", "content": "Add 5 to my number. " + ASK}], "max_tokens": {"off": 64, "on": 512},
          "oracle": {"type": "state_recall", "expect": "12", "normalize": "int"}}
CODE = {"id": "c", "verb": ["code"], "messages": [{"role": "user", "content": "Write add(a, b) in a ```python block."}],
        "max_tokens": {"off": 256, "on": 1024},
        "oracle": {"type": "code_tests", "lang": "python", "entry": "add", "timeout_s": 3,
                   "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0"}}
STRUCT = {"id": "s", "verb": ["run"], "messages": [{"role": "user", "content": "first 3 primes " + ASK}],
          "max_tokens": {"off": 64, "on": 512}, "oracle": {"type": "structure", "lines": [2, 3, 5]}}
def py(body): return "Here:\n```python\n" + body + "\n```\n"

# (name, prompt, reply, want_correct, want_why_prefix); why is None exactly when correct
ROWS = [
  ("GREEN answer tagged",                 ans("Paris"), {"text": "<answer>Paris</answer>"}, True, None),
  ("GREEN answer casefold + trailing dot", ans("Paris"), {"text": "It is <answer> paris. </answer>"}, True, None),
  ("GREEN int with thousands separator",  ans("1000", "int"), {"text": "<answer>1,000</answer>"}, True, None),
  ("GREEN the LAST tag wins",             ans("12", "int"), {"text": "<answer>4</answer> then x3: <answer>12</answer>"}, True, None),
  ("RED no tag: 'not Paris' is not Paris (Q3)", ans("Paris"), {"text": "The capital is not Paris."}, False, "no_answer_tag"),
  ("RED untagged right answer is still unjudged", ans("4", "int"), {"text": "4"}, False, "no_answer_tag"),
  ("RED negation inside the tag",         ans("Paris"), {"text": "<answer>not Paris</answer>"}, False, "mismatch"),
  ("RED an earlier right tag, a later wrong one", ans("12", "int"), {"text": "<answer>12</answer> no wait <answer>13</answer>"}, False, "mismatch"),
  ("RED think block never closed (Q5)",   ans("4", "int"), {"text": "<think>let me see 2+2 is <answer>4</answer>"}, False, "unclosed think"),
  ("GREEN a closed think, then the tag",   ans("4", "int"), {"text": "<think>2+2=4</think><answer>4</answer>"}, True, None),
  ("RED the only tag was drafted inside think", ans("4", "int"), {"text": "<think>maybe <answer>4</answer></think>I am not sure."}, False, "no_answer_tag"),
  ("RED a right draft in think, a wrong final", ans("4", "int"), {"text": "<think><answer>4</answer></think><answer>5</answer>"}, False, "mismatch"),
  ("GREEN a prefilled <think>: reasoning ends at </think>", ans("4", "int"), {"text": "2+2 <answer>5</answer>?\n</think>\n<answer>4</answer>"}, True, None),
  ("RED a prefilled <think>: only a drafted tag", ans("4", "int"), {"text": "2+2 is <answer>4</answer> I think\n</think>\nIt is four."}, False, "no_answer_tag"),
  ("RED int oracle given prose",          ans("4", "int"), {"text": "<answer>four</answer>"}, False, "answer_not_int"),
  ("RED no text at all",                  ans("4", "int"), {"text": None}, False, "no_text"),
  ("GREEN recall on the final turn",      RECALL, {"text": "<answer>12</answer>", "turns": ["<answer>7</answer>", "<answer>12</answer>"]}, True, None),
  ("RED recall right in text, wrong in turns[-1]", RECALL, {"text": "<answer>12</answer>", "turns": ["<answer>12</answer>", "<answer>5</answer>"]}, False, "mismatch"),
  ("RED recall with a turn missing",      RECALL, {"text": "<answer>12</answer>", "turns": ["<answer>12</answer>"]}, False, "turns_missing"),
  ("GREEN code passes its tests",         CODE, {"text": py("def add(a, b):\n    return a + b")}, True, None),
  ("RED code fails an assert",            CODE, {"text": py("def add(a, b):\n    return a - b")}, False, "tests_failed"),
  ("RED code exits 0 before the asserts", CODE, {"text": py("import sys\ndef add(a, b):\n    return 0\nsys.exit(0)")}, False, "tests_failed"),
  ("RED code that reaches the network",   CODE, {"text": py("import socket\nsocket.create_connection(('1.1.1.1', 53), timeout=2)\ndef add(a, b):\n    return a + b")}, False, "tests_failed"),
  ("RED code that never returns",         CODE, {"text": py("def add(a, b):\n    while True:\n        pass")}, False, "tests_"),
  ("RED code under the wrong name",       CODE, {"text": py("def plus(a, b):\n    return a + b")}, False, "entry_not_defined"),
  ("RED prose, no code block",            CODE, {"text": "def add(a, b): return a + b"}, False, "no_code_block"),
  ("GREEN structure exact lines",         STRUCT, {"text": "<answer>\n2\n3\n5\n</answer>"}, True, None),
  ("RED structure out of order",          STRUCT, {"text": "<answer>\n2\n5\n3\n</answer>"}, False, "lines_differ_at_1"),
  ("RED structure one line short",        STRUCT, {"text": "<answer>\n2\n3\n</answer>"}, False, "lines_differ_at_2"),
]

fail = 0
for name, prompt, reply, want, why in ROWS:
    v = o.evaluate(prompt, reply.get("text"), reply.get("turns"))
    good = v["correct"] is want and (v["why"] is None if want else (v["why"] or "").startswith(why))
    print(f"  {'ok   ' if good else 'BROKE'} {name}  ->  {v['correct']} {v['why']}")
    fail += not good

# extract(): the unit a same-representation comparison compares (apr vs ggml on one GGUF).
EXTRACT = [
  ("extract normalizes an int answer",       ans("1000", "int"), "<answer> 1,000 </answer>", "1000"),
  ("extract skips an answer drafted in think", ans("4", "int"), "<think><answer>5</answer></think><answer>4</answer>", "4"),
  ("extract is None when the think never closed", ans("4", "int"), "<think><answer>4</answer>", None),
  ("extract of a code cell is the last block", CODE, py("def add(a, b):\n    return a + b"), "def add(a, b):\n    return a + b"),
  ("extract is None with no tag",           ans("4", "int"), "4", None),
]
for name, prompt, text, want in EXTRACT:
    got = o.extract(prompt, text)
    good = got == want
    print(f"  {'ok   ' if good else 'BROKE'} {name}  ->  {got!r}")
    fail += not good

# Schema rows: a set the lint must REFUSE.
base = {"schema": o.SCHEMA, "prompts": [dict(ans("4", "int"), id="c1", verb=list(o.VERBS), control=True, negative="<answer>5</answer>")]}
LINT = [
  ("GREEN minimal set with one control for every verb", base, True),
  ("RED no control serves `code`", {"schema": o.SCHEMA, "prompts": [dict(base["prompts"][0], verb=["run", "chat", "serve run", "serve stream"])]}, False),
  ("RED a control with no `negative`", {"schema": o.SCHEMA, "prompts": [{k: v for k, v in base["prompts"][0].items() if k != "negative"}]}, False),
  ("RED a `negative` its own oracle accepts", {"schema": o.SCHEMA, "prompts": [dict(base["prompts"][0], negative="<answer>4</answer>")]}, False),
  ("RED prompt never asks for <answer>", {"schema": o.SCHEMA, "prompts": [dict(base["prompts"][0], messages=[{"role": "user", "content": "What is 2+2?"}])]}, False),
  ("RED v1 schema", dict(base, schema="crux-inference-prompts/v1"), False),
  ("RED max_tokens without the thinking-ON budget", {"schema": o.SCHEMA, "prompts": [dict(base["prompts"][0], max_tokens={"off": 64})]}, False),
  ("RED duplicate ids", {"schema": o.SCHEMA, "prompts": base["prompts"] * 2}, False),
]
for name, doc, want in LINT:
    errs = o.validate_set(doc)
    good = (not errs) is want
    print(f"  {'ok   ' if good else 'BROKE'} lint: {name}  ->  {errs[:1] or 'valid'}")
    fail += not good

# The certifier (crux_prompt_certify.py): admission needs EVERY leg correct; missing is never agreement.
import os, subprocess, tempfile
CERT = os.path.join(os.path.dirname(sys.argv[1]), "crux_prompt_certify.py")
SRC = {"repo": "Q/M", "revision": "r1"}
MODEL = {"model": "M", "source": SRC, "bf16_gguf": "b" * 64, "quants": {"Q4": "q" * 64}, "thinking": ["off"]}
PSET = {"schema": o.SCHEMA, "prompts": [
    dict(base["prompts"][0], id="ctl"),
    dict(ans("391", "int"), id="arith", verb=["run"])]}
LEGROW = {"ggml@bf16": ("llamafile", "b" * 64, None), "hf@bf16": ("hf", "b" * 64, SRC),
          "vllm@bf16": ("vllm", "b" * 64, SRC), "ggml@quant": ("llamafile", "q" * 64, None)}
RIGHT = {"ctl": "<answer>4</answer>", "arith": "<answer>391</answer>"}

def certify(d, drop=(), wrong=(), source=SRC):
    rows = []
    for pid, text in RIGHT.items():
        for leg, (eng, sha, src) in LEGROW.items():
            if (pid, leg) in drop:
                continue
            out = os.path.join(d, f"{pid}-{leg}.json")
            json.dump({"text": "<answer>0</answer>" if (pid, leg) in wrong else text}, open(out, "w"))
            open(out + ".err", "w").close()
            rows.append({"kind": "gen", "engine": eng, "model_sha256": sha, "host": "h", "verb": "run",
                         "thinking": "off", "backend": "gpu", "prompt_id": pid, "rc": 0, "stdout": out,
                         "stderr": out + ".err", "refused": None, **({"source": source} if src else {})})
    for f, doc in (("m.jsonl", None), ("p.json", PSET), ("i.json", [MODEL])):
        with open(os.path.join(d, f), "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows) if doc is None else json.dumps(doc))
    r = subprocess.run([sys.executable, CERT, "certify", "--prompts", f"{d}/p.json", "--inventory", f"{d}/i.json",
                        "--apr-commit", "c" * 40, "-o", f"{d}/r.json", f"{d}/m.jsonl"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.load(open(f"{d}/r.json"))

CERTROWS = [
  ("GREEN every leg right: both prompts admitted", {}, ["arith", "ctl"], []),
  ("RED vLLM wrong on one prompt rejects it",      {"wrong": [("arith", "vllm@bf16")]}, ["ctl"], []),
  ("RED a missing ggml@bf16 row is not agreement", {"drop": [("arith", "ggml@bf16")]}, ["ctl"], []),
  ("RED ggml wrong on the QUANT rejects it",       {"wrong": [("arith", "ggml@quant")]}, ["ctl"], []),
  ("RED an unpinned source revision is no hf/vllm row", {"source": {"repo": "Q/M", "revision": "main"}}, [], ["M/Q4"]),
  ("RED an uncertified control marks the model uncontrolled", {"wrong": [("ctl", "hf@bf16")]}, ["arith"], ["M/Q4"]),
]
for name, kw, want_adm, want_unc in CERTROWS:
    with tempfile.TemporaryDirectory() as d:
        r = certify(d, **kw)
    got_adm, got_unc = sorted(r["admitted"]["M/Q4"]), r["uncontrolled"]
    good = got_adm == want_adm and got_unc == want_unc
    print(f"  {'ok   ' if good else 'BROKE'} certify: {name}  ->  admitted {got_adm}, uncontrolled {got_unc}")
    fail += not good

# Thinking ON: hf/vLLM split the think off (split_think); an UNCLOSED one is reasoning + an EMPTY text.
# The receipt must record it as unclosed (the cop's close/loop join), and a tag drafted inside the loop
# must never count.
def think_cert(d, hf_doc):
    m = dict(MODEL, thinking=["on"])
    rows = []
    for leg, (eng, sha, src) in LEGROW.items():
        out = os.path.join(d, f"{leg}.json")
        json.dump(hf_doc if eng in ("hf", "vllm") else {"text": "<think>ok</think><answer>4</answer>"}, open(out, "w"))
        open(out + ".err", "w").close()
        rows.append({"kind": "gen", "engine": eng, "model_sha256": sha, "host": "h", "verb": "chat", "thinking": "on",
                     "backend": "gpu", "prompt_id": "ctl", "rc": 0, "stdout": out, "stderr": out + ".err",
                     "refused": None, **({"source": SRC} if src else {})})
    for f, doc in (("m.jsonl", None), ("p.json", {"schema": o.SCHEMA, "prompts": [PSET["prompts"][0]]}), ("i.json", [m])):
        with open(os.path.join(d, f), "w") as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in rows) if doc is None else json.dumps(doc))
    subprocess.run([sys.executable, CERT, "certify", "--prompts", f"{d}/p.json", "--inventory", f"{d}/i.json",
                    "--apr-commit", "c" * 40, "-o", f"{d}/r.json", f"{d}/m.jsonl"], capture_output=True, check=True)
    r = json.load(open(f"{d}/r.json"))
    return r["admitted"]["M/Q4"], r["think_closure"]["M/Q4|ctl"]["hf@bf16:hf"]

OPENS = {"reported": {"prompt_opens_think": True}}
THINKROWS = [
  ("RED hf looped (opener prefilled, no </think>) is recorded unclosed",
   dict(OPENS, text="", reasoning="x", raw_text="2+2 <answer>4</answer> wait <answer>4</answer>"), [], "unclosed"),
  ("GREEN hf closed its think and answered",
   dict(OPENS, text="<answer>4</answer>", reasoning="2+2", raw_text="2+2\n</think>\n<answer>4</answer>"), ["ctl"], "closed"),
  ("RED a pre-#3990 ON row (no prompt_opens_think) is refused, even when its text looks right (#3990)",
   {"text": "2+2 <answer>4</answer> wait <answer>4</answer>"}, [], "none"),
]
for name, doc, want_adm, want_think in THINKROWS:
    with tempfile.TemporaryDirectory() as d:
        adm, think = think_cert(d, doc)
    good = adm == want_adm and think == want_think
    print(f"  {'ok   ' if good else 'BROKE'} certify: {name}  ->  admitted {adm}, think {think}")
    fail += not good

with tempfile.TemporaryDirectory() as d:
    certify(d)
    def chk():
        return subprocess.run([sys.executable, CERT, "check", "--prompts", f"{d}/p.json", "--receipt", f"{d}/r.json"],
                              capture_output=True, text=True).returncode
    rc_same = chk()
    with open(f"{d}/p.json", "a") as fh:
        fh.write(" ")
    rc_edited = chk()
good = rc_same == 0 and rc_edited == 1
print(f"  {'ok   ' if good else 'BROKE'} check: certified bytes pass (rc {rc_same}); one added byte is refused (rc {rc_edited})")
fail += not good
sys.exit(1 if fail else 0)
PY
table=$?

# Drift against golden_output.rs (#3962 done-when 5: extended, not bypassed): every golden question opens
# an `answer` prompt whose expect is one of its patterns, or is excluded by name with a reason.
GOLDEN="$ROOT/crates/apr-cli/src/commands/golden_output.rs"
[ -f "$GOLDEN" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$GOLDEN" >&2; exit 2; }
DRIFT_OUT=$(python3 - "$GOLDEN" "$PROMPTS" <<'DRIFT'
import json, re, sys
src = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"fn golden_questions\(\)[^{]*\{(.*?)\n\}", src, re.S)
body = re.sub(r"//[^\n]*", "", m.group(1)) if m else ""
cases = [(q, re.findall(r'"((?:[^"\\]|\\.)*)"', pats))
         for q, pats in re.findall(r'\(\s*"((?:[^"\\]|\\.)*)"\s*,\s*vec!\[(.*?)\]', body, re.S)]
doc = json.load(open(sys.argv[2], encoding="utf-8"))
if not cases:
    print("golden_questions() yielded 0 cases - a parse that finds nothing is a refusal, never agreement"); sys.exit(1)
excluded = doc.get("golden_excluded") or {}
bad = []
for q, pats in cases:
    hit = [p["id"] for p in doc["prompts"] if p["oracle"].get("type") == "answer" and len(p["messages"]) == 1
           and p["messages"][0]["content"].startswith(q) and str(p["oracle"].get("expect")) in pats]
    if not hit and not (excluded.get(q) or "").strip():
        bad.append(f"golden question {q!r} (patterns {pats}) has no answer prompt and no golden_excluded reason")
for q in excluded:
    if q not in [c[0] for c in cases]:
        bad.append(f"golden_excluded names {q!r}, which golden_questions() no longer has")
print("\n".join(bad) if bad else f"{len(cases)} golden questions accounted for")
sys.exit(1 if bad else 0)
DRIFT
)
drift=$?
if [ "$drift" -eq 0 ]; then printf '  ok    drift vs golden_output.rs: %s\n' "$DRIFT_OUT"
else printf '  BROKE drift vs golden_output.rs:\n'; printf '%s\n' "$DRIFT_OUT" | sed 's/^/          /'; fi

LINT_OUT=$(python3 "$ORACLES" lint "$PROMPTS" 2>&1)
lint=$?
if [ "$lint" -eq 0 ]; then
  printf '  ok    %s is a valid v2 prompt set\n' "${PROMPTS#"$ROOT"/}"
else
  printf '  BROKE %s:\n' "${PROMPTS#"$ROOT"/}"; printf '%s\n' "$LINT_OUT" | sed 's/^/          /'
fi

if [ "$table" -eq 0 ] && [ "$lint" -eq 0 ] && [ "$drift" -eq 0 ]; then
  printf '%s: PASS\n' "$PROG"; exit 0
fi
printf '%s: FAIL (table rc=%s, prompt-set lint rc=%s, golden drift rc=%s)\n' "$PROG" "$table" "$lint" "$drift"; exit 1
