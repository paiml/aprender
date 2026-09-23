#!/usr/bin/env bash
# check_crux_inference_judge.sh: the case table for the CRUX inference judge
# (scripts/lib/crux_inference_judge.py, #3739). Hermetic: every engine output
# is a fixture written under mktemp, so it needs no model, GPU or comparator
# and runs bare wherever guard_tree runs it.
#
# WHY A TABLE. The judge's one rule, "a comparator right and apr wrong is RED",
# is only a gate if it has been seen to go RED. Row 2 is the falsifier #3715
# names: llama.cpp correct and apr wrong on one cell turns the run RED. The
# other rows pin each way apr can fail to answer (a wrong answer, a non-zero
# exit, a fallback off the lane's backend, a MISSING row), and each way a run
# can look fine while proving nothing: no comparator answered, the positive
# control came back ALL_WRONG or was never measured, no control was declared.
# ALL_WRONG anywhere else is named and counted per model, never a violation.
#
# It also refuses DRIFT between scripts/crux_inference_prompts.json and the
# golden cases it was derived from (golden_output.rs::golden_test_cases): the
# same user messages and expected patterns, in the same order. Zero cases found
# on either side is a refusal, never agreement.
#
# Exit: 0 every row behaved · 1 a row broke or the prompt set drifted · 2 ENV.
set -uo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd) || exit 2
PROG=check_crux_inference_judge
command -v python3 >/dev/null 2>&1 || { printf '%s: ENV - python3 is missing\n' "$PROG" >&2; exit 2; }
JUDGE="${CRUX_JUDGE_OVERRIDE:-$ROOT/scripts/lib/crux_inference_judge.py}"
# #3957 F6: the verdict cases run on a v2 FIXTURE prompt set (constrained <answer> oracles, #3962)
# written below, certified by a fixture receipt. The REAL v1 set is still what the golden_output.rs
# drift check (row 12) reads.
PROMPTS_V1="$ROOT/scripts/crux_inference_prompts.json"
PROMPTS="$PROMPTS_V1"
GOLDEN="$ROOT/crates/apr-cli/src/commands/golden_output.rs"
for f in "$JUDGE" "$PROMPTS_V1" "$GOLDEN"; do
  [ -f "$f" ] || { printf '%s: ENV - %s not found\n' "$PROG" "$f" >&2; exit 2; }
done

TMP=$(mktemp -d) || exit 2
_rm_tmp() {
  case "${TMP:-}" in
    /tmp/?*|/var/folders/?*) rm -rf -- "$TMP" || : ;;
    *) : ;;
  esac
}
trap _rm_tmp EXIT

PASS=0
FAIL=0
# The fixture v2 prompt set: the old ids and message texts (the llama.cpp parser finds the echoed
# prompt), with verifiable oracles. golden-2plus2 is the control for run/serve, chat-arith-2turn for chat.
PROMPTS="$TMP/prompts.v2.json"
python3 - "$PROMPTS" <<'PY'
import json, sys
def p(pid, rung, content, oracle, verb, control=False, negative=None, messages=None):
    d = {"id": pid, "rung": rung, "verb": verb, "tier": "easy", "max_tokens": {"off": 128, "on": 2048},
         "messages": messages or [{"role": "user", "content": content}], "oracle": oracle}
    if control:
        d.update(control=True, negative=negative)
    return d
json.dump({"schema": "crux-inference-prompts/v2", "prompts": [
    p("golden-2plus2", "golden", "What is 2+2?", {"type": "answer", "expect": "4", "normalize": "int"},
      ["run", "chat", "serve run"], True, "<answer>5</answer>"),
    p("golden-greeting", "golden", "Hello there, how are you doing today my friend?",
      {"type": "answer", "expect": "hello", "normalize": "casefold_strip"}, ["run"]),
    p("golden-paris", "golden", "What is the capital of France?",
      {"type": "answer", "expect": "Paris", "normalize": "casefold_strip"}, ["run", "chat"]),
    p("chat-arith-2turn", "chat", None, {"type": "state_recall", "expect": "12", "normalize": "int"}, ["chat"],
      True, "<answer>13</answer>",
      [{"role": "user", "content": "What is 2+2?"}, {"role": "user", "content": "Now multiply that by 3."}]),
]}, open(sys.argv[1], "w"), indent=1)
PY
# ...and its certification (#3962 J2): the fixture receipt covers exactly these bytes.
CERT="$TMP/cert.json"
cert_for() { # cert_for <prompts> <receipt>: a fixture certification covering exactly those bytes
python3 - "$ROOT/scripts/lib" "$1" "$2" <<'PY'
import hashlib, json, sys
sys.path.insert(0, sys.argv[1]); import crux_prompt_certify as c
json.dump({"schema": c.SCHEMA, "prompts": sys.argv[2], "prompts_sha256": hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest(),
           "admitted": {"fixture/Q4_K_M": ["golden-2plus2", "golden-greeting", "golden-paris", "chat-arith-2turn"]},
           "admitted_by_sha": {"a" * 64: __import__("os").environ.get("ADMIT", "golden-2plus2,golden-greeting,golden-paris,chat-arith-2turn").split(",")},
           **({"admitted_by_sha_thinking": {"a" * 64: json.loads(__import__("os").environ["ADMIT_THINKING"])}}
              if __import__("os").environ.get("ADMIT_THINKING") else {}),
           "rejected": {}, "uncontrolled": [], "cells": []}, open(sys.argv[3], "w"))
PY
}
cert_for "$PROMPTS" "$CERT"
ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }

# ---- fixture writers: one engine's raw output, in the shape it really prints ---
SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
apr_out() { # apr_out <dir> <pid> <text> <ran> <fell_back> [ids]
  printf '{"text": %s, "tokens_generated": 3, "tok_per_sec": 9.5, "backend": {"requested": "gpu", "ran": "%s", "fell_back": %s}}\n' \
    "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$3")" "$4" "$5" > "$1/apr-$2.out"
  printf '[DEBUG] formatted_prompt="%s"\n[DEBUG] add_bos=false, encoded %s tokens: [%s]\n' "${APR_RENDER:-<|im_start|>user\\nq<|im_end|>\\n}" \
    "$(printf '%s' "${6:-1,2,3}" | tr ',' '\n' | grep -c .)" "${6:-1,2,3}" > "$1/apr-$2.err"
}
llama_out() { # llama_out <dir> <pid> <prompt> <answer> — the pinned chat CLI's stdout
  printf 'build : b10987-d1d3c3396\n\n> %s\n%s\n\n[ Prompt: 282.9 t/s | Generation: 104.9 t/s ]\n\nExiting...\n' "$3" "$4" > "$1/llama-$2.out"
  : > "$1/llama-$2.err"
}
ollama_out() { # ollama_out <dir> <pid> <answer>
  # ollama's real --verbose stderr uses the token `eval` three times ("prompt eval
  # count", "eval count", "eval rate") and this fixture must reproduce those bytes
  # exactly, or the judge parses a shape ollama never emits. bashrs SEC001 pattern-
  # matches the bare word `eval` and cannot tell a literal inside a single-quoted
  # format string from a call, so the token is passed as an argument instead. The
  # emitted bytes are unchanged; only the shape of the source line is. The gate runs
  # `bashrs lint --no-ignore`, so a .bashrsignore entry would be ignored by design.
  local ev='eval'
  printf '%s\n' "$3" > "$1/ollama-$2.out"
  printf 'total duration:       1.2s\nprompt %s count:    30 token(s)\n%s count:           8 token(s)\n%s rate:            90.00 tokens/s\n' "$ev" "$ev" "$ev" > "$1/ollama-$2.err"
}
row() { # row <manifest> <engine> <pid> <rc> <stdout> <stderr> [refused]
  python3 - "$@" <<'PY'
import json, os, sys
m, eng, pid, rc, o, e = sys.argv[1:7]
ref = sys.argv[7] if len(sys.argv) > 7 else ""
open(m, "a").write(json.dumps({"kind": "gen", "engine": eng, "prompt_id": pid, "rc": int(rc) if rc else None,
    "stdout": o or None, "stderr": e or None, "refused": ref or None, "model_sha256": "%s",
    "host": "fixture", "verb": os.environ.get("ROW_VERB", "run"), "thinking": os.environ.get("ROW_THINKING", "off"),
    "backend": "gpu"}) + "\n")
PY
}
tokrow() { # tokrow <manifest> <dir> <pid> <ids csv>
  printf '{"tokens": [%s]}\n' "$4" > "$2/ids-$3.json"
  printf '{"kind": "tok", "engine": "llama.cpp", "model_sha256": "%%s", "prompt_id": "%s", "ids": "%s"}\n' "$3" "$2/ids-$3.json" >> "$1"
}
# The manifest rows carry the literal "%s" for the sha; fill it in one place.
seal() { sed -i "s/\"%s\"/\"$SHA\"/g" "$1"; }

engine_out() { # engine_out <dir> <engine> <pid> <answer> [device] — row contract v1: {"text": ...}
  python3 -c 'import json,sys; json.dump({"text": sys.argv[2], "reported": {"reported_by": "fixture", "device": sys.argv[3]}}, open(sys.argv[1], "w"))' \
    "$1/$2-$3.json" "$4" "${5-cuda:0 fixture}"
}
detrow() { # detrow <manifest> <kind> <engine> <pid> <artifact path> [thinking] [refused]
  python3 - "$@" <<'PY'
import json, sys
m, kind, eng, pid, path = sys.argv[1:6]
thinking = sys.argv[6] if len(sys.argv) > 6 else "unset"
ref = sys.argv[7] if len(sys.argv) > 7 else ""
row = {"kind": kind, "engine": eng, "model_sha256": "%s", "host": "fixture", "prompt_id": pid, "refused": ref or None}
if kind == "tok":
    row.update({"input": path + ".txt", "ids": path})
elif kind == "tmpl":
    row.update({"thinking": thinking, "messages": path + ".messages.json", "rendered": path})
else:
    row.update({"steps": 4, "tokens": path, "logits": path + ".npy"})
    if len(sys.argv) > 6:
        row["thinking"] = thinking   # #3957 F9: a greedy row's thinking mode is part of its key
open(m, "a").write(json.dumps(row) + "\n")
PY
}
npy() { # npy <path> <rows as a,b,c;d,e,f> — a float32 C-order 2-D .npy, stdlib only
  python3 - "$1" "$2" <<'PY'
import struct, sys
rows = [[float(x) for x in r.split(",")] for r in sys.argv[2].split(";")]
n, m = len(rows), len(rows[0])
hdr = ("{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (n, m)).encode("latin1")
hdr += b" " * ((64 - (10 + len(hdr) + 1) % 64) % 64) + b"\n"
with open(sys.argv[1], "wb") as fh:
    fh.write(b"\x93NUMPY\x01\x00" + struct.pack("<H", len(hdr)) + hdr)
    fh.write(struct.pack("<%df" % (n * m), *[v for r in rows for v in r]))
PY
}
control_green() { # control_green <dir>: the positive control answered right by apr, llama.cpp (the
  #                   same-representation oracle) and hf (the bf16 ground-truth control, #3957 Q2)
  apr_out "$1" golden-2plus2 "<answer>4</answer>" gpu false; llama_out "$1" golden-2plus2 "What is 2+2?" "<answer>4</answer>"
  engine_out "$1" hf golden-2plus2 "<answer>4</answer>"
  row "$1/manifest.jsonl" apr golden-2plus2 0 "$1/apr-golden-2plus2.out" "$1/apr-golden-2plus2.err"
  row "$1/manifest.jsonl" llama.cpp golden-2plus2 0 "$1/llama-golden-2plus2.out" "$1/llama-golden-2plus2.err"
  row "$1/manifest.jsonl" hf golden-2plus2 0 "$1/hf-golden-2plus2.json" ""
}
hf_ok() { # hf_ok <dir> <pid> [answer]: a bf16 ground-truth control row that answered right
  engine_out "$1" hf "$2" "${3:-<answer>4</answer>}"; row "$1/manifest.jsonl" hf "$2" 0 "$1/hf-$2.json" ""
}
reason_has() { # reason_has <name> <case dir> <prompt id> <substring>: the cell's reasons name it
  local got; got=$(python3 -c 'import json,sys
r = json.load(open(sys.argv[1])); c = next((x for x in r["cells"] if x["key"]["prompt_id"] == sys.argv[2]), None)
print("ABSENT" if c is None else " | ".join(c.get("reasons") or []))' "$2/receipt.json" "$3" 2>/dev/null)
  case "$got" in *"$4"*) ok "$1" ;; *) broke "$1: reasons were '$got'" ;; esac
}
declined_has() { # declined_has <name> <case dir> <substring>
  local got; got=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["declined_because"])' "$2/receipt.json" 2>/dev/null)
  case "$got" in *"$3"*) ok "$1" ;; *) broke "$1: declined_because was '$got'" ;; esac
}
det_verdict() { # det_verdict <case dir> <kind> <prompt id> -> "<verdict> <first_difference of the first reference>"
  python3 -c 'import json,sys
r = json.load(open(sys.argv[1]))
d = next((x for x in r.get("deterministic", []) if x["kind"] == sys.argv[2] and x["key"]["prompt_id"] == sys.argv[3]), None)
if d is None: print("ABSENT"); sys.exit(0)
ref = next(iter(d["references"].values()), {})
print(d["verdict"], ref.get("first_difference"))' "$1/receipt.json" "$2" "$3" 2>/dev/null
}

run_judge() { # run_judge <case dir> — prints the receipt path; returns the judge's rc
  # Every engine's version is recorded, as the producer does; UNPIN_ENGINE=<e> drops one (#3952).
  python3 - "$1/meta.json" "${META_ENGINES:-[\"apr\", \"llama.cpp\", \"ollama\"]}" "${UNPIN_ENGINE:-}" <<'PY'
import json, sys
out, engines, unpin = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
meta = {"version": "0.0.0", "host": "fixture", "backend": "gpu", "engines": engines,
        "models": [{"name": __import__("os").environ.get("MODEL_NAME", "fixture-q4_k_m.gguf"),
                    "sha256": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],
        "apr": {"version_line": "apr 0.0.0 (fixture)"}, "llama_cpp": {"build": "b0-fixture"},
        "ollama": {"server_version": "0.0.0-fixture"}}
for e in ("hf", "llamafile", "vllm"):
    meta[e] = {"probe": "%s=fixture" % e}
if unpin:
    meta[unpin if unpin != "llama.cpp" else "llama_cpp"] = {}
json.dump(meta, open(out, "w"))
PY
  seal "$1/manifest.jsonl"
  PYTHONPATH="$ROOT/scripts/lib" python3 "$JUDGE" collect --manifest "$1/manifest.jsonl" --prompts "$PROMPTS" --meta "$1/meta.json" \
    ${CERT:+--certification "$CERT"} --out-json "$1/receipt.json" --out-md "$1/receipt.md" > "$1/judge.out" 2> "$1/judge.err"
}
verdict_of() { # verdict_of <case dir> <prompt id>
  python3 -c 'import json,sys
r = json.load(open(sys.argv[1]))
print(next((c["verdict"] for c in r["cells"] if c["key"]["prompt_id"] == sys.argv[2]), "ABSENT"))' "$1/receipt.json" "$2" 2>/dev/null
}
expect() { # expect <name> <case dir> <want rc> <prompt id> <want verdict>
  local rc=$3 got_v
  got_v=$(verdict_of "$2" "$4")
  if [ "$GOT_RC" = "$rc" ] && [ "$got_v" = "$5" ]; then ok "$1 (rc $GOT_RC, $4 $got_v)"
  else broke "$1: want rc $rc + $5, got rc $GOT_RC + $got_v ($(tail -1 "$2/judge.err" 2>/dev/null))"; fi
}
newcase() { local d="$TMP/$1"; mkdir -p "$d"; : > "$d/manifest.jsonl"; printf '%s' "$d"; }

Q="What is 2+2?"
P=golden-2plus2
T4="<answer>4</answer>"
T5="<answer>5</answer>"

printf '=== CRUX inference judge case table ===\n'

# 1. every engine right: GREEN, and the run passes.
d=$(newcase all_right)
apr_out "$d" $P "2+2 equals <answer>4</answer>." gpu false; llama_out "$d" $P "$Q" "2 + 2 = $T4"; ollama_out "$d" $P "$T4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" ollama $P 0 "$d/ollama-$P.out" "$d/ollama-$P.err"
hf_ok "$d" $P
run_judge "$d"; GOT_RC=$?
expect "every engine right is GREEN and passes" "$d" 0 $P GREEN

# 2. THE FALSIFIER: llama.cpp right, apr wrong.
d=$(newcase llama_right_apr_wrong)
apr_out "$d" $P "$T5" gpu false; llama_out "$d" $P "$Q" "$T4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" ollama $P "" "" "" "not requested here"; hf_ok "$d" $P
run_judge "$d"; GOT_RC=$?
expect "llama.cpp right and apr wrong is RED" "$d" 1 $P RED

# 3. ollama right, apr exited 14 after falling back off the GPU.
d=$(newcase apr_fell_back)
apr_out "$d" $P "$T4" cpu true; ollama_out "$d" $P "$T4"; hf_ok "$d" $P
row "$d/manifest.jsonl" apr $P 14 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P "" "" "" "unresolved"
row "$d/manifest.jsonl" ollama $P 0 "$d/ollama-$P.out" "$d/ollama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "apr right text but fell back and exited 14 is RED" "$d" 1 $P RED

# 4. apr ran on another backend with rc 0 and fell_back=false: still not the lane.
d=$(newcase apr_wrong_backend)
apr_out "$d" $P "$T4" cpu false; llama_out "$d" $P "$Q" "$T4"; hf_ok "$d" $P
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "apr on a backend the lane did not ask for is RED" "$d" 1 $P RED

# 4b. apr printed a right-looking answer on the lane's backend and still exited non-zero.
d=$(newcase apr_nonzero_exit)
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"; hf_ok "$d" $P
row "$d/manifest.jsonl" apr $P 1 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "apr exiting non-zero is not an answer, however right it reads" "$d" 1 $P RED

# 4c. the same on the comparator side: a comparator that exited non-zero cannot vouch -- and with no
#     other same-representation engine the cell is RED, never a third state (#3957 Q1).
d=$(newcase llama_nonzero_exit)
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"; hf_ok "$d" $P
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 1 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "a comparator exiting non-zero is no oracle: RED, not UNJUDGED (#3957 Q1)" "$d" 1 $P RED
reason_has "and the cell names the missing same-representation oracle" "$d" $P "no same-representation oracle"

# 5. the apr row is MISSING while llama.cpp answered right: absence is a violation.
d=$(newcase apr_missing)
llama_out "$d" $P "$Q" "$T4"; hf_ok "$d" $P
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "a missing apr row where llama.cpp is right is RED" "$d" 1 $P RED

# 6. no comparator answered: apr right about itself is not evidence -- RED (#3957 Q1), never GREEN.
d=$(newcase no_oracle)
apr_out "$d" $P "$T4" gpu false
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P "" "" "" "unresolved: no_binary_named"
row "$d/manifest.jsonl" ollama $P "" "" "" "imported with no chat template"
run_judge "$d"; GOT_RC=$?
expect "no comparator answered is RED (no oracle), never UNJUDGED" "$d" 1 $P RED
reason_has "and it names both missing oracles" "$d" $P "no ground-truth control"

# 7. everyone answered wrong: RED, with ALL_WRONG as a flag on it (#3957 F6: ALL_WRONG is RED).
d=$(newcase all_wrong)
apr_out "$d" $P "$T5" gpu false; llama_out "$d" $P "$Q" "<answer>22</answer>"; ollama_out "$d" $P "<answer>22</answer>"
hf_ok "$d" $P "<answer>3</answer>"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" ollama $P 0 "$d/ollama-$P.out" "$d/ollama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "everyone wrong is RED (ALL_WRONG is RED, #3957 F6)" "$d" 1 $P RED
field_all_wrong=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][0]["all_wrong"])' "$d/receipt.json" 2>/dev/null)
[ "$field_all_wrong" = True ] && ok "and the cell is flagged all_wrong" || broke "all_wrong flag: '$field_all_wrong'"

# 8. a comparator whose output cannot be parsed did not answer; it cannot vouch.
d=$(newcase llama_unparsed)
apr_out "$d" $P "$T4" gpu false; hf_ok "$d" $P
printf 'banner only, the turn never finished\n' > "$d/llama-$P.out"; : > "$d/llama-$P.err"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "an unparseable comparator is no oracle: RED" "$d" 1 $P RED
reason_has "and the cell says no same-representation engine answered" "$d" $P "no ggml-family engine answered"

# 9. one RED among GREENs still fails the whole run.
d=$(newcase one_red_among_green)
for p in golden-2plus2 golden-paris; do
  case $p in golden-2plus2) q="$Q"; a_apr="$T4" ;; golden-paris) q="What is the capital of France?"; a_apr="<answer>Lyon</answer>" ;; esac
  case $p in golden-2plus2) a_ref="$T4" ;; golden-paris) a_ref="<answer>Paris</answer>" ;; esac
  apr_out "$d" $p "$a_apr" gpu false; llama_out "$d" $p "$q" "$a_ref"; hf_ok "$d" $p "$a_ref"
  row "$d/manifest.jsonl" apr $p 0 "$d/apr-$p.out" "$d/apr-$p.err"
  row "$d/manifest.jsonl" llama.cpp $p 0 "$d/llama-$p.out" "$d/llama-$p.err"
done
run_judge "$d"; GOT_RC=$?
expect "one RED cell among GREEN ones fails the run" "$d" 1 golden-paris RED

# 9b. one uncorroborated cell among GREEN ones: it was never compared -- RED (#3957 Q1), the run RED.
d=$(newcase one_unjudged_among_green)
control_green "$d"
apr_out "$d" golden-paris "<answer>Paris</answer>" gpu false; hf_ok "$d" golden-paris "<answer>Paris</answer>"
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 124 "" "" ""
run_judge "$d"; GOT_RC=$?
expect "one uncorroborated cell among GREEN ones is RED and fails the run" "$d" 1 golden-paris RED

# 9c. #3957 F6 REVERSES the old ruling: a non-control ALL_WRONG is RED (the run fails), still counted per model.
d=$(newcase all_wrong_named)
control_green "$d"
apr_out "$d" golden-paris "<answer>Lyon</answer>" gpu false; llama_out "$d" golden-paris "What is the capital of France?" "<answer>Marseille</answer>"
hf_ok "$d" golden-paris "<answer>Nice</answer>"
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 0 "$d/llama-golden-paris.out" "$d/llama-golden-paris.err"
run_judge "$d"; GOT_RC=$?
expect "a non-control ALL_WRONG is RED and fails the run (#3957 F6)" "$d" 1 golden-paris RED
got=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["all_wrong_by_model"])' "$d/receipt.json" 2>/dev/null)
[ "$got" = "{'$SHA': 1}" ] && ok "ALL_WRONG is counted per model (1)" || broke "all_wrong_by_model: got '$got'"

# 9d. the positive control comes back ALL_WRONG: RED, beside a GREEN cell.
d=$(newcase control_all_wrong)
apr_out "$d" golden-2plus2 "<answer>22</answer>" gpu false; llama_out "$d" golden-2plus2 "$Q" "<answer>22</answer>"
hf_ok "$d" golden-2plus2 "<answer>22</answer>"
row "$d/manifest.jsonl" apr golden-2plus2 0 "$d/apr-golden-2plus2.out" "$d/apr-golden-2plus2.err"
row "$d/manifest.jsonl" llama.cpp golden-2plus2 0 "$d/llama-golden-2plus2.out" "$d/llama-golden-2plus2.err"
apr_out "$d" golden-paris "<answer>Paris</answer>" gpu false; llama_out "$d" golden-paris "What is the capital of France?" "<answer>Paris</answer>"
hf_ok "$d" golden-paris "<answer>Paris</answer>"
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 0 "$d/llama-golden-paris.out" "$d/llama-golden-paris.err"
run_judge "$d"; GOT_RC=$?
expect "the positive control ALL_WRONG is RED beside a GREEN cell" "$d" 1 golden-2plus2 RED

# 9e. a prompt set that declares no positive control declines, however green the cells.
d=$(newcase no_control_declared)
python3 -c 'import json,sys
d = json.load(open(sys.argv[1]))
for p in d["prompts"]: p.pop("control", None)
json.dump(d, open(sys.argv[2], "w"))' "$PROMPTS" "$d/prompts.json"
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"; hf_ok "$d" $P
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
PROMPTS_SAVED=$PROMPTS; PROMPTS="$d/prompts.json"
run_judge "$d"; GOT_RC=$?
PROMPTS=$PROMPTS_SAVED
expect "a prompt set with no positive control declines" "$d" 2 $P GREEN
why=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["declined_because"])' "$d/receipt.json" 2>/dev/null)
case "$why" in
  *"declares no positive control"*) ok "the decline names the undeclared control, not a symptom of it" ;;
  *) broke "undeclared control: declined_because was '$why'" ;;
esac

# 9e2. every cell says whether it is the positive control (pv reads the flag, #3715).
got=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sorted((c["key"]["prompt_id"], c["positive_control"]) for c in r["cells"]))' "$TMP/control_all_wrong/receipt.json" 2>/dev/null)
[ "$got" = "[('golden-2plus2', True), ('golden-paris', False)]" ] && ok "cells carry positive_control (true only on the control)" || broke "positive_control flags: '$got'"

# 9f. a model whose run never measured the control prompt declines, however green:
#     an unmeasured control proves nothing about the harness.
d=$(newcase no_control_cell)
apr_out "$d" golden-paris "<answer>Paris</answer>" gpu false; llama_out "$d" golden-paris "What is the capital of France?" "<answer>Paris</answer>"
hf_ok "$d" golden-paris "<answer>Paris</answer>"
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 0 "$d/llama-golden-paris.out" "$d/llama-golden-paris.err"
run_judge "$d"; GOT_RC=$?
expect "a model with no measured control cell declines" "$d" 2 golden-paris GREEN

# ---- the 19:03Z / 19:04Z engines: hf and llamafile (row contract v1) --------------
META_ENGINES='["apr", "llama.cpp", "ollama", "hf", "llamafile"]'

# E1. hf right, apr wrong: RED, exactly as for llama.cpp or ollama.
d=$(newcase hf_right_apr_wrong)
apr_out "$d" $P "$T5" gpu false; engine_out "$d" hf $P "2 + 2 = $T4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf right and apr wrong is RED" "$d" 1 $P RED

# E2. llamafile refuses the model (its own text quoted); llama.cpp answers: still judged, refusal named.
d=$(newcase llamafile_refused)
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" llamafile $P "" "" "" "error loading model architecture: unknown model architecture: 'qwen35'"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""; engine_out "$d" hf $P "$T4"
run_judge "$d"; GOT_RC=$?
expect "a llamafile refusal beside a right llama.cpp is GREEN" "$d" 0 $P GREEN
why=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][0]["engines"]["llamafile"]["why"])' "$d/receipt.json" 2>/dev/null)
case "$why" in *"unknown model architecture: 'qwen35'"*) ok "llamafile's refusal is quoted, not dropped" ;; *) broke "llamafile refusal: '$why'" ;; esac

# E3. an hf answer that is not the contract's JSON is no answer, so it cannot vouch.
d=$(newcase hf_bad_json)
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"; row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"; printf '2 + 2 = 4\n' > "$d/hf-$P.json"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf output that is not the contract JSON is no control: RED" "$d" 1 $P RED
reason_has "  ...because no bf16 engine answered" "$d" $P "no ground-truth control"

# E3b. valid JSON that is not the contract's shape (no "text") is no answer either.
d=$(newcase hf_json_without_text)
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"; row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"; printf '{"answer": "$T4"}\n' > "$d/hf-$P.json"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf JSON without a text field is no control: RED" "$d" 1 $P RED
reason_has "  ...because no bf16 engine answered" "$d" $P "no ground-truth control"

# E3c. a plugin engine is held to its lane: a gpu-lane row that ran on the CPU,
#      or a row with no reported device, cannot vouch.
d=$(newcase hf_wrong_device)
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"; row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"; engine_out "$d" hf $P "$T4" "cpu"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf on the CPU in the gpu lane is no control: RED" "$d" 1 $P RED
reason_has "  ...because no bf16 engine answered" "$d" $P "no ground-truth control"
d=$(newcase hf_no_device)
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"; row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"; engine_out "$d" hf $P "$T4" ""
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf with no reported device is no control: RED" "$d" 1 $P RED
reason_has "  ...because no bf16 engine answered" "$d" $P "no ground-truth control"

# E4. a DEGENERATE answer is no answer from any engine: the golden greeting's "!"
#     pattern would otherwise score "!!!!!!!!" (token id 0, x64) as correct.
d=$(newcase degenerate_answers)
apr_out "$d" golden-greeting "!!!!!!!!!!!!!!!!" gpu false; engine_out "$d" hf golden-greeting "!!!!!!!!!!!!!!!!"
llama_out "$d" golden-greeting "Hello there, how are you doing today my friend?" "<answer>Hello</answer> I am well."
row "$d/manifest.jsonl" apr golden-greeting 0 "$d/apr-golden-greeting.out" "$d/apr-golden-greeting.err"
row "$d/manifest.jsonl" hf golden-greeting 0 "$d/hf-golden-greeting.json" ""
row "$d/manifest.jsonl" llama.cpp golden-greeting 0 "$d/llama-golden-greeting.out" "$d/llama-golden-greeting.err"
control_green "$d"
run_judge "$d"; GOT_RC=$?
expect "apr's degenerate '!!!!' is no answer: RED where llama.cpp answered" "$d" 1 golden-greeting RED
got=$(python3 -c 'import json,sys; c=[x for x in json.load(open(sys.argv[1]))["cells"] if x["key"]["prompt_id"]=="golden-greeting"][0]; print(c["engines"]["hf"]["answered"], c["engines"]["hf"]["why"][:18])' "$d/receipt.json" 2>/dev/null)
case "$got" in "False degenerate output"*) ok "hf's degenerate '!!!!' cannot vouch either" ;; *) broke "hf degenerate: '$got'" ;; esac

# V1-V5. vLLM (#3952): a plugin engine under the same rule. vLLM 0.30.0 cannot load
#        the GGUF, so its row runs the SOURCE weights and says so in `source`.
vllm_row() { # vllm_row <manifest> <pid> <rc> <stdout> [refused] — the driver's row, with its source
  python3 - "$@" <<'PY'
import json, os, sys
m, pid, rc, o = sys.argv[1:5]
ref = sys.argv[5] if len(sys.argv) > 5 else ""
open(m, "a").write(json.dumps({"kind": "gen", "engine": "vllm", "prompt_id": pid, "rc": int(rc) if rc else None,
    "stdout": o or None, "stderr": None, "refused": ref or None, "model_sha256": "%s", "host": "fixture",
    "verb": os.environ.get("ROW_VERB", "run"), "thinking": "off", "backend": "gpu",
    "source": {"repo": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "revision": "ea3f2471", "dtype": "bfloat16",
               "compares": "apr quantized file vs source weights (vLLM 0.30.0 cannot load the GGUF, #3952)"}}) + "\n")
PY
}
META_ENGINES='["apr", "llama.cpp", "ollama", "hf", "llamafile", "vllm"]'
d=$(newcase vllm_right_apr_wrong)
apr_out "$d" $P "$T5" gpu false; engine_out "$d" vllm $P "2 + 2 = $T4" "cuda:0 NVIDIA GeForce RTX 4090"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
vllm_row "$d/manifest.jsonl" $P 0 "$d/vllm-$P.json"
run_judge "$d"; GOT_RC=$?
expect "vllm right and apr wrong is RED" "$d" 1 $P RED
src=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][0]["engines"]["vllm"]["source"]["compares"])' "$d/receipt.json" 2>/dev/null)
case "$src" in *"source weights"*) ok "a vllm verdict carries what it compared (source weights, not the file)" ;; *) broke "vllm source: '$src'" ;; esac

d=$(newcase vllm_refused)
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"; hf_ok "$d" $P
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
vllm_row "$d/manifest.jsonl" $P "" "" "RuntimeError: Engine core initialization failed — engine core: AssertionError: Error in memory profiling"
run_judge "$d"; GOT_RC=$?
expect "a vllm refusal beside a right llama.cpp is GREEN" "$d" 0 $P GREEN
why=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][0]["engines"]["vllm"]["why"])' "$d/receipt.json" 2>/dev/null)
case "$why" in *"AssertionError: Error in memory profiling"*) ok "vllm's refusal is quoted, root cause included" ;; *) broke "vllm refusal: '$why'" ;; esac

# V6. #3952: a comparator with no recorded version cannot vouch. Before this, every plugin cell's version
#     was None (engine_versions read `version`; the producer writes `probe`) and it vouched anyway.
d=$(newcase vllm_unpinned)
apr_out "$d" $P "$T4" gpu false; engine_out "$d" vllm $P "2 + 2 = $T4" "cuda:0 NVIDIA GeForce RTX 4090"
llama_out "$d" $P "$Q" "$T4"; row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
vllm_row "$d/manifest.jsonl" $P 0 "$d/vllm-$P.json"
UNPIN_ENGINE=vllm run_judge "$d"; GOT_RC=$?
expect "an UNPINNED vllm cannot vouch: RED, not proven (no third state)" "$d" 1 $P RED
why=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][0]["engines"]["vllm"]["why"])' "$d/receipt.json" 2>/dev/null)
case "$why" in "unpinned:"*) ok "the unpinned vllm is named as unpinned" ;; *) broke "vllm unpinned why: '$why'" ;; esac
rsn=$(python3 -c 'import json,sys; print(" | ".join(json.load(open(sys.argv[1]))["cells"][0]["reasons"]))' "$d/receipt.json" 2>/dev/null)
case "$rsn" in *"oracle unpinned: vllm"*"not a finding that apr was wrong"*) ok "the RED says it is an unpinned oracle, not apr being wrong" ;; *) broke "unpinned reasons: '$rsn'" ;; esac
# A CONTROLLED PAIR: apr right, llama.cpp (same representation) agreeing, vllm the bf16 control. Pinned it
# is GREEN; the ONLY change, unpinning vllm, turns it RED. Without the pair an unpinned case could be RED for
# another reason (F6 needs a same-representation engine AND a control) and the rule would go unmeasured.
for pin in pinned unpinned; do
  d=$(newcase "vllm_${pin}_apr_right")
  apr_out "$d" $P "$T4" gpu false; engine_out "$d" vllm $P "2 + 2 = $T4" "cuda:0 NVIDIA GeForce RTX 4090"
  llama_out "$d" $P "$Q" "$T4"; row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
  row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
  vllm_row "$d/manifest.jsonl" $P 0 "$d/vllm-$P.json"
  if [ "$pin" = pinned ]; then run_judge "$d"; else UNPIN_ENGINE=vllm run_judge "$d"; fi; GOT_RC=$?
  if [ "$pin" = pinned ]; then
    expect "control half: the same cell with vllm PINNED is GREEN" "$d" 0 $P GREEN
  else
    expect "an UNPINNED vllm agreeing with a right apr is still RED, never GREEN" "$d" 1 $P RED
    rsn=$(python3 -c 'import json,sys; print(" | ".join(json.load(open(sys.argv[1]))["cells"][0]["reasons"]))' "$d/receipt.json" 2>/dev/null)
    case "$rsn" in *"oracle unpinned: vllm"*) ok "...and its reason is the unpinned oracle" ;; *) broke "unpinned-apr-right reasons: '$rsn'" ;; esac
  fi
done
rsn=$(python3 -c 'import json,sys; print(" | ".join(json.load(open(sys.argv[1]))["cells"][0]["reasons"]))' "$TMP/vllm_right_apr_wrong/receipt.json" 2>/dev/null)
case "$rsn" in *"oracle unpinned"*) broke "a plain-rule RED names an unpinned oracle: '$rsn'" ;; *) ok "a plain-rule RED names no unpinned oracle" ;; esac
ver=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][0]["engines"]["vllm"]["version"])' "$TMP/vllm_right_apr_wrong/receipt.json" 2>/dev/null)
[ "$ver" = "vllm=fixture" ] && ok "a pinned vllm's cell carries its probe line as its version" || broke "vllm cell version: '$ver'"

d=$(newcase vllm_wrong_device)
apr_out "$d" $P "$T4" gpu false; engine_out "$d" vllm $P "$T4" "cpu"
llama_out "$d" $P "$Q" "$T4"; row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
vllm_row "$d/manifest.jsonl" $P 0 "$d/vllm-$P.json"
run_judge "$d"; GOT_RC=$?
expect "vllm on the CPU in the gpu lane is no control: RED" "$d" 1 $P RED
# The verdict alone would pass for an engine the judge does not know at all ("unknown engine");
# the reason is what proves the lane check ran.
why=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][0]["engines"]["vllm"]["why"])' "$d/receipt.json" 2>/dev/null)
case "$why" in *"is not the gpu lane"*) ok "vllm is held to its lane by the device check, not refused as unknown" ;; *) broke "vllm lane: '$why'" ;; esac

d=$(newcase vllm_serve_right_apr_wrong)
apr_out "$d" $P "$T5" gpu false; engine_out "$d" vllm $P "$T4" "cuda:0 NVIDIA GB10 (vllm serve)"
python3 -c 'import json,sys; json.dump({"text": "<answer>5</answer>", "reported": {}}, open(sys.argv[1], "w"))' "$d/apr-$P.json"
ROW_VERB="serve run" row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.json" ""
ROW_VERB="serve run" vllm_row "$d/manifest.jsonl" $P 0 "$d/vllm-$P.json"
run_judge "$d"; GOT_RC=$?
expect "vllm serve right and apr serve wrong is RED" "$d" 1 $P RED

# C1-C3. the chat verb (#3739 slice 3): judged on the FINAL turn; apr's backend is
#        recorded as unverified until apr chat reports one (#3794).
apr_chat_out() { # apr_chat_out <dir> <pid> <reply 1> <reply 2> — apr chat's transcript shape
  printf 'Detected ChatML chat template\nYou: \nAssistant: %s\nYou: \nAssistant: %s\nYou: \nGoodbye!\n' "$3" "$4" > "$1/apr-$2.out"
  : > "$1/apr-$2.err"
}
chat_json() { # chat_json <dir> <engine> <pid> <reply 1> <reply 2> — the pty helper's JSON
  python3 -c 'import json,sys; json.dump({"text": sys.argv[3], "turns": [sys.argv[2], sys.argv[3]], "reported": {"device": "cuda (-ngl 999)", "interface": "pty"}}, open(sys.argv[1], "w"))' \
    "$1/$2-$3.json" "$4" "$5"
}
chat_hf() { # chat_hf <dir> <pid> <reply 1> <reply 2>: the bf16 control's chat row
  chat_json "$1" hf "$2" "$3" "$4"; ROW_VERB=chat row "$1/manifest.jsonl" hf "$2" 0 "$1/hf-$2.json" ""
}
CP=chat-arith-2turn
d=$(newcase chat_green); control_green "$d"
apr_chat_out "$d" $CP "2 + 2 equals <answer>4</answer>." "4 * 3 equals <answer>12</answer>."; chat_json "$d" llama $CP "<answer>4</answer>" "<answer>12</answer>"
chat_hf "$d" $CP "<answer>4</answer>" "<answer>12</answer>"
ROW_VERB=chat row "$d/manifest.jsonl" apr $CP 0 "$d/apr-$CP.out" "$d/apr-$CP.err"
ROW_VERB=chat row "$d/manifest.jsonl" llama.cpp $CP 0 "$d/llama-$CP.json" ""
run_judge "$d"; GOT_RC=$?
expect "a chat whose final turn carries the first turn's answer is GREEN" "$d" 0 $CP GREEN
got=$(python3 -c 'import json,sys; c=[x for x in json.load(open(sys.argv[1]))["cells"] if x["key"]["verb"]=="chat"][0]; a=c["engines"]["apr"]; print(a["turns"], a["backend_verified"], c["token_parity"].get("why"))' "$d/receipt.json" 2>/dev/null)
[ "$got" = "['2 + 2 equals <answer>4</answer>.', '4 * 3 equals <answer>12</answer>.'] False not measured for the chat verb" ] && ok "every apr turn recorded; backend unverified (#3794); no run-verb token parity" || broke "chat record: '$got'"
d=$(newcase chat_red); control_green "$d"
apr_chat_out "$d" $CP "<answer>4</answer>" "<answer>9</answer>"; chat_json "$d" llama $CP "<answer>4</answer>" "<answer>12</answer>"
chat_hf "$d" $CP "<answer>4</answer>" "<answer>12</answer>"
ROW_VERB=chat row "$d/manifest.jsonl" apr $CP 0 "$d/apr-$CP.out" "$d/apr-$CP.err"
ROW_VERB=chat row "$d/manifest.jsonl" llama.cpp $CP 0 "$d/llama-$CP.json" ""
run_judge "$d"; GOT_RC=$?
expect "apr losing the first turn's answer while llama.cpp carries it is RED" "$d" 1 $CP RED
d=$(newcase chat_no_turn); control_green "$d"
printf 'Detected ChatML chat template\nYou: \nGoodbye!\n' > "$d/apr-$CP.out"; : > "$d/apr-$CP.err"; chat_json "$d" llama $CP "<answer>4</answer>" "<answer>12</answer>"
chat_hf "$d" $CP "<answer>4</answer>" "<answer>12</answer>"
ROW_VERB=chat row "$d/manifest.jsonl" apr $CP 0 "$d/apr-$CP.out" "$d/apr-$CP.err"
ROW_VERB=chat row "$d/manifest.jsonl" llama.cpp $CP 0 "$d/llama-$CP.json" ""
run_judge "$d"; GOT_RC=$?
expect "an apr chat transcript with no Assistant turn is no answer (RED)" "$d" 1 $CP RED

# S1-S3. the serve verb (#3739 slice 4): every server through the ONE OpenAI client,
#        nonstream and stream as distinct cells (the key's additive `mode`).
serve_json() { # serve_json <dir> <engine> <pid> <mode> <text|""> [error] — crux_openai_client.py's output shape
  python3 -c 'import json,sys
t = sys.argv[5] or None
d = {"text": t, "reported": {"device": "fixture", "stream": sys.argv[4] == "stream", "finish_reason": "stop" if t else None, "usage": None, "chunks": 0}}
if len(sys.argv) > 6: d["error"] = sys.argv[6]
json.dump(d, open(sys.argv[1], "w"))' "$1/$2-$3-$4.json" "$2" "$3" "$4" "$5" "${@:6}"
}
serve_row() { # serve_row <manifest> <engine> <pid> <mode> <rc> <json>
  python3 - "$@" <<'PY2'
import json, sys
m, eng, pid, mode, rc, o = sys.argv[1:7]
open(m, "a").write(json.dumps({"kind": "gen", "engine": eng, "prompt_id": pid, "rc": int(rc), "stdout": o, "stderr": None,
    "refused": None, "model_sha256": "%s", "host": "fixture", "verb": "serve run", "thinking": "off", "backend": "gpu",
    "mode": mode}) + "\n")
PY2
}
d=$(newcase serve_green); control_green "$d"
for mode in nonstream stream; do
  serve_json "$d" apr $P $mode "$T4"; serve_json "$d" llama $P $mode "2 + 2 = $T4"; serve_json "$d" hf $P $mode "$T4"
  serve_row "$d/manifest.jsonl" apr $P $mode 0 "$d/apr-$P-$mode.json"; serve_row "$d/manifest.jsonl" llama.cpp $P $mode 0 "$d/llama-$P-$mode.json"
  serve_row "$d/manifest.jsonl" hf $P $mode 0 "$d/hf-$P-$mode.json"
done
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sorted((c["key"].get("mode"), c["verdict"], c["engines"]["apr"].get("backend_verified")) for c in r["cells"] if c["key"]["verb"]=="serve run"))' "$d/receipt.json" 2>/dev/null)
[ "$GOT_RC" = 0 ] && [ "$got" = "[('nonstream', 'GREEN', False), ('stream', 'GREEN', False)]" ] && ok "serve nonstream and stream are distinct GREEN cells, and the apr serve backend is unverified" || broke "serve green: rc $GOT_RC '$got'"
d=$(newcase serve_apr_refused); control_green "$d"
serve_json "$d" apr $P nonstream "" "URLError: [Errno 111] Connection refused"; serve_json "$d" llama $P nonstream "2 + 2 = $T4"
serve_row "$d/manifest.jsonl" apr $P nonstream 3 "$d/apr-$P-nonstream.json"; serve_row "$d/manifest.jsonl" llama.cpp $P nonstream 0 "$d/llama-$P-nonstream.json"
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print([c["verdict"] for c in r["cells"] if c["key"]["verb"]=="serve run"])' "$d/receipt.json" 2>/dev/null)
[ "$GOT_RC" = 1 ] && [ "$got" = "['RED']" ] && ok "an apr serve that answered nothing while llama-server answered is RED" || broke "serve apr refused: rc $GOT_RC '$got'"

# D1-D4. tok: raw-text ids, BYTE-EQUAL or RED.
d=$(newcase tok_equal); control_green "$d"
printf '{"tokens": [9707, 11, 1879]}\n' > "$d/apr-tok.json"; printf '{"tokens": [9707, 11, 1879]}\n' > "$d/hf-tok.json"
detrow "$d/manifest.jsonl" tok apr tok-cjk-01 "$d/apr-tok.json"; detrow "$d/manifest.jsonl" tok hf tok-cjk-01 "$d/hf-tok.json"
run_judge "$d"; GOT_RC=$?; got=$(det_verdict "$d" tok tok-cjk-01)
[ "$GOT_RC" = 0 ] && [ "$got" = "GREEN None" ] && ok "tok ids equal to HF are GREEN (rc 0)" || broke "tok equal: rc $GOT_RC, '$got'"
d=$(newcase tok_differ); control_green "$d"
printf '{"tokens": [9707, 0, 0, 0]}\n' > "$d/apr-tok.json"; printf '{"tokens": [9707, 11, 1879]}\n' > "$d/hf-tok.json"
detrow "$d/manifest.jsonl" tok apr tok-cjk-01 "$d/apr-tok.json"; detrow "$d/manifest.jsonl" tok hf tok-cjk-01 "$d/hf-tok.json"
run_judge "$d"; GOT_RC=$?; got=$(det_verdict "$d" tok tok-cjk-01)
[ "$GOT_RC" = 1 ] && [ "$got" = "RED 1" ] && ok "tok ids differing from HF are RED at the first differing index (1)" || broke "tok differ: rc $GOT_RC, '$got'"
d=$(newcase tok_apr_missing); control_green "$d"
printf '{"tokens": [9707]}\n' > "$d/hf-tok.json"; detrow "$d/manifest.jsonl" tok hf tok-cjk-01 "$d/hf-tok.json"
run_judge "$d"; GOT_RC=$?; got=$(det_verdict "$d" tok tok-cjk-01)
[ "$GOT_RC" = 1 ] && [ "${got%% *}" = RED ] && ok "an HF tok row with no apr row is RED (absence)" || broke "tok apr missing: rc $GOT_RC, '$got'"
d=$(newcase tok_hf_refused); control_green "$d"
printf '{"tokens": [9707]}\n' > "$d/apr-tok.json"; detrow "$d/manifest.jsonl" tok apr tok-cjk-01 "$d/apr-tok.json"
detrow "$d/manifest.jsonl" tok hf tok-cjk-01 "$d/hf-tok.json" unset "no tokenizer.json in the source repo"
run_judge "$d"; GOT_RC=$?; got=$(det_verdict "$d" tok tok-cjk-01)
[ "$GOT_RC" = 1 ] && [ "${got%% *}" = RED ] && ok "a tok row HF refused is RED -- no third state (#3957 Q1)" || broke "tok hf refused: rc $GOT_RC, '$got'"

# D5-D6. tmpl: the chat rendering, BYTE-EQUAL or RED (the #3672 double wrap is the RED case).
d=$(newcase tmpl_equal); control_green "$d"
printf '<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n' > "$d/apr-t"; cp "$d/apr-t" "$d/hf-t"
detrow "$d/manifest.jsonl" tmpl apr golden-2plus2 "$d/apr-t" off; detrow "$d/manifest.jsonl" tmpl hf golden-2plus2 "$d/hf-t" off
run_judge "$d"; GOT_RC=$?; got=$(det_verdict "$d" tmpl golden-2plus2)
[ "$GOT_RC" = 0 ] && [ "$got" = "GREEN None" ] && ok "a rendering byte-equal to apply_chat_template is GREEN" || broke "tmpl equal: rc $GOT_RC, '$got'"
d=$(newcase tmpl_double_wrap); control_green "$d"
printf '<|im_start|>user\n<\xe2\x80\x8b|im_start|>user\nhi' > "$d/apr-t"; printf '<|im_start|>user\nhi<|im_end|>\n' > "$d/hf-t"
detrow "$d/manifest.jsonl" tmpl apr golden-2plus2 "$d/apr-t" off; detrow "$d/manifest.jsonl" tmpl hf golden-2plus2 "$d/hf-t" off
run_judge "$d"; GOT_RC=$?; got=$(det_verdict "$d" tmpl golden-2plus2)
[ "$GOT_RC" = 1 ] && [ "$got" = "RED 17" ] && ok "a double-wrapped rendering is RED at byte 17, where the second wrap starts" || broke "tmpl differ: rc $GOT_RC, '$got'"

# R1. greedy: the first divergence and the logit cosine there are REPORTED, never a verdict.
d=$(newcase greedy_report); control_green "$d"
printf '{"prompt_ids": [1], "generated_ids": [5, 6, 7, 8]}\n' > "$d/apr-g.json"; npy "$d/apr-g.json.npy" "1,0;0,1;1,1;0,0"
printf '{"prompt_ids": [1], "generated_ids": [5, 6, 9, 8]}\n' > "$d/hf-g.json";  npy "$d/hf-g.json.npy" "1,0;0,1;1,0;0,0"
detrow "$d/manifest.jsonl" greedy apr golden-paris "$d/apr-g.json"; detrow "$d/manifest.jsonl" greedy hf golden-paris "$d/hf-g.json"
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; g=json.load(open(sys.argv[1]))["greedy"][0]["hf"]; print(g.get("first_divergence"), round(g.get("logit_cosine_at_divergence") or -9, 4))' "$d/receipt.json" 2>/dev/null)
[ "$GOT_RC" = 0 ] && [ "$got" = "2 0.7071" ] && ok "greedy divergence at step 2 is reported with its logit cosine (0.7071), not judged" || broke "greedy: rc $GOT_RC, '$got'"
# R1b (#3957 F9). The receipt carries each engine's RAW greedy record, keyed by thinking, so the
# ladder judge compares the id lists itself; an engine that refused carries its refusal, never ids.
d=$(newcase greedy_raw_carried); control_green "$d"
printf '{"generated_ids": [5, 6, 7, 8], "generated_text": "<think>\\nx", "greedy": true, "special": true, "max_tokens": 4}\n' > "$d/ll-g.json"
detrow "$d/manifest.jsonl" greedy llama.cpp golden-paris "$d/ll-g.json" on
detrow "$d/manifest.jsonl" greedy apr golden-paris "$d/none.json" on "#3723: apr has no thinking toggle"
# #3990: llama.cpp on the model's OWN template is a second row for the same engine, kept under "llama.cpp@official".
printf '{"generated_ids": [9], "generated_text": "<think>\\ny", "greedy": true, "special": true, "max_tokens": 4}\n' > "$d/off-g.json"
detrow "$d/manifest.jsonl" greedy llama.cpp golden-paris "$d/off-g.json" on
python3 - "$d/manifest.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows[-1]["prompt_source"] = "official"
open(sys.argv[1], "w").write("".join(json.dumps(r) + "\n" for r in rows))
PY
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; g=json.load(open(sys.argv[1]))["greedy"][0]; print(g["key"].get("thinking"), (g.get("llama.cpp") or {}).get("raw", {}).get("generated_ids"), (g.get("llama.cpp@official") or {}).get("raw", {}).get("generated_ids"), (g.get("apr") or {}).get("refused"), "raw" in (g.get("apr") or {}))' "$d/receipt.json" 2>/dev/null)
[ "$GOT_RC" = 0 ] && [ "$got" = "on [5, 6, 7, 8] [9] #3723: apr has no thinking toggle False" ] && ok "greedy receipt carries the raw ids keyed by thinking, the official-template row apart, and a refusal in place of ids (#3957 F9, #3990)" || broke "greedy raw: rc $GOT_RC, '$got'"
unset META_ENGINES

# 10-11. token parity: equal ids agree; a divergence is located, never averaged away.
d=$(newcase parity)
apr_out "$d" $P "$T4" gpu false "151644,872,198,3838"; llama_out "$d" $P "$Q" "$T4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
tokrow "$d/manifest.jsonl" "$d" $P "151644,872,198,3838"
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1]))["cells"][0]["token_parity"]; print(t.get("parity"), t.get("first_divergence"))' "$d/receipt.json" 2>/dev/null)
[ "$got" = "True None" ] && ok "identical prompt ids are parity" || broke "identical prompt ids: got '$got'"
d=$(newcase parity_diverges)
apr_out "$d" $P "$T4" gpu false "151644,872,198,27,0,0,0"; llama_out "$d" $P "$Q" "$T4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
tokrow "$d/manifest.jsonl" "$d" $P "151644,872,198,3838,374"
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1]))["cells"][0]["token_parity"]; print(t.get("parity"), t.get("first_divergence"))' "$d/receipt.json" 2>/dev/null)
[ "$got" = "False 3" ] && ok "a divergent prompt is located at its first differing id (3)" || broke "divergent prompt ids: got '$got'"

# 12. the prompt set has not drifted from the golden cases it was derived from.
drift=$(python3 - "$GOLDEN" "$PROMPTS_V1" <<'PY'
import json, re, sys
src = open(sys.argv[1]).read()
# #3724 restructured this. golden_test_cases_for(architecture) now DELEGATES to
# golden_questions() and moves the ChatML templating into golden_prompt_for(), so the
# literals this gate compares live in golden_questions. Reading them there is also more
# honest: the questions are BARE, which is exactly what crux_inference_prompts.json
# stores as messages[-1].content, so no un-templating step can silently mis-parse.
# Anchored on the name; a regex that accepted any function would stop detecting the real
# drift this check exists for.
m = re.search(r"fn golden_questions\([^)]*\)[^{]*\{(.*?)\n\}", src, re.S)
if not m:
    print("golden_questions() not found in %s (did #3724's refactor move again?)" % sys.argv[1]); sys.exit(0)
body = re.sub(r"//[^\n]*", "", m.group(1))
lit = r'"((?:[^"\\]|\\.)*)"'
cases = re.findall(r'\(\s*' + lit + r'\s*,\s*vec!\[(.*?)\]\s*,?\s*\)', body, re.S)
golden = [(prompt, re.findall(lit, pats)) for prompt, pats in cases]
mine = [(p["messages"][-1]["content"], p["expect_any"]) for p in json.load(open(sys.argv[2]))["prompts"] if p["rung"] == "golden"]
if not golden or not mine:
    print("refused: %d golden case(s), %d prompt(s); zero on either side is not agreement" % (len(golden), len(mine)))
elif golden != mine:
    print("DRIFT golden=%r prompts=%r" % (golden, mine))
PY
)
[ -z "$drift" ] && ok "the prompt set matches golden_test_cases (3 cases, same order)" || broke "prompt set vs golden: $drift"


# ── #3832: the cell states its own COVERAGE and PROVENANCE ───────────────────
# A verdict read three weeks later collapses to a colour. These rows keep the
# count, the versions and the reason-an-engine-was-absent attached to it.
cell_field() { # cell_field <case dir> <prompt id> <python expr over `c`>
  python3 -c 'import json,sys
r = json.load(open(sys.argv[1]))
c = next((x for x in r["cells"] if x["key"]["prompt_id"] == sys.argv[2]), None)
print("ABSENT" if c is None else eval(sys.argv[3]))' "$1/receipt.json" "$2" "$3" 2>/dev/null
}
field_is() { # field_is <name> <case dir> <prompt id> <expr> <want>
  local got; got=$(cell_field "$2" "$3" "$4")
  if [ "$got" = "$5" ]; then ok "$1 ($4 = $got)"; else broke "$1: want $5, got $got"; fi
}

# A corroborated GREEN records who corroborated it.
d=$(newcase quorum_recorded)
apr_out "$d" $P "2+2 equals $T4" gpu false; llama_out "$d" $P "$Q" "2 + 2 = $T4"; ollama_out "$d" $P "$T4"; hf_ok "$d" $P
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" ollama $P 0 "$d/ollama-$P.out" "$d/ollama-$P.err"
run_judge "$d"; GOT_RC=$?
field_is "a GREEN cell records the engines that corroborated it" "$d" $P 'c["quorum"]["engines_answered"]' 4
field_is "and records that the floor was met"                    "$d" $P 'c["quorum"]["met"]' True

# apr alone is not an oracle: RED (#3957 Q1), and the cell SAYS it stood alone.
d=$(newcase apr_alone_says_so)
apr_out "$d" $P "2+2 equals $T4" gpu false
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P "" "" "" "zsh:1: command not found: llama-cli"
row "$d/manifest.jsonl" ollama $P "" "" "" "/usr/local/bin/ollama: No such file or directory"
run_judge "$d"; GOT_RC=$?
expect "apr correct but alone is RED, never GREEN" "$d" 1 $P RED
field_is "and the cell records that only one engine answered" "$d" $P 'c["quorum"]["engines_answered"]' 1
field_is "and that the floor was not met"                    "$d" $P 'c["quorum"]["met"]' False

# THE DISTINCTION #3832 EXISTS FOR. lambda's pinned llama-cli WORKS and reports
# `command not found` over ssh because ~/.local/bin is off the non-interactive
# PATH. That is a HARNESS defect. An absent binary is a FLEET fact. One string
# for both would blame the comparator for the harness's mistake.
field_is "a working install the harness could not NAME is not_on_PATH" \
  "$d" $P 'c["engines"]["llama.cpp"]["not_ran_reason"]' not_on_PATH
field_is "an absent binary is binary_not_found_at_path"  \
  "$d" $P 'c["engines"]["ollama"]["not_ran_reason"]' binary_not_found_at_path

# MUTANT: collapse the two reasons to one, as the receipt used to. The row above
# must go RED — otherwise the distinction is decoration.
mut="$TMP/judge-collapsed.py"
sed 's/("command not found", "not_on_PATH")/("command not found", "binary_not_found_at_path")/' "$JUDGE" > "$mut"
if ! cmp -s "$JUDGE" "$mut"; then
  mut_got=$(JUDGE="$mut" PYTHONPATH="$ROOT/scripts/lib" python3 "$mut" collect --manifest "$d/manifest.jsonl" --prompts "$PROMPTS" \
              --meta "$d/meta.json" --out-json "$d/mut.json" --out-md "$d/mut.md" > /dev/null 2>&1;
            python3 -c 'import json,sys
r = json.load(open(sys.argv[1]))
c = next(x for x in r["cells"] if x["key"]["prompt_id"] == sys.argv[2])
print(c["engines"]["llama.cpp"]["not_ran_reason"])' "$d/mut.json" $P 2>/dev/null)
  if [ "$mut_got" = "binary_not_found_at_path" ]; then
    ok "MUTANT collapsing not_on_PATH into binary_not_found_at_path is detected (got $mut_got)"
  else
    broke "MUTANT not detected: collapsing the reasons still reported '$mut_got'"
  fi
else
  broke "MUTANT could not be planted: the not_on_PATH pattern was not found in $JUDGE"
fi

# ── #3957 F6: the quorum-revised oracle. Each row below was a WRONG verdict on the unfixed judge
# (recorded before the fix: a comparator that answered wrong corroborated apr, ALL_WRONG was not
# RED, an answer inside <think> scored, "not Paris" matched "Paris", "Parisian" matched, and
# "NavController" x12 was an ANSWER -- the live #3971 receipt was PASS 4/4 GREEN on it).
META_ENGINES='["apr", "llama.cpp", "ollama", "hf", "llamafile", "vllm"]'
three() { # three <dir> <apr text> <llama text> <hf text> — apr + the same-rep oracle + the bf16 control
  apr_out "$1" $P "$2" gpu false; llama_out "$1" $P "$Q" "$3"
  row "$1/manifest.jsonl" apr $P 0 "$1/apr-$P.out" "$1/apr-$P.err"
  row "$1/manifest.jsonl" llama.cpp $P 0 "$1/llama-$P.out" "$1/llama-$P.err"
  hf_ok "$1" $P "$4"
}
d=$(newcase f6_think_only); three "$d" "<think><answer>4</answer></think>I cannot say" "$T4" "$T4"
run_judge "$d"; GOT_RC=$?
expect "F6: an answer that exists only inside <think> is not an answer (RED)" "$d" 1 $P RED
reason_has "  ...apr has no answer tag once the think block is stripped" "$d" $P "apr is wrong: no_answer_tag"
d=$(newcase f6_unclosed_think); three "$d" "<think>2+2 is, let me see" "$T4" "$T4"
run_judge "$d"; GOT_RC=$?
expect "F6/Q5: an UNCLOSED think block is RED (budget exhausted), never an empty answer" "$d" 1 $P RED
reason_has "  ...named as the budget running out" "$d" $P "unclosed think (budget exhausted)"
d=$(newcase f6_v1_negation)
apr_out "$d" golden-paris "It is not Paris." gpu false; llama_out "$d" golden-paris "What is the capital of France?" "Paris"
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 0 "$d/llama-golden-paris.out" "$d/llama-golden-paris.err"
hf_ok "$d" golden-paris "Paris"
PROMPTS_SAVED=$PROMPTS; PROMPTS=$PROMPTS_V1; run_judge "$d"; GOT_RC=$?; PROMPTS=$PROMPTS_SAVED
expect "F6/Q3: 'It is not Paris.' on a v1 substring prompt is RED, never a match" "$d" 1 golden-paris RED
reason_has "  ...because a v1 expect_any prompt is not a constrained oracle" "$d" golden-paris "v1 prompt"
d=$(newcase f6_comparator_wrong); three "$d" "$T4" "$T5" "$T4"
run_judge "$d"; GOT_RC=$?
expect "F6: a comparator that ANSWERED but WRONG does not corroborate apr (RED)" "$d" 1 $P RED
reason_has "  ...apr differs from the same-representation engine" "$d" $P "apr differs from the ggml family"
d=$(newcase f6_ggml_split); three "$d" "$T4" "$T4" "$T4"
ollama_out "$d" $P "$T5"; row "$d/manifest.jsonl" ollama $P 0 "$d/ollama-$P.out" "$d/ollama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "F6/Q1: llama.cpp and ollama disagreeing on the identical GGUF is a SPLIT: RED" "$d" 1 $P RED
reason_has "  ...named as a same-representation split" "$d" $P "same-representation SPLIT"
# aprender-83's live #3971 shape, verbatim: vLLM on a poisoned HF cache, apr, and no other comparator.
NAV="NavControllerNavControllerNavControllerNavControllerNavControllerNavControllerNavControllerNavControllerNavControllerNavControllerNavControllerNavController"
d=$(newcase f6_navcontroller_live)
apr_out "$d" $P "2 + 2 equals 4." gpu false; row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
engine_out "$d" vllm $P "$NAV" "cuda:0 NVIDIA GeForce RTX 4090"; vllm_row "$d/manifest.jsonl" $P 0 "$d/vllm-$P.json"
run_judge "$d"; GOT_RC=$?
expect "F6/#3971: vLLM answering 'NavController' x12 never makes a cell GREEN" "$d" 1 $P RED
got=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][0]["engines"]["vllm"]["why"][:10])' "$d/receipt.json" 2>/dev/null)
[ "$got" = "degenerate" ] && ok "  ...and the token loop is named DEGENERATE, not an answer" || broke "  ...and the token loop is named DEGENERATE: got '$got'"
d=$(newcase f6_navcontroller_control); three "$d" "$T4" "$T4" "$T4"
: > "$d/hf-$P.json.unused"; python3 -c 'import sys; lines=[l for l in open(sys.argv[1]) if "\"engine\": \"hf\"" not in l]; open(sys.argv[1],"w").writelines(lines)' "$d/manifest.jsonl"
engine_out "$d" vllm $P "$NAV" "cuda:0 NVIDIA GeForce RTX 4090"; vllm_row "$d/manifest.jsonl" $P 0 "$d/vllm-$P.json"
run_judge "$d"; GOT_RC=$?
expect "F6/#3971: apr AND llama.cpp right, the only control a token loop: RED" "$d" 1 $P RED
reason_has "  ...because the ground-truth control FAILED" "$d" $P "ground-truth control FAILED"
d=$(newcase f6_safetensors_green)
apr_out "$d" $P "$T4" gpu false; row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"; hf_ok "$d" $P
MODEL_NAME=fixture-bf16.safetensors run_judge "$d"; GOT_RC=$?
expect "F6/Q2: a SafeTensors cell's same-representation oracle is hf (bf16): GREEN with no ggml" "$d" 0 $P GREEN
d=$(newcase f6_safetensors_ggml_only)
apr_out "$d" $P "$T4" gpu false; row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
llama_out "$d" $P "$Q" "$T4"; row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
MODEL_NAME=fixture-bf16.safetensors run_judge "$d"; GOT_RC=$?
expect "F6/Q2: a SafeTensors cell judged only by ggml has no same-representation oracle: RED" "$d" 1 $P RED
reason_has "  ...no bf16-family engine answered" "$d" $P "no bf16-family engine answered"
d=$(newcase f6_apr_format); three "$d" "$T4" "$T4" "$T4"
MODEL_NAME=fixture-q4k.apr run_judge "$d"; GOT_RC=$?
expect "F6/F8: a .apr cell has no same-representation engine at all: RED here, proven only by the ladder chain" "$d" 1 $P RED
reason_has "  ...named as no engine but apr reading .apr" "$d" $P "no engine but apr reads a apr file"
d=$(newcase f6_control_per_verb); control_green "$d"
for e in apr llama hf; do serve_json "$d" $e golden-paris nonstream "<answer>Paris</answer>"; done
serve_row "$d/manifest.jsonl" apr golden-paris nonstream 0 "$d/apr-golden-paris-nonstream.json"
serve_row "$d/manifest.jsonl" llama.cpp golden-paris nonstream 0 "$d/llama-golden-paris-nonstream.json"
serve_row "$d/manifest.jsonl" hf golden-paris nonstream 0 "$d/hf-golden-paris-nonstream.json"
run_judge "$d"; GOT_RC=$?
expect "F6: a run-verb control does not control the serve lane: declines" "$d" 2 golden-paris GREEN
declined_has "  ...naming the uncontrolled (model/host/verb/thinking)" "$d" "/fixture/serve run/off"
d=$(newcase f6_negative_blind); three "$d" "$T4" "$T4" "$T4"
python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); [p.update(negative="<answer>4</answer>") for p in d["prompts"] if p["id"]=="golden-2plus2"]; json.dump(d, open(sys.argv[2],"w"))' "$PROMPTS" "$d/prompts.json"
# certified for ITS bytes, so the negative control is the ONLY reason this case can decline (#3887)
cert_for "$d/prompts.json" "$d/cert.json"
PROMPTS_SAVED=$PROMPTS; CERT_SAVED=$CERT; PROMPTS="$d/prompts.json"; CERT="$d/cert.json"
run_judge "$d"; GOT_RC=$?; PROMPTS=$PROMPTS_SAVED; CERT=$CERT_SAVED
expect "F6/J3: a planted 'wrong' answer the judge scores GREEN means the lane is blind: declines" "$d" 2 $P GREEN
declined_has "  ...named as the negative control" "$d" "negative control"
d=$(newcase f6_uncertified); three "$d" "$T4" "$T4" "$T4"
CERT_SAVED=$CERT; CERT=""; run_judge "$d"; GOT_RC=$?; CERT=$CERT_SAVED
expect "J2: a v2 prompt set with no certification receipt declines" "$d" 2 $P GREEN
declined_has "  ...named as uncertified" "$d" "not certified"
d=$(newcase f6_cert_stale); three "$d" "$T4" "$T4" "$T4"
python3 -c 'import sys; open(sys.argv[2],"w").write(open(sys.argv[1]).read() + "\n")' "$PROMPTS" "$d/prompts.json"
PROMPTS_SAVED=$PROMPTS; PROMPTS="$d/prompts.json"; run_judge "$d"; GOT_RC=$?; PROMPTS=$PROMPTS_SAVED
expect "J2: one changed byte in the prompt set makes the certification stale: declines" "$d" 2 $P GREEN
declined_has "  ...the certifier's own refusal is quoted" "$d" "an edited prompt set is uncertified"
# ── #3962 joins (aprender-dd, aprender-19): each was a wrong verdict on the judge before this block.
# hf/vLLM split the think block off before writing `text`: an UNCLOSED one arrives as `reasoning` + "".
d=$(newcase j_hf_unclosed_reasoning)
apr_out "$d" $P "$T4" gpu false; llama_out "$d" $P "$Q" "$T4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"; row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
python3 -c 'import json,sys; json.dump({"text": "", "reasoning": "2+2, let me carefully consider the", "reported": {"device": "cuda:0 fixture"}}, open(sys.argv[1], "w"))' "$d/hf-$P.json"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "J: an hf control whose think block never closed is RED" "$d" 1 $P RED
reason_has "  ...read as UNCLOSED (the reasoning field rebuilt), not as a missing tag" "$d" $P "unclosed think"
# aprender-19 R1: each serve ROUTE is its own cell; a wrong route cannot hide behind a right one.
d=$(newcase j_route_is_a_key); control_green "$d"
for rt in "POST /api/chat" "POST /v1/chat/completions"; do
  f="$d/apr-route-${rt//[^a-z]/}.json"
  case $rt in *api*) txt="<answer>5</answer>" ;; *) txt="$T4" ;; esac
  python3 -c 'import json,sys; json.dump({"text": sys.argv[2], "reported": {"device": "fixture"}}, open(sys.argv[1], "w"))' "$f" "$txt"
  python3 -c 'import json,sys; open(sys.argv[1],"a").write(json.dumps({"kind":"gen","engine":"apr","prompt_id":"golden-2plus2","rc":0,"stdout":sys.argv[2],"stderr":None,"refused":None,"model_sha256":"%s","host":"fixture","verb":"serve run","thinking":"off","backend":"gpu","mode":"nonstream","route":sys.argv[3]})+"\n")' "$d/manifest.jsonl" "$f" "$rt"
done
for e in llama hf; do serve_json "$d" $e $P nonstream "$T4"; done
for rt in "POST /api/chat" "POST /v1/chat/completions"; do
  for e in llama.cpp hf; do f="$d/${e%%.*}-$P-nonstream.json"; [ "$e" = llama.cpp ] && f="$d/llama-$P-nonstream.json"
    python3 -c 'import json,sys; open(sys.argv[1],"a").write(json.dumps({"kind":"gen","engine":sys.argv[2],"prompt_id":"golden-2plus2","rc":0,"stdout":sys.argv[3],"stderr":None,"refused":None,"model_sha256":"%s","host":"fixture","verb":"serve run","thinking":"off","backend":"gpu","mode":"nonstream","route":sys.argv[4]})+"\n")' "$d/manifest.jsonl" "$e" "$f" "$rt"
  done
done
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sorted((c["key"].get("route"), c["verdict"]) for c in r["cells"] if c["key"]["verb"]=="serve run"))' "$d/receipt.json" 2>/dev/null)
[ "$GOT_RC" = 1 ] && [ "$got" = "[('POST /api/chat', 'RED'), ('POST /v1/chat/completions', 'GREEN')]" ] && ok "J/R1: a wrong /api/chat is its own RED cell beside a right /v1 route" || broke "J/R1 route key: rc $GOT_RC '$got'"
# aprender-19 R2: a broken wire is named, never read as a missing text field.
d=$(newcase j_protocol_fault); control_green "$d"
python3 -c 'import json,sys; json.dump({"text": None, "protocol_fault": "stream_truncated", "reported": {"device": "fixture"}}, open(sys.argv[1], "w"))' "$d/apr-$P-stream.json"
serve_row "$d/manifest.jsonl" apr $P stream 0 "$d/apr-$P-stream.json"
for e in llama hf; do serve_json "$d" $e $P stream "$T4"; done
serve_row "$d/manifest.jsonl" llama.cpp $P stream 0 "$d/llama-$P-stream.json"; serve_row "$d/manifest.jsonl" hf $P stream 0 "$d/hf-$P-stream.json"
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; c=[x for x in json.load(open(sys.argv[1]))["cells"] if x["key"].get("mode")=="stream"][0]; print(c["verdict"], c["engines"]["apr"]["why"])' "$d/receipt.json" 2>/dev/null)
case "$got" in "RED protocol fault: stream_truncated"*) ok "J/R2: a protocol_fault is named on the apr entry" ;; *) broke "J/R2 protocol fault: '$got'" ;; esac
# dd's admitted_by_sha: a prompt the certification did not admit FOR THIS MODEL is RED, even if right.
d=$(newcase j_not_admitted); control_green "$d"
apr_out "$d" golden-paris "<answer>Paris</answer>" gpu false; llama_out "$d" golden-paris "What is the capital of France?" "<answer>Paris</answer>"
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 0 "$d/llama-golden-paris.out" "$d/llama-golden-paris.err"; hf_ok "$d" golden-paris "<answer>Paris</answer>"
ADMIT=golden-2plus2 cert_for "$PROMPTS" "$d/cert.json"
CERT_SAVED=$CERT; CERT="$d/cert.json"; run_judge "$d"; GOT_RC=$?; CERT=$CERT_SAVED
expect "J2: a right answer to a prompt NOT admitted for this model is RED" "$d" 1 golden-paris RED
reason_has "  ...named as not certified for this model" "$d" golden-paris "not admitted for this model"

# dd 292645efb: admission PER THINKING MODE. A model whose thinking-ON cells loop certifies no prompt
# under the strict key, and its right thinking-OFF cells must not go RED "not certified" for it.
d=$(newcase j_admitted_off_only); control_green "$d"
ADMIT="" ADMIT_THINKING='{"off": ["golden-2plus2"], "on": []}' cert_for "$PROMPTS" "$d/cert.json"
CERT_SAVED=$CERT; CERT="$d/cert.json"; run_judge "$d"; GOT_RC=$?; CERT=$CERT_SAVED
expect "J2/thinking: strict admits nothing, the OFF mode admits -- the right OFF cell is GREEN" "$d" 0 $P GREEN
d=$(newcase j_not_admitted_in_this_mode); control_green "$d"
ADMIT_THINKING='{"off": [], "on": ["golden-2plus2"]}' cert_for "$PROMPTS" "$d/cert.json"
CERT_SAVED=$CERT; CERT="$d/cert.json"; run_judge "$d"; GOT_RC=$?; CERT=$CERT_SAVED
expect "J2/thinking: admitted only for thinking ON -- the OFF cell is RED" "$d" 1 $P RED
reason_has "  ...named as not admitted in this thinking mode" "$d" $P "not admitted for this model"

d=$(newcase f6_all_green); three "$d" "$T4" "$T4" "$T4"
run_judge "$d"; GOT_RC=$?
expect "F6 positive control of this section: apr, ggml and the bf16 control all right is GREEN and PASSES" "$d" 0 $P GREEN
neg=$(python3 -c 'import json,sys; n=json.load(open(sys.argv[1]))["summary"]["negative_controls"]["run"]; print(n["planted"], n["verdict"])' "$d/receipt.json" 2>/dev/null)
[ "$neg" = "<answer>5</answer> RED" ] && ok "  ...and its negative control planted <answer>5</answer> and saw RED" || broke "negative control record: '$neg'"
unset META_ENGINES

# ── #3962 B2: thinking ON with the OFFICIAL template prefills `<think>\n` in the prompt (#3990), so apr's
# reply starts INSIDE the block, and llama-cli prints `[Start thinking] ... [End thinking]`. The judge
# re-attaches the tags; crux_oracles.strip_think decides. Measured: every ON cell of the smoke (apr
# c08437cdd) was RED "answer_not_int" because the REASONING was judged as the answer.
OPEN_RENDER='<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n'
b2() { # b2 <case> <apr reply> <llama answer> [apr render]: one thinking-ON run cell, apr + llama.cpp + hf
  local d; d=$(newcase "$1")
  APR_RENDER="${4:-$OPEN_RENDER}" apr_out "$d" $P "$2" gpu false; llama_out "$d" $P "$Q" "$3"
  ROW_THINKING=on row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
  ROW_THINKING=on row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
  engine_out "$d" hf $P "$T4"; ROW_THINKING=on row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
  printf '%s' "$d"
}
LL_CLOSED="[Start thinking]
Two and two make four.
[End thinking]

$T4"
d=$(b2 b2_prefilled_closed "Two and two make four.
</think>

$T4" "$LL_CLOSED"); run_judge "$d"; GOT_RC=$?
expect "B2: a prefilled block that CLOSES, then the right answer, is GREEN in apr and llama-cli (was RED answer_not_int)" "$d" 0 $P GREEN
d=$(b2 b2_prefilled_unclosed "Thinking: the answer is <answer>4</answer>, let me check again. The answer" "$LL_CLOSED"); run_judge "$d"; GOT_RC=$?
expect "B2: a prefilled block that NEVER closes is RED, even with a right draft inside the reasoning" "$d" 1 $P RED
reason_has "  ...named as an unclosed think block, never judged on the reasoning" "$d" $P "unclosed think"
d=$(b2 b2_llama_unclosed "Two and two make four.
</think>

$T4" "[Start thinking]
The answer is $T4 but let me") ; run_judge "$d"; GOT_RC=$?
expect "B2: llama-cli [Start thinking] with no [End thinking] does not answer (unclosed), so it corroborates nothing" "$d" 1 $P RED
d=$(b2 b2_prompt_unknown "Two and two make <answer>4</answer>" "$LL_CLOSED" "$(printf 'x%.0s' $(seq 1 200))"); run_judge "$d"; GOT_RC=$?
expect "B2: nothing shows whether the prompt opened a block and the reply has no tag: RED, never judged on reasoning" "$d" 1 $P RED
reason_has "  ...named as undecidable" "$d" $P "nothing shows whether the prompt opened a think block"
# aprender-6c [8b6b78]: realizar logs the first 200 BYTES, and {:?} leaves CJK unescaped -- 70 CJK chars
# (210 bytes) print SHORT. Counting the printed chars called that "whole, does not open" and judged the
# cell on its reasoning (FALSE GREEN); the raw byte count makes it unknown.
d=$(b2 b2_cjk_render_cut "Two and two make <answer>4</answer>" "$LL_CLOSED" "$(printf '问%.0s' $(seq 1 70))"); run_judge "$d"; GOT_RC=$?
expect "B2: a CJK rendering cut at 200 bytes (70 chars) is NOT read as whole -- unknown, so RED by name" "$d" 1 $P RED
reason_has "  ...named as undecidable, never judged on the reasoning" "$d" $P "nothing shows whether the prompt opened a think block"

# ── #3957 F6 MUTANTS. Each rule deleted in a copy of the judge; the WHOLE table must then break.
if [ -z "${CRUX_NO_MUTANTS:-}" ]; then
  # label|the row that MUST break (#3887: a kill for the wrong reason is no kill)|sed deleting the rule
  while IFS='|' read -r label must expr; do
    [ -n "$label" ] || continue
    m="$TMP/mut-$label.py"; sed "$expr" "$JUDGE" > "$m"
    if cmp -s "$JUDGE" "$m"; then broke "F6 mutant $label did not apply"; continue; fi
    CRUX_JUDGE_OVERRIDE="$m" CRUX_NO_MUTANTS=1 bash "$ROOT/scripts/check_crux_inference_judge.sh" > "$TMP/mut-$label.log" 2>&1
    nb=$(grep -c '^  BROKE' "$TMP/mut-$label.log")
    if grep -q "^  BROKE.*$must" "$TMP/mut-$label.log"; then ok "F6 mutant $label killed by '$must' ($nb row(s) broke)"
    else broke "F6 mutant $label SURVIVED: '$must' stayed ok ($nb other row(s) broke)"; fi
  done <<'MUT'
bad-control-ignored|the only control a token loop|s/^        if bad:$/        if False:/
no-control-ok|is no control: RED|s/^    if not ctl:$/    if False:/
split-ignored|is a SPLIT|s/^        elif len(set(vals.values())) > 1 or None in vals.values():$/        elif False:/
apr-differs-ignored|ANSWERED but WRONG does not corroborate|s/^        elif a.get("answered") and ext.get("apr") != next(iter(vals.values())):$/        elif False:/
token-loop-off|named DEGENERATE|s/^    return top >= 0.9 \* len(chars) or token_loop(text) is not None$/    return top >= 0.9 * len(chars)/
b2-no-opener|a prefilled block that NEVER closes is RED|s/^            p\["answer"\] = "<think>\\n" + p\["answer"\]$/            pass/
b2-no-llama-map|llama-cli \[Start thinking\] with no \[End thinking\]|s/^        ans = ans.replace(marker, tag)$/        pass/
b2-chars-not-bytes|a CJK rendering cut at 200 bytes|s/^    return False if len(raw.encode("utf-8")) < FORMATTED_PROMPT_LOG_BYTES else None$/    return False if len(rendered) < 180 else None/
b2-unknown-judged|nothing shows whether the prompt opened a block|s/^        elif opened is None and not re.search(r"<\/?think>", p\["answer"\], re.I):$/        elif False:/
negative-control-off|the lane is blind|s/^    blind = sorted(v for v, r in negative.items() if r\["verdict"\] != "RED")$/    blind = []/
per-verb-control-off|does not control the serve lane|s/^    uncontrolled = \["%s/    uncontrolled = [] and ["%s/
reasoning-not-rebuilt|read as UNCLOSED|s/^    if isinstance(doc, dict) and isinstance(doc.get("reasoning"), str) and doc.get("reasoning"):$/    if False:/
route-not-keyed|J\/R1 route key|s/, mode, r.get("route") or "")$/, mode, "")/
admission-mode-off|admitted only for thinking ON|s/^        if admitted_mode is not None:$/        if False:/
admission-off|NOT admitted for this model is RED|s/^        elif admitted is not None and k\[5\] not in admitted.get(k\[0\], ()):$/        elif False:/
certification-off|no certification receipt declines|s/^        certified = certification_ok(args.prompts, getattr(args, "certification", None))$/        certified = True/
MUT
fi

printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
