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
JUDGE="$ROOT/scripts/lib/crux_inference_judge.py"
PROMPTS="$ROOT/scripts/crux_inference_prompts.json"
GOLDEN="$ROOT/crates/apr-cli/src/commands/golden_output.rs"
for f in "$JUDGE" "$PROMPTS" "$GOLDEN"; do
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
ok()   { printf '  ok    %s\n' "$1"; PASS=$((PASS + 1)); }
broke(){ printf '  BROKE %s\n' "$1"; FAIL=$((FAIL + 1)); }

# ---- fixture writers: one engine's raw output, in the shape it really prints ---
SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
apr_out() { # apr_out <dir> <pid> <text> <ran> <fell_back> [ids]
  printf '{"text": %s, "tokens_generated": 3, "tok_per_sec": 9.5, "backend": {"requested": "gpu", "ran": "%s", "fell_back": %s}}\n' \
    "$(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$3")" "$4" "$5" > "$1/apr-$2.out"
  printf '[DEBUG] formatted_prompt="<|im_start|>user\\nq<|im_end|>\\n"\n[DEBUG] add_bos=false, encoded %s tokens: [%s]\n' \
    "$(printf '%s' "${6:-1,2,3}" | tr ',' '\n' | grep -c .)" "${6:-1,2,3}" > "$1/apr-$2.err"
}
llama_out() { # llama_out <dir> <pid> <prompt> <answer> — the pinned chat CLI's stdout
  printf 'build : b10987-d1d3c3396\n\n> %s\n%s\n\n[ Prompt: 282.9 t/s | Generation: 104.9 t/s ]\n\nExiting...\n' "$3" "$4" > "$1/llama-$2.out"
  : > "$1/llama-$2.err"
}
ollama_out() { # ollama_out <dir> <pid> <answer>
  printf '%s\n' "$3" > "$1/ollama-$2.out"
  printf 'total duration:       1.2s\nprompt eval count:    30 token(s)\neval count:           8 token(s)\neval rate:            90.00 tokens/s\n' > "$1/ollama-$2.err"
}
row() { # row <manifest> <engine> <pid> <rc> <stdout> <stderr> [refused]
  python3 - "$@" <<'PY'
import json, os, sys
m, eng, pid, rc, o, e = sys.argv[1:7]
ref = sys.argv[7] if len(sys.argv) > 7 else ""
open(m, "a").write(json.dumps({"kind": "gen", "engine": eng, "prompt_id": pid, "rc": int(rc) if rc else None,
    "stdout": o or None, "stderr": e or None, "refused": ref or None, "model_sha256": "%s",
    "host": "fixture", "verb": os.environ.get("ROW_VERB", "run"), "thinking": "off", "backend": "gpu"}) + "\n")
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
control_green() { # control_green <dir>: the positive control answered right by apr and llama.cpp
  apr_out "$1" golden-2plus2 "4" gpu false; llama_out "$1" golden-2plus2 "What is 2+2?" "4"
  row "$1/manifest.jsonl" apr golden-2plus2 0 "$1/apr-golden-2plus2.out" "$1/apr-golden-2plus2.err"
  row "$1/manifest.jsonl" llama.cpp golden-2plus2 0 "$1/llama-golden-2plus2.out" "$1/llama-golden-2plus2.err"
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
  printf '{"version": "0.0.0", "host": "fixture", "backend": "gpu", "engines": %s}\n' \
    "${META_ENGINES:-[\"apr\", \"llama.cpp\", \"ollama\"]}" > "$1/meta.json"
  seal "$1/manifest.jsonl"
  python3 "$JUDGE" collect --manifest "$1/manifest.jsonl" --prompts "$PROMPTS" --meta "$1/meta.json" \
    --out-json "$1/receipt.json" --out-md "$1/receipt.md" > "$1/judge.out" 2> "$1/judge.err"
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

printf '=== CRUX inference judge case table ===\n'

# 1. every engine right: GREEN, and the run passes.
d=$(newcase all_right)
apr_out "$d" $P "2+2 equals 4." gpu false; llama_out "$d" $P "$Q" "2 + 2 equals 4."; ollama_out "$d" $P "4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" ollama $P 0 "$d/ollama-$P.out" "$d/ollama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "every engine right is GREEN and passes" "$d" 0 $P GREEN

# 2. THE FALSIFIER: llama.cpp right, apr wrong.
d=$(newcase llama_right_apr_wrong)
apr_out "$d" $P "2+2 equals 5." gpu false; llama_out "$d" $P "$Q" "2 + 2 equals 4."
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" ollama $P "" "" "" "not requested here"
run_judge "$d"; GOT_RC=$?
expect "llama.cpp right and apr wrong is RED" "$d" 1 $P RED

# 3. ollama right, apr exited 14 after falling back off the GPU.
d=$(newcase apr_fell_back)
apr_out "$d" $P "4" cpu true; ollama_out "$d" $P "4"
row "$d/manifest.jsonl" apr $P 14 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P "" "" "" "unresolved"
row "$d/manifest.jsonl" ollama $P 0 "$d/ollama-$P.out" "$d/ollama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "apr right text but fell back and exited 14 is RED" "$d" 1 $P RED

# 4. apr ran on another backend with rc 0 and fell_back=false: still not the lane.
d=$(newcase apr_wrong_backend)
apr_out "$d" $P "4" cpu false; llama_out "$d" $P "$Q" "4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "apr on a backend the lane did not ask for is RED" "$d" 1 $P RED

# 4b. apr printed a right-looking answer on the lane's backend and still exited non-zero.
d=$(newcase apr_nonzero_exit)
apr_out "$d" $P "4" gpu false; llama_out "$d" $P "$Q" "4"
row "$d/manifest.jsonl" apr $P 1 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "apr exiting non-zero is not an answer, however right it reads" "$d" 1 $P RED

# 4c. the same on the comparator side: a comparator that exited non-zero cannot vouch.
d=$(newcase llama_nonzero_exit)
apr_out "$d" $P "5" gpu false; llama_out "$d" $P "$Q" "4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 1 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "a comparator exiting non-zero is no oracle (UNJUDGED)" "$d" 2 $P UNJUDGED

# 5. the apr row is MISSING while llama.cpp answered right: absence is a violation.
d=$(newcase apr_missing)
llama_out "$d" $P "$Q" "4"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "a missing apr row where llama.cpp is right is RED" "$d" 1 $P RED

# 6. no comparator answered: UNJUDGED, and a run of only that declines.
d=$(newcase no_oracle)
apr_out "$d" $P "4" gpu false
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P "" "" "" "unresolved: no_binary_named"
row "$d/manifest.jsonl" ollama $P "" "" "" "imported with no chat template"
run_judge "$d"; GOT_RC=$?
expect "no comparator answered is UNJUDGED and declines" "$d" 2 $P UNJUDGED

# 7. everyone answered wrong on the control prompt: ALL_WRONG, and the run declines.
d=$(newcase all_wrong)
apr_out "$d" $P "5" gpu false; llama_out "$d" $P "$Q" "22"; ollama_out "$d" $P "five"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" ollama $P 0 "$d/ollama-$P.out" "$d/ollama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "everyone wrong is ALL_WRONG and declines" "$d" 2 $P ALL_WRONG

# 8. a comparator whose output cannot be parsed did not answer; it cannot vouch.
d=$(newcase llama_unparsed)
apr_out "$d" $P "5" gpu false
printf 'banner only, the turn never finished\n' > "$d/llama-$P.out"; : > "$d/llama-$P.err"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
run_judge "$d"; GOT_RC=$?
expect "an unparseable comparator is no oracle (UNJUDGED)" "$d" 2 $P UNJUDGED

# 9. one RED among GREENs still fails the whole run.
d=$(newcase one_red_among_green)
for p in golden-2plus2 golden-paris; do
  case $p in golden-2plus2) q="$Q"; a_apr="4" ;; golden-paris) q="What is the capital of France?"; a_apr="Lyon" ;; esac
  case $p in golden-2plus2) a_ref="4" ;; golden-paris) a_ref="Paris" ;; esac
  apr_out "$d" $p "$a_apr" gpu false; llama_out "$d" $p "$q" "$a_ref"
  row "$d/manifest.jsonl" apr $p 0 "$d/apr-$p.out" "$d/apr-$p.err"
  row "$d/manifest.jsonl" llama.cpp $p 0 "$d/llama-$p.out" "$d/llama-$p.err"
done
run_judge "$d"; GOT_RC=$?
expect "one RED cell among GREEN ones fails the run" "$d" 1 golden-paris RED

# 9b. one UNJUDGED cell among GREEN ones: that cell was never compared, and the
#     amended scope makes a missing cell a NO-GO, so the run declines.
d=$(newcase one_unjudged_among_green)
apr_out "$d" golden-2plus2 "4" gpu false; llama_out "$d" golden-2plus2 "$Q" "4"
row "$d/manifest.jsonl" apr golden-2plus2 0 "$d/apr-golden-2plus2.out" "$d/apr-golden-2plus2.err"
row "$d/manifest.jsonl" llama.cpp golden-2plus2 0 "$d/llama-golden-2plus2.out" "$d/llama-golden-2plus2.err"
apr_out "$d" golden-paris "Paris" gpu false
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 124 "" "" ""
run_judge "$d"; GOT_RC=$?
expect "one UNJUDGED cell among GREEN ones declines the run" "$d" 2 golden-paris UNJUDGED

# 9c-9e. ALL_WRONG is NAMED, not a violation (the cop's ruling), and the positive
#     control keeps it from becoming the escape.
# 9c. a non-control ALL_WRONG beside a GREEN control passes, and is counted per model.
d=$(newcase all_wrong_named)
apr_out "$d" golden-2plus2 "4" gpu false; llama_out "$d" golden-2plus2 "$Q" "4"
row "$d/manifest.jsonl" apr golden-2plus2 0 "$d/apr-golden-2plus2.out" "$d/apr-golden-2plus2.err"
row "$d/manifest.jsonl" llama.cpp golden-2plus2 0 "$d/llama-golden-2plus2.out" "$d/llama-golden-2plus2.err"
apr_out "$d" golden-paris "Lyon" gpu false; llama_out "$d" golden-paris "What is the capital of France?" "Marseille"
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 0 "$d/llama-golden-paris.out" "$d/llama-golden-paris.err"
run_judge "$d"; GOT_RC=$?
expect "a non-control ALL_WRONG is named and the run still passes" "$d" 0 golden-paris ALL_WRONG
got=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["summary"]["all_wrong_by_model"])' "$d/receipt.json" 2>/dev/null)
[ "$got" = "{'$SHA': 1}" ] && ok "ALL_WRONG is counted per model (1)" || broke "all_wrong_by_model: got '$got'"

# 9d. the positive control comes back ALL_WRONG: a broken harness, so the run declines.
d=$(newcase control_all_wrong)
apr_out "$d" golden-2plus2 "22" gpu false; llama_out "$d" golden-2plus2 "$Q" "22"
row "$d/manifest.jsonl" apr golden-2plus2 0 "$d/apr-golden-2plus2.out" "$d/apr-golden-2plus2.err"
row "$d/manifest.jsonl" llama.cpp golden-2plus2 0 "$d/llama-golden-2plus2.out" "$d/llama-golden-2plus2.err"
apr_out "$d" golden-paris "Paris" gpu false; llama_out "$d" golden-paris "What is the capital of France?" "Paris"
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 0 "$d/llama-golden-paris.out" "$d/llama-golden-paris.err"
run_judge "$d"; GOT_RC=$?
expect "the positive control ALL_WRONG declines the run beside a GREEN cell" "$d" 2 golden-2plus2 ALL_WRONG

# 9e. a prompt set that declares no positive control declines, however green the cells.
d=$(newcase no_control_declared)
python3 -c 'import json,sys
d = json.load(open(sys.argv[1]))
for p in d["prompts"]: p.pop("control", None)
json.dump(d, open(sys.argv[2], "w"))' "$PROMPTS" "$d/prompts.json"
apr_out "$d" $P "4" gpu false; llama_out "$d" $P "$Q" "4"
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
apr_out "$d" golden-paris "Paris" gpu false; llama_out "$d" golden-paris "What is the capital of France?" "Paris"
row "$d/manifest.jsonl" apr golden-paris 0 "$d/apr-golden-paris.out" "$d/apr-golden-paris.err"
row "$d/manifest.jsonl" llama.cpp golden-paris 0 "$d/llama-golden-paris.out" "$d/llama-golden-paris.err"
run_judge "$d"; GOT_RC=$?
expect "a model with no measured control cell declines" "$d" 2 golden-paris GREEN

# ---- the 19:03Z / 19:04Z engines: hf and llamafile (row contract v1) --------------
META_ENGINES='["apr", "llama.cpp", "ollama", "hf", "llamafile"]'

# E1. hf right, apr wrong: RED, exactly as for llama.cpp or ollama.
d=$(newcase hf_right_apr_wrong)
apr_out "$d" $P "5" gpu false; engine_out "$d" hf $P "2 + 2 = 4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf right and apr wrong is RED" "$d" 1 $P RED

# E2. llamafile refuses the model (its own text quoted); llama.cpp answers: still judged, refusal named.
d=$(newcase llamafile_refused)
apr_out "$d" $P "4" gpu false; llama_out "$d" $P "$Q" "4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
row "$d/manifest.jsonl" llamafile $P "" "" "" "error loading model architecture: unknown model architecture: 'qwen35'"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""; engine_out "$d" hf $P "4"
run_judge "$d"; GOT_RC=$?
expect "a llamafile refusal beside a right llama.cpp is GREEN" "$d" 0 $P GREEN
why=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["cells"][0]["engines"]["llamafile"]["why"])' "$d/receipt.json" 2>/dev/null)
case "$why" in *"unknown model architecture: 'qwen35'"*) ok "llamafile's refusal is quoted, not dropped" ;; *) broke "llamafile refusal: '$why'" ;; esac

# E3. an hf answer that is not the contract's JSON is no answer, so it cannot vouch.
d=$(newcase hf_bad_json)
apr_out "$d" $P "5" gpu false; printf '2 + 2 = 4\n' > "$d/hf-$P.json"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf output that is not the contract JSON is no oracle (UNJUDGED)" "$d" 2 $P UNJUDGED

# E3b. valid JSON that is not the contract's shape (no "text") is no answer either.
d=$(newcase hf_json_without_text)
apr_out "$d" $P "5" gpu false; printf '{"answer": "4"}\n' > "$d/hf-$P.json"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf JSON without a text field is no oracle (UNJUDGED)" "$d" 2 $P UNJUDGED

# E3c. a plugin engine is held to its lane: a gpu-lane row that ran on the CPU,
#      or a row with no reported device, cannot vouch.
d=$(newcase hf_wrong_device)
apr_out "$d" $P "5" gpu false; engine_out "$d" hf $P "4" "cpu"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf on the CPU in the gpu lane is no oracle (UNJUDGED)" "$d" 2 $P UNJUDGED
d=$(newcase hf_no_device)
apr_out "$d" $P "5" gpu false; engine_out "$d" hf $P "4" ""
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" hf $P 0 "$d/hf-$P.json" ""
run_judge "$d"; GOT_RC=$?
expect "hf with no reported device is no oracle (UNJUDGED)" "$d" 2 $P UNJUDGED

# E4. a DEGENERATE answer is no answer from any engine: the golden greeting's "!"
#     pattern would otherwise score "!!!!!!!!" (token id 0, x64) as correct.
d=$(newcase degenerate_answers)
apr_out "$d" golden-greeting "!!!!!!!!!!!!!!!!" gpu false; engine_out "$d" hf golden-greeting "!!!!!!!!!!!!!!!!"
llama_out "$d" golden-greeting "Hello there, how are you doing today my friend?" "Hello! I am well."
row "$d/manifest.jsonl" apr golden-greeting 0 "$d/apr-golden-greeting.out" "$d/apr-golden-greeting.err"
row "$d/manifest.jsonl" hf golden-greeting 0 "$d/hf-golden-greeting.json" ""
row "$d/manifest.jsonl" llama.cpp golden-greeting 0 "$d/llama-golden-greeting.out" "$d/llama-golden-greeting.err"
control_green "$d"
run_judge "$d"; GOT_RC=$?
expect "apr's degenerate '!!!!' is no answer: RED where llama.cpp answered" "$d" 1 golden-greeting RED
got=$(python3 -c 'import json,sys; c=[x for x in json.load(open(sys.argv[1]))["cells"] if x["key"]["prompt_id"]=="golden-greeting"][0]; print(c["engines"]["hf"]["answered"], c["engines"]["hf"]["why"][:18])' "$d/receipt.json" 2>/dev/null)
case "$got" in "False degenerate output"*) ok "hf's degenerate '!!!!' cannot vouch either" ;; *) broke "hf degenerate: '$got'" ;; esac

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
CP=chat-arith-2turn
d=$(newcase chat_green); control_green "$d"
apr_chat_out "$d" $CP "2 + 2 equals 4." "4 * 3 equals 12."; chat_json "$d" llama $CP "4" "12"
ROW_VERB=chat row "$d/manifest.jsonl" apr $CP 0 "$d/apr-$CP.out" "$d/apr-$CP.err"
ROW_VERB=chat row "$d/manifest.jsonl" llama.cpp $CP 0 "$d/llama-$CP.json" ""
run_judge "$d"; GOT_RC=$?
expect "a chat whose final turn carries the first turn's answer is GREEN" "$d" 0 $CP GREEN
got=$(python3 -c 'import json,sys; c=[x for x in json.load(open(sys.argv[1]))["cells"] if x["key"]["verb"]=="chat"][0]; a=c["engines"]["apr"]; print(a["turns"], a["backend_verified"], c["token_parity"].get("why"))' "$d/receipt.json" 2>/dev/null)
[ "$got" = "['2 + 2 equals 4.', '4 * 3 equals 12.'] False not measured for the chat verb" ] && ok "every apr turn recorded; backend unverified (#3794); no run-verb token parity" || broke "chat record: '$got'"
d=$(newcase chat_red); control_green "$d"
apr_chat_out "$d" $CP "2 + 2 equals 4." "4 * 3 equals 9."; chat_json "$d" llama $CP "4" "12"
ROW_VERB=chat row "$d/manifest.jsonl" apr $CP 0 "$d/apr-$CP.out" "$d/apr-$CP.err"
ROW_VERB=chat row "$d/manifest.jsonl" llama.cpp $CP 0 "$d/llama-$CP.json" ""
run_judge "$d"; GOT_RC=$?
expect "apr losing the first turn's answer while llama.cpp carries it is RED" "$d" 1 $CP RED
d=$(newcase chat_no_turn); control_green "$d"
printf 'Detected ChatML chat template\nYou: \nGoodbye!\n' > "$d/apr-$CP.out"; : > "$d/apr-$CP.err"; chat_json "$d" llama $CP "4" "12"
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
  serve_json "$d" apr $P $mode "4"; serve_json "$d" llama $P $mode "2 + 2 equals 4."
  serve_row "$d/manifest.jsonl" apr $P $mode 0 "$d/apr-$P-$mode.json"; serve_row "$d/manifest.jsonl" llama.cpp $P $mode 0 "$d/llama-$P-$mode.json"
done
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sorted((c["key"].get("mode"), c["verdict"], c["engines"]["apr"].get("backend_verified")) for c in r["cells"] if c["key"]["verb"]=="serve run"))' "$d/receipt.json" 2>/dev/null)
[ "$GOT_RC" = 0 ] && [ "$got" = "[('nonstream', 'GREEN', False), ('stream', 'GREEN', False)]" ] && ok "serve nonstream and stream are distinct GREEN cells; apr serve's backend unverified" || broke "serve green: rc $GOT_RC '$got'"
d=$(newcase serve_apr_refused); control_green "$d"
serve_json "$d" apr $P nonstream "" "URLError: [Errno 111] Connection refused"; serve_json "$d" llama $P nonstream "2 + 2 equals 4."
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
[ "$GOT_RC" = 2 ] && [ "${got%% *}" = UNJUDGED ] && ok "a tok row HF refused is UNJUDGED and declines" || broke "tok hf refused: rc $GOT_RC, '$got'"

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
unset META_ENGINES

# 10-11. token parity: equal ids agree; a divergence is located, never averaged away.
d=$(newcase parity)
apr_out "$d" $P "4" gpu false "151644,872,198,3838"; llama_out "$d" $P "$Q" "4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
tokrow "$d/manifest.jsonl" "$d" $P "151644,872,198,3838"
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1]))["cells"][0]["token_parity"]; print(t.get("parity"), t.get("first_divergence"))' "$d/receipt.json" 2>/dev/null)
[ "$got" = "True None" ] && ok "identical prompt ids are parity" || broke "identical prompt ids: got '$got'"
d=$(newcase parity_diverges)
apr_out "$d" $P "4" gpu false "151644,872,198,27,0,0,0"; llama_out "$d" $P "$Q" "4"
row "$d/manifest.jsonl" apr $P 0 "$d/apr-$P.out" "$d/apr-$P.err"
row "$d/manifest.jsonl" llama.cpp $P 0 "$d/llama-$P.out" "$d/llama-$P.err"
tokrow "$d/manifest.jsonl" "$d" $P "151644,872,198,3838,374"
run_judge "$d"; GOT_RC=$?
got=$(python3 -c 'import json,sys; t=json.load(open(sys.argv[1]))["cells"][0]["token_parity"]; print(t.get("parity"), t.get("first_divergence"))' "$d/receipt.json" 2>/dev/null)
[ "$got" = "False 3" ] && ok "a divergent prompt is located at its first differing id (3)" || broke "divergent prompt ids: got '$got'"

# 12. the prompt set has not drifted from the golden cases it was derived from.
drift=$(python3 - "$GOLDEN" "$PROMPTS" <<'PY'
import json, re, sys
src = open(sys.argv[1]).read()
m = re.search(r"fn golden_test_cases\(\)[^{]*\{(.*?)\n\}", src, re.S)
if not m:
    print("golden_test_cases() not found in %s" % sys.argv[1]); sys.exit(0)
body = re.sub(r"//[^\n]*", "", m.group(1))
lit = r'"((?:[^"\\]|\\.)*)"'
cases = re.findall(r'\(\s*' + lit + r'\s*,\s*vec!\[(.*?)\]\s*,?\s*\)', body, re.S)
golden = []
for prompt, pats in cases:
    u = re.fullmatch(r"<\|im_start\|>user\\n(.*)<\|im_end\|>\\n<\|im_start\|>assistant\\n", prompt, re.S)
    golden.append((u.group(1) if u else None, re.findall(lit, pats)))
mine = [(p["messages"][-1]["content"], p["expect_any"]) for p in json.load(open(sys.argv[2]))["prompts"] if p["rung"] == "golden"]
if not golden or not mine:
    print("refused: %d golden case(s), %d prompt(s); zero on either side is not agreement" % (len(golden), len(mine)))
elif golden != mine:
    print("DRIFT golden=%r prompts=%r" % (golden, mine))
PY
)
[ -z "$drift" ] && ok "the prompt set matches golden_test_cases (3 cases, same order)" || broke "prompt set vs golden: $drift"

printf '%s: %d ok, %d broke\n' "$PROG" "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
