#!/usr/bin/env bash
# crux_sweep_json_test.sh — the case table of scripts/lib/crux_sweep_json.sh, the bash + jq port of the
# JSON steps of scripts/crux_sweep_shards.sh, which now need no Python. Two checks:
#
#   1. CASES. Each case feeds a function a small synthetic input and compares its output, byte for byte,
#      with the output the replaced python3 snippet gave (written out below, not computed by the port), or
#      expects the refusal the library header lists.
#   2. RATCHET. scripts/crux_sweep_shards.sh calls python3 for the judge only. When the judge is ported,
#      this check moves to zero.
#
#   bash scripts/tests/crux_sweep_json_test.sh              # both checks; exit 1 on any failure
#   bash scripts/tests/crux_sweep_json_test.sh --self-test  # every planted defect must turn a check RED
#
# The self-test plants each defect in a copy of the library (or of the sweep), asserts the plant changed
# the file, and requires the case table to FAIL on it. A plant that changes nothing, or a case table that
# stays green, fails the self-test.
#
# Not wired yet. It lives under scripts/tests/ rather than scripts/check_*.sh because guard_tree.sh runs
# every scripts/check_*.sh in the required CI job, and a new check blocks only after it has been run green
# by hand for three nights. Wiring it is its own change.
set -euo pipefail
cd "$(dirname "$0")/../.."
PROG=crux_sweep_json_test
LIB=scripts/lib/crux_sweep_json.sh
SWEEP=scripts/crux_sweep_shards.sh
case "${1:-}" in
  "" | --self-test) ;;
  -h | --help) printf 'usage: bash scripts/tests/%s.sh [--self-test]\n' "$PROG"; exit 0 ;;
  *) printf '%s: usage: bash scripts/tests/%s.sh [--self-test]\n' "$PROG" "$PROG" >&2; exit 2 ;;
esac
command -v jq > /dev/null || { printf '%s: FAIL: jq not found; nothing was measured\n' "$PROG" >&2; exit 1; }
T=$(mktemp -d)
trap 'rm -rf -- "${T:?}"' EXIT

# ---------------------------------------------------------------------------------------------------------
# Fixtures (synthetic; built once, read-only to the cases).
FX="$T/fx"
mkdir -p "$FX/m1" "$FX/m2" "$FX/m3/d.gguf" "$FX/m4" "$FX/W1" "$FX/W2" "$FX/W3" "$FX/W4" "$FX/W5" "$FX/WG"
printf 'model-a\n' > "$FX/m1/a.gguf"
printf 'model-a\n' > "$FX/m1/z.gguf"        # same bytes as a.gguf: the first name in order wins
printf 'model-b\n' > "$FX/m1/b.gguf"
printf 'model-h\n' > "$FX/m1/.h.gguf"       # a dotfile is a model too (os.listdir lists it)
printf 'model-c\n' > "$FX/m1/notes.txt"     # not a .gguf: never hashed
printf 'model-a\n' > "$FX/m2/a2.gguf"       # same bytes again, later dir: m1's path wins
printf 'model-c\n' > "$FX/m2/c.gguf"
printf 'model-t\n' > "$FX/m4/t"$'\t'"ab.gguf"
sha() { sha256sum < "$1" | cut -c1-64; }
noeol() { local s; s=$(cat); printf '%s' "$s"; }   # a heredoc without its final newline, as json.dump wrote
A=$(sha "$FX/m1/a.gguf"); B=$(sha "$FX/m1/b.gguf"); H=$(sha "$FX/m1/.h.gguf"); C=$(sha "$FX/m2/c.gguf")
TB=$(sha "$FX/m4/t"$'\t'"ab.gguf")
Z0=0000000000000000000000000000000000000000000000000000000000000000
ZF=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
# Controls: p1 (true) and p4 ("yes"). 0, "" and [] are false, as in Python.
printf '%s\n' '{"prompts": [{"id": "p1", "control": true}, {"id": "p2"}, {"id": "p3", "control": 0},
  {"id": "p4", "control": "yes"}, {"id": "p5", "control": ""}, {"id": "p6", "control": []}]}' > "$FX/prompts.json"
printf '%s\n' '{"prompts": [{"id": "q0", "control": 0}, {"id": "q1", "control": ""}, {"id": "q2", "control": "yes"}]}' \
  > "$FX/prompts-b.json"
printf '%s\n' '{"prompts": [{"id": "r0"}, {"id": "r1", "control": false}]}' > "$FX/prompts-none.json"
printf '%s\n' '{"prompts": [{"id": "p1", "control": true}]}' '{"prompts": [{"id": "p9", "control": true}]}' > "$FX/prompts-two.json"
PSHA=$(sha "$FX/prompts.json")
adm() { # the admitted table, every certified sha in an unsorted order
  printf '{"%s": {"off": ["p1"]}, "%s": {"off": ["p1", "p2"], "on": ["p2"]}, "%s": {"on": ["p4", "p3", "p1"]},
    "%s": {"off": []}, "%s": {"off": ["p3"], "on": ["p1"]}, "%s": {"off": ["p1"]}}' "$ZF" "$A" "$B" "$H" "$C" "$Z0"
}
printf '{"schema": "s", "prompts_sha256": "%s", "admitted_by_sha_thinking": %s}\n' "$PSHA" "$(adm)" > "$FX/cert.json"
printf '{"prompts_sha256": "%s", "admitted_by_sha_thinking": %s}\n' \
  abcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcabcd "$(adm)" > "$FX/cert-mismatch.json"
printf '{"prompts_sha256": "", "admitted_by_sha_thinking": %s}\n' "$(adm)" > "$FX/cert-unbound.json"
printf '{"admitted_by_sha_thinking": {"%s": {"off": ["p1", 5]}}}\n' "$A" > "$FX/cert-nonstring.json"
printf '{"admitted_by_sha_thinking": {"%s": {"off": ["p1"]}}}\n' "$TB" > "$FX/cert-tab.json"

# The expected plans. Lines are written grouped per sha (off, then on), and sorted by sha with a stable sort.
DIRS="$FX/m1/ $FX/m2 $FX/nope"
plan_lines() { # <scope>
  local t=$'\t'
  if [ "$1" = controls ]; then
    printf '%s\n' "RUN${t}$A${t}off${t}$FX/m1/a.gguf${t}p1" \
      "SKIP${t}$A${t}on${t}$FX/m1/a.gguf${t}no CONTROL admitted (scope=controls)" \
      "SKIP${t}$B${t}off${t}$FX/m1/b.gguf${t}nothing admitted" "RUN${t}$B${t}on${t}$FX/m1/b.gguf${t}p4,p1" \
      "SKIP${t}$C${t}off${t}$FX/m2/c.gguf${t}no CONTROL admitted (scope=controls)" "RUN${t}$C${t}on${t}$FX/m2/c.gguf${t}p1"
  else
    printf '%s\n' "RUN${t}$A${t}off${t}$FX/m1/a.gguf${t}p1,p2" "RUN${t}$A${t}on${t}$FX/m1/a.gguf${t}p2" \
      "SKIP${t}$B${t}off${t}$FX/m1/b.gguf${t}nothing admitted" "RUN${t}$B${t}on${t}$FX/m1/b.gguf${t}p4,p3,p1" \
      "RUN${t}$C${t}off${t}$FX/m2/c.gguf${t}p3" "RUN${t}$C${t}on${t}$FX/m2/c.gguf${t}p1"
  fi
  printf '%s\n' "SKIP${t}$H${t}off${t}$FX/m1/.h.gguf${t}nothing admitted" "SKIP${t}$H${t}on${t}$FX/m1/.h.gguf${t}nothing admitted" \
    "ABSENT${t}$Z0${t}-${t}-${t}certified, but no file with this sha256 in $DIRS" \
    "ABSENT${t}$ZF${t}-${t}-${t}certified, but no file with this sha256 in $DIRS"
}
plan_lines controls | LC_ALL=C sort -s -t $'\t' -k2,2 > "$FX/plan-controls.expected"
plan_lines admitted | LC_ALL=C sort -s -t $'\t' -k2,2 > "$FX/plan-admitted.expected"
# A models dir named like an option: the ABSENT reason names it, and nothing reads it as one.
DIRS="$FX/m1/ $FX/m2 -nodir" plan_lines controls | LC_ALL=C sort -s -t $'\t' -k2,2 > "$FX/plan-dash.expected"

# Greedy rows: copied verbatim (spacing, exponents, NaN, escapes), the unterminated last row gets a newline.
printf '%s\n' '{"kind":"gen","id":"a"}' '{"kind": "greedy",  "x": 1e-05, "u": "é"}' \
  '{"kind":"greedy","x":NaN}' '{"kind":"tok"}' '{"id":"no-kind"}' > "$FX/manifest.jsonl"
printf '%s' '{"kind":"greedy","last":true}' >> "$FX/manifest.jsonl"
printf '%s\n' '{"kind": "greedy",  "x": 1e-05, "u": "é"}' '{"kind":"greedy","x":NaN}' \
  '{"kind":"greedy","last":true}' > "$FX/greedy.expected"
printf '%s\n' '{"kind":"greedy","n":1}' '[1]' '{"kind":"greedy","n":2}' > "$FX/manifest-bad.jsonl"
printf '%s\n' '{"kind":"greedy","n":1}' > "$FX/greedy-bad.expected"
# CR LF and a final lone CR end a row (Python's text mode); a lone CR between two rows is refused.
printf '{"kind":"greedy","a":1}\r\n{"kind":"gen"}\r\n{"kind":"greedy","b":2}\r' > "$FX/manifest-crlf.jsonl"
printf '%s\n' '{"kind":"greedy","a":1}' '{"kind":"greedy","b":2}' > "$FX/greedy-crlf.expected"
printf '{"kind":"greedy","a":1}\r{"kind":"greedy","b":2}\n' > "$FX/manifest-lone-cr.jsonl"

# Shard metas and the shards.tsv that names them.
printf '%s\n' '{"host": "syn", "models": [{"sha256": "aa", "format": "gguf"}, {"sha256": "bb"}], "x": "é", "f": 1.5}' \
  > "$FX/W1/meta.json"
printf '%s\n' '{"models": [{"sha256": "bb", "format": "other"}, {"sha256": "cc"}]}' > "$FX/W2/meta.json"
printf '%s\n' '{"models": [{"name": "no-sha"}, {"name": "no-sha-2"}], "host": "other"}' > "$FX/W3/meta.json"
printf '%s\n' '{"models": [{"sha256": "nn", "score": NaN}]}' > "$FX/W5/meta.json"
printf '%s\n' '{"models": [{"sha256": "gg"}]}' > "$FX/WG/meta.json"
printf '%s\n' '{"host": "h"}' > "$FX/meta-bare.json"
mkdir -p "$FX/W6" "$FX/W7"; : > "$FX/W6/meta.json"   # an empty meta (a shard killed mid-write)
printf '%s\n' '{"models": []}' '{"models": []}' > "$FX/W7/meta.json"
mkdir -p "$FX/W8"   # numbers jq writes in another form than Python did (0.00001, 1E+5, -0): same doubles
printf '%s\n' '{"x": 1e-05, "y": 1e5, "z": -0, "models": [{"sha256": "ee", "e": 0.0025}]}' > "$FX/W8/meta.json"
t=$'\t'
printf '%s\n' "s1${t}0${t}$FX/W1" "greedy-g${t}0${t}$FX/WG" "s2${t}1${t}$FX/W2" "s3${t}0${t}" "s4${t}0${t}$FX/W4" \
  "s5${t}0${t}$FX/W3" "s6" > "$FX/shards.tsv"
printf '%s\n' "greedy-g${t}0${t}$FX/WG" > "$FX/shards-greedy.tsv"
printf '%s\n' "s1${t}0${t}$FX/W1" "s5${t}0${t}$FX/W5" > "$FX/shards-nan.tsv"
printf '%s\n' "s8${t}0${t}$FX/W8" > "$FX/shards-num.tsv"
printf '%s\n' "s1${t}0${t}$FX/W1" "s6${t}0${t}$FX/W6" > "$FX/shards-empty.tsv"
printf '%s\n' "s1${t}0${t}$FX/W1" "s7${t}0${t}$FX/W7" > "$FX/shards-two.tsv"
noeol > "$FX/meta.expected" <<'JSON'
{
  "host": "syn",
  "models": [
    {
      "sha256": "aa",
      "format": "gguf"
    },
    {
      "sha256": "bb"
    },
    {
      "sha256": "cc"
    },
    {
      "name": "no-sha"
    }
  ],
  "x": "\u00e9",
  "f": 1.5,
  "merged_shards": 7
}
JSON
noeol > "$FX/meta-bare.expected" <<'JSON'
{
  "host": "h",
  "merged_shards": 1
}
JSON

# Judge receipts for the greedy-only mark.
printf '%s\n' '{"schema": "x", "cells": [], "greedy_only": false, "greedy": [{"d": 1}],
  "summary": {"verdict": "DECLINE"}, "u": "é"}' > "$FX/r-ok.json"
noeol > "$FX/r-ok.expected" <<'JSON'
{
 "schema": "x",
 "cells": [],
 "greedy_only": true,
 "greedy": [
  {
   "d": 1
  }
 ],
 "summary": {
  "verdict": "DECLINE"
 },
 "u": "\u00e9"
}
JSON
printf '%s\n' '{"greedy": [1, 2], "summary": {}}' > "$FX/r-noverdict.json"
noeol > "$FX/r-noverdict.expected" <<'JSON'
{
 "greedy": [
  1,
  2
 ],
 "summary": {},
 "greedy_only": true,
 "cells": []
}
JSON
printf '%s\n' '{"cells": [{"c": 1}, {"c": 2}], "greedy": [1], "summary": {}}' > "$FX/r-cells.json"
printf '%s\n' '{"cells": [], "greedy": [], "summary": {}}' > "$FX/r-nogreedy.json"
printf '%s\n' '{"cells": [], "greedy": [{"cos": NaN}], "summary": {}}' > "$FX/r-nan.json"
printf '%s\n' '{"greedy": [1], "summary": {}}' '{"greedy": [2], "summary": {}}' > "$FX/r-two.json"

# ---------------------------------------------------------------------------------------------------------
# The case table. run_cases <lib> prints one FAIL line per failed case and exits with the failure count.
run_cases() (
  # shellcheck source=scripts/lib/crux_sweep_json.sh
  . "$1"
  W=$(mktemp -d "$T/run.XXXXXX")
  n=0
  bad() { printf 'FAIL %s: %s\n' "$1" "$2"; n=$((n + 1)); }
  same() { cmp -s "$2" "$3" || bad "$1" "output differs from $(basename "$2"): $(cmp "$2" "$3" 2>&1 | head -n 1)"; }
  ok() { [ "$2" = 0 ] || bad "$1" "rc $2, expected 0: $(head -c 300 "$W/err")"; }
  refused() { [ "$2" != 0 ] || bad "$1" "rc 0, expected a refusal"; }

  # crux_sweep_plan
  rc=0; crux_sweep_plan "$FX/cert.json" "$FX/prompts.json" controls "$W/plan" "$FX/m1/" "$FX/m2" "$FX/nope" \
    > "$W/out" 2> "$W/err" || rc=$?
  ok plan-controls "$rc"; same plan-controls "$FX/plan-controls.expected" "$W/plan"
  rc=0; crux_sweep_plan "$FX/cert.json" "$FX/prompts.json" admitted "$W/plan" "$FX/m1/" "$FX/m2" "$FX/nope" \
    > "$W/out" 2> "$W/err" || rc=$?
  ok plan-admitted "$rc"; same plan-admitted "$FX/plan-admitted.expected" "$W/plan"
  rc=0; crux_sweep_plan "$FX/cert.json" "$FX/prompts.json" controls "$W/plan-dash" "$FX/m1/" "$FX/m2" -nodir \
    > "$W/out" 2> "$W/err" || rc=$?
  ok plan-dash-dir "$rc"; same plan-dash-dir "$FX/plan-dash.expected" "$W/plan-dash"
  rc=0; crux_sweep_plan "$FX/cert-mismatch.json" "$FX/prompts.json" controls "$W/plan-mm" "$FX/m1" \
    > "$W/out" 2> "$W/err" || rc=$?
  refused plan-prompts-not-bound "$rc"
  printf 'the certification binds prompts sha abcabcabcabc, but %s is %s\n' "$FX/prompts.json" "${PSHA:0:12}" > "$W/mm.expected"
  same plan-prompts-not-bound "$W/mm.expected" "$W/err"
  [ ! -e "$W/plan-mm" ] || bad plan-prompts-not-bound "a plan was written"
  rc=0; crux_sweep_plan "$FX/cert-unbound.json" "$FX/prompts-b.json" controls "$W/plan-ub" "$FX/m1" \
    > "$W/out" 2> "$W/err" || rc=$?
  ok plan-unbound-prompts "$rc"
  rc=0; crux_sweep_plan "$FX/cert.json" "$FX/prompts.json" controls "$W/plan-dir" "$FX/m3" > "$W/out" 2> "$W/err" || rc=$?
  refused plan-gguf-is-a-directory "$rc"
  rc=0; crux_sweep_plan "$FX/cert-nonstring.json" "$FX/prompts.json" admitted "$W/plan-ns" "$FX/m1" \
    > "$W/out" 2> "$W/err" || rc=$?
  refused plan-nonstring-id "$rc"
  rc=0; crux_sweep_plan "$FX/cert-tab.json" "$FX/prompts.json" admitted "$W/plan-tab" "$FX/m4" \
    > "$W/out" 2> "$W/err" || rc=$?
  refused plan-path-with-tab "$rc"

  # crux_first_control
  [ "$(crux_first_control "$FX/prompts.json" 2> /dev/null)" = p1 ] || bad first-control "not p1"
  [ "$(crux_first_control "$FX/prompts-b.json" 2> /dev/null)" = q2 ] || bad first-control-python-truth "not q2"
  rc=0; out=$(crux_first_control "$FX/prompts-none.json" 2> /dev/null) || rc=$?
  refused first-control-none "$rc"; [ -z "$out" ] || bad first-control-none "printed '$out'"
  rc=0; out=$(crux_first_control "$FX/prompts-two.json" 2> /dev/null) || rc=$?
  refused first-control-two-values "$rc"

  # crux_greedy_rows
  rc=0; crux_greedy_rows "$FX/manifest.jsonl" > "$W/greedy" 2> "$W/err" || rc=$?
  ok greedy-rows "$rc"; same greedy-rows "$FX/greedy.expected" "$W/greedy"
  rc=0; crux_greedy_rows "$FX/manifest-bad.jsonl" > "$W/greedy-bad" 2> "$W/err" || rc=$?
  refused greedy-rows-stop-at-bad-row "$rc"; same greedy-rows-stop-at-bad-row "$FX/greedy-bad.expected" "$W/greedy-bad"
  rc=0; crux_greedy_rows "$FX/manifest-crlf.jsonl" > "$W/greedy-crlf" 2> "$W/err" || rc=$?
  ok greedy-rows-crlf "$rc"; same greedy-rows-crlf "$FX/greedy-crlf.expected" "$W/greedy-crlf"
  rc=0; crux_greedy_rows "$FX/manifest-lone-cr.jsonl" > "$W/greedy-lone-cr" 2> "$W/err" || rc=$?
  refused greedy-rows-lone-cr "$rc"

  # crux_merge_meta
  rc=0; crux_merge_meta "$FX/W1/meta.json" "$W/meta" "$FX/shards.tsv" > "$W/out" 2> "$W/err" || rc=$?
  ok merge-meta "$rc"; same merge-meta "$FX/meta.expected" "$W/meta"
  printf 'merged meta: 4 model(s)\n' > "$W/out.expected"; same merge-meta-stdout "$W/out.expected" "$W/out"
  rc=0; crux_merge_meta "$FX/meta-bare.json" "$W/meta-bare" "$FX/shards-greedy.tsv" > "$W/out" 2> "$W/err" || rc=$?
  ok merge-meta-no-models "$rc"; same merge-meta-no-models "$FX/meta-bare.expected" "$W/meta-bare"
  rc=0; crux_merge_meta "$FX/W1/meta.json" "$W/meta-nan" "$FX/shards-nan.tsv" > "$W/out" 2> "$W/err" || rc=$?
  refused merge-meta-nan "$rc"
  rc=0; crux_merge_meta "$FX/W8/meta.json" "$W/meta-num" "$FX/shards-num.tsv" > "$W/out" 2> "$W/err" || rc=$?
  ok merge-meta-number-values "$rc"
  jq -e '.x == 1e-05 and .y == 100000 and .z == 0 and .models == [{sha256: "ee", e: 0.0025}] and .merged_shards == 1' \
    "$W/meta-num" > /dev/null 2>&1 || bad merge-meta-number-values "a number changed its value"
  for t in shards-empty shards-two; do
    rc=0; crux_merge_meta "$FX/W1/meta.json" "$W/meta-$t" "$FX/$t.tsv" > "$W/out" 2> "$W/err" || rc=$?
    refused "merge-meta-$t" "$rc"
  done

  # crux_mark_greedy_only (each on its own copy: the mark rewrites in place)
  for r in r-ok r-noverdict r-cells r-nogreedy r-nan; do cp "$FX/$r.json" "$W/$r.json"; done
  rc=0; crux_mark_greedy_only "$W/r-ok.json" > "$W/out" 2> "$W/err" || rc=$?
  ok mark "$rc"; same mark "$FX/r-ok.expected" "$W/r-ok.json"
  printf 'greedy_only receipt: 1 greedy entries, verdict DECLINE\n' > "$W/out.expected"; same mark-stdout "$W/out.expected" "$W/out"
  rc=0; crux_mark_greedy_only "$W/r-noverdict.json" > "$W/out" 2> "$W/err" || rc=$?
  ok mark-no-verdict "$rc"; same mark-no-verdict "$FX/r-noverdict.expected" "$W/r-noverdict.json"
  printf 'greedy_only receipt: 2 greedy entries, verdict None\n' > "$W/out.expected"; same mark-no-verdict-stdout "$W/out.expected" "$W/out"
  for r in r-cells r-nogreedy r-nan; do
    rc=0; crux_mark_greedy_only "$W/$r.json" > "$W/out" 2> "$W/err" || rc=$?
    refused "mark-refuses-$r" "$rc"; same "mark-refuses-$r-unchanged" "$FX/$r.json" "$W/$r.json"
  done
  rc=0; crux_mark_greedy_only "$W/r-cells.json" > "$W/out" 2> "$W/err" || rc=$?
  grep -q 'a greedy-only merge produced 2 judged cells; it must produce none' "$W/err" || bad mark-cells-message "not the cell count"
  cp "$FX/r-two.json" "$W/r-two.json"; rc=0; crux_mark_greedy_only "$W/r-two.json" > "$W/out" 2> "$W/err" || rc=$?
  refused mark-two-values "$rc"; same mark-two-values-unchanged "$FX/r-two.json" "$W/r-two.json"
  exit "$n"
)

# ---------------------------------------------------------------------------------------------------------
# The ratchet: the python lines of a sweep script, comments excluded, must be exactly the judge call.
python_lines() { grep -nE 'python|\.py([^[:alnum:]_]|$)' "$1" | grep -vE '^[0-9]+:[[:space:]]*#' || true; }
only_judge_python() { # <sweep script>
  local lines
  lines=$(python_lines "$1")
  [ -n "$lines" ] && [ "$(printf '%s\n' "$lines" | wc -l)" -eq 1 ] \
    && printf '%s\n' "$lines" | grep -qE '^[0-9]+:python3 scripts/lib/crux_inference_judge\.py collect '
}

if [ "${1:-}" != --self-test ]; then
  fails=0
  run_cases "$LIB" || fails=$?
  if only_judge_python "$SWEEP"; then
    printf 'PASS ratchet: %s calls python3 for the judge only\n' "$SWEEP"
  else
    printf 'FAIL ratchet: %s has a python line other than the judge call:\n' "$SWEEP"; python_lines "$SWEEP" | sed 's/^/  /'
    fails=$((fails + 1))
  fi
  if [ "$fails" -gt 0 ]; then printf '%s: FAIL (%s)\n' "$PROG" "$fails"; exit 1; fi
  printf '%s: PASS — case table green, sweep python3 = judge only\n' "$PROG"
  exit 0
fi

# ---------------------------------------------------------------------------------------------------------
# --self-test: the unplanted library is green; every plant below must turn the case table RED.
st=0
if run_cases "$LIB" > "$T/base.out"; then printf 'ok   unplanted library: case table green\n'
else printf 'FAIL unplanted library is not green:\n'; sed 's/^/  /' "$T/base.out"; st=1; fi
plant() { # <name> <case> <perl substitution> — on a copy of the library; <case> must go RED with a wrong
  # answer (a crash of a case that expects rc 0 is a RED that proves nothing about the planted defect)
  local m="$T/lib.$1.sh" out rc=0
  perl -pe "$3" "$LIB" > "$m"
  if cmp -s "$LIB" "$m"; then printf 'FAIL plant %s changed nothing (the substitution no longer matches)\n' "$1"; st=1; return; fi
  out=$(run_cases "$m") || rc=$?
  if [ "$rc" = 0 ]; then printf 'FAIL plant %s SURVIVED: the case table stayed green\n' "$1"; st=1
  elif ! printf '%s\n' "$out" | grep -q "^FAIL $2:"; then printf 'FAIL plant %s: RED, but not on case %s\n' "$1" "$2"; st=1
  elif printf '%s\n' "$out" | grep -qE "^FAIL $2: rc [1-9][0-9]*, expected 0"; then
    printf 'FAIL plant %s: case %s crashed instead of answering wrong\n' "$1" "$2"; st=1
  else printf 'ok   plant %s: %s RED\n' "$1" "$2"; fi
}
plant jq-truthiness       plan-controls 's/def pytrue: .*;/def pytrue: . != null and . != false;/'
plant unsorted-shas       plan-controls 's/keys\[\] as \$sha/keys_unsorted[] as \$sha/'
plant mode-order          plan-controls 's/\("off", "on"\) as \$mode/("on", "off") as \$mode/'
plant scope-ignored       plan-controls 's/select\(\$scope == "admitted" or/select(\$scope != "admitted" or/'
plant skip-reason-swapped plan-controls 's/then "no CONTROL admitted \(scope=controls\)" else "nothing admitted" end/then "nothing admitted" else "no CONTROL admitted (scope=controls)" end/'
plant last-path-wins      plan-controls 's/if has\(\$s\) then \. else \.\[\$s\] = \$p end/.[\$s] = \$p/'
plant dotfiles-skipped    plan-controls 's/-name \x27\*\.gguf\x27/-name \x27[!.]*.gguf\x27/'
plant double-slash-join   plan-controls 's/p="\$\{d%\/\}\/\$f"/p="\$d\/\$f"/'
plant prompts-unbound     plan-prompts-not-bound 's/if \[ "\$bound" != "\$actual" \]; then/if false; then/'
plant greedy-loosened     greedy-rows 's/then \.kind == "greedy" else/then .kind != "gen" else/'
plant greedy-past-bad-row greedy-rows-stop-at-bad-row 's/jq -nRr \x27inputs$/jq -Rr \x27./'
plant merge-takes-greedy  merge-meta 's/select\(\(\.\[0\] \| startswith\("greedy-"\) \| not\) and/select(true and/'
plant merge-no-dedup      merge-meta 's/if any\(\.seen\[\]; \. == \$s\) then \. else \.seen \+= \[\$s\] \| \.models \+= \[\$x\] end/.models += [\$x]/'
plant merge-count-short   merge-meta 's/n=\$\(jq -nR \x27\[inputs\] \| length\x27/n=\$(jq -nR \x27[inputs | select(contains("\\t"))] | length\x27/'
plant merge-not-ascii     merge-meta 's/jq -nj -a --indent 2/jq -nj --indent 2/'
plant merge-final-newline merge-meta 's/jq -nj -a --indent 2/jq -n -a --indent 2/'
plant mark-indent         mark 's/jq -nj -a --indent 1/jq -nj -a --indent 2/'
plant mark-takes-cells    mark-refuses-r-cells 's/elif \.cells \| pytrue then error/elif false then error/'
plant mark-empty-greedy   mark-refuses-r-nogreedy 's/elif \.greedy \| pytrue \| not then error/elif false then error/'
plant nonfinite-written   merge-meta-nan 's/select\(isnan or isinfinite\)/select(false)/'
plant greedy-crlf-kept    greedy-rows-crlf 's/\| rtrimstr\("\\r"\)/| ./'
plant control-jq-truth    first-control-python-truth 's/select\(\.control \| pytrue\)\) \/\/ error/select(.control != null)) \/\/ error/'
plant one-value-loosened   mark-two-values 's/def one\(\$what\): if length == 1 then/def one(\$what): if length >= 1 then/'
plant meta-count-skipped  merge-meta-shards-empty 's/if \[ "\$k" != 1 \]; then/if false; then/'

# The ratchet's own table: must go RED on a second python call or a changed judge call, and stay green on
# a comment that names python.
ratchet() { # <name> <want: green|red> <perl substitution> — on a copy of the sweep
  local s="$T/sweep.$1.sh" got=red
  perl -pe "$3" "$SWEEP" > "$s"
  if cmp -s "$SWEEP" "$s"; then printf 'FAIL ratchet plant %s changed nothing\n' "$1"; st=1; return; fi
  only_judge_python "$s" && got=green
  if [ "$got" = "$2" ]; then printf 'ok   ratchet %s: %s\n' "$1" "$got"
  else printf 'FAIL ratchet %s: %s, expected %s\n' "$1" "$got" "$2"; st=1; fi
}
if only_judge_python "$SWEEP"; then printf 'ok   ratchet on the real sweep: green\n'
else printf 'FAIL ratchet on the real sweep is not green\n'; st=1; fi
ratchet second-python3   red   's/^(CTL=)/X=\$(python3 -c 1)\n$1/'
ratchet bare-python      red   's/^(CTL=)/python -c 1\n$1/'
ratchet env-python       red   's/^(CTL=)/\/usr\/bin\/env python3 x.py\n$1/'
ratchet a-py-script      red   's/^(CTL=)/bash -c "x scripts\/lib\/y.py"\n$1/'
ratchet judge-changed    red   's/^python3 scripts\/lib\/crux_inference_judge\.py collect /python3 -m crux_judge collect /'
ratchet judge-removed    red   's/^python3 scripts\/lib\/crux_inference_judge\.py collect /: scripts\/lib\/crux_judge collect /'
ratchet comment-python   green 's/^(CTL=)/# python3 is named in a comment only\n$1/'
ratchet indented-comment green 's/^(CTL=)/  # see crux_inference_judge.py\n$1/'
if [ "$st" = 0 ]; then printf '%s --self-test: PASS\n' "$PROG"; else printf '%s --self-test: FAIL\n' "$PROG"; fi
exit "$st"
