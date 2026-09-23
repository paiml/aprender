#!/usr/bin/env bash
# check_release_shift_left.sh -- the case table for #4045 M8 (shift-left), every mutant killed by its NAMED row.
#   1. the class registry covers exactly the derived publish-blocking gate set (live, on this tree);
#   2. scripts/lib/release_gate_classes_cases.py (derivation + refusals, with mutants);
#   3. scripts/check_no_shadowed_repo_skill.sh --self-test, plus its mutants;
#   4. scripts/release/candidate_watch.sh on THIS checkout with its seams: a bookkeeping red never blocks, a real red
#      andons, an unclassified gate is real, first-red-at persists, a stale receipt is red, a shadow refuses.
set -uo pipefail
case "${1:-}" in -h|--help) echo "usage: bash scripts/check_release_shift_left.sh   (the #4045 M8 case table; no arguments)"; exit 0 ;; esac
cd "$(dirname "$0")/.." || exit 2
bad=0
T=$(mktemp -d) || exit 2
MUTS=()
cleanup() {
  for f in "${MUTS[@]}"; do [ -n "$f" ] && [ -f "$f" ] && rm -f -- "$f"; done
  if [ -n "$T" ] && [ "$T" != "/" ] && [ -d "$T" ]; then rm -rf -- "$T"; fi
}
trap cleanup EXIT

# 1 + 2 (cargo-free: the declared gates are read from the root manifest, so guard_tree's --no-cargo PR job runs this)
if python3 scripts/lib/release_gate_classes.py check . > "$T/live.out" 2>&1; then echo "ok    live: $(tail -1 "$T/live.out")"
else cat "$T/live.out"; bad=1; fi
if python3 scripts/lib/release_gate_classes_cases.py --mutants > "$T/cls.out" 2>&1; then echo "ok    classes: $(tail -1 "$T/cls.out")"
else grep -E 'FAIL|SURVIVED|ANCHOR|CRASHED' "$T/cls.out"; bad=1; fi

# 3
if bash scripts/check_no_shadowed_repo_skill.sh --self-test > "$T/sh.out" 2>&1; then echo "ok    shadow: $(tail -1 "$T/sh.out")"
else cat "$T/sh.out"; bad=1; fi
smut() { # smut <label> <row> <python old> <python new>
  local m="$T/sh-$1.sh"
  python3 -c 'import sys; s=open(sys.argv[1]).read(); assert s.count(sys.argv[3])==1; open(sys.argv[2],"w").write(s.replace(sys.argv[3],sys.argv[4]))' \
    scripts/check_no_shadowed_repo_skill.sh "$m" "$3" "$4" || { echo "FAIL  shadow mutant $1 did not apply"; bad=1; return; }
  local so; so=$(bash "$m" --self-test 2>&1)   # never through a pipe: the mutated self-test exits 1 (pipefail)
  if grep -q "^FAIL  $2 " <<< "$so"; then echo "ok    shadow mutant $1 killed by $2"; else echo "FAIL  shadow mutant $1 SURVIVED $2"; bad=1; fi
}
smut name-ignored   same-name-shadows 'out[f] = {d, m.group(1)} if m else {d}' 'out[f] = {d}'
smut dir-ignored    same-directory-shadows 'out[f] = {d, m.group(1)} if m else {d}' 'out[f] = {m.group(1)} if m else set()'
smut any-dir-skill  a-dir-without-SKILL.md-is-not-a-skill 'glob.glob(os.path.join(root, "*", "SKILL.md"))' 'glob.glob(os.path.join(root, "*", "*"))'

# 4. the watch, on this checkout's HEAD, with fake gate producers
HEADSHA=$(git rev-parse HEAD)
mkdir -p "$T/home/.claude/skills"
# a scratch infra checkout for G-ONT: every ONT row bound, its one done_when probe passing
ONTI="$T/ont-infra"; mkdir -p "$ONTI/scripts/ont/done_when" "$ONTI/docs/specifications" "$ONTI/docs/audits/ONT-001"
echo "# ONT-001" > "$ONTI/docs/specifications/paiml-ontology.md"; : > "$ONTI/docs/audits/ONT-001/ledger.jsonl"
printf '#!/bin/bash\necho "precondition-lint: rows=2 bound=2 unbound=0 probe_paths=1 declared=0 violations=0"\n' > "$ONTI/scripts/ont/precondition-lint.sh"
printf '#!/bin/bash\nexit 0\n' > "$ONTI/scripts/ont/done_when/ONT-1.sh"
git -C "$ONTI" init -q && git -C "$ONTI" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t add -A \
  && git -C "$ONTI" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm ont
receipt() { # receipt <file> <commit> <gate=RESULT>...
  local f=$1 c=$2; shift 2
  python3 - "$f" "$c" "$@" <<'PY'
import json, sys
f, c = sys.argv[1], sys.argv[2]
gates = [{"gate": a.split("=")[0], "result": a.split("=")[1], "note": "fixture"} for a in sys.argv[3:]]
json.dump({"crate": "aprender", "commit": c, "gates": gates, "verdict": "GO"}, open(f, "w"))
PY
}
watch() { # watch <script> <state> <receipt file> [extra watch args...] -> sets WRC, WJ (the watch json)
  local s=$1 st=$2 rf=$3; shift 3
  env WATCH_DOGFOOD_CMD="echo $rf" WATCH_PREFLIGHT_CMD="${PF:-true}" WATCH_BUMP_CMD=true WATCH_HOME="${WH:-$T/home}" APR_ONT_INFRA="${OI-$ONTI}" \
    bash "$s" 0.70.0 --state "$st" "$@" > "$T/w.out" 2>&1; WRC=$?
  WJ=$(ls -t "$st"/0.70.0/watch-*.json 2> /dev/null | head -1)
}
# the predicate is THIS script's own literal (never data from the receipt): python evaluates it against the watch JSON
jq_has() { python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if eval(sys.argv[2]) else 1)' "$WJ" "$1" 2> /dev/null; }  # bashrs disable-line=SEC012
wtable() { # wtable <script> -> ok/FAIL lines
  local s=$1 st
  st=$(mktemp -d -p "$T"); receipt "$T/r1.json" "$HEADSHA" model-parity=PASS bashrs=PASS; watch "$s" "$st" "$T/r1.json"
  if [ "$WRC" = 0 ] && jq_has 'not d["andon"]'; then echo "ok    watch-green"; else echo "FAIL  watch-green rc $WRC"; fi
  st=$(mktemp -d -p "$T"); receipt "$T/r2.json" "$HEADSHA" model-parity=PASS bashrs=FAIL; watch "$s" "$st" "$T/r2.json"
  if [ "$WRC" = 0 ] && jq_has '"dogfood:bashrs" in d["bookkeeping_red"] and not d["andon"]'; then echo "ok    watch-bookkeeping-red-never-blocks"
  else echo "FAIL  watch-bookkeeping-red-never-blocks rc $WRC"; fi
  st=$(mktemp -d -p "$T"); receipt "$T/r3.json" "$HEADSHA" model-parity=FAIL bashrs=PASS; watch "$s" "$st" "$T/r3.json"
  if [ "$WRC" = 1 ] && jq_has '"dogfood:model-parity" in d["real_red"]'; then echo "ok    watch-real-red-andons"
  else echo "FAIL  watch-real-red-andons rc $WRC"; fi
  st=$(mktemp -d -p "$T"); receipt "$T/r4.json" "$HEADSHA" brand-new-gate=FAIL; watch "$s" "$st" "$T/r4.json"
  if [ "$WRC" = 1 ] && jq_has 'any(r["gate"] == "dogfood:brand-new-gate" and "UNCLASSIFIED" in r["class"] for r in d["rows"])'; then  # bashrs disable-line=SC2135
    echo "ok    watch-unclassified-is-real"; else echo "FAIL  watch-unclassified-is-real rc $WRC"; fi
  st=$(mktemp -d -p "$T"); receipt "$T/r5.json" "$HEADSHA" bashrs=FAIL; watch "$s" "$st" "$T/r5.json"
  local first; first=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["dogfood:bashrs"]["at"])' "$st/0.70.0/first-red.json" 2> /dev/null)
  sleep 1.1; watch "$s" "$st" "$T/r5.json"
  if jq_has 'any(r["gate"] == "dogfood:bashrs" and r["first_red_at"] == "'"$first"'" for r in d["rows"])' && [ -n "$first" ]; then  # bashrs disable-line=SC2135
    receipt "$T/r5b.json" "$HEADSHA" bashrs=PASS; watch "$s" "$st" "$T/r5b.json"
    if ! grep -q 'dogfood:bashrs' "$st/0.70.0/first-red.json"; then echo "ok    watch-first-red-persists-then-clears"
    else echo "FAIL  watch-first-red-persists-then-clears (not cleared)"; fi
  else echo "FAIL  watch-first-red-persists-then-clears (first $first)"; fi
  st=$(mktemp -d -p "$T"); receipt "$T/r6.json" 0000000000000000000000000000000000000000 model-parity=PASS; watch "$s" "$st" "$T/r6.json"
  if [ "$WRC" = 1 ] && jq_has '"step:dogfood" in d["real_red"]'; then echo "ok    watch-stale-receipt-is-red"
  else echo "FAIL  watch-stale-receipt-is-red rc $WRC"; fi
  st=$(mktemp -d -p "$T"); mkdir -p "$T/home2/.claude/skills/dogfood"; printf -- '---\nname: dogfood\n---\n' > "$T/home2/.claude/skills/dogfood/SKILL.md"
  WH="$T/home2" watch "$s" "$st" "$T/r1.json"
  if [ "$WRC" = 1 ] && jq_has '"shadow:skills" in d["real_red"]'; then echo "ok    watch-shadow-refuses"
  else echo "FAIL  watch-shadow-refuses rc $WRC"; fi
  st=$(mktemp -d -p "$T"); receipt "$T/r7.json" "$HEADSHA" declared:check_no_claim_literals=FAIL pmat-verify=FAIL model-parity=PASS
  WATCH_AUTOFIX_CMD="echo AUTOFIX-REQUESTED" watch "$s" "$st" "$T/r7.json" --autofix
  local ok7=0; [ "$WRC" = 0 ] && grep -q '^AUTOFIX-REQUESTED claims complexity$' "$T/w.out" && grep -q 'Bookkeeping auto-fixes' "$(ls -t "$st"/0.70.0/watch-*.md | head -1)" && ok7=1
  # a REAL red with a fixer (contracts: pv lint + census) gets its proposal AND keeps its andon
  st=$(mktemp -d -p "$T"); receipt "$T/r8.json" "$HEADSHA" contracts=FAIL; WATCH_AUTOFIX_CMD="echo AUTOFIX-REQUESTED" watch "$s" "$st" "$T/r8.json" --autofix
  [ "$WRC" = 1 ] && grep -q '^AUTOFIX-REQUESTED census readme$' "$T/w.out" || ok7=0
  if [ "$ok7" = 1 ]; then
    echo "ok    watch-autofix-runs-the-fixer-of-a-bookkeeping-red"; else echo "FAIL  watch-autofix-runs-the-fixer-of-a-bookkeeping-red rc $WRC"; fi
  st=$(mktemp -d -p "$T"); OI="" watch "$s" "$st" "$T/r1.json"
  if [ "$WRC" = 1 ] && jq_has '"g-ont:complete" in d["real_red"]'; then echo "ok    watch-g-ont-unconfigured-is-red"
  else echo "FAIL  watch-g-ont-unconfigured-is-red rc $WRC"; fi
  st=$(mktemp -d -p "$T"); PF=false watch "$s" "$st" "$T/r1.json"
  if [ "$WRC" = 1 ] && jq_has '"preflight:R5" in d["real_red"]'; then echo "ok    watch-preflight-red-andons"
  else echo "FAIL  watch-preflight-red-andons rc $WRC"; fi
}
out=$(wtable scripts/release/candidate_watch.sh); printf '%s\n' "$out"; grep -q '^FAIL' <<< "$out" && bad=1
wmut() { # wmut <label> <row> <python old> <python new>
  local m="scripts/release/.m-$1.sh"; MUTS+=("$m")
  python3 -c 'import sys; s=open(sys.argv[1]).read(); assert s.count(sys.argv[3])==1; open(sys.argv[2],"w").write(s.replace(sys.argv[3],sys.argv[4]))' \
    scripts/release/candidate_watch.sh "$m" "$3" "$4" || { echo "FAIL  watch mutant $1 did not apply"; bad=1; return; }
  local mo; mo=$(wtable "$m")
  if grep -q "^FAIL  $2" <<< "$mo"; then echo "ok    watch mutant $1 killed by $2"; else echo "FAIL  watch mutant $1 SURVIVED $2"; bad=1; fi
}
wmut bookkeeping-blocks watch-bookkeeping-red-never-blocks 'sys.exit(1 if real_red else 0)' 'sys.exit(1 if real_red or book_red else 0)'
wmut real-ignored       watch-real-red-andons 'sys.exit(1 if real_red else 0)' 'sys.exit(0)'
wmut unclassified-waived watch-unclassified-is-real 'else "real"   # UNCLASSIFIED' 'else "bookkeeping"   # UNCLASSIFIED'
wmut first-red-reset    watch-first-red-persists-then-clears 'first.setdefault(gid, {"at": now, "sha": sha})' 'first[gid] = {"at": now, "sha": sha}'
wmut stale-receipt-ok   watch-stale-receipt-is-red 'if r.get("commit") and not sha.startswith(r["commit"][:7]):' 'if False:'
wmut autofix-skipped    watch-autofix-runs-the-fixer-of-a-bookkeeping-red 'if [ "$AUTOFIX" = 1 ]; then' 'if false; then'
wmut g-ont-skipped      watch-g-ont-unconfigured-is-red 'printf '"'"'g-ont:complete\tFAIL' ': printf '"'"'g-ont:complete\tFAIL'
wmut shadow-skipped     watch-shadow-refuses 'if ! bash scripts/check_no_shadowed_repo_skill.sh' 'if false && bash scripts/check_no_shadowed_repo_skill.sh'

echo "check_release_shift_left: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
exit "$bad"
