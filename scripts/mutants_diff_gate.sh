#!/usr/bin/env bash
# mutants_diff_gate.sh -- the PR mutation gate (#4142): mutate the PR diff across the WORKSPACE and judge it, and
# never pass on anything it did not judge.
#
# WHY. Until #4142 the `mutants` job ran `cargo mutants --in-diff pr.diff -- --lib` at the workspace ROOT without
# `--workspace`. cargo-mutants then mutates only the package in the cwd, the near-empty root facade, so every diff
# under crates/** listed ZERO mutants and the job printed "0 mutants in diff. Pass." (measured on #3772, #3737,
# #3707 and #3706). The blocking gate had never mutated a line of crate code. `.cargo/mutants.toml` cannot fix
# that: cargo-mutants 27 refuses `workspace = true` ("unknown field"). The flag is the only way.
#
# RULES, in order (operator ruling via the cop, 2026-09-24: "go with recommended"):
#   1. LIST with --workspace. A list that fails is RED: the gate cannot say what it would judge.
#   2. 0 mutants listed -> pass, saying so. (A diff of tests, docs or data has nothing to mutate.)
#   3. more than --cap mutants -> RED "too many to judge": the job's hour holds about 60 at -j 4. Split the PR.
#      The nightly run (mutants-nightly.yml, --cap 0 = no cap) judges main's day of merges in full.
#   4. RUN with --workspace -j N (copies, not --in-place). No outcomes.json is RED, whatever the exit code: the
#      list said there were mutants, so a run that wrote nothing died.
#   5. An unparseable outcomes.json is RED (not being able to measure is not measuring zero), and so is a
#      TESTED count that differs from the LISTED count.
#   6. missed + timeout > --max-missed is RED, naming each survivor.
#   --exclude-crate NAME (repeatable) keeps crates/NAME/** out of mutation, for the crates CI's workspace-test also
#   excludes (aprender-gpu, aprender-cuda-edge, aprender-compute: hardware, or a harness that segfaults at exit on a
#   clean pass, so cargo-mutants would read its baseline as failed). The exclusion is printed, never silent.
#
#   bash scripts/mutants_diff_gate.sh <diff> [--cap N] [--jobs J] [--max-missed M] [--out DIR] [--exclude-crate NAME]...
#   bash scripts/mutants_diff_gate.sh --self-test
# exit 0 judged and within --max-missed . 1 RED . 2 usage
# Seam (the self-test only): MUTANTS_GATE_CARGO replaces `cargo`.
set -uo pipefail

gate() { # gate <diff> <cap> <jobs> <max-missed> <out> [excluded crate...]
  local diff=$1 cap=$2 jobs=$3 max=$4 out=$5 cargo=${MUTANTS_GATE_CARGO:-cargo} n rc oc c
  shift 5
  local -a ex=()
  for c in "$@"; do ex+=(--exclude "crates/$c/**"); done
  [ -s "$diff" ] || { echo "ok    empty diff: nothing to mutate"; return 0; }
  mkdir -p "$out" || return 1
  [ "$#" -gt 0 ] && echo "excluded from mutation (judged by their own CI steps): $*"
  "$cargo" mutants --workspace "${ex[@]}" --in-diff "$diff" --list > "$out/list.txt" 2> "$out/list.err"; rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "RED   cargo mutants --list exited $rc: the gate cannot say what it would judge -- $(tail -1 "$out/list.err")"
    return 1
  fi
  n=$(grep -c . "$out/list.txt")
  if [ "$n" -eq 0 ]; then
    echo "ok    0 mutants in the diff, listed across the workspace"
    return 0
  fi
  if [ "$cap" -gt 0 ] && [ "$n" -gt "$cap" ]; then
    echo "RED   too many to judge: $n mutants in the diff > cap $cap -- split the PR (mutants-nightly judges main in full)"
    return 1
  fi
  echo "mutating $n mutant(s) across the workspace, -j $jobs"
  "$cargo" mutants --workspace "${ex[@]}" --no-times --timeout 300 -j "$jobs" --in-diff "$diff" --output "$out" -- --lib; rc=$?
  oc="$out/mutants.out/outcomes.json"
  if [ ! -f "$oc" ]; then
    echo "RED   cargo mutants exited $rc and wrote no outcomes.json although $n mutant(s) were listed: the run died"
    return 1
  fi
  judge "$oc" "$n" "$max" "$out"
}

field() { # field <json> <key> -> the first `"key": <int>` (top-level summary keys; outcome entries carry none)
  grep -oE "\"$2\": ?[0-9]+" "$1" | head -1 | grep -oE '[0-9]+$'
}

judge() { # judge <outcomes.json> <listed> <max-missed> <out>
  local oc=$1 n=$2 max=$3 out=$4 total missed timeout
  total=$(field "$oc" total_mutants); missed=$(field "$oc" missed); timeout=$(field "$oc" timeout)
  if [ -z "$total" ] || [ -z "$missed" ] || [ -z "$timeout" ]; then
    echo "RED   cannot parse total_mutants/missed/timeout out of $oc -- did cargo-mutants change its JSON shape?"
    return 1
  fi
  if [ "$total" -ne "$n" ]; then
    echo "RED   cargo mutants tested $total mutant(s) but listed $n: what was judged is not what the diff holds"
    return 1
  fi
  echo "judged $total mutant(s): missed=$missed timeout=$timeout (max allowed $max)"
  if [ $((missed + timeout)) -gt "$max" ]; then
    echo "RED   $((missed + timeout)) mutant(s) survived or timed out on the diff (> $max): add tests that kill them"
    cat "$out/mutants.out/missed.txt" "$out/mutants.out/timeout.txt" 2> /dev/null | sed 's/^/  /'
    return 1
  fi
  echo "ok    every mutant in the diff was caught (or within --max-missed)"
}

self_test() {
  local T bad=0 SELF
  SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")
  T=$(mktemp -d) || return 2
  # shellcheck disable=SC2064
  trap "rm -rf -- '$T'" RETURN
  cat > "$T/cargo" <<'STUB'
#!/usr/bin/env bash
# A fake `cargo mutants`: STUB_LIST (lines), STUB_LIST_RC, STUB_OUTCOMES (json text or "none"), STUB_MISSED.
echo "$*" >> "$STUB_ARGS"
case " $* " in
  *" --list "*) printf '%b' "${STUB_LIST:-}"; exit "${STUB_LIST_RC:-0}" ;;
esac
out=""; prev=""
for a in "$@"; do [ "$prev" = "--output" ] && out=$a; prev=$a; done
[ "${STUB_OUTCOMES:-none}" = none ] && exit "${STUB_RUN_RC:-0}"
mkdir -p "$out/mutants.out"
printf '%s' "$STUB_OUTCOMES" > "$out/mutants.out/outcomes.json"
printf '%b' "${STUB_MISSED:-}" > "$out/mutants.out/missed.txt"
exit "${STUB_RUN_RC:-0}"
STUB
  chmod +x "$T/cargo"
  printf 'diff --git a/x.rs b/x.rs\n+fn f() {}\n' > "$T/pr.diff"
  row() { # row <name> <want rc> <needle> [ENV=VAL...] -- runs THIS script's gate against the stub
    local name=$1 want=$2 needle=$3 out rc; shift 3
    : > "$T/args-$name"   # a fresh log per row: a stale one would answer for a mutant that never ran
    out=$(env MUTANTS_GATE_CARGO="$T/cargo" STUB_ARGS="$T/args-$name" "$@" \
          bash "$GATE_SCRIPT" "${GATE_DIFF:-$T/pr.diff}" --cap 3 --jobs 2 --max-missed "${MAXM:-0}" --out "$T/out-$name" ${EXCL:-} 2>&1); rc=$?
    if [ "$rc" = "$want" ] && grep -qF -- "$needle" <<< "$out"; then echo "ok    $name"
    else echo "FAIL  $name -- rc $rc (want $want): $(tr '\n' ' ' <<< "$out" | cut -c1-220)"; fi
  }
  OK2='{"total_mutants": 2, "missed": 0, "caught": 2, "timeout": 0, "unviable": 0, "outcomes": [{"summary": "CaughtMutant"}]}'
  table() {
    row zero-listed-passes 0 "0 mutants in the diff" STUB_LIST=""
    row all-caught-passes 0 "every mutant in the diff was caught" STUB_LIST='a\nb\n' STUB_OUTCOMES="$OK2"
    if grep -q -- "--workspace --in-diff .* --list" "$T/args-all-caught-passes" 2> /dev/null \
       && grep -q -- "--workspace --no-times .* -j 2 " "$T/args-all-caught-passes" 2> /dev/null; then
      echo "ok    workspace-on-list-and-run"
    else echo "FAIL  workspace-on-list-and-run -- $(tr '\n' '|' < "$T/args-all-caught-passes" 2> /dev/null)"; fi
    EXCL="--exclude-crate aprender-compute" row excluded-crate-reaches-list-and-run 0 "excluded from mutation (judged by their own CI steps): aprender-compute" \
        STUB_LIST='a\nb\n' STUB_OUTCOMES="$OK2"
    if [ "$(grep -c -- "--exclude crates/aprender-compute/\*\*" "$T/args-excluded-crate-reaches-list-and-run" 2> /dev/null)" = 2 ]; then
      echo "ok    exclusion-on-list-and-run"
    else echo "FAIL  exclusion-on-list-and-run -- $(tr '\n' '|' < "$T/args-excluded-crate-reaches-list-and-run" 2> /dev/null)"; fi
    row list-failure-is-red 1 "--list exited 101" STUB_LIST_RC=101
    row over-cap-is-red 1 "too many to judge: 4 mutants in the diff > cap 3" STUB_LIST='a\nb\nc\nd\n'
    row missed-is-red-and-named 1 "  src/x.rs:1: replace f with ()" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants":2,"missed":1,"caught":1,"timeout":0}' STUB_MISSED='src/x.rs:1: replace f with ()\n'
    row timeout-counts-as-uncaught 1 "1 mutant(s) survived or timed out" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants": 2, "missed": 0, "caught": 1, "timeout": 1}'
    MAXM=1 row missed-within-tolerance-passes 0 "missed=1" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants": 2, "missed": 1, "caught": 1, "timeout": 0}'
    row no-outcomes-is-red-even-at-rc-0 1 "wrote no outcomes.json although 2 mutant(s)" STUB_LIST='a\nb\n' STUB_OUTCOMES=none
    row unparseable-outcomes-is-red 1 "cannot parse" STUB_LIST='a\nb\n' STUB_OUTCOMES='{"total": 2}'
    row tested-count-mismatch-is-red 1 "tested 1 mutant(s) but listed 2" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES='{"total_mutants": 1, "missed": 0, "caught": 1, "timeout": 0}'
    # the REAL cargo-mutants 27 shape (captured from a run, trimmed): pretty-printed, "outcomes" FIRST with nested
    # phase results, the summary keys AFTER it. The pre-#4142 inline parser read nothing out of this file.
    row real-outcomes-shape-is-parsed 1 "judged 2 mutant(s): missed=1 timeout=0" STUB_LIST='a\nb\n' \
        STUB_OUTCOMES="$(cat "$(dirname "$SELF")/mutants_diff_gate.real-outcomes.json")"
    : > "$T/empty.diff"
    GATE_DIFF="$T/empty.diff" row empty-diff-passes 0 "empty diff" STUB_LIST='a\n'
  }
  GATE_SCRIPT=$SELF
  local o; o=$(table); printf '%s\n' "$o"; grep -q '^FAIL' <<< "$o" && bad=1
  # every rule deleted in a copy of THIS script must break the row that names it (code above self_test only)
  while IFS='~' read -r label must old new; do
    [ -n "$label" ] || continue
    python3 -c 'import sys
s = open(sys.argv[1]).read(); code, cut, rest = s.partition("\nself_test() {")
assert code.count(sys.argv[3]) == 1, "anchor"
open(sys.argv[2], "w").write(code.replace(sys.argv[3], sys.argv[4]) + cut + rest)' "$SELF" "$T/m.sh" "$old" "$new" 2> /dev/null \
      || { echo "FAIL  mutant $label did not apply"; bad=1; continue; }
    GATE_SCRIPT="$T/m.sh"; o=$(table); GATE_SCRIPT=$SELF
    if grep -q "^FAIL  $must " <<< "$o"; then echo "ok    mutant $label killed by $must"
    else echo "FAIL  mutant $label SURVIVED $must"; bad=1; fi
  done <<'MUT'
no-workspace-on-list~workspace-on-list-and-run~"$cargo" mutants --workspace "${ex[@]}" --in-diff "$diff" --list~"$cargo" mutants "${ex[@]}" --in-diff "$diff" --list
no-workspace-on-run~workspace-on-list-and-run~"$cargo" mutants --workspace "${ex[@]}" --no-times~"$cargo" mutants "${ex[@]}" --no-times
list-rc-ignored~list-failure-is-red~  if [ "$rc" -ne 0 ]; then~  if false; then
cap-ignored~over-cap-is-red~  if [ "$cap" -gt 0 ] && [ "$n" -gt "$cap" ]; then~  if false; then
no-outcomes-passes~no-outcomes-is-red-even-at-rc-0~    echo "RED   cargo mutants exited $rc and wrote no outcomes.json although $n mutant(s) were listed: the run died"~    echo "RED   cargo mutants exited $rc and wrote no outcomes.json although $n mutant(s) were listed: the run died"; return 0
compact-json-only-parser~real-outcomes-shape-is-parsed~  grep -oE "\"$2\": ?[0-9]+" "$1" | head -1 | grep -oE '[0-9]+$'~  grep -oE "\"$2\":[0-9]+" "$1" | head -1 | grep -oE '[0-9]+$'
missing-field-is-zero~unparseable-outcomes-is-red~  if [ -z "$total" ] || [ -z "$missed" ] || [ -z "$timeout" ]; then~  total=${total:-2}; missed=${missed:-0}; timeout=${timeout:-0}; if false; then
count-mismatch-ok~tested-count-mismatch-is-red~  if [ "$total" -ne "$n" ]; then~  if false; then
timeout-not-counted~timeout-counts-as-uncaught~  if [ $((missed + timeout)) -gt "$max" ]; then~  if [ "$missed" -gt "$max" ]; then
exclusion-dropped-on-run~exclusion-on-list-and-run~  "$cargo" mutants --workspace "${ex[@]}" --no-times~  "$cargo" mutants --workspace --no-times
exclusion-dropped-on-list~exclusion-on-list-and-run~  "$cargo" mutants --workspace "${ex[@]}" --in-diff "$diff" --list~  "$cargo" mutants --workspace --in-diff "$diff" --list
survivors-unnamed~missed-is-red-and-named~    cat "$out/mutants.out/missed.txt" "$out/mutants.out/timeout.txt" 2> /dev/null | sed 's/^/  /'~    :
MUT
  echo "mutants_diff_gate self-test: $([ "$bad" = 0 ] && echo PASS || echo FAIL)"
  return "$bad"
}

main() {
  local diff="" cap=60 jobs=4 max=0 out=mutants-gate
  local -a excl=()
  [ "${1:-}" = "--self-test" ] && { self_test; exit $?; }
  while [ $# -gt 0 ]; do
    case "$1" in
      --cap) cap=$2; shift 2 ;;
      --jobs) jobs=$2; shift 2 ;;
      --max-missed) max=$2; shift 2 ;;
      --out) out=$2; shift 2 ;;
      --exclude-crate) excl+=("$2"); shift 2 ;;
      -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
      -*) echo "mutants_diff_gate: unknown option $1" >&2; exit 2 ;;
      *) diff=$1; shift ;;
    esac
  done
  [ -n "$diff" ] || { echo "usage: mutants_diff_gate.sh <diff> [--cap N] [--jobs J] [--max-missed M] [--out DIR]" >&2; exit 2; }
  for v in "$cap" "$jobs" "$max"; do [[ "$v" =~ ^[0-9]+$ ]] || { echo "mutants_diff_gate: not a count: $v" >&2; exit 2; }; done
  gate "$diff" "$cap" "$jobs" "$max" "$out" "${excl[@]}"
}

main "$@"
