#!/usr/bin/env bash
# check_story_pmat_hunt.sh - prove the qwen story's pmat bug-hunt manifest can
# actually produce rows, and that it goes RED when it cannot.
#
# The nightly emitted eight manifest headers and zero rows every night since the
# cron was added (#2356), so the workflow's "manifest grew by >5" alert branch
# never had a non-zero input. Three causes, all silent:
#
#   1. the jq filters named `.function` / `.churn.commit_count` / `.faults`;
#      pmat emits `function_name` / `commit_count` / `fault_annotations`.
#      `select(.function != null)` discarded every record before formatting.
#   2. the churn and fault hunts passed the beat label as a free-text query.
#      That is a relevance filter applied BEFORE `--path`, so a module-scoped
#      hunt collapsed to nothing.
#   3. three of the hunted paths had been deleted or moved.
#
# These are BEHAVIOURAL assertions: they run the REAL pmat_rows / pmat_hunt from
# scripts/lib_story_pmat.sh against a stub `pmat` whose JSON shape and whose
# query-narrows-before-path behaviour are both taken from measurements on pmat
# 3.30.0. A grep-for-the-field-name guard would not have caught cause 2 at all,
# and would have to be rewritten every time the formatter changes.
#
# Exit 0 = the manifest formats rows from pmat's real field names, an empty
#          manifest fails the beat instead of passing silently, and coverage
#          gaps not measured on this commit fail it too (#4715).
# Exit 1 = at least one assertion failed.

set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

LIB="scripts/lib_story_pmat.sh"
STORY="scripts/qwen-story.sh"
[ -f "$LIB" ] || { echo "check_story_pmat_hunt: missing $LIB"; exit 1; }

fails=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { fails=$((fails+1)); printf '  FAIL  %s\n     expected: %s\n     actual:   %s\n' "$1" "$2" "$3"; }
want() { # name expected actual
  # Body on its own line: with the `[ ... ]` on the same line as `want()`,
  # bashrs reads the function-definition parentheses as unescaped parens inside
  # a test and errors (SC1028).
  if [ "$2" = "$3" ]; then ok "$1"; else bad "$1" "$2" "$3"; fi
}

TMP=$(mktemp -d "${TMPDIR:-/tmp}/story-pmat-check.XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT

# -- the stub ---------------------------------------------------------------
# Mirrors pmat 3.30.0 as measured on this tree:
#   * records are keyed function_name / impact_score / commit_count /
#     churn_score / fault_annotations, and fault_annotations is null for a
#     function with no annotations;
#   * a free-text query is a relevance filter applied BEFORE --path, so
#     `query "some words" --path <module>` returns [] while the same call
#     without the words returns rows. That asymmetry is cause 2, and encoding
#     it here is what makes re-adding the beat label turn this check red.
#   * (#4715, pmat 3.42.0) records come in a fixed order per file, --path is a
#     substring filter, and --limit cuts the answer. A directory --path
#     answers for every file directly in it, each name prefixed with its
#     file's basename, so a cached directory answer shows which file each row
#     was cut from. Each file has FOUR records, and only the fourth is past
#     --limit 3, so a cut that applies the row filter before the limit shows
#     up as a third fault row.
mkdir -p "$TMP/bin"
# The stub's shebang is printf'd rather than written into the heredoc: a literal
# `#!` at the start of a line makes bashrs report SC1128 ("the shebang must be
# on the first line") against THIS file, at the heredoc's line number.
printf '#!/usr/bin/env bash\n' > "$TMP/bin/pmat"
cat >> "$TMP/bin/pmat" <<'STUB'
# Stub pmat for check_story_pmat_hunt.sh.
[ -z "${STUB_ARGS_LOG:-}" ] || printf '%s\n' "$*" >> "$STUB_ARGS_LOG"
case "${STUB_MODE:-rows}" in
  empty) printf '[]\n'; exit 0 ;;
  docs)  printf '{"documents":[{"path":"a.rs"},{"path":"b.rs"}]}\n'; exit 0 ;;
  fail)  exit 3 ;;
esac
# STUB_GAP_MODE breaks only the --coverage-gaps query; churn and fault stay normal.
case " $* " in
  *" --coverage-gaps "*)
    case "${STUB_GAP_MODE:-}" in
      fail)    exit 3 ;;
      nonjson) printf 'error: cannot parse coverage file\n'; exit 0 ;;
      empty)   printf '[]\n'; exit 0 ;;
      blank)   printf 'pmat: coverage file unreadable\n' >&2; exit 0 ;;
    esac ;;
esac
shift  # drop the `query` subcommand
# A leading non-flag argument is a free-text semantic query.
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  printf '[]\n'
  exit 0
fi
path="" limit=1000000
while [ "$#" -gt 0 ]; do
  case "$1" in
    --path)  path="$2"; shift ;;
    --limit) limit="$2"; shift ;;
  esac
  shift
done
records() { # file_path name-prefix
  printf '{"function_name":"%scache_path","file_path":"%s","impact_score":42,"commit_count":7,"churn_score":0.5,"fault_annotations":["CLONE","UNWRAP"]},\n' "$2" "$1"
  printf '{"function_name":"%sparse","file_path":"%s","impact_score":9,"commit_count":3,"churn_score":0.1,"fault_annotations":null},\n' "$2" "$1"
  printf '{"function_name":"%sModelSource","file_path":"%s","impact_score":0,"commit_count":3,"churn_score":0.1,"fault_annotations":["PANIC"]},\n' "$2" "$1"
  printf '{"function_name":"%slate_panic","file_path":"%s","impact_score":0,"commit_count":1,"churn_score":0.0,"fault_annotations":["PANIC"]},\n' "$2" "$1"
}
{
  printf '[\n'
  if [ -n "$path" ] && [ -d "$path" ]; then
    find "$path" -maxdepth 1 -type f | sort | while read -r f; do records "$f" "${f##*/}:"; done
  else
    records "${path:-x.rs}" ""
  fi
  printf '{}]\n'
} | jq -c --argjson n "$limit" '[.[] | select(.function_name != null)] | .[0:$n]'
STUB
chmod +x "$TMP/bin/pmat"
PATH="$TMP/bin:$PATH"
export PATH

echo "check_story_pmat_hunt: manifest row formatting and the zero-row andon"

# -- 0. The library refuses to load without the caller's tally function -----
# pmat_hunt calls emit_fail. Discovering that is missing halfway through a
# nightly is how an advisory helper takes the whole run down.
( unset -f emit_fail 2>/dev/null; . "$LIB" ) >/dev/null 2>&1
want "library refuses to source when emit_fail is undefined" "1" "$?"

FAILLOG="$TMP/emit_fail.log"
emit_fail() { printf '%s :: %s\n' "$1" "$2" >> "$FAILLOG"; }
# shellcheck source=scripts/lib_story_pmat.sh
. "$LIB" || { echo "check_story_pmat_hunt: could not source $LIB"; exit 1; }

# -- 1. Each filter reads a field pmat ACTUALLY emits ------------------------
# Under the pre-fix filters every one of these is empty: the row guard was
# `select(.function != null)` and `.function` is null on every pmat record.
got=$(pmat_rows "$PMAT_FILTER_GAP" --coverage-gaps --path x.rs --rank-by impact --limit 3)
want "gap rows carry function_name and impact_score" \
  "        gap   cache_path (impact=42)" "$(printf '%s\n' "$got" | head -1)"
want "gap block has 3 rows" "3" "$(printf '%s\n' "$got" | grep -c .)"

got=$(pmat_rows "$PMAT_FILTER_CHURN" --path x.rs --churn --limit 3)
want "churn rows read commit_count, not .churn.commit_count" \
  "        churn cache_path (commits=7)" "$(printf '%s\n' "$got" | head -1)"

got=$(pmat_rows "$PMAT_FILTER_FAULT" --path x.rs --faults --exclude-tests --limit 3)
want "fault rows read fault_annotations, not .faults" \
  "        fault cache_path (CLONE,UNWRAP)" "$(printf '%s\n' "$got" | head -1)"
# A function with no annotations must not produce `fault parse ()`: that row is
# noise, and being non-empty it would also defeat the zero-row andon below.
want "fault block skips records whose fault_annotations is null" "2" \
  "$(printf '%s\n' "$got" | grep -c .)"

# -- 2. The document-fallback shape yields nothing, not a jq error -----------
got=$(STUB_MODE=docs pmat_rows "$PMAT_FILTER_GAP" --coverage-gaps --path x.rs)
want "the {\"documents\":[...]} fallback shape yields no rows" "" "$got"

# -- 3. A free-text query before --path collapses a module-scoped hunt -------
# This is cause 2 stated as behaviour rather than as a comment. If a future
# edit re-adds the beat label as a query argument, this row goes empty and the
# hunt in assertion 4 turns red.
got=$(pmat_rows "$PMAT_FILTER_CHURN" "qa validate lint" --path x.rs --churn --limit 3)
want "a leading free-text query returns nothing for a module-scoped path" "" "$got"

# A coverage file measured on THIS checkout, so the gap block is measured
# (section 9 covers the cases where it is not).
# Its source files are the paths the hunts below ask about, absolute as
# llvm-cov writes them.
COVFILE="$TMP/coverage.json"
printf '{"data":[{"files":[{"filename":"/build/aprender/%s"},{"filename":"/build/aprender/%s"}]}]}\n' "$LIB" "$STORY" > "$COVFILE"
HEAD_SHA=$(git rev-parse HEAD)

# -- 4. A hunt that finds rows passes and does not tally a failure ----------
: > "$FAILLOG"
out=$(STORY_COVERAGE_FILE="$COVFILE" STORY_COVERAGE_SHA="$HEAD_SHA" PMAT_HUNT=1 pmat_hunt "check" "$LIB"); rc=$?
want "a hunt with rows returns 0" "0" "$rc"
want "a hunt with rows tallies no failure" "" "$(cat "$FAILLOG")"

# EACH of the three query kinds must contribute, checked SEPARATELY.
#
# This started as one grep for `(gap|churn|fault)`, and re-adding the beat label
# to the churn call - the exact pre-fix shape, cause 2 - left it GREEN: the gap
# call carries no free-text query, so it still produced rows, PMAT_HUNT_ROWS was
# non-zero and the hunt passed. An aggregate row count cannot see one of three
# queries go silent, which is the same blindness that let the whole manifest go
# inert. Per-kind assertions are what turn that mutation red.
for kind in gap churn fault; do
  n=$(printf '%s\n' "$out" | grep -cE "^        $kind ")
  if [ "$n" -gt 0 ]; then
    ok "the hunt emitted $kind rows ($n)"
  else
    bad "the hunt emitted $kind rows" ">= 1" "0 (whole manifest: $out)"
  fi
done

# -- 5. A hunt that finds NOTHING must go RED -------------------------------
# The whole point of #2356: pmat_hunt used to `return 0` unconditionally, so
# eight empty manifests a night were indistinguishable from clean code.
: > "$FAILLOG"
out=$(STUB_MODE=empty PMAT_HUNT=1 pmat_hunt "check" "$LIB"); rc=$?
want "a hunt with zero rows returns 1" "1" "$rc"
if grep -q 'pmat-hunt check' "$FAILLOG"; then
  ok "a hunt with zero rows calls emit_fail"
else
  bad "a hunt with zero rows calls emit_fail" "a pmat-hunt failure" "$(cat "$FAILLOG")"
fi
if grep -q -- '-- pmat bug-hunt manifest' <<< "$out" ; then
  ok "the empty hunt did print a header (this is the inert shape being caught)"
else
  bad "the empty hunt did print a header" "a header" "$out"
fi

# -- 6. A deleted path is named, not merely counted -------------------------
: > "$FAILLOG"
STUB_MODE=empty PMAT_HUNT=1 pmat_hunt "check" "scripts/does_not_exist.rs" >/dev/null 2>&1
if grep -q 'do not exist:.*does_not_exist.rs' "$FAILLOG"; then
  ok "a hunted path that no longer exists is named in the failure"
else
  bad "a hunted path that no longer exists is named in the failure" \
    "message naming does_not_exist.rs" "$(cat "$FAILLOG")"
fi

# -- 7. PMAT_HUNT=0 stays silent and stays green ----------------------------
# The opt-out must not print a header, or the workflow's manifest extractor
# would count an opt-out as growth.
: > "$FAILLOG"
out=$(PMAT_HUNT=0 pmat_hunt "check" "$LIB"); rc=$?
want "PMAT_HUNT=0 returns 0" "0" "$rc"
want "PMAT_HUNT=0 prints nothing" "" "$out"
want "PMAT_HUNT=0 tallies no failure" "" "$(cat "$FAILLOG")"

# -- 9. Coverage gaps come from a coverage file of THIS commit, or fail ------
# #4715: without --coverage-file one `pmat query --coverage-gaps` ran out a
# 900 s timeout on intel, once per hunted path per beat, so the nightly was
# cancelled each night. So without a coverage file for HEAD the query must not
# run at all, and the hunt must go RED as not_measured. Dropping the gap block
# and passing on churn + fault rows alone would be a green with no measurement.
ARGS="$TMP/pmat.args"
cov_case() { # name file sha want-rc want-coverage-query(yes|no) want-fail-text
  : > "$FAILLOG"; : > "$ARGS"
  out=$(STUB_GAP_MODE="${GAP_MODE:-}" STUB_ARGS_LOG="$ARGS" STORY_COVERAGE_FILE="$2" STORY_COVERAGE_SHA="$3" PMAT_HUNT=1 pmat_hunt "check" "$LIB"); rc=$?
  want "$1: hunt returns $4" "$4" "$rc"
  if grep -q -- '--coverage-gaps' "$ARGS"; then ran=yes; else ran=no; fi
  want "$1: coverage-gaps query ran = $5" "$5" "$ran"
  if [ -n "$6" ]; then
    if grep -q -- "$6" "$FAILLOG"; then ok "$1: failure says '$6'"; else bad "$1: failure says '$6'" "$6" "$(cat "$FAILLOG")"; fi
    if ! grep -q -- '-- coverage gaps not_measured:' <<<"$out"; then
      bad "$1: the manifest prints the not_measured line" "a '-- coverage gaps not_measured:' line" "$out"
    elif grep -qE '^[[:space:]]+(gap|churn|fault)[[:space:]].*not_measured' <<<"$out"; then
      bad "$1: the not_measured line is not counted as a manifest row" "no row-shaped line" "$out"
    else
      ok "$1: the not_measured line is not counted as a manifest row"
    fi
  else
    want "$1: no failure tallied" "" "$(cat "$FAILLOG")"
  fi
}
cov_case "no coverage file"        ""              ""                 1 no  "coverage gaps not_measured: no coverage file"
: > "$TMP/empty.json"   # zero bytes: exercises the -s branch, not the missing-file one
cov_case "empty coverage file"     "$TMP/empty.json" "$HEAD_SHA"      1 no  "coverage gaps not_measured: no coverage file"
cov_case "coverage path that does not exist" "$TMP/absent.json" "$HEAD_SHA" 1 no "coverage gaps not_measured: no coverage file"
cov_case "file from another commit" "$COVFILE"     "0000000000000000000000000000000000000000" 1 no "measured on '0000000000000000000000000000000000000000'"
cov_case "file with no recorded sha" "$COVFILE"    ""                 1 no  "measured on '(none recorded)'"
cov_case "file from this commit"   "$COVFILE"      "$HEAD_SHA"        0 yes ""
# The file is this commit's, but the gap query itself is not a measurement: pmat
# fails on it, prints no JSON, or exits 0 with an empty stdout (a real pmat failure mode: stderr only). Churn and fault rows alone must not pass the beat.
GAP_MODE=fail    cov_case "gap query exits non-zero" "$COVFILE" "$HEAD_SHA" 1 yes "coverage gaps not_measured: the coverage-gaps query failed"
GAP_MODE=nonjson cov_case "gap query prints non-JSON" "$COVFILE" "$HEAD_SHA" 1 yes "coverage gaps not_measured: the coverage-gaps query on"
GAP_MODE=blank   cov_case "gap query prints nothing" "$COVFILE" "$HEAD_SHA" 1 yes "coverage gaps not_measured: the coverage-gaps query on"
# A parsed [] is a measured zero: no gaps on this path, and churn + fault carry the manifest.
GAP_MODE=empty   cov_case "gap query returns []"     "$COVFILE" "$HEAD_SHA" 0 yes ""
cov_case "file from this commit, again" "$COVFILE" "$HEAD_SHA"        0 yes ""
if grep -q -- "--coverage-file $COVFILE" "$ARGS"; then
  ok "the measured gap query reads the coverage file instead of deriving coverage"
else
  bad "the measured gap query reads the coverage file" "--coverage-file $COVFILE" "$(cat "$ARGS")"
fi
# A path the coverage file does not contain is not asked about: pmat answers
# it with an empty stdout, which was not_measured already. It stays
# not_measured, with the reason named, and no query is spent on it (#4715).
COV_OTHER="$TMP/coverage-other.json"
printf '{"data":[{"files":[{"filename":"/build/aprender/crates/x/src/lib.rs"}]}]}\n' > "$COV_OTHER"
cov_case "path outside the coverage file's scope" "$COV_OTHER" "$HEAD_SHA" 1 no "is outside the coverage file's scope"
# A coverage file jq cannot read as LLVM JSON gets no scope list, so the gap
# query runs as before and pmat's own answer decides.
printf 'not json\n' > "$TMP/coverage-bad.json"
cov_case "unreadable coverage file is still asked" "$TMP/coverage-bad.json" "$HEAD_SHA" 0 yes ""
# Nor does a file that names no source file: an empty list is a shape this
# reader does not know, not proof that the path is out of scope.
printf '{"data":[]}\n' > "$TMP/coverage-nofiles.json"
cov_case "coverage file naming no source file is still asked" "$TMP/coverage-nofiles.json" "$HEAD_SHA" 0 yes ""

# -- 10. One cached directory query serves every hunted path under it -------
# #4715: pmat 3.42.0 rebuilds its index on every query, so the story ran out
# its 30 minutes on 42 churn and fault queries. They now come from one query
# per (directory, flag set), cut per path. The cut must give each path its OWN
# first three records, in pmat's order, BEFORE the row filter.
CACHE="$TMP/cache"
: > "$ARGS"; : > "$FAILLOG"
out1=$(STUB_ARGS_LOG="$ARGS" PMAT_HUNT_CACHE_DIR="$CACHE" STORY_COVERAGE_FILE="$COVFILE" STORY_COVERAGE_SHA="$HEAD_SHA" PMAT_HUNT=1 pmat_hunt "one" "$LIB" "$STORY")
out2=$(STUB_ARGS_LOG="$ARGS" PMAT_HUNT_CACHE_DIR="$CACHE" STORY_COVERAGE_FILE="$COVFILE" STORY_COVERAGE_SHA="$HEAD_SHA" PMAT_HUNT=1 pmat_hunt "two" "$LIB"); rc=$?
want "cached: the second beat returns 0" "0" "$rc"
want "cached: 3 paths over 2 beats ran ONE churn query" "1" "$(grep -c -- '--churn' "$ARGS")"
want "cached: 3 paths over 2 beats ran ONE fault query" "1" "$(grep -c -- '--faults' "$ARGS")"
want "cached: the churn query asked for the parent directory" "1" "$(grep -c -- '^query --path scripts --churn' "$ARGS")"
want "cached: churn rows for the library are its own 3" "3" "$(printf '%s\n' "$out2" | grep -c "^        churn ${LIB##*/}:")"
want "cached: no other file's churn rows" "3" "$(printf '%s\n' "$out2" | grep -c '^        churn ')"
want "cached: each path in a beat gets its own rows" "3" "$(printf '%s\n' "$out1" | grep -c "^        churn ${STORY##*/}:")"
# Of the first three records, two carry fault annotations. A cut that filters
# before taking three would add the fourth, late_panic.
want "cached: fault rows are the annotated ones among the first 3" "2" "$(printf '%s\n' "$out2" | grep -c '^        fault ')"
want "cached: the fourth record never becomes a row" "0" "$(printf '%s\n' "$out1$out2" | grep -c 'late_panic')"
want "cached: the gap query stays per path" "2" "$(grep -c -- "--coverage-gaps.*--path $LIB" "$ARGS")"
want "cached: no failure tallied" "" "$(cat "$FAILLOG")"

# A directory answer that reaches PMAT_HUNT_DIR_LIMIT may be cut short, so
# the path is asked on its own.
: > "$ARGS"
out=$(PMAT_HUNT_DIR_LIMIT=2 STUB_ARGS_LOG="$ARGS" PMAT_HUNT_CACHE_DIR="$TMP/cache-limit" STORY_COVERAGE_FILE="$COVFILE" STORY_COVERAGE_SHA="$HEAD_SHA" PMAT_HUNT=1 pmat_hunt "limit" "$LIB")
want "truncated: the path is asked on its own" "1" "$(grep -c -- "^query --path $LIB --churn --max-complexity 30 --limit 3" "$ARGS")"
want "truncated: the per-path rows are printed" "3" "$(printf '%s\n' "$out" | grep -cE '^        churn (cache_path|parse|ModelSource) ')"

# A failed directory query is cached as failed: no rows for any path under
# it, one query, and the zero-row andon fails the beat.
: > "$ARGS"; : > "$FAILLOG"
STUB_MODE=fail STUB_ARGS_LOG="$ARGS" PMAT_HUNT_CACHE_DIR="$TMP/cache-fail" PMAT_HUNT=1 pmat_hunt "fail" "$LIB" "$STORY" >/dev/null; rc=$?
want "failed directory query: the hunt returns 1" "1" "$rc"
want "failed directory query: asked once for two paths" "1" "$(grep -c -- '--churn' "$ARGS")"

# Without PMAT_HUNT_CACHE_DIR the hunt keeps a private cache and removes it.
mkdir -p "$TMP/t"
( TMPDIR="$TMP/t" PMAT_HUNT=1 pmat_hunt "own" "$LIB" >/dev/null 2>&1; printf '%s' "${PMAT_HUNT_CACHE_DIR:-}" > "$TMP/own.var" )
want "a private cache is removed after the hunt" "" "$(find "$TMP/t" -mindepth 1 | head -1)"
want "a private cache is not left in PMAT_HUNT_CACHE_DIR" "" "$(cat "$TMP/own.var")"

# -- 11. The story's own preamble removes the shared cache on exit ----------
# The cache lives under TMPDIR_STORY, so the story's EXIT trap is what removes
# it. Run the story's real preamble, at most 12 lines from TMPDIR_STORY= to the
# trap, rather than trusting a reading of it: `trap'...' EXIT` (one lost space)
# is valid syntax, fails at run time with "command not found", and installs no
# trap at all.
pre=$(awk '/^TMPDIR_STORY=/{p=1} p{print; n++} p && /^trap/{exit} n >= 12{exit}' "$STORY")
TMPDIR_STORY="$TMP/story-tmp" bash -c "$pre"'
mkdir -p "$PMAT_HUNT_CACHE_DIR" && : > "$PMAT_HUNT_CACHE_DIR/x"
trap -p EXIT' > "$TMP/trap.out" 2>&1
want "qwen-story.sh: the cache is under TMPDIR_STORY" "1" "$(printf '%s\n' "$pre" | grep -c '^PMAT_HUNT_CACHE_DIR="\$TMPDIR_STORY/')"
want "qwen-story.sh: the preamble installs an EXIT trap" "1" "$(grep -c '^trap -- .*rm -rf.* EXIT$' "$TMP/trap.out")"
want "qwen-story.sh: the EXIT trap removes the story dir and its cache" "gone" "$([ -e "$TMP/story-tmp" ] && echo left || echo gone)"

# -- 8. Every path the story hunts still exists -----------------------------
# Cause 3. Static, because a path can rot without anyone running the nightly.
if [ -f "$STORY" ]; then
  missing=""
  checked=0
  while read -r p; do
    checked=$((checked + 1))
    [ -e "$p" ] || missing="$missing $p"
  done < <(grep -oE '(crates|src)/[A-Za-z0-9_./-]+' "$STORY" \
             | grep -E '\.rs$|/serve$' | sort -u)
  want "every source path hunted by qwen-story.sh exists" "" "$missing"
  # Fail closed: a discovery step that found nothing must not report success.
  if [ "$checked" -ge 15 ]; then
    ok "path discovery found $checked hunted paths"
  else
    bad "path discovery found enough hunted paths" ">= 15" "$checked"
  fi
fi

echo
if [ "$fails" -eq 0 ]; then
  echo "check_story_pmat_hunt: OK - the manifest produces rows, and an empty one fails the beat"
  exit 0
fi
echo "check_story_pmat_hunt: $fails assertion(s) FAILED"
exit 1
