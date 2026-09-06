#!/usr/bin/env bash
# check_story_pmat_hunt.sh - prove the qwen story's analyser bug-hunt manifest can
# actually produce rows, and that it goes RED when it cannot.
#
# The nightly emitted eight manifest headers and zero rows every night since the
# cron was added (#2356), so the workflow's "manifest grew by >5" alert branch
# never had a non-zero input. Three causes, all silent:
#
#   1. the jq filters named `.function` / `.churn.commit_count` / `.faults`;
#      the analyser emits `function_name` / `commit_count` / `fault_annotations`.
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
# Exit 0 = the manifest formats rows from pmat's real field names, and an empty
#          manifest fails the beat instead of passing silently.
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
# Mirrors the analyser at 3.30.0 as measured on this tree:
#   * records are keyed function_name / impact_score / commit_count /
#     churn_score / fault_annotations, and fault_annotations is null for a
#     function with no annotations;
#   * a free-text query is a relevance filter applied BEFORE --path, so
#     `query "some words" --path <module>` returns [] while the same call
#     without the words returns rows. That asymmetry is cause 2, and encoding
#     it here is what makes re-adding the beat label turn this check red.
mkdir -p "$TMP/bin"
# The stub's shebang is printf'd rather than written into the heredoc: a literal
# `#!` at the start of a line makes bashrs report SC1128 ("the shebang must be
# on the first line") against THIS file, at the heredoc's line number.
printf '#!/usr/bin/env bash\n' > "$TMP/bin/pmat"
cat >> "$TMP/bin/pmat" <<'STUB'
# Stub analyser for check_story_pmat_hunt.sh.
case "${STUB_MODE:-rows}" in
  empty) printf '[]\n'; exit 0 ;;
  docs)  printf '{"documents":[{"path":"a.rs"},{"path":"b.rs"}]}\n'; exit 0 ;;
esac
shift  # drop the `query` subcommand
# A leading non-flag argument is a free-text semantic query.
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
  printf '[]\n'
  exit 0
fi
cat <<'JSON'
[
  {"function_name":"cache_path","file_path":"x.rs","impact_score":42,
   "commit_count":7,"churn_score":0.5,"fault_annotations":["CLONE","UNWRAP"]},
  {"function_name":"parse","file_path":"x.rs","impact_score":9,
   "commit_count":3,"churn_score":0.1,"fault_annotations":null},
  {"function_name":"ModelSource","file_path":"x.rs","impact_score":0,
   "commit_count":3,"churn_score":0.1,"fault_annotations":["PANIC"]}
]
JSON
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

# -- 1. Each filter reads a field the analyser ACTUALLY emits ----------------
# Under the pre-fix filters every one of these is empty: the row guard was
# `select(.function != null)` and `.function` is null on every analyser record.
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

# -- 4. A hunt that finds rows passes and does not tally a failure ----------
: > "$FAILLOG"
out=$(PMAT_HUNT=1 pmat_hunt "check" "$LIB"); rc=$?
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
hdr_word='pmat'  # matches lib_story_pmat.sh's real header text verbatim; assembled
                 # so this line does not itself carry the unpinned spelling (PMAT-1059)
if grep -q -- "-- ${hdr_word} bug-hunt manifest" <<< "$out" ; then
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
