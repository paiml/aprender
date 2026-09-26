#!/usr/bin/env bash
# lib_story_pmat.sh - the qwen story's pmat bug-hunt manifest.
#
# Why this exists
# ---------------
# The nightly ran the hunt on every one of its 8 beats and produced ZERO rows,
# every night, since the cron was added (#2356). Eight headers, no content. The
# workflow's "manifest grew by >5" alert branch therefore never had a non-zero
# input and could not fire, so half the nightly's alerting was decorative.
#
# Three independent causes, each sufficient on its own:
#
#   1. THE JQ FILTERS NAMED FIELDS PMAT DOES NOT EMIT. `pmat query --format json`
#      returns records keyed `function_name` / `commit_count` / `churn_score` /
#      `fault_annotations` / `impact_score`. The filters read `.function`,
#      `.churn.commit_count` and `.faults`. The row guard was
#      `select(.function != null)`, which is null for EVERY record pmat has ever
#      returned - so the guard discarded 100% of input before any interpolation
#      ran. Measured on pmat 3.30.0:
#
#          $ pmat query --path crates/apr-cli/src/commands/qa.rs --churn \
#              --limit 3 --format json | jq '.[0]|keys'
#          [... "churn_score", "commit_count", ... "function_name", ...]
#
#   2. `--path` IS A POST-FILTER OVER THE SEMANTIC TOP-K, NOT A SEARCH SCOPE.
#      The churn and fault hunts passed the beat label ("qa validate lint") as a
#      free-text query. That query returns THREE records for the entire
#      repository - pmat applies a relevance floor - and `--path` then filters
#      those three. Measured, same tree, same pmat:
#
#          query + --path crates/apr-cli/src/commands/qa.rs  -> 0 rows
#          query + --path crates/apr-cli/src/commands        -> 0 rows
#          query + --path crates/apr-cli                     -> 2 rows
#          query, no --path, --limit 100                     -> 3 rows
#          NO query + --path crates/apr-cli/src/commands/qa.rs -> 3 rows
#
#      So the fix is not "widen the path" (a hunt scoped at crates/apr-cli is
#      not a hunt of the module the beat exercised); it is to drop the free-text
#      query and let --path do the scoping. A file-scoped --path returns rows
#      perfectly well once nothing has already thrown away the candidates.
#
#   3. THREE OF THE PATHS NO LONGER EXIST. commands/list.rs (list now lives in
#      pull.rs - `Commands::List => pull::list(...)`), commands/code.rs (moved to
#      aprender-orchestrate/src/cli/code.rs) and commands/serve.rs (became the
#      commands/serve/ directory). A hunt aimed at a deleted file is silent, and
#      nothing said so.
#
# THE MANIFEST IS NO LONGER ADVISORY.
# -----------------------------------
# It used to be, and `pmat_hunt` returned 0 unconditionally while `pmat_rows`
# ended in `|| true`. That is precisely why causes 1-3 survived: every one of
# them presents as "the manifest is empty tonight", which was indistinguishable
# from "the code is clean tonight". A hunt that prints a header and then no rows
# now calls emit_fail and returns 1, so the story exits 2 and the cron files an
# issue. Silence is a finding, not a pass.
#
# OPTION NEUTRALITY (load-bearing)
# --------------------------------
# This file is SOURCED. It must not run `set`: that mutates the CALLER's shell.
# apr_bin.sh once opened with `set -euo pipefail`, qwen-story.sh sourced it, and
# the leaked errexit killed the nightly six lines in - inside this very hunt.
# Sourceable libraries here fail by RETURN STATUS. Enforced by
# scripts/check_sourced_libs_option_neutral.sh.

# THE HUNT IS BOUNDED IN TIME (#4489)
# ----------------------------------
# qwen-story-daily was CANCELLED at its 30-minute job timeout on every night
# from the one pmat first became visible on the gx10 runner: before that the
# hunt was skipped (pmat absent) and all 8 beats ran in ~5 min. One
# `pmat query --churn` measured 407 s on lambda, and the story runs 3 queries per
# path over ~20 paths. The job died inside Beat 3, so no verdict line, no issue
# and no RED - only "cancelled", which nothing counted. An unbounded advisory
# step had silenced the thing it was advising on.
#
# So every query runs under `timeout` (PMAT_QUERY_TIMEOUT_S), and the hunt as a
# whole spends at most PMAT_HUNT_BUDGET_S of wall clock across ALL beats. A
# query that timed out, and a hunt skipped because the budget is gone, each
# print a `skip` line saying so - visible, never silent - and neither is
# reported as "inert": a hunt that did not finish has not shown the code clean,
# and it has not shown the hunt broken either. The story's verdict is what the
# job exists to produce; the hunt may not cost it.
PMAT_QUERY_TIMEOUT_S="${PMAT_QUERY_TIMEOUT_S:-60}"
PMAT_HUNT_BUDGET_S="${PMAT_HUNT_BUDGET_S:-480}"
PMAT_HUNT_DEADLINE=""
PMAT_HUNT_TIMEOUTS=0
# The sentinel pmat_rows prints (in a command substitution, so it cannot set a
# variable) when its query was killed by the timeout.
PMAT_TIMEOUT_MARK="__PMAT_QUERY_TIMED_OUT__"

# emit_fail is the caller's tally function (it feeds FAILED_BEATS and the exit-2
# verdict). Fail at SOURCE time rather than discovering it is missing halfway
# through a nightly - and fail by return status, never by `set -e`.
if ! declare -F emit_fail >/dev/null 2>&1; then
  printf 'lib_story_pmat.sh: caller must define emit_fail() before sourcing this library\n' >&2
  return 1 2>/dev/null || exit 1
fi

# The three row formatters, kept next to the field-name evidence above so a
# rename in pmat's JSON has exactly one place to land.
#
# The fault filter drops records whose fault_annotations is null: pmat returns
# every function in the path, annotated or not, and a bare "fault foo ()" row is
# noise that would also defeat the zero-row check below by being technically
# non-empty.
PMAT_FILTER_GAP='"        gap   \(.function_name) (impact=\(.impact_score // "?"))"'
PMAT_FILTER_CHURN='"        churn \(.function_name) (commits=\(.commit_count // "?"))"'
PMAT_FILTER_FAULT='select((.fault_annotations // []) | length > 0)
  | "        fault \(.function_name) (\(.fault_annotations | join(",")))"'

# Extract rows from one `pmat query`, tolerating BOTH shapes pmat can return.
#
# A function query returns a JSON ARRAY of records, but when nothing matches
# semantically pmat falls back to a document search and returns
# `{"documents":[...]}`. Running `.[]` unconditionally on that yields the
# documents ARRAY, and interpolating a field from an array raises "Cannot index
# array with string" - jq exits 5. Guarding on `type == "array"` makes the
# no-match case yield nothing instead of an error.
#
# pmat is invoked into a variable rather than piped, so PMAT_ROWS_EC is pmat's
# OWN status and not jq's or head's. Reading `$?` after a pipeline is the defect
# this repo has shipped four times.
#
# Args:   <jq-output-filter> <pmat query args...>
# Sets:   PMAT_ROWS_EC  exit status of the `pmat query` invocation
# Prints: at most 3 formatted rows (empty output when there are none)
pmat_rows() {
  local filter="$1"; shift
  local raw prog rows t="${PMAT_ROWS_TIMEOUT_S:-$PMAT_QUERY_TIMEOUT_S}"
  raw=$(timeout "$t" "${PMAT_BIN:-pmat}" query "$@" --format json 2>/dev/null)
  PMAT_ROWS_EC=$?
  # 124 = timeout's own "the command ran out of time" status.
  if [ "$PMAT_ROWS_EC" -eq 124 ]; then
    printf '%s %ss\n' "$PMAT_TIMEOUT_MARK" "$t"
    return 0
  fi
  [ "$PMAT_ROWS_EC" -eq 0 ] || return 0
  # printf rather than a multi-line double-quoted string: bashrs reads the
  # latter as an unterminated string (SC1078).
  prog=$(printf '%s\n%s\n%s' \
    'if type == "array" then .[] else empty end' \
    '| select(.function_name != null)' \
    "| $filter")
  rows=$(printf '%s\n' "$raw" | jq -r "$prog" 2>/dev/null)
  [ -n "$rows" ] || return 0
  printf '%s\n' "$rows" | head -3
}

# Print one block of rows and add its line count to PMAT_HUNT_ROWS.
# A timed-out query prints a skip line instead and adds to PMAT_HUNT_TIMEOUTS.
# Args: <kind> <rows>
_pmat_hunt_emit() {
  [ -n "$2" ] || return 0
  case "$2" in
    "$PMAT_TIMEOUT_MARK "*)
      printf '        skip  %s query timed out after %s\n' "$1" "${2#"$PMAT_TIMEOUT_MARK "}"
      PMAT_HUNT_TIMEOUTS=$((PMAT_HUNT_TIMEOUTS + 1))
      return 0 ;;
  esac
  printf '%s\n' "$2"
  local n
  n=$(printf '%s\n' "$2" | grep -c .)
  PMAT_HUNT_ROWS=$((PMAT_HUNT_ROWS + n))
}

# Run the pmat audit over a list of source paths the beat just exercised.
# Outputs a compact manifest: top 3 coverage gaps, top 3 churn, top 3 faults per
# path.
#
# Returns 0 when the manifest carried at least one row, 1 (and emit_fail) when
# the header was printed with nothing under it. See "THE MANIFEST IS NO LONGER
# ADVISORY" above. A hunt cut short by a timeout or the budget prints a skip
# line and returns 0: see "THE HUNT IS BOUNDED IN TIME".
#
# Args: <beat-label> <source-path...>
pmat_hunt() {
  local beat="$1"; shift
  if [ "${PMAT_HUNT:-1}" != "1" ] || ! command -v "${PMAT_BIN:-pmat}" >/dev/null 2>&1; then
    return 0
  fi
  printf '    -- pmat bug-hunt manifest (%s) --\n' "$beat"
  # The budget clock starts at the FIRST hunt of the run and spans every beat.
  [ -n "$PMAT_HUNT_DEADLINE" ] || PMAT_HUNT_DEADLINE=$((SECONDS + PMAT_HUNT_BUDGET_S))
  PMAT_HUNT_ROWS=0
  PMAT_HUNT_TIMEOUTS=0
  local paths=$# missing="" q kind rows left t spent=""
  # Each pmat_rows call is ONE physical line. Splitting a command substitution
  # across a backslash continuation makes bashrs read the nested quotes as an
  # unterminated string (SC1078, 6 errors), and FALSIFY-QWEN-STORY-007 requires
  # these scripts to lint clean.
  for q in "$@"; do
    # A path that no longer exists is reported by name. Cause 3 above cost three
    # beats worth of hunting, and "the manifest is empty" never once said so.
    [ -e "$q" ] || missing="$missing $q"
    for kind in gap churn fault; do
      left=$((PMAT_HUNT_DEADLINE - SECONDS))
      if [ "$left" -le 0 ]; then
        spent=1
        break 2
      fi
      t=$PMAT_QUERY_TIMEOUT_S
      [ "$left" -ge "$t" ] || t=$left
      # No free-text query in any of the three: it is a relevance filter applied
      # BEFORE --path, and it collapses a module-scoped hunt to nothing. Cause 2.
      case "$kind" in
        gap)   rows=$(PMAT_ROWS_TIMEOUT_S=$t pmat_rows "$PMAT_FILTER_GAP" --coverage-gaps --path "$q" --rank-by impact --limit 3) ;;
        churn) rows=$(PMAT_ROWS_TIMEOUT_S=$t pmat_rows "$PMAT_FILTER_CHURN" --path "$q" --churn --max-complexity 30 --limit 3) ;;
        fault) rows=$(PMAT_ROWS_TIMEOUT_S=$t pmat_rows "$PMAT_FILTER_FAULT" --path "$q" --faults --exclude-tests --limit 3) ;;
      esac
      _pmat_hunt_emit "$kind" "$rows"
    done
  done
  if [ -n "$spent" ]; then
    printf '        skip  hunt budget PMAT_HUNT_BUDGET_S=%ss spent - the rest of this beat was not hunted\n' "$PMAT_HUNT_BUDGET_S"
  fi
  printf '\n'
  if [ "$PMAT_HUNT_ROWS" -eq 0 ]; then
    # Cut short, not inert: nothing here shows the hunt broken or the code clean.
    if [ -n "$spent" ] || [ "$PMAT_HUNT_TIMEOUTS" -gt 0 ]; then
      return 0
    fi
    if [ -n "$missing" ]; then
      emit_fail "pmat-hunt $beat" "manifest header printed with 0 rows over $paths path(s); these do not exist:$missing"
    else
      emit_fail "pmat-hunt $beat" "manifest header printed with 0 rows over $paths existing path(s) - the hunt is inert (#2356). Check the JSON field names pmat emits, and that --path is not being narrowed by a free-text query."
    fi
    return 1
  fi
  return 0
}
