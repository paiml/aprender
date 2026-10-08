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
# ONE QUERY PER DIRECTORY, NOT PER PATH (#4715)
# ---------------------------------------------
# pmat 3.42.0 rebuilds its whole function index on every `pmat query`: about
# 53 s on the story's runner and 92 s on a dev box, for a 94,407-function
# index (paiml/paiml-mcp-agent-toolkit#1481). So what this hunt costs is the
# NUMBER of queries, not their scope. Two per hunted path over 21 paths, plus
# the gap queries, kept qwen-story-daily inside Beat 4 when its 30-minute
# limit cancelled it, every night from 2026-09-29. Churn and fault rows now
# come from one query per (parent directory, flag set), cached for the story
# run and cut per path in jq: 10 queries instead of 42.
#
# For churn the cut is exact, because of how pmat answers a query with no
# free-text words (`browse_all`, src/services/agent_context/query/
# engine_query.rs, pmat 3.42.0). --path, a substring test when it holds no `*`,
# and --max-complexity filter the index BEFORE a stable sort by pagerank and
# BEFORE --limit. Every file_path that contains a hunted path also contains
# that path's parent directory. So the directory's list, filtered to the path,
# is the path's own list in the same order, and its first three records are
# what `--path P --limit 3` returns.
#
# For faults the cut is a superset of the old rows. --exclude-tests is a
# post-filter that pmat applies AFTER --limit
# (src/cli/handlers/query_handler/options.rs `apply_result_filters`). So
# `--limit 3` on a path whose top three held a test function printed fewer
# than three rows. The first three non-test records keep every row the old
# query printed, because those were the first non-test records in the same
# order, and fill the rest.
#
# Gap queries stay one per path, but a path the coverage file does not
# contain is not asked at all. coverage-nightly's report scope leaves
# crates/apr-cli/ out (COVERAGE_EXCLUDE_REGEX in the Makefile), and pmat
# answers such a path with "No coverage gaps found (100% coverage or no data)"
# and an empty stdout. That verdict was already not_measured. It stays
# not_measured, now with its real reason and without spending a query.

# OPTION NEUTRALITY (load-bearing)
# --------------------------------
# This file is SOURCED. It must not run `set`: that mutates the CALLER's shell.
# apr_bin.sh once opened with `set -euo pipefail`, qwen-story.sh sourced it, and
# the leaked errexit killed the nightly six lines in - inside this very hunt.
# Sourceable libraries here fail by RETURN STATUS. Enforced by
# scripts/check_sourced_libs_option_neutral.sh.

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
# Callers read rows through `$(...)`, a subshell, so PMAT_ROWS_EC never reaches
# them; the return status is the only channel out. With PMAT_ROWS_STRICT=1 a
# query that was not measured says so: rc 2 when pmat exits non-zero, rc 3 when
# its output is empty or not JSON. A parsed `[]` (or the no-match document
# shape) is a measured zero, rc 0. Without it the call stays tolerant (rc 0).
#
# Args:   <jq-output-filter> <pmat query args...>
# Sets:   PMAT_ROWS_EC  exit status of the `pmat query` invocation
# Prints: at most 3 formatted rows (empty output when there are none)
pmat_rows() {
  local filter="$1"; shift
  local raw prog rows jq_ec strict=0
  [ "${PMAT_ROWS_STRICT:-0}" != "1" ] || strict=2
  raw=$("${PMAT_BIN:-pmat}" query "$@" --format json 2>/dev/null)
  PMAT_ROWS_EC=$?
  [ "$PMAT_ROWS_EC" -eq 0 ] || return "$strict"
  if [ -z "$raw" ] && [ "$strict" -ne 0 ]; then return 3; fi
  # printf rather than a multi-line double-quoted string: bashrs reads the
  # latter as an unterminated string (SC1078).
  prog=$(printf '%s\n%s\n%s' \
    'if type == "array" then .[] else empty end' \
    '| select(.function_name != null)' \
    "| $filter")
  rows=$(printf '%s\n' "$raw" | jq -r "$prog" 2>/dev/null)
  jq_ec=$?
  if [ "$jq_ec" -ne 0 ] && [ "$strict" -ne 0 ]; then return 3; fi
  [ -n "$rows" ] || return 0
  printf '%s\n' "$rows" | head -3
}

# A directory query whose record count reaches this limit may be truncated,
# and then the cut falls back to the per-path query.
PMAT_HUNT_DIR_LIMIT="${PMAT_HUNT_DIR_LIMIT:-1000000}"

# Rows for one hunted path, cut from ONE cached `pmat query` over the path's
# parent directory. See "ONE QUERY PER DIRECTORY, NOT PER PATH" above.
#
# The cache lives in PMAT_HUNT_CACHE_DIR, keyed by directory, flags and
# PMAT_HUNT_HEAD, and records pmat's exit status beside its JSON, so a failed
# directory query fails every path under it once rather than being re-run per
# path. As with pmat_rows outside strict mode, a failed or non-JSON query
# yields no rows, and the zero-row andon in pmat_hunt decides. With no cache
# directory, or a directory answer that reached PMAT_HUNT_DIR_LIMIT, this is
# exactly the per-path pmat_rows call it replaces.
#
# Args:   <jq-output-filter> <path> <pmat flags, without --path/--limit/--format...>
# Prints: at most 3 formatted rows
pmat_rows_cached() {
  local filter="$1" q="$2"; shift 2
  local dir key cache n prog
  dir="${q%/}"
  dir="${dir%/*}"
  if [ -z "${PMAT_HUNT_CACHE_DIR:-}" ] || [ ! -d "${PMAT_HUNT_CACHE_DIR:-}" ]; then
    pmat_rows "$filter" --path "$q" "$@" --limit 3
    return
  fi
  # A hash, so two directories or flag sets can never share a file.
  key=$(printf '%s\n%s\n%s' "$dir" "$*" "${PMAT_HUNT_HEAD:-}" | sha256sum | cut -c1-32)
  if [ "${#key}" -ne 32 ]; then
    pmat_rows "$filter" --path "$q" "$@" --limit 3
    return
  fi
  cache="$PMAT_HUNT_CACHE_DIR/$key"
  if [ ! -f "$cache.ec" ]; then
    "${PMAT_BIN:-pmat}" query --path "$dir" "$@" --limit "$PMAT_HUNT_DIR_LIMIT" --format json >"$cache.json" 2>/dev/null
    printf '%s\n' "$?" >"$cache.ec"
    jq 'if type == "array" then length else 0 end' "$cache.json" >"$cache.n" 2>/dev/null || : >"$cache.n"
  fi
  [ "$(cat "$cache.ec")" = "0" ] || return 0
  n=$(cat "$cache.n")
  [ -n "$n" ] || return 0
  if [ "$n" -ge "$PMAT_HUNT_DIR_LIMIT" ]; then
    pmat_rows "$filter" --path "$q" "$@" --limit 3
    return
  fi
  # `contains` on two strings is jq's substring test, the same test pmat's
  # --path applies. [0:3] is the per-path --limit 3, taken BEFORE the row
  # filter, exactly as pmat_rows formats the three records pmat returned.
  prog=$(printf '%s\n%s\n%s\n%s' \
    '[ if type == "array" then .[] else empty end | select((.file_path // "") | contains($q)) ]' \
    '| .[0:3][]' \
    '| select(.function_name != null)' \
    "| $filter")
  jq -r --arg q "$q" "$prog" "$cache.json" 2>/dev/null
}

# The coverage file's source paths, extracted once per cache directory. Prints
# the list's path, or nothing when the list cannot be built (no cache
# directory, a file jq cannot read as LLVM coverage JSON, or one that names no
# source file). Then pmat_hunt asks pmat as it did before.
_pmat_cov_list() {
  if [ -z "${PMAT_HUNT_CACHE_DIR:-}" ] || [ ! -d "${PMAT_HUNT_CACHE_DIR:-}" ]; then
    return 0
  fi
  local list key
  key=$(printf '%s\n%s' "${STORY_COVERAGE_FILE:-}" "${PMAT_HUNT_HEAD:-}" | sha256sum | cut -c1-32)
  [ "${#key}" -eq 32 ] || return 0
  list="$PMAT_HUNT_CACHE_DIR/coverage-files.$key"
  if [ ! -f "$list.ec" ]; then
    jq -r '.data[]?.files[]?.filename // empty' "${STORY_COVERAGE_FILE:-}" >"$list" 2>/dev/null
    printf '%s\n' "$?" >"$list.ec"
  fi
  [ "$(cat "$list.ec")" = "0" ] || return 0
  # A file that names no source file is a shape this reader does not know,
  # not proof that every path is out of scope: pmat is asked instead.
  [ -s "$list" ] || return 0
  printf '%s\n' "$list"
}

# Print one block of rows and add its line count to PMAT_HUNT_ROWS.
_pmat_hunt_emit() {
  [ -n "$1" ] || return 0
  printf '%s\n' "$1"
  local n
  n=$(printf '%s\n' "$1" | grep -c .)
  PMAT_HUNT_ROWS=$((PMAT_HUNT_ROWS + n))
}

# Run the pmat audit over a list of source paths the beat just exercised.
# Outputs a compact manifest: top 3 coverage gaps, top 3 churn, top 3 faults per
# path.
#
# Returns 0 when the manifest carried at least one row and the coverage gaps
# were measured, 1 (and emit_fail) when the header was printed with nothing
# under it or when the coverage gaps were not measured. See "THE MANIFEST IS NO
# LONGER ADVISORY" above.
#
# COVERAGE GAPS READ A COVERAGE FILE, NEVER A FRESH RUN (#4715). Without
# --coverage-file, `pmat query --coverage-gaps` derives coverage itself, and
# measured on intel one such query ran out its 900 s timeout. Every beat runs
# one per hunted path, so qwen-story-daily hit its job timeout and was
# cancelled each night. The gaps now come only from STORY_COVERAGE_FILE (an
# LLVM coverage JSON from coverage-nightly), and only when STORY_COVERAGE_SHA,
# the commit that file was measured on, is this checkout's HEAD. In every
# other case the gaps are not_measured, and not_measured fails the beat exactly
# as an empty manifest does. A gap list measured on another commit is not a
# measurement of this one, and silently dropping the gap block would turn a
# red hunt green with nothing measured.
#
# Args: <beat-label> <source-path...>
pmat_hunt() {
  local beat="$1"; shift
  if [ "${PMAT_HUNT:-1}" != "1" ] || ! command -v "${PMAT_BIN:-pmat}" >/dev/null 2>&1; then
    return 0
  fi
  printf '    -- pmat bug-hunt manifest (%s) --\n' "$beat"
  PMAT_HUNT_ROWS=0
  local paths=$# missing="" q gaps gap_ec churn faults cov_why="" head own_cache="" cov_list=""
  head=$(git rev-parse HEAD 2>/dev/null)
  PMAT_HUNT_HEAD="$head"
  # The story sets PMAT_HUNT_CACHE_DIR once so every beat shares one cache.
  # Without it, this call keeps a private cache and removes it when done.
  if [ -z "${PMAT_HUNT_CACHE_DIR:-}" ]; then
    own_cache=$(mktemp -d "${TMPDIR:-/tmp}/pmat-hunt.XXXXXX" 2>/dev/null) || own_cache=""
    PMAT_HUNT_CACHE_DIR="$own_cache"
  else
    mkdir -p "$PMAT_HUNT_CACHE_DIR" 2>/dev/null
  fi
  if [ -z "${STORY_COVERAGE_FILE:-}" ] || [ ! -s "${STORY_COVERAGE_FILE:-}" ]; then
    cov_why="no coverage file (STORY_COVERAGE_FILE='${STORY_COVERAGE_FILE:-}' is unset, missing or empty)"
  elif [ -z "$head" ] || [ "${STORY_COVERAGE_SHA:-}" != "$head" ]; then
    cov_why="the coverage file was measured on '${STORY_COVERAGE_SHA:-(none recorded)}', this checkout is '${head:-(unknown)}'"
  else
    cov_list=$(_pmat_cov_list)
  fi
  # Each pmat_rows call is ONE physical line. Splitting a command substitution
  # across a backslash continuation makes bashrs read the nested quotes as an
  # unterminated string (SC1078, 6 errors), and FALSIFY-QWEN-STORY-007 requires
  # these scripts to lint clean.
  for q in "$@"; do
    # A path that no longer exists is reported by name. Cause 3 above cost three
    # beats worth of hunting, and "the manifest is empty" never once said so.
    [ -e "$q" ] || missing="$missing $q"
    # No free-text query in any of the three: it is a relevance filter applied
    # BEFORE --path, and it collapses a module-scoped hunt to nothing. Cause 2.
    # A gap query that fails or prints no JSON (a coverage file pmat cannot
    # parse, a truncated artifact) is not_measured too, never an empty block.
    gaps=""
    if [ -z "$cov_why" ] && [ -n "$cov_list" ] && ! grep -qF -- "$q" "$cov_list"; then
      cov_why="$q is outside the coverage file's scope (none of its $(grep -c . "$cov_list") files is under it)"
    fi
    if [ -z "$cov_why" ]; then
      gaps=$(PMAT_ROWS_STRICT=1 pmat_rows "$PMAT_FILTER_GAP" --coverage-gaps --coverage-file "${STORY_COVERAGE_FILE:-}" --path "$q" --rank-by impact --limit 3) && gap_ec=0 || gap_ec=$?
      case "$gap_ec" in
        0) ;;
        2) gaps=""; cov_why="the coverage-gaps query failed on $q (pmat exited non-zero)" ;;
        *) gaps=""; cov_why="the coverage-gaps query on $q printed no JSON (rc $gap_ec)" ;;
      esac
    fi
    churn=$(pmat_rows_cached "$PMAT_FILTER_CHURN" "$q" --churn --max-complexity 30)
    faults=$(pmat_rows_cached "$PMAT_FILTER_FAULT" "$q" --faults --exclude-tests)
    _pmat_hunt_emit "$gaps"
    _pmat_hunt_emit "$churn"
    _pmat_hunt_emit "$faults"
  done
  if [ -n "$own_cache" ]; then
    rm -rf "${own_cache:?}"
    PMAT_HUNT_CACHE_DIR=""
  fi
  printf '\n'
  if [ "$PMAT_HUNT_ROWS" -eq 0 ]; then
    if [ -n "$missing" ]; then
      emit_fail "pmat-hunt $beat" "manifest header printed with 0 rows over $paths path(s); these do not exist:$missing"
    else
      emit_fail "pmat-hunt $beat" "manifest header printed with 0 rows over $paths existing path(s) - the hunt is inert (#2356). Check the JSON field names pmat emits, and that --path is not being narrowed by a free-text query."
    fi
    return 1
  fi
  if [ -n "$cov_why" ]; then
    # Not indented like a row: the workflow's manifest extractor counts lines
    # that start with whitespace + gap/churn/fault.
    printf '    -- coverage gaps not_measured: %s --\n\n' "$cov_why"
    emit_fail "pmat-hunt $beat" "coverage gaps not_measured: $cov_why"
    return 1
  fi
  return 0
}
