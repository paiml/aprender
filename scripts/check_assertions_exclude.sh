#!/usr/bin/env bash
#
# check_assertions_exclude.sh — an assertion must EXCLUDE an outcome.
#
# WHY THIS EXISTS (docs/audits/dogfood-0.63.0-hansei.md, 5 Whys)
# --------------------------------------------------------------
# The 0.63.0 dogfood audit found 201 defects in shipped, tested, covered code.
# The root cause was not "we forgot to test". It was that nothing mechanical
# requires an assertion to exclude an outcome. A test that says
#
#     assert!(status == OK || status == BAD_REQUEST || status == NOT_IMPLEMENTED);
#
# passes whatever the endpoint does, earns full line coverage, and survives
# diff-scoped mutation testing. It is a claim about reachability, not about
# behaviour. `/v1/explain` was "covered" by exactly that shape while returning a
# hardcoded 0.95 for every input.
#
# THE RULE
# --------
# An assertion is IN SCOPE if it is `assert!` / `prop_assert!` / `debug_assert!`
# whose expression contains `||` and mentions an outcome token
# (StatusCode::, .status(), is_ok(), is_err(), .success(), .failure(), .code(,
# exit_code). It FAILS if it admits more than one OUTCOME CLASS:
#
#   domain    classes                 fails                     passes
#   HTTP      2xx / 3xx / 4xx / 5xx   OK || BAD_REQUEST         BAD_REQUEST || CONFLICT
#   Result    ok / err                is_ok() || is_err()       is_err() || is_err()
#   process   zero / non-zero exit    code==0 || code==3        code==3 || code==5
#
# A disjunction WITHIN one class stays legal: "the client was told it was wrong
# and I don't care which way" excludes a real outcome. Only "I don't care
# whether it worked" is banned.
#
# BASELINE RATCHET
# ----------------
# Pre-existing debt lives in scripts/assertion_exclusion_baseline.txt as
# `path<TAB>count`. The guard fails if a file exceeds its baseline, or if a file
# NOT in the baseline has any finding. New code is at zero from day one; the
# committed sum can only fall. Regenerate with --update-baseline (which refuses
# to raise a count).
#
# VACUITY GUARD
# -------------
# A guard that silently matches nothing must not pass as clean. This one asserts
# it scanned >= MIN_FILES .rs files and that the scanner still finds >0 sites.
#
# SELF-TEST
# ---------
#   bash scripts/check_assertions_exclude.sh --self-test
# runs a 7-case must-match / must-not-match table (Verification Discipline #7:
# re-run the table, never re-read the pattern).

set -uo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${REPO_ROOT}/scripts/assertion_exclusion_baseline.txt"
AWK_PROG="${REPO_ROOT}/scripts/lib/assertions_exclude.awk"
MIN_FILES="${MIN_FILES:-2000}"

# ---------------------------------------------------------------------------
# The scanner. Emits `path<TAB>line<TAB>domain<TAB>classes` for every in-scope
# assertion that admits more than one outcome class.
#
# Implemented in awk because it must span multiple lines (the motivating site
# spans 26) and must ignore `||` inside string literals.
# ---------------------------------------------------------------------------
scan_awk() {
  awk -f "$AWK_PROG" "$@"
}
# List the .rs files to scan (tracked source only; never target/ or vendored).
list_rs_files() {
  local root="$1"
  if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
    git -C "$root" ls-files -z '*.rs' | tr '\0' '\n' | sed "s|^|${root}/|"
  else
    find "$root" -name '*.rs' -type f -not -path '*/target/*' | sort
  fi
}

# Emit `relpath<TAB>count` for every file with at least one finding.
#
# Never swallows the scanner's stderr: an awk syntax error or an unrecognised
# StatusCode name must abort the run, not silently produce "0 findings".
scan_counts() {
  local root="$1" files raw errf rc
  files="$(list_rs_files "$root")"
  [ -z "$files" ] && return 0
  raw="$(mktemp)"; errf="$(mktemp)"
  printf '%s\n' "$files" | xargs -d '\n' -r -n 200 bash "$SELF" --scan-batch > "$raw" 2>"$errf"
  rc=${PIPESTATUS[1]}
  if [ "$rc" -ne 0 ]; then
    printf 'SCANNER ERROR (xargs rc=%s) - refusing to report a verdict:\n' "$rc" >&2
    sed 's|^|  |' "$errf" >&2
    rm -f "$raw" "$errf"
    exit 2
  fi
  sed "s|^${root}/||" "$raw" \
    | awk -F'\t' '{ c[$1]++ } END { for (f in c) printf("%s\t%s\n", f, c[f]) }' \
    | sort
  rm -f "$raw" "$errf"
}

# ---------------------------------------------------------------------------
# --scan-batch: internal re-entry so xargs can chunk the file list.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--scan-batch" ]; then
  shift
  # Propagate awk's status. A scanner that dies on a syntax error and still
  # exits 0 is exactly the defect class this guard exists to catch — it
  # happened during development and the self-test caught it.
  scan_awk "$@"
  exit $?
fi

# ---------------------------------------------------------------------------
# --self-test: the 7-case table. Cases 1-3 MUST be flagged; 4-7 MUST NOT be.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
  TD="$(mktemp -d)"
  if [ -z "${TD:-}" ] || [ ! -d "$TD" ]; then
    printf 'FAIL: could not create a temp dir for the case table.\n' >&2
    exit 1
  fi
  trap 'rm -rf "${TD:?}"' EXIT

  # Case 1 — the verbatim shape from gpu_warmup.rs that "covered" /v1/explain
  # while it returned a hardcoded 0.95 for every input.
  cat > "$TD/probe1.rs" <<'RS'
#[test]
fn case1() {
    assert!(
        status == StatusCode::OK
            || status == StatusCode::BAD_REQUEST
            || status == StatusCode::NOT_IMPLEMENTED
            || status == StatusCode::SERVICE_UNAVAILABLE,
        "explain endpoint responded"
    );
}
RS
  # Case 2 — the literal tautology.
  cat > "$TD/probe2.rs" <<'RS'
#[test]
fn case2() { assert!(result.is_ok() || result.is_err()); }
RS
  # Case 3 — process exit that admits success and failure.
  cat > "$TD/probe3.rs" <<'RS'
#[test]
fn case3() { assert!(out.status.code() == Some(0) || out.status.code() == Some(5)); }
RS
  # Case 4 — one class (4xx). Legal: it excludes 2xx and 5xx.
  cat > "$TD/probe4.rs" <<'RS'
#[test]
fn case4() { assert!(status == StatusCode::BAD_REQUEST || status == StatusCode::UNPROCESSABLE_ENTITY); }
RS
  # Case 5 — no disjunction at all.
  cat > "$TD/probe5.rs" <<'RS'
#[test]
fn case5() { assert_eq!(status, StatusCode::OK); }
RS
  # Case 6 — a disjunction with no outcome token: out of scope.
  cat > "$TD/probe6.rs" <<'RS'
#[test]
fn case6() { assert!(n == 1 || n == 2); }
RS
  # Case 7 — `||` inside a string literal must not be read as a disjunction.
  cat > "$TD/probe7.rs" <<'RS'
#[test]
fn case7() { assert!(msg.contains("a || b") && result.is_ok(), "status || code"); }
RS

  fails=0
  # Never suppress the scanner's stderr here: the first run of this table
  # reported "3 cases blind" when the real cause was an awk syntax error that
  # `2>/dev/null` had hidden. A silent scanner must fail the table, loudly.
  run_case() {
    CASE_OUT="$(scan_awk "$TD/probe${1}.rs" 2>"$TD/err")"
    CASE_RC=$?
    if [ "$CASE_RC" -ne 0 ]; then
      printf 'FAIL  case %s: scanner exited %s - not a verdict:\n' "$1" "$CASE_RC"
      sed 's|^|        |' "$TD/err"
      return 1
    fi
    return 0
  }
  for c in 1 2 3; do
    if ! run_case "$c"; then fails=$((fails + 1)); continue; fi
    if [ -n "$CASE_OUT" ]; then
      printf 'ok    case %s flagged (must turn RED): %s\n' "$c" "$CASE_OUT"
    else
      printf 'FAIL  case %s NOT flagged - the guard is blind to a real defect shape\n' "$c"
      fails=$((fails + 1))
    fi
  done
  for c in 4 5 6 7; do
    if ! run_case "$c"; then fails=$((fails + 1)); continue; fi
    if [ -z "$CASE_OUT" ]; then
      printf 'ok    case %s clean (must stay GREEN)\n' "$c"
    else
      printf 'FAIL  case %s flagged - false positive: %s\n' "$c" "$CASE_OUT"
      fails=$((fails + 1))
    fi
  done

  if [ "$fails" -ne 0 ]; then
    printf '\nSELF-TEST FAILED (%s/7 cases wrong)\n' "$fails"
    exit 1
  fi
  printf '\nSELF-TEST PASSED (7/7)\n'
  exit 0
fi

# ---------------------------------------------------------------------------
# --update-baseline: regenerate, refusing to RAISE any count.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--update-baseline" ]; then
  new="$(scan_counts "$REPO_ROOT")" || exit 2
  if [ -f "$BASELINE" ]; then
    raised=0
    while IFS=$'\t' read -r f n; do
      [ -z "$f" ] && continue
      case "$f" in \#*) continue ;; esac
      old="$(awk -F'\t' -v k="$f" '$1 == k { print $2 }' "$BASELINE")"
      if [ -n "$old" ] && [ "$n" -gt "$old" ]; then
        printf 'REFUSED: %s would rise %s -> %s. The baseline is a ratchet.\n' "$f" "$old" "$n"
        raised=1
      elif [ -z "$old" ]; then
        printf 'REFUSED: %s is new debt (%s). Fix the assertion instead.\n' "$f" "$n"
        raised=1
      fi
    done <<< "$new"
    [ "$raised" -ne 0 ] && exit 1
  fi
  {
    printf '# assertion_exclusion_baseline.txt - pre-existing debt, monotonically decreasing.\n'
    printf '# Regenerate with: bash scripts/check_assertions_exclude.sh --update-baseline\n'
    printf '# See docs/audits/dogfood-0.63.0-hansei.md for why this file exists.\n'
    printf '%s\n' "$new"
  } > "$BASELINE"
  printf 'baseline updated: %s files, %s sites\n' \
    "$(printf '%s\n' "$new" | grep -c .)" \
    "$(printf '%s\n' "$new" | awk -F'\t' '{s+=$2} END {print s+0}')"
  exit 0
fi

# ---------------------------------------------------------------------------
# Normal mode: scan, ratchet, vacuity.
# ---------------------------------------------------------------------------
printf '=== assertions must EXCLUDE an outcome (check_assertions_exclude.sh) ===\n'

nfiles="$(list_rs_files "$REPO_ROOT" | grep -c .)"
current="$(scan_counts "$REPO_ROOT")" || exit 2
nsites="$(printf '%s\n' "$current" | awk -F'\t' '{s+=$2} END {print s+0}')"

# --- vacuity: a guard that matched nothing must not report clean -----------
if [ "$nfiles" -lt "$MIN_FILES" ]; then
  printf 'FAIL (vacuity): scanned only %s .rs files, expected >= %s.\n' "$nfiles" "$MIN_FILES"
  printf '  The scanner found nothing to look at. Fix the file list, not this number.\n'
  exit 1
fi
if [ "$nsites" -eq 0 ] && [ -s "$BASELINE" ]; then
  printf 'FAIL (vacuity): scanner found 0 sites while the baseline is non-empty.\n'
  printf '  Either every site was fixed (then run --update-baseline) or the scanner broke.\n'
  exit 1
fi

# THE RATCHET IS A PROPERTY OF THE DIFF, NOT OF THE TREE.
#
# Everything above compares the scan against the baseline AS IT STANDS IN THE
# WORKING TREE, and that is not a ratchet. NEW (a finding with no entry) and
# STALE (an entry with no finding) are the only two properties a working tree
# can answer, and a commit that appends one line AND lands the matching
# violation satisfies both at once: not new, because it is baselined; not
# stale, because the finding is real.
#
# Measured, not argued: appending one entry cloned from this file's own last
# real entry returned rc=0 from this guard, under its own words:
#     "the committed sum can only fall"
# Twelve guards in scripts/ failed the same probe.
#
# So growth is now compared against merge-base(HEAD, origin/main), falling
# back to the origin/main TIP because CI checks out shallow — a ref this
# branch cannot rewrite, and never the branch against itself.
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
baseline_ratchet_check "${REPO_ROOT}" scripts/assertion_exclusion_baseline.txt keyed || exit 1

if [ ! -f "$BASELINE" ]; then
  printf 'FAIL: %s is missing. Create it with --update-baseline.\n' "$BASELINE"
  exit 1
fi

violations=0
base_sum="$(awk -F'\t' '!/^#/ && NF==2 {s+=$2} END {print s+0}' "$BASELINE")"

while IFS=$'\t' read -r f n; do
  [ -z "$f" ] && continue
  old="$(awk -F'\t' -v k="$f" '!/^#/ && $1 == k { print $2 }' "$BASELINE")"
  if [ -z "$old" ]; then
    printf '\nFAIL %s: %s assertion(s) that admit more than one outcome class, and this file\n' "$f" "$n"
    printf '     is not in the baseline. New code starts at zero.\n'
    bash "$SELF" --scan-batch "$REPO_ROOT/$f" 2>/dev/null | sed 's|^|       |'
    violations=$((violations + 1))
  elif [ "$n" -gt "$old" ]; then
    printf '\nFAIL %s: %s sites, baseline %s. The ratchet only turns one way.\n' "$f" "$n" "$old"
    bash "$SELF" --scan-batch "$REPO_ROOT/$f" 2>/dev/null | sed 's|^|       |'
    violations=$((violations + 1))
  fi
done <<< "$current"

printf '\nscanned %s .rs files; %s site(s) admit more than one outcome class (baseline sum %s, delta %s)\n' \
  "$nfiles" "$nsites" "$base_sum" "$((nsites - base_sum))"

if [ "$violations" -ne 0 ]; then
  printf '\n%s file(s) violated the ratchet.\n' "$violations"
  printf 'An assertion that admits both success and failure is a reachability claim, not a\n'
  printf 'behaviour claim. State what the code must reject. See\n'
  printf 'docs/audits/dogfood-0.63.0-hansei.md.\n'
  exit 1
fi

printf 'PASS: no file exceeds its baseline.\n'
exit 0
