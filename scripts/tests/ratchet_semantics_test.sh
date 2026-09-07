#!/usr/bin/env bash
# ratchet_semantics_test.sh — the polarity table of the D2 ratchet normaliser,
# executed on a throwaway git repository (BSE-03 phase A,
# docs/specifications/build-system-enhancement.md, `paiml/infra` §4 wave 4;
# Pmat-Ticket: PMAT-1068).
#
# The model this exercises is docs/audits/threat-model-bse-03-ratchets.md:
# a ratchet verdict is a function of (comparand SHA, merge SHA) and of nothing
# on disk. Phase A covers ONE class, `readme` — the contract-count claim:
#
#   * scripts/readme_sync.sh          the generator that makes the count DERIVED
#   * scripts/check_readme_claims.sh  the D2 comparand diff that judges it
#
# Rows (`--class readme`), each asserting an exit code AND a named verdict
# line — never a substring the banner already prints:
#
#   1  block states the truth                      -> GREEN
#   2  block hand-edited to a wrong number         -> RED, BOTH numbers printed
#   3  authored literal, no markers, overstating    -> RED, BOTH numbers printed
#   4  no block and no literal (count is derived)   -> GREEN, and the bytes
#      readme_sync.sh would write are printed and are stable across two runs
#   5  readme_sync.sh --write is idempotent         (byte-identical 2nd run)
#   6  comparand unresolvable                       -> RED at PREFLIGHT, and the
#      measurement line is ABSENT (the failure precedes the measurement)
#   7  the count FELL vs the comparand              -> GREEN (the verdict is
#      about truth, not direction)
#   8  MUTATION — a temp copy of check_readme_claims.sh whose merge-tree
#      measurement reads a number from a FILE ON DISK instead of from the
#      checkout must turn rows 2 and 3 GREEN. If it does not, rows 2/3 are not
#      measuring the property and this suite is vacuous.
#
# Phase B (BSE-03, this commit) adds the other two classes.
#
# `--class complexity` — scripts/check_complexity_ratchet.sh, on a throwaway git
# repository with a BASE commit and a merge commit, and a `pmat` SHIM on PATH.
# The shim is not an analyser: it reports the numbers each .rs file declares in
# `// CX <fn> <cyclomatic> <cognitive>` and the version the tree declares in
# `// PMAT-VERSION <v>`, so every measurement is a pure function of the revision
# measured — which is the property under test — and the pair costs milliseconds
# instead of 18 s. Rows:
#
#   C1  base and merge measure identically              -> GREEN
#   C2  a function crosses a threshold in the merge     -> RED, named NEW
#   C3  a recorded function is FIXED, baseline file left
#       exactly as the comparand carries it             -> GREEN, named RESOLVED
#       (this is the merge-commit failure the ticket exists to remove: the
#       pre-BSE-03 verdict calls it STALE and reds)
#   C4  the comparand ref cannot be resolved            -> RED at PREFLIGHT, and
#       NO measurement line is printed (the failure precedes the scan)
#   C5  the two revisions measure under DIFFERENT pmat
#       versions (attack A6)                            -> RED, BOTH versions
#   C6  the recorded `# tool_version=` header disagrees
#       with the pmat that ran                          -> RED, BOTH versions
#   C7  MUTATION — a temp copy of the guard whose verdict line is replaced by
#       the pre-BSE-03 `cx_verdict "$BASELINE" …` (a LOWER BOUND, read from a
#       FILE ON DISK) must turn C3 RED. If it does not, C3 is not measuring the
#       property and this class is vacuous.
#
# `--class satd` — F-CHECKLIST-005 / F-DOD-001 in
# crates/aprender-core/tests/includes/falsification_spec_v10_checklist.rs. A
# throwaway crate is too heavy, so the rows EXECUTE THE ARTIFACT: the test
# binary is built once (`--features model-tests`, without which this suite is
# dark) and then run per row with a different `SATD_BASELINE`. The marker count
# N is READ OUT of the instrument's own failure text rather than pinned here, so
# no number in this file goes stale when debt is paid down.
#
#   S1  the parsing/fallback unit test                  -> GREEN
#   S2  SATD_BASELINE unset                             -> GREEN, "source: constant"
#   S3  ceiling N-1 (a regression)                      -> RED, names N and the
#       ceiling and "source: comparand"
#   S4  ceiling N (at the ceiling)                      -> GREEN
#   S5  ceiling N+20 (the comparand carried more debt — an IMPROVEMENT) -> GREEN.
#       This is the row the removed lower bound redded: the pre-BSE-03 code
#       asserted `violations.len() + 8 >= baseline`, and N + 8 >= N + 20 is
#       false. The lower-bound MUTANT itself is executed in --class complexity
#       (C7), where reintroducing it costs a sed rather than a rebuild.
#   S6  SATD_BASELINE set but EMPTY                     -> RED, "set but empty"
#       (the comparand measurement did not happen, and a silent fall back to the
#       constant would be a verdict from a measurement nobody made)
#   S7  SATD_BASELINE unparseable                       -> RED, names the variable
#
# Prints "N checks, M failed" and exits 1 if M > 0. A fixture that cannot be
# built is a FAILURE, not a skip.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GUARD="$ROOT/scripts/check_readme_claims.sh"
SYNC="$ROOT/scripts/readme_sync.sh"
START='<!-- CONTRACT_COUNT_START -->'
END='<!-- CONTRACT_COUNT_END -->'

CHECKS=0
FAILED=0

ok()   { CHECKS=$((CHECKS + 1)); printf 'ok    %s\n' "$1"; }
bad()  { CHECKS=$((CHECKS + 1)); FAILED=$((FAILED + 1)); printf 'FAIL  %s\n' "$1"; }

die() { printf 'FAIL  fixture: %s\n' "$1"; printf '\n%s checks, %s failed\n' "$((CHECKS + 1))" "$((FAILED + 1))"; exit 1; }

usage() {
  cat >&2 <<'USAGE'
usage: bash scripts/tests/ratchet_semantics_test.sh --class <readme|complexity|satd>
  --class readme      the README contract-count ratchet (BSE-03 phase A)
  --class complexity  scripts/check_complexity_ratchet.sh, D2 comparand diff (phase B)
  --class satd        F-CHECKLIST-005 / F-DOD-001, SATD_BASELINE from the comparand (phase B)
USAGE
  exit 2
}

finish() { printf '\n%s checks, %s failed\n' "$CHECKS" "$FAILED"; [ "$FAILED" -eq 0 ]; }

CLASS=""
while [ $# -gt 0 ]; do
  case "$1" in
    --class) [ $# -ge 2 ] || usage; CLASS="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) printf 'unknown arg: %s\n' "$1" >&2; usage ;;
  esac
done
[ -n "$CLASS" ] || usage

case "$CLASS" in
  readme | complexity | satd) : ;;
  *) printf 'unknown class: %s\n' "$CLASS" >&2; usage ;;
esac

if [ "$CLASS" = 'readme' ]; then
  [ -f "$GUARD" ] || die "missing $GUARD"
  [ -f "$SYNC" ]  || die "missing $SYNC — the generator that makes the count derived (BSE-03 decision 1)"
fi
CXGUARD="$ROOT/scripts/check_complexity_ratchet.sh"
if [ "$CLASS" = 'complexity' ]; then
  [ -f "$CXGUARD" ] || die "missing $CXGUARD"
fi

WORK="$(mktemp -d -t ratchet_semantics_test.XXXXXX)" || die "mktemp -d failed"
trap 'rm -rf "${WORK:?}"' EXIT
mkdir -p "$WORK/empty-template" || die "cannot create the empty git template dir"

# --- fixture ---------------------------------------------------------------
# A throwaway repository, the pattern scripts/check_baseline_ratchets.sh:213
# uses: pinned identity, no signing, and an EMPTY template so this machine's
# global hooks/templates cannot reach into the fixture.
git_fx() { GIT_TERMINAL_PROMPT=0 git -C "$1" -c commit.gpgsign=false -c user.email=t@example.com -c user.name=t "${@:2}"; }

fixture() { # fixture <dir> <n contracts> <readme body>
  local dir="$1" n="$2" body="$3" i
  mkdir -p "$dir/contracts" "$dir/scripts" || return 1
  for i in $(seq 1 "$n"); do printf 'id: c%s\n' "$i" > "$dir/contracts/c$i.yaml" || return 1; done
  printf '%s\n' "$body" > "$dir/README.md" || return 1
  # The guard and the generator are copied IN, so REPO_ROOT resolves to the
  # fixture (they derive it from their own location) and the mutant copy below
  # sits beside them with the same resolution.
  cp "$GUARD" "$SYNC" "$ROOT/scripts/cargo_classify.sh" "$ROOT/scripts/lib_baseline_ratchet.sh" "$dir/scripts/" || return 1
  cp -R "$ROOT/scripts/lib" "$dir/scripts/lib" || return 1
  GIT_TERMINAL_PROMPT=0 git init -q --template="$WORK/empty-template" "$dir" >/dev/null 2>&1 || return 1
  git_fx "$dir" add -A -f contracts README.md >/dev/null 2>&1 || return 1
  git_fx "$dir" commit -q -m "fixture: $n contracts" >/dev/null 2>&1 || return 1
  git_fx "$dir" update-ref refs/remotes/origin/main HEAD || return 1
}

readme_with_block() { # readme_with_block <claimed>
  printf '# fixture\n\n| Metric | Count | Source |\n|---|---|---|\n| Provable contracts | **%s%s%s** provable contracts | `find contracts/ -name '"'"'*.yaml'"'"'` |\n' \
    "$START" "$1" "$END"
}
readme_with_literal() { printf '# fixture\n\nThis tree carries %s provable contracts.\n' "$1"; }
readme_derived()      { printf '# fixture\n\nThe contract count is derived; this file states no literal.\n'; }

run_guard() { # run_guard <fixture dir> <script name> [env assignments...]
  local dir="$1" script="$2"; shift 2
  ( cd "$dir" && env "$@" bash "$dir/scripts/$script" --claim contract_count ) 2>&1
}

# ===========================================================================
# --class complexity
# ===========================================================================

# The shim. Written once per run into $WORK/bin and put FIRST on PATH.
cx_write_shim() { # cx_write_shim <bindir>
  mkdir -p "$1" || return 1
  cat > "$1/pmat" <<'SHIM'
#!/usr/bin/env bash
# A `pmat` stand-in for the ratchet fixture (scripts/tests/ratchet_semantics_test.sh).
# It analyses nothing: it reports what the .rs files it is handed DECLARE, in
#   // CX <function> <cyclomatic> <cognitive>
# and the version the tree declares in
#   // PMAT-VERSION <version>
# so a measurement is a pure function of the revision being measured. That is
# the property the D2 normaliser is under test for, and it makes two full
# measurements cost milliseconds. It emits the two keys complexity_rows.py
# reads: files_analyzed and files[].functions[].metrics.{cyclomatic,cognitive}.
set -uo pipefail
shim_version() {
  local v
  v=$(grep -rh '^// PMAT-VERSION ' --include='*.rs' . 2>/dev/null | awk 'NR == 1 { print $3 }')
  if [ -z "$v" ]; then v='0.0.0-shim'; fi
  printf '%s\n' "$v"
}
if [ "${1:-}" = '--version' ]; then
  printf 'pmat %s\n' "$(shim_version)"
  exit 0
fi
files=''
while [ $# -gt 0 ]; do
  case "$1" in
    --files) files="${2:-}"; shift 2 ;;
    *) shift ;;
  esac
done
count=$(printf '%s' "$files" | tr ',' '\n' | grep -c . || true)
printf '{ "files_analyzed": %s, "files": [' "$count"
first=1
for f in $(printf '%s' "$files" | tr ',' ' '); do
  if [ "$first" -eq 0 ]; then printf ','; fi
  first=0
  printf '{"path": "%s", "functions": [' "$f"
  awk '/^\/\/ CX / { if (n++) printf(","); printf("{\"name\": \"%s\", \"metrics\": {\"cyclomatic\": %s, \"cognitive\": %s}}", $3, $4, $5) }' "$f"
  printf ']}'
done
printf '] }\n'
SHIM
  chmod +x "$1/pmat" || return 1
  [ -x "$1/pmat" ]
}

cx_lib() { # cx_lib <declared pmat version> <"<fn> <cyc> <cog>">... -> a src/lib.rs
  local v="$1" row name
  shift
  printf '// PMAT-VERSION %s\n' "$v"
  for row in "$@"; do
    name=$(printf '%s' "$row" | awk '{ print $1 }')
    printf '// CX %s\n' "$row"
    printf 'pub fn %s() {}\n' "$name"
  done
}

cx_commit() { # cx_commit <dir> <message>
  git_fx "$1" add -A -f . >/dev/null 2>&1 || return 1
  git_fx "$1" commit -q -m "$2" >/dev/null 2>&1 || return 1
}

# The BASE commit, and origin/main pinned to it so the real (non-overridden)
# resolver path is exercised and BOOTSTRAP stays unreachable.
cx_fixture() { # cx_fixture <dir> [recorded tool_version]
  local dir="$1" recorded="${2:-pmat 0.0.0-shim}"
  mkdir -p "$dir/scripts/lib" "$dir/src" || return 1
  cp "$CXGUARD" "$ROOT/scripts/lib_baseline_ratchet.sh" "$dir/scripts/" || return 1
  cp "$ROOT/scripts/lib/complexity_rows.py" "$dir/scripts/lib/" || return 1
  cx_lib '0.0.0-shim' 'branchy 35 34' 'cognitive_only 8 28' 'tidy 1 0' > "$dir/src/lib.rs" || return 1
  { printf '# complexity_baseline.txt — fixture inventory, NOT the comparand\n'
    printf '# tool_version=%s\n' "$recorded"
    printf 'src/lib.rs::branchy 35 34\n'
    printf 'src/lib.rs::cognitive_only 8 28\n'; } > "$dir/scripts/complexity_baseline.txt" || return 1
  printf '609\n' > "$dir/scripts/cb200_baseline.txt" || return 1
  printf '[tdg]\nbaseline = 609\n' > "$dir/.pmat-gates.toml" || return 1
  GIT_TERMINAL_PROMPT=0 git init -q --template="$WORK/empty-template" "$dir" >/dev/null 2>&1 || return 1
  cx_commit "$dir" 'fixture: base' || return 1
  git_fx "$dir" update-ref refs/remotes/origin/main HEAD || return 1
}

cx_run() { # cx_run <dir> <script name> [VAR=value...]
  local dir="$1" script="$2"
  shift 2
  ( cd "$dir" && PATH="$WORK/bin:$PATH" env CX_MIN_RS_FILES=1 "$@" \
      bash "$dir/scripts/$script" ) 2>&1
}

run_complexity() {
  cx_write_shim "$WORK/bin" || die "the pmat shim could not be written"
  command -v python3 >/dev/null 2>&1 || die "python3 is required: the guard reads pmat JSON with scripts/lib/complexity_rows.py"

  local out rc

  # --- C1: nothing changed ------------------------------------------------
  local C1="$WORK/cx1"
  cx_fixture "$C1" || die "C1 fixture could not be built"
  cx_lib '0.0.0-shim' 'branchy 35 34' 'cognitive_only 8 28' 'tidy 2 0' > "$C1/src/lib.rs" || die "C1: cannot rewrite src/lib.rs"
  cx_commit "$C1" 'merge: an unrelated edit under both thresholds' || die "C1: commit failed"
  out=$(cx_run "$C1" check_complexity_ratchet.sh); rc=$?
  if [ "$rc" -eq 0 ] \
     && printf '%s' "$out" | grep -qE '^  measured    base .*over cyclomatic>30' \
     && printf '%s' "$out" | grep -qE '^  measured    merge .*over cyclomatic>30' \
     && printf '%s' "$out" | grep -q '^PASS (D2)'; then
    ok "C1  base and merge measure the same offenders: GREEN, both measurements printed"
  else
    bad "C1  no change: rc=$rc, wanted 0, both 'measured' lines and a 'PASS (D2)' line"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi
  # and the two SHAs are printed BEFORE the verdict, which is what makes the
  # verdict auditable at all
  if printf '%s' "$out" | grep -qE '^  comparand   (MERGEBASE|TIP) ' \
     && printf '%s' "$out" | grep -qE '^  merge       HEAD ' \
     && printf '%s' "$out" | grep -q 'polarity    the verdict is about the DIFF of two measurements'; then
    ok "C1  the D2 header names the comparand mode, both revisions and the polarity"
  else
    bad "C1  D2 header missing: wanted a 'comparand', a 'merge' and a 'polarity' line"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # --- C2: a regression ---------------------------------------------------
  local C2="$WORK/cx2"
  cx_fixture "$C2" || die "C2 fixture could not be built"
  cx_lib '0.0.0-shim' 'branchy 35 34' 'cognitive_only 8 28' 'tidy 1 0' 'newbie 40 40' > "$C2/src/lib.rs" || die "C2: cannot rewrite src/lib.rs"
  cx_commit "$C2" 'merge: land a function over both thresholds' || die "C2: commit failed"
  out=$(cx_run "$C2" check_complexity_ratchet.sh); rc=$?
  if [ "$rc" -eq 1 ] \
     && printf '%s' "$out" | grep -q 'RED    NEW      src/lib.rs::newbie' \
     && printf '%s' "$out" | grep -q 'complexity regressed against'; then
    ok "C2  a function over a threshold and absent from the comparand: RED, named NEW"
  else
    bad "C2  regression: rc=$rc, wanted 1 and a 'RED    NEW      src/lib.rs::newbie' finding"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # --- C3: an improvement, with the baseline FILE left alone --------------
  # This is the merge-commit shape: nobody rewrites scripts/complexity_baseline.txt
  # on a merge, so it still carries the row for a function that is now fixed.
  local C3="$WORK/cx3"
  cx_fixture "$C3" || die "C3 fixture could not be built"
  cx_lib '0.0.0-shim' 'branchy 12 10' 'cognitive_only 8 28' 'tidy 1 0' > "$C3/src/lib.rs" || die "C3: cannot rewrite src/lib.rs"
  cx_commit "$C3" 'merge: fix branchy, leave the baseline file untouched' || die "C3: commit failed"
  out=$(cx_run "$C3" check_complexity_ratchet.sh); rc=$?
  if [ "$rc" -eq 0 ] \
     && printf '%s' "$out" | grep -q 'NOTE   RESOLVED src/lib.rs::branchy' \
     && grep -q '^src/lib.rs::branchy 35 34$' "$C3/scripts/complexity_baseline.txt"; then
    ok "C3  a recorded function fixed, its row still in the baseline file: GREEN, named RESOLVED"
  else
    bad "C3  improvement: rc=$rc, wanted 0 and a 'NOTE   RESOLVED src/lib.rs::branchy' line, with the row still on disk"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # --- C4: the comparand cannot be resolved -------------------------------
  out=$(cx_run "$C3" check_complexity_ratchet.sh BASELINE_RATCHET_BASE_REF=refs/heads/no-such-xyzzy); rc=$?
  if [ "$rc" -eq 1 ] \
     && printf '%s' "$out" | grep -q 'FAIL PREFLIGHT' \
     && printf '%s' "$out" | grep -q 'refs/heads/no-such-xyzzy' \
     && ! printf '%s' "$out" | grep -qE '^  measured    base '; then
    ok "C4  unresolvable comparand: RED at preflight, ref named, and NO measurement happened"
  else
    bad "C4  unresolvable comparand: rc=$rc, wanted 1, a PREFLIGHT failure naming the ref, and no measurement line"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # --- C5: two revisions, two pmats (attack A6) ---------------------------
  local C5="$WORK/cx5"
  cx_fixture "$C5" || die "C5 fixture could not be built"
  cx_lib '9.9.9-shim' 'branchy 35 34' 'cognitive_only 8 28' 'tidy 1 0' > "$C5/src/lib.rs" || die "C5: cannot rewrite src/lib.rs"
  cx_commit "$C5" 'merge: the same tree, measured by a different pmat' || die "C5: commit failed"
  out=$(cx_run "$C5" check_complexity_ratchet.sh); rc=$?
  if [ "$rc" -eq 1 ] \
     && printf '%s' "$out" | grep -q 'DIFFERENT pmat versions' \
     && printf '%s' "$out" | grep -q 'pmat 0.0.0-shim' \
     && printf '%s' "$out" | grep -q 'pmat 9.9.9-shim'; then
    ok "C5  the comparand and the merge measured by different pmats: RED, BOTH versions printed"
  else
    bad "C5  instrument mismatch: rc=$rc, wanted 1 and both 'pmat 0.0.0-shim' and 'pmat 9.9.9-shim'"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # --- C6: the recorded tool_version header disagrees ---------------------
  local C6="$WORK/cx6"
  cx_fixture "$C6" 'pmat 1.2.3-recorded' || die "C6 fixture could not be built"
  cx_commit "$C6" 'merge: no change' >/dev/null 2>&1 || git_fx "$C6" commit -q --allow-empty -m 'merge: no change' >/dev/null 2>&1 || die "C6: commit failed"
  out=$(cx_run "$C6" check_complexity_ratchet.sh); rc=$?
  if [ "$rc" -eq 1 ] \
     && printf '%s' "$out" | grep -q 'the recorded tool_version does not match' \
     && printf '%s' "$out" | grep -q 'pmat 1.2.3-recorded' \
     && printf '%s' "$out" | grep -q 'pmat 0.0.0-shim'; then
    ok "C6  the recorded tool_version header disagrees with the pmat that ran: RED, BOTH versions printed"
  else
    bad "C6  recorded instrument: rc=$rc, wanted 1 and both 'pmat 1.2.3-recorded' and 'pmat 0.0.0-shim'"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # --- C7: THE REGISTERED MUTATION ----------------------------------------
  # Put the lower bound back by restoring the pre-BSE-03 verdict: read the
  # baseline FILE and red a row whose function no longer offends (STALE). C3
  # must go RED under it, or C3's GREEN is not evidence of anything.
  local MUT='check_complexity_ratchet_mutant.sh'
  if sed 's|^FINDINGS=.*# RATCHET-MUTATION-POINT.*$|FINDINGS=$(cx_verdict "$BASELINE" "$WORK/merge/rows.txt") \|\| VERDICT_RC=$?|' \
        "$C3/scripts/check_complexity_ratchet.sh" > "$C3/scripts/$MUT" \
     && grep -q 'cx_verdict "\$BASELINE"' "$C3/scripts/$MUT" \
     && ! grep -q 'RATCHET-MUTATION-POINT' "$C3/scripts/$MUT"; then
    ok "C7  the mutation point is present in the shipped guard and the mutant was derived from a COPY of it"
  else
    bad "C7  the mutation could not be derived: no '# RATCHET-MUTATION-POINT' verdict line in scripts/check_complexity_ratchet.sh"
  fi
  out=$(cx_run "$C3" "$MUT"); rc=$?
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q 'STALE'; then
    ok "C7  mutation caught: a lower bound read from the baseline FILE turns C3 (a fixed function) RED — so C3's GREEN is load-bearing"
  else
    bad "C7  mutation NOT demonstrated: the mutant still exits $rc on the improvement fixture, so C3 does not discriminate 'measured comparand' from 'file on disk'"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi
}

# ===========================================================================
# --class satd
# ===========================================================================

satd_run() { # satd_run <binary> <filter> [VAR=value...] -> output; rc is the test rc
  local bin="$1" filter="$2"
  shift 2
  env "$@" "$bin" "$filter" --nocapture --test-threads=1 2>&1
}

run_satd() {
  local bin log out rc n
  log="$WORK/satd-build.log"
  # `--features model-tests` is not optional: the whole suite is
  # `#![cfg(feature = "model-tests")]`, so without it the binary contains ZERO
  # tests and every row below would "pass" over nothing.
  if ! ( cd "$ROOT" && CARGO_TERM_COLOR=never cargo test -p aprender-core --features model-tests \
           --test falsification_spec_v10_tests --no-run ) > "$log" 2>&1; then
    die "cargo could not build the falsification suite: $(tail -3 "$log" | tr '\n' ' ')"
  fi
  bin=$(sed -n 's|.*Executable tests/falsification_spec_v10_tests\.rs (\(.*\))$|\1|p' "$log" | tail -1)
  [ -n "$bin" ] && [ -x "$bin" ] || die "cargo did not name an executable for the falsification suite"

  # S1 — the parsing/fallback unit test.
  out=$(satd_run "$bin" f_checklist_005_satd_baseline_source_is_env_then_constant); rc=$?
  if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q '^test result: ok. 1 passed'; then
    ok "S1  satd_baseline_from: the parsing/fallback unit test runs and passes"
  else
    bad "S1  parsing test: rc=$rc, wanted 0 and exactly 1 passed"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # THE COUNT IS READ OUT OF THE INSTRUMENT, never pinned here: a ceiling of 0
  # forces the guard to state how many markers it measured.
  out=$(satd_run "$bin" f_dod_001_satd_count_is_zero SATD_BASELINE=0); rc=$?
  n=$(printf '%s' "$out" | sed -n 's/.*BROKEN -- \([0-9][0-9]*\) markers in production source.*/\1/p' | head -1)
  if [ "$rc" -ne 0 ] && [ -n "$n" ]; then
    ok "S0  a ceiling of 0 over $n measured marker(s): RED, and the guard states the number it measured"
  else
    bad "S0  ceiling 0: rc=$rc, wanted a failure naming the measured marker count"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
    printf '\n%s checks, %s failed\n' "$CHECKS" "$FAILED"
    exit 1
  fi

  # S2 — unset falls back to the constant, and SAYS SO.
  out=$(satd_run "$bin" f_dod_001_satd_count_is_zero); rc=$?
  if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q 'SATD ceiling .* (source: constant)'; then
    ok "S2  SATD_BASELINE unset: GREEN, and the run names the constant as the source"
  else
    bad "S2  unset: rc=$rc, wanted 0 and a 'SATD ceiling N (source: constant)' line"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # S3b — a regression: the comparand carried one marker fewer than this tree.
  out=$(satd_run "$bin" f_dod_001_satd_count_is_zero "SATD_BASELINE=$((n - 1))"); rc=$?
  if [ "$rc" -ne 0 ] \
     && printf '%s' "$out" | grep -q "the ceiling is $((n - 1)) (source: comparand)" \
     && printf '%s' "$out" | grep -q "$n markers in production source"; then
    ok "S3  ceiling $((n - 1)) against $n measured: RED, both numbers printed, source named as the comparand"
  else
    bad "S3  regression: rc=$rc, wanted a failure naming both $n and $((n - 1))"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # S4 — exactly at the ceiling passes.
  out=$(satd_run "$bin" f_dod_001_satd_count_is_zero "SATD_BASELINE=$n"); rc=$?
  if [ "$rc" -eq 0 ]; then
    ok "S4  ceiling $n against $n measured: GREEN (at the ceiling is not over it)"
  else
    bad "S4  at the ceiling: rc=$rc, wanted 0"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # S5 — THE IMPROVEMENT ROW. The comparand carried 20 markers more than this
  # tree. The pre-BSE-03 code asserted `violations.len() + 8 >= baseline`, and
  # n + 8 >= n + 20 is false, so this row was RED before this ticket.
  out=$(satd_run "$bin" f_dod_001_satd_count_is_zero "SATD_BASELINE=$((n + 20))"); rc=$?
  if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q "SATD ceiling $((n + 20)) (source: comparand)"; then
    ok "S5  ceiling $((n + 20)) against $n measured: GREEN — an improvement is not a failure, and there is no lower bound"
  else
    bad "S5  improvement: rc=$rc, wanted 0 with the comparand ceiling named — a lower bound has been reintroduced"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # S6 — set but EMPTY: the measurement did not happen.
  out=$(satd_run "$bin" f_dod_001_satd_count_is_zero SATD_BASELINE=); rc=$?
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q 'set but empty'; then
    ok "S6  SATD_BASELINE set but empty: RED — a failed comparand measurement never falls back to the constant"
  else
    bad "S6  empty: rc=$rc, wanted a failure saying the comparand measurement did not happen"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi

  # S7 — unparseable.
  out=$(satd_run "$bin" f_dod_001_satd_count_is_zero SATD_BASELINE=not-a-number); rc=$?
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q 'SATD_BASELINE'; then
    ok "S7  an unparseable SATD_BASELINE: RED, and the failure names the variable"
  else
    bad "S7  unparseable: rc=$rc, wanted a failure naming SATD_BASELINE"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi
  # S8 — MUTATION. The compiled-in constant is the one number a PR can rewrite
  # (threat-model class 2). Raise it to 999999 in the include, rebuild the
  # suite, and run the MUTANT twice: with SATD_BASELINE unset it is GREEN — the
  # pre-BSE-03 path, where the rewrite "passed"; with SATD_BASELINE=$((n - 1))
  # it is RED naming $n, judged by the export and not by 999999. If the unset
  # run were RED, or the exported run were judged by the constant, this class
  # would not be measuring the property. The include is restored whatever
  # happens (EXIT trap while mutated, then explicitly) and the suite is
  # rebuilt from the restored source so no mutant binary outlives this row.
  local inc="$ROOT/crates/aprender-core/tests/includes/falsification_spec_v10_checklist.rs" mbin mlog
  mlog="$WORK/satd-mutant-build.log"
  cp "$inc" "$WORK/checklist.orig" || die "cannot copy the checklist include for S8"
  grep -q '^const SATD_PRODUCTION_BASELINE: usize = [0-9][0-9]*;$' "$inc" \
    || die "S8: the constant SATD_PRODUCTION_BASELINE is not where this row expects it"
  trap 'cp "$WORK/checklist.orig" "$inc" 2>/dev/null; rm -rf "${WORK:?}"' EXIT
  sed -i 's/^const SATD_PRODUCTION_BASELINE: usize = [0-9][0-9]*;$/const SATD_PRODUCTION_BASELINE: usize = 999999;/' "$inc"
  cmp -s "$inc" "$WORK/checklist.orig" && die "S8: the mutant is byte-identical to the include (sed matched nothing)"
  mbin=""
  if ( cd "$ROOT" && cargo test -p aprender-core --features model-tests \
         --test falsification_spec_v10_tests --no-run ) > "$mlog" 2>&1; then
    mbin=$(sed -n 's|.*Executable tests/falsification_spec_v10_tests\.rs (\(.*\))$|\1|p' "$mlog" | tail -1)
  fi
  cp "$WORK/checklist.orig" "$inc"; trap 'rm -rf "${WORK:?}"' EXIT
  [ -n "$mbin" ] && [ -x "$mbin" ] || die "S8: cargo could not build the mutant: $(tail -3 "$mlog" | tr '\n' ' ')"
  out=$(satd_run "$mbin" f_dod_001_satd_count_is_zero); rc=$?
  if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q 'SATD ceiling 999999 (source: constant)'; then
    ok "S8a mutant (constant rewritten to 999999), SATD_BASELINE unset: GREEN — the rewritable path a PR had before"
  else
    bad "S8a mutant unset: rc=$rc, wanted 0 naming the rewritten constant"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi
  out=$(satd_run "$mbin" f_dod_001_satd_count_is_zero "SATD_BASELINE=$((n - 1))"); rc=$?
  if [ "$rc" -ne 0 ] && printf '%s' "$out" | grep -q "the ceiling is $((n - 1)) (source: comparand)"; then
    ok "S8b the same mutant with the comparand exported: RED at $((n - 1)) against $n — the rewritten constant is unread"
  else
    bad "S8b mutant + comparand: rc=$rc, wanted RED judged by the comparand, not by 999999"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
  fi
  ( cd "$ROOT" && cargo test -p aprender-core --features model-tests \
      --test falsification_spec_v10_tests --no-run ) > "$WORK/satd-rebuild.log" 2>&1 \
    || die "S8: the suite could not be rebuilt from the restored include: $(tail -2 "$WORK/satd-rebuild.log" | tr '\n' ' ')"
}

case "$CLASS" in
  complexity) run_complexity; finish; exit ;;
  satd)       run_satd;       finish; exit ;;
esac

# ---------------------------------------------------------------------------
# Row 1 — the block states the truth.
F1="$WORK/f1"
fixture "$F1" 2 "$(readme_with_block 2)" || die "row 1 fixture could not be built"
out="$(run_guard "$F1" check_readme_claims.sh)"; rc=$?
if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q '^PASS FALSIFY-README-002 contract_count: 2 '; then
  ok "row 1  block states the truth (2 of 2): GREEN, PASS line names the derived count"
else
  bad "row 1  block states the truth: rc=$rc, wanted 0 and a 'PASS FALSIFY-README-002 contract_count: 2' line"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
fi
# the D2 header must name BOTH revs and BOTH measurements, before any verdict
if printf '%s' "$out" | grep -q '^  comparand ' && printf '%s' "$out" | grep -qE '^  contracts +base=2 +merge=2'; then
  ok "row 1  D2 header prints the comparand SHA and both measurements"
else
  bad "row 1  D2 header missing: wanted a 'comparand' line and 'contracts base=2 merge=2'"$'\n'"$(printf '%s' "$out" | sed 's/^/        /')"
fi

# ---------------------------------------------------------------------------
# Row 2 — the block hand-edited to a number the tree does not carry.
F2="$WORK/f2"
fixture "$F2" 2 "$(readme_with_block 9)" || die "row 2 fixture could not be built"
out2="$(run_guard "$F2" check_readme_claims.sh)"; rc=$?
if [ "$rc" -eq 1 ] \
   && printf '%s' "$out2" | grep -q 'FAIL FALSIFY-README-002 contract_count: the CONTRACT_COUNT block states 9' \
   && printf '%s' "$out2" | grep -q 'merge tree carries 2'; then
  ok "row 2  block hand-edited to 9 over 2 files: RED, both numbers printed"
else
  bad "row 2  hand-edited block: rc=$rc, wanted 1 and a FAIL line carrying BOTH 9 and 2"$'\n'"$(printf '%s' "$out2" | sed 's/^/        /')"
fi

# ---------------------------------------------------------------------------
# Row 3 — an AUTHORED literal (no markers) that overstates.
F3="$WORK/f3"
fixture "$F3" 2 "$(readme_with_literal 9)" || die "row 3 fixture could not be built"
out3="$(run_guard "$F3" check_readme_claims.sh)"; rc=$?
if [ "$rc" -eq 1 ] && printf '%s' "$out3" | grep -q 'README claims 9' && printf '%s' "$out3" | grep -q 'has 2'; then
  ok "row 3  authored literal 9 over 2 files: RED, both numbers printed"
else
  bad "row 3  authored literal: rc=$rc, wanted 1 and both numbers"$'\n'"$(printf '%s' "$out3" | sed 's/^/        /')"
fi

# ---------------------------------------------------------------------------
# Row 4 — no block, no literal: the count is DERIVED. GREEN only because
# readme_sync.sh regenerates it deterministically IN THIS RUN and the bytes are
# printed. (P7 of the threat model, inverted only once the generator exists.)
F4="$WORK/f4"
fixture "$F4" 2 "$(readme_derived)" || die "row 4 fixture could not be built"
out4="$(run_guard "$F4" check_readme_claims.sh)"; rc=$?
if [ "$rc" -eq 0 ] \
   && printf '%s' "$out4" | grep -q 'PASS FALSIFY-README-002 contract_count: DERIVED' \
   && printf '%s' "$out4" | grep -qF "${START}2${END}"; then
  ok "row 4  no block and no literal: GREEN, and the bytes readme_sync would write are printed"
else
  bad "row 4  derived claim: rc=$rc, wanted 0, a 'DERIVED' PASS line and the regenerated block bytes"$'\n'"$(printf '%s' "$out4" | sed 's/^/        /')"
fi

# ---------------------------------------------------------------------------
# Row 5 — readme_sync.sh --write is idempotent: a stale block converges in one
# run and the second run changes nothing, byte for byte.
F5="$WORK/f5"
fixture "$F5" 3 "$(readme_with_block 9)" || die "row 5 fixture could not be built"
( cd "$F5" && bash "$F5/scripts/readme_sync.sh" --write ) >"$WORK/sync1.log" 2>&1; rc=$?
h1="$(sha256sum "$F5/README.md" | cut -d' ' -f1)"
( cd "$F5" && bash "$F5/scripts/readme_sync.sh" --write ) >"$WORK/sync2.log" 2>&1; rc2=$?
h2="$(sha256sum "$F5/README.md" | cut -d' ' -f1)"
if [ "$rc" -eq 0 ] && [ "$rc2" -eq 0 ] && [ "$h1" = "$h2" ] && grep -qF "${START}3${END}" "$F5/README.md"; then
  ok "row 5  readme_sync.sh --write rewrites the stale block to 3 and is idempotent (identical sha256 on the second run)"
else
  bad "row 5  readme_sync idempotency: rc=$rc/$rc2, sha $h1 vs $h2"$'\n'"$(sed 's/^/        /' "$WORK/sync1.log" "$WORK/sync2.log")"
fi
# and the guard now agrees with the generator's output, in the same fixture
out5="$(run_guard "$F5" check_readme_claims.sh)"; rc=$?
if [ "$rc" -eq 0 ]; then
  ok "row 5  the guard passes on what the generator wrote (generator and guard share one instrument)"
else
  bad "row 5  the guard REJECTS what readme_sync.sh wrote: rc=$rc"$'\n'"$(printf '%s' "$out5" | sed 's/^/        /')"
fi

# ---------------------------------------------------------------------------
# Row 6 — the comparand cannot be resolved: RED at PREFLIGHT, before any
# measurement. The proof that it is a preflight failure is that the
# measurement line is ABSENT from the output.
out6="$(run_guard "$F1" check_readme_claims.sh BASELINE_RATCHET_BASE_REF=refs/heads/no-such-xyzzy)"; rc=$?
if [ "$rc" -eq 1 ] \
   && printf '%s' "$out6" | grep -q 'FAIL FALSIFY-README-002 contract_count: PREFLIGHT' \
   && printf '%s' "$out6" | grep -q 'refs/heads/no-such-xyzzy' \
   && ! printf '%s' "$out6" | grep -qE '^  contracts +base='; then
  ok "row 6  unresolvable comparand: RED at preflight, named ref, and NO measurement line"
else
  bad "row 6  unresolvable comparand: rc=$rc, wanted 1, a PREFLIGHT FAIL naming the ref, and no measurement line"$'\n'"$(printf '%s' "$out6" | sed 's/^/        /')"
fi

# ---------------------------------------------------------------------------
# Row 7 — the count FELL vs the comparand. That is an improvement, and a README
# that states the fallen count is TRUE. The verdict is about truth, not
# direction, and the header must say so.
F7="$WORK/f7"
fixture "$F7" 3 "$(readme_with_block 3)" || die "row 7 fixture could not be built"
rm -f "$F7/contracts/c3.yaml" || die "row 7: cannot remove a contract"
readme_with_block 2 > "$F7/README.md" || die "row 7: cannot rewrite the README"
git_fx "$F7" add -A -f contracts README.md >/dev/null 2>&1 || die "row 7: git add failed"
git_fx "$F7" commit -q -m "drop one contract" >/dev/null 2>&1 || die "row 7: git commit failed"
out7="$(run_guard "$F7" check_readme_claims.sh)"; rc=$?
if [ "$rc" -eq 0 ] \
   && printf '%s' "$out7" | grep -qE '^  contracts +base=3 +merge=2 +delta=-1' \
   && printf '%s' "$out7" | grep -q 'TRUTH, not DIRECTION'; then
  ok "row 7  count fell 3 -> 2 and the README states 2: GREEN, delta=-1 printed, polarity stated"
else
  bad "row 7  improvement: rc=$rc, wanted 0, 'contracts base=3 merge=2 delta=-1' and the truth-not-direction line"$'\n'"$(printf '%s' "$out7" | sed 's/^/        /')"
fi

# ---------------------------------------------------------------------------
# Row 8 — THE REGISTERED MUTATION. A copy of the real guard whose merge-tree
# measurement reads a number from a file on disk instead of from the checkout.
# Rows 2 and 3 must turn GREEN under it. If they do not, they are not measuring
# "the claim agrees with the TREE" and this suite proves nothing.
MUTANT="check_readme_claims_mutant.sh"
mutate() { # mutate <fixture dir>
  sed 's|^.*# RATCHET-MUTATION-POINT.*$|  merge_count=$(cat "$RATCHET_MUTANT_COUNT_FILE")|' \
      "$1/scripts/check_readme_claims.sh" > "$1/scripts/$MUTANT" || return 1
  grep -q 'RATCHET_MUTANT_COUNT_FILE' "$1/scripts/$MUTANT"
}
printf '9\n' > "$WORK/mutant-count.txt"
if mutate "$F2" && mutate "$F3"; then
  ok "row 8  the mutation point is present in the shipped guard and the mutant was derived from a COPY of it"
else
  bad "row 8  the mutation could not be derived: no '# RATCHET-MUTATION-POINT' line in scripts/check_readme_claims.sh"
fi
m2="$(run_guard "$F2" "$MUTANT" RATCHET_MUTANT_COUNT_FILE="$WORK/mutant-count.txt")"; mrc2=$?
m3="$(run_guard "$F3" "$MUTANT" RATCHET_MUTANT_COUNT_FILE="$WORK/mutant-count.txt")"; mrc3=$?
if [ "$mrc2" -eq 0 ]; then
  ok "row 8  mutation caught: reading the count from a file on disk turns row 2 (hand-edited block) GREEN — so row 2's RED is load-bearing"
else
  bad "row 8  mutation NOT demonstrated on row 2: the mutant still exits $mrc2, so row 2 does not discriminate 'measured from the tree' from 'read from disk'"$'\n'"$(printf '%s' "$m2" | sed 's/^/        /')"
fi
if [ "$mrc3" -eq 0 ]; then
  ok "row 8  mutation caught: the same mutant turns row 3 (authored literal) GREEN"
else
  bad "row 8  mutation NOT demonstrated on row 3: the mutant still exits $mrc3"$'\n'"$(printf '%s' "$m3" | sed 's/^/        /')"
fi

printf '\n%s checks, %s failed\n' "$CHECKS" "$FAILED"
[ "$FAILED" -eq 0 ]
