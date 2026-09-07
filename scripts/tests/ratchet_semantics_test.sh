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
# `--class complexity` and `--class satd` are documented stubs: they exit 3.
# They never exit 0 — a stub that passes is a guard reporting a verdict it did
# not reach.
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
  --class complexity  stub, exit 3 — not implemented in this phase
  --class satd        stub, exit 3 — not implemented in this phase
USAGE
  exit 2
}

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
  complexity|satd)
    # A stub exits 3, never 0. The class is named in the threat model
    # (docs/audits/threat-model-bse-03-ratchets.md, classes 2 and 3) and its
    # rows P1-P6/P11/P12 are written there; the measurement side
    # (cx_measure at two revs, pmat comply at two revs) is not built in phase A.
    printf '%s: not implemented in this phase — BSE-03 phase A covers --class readme only.\n' "$CLASS" >&2
    printf 'The rows for this class are specified in docs/audits/threat-model-bse-03-ratchets.md (Acceptance).\n' >&2
    exit 3 ;;
  readme) : ;;
  *) printf 'unknown class: %s\n' "$CLASS" >&2; usage ;;
esac

[ -f "$GUARD" ] || die "missing $GUARD"
[ -f "$SYNC" ]  || die "missing $SYNC — the generator that makes the count derived (BSE-03 decision 1)"

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
