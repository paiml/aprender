#!/usr/bin/env bash
# witness_diff_test.sh — self-test for scripts/witness_diff.sh (BSE-11,
# docs/specifications/build-system-enhancement.md wave 3, `paiml/infra`,
# Pmat-Ticket: PMAT-1067).
#
# Cases:
#   1. identical files                          -> UNCHANGED, exit 0
#   2. one byte different                       -> CHANGED,   exit 0
#   3. a colour word present but unchanged       -> UNCHANGED (a colour word
#      in the input must not itself change the verdict — the verdict comes
#      from bytes, never from recognising or normalising any particular word)
#   4. a colour word is the ONLY difference      -> CHANGED (same point from
#      the other side: a colour word is not exempt from the byte compare
#      either — swapping "green" for "red" is still a byte that differs)
#   5. the mutation the spec names, "compare status instead of bytes", is
#      applied to a temp copy of the script and shown to get case 2's shape
#      WRONG (a status field can agree while the surrounding witness line
#      does not) — proving this test suite would go RED if that mutation
#      ever landed in the real script.
#
# Prints "N checks, M failed" and exits 1 if M > 0.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT/scripts/witness_diff.sh"

WORK="$(mktemp -d -t witness_diff_test.XXXXXX)"
STDERR_SINK="$(mktemp -t witness_diff_test_stderr.XXXXXX)"
trap 'rm -rf "${WORK:?}" "${STDERR_SINK:?}"' EXIT

CHECKS=0
FAILED=0

check() {
  # check <name> <expected-stdout> <expected-exit> -- <cmd...>
  local name="$1" want_out="$2" want_rc="$3"
  shift 3
  [ "$1" = "--" ] || { echo "check(): malformed call, missing --" >&2; exit 2; }
  shift
  CHECKS=$((CHECKS + 1))
  local got_out got_rc
  got_out="$("$@" 2>"$STDERR_SINK")"
  got_rc=$?
  if [ "$got_out" = "$want_out" ] && [ "$got_rc" = "$want_rc" ]; then
    printf 'ok    %s\n' "$name"
  else
    printf 'FAIL  %s: got output=%q rc=%s, want output=%q rc=%s\n' \
      "$name" "$got_out" "$got_rc" "$want_out" "$want_rc"
    FAILED=$((FAILED + 1))
  fi
}

# --- Case 1: identical files -> UNCHANGED, exit 0 -----------------------
printf 'PP-26-WITNESS: band=c4 divergence_at=3 status=PASS\n' > "$WORK/a.txt"
cp "$WORK/a.txt" "$WORK/b_identical.txt"
check "identical files -> UNCHANGED" "UNCHANGED" 0 -- \
  bash "$SCRIPT" "$WORK/a.txt" "$WORK/b_identical.txt"

# --- Case 2: one byte different -> CHANGED, exit 0 -----------------------
printf 'PP-26-WITNESS: band=c4 divergence_at=4 status=PASS\n' > "$WORK/b_onebyte.txt"
check "one byte different -> CHANGED" "CHANGED" 0 -- \
  bash "$SCRIPT" "$WORK/a.txt" "$WORK/b_onebyte.txt"

# --- Case 3: a colour word present, files otherwise identical -> UNCHANGED
printf 'PP-26-WITNESS: band=c4 divergence_at=3 status=PASS colour=green\n' \
  > "$WORK/colour_a.txt"
cp "$WORK/colour_a.txt" "$WORK/colour_b_same.txt"
check "colour word present but unchanged -> UNCHANGED" "UNCHANGED" 0 -- \
  bash "$SCRIPT" "$WORK/colour_a.txt" "$WORK/colour_b_same.txt"

# --- Case 4: only the colour word differs -> CHANGED ----------------------
sed 's/colour=green/colour=red/' "$WORK/colour_a.txt" > "$WORK/colour_b_diff.txt"
check "only a colour word differs -> CHANGED" "CHANGED" 0 -- \
  bash "$SCRIPT" "$WORK/colour_a.txt" "$WORK/colour_b_diff.txt"

# --- Case 5: the named mutation ("compare status instead of bytes") -------
# Two witness lines that AGREE on status=PASS but differ everywhere else
# (a real divergence_at moved from 3 to 64, i.e. the actual thing this tool
# exists to catch).
printf 'PP-26-WITNESS: band=c4 divergence_at=3 status=PASS\n' > "$WORK/mut_expected.txt"
printf 'PP-26-WITNESS: band=c4 divergence_at=64 status=PASS\n' > "$WORK/mut_observed.txt"

# The real script must call this CHANGED (bytes differ).
check "mutation fixture, real script -> CHANGED" "CHANGED" 0 -- \
  bash "$SCRIPT" "$WORK/mut_expected.txt" "$WORK/mut_observed.txt"

# A mutant that compares only the `status=` token instead of the file's
# bytes. This is written standalone (not derived by editing a copy of the
# real script) so the mutation is exactly the one the spec names, not an
# artifact of how the real script happens to be phrased today.
MUTANT="$WORK/witness_diff_mutant.sh"
cat > "$MUTANT" <<'MUT'
#!/usr/bin/env bash
# MUTATION UNDER TEST: compares a parsed `status=` field instead of raw
# bytes. This file must never be shipped; it exists only so the test above
# can prove the real script does NOT behave this way.
set -uo pipefail
expected="$1"; observed="$2"
st_e="$(grep -oE 'status=[A-Za-z_]+' "$expected" | head -1)"
st_o="$(grep -oE 'status=[A-Za-z_]+' "$observed" | head -1)"
if [ "$st_e" = "$st_o" ]; then
  echo "UNCHANGED"
else
  echo "CHANGED"
fi
exit 0
MUT
chmod +x "$MUTANT"

CHECKS=$((CHECKS + 1))
mutant_out="$(bash "$MUTANT" "$WORK/mut_expected.txt" "$WORK/mut_observed.txt")"
if [ "$mutant_out" = "UNCHANGED" ]; then
  printf 'ok    mutation caught: a status-only comparator reports UNCHANGED on this fixture (would be RED if it were the real script — real script correctly reports CHANGED)\n'
else
  printf 'FAIL  mutation not demonstrated: the status-only mutant reported %q, expected it to wrongly say UNCHANGED so the contrast with the real script (CHANGED) proves discrimination\n' "$mutant_out"
  FAILED=$((FAILED + 1))
fi

# --- Case 6: usage errors are distinct from a witness verdict -------------
check "missing argument -> usage error, exit 2" "" 2 -- \
  bash "$SCRIPT" "$WORK/a.txt"

check "nonexistent file -> usage error, exit 2" "" 2 -- \
  bash "$SCRIPT" "$WORK/a.txt" "$WORK/does-not-exist.txt"

echo ""
echo "$CHECKS checks, $FAILED failed"
[ "$FAILED" -eq 0 ]
