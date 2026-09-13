#!/usr/bin/env bash
# gate_touched_crates_test.sh — self-test for scripts/gate_touched_crates.sh
# (BSE-16, docs/specifications/build-system-enhancement.md, infra repo).
#
# Every case runs with --dry-run: no `cargo test`/`cargo check` is ever
# executed here, only the PLAN gate_touched_crates.sh would run — the
# forbidden list for this ticket bans running the full workspace test suite
# and this self-test does not need it: only the SELECTION logic is asserted.
#
# Deliberately NOT `set -e`: most of this file is assertions that are
# expected to fail on a broken script (that's the FAIL path a mutation must
# hit), so failure is aggregated in $FAIL/check() instead of aborting the
# whole run on the first RED case. `set -u` and `pipefail` are still on;
# setup steps that must succeed are each followed by an explicit `|| exit 1`.
#
# Fixture crates were measured directly with `cargo metadata` on 2026-09-07
# (this worktree), not assumed:
#   - aprender-core has 13 direct reverse dependents -> touching it selects
#     14 crates total, over CAP(3) -> `cargo check --workspace --tests`.
#     This is also the ideal mutation fixture: WITHOUT the reverse-dependents
#     step the same touch is just 1 crate (under cap) -> `cargo test -p
#     aprender-core` — a different action, so the mutation is observable.
#   - aprender-rand has exactly 2 direct reverse dependents (aprender-core,
#     aprender-contrastive-data) -> touching it selects 3 crates, AT cap
#     (not exceeded) -> the targeted `cargo test -p ...` path.
#   - aprender-fft/-tensor/-sparse/-solve have 0 reverse dependents each;
#     touching all four directly (no expansion needed) exceeds CAP(3) via
#     plain touched-crate count -> `cargo check --workspace --tests`.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1

PASS=0
FAIL=0

check() {
  local desc="$1"
  local ok="$2"
  if [ "$ok" -eq 0 ]; then
    echo "PASS: $desc"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $desc"
    FAIL=$((FAIL + 1))
  fi
}

tmpdir="$(mktemp -d)" || exit 1
cleanup() {
  rm -rf "${tmpdir:?tmpdir must be set}"
  rm -f scripts/.gate_touched_crates_mutant.sh
}
trap cleanup EXIT

GATE_SCRIPT="scripts/gate_touched_crates.sh"

# --- Case 1: touching aprender-core selects it + its direct reverse
#     dependents; that selection (measured: 14 crates) exceeds the cap, so
#     the gate fails closed to a full workspace check.
printf 'crates/aprender-core/src/lib.rs\n' > "$tmpdir/diff-core.txt" || exit 1
out_core="$(bash "$GATE_SCRIPT" --dry-run --diff-from "$tmpdir/diff-core.txt" 2>&1)"
grep -q 'aprender-core' <<< "$out_core"
check "aprender-core touch: selection includes aprender-core" $?
grep -qE 'cargo check --workspace --tests' <<< "$out_core"
check "aprender-core touch: reverse-dependent fan-out exceeds cap -> check --workspace" $?

# --- Case 2: a Cargo.lock diff selects a full workspace check, regardless of
#     how few crates it maps to (it maps to none).
printf 'Cargo.lock\n' > "$tmpdir/diff-lock.txt" || exit 1
out_lock="$(bash "$GATE_SCRIPT" --dry-run --diff-from "$tmpdir/diff-lock.txt" 2>&1)"
grep -qE 'cargo check --workspace --tests' <<< "$out_lock"
check "Cargo.lock diff selects check --workspace" $?

# --- Case 3: touching more than CAP crates directly (no reverse-dependent
#     expansion needed — these four have none) selects a full workspace check.
printf 'crates/aprender-fft/src/lib.rs\ncrates/aprender-tensor/src/lib.rs\ncrates/aprender-sparse/src/lib.rs\ncrates/aprender-solve/src/lib.rs\n' > "$tmpdir/diff-four.txt" || exit 1
out_four="$(bash "$GATE_SCRIPT" --dry-run --diff-from "$tmpdir/diff-four.txt" 2>&1)"
grep -qE 'cargo check --workspace --tests' <<< "$out_four"
check "more than 3 touched crates selects check --workspace" $?

# --- Case 4: an under-cap touch runs the targeted `cargo test -p ...`
#     selection, including the touched crate's direct reverse dependents.
printf 'crates/aprender-rand/src/lib.rs\n' > "$tmpdir/diff-rand.txt" || exit 1
out_rand="$(bash "$GATE_SCRIPT" --dry-run --diff-from "$tmpdir/diff-rand.txt" 2>&1)"
grep -qE -- '\+ cargo test .*-p aprender-rand' <<< "$out_rand"
check "under-cap touch runs targeted cargo test -p (includes touched crate)" $?
grep -qE -- '-p aprender-core' <<< "$out_rand"
check "under-cap touch includes direct reverse dependent aprender-core" $?
case "$out_rand" in
  *'cargo check --workspace'*) check "under-cap touch does NOT fall back to check --workspace" 1 ;;
  *) check "under-cap touch does NOT fall back to check --workspace" 0 ;;
esac

# --- Mutation 1: drop the reverse-dependents-expansion step -> Case 1 must
#     go RED (the selection flips from check-workspace to a single-crate test).
sed '/# BEGIN reverse-dependents-expansion/,/# END reverse-dependents-expansion/d' \
  "$GATE_SCRIPT" > scripts/.gate_touched_crates_mutant.sh || exit 1
chmod +x scripts/.gate_touched_crates_mutant.sh || exit 1
mutant_out="$(bash scripts/.gate_touched_crates_mutant.sh --dry-run --diff-from "$tmpdir/diff-core.txt" 2>&1)"
case "$mutant_out" in
  *'cargo check --workspace --tests'*)
    check "mutation (no reverse-dependents step): aprender-core case no longer selects check --workspace (RED, as required)" 1 ;;
  *)
    check "mutation (no reverse-dependents step): aprender-core case no longer selects check --workspace (RED, as required)" 0 ;;
esac

# --- Mutation 2: appending `|| true` to the gate recipe must trip the
#     acceptance grep (`grep -cE '\|\| *(true|echo)'` on `make -n gate`).
cp Makefile "$tmpdir/Makefile.mutant" || exit 1
sed -i 's#^\tscripts/gate_touched_crates\.sh$#\tscripts/gate_touched_crates.sh || true#' "$tmpdir/Makefile.mutant" || exit 1
grep -q 'gate_touched_crates.sh || true' "$tmpdir/Makefile.mutant"
check "mutation fixture actually inserted || true into the gate recipe" $?
mutant_lines="$(make -n -f "$tmpdir/Makefile.mutant" gate 2>/dev/null | grep -cE '\|\| *(true|echo)')"
[ -z "$mutant_lines" ] && mutant_lines=0
[ "$mutant_lines" -gt 0 ]
check "mutation (|| true in recipe): acceptance grep trips (RED, as required)" $?

# --- discover.sh must resolve `make gate` without the cargo-test fallback.
bash "$HOME/.claude/skills/paiml-implement/scripts/discover.sh" -o "$tmpdir/discover.json" >/dev/null 2>&1
discover_rc=$?
if [ "$discover_rc" -eq 0 ]; then
  grep -q '"gate_cmd_fallback": "false"' "$tmpdir/discover.json"
  check "discover.sh resolves make gate (gate_cmd_fallback=false)" $?
else
  echo "SKIP: discover.sh not runnable in this environment (exit $discover_rc)"
fi

echo ""
echo "$((PASS + FAIL)) checks, $FAIL failed"
[ "$FAIL" -eq 0 ]
