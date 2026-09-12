#!/usr/bin/env bash
#
# guard_tree_parallel_test.sh -- the case table for guard_tree.sh's PARALLEL
# dispatcher (PMAT-1098, row 67-E3 of docs/specifications/06x-release-schedule.md).
#
# WHY THIS EXISTS
# ---------------
# `bash scripts/guard_tree.sh --no-cargo` is ONE step of the required
# `guard-tree` job and it measured 1105 s -- 18.4 of that job's 27.3 minutes --
# on run 34449608126 (PR #3074, 2026-09-10). It ran ~50 independent, mostly
# I/O- and grep-bound guards strictly one after another on a 48-core box.
# Row 67-E3 requires the whole job under the operator's 20-minute PR budget,
# so the dispatcher runs its guards concurrently.
#
# Concurrency is where a runner silently stops being a runner: output
# interleaves, a row goes missing, a worker dies before it writes its verdict
# and the summary counts one fewer check instead of one more failure. Every
# row below pins one of those, against a THROWAWAY GIT REPO -- guard_tree.sh
# derives its universe from `git ls-files 'scripts/check_*.sh'`, so a fixture
# repo with three planted guards is a complete universe and the table costs
# seconds rather than the 18 minutes the real one costs.
#
# THE ROW THAT WAS RED BEFORE THE CHANGE
# ---------------------------------------
# Row (e) is the one that discriminates a parallel dispatcher from a serial
# one, and it does it WITHOUT reading a clock: two planted guards rendezvous
# through marker files, each waiting a bounded time for the other to arrive.
# Under serial dispatch the first guard can never see the second -- it has not
# started -- so it exits 1 and the row is RED. Under concurrent dispatch both
# see each other and exit 0. No duration is asserted anywhere in this file:
# eleven wall-clock assertions have failed in a required check in this repo
# (scripts/check_no_timing_in_required.sh), and this table runs in the
# required `guard-tree` job.
#
#   bash scripts/tests/guard_tree_parallel_test.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GUARD_TREE="$REPO_ROOT/scripts/guard_tree.sh"
[ -f "$GUARD_TREE" ] || { printf 'FAIL: %s missing\n' "$GUARD_TREE"; exit 1; }

fails=0
ok()  { printf 'ok    %s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; fails=1; }

WORK="$(mktemp -d)" || exit 1
trap 'rm -rf "${WORK:?}"' EXIT

# new_fixture NAME -- a git repo carrying a copy of the dispatcher and nothing
# else. The caller plants scripts/check_*.sh guards, then calls commit_fixture.
new_fixture() {
    local d="$WORK/$1"
    mkdir -p "$d/scripts"
    cp "$GUARD_TREE" "$d/scripts/guard_tree.sh"
    printf '%s\n' "$d"
}

commit_fixture() {
    local d="$1"
    git -C "$d" init -q
    # hooksPath: a user-scope pre-commit hook (pmat) prints into this table's
    # output and is not part of what it measures.
    git -C "$d" -c core.hooksPath=/dev/null config core.hooksPath /dev/null
    git -C "$d" config user.email test@example.invalid
    git -C "$d" config user.name "guard_tree parallel case table"
    git -C "$d" add -A
    git -C "$d" -c commit.gpgsign=false commit -q -m fixture
}

# A planted guard that exits with $2 after printing $3. `--help` returns
# immediately and never mentions self-test: guard_tree.sh probes `--help` to
# decide whether a guard advertises a --self-test mode, and a fixture guard
# that ran its body on that probe would be counted twice.
plant_guard() {
    local path="$1" rc="$2" msg="$3"
    cat > "$path" <<EOF
#!/usr/bin/env bash
case "\${1:-}" in --help) printf 'usage: fixture guard\n'; exit 0 ;; esac
printf '%s\n' "$msg"
exit $rc
EOF
}

# ---------------------------------------------------------------------------
# Fixture A -- two passing guards and one failing one.
FA="$(new_fixture a)"
plant_guard "$FA/scripts/check_alpha.sh"   0 "alpha ran"
plant_guard "$FA/scripts/check_bravo.sh"   0 "bravo ran"
plant_guard "$FA/scripts/check_charlie.sh" 1 "charlie is the planted failure"
commit_fixture "$FA"

outA1="$(cd "$FA" && bash scripts/guard_tree.sh --no-cargo 2>&1)"; rcA1=$?
listA="$(cd "$FA" && bash scripts/guard_tree.sh --list --no-cargo 2>&1)"

# (a) the run reaches exactly the guard set --list --no-cargo advertises.
#     Derived from the run's own rows (PASS/FAIL/skipped), never from a list
#     written here: a dispatcher that dropped a guard under concurrency is
#     exactly what this row exists to catch.
run_set="$(printf '%s\n' "$outA1" \
    | sed -n -e 's/^\(PASS\|FAIL\)  \(scripts\/check_[A-Za-z0-9_.-]*\.sh\) .*/\2/p' \
             -e 's/^skipped: \(scripts\/check_[A-Za-z0-9_.-]*\.sh\) .*/\1/p' \
    | LC_ALL=C sort -u)"
list_set="$(printf '%s\n' "$listA" | grep -E '^scripts/check_' | LC_ALL=C sort -u)"
if [ "$run_set" = "$list_set" ] && [ -n "$list_set" ]; then
    ok "(a) --no-cargo reaches the same guard set --list --no-cargo advertises"
else
    bad "(a) run set [$(tr '\n' ' ' <<< "$run_set")] != list set [$(tr '\n' ' ' <<< "$list_set")]"
fi

# (b) the planted failure turns the run RED and is NAMED.
if [ "$rcA1" -ne 0 ] \
   && grep -q '^FAIL  scripts/check_charlie.sh \[run\]$' <<< "$outA1" \
   && grep -q 'charlie is the planted failure' <<< "$outA1"; then
    ok "(b) a planted failing guard turns the run RED and names it"
else
    bad "(b) rc=$rcA1, and the FAIL row / its captured output was not found"
fi

# (c) the planted passing guards are COUNTED -- a runner that silently
#     dropped them would still be red from (b).
if grep -q '^3 checks, 1 failed$' <<< "$outA1" \
   && grep -q '^PASS  scripts/check_alpha.sh \[run\]$' <<< "$outA1" \
   && grep -q '^PASS  scripts/check_bravo.sh \[run\]$' <<< "$outA1"; then
    ok "(c) the passing guards are counted: 3 checks, 1 failed"
else
    bad "(c) summary/PASS rows wrong: $(grep -E 'checks, [0-9]+ failed' <<< "$outA1")"
fi

# (d) two runs of the same fixture produce byte-identical output. Concurrency
#     must not reorder or interleave rows; the per-guard output is captured to
#     its own file and printed in universe order afterwards.
outA2="$(cd "$FA" && bash scripts/guard_tree.sh --no-cargo 2>&1)"
if [ "$outA1" = "$outA2" ]; then
    ok "(d) output ordering is deterministic across two runs"
else
    bad "(d) two runs differ:"
    diff <(printf '%s\n' "$outA1") <(printf '%s\n' "$outA2") | sed 's/^/      | /'
fi

# ---------------------------------------------------------------------------
# Fixture B -- the concurrency row. Two guards that can only both succeed if
# they run at the same time. No clock is asserted: the verdict is "did the
# peer arrive", and the bounded wait exists only so a serial dispatcher fails
# in ~20 s instead of hanging.
FB="$(new_fixture b)"
cat > "$FB/scripts/check_rendezvous_one.sh" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in --help) printf 'usage: fixture guard\n'; exit 0 ;; esac
d="${GUARD_RENDEZVOUS_DIR:?GUARD_RENDEZVOUS_DIR unset}"
: > "$d/one"
i=0
while [ "$i" -lt 200 ]; do
    [ -e "$d/two" ] && { printf 'one: peer arrived\n'; exit 0; }
    sleep 0.1; i=$((i + 1))
done
printf 'one: peer never arrived -- this dispatcher is serial\n'
exit 1
EOF
cat > "$FB/scripts/check_rendezvous_two.sh" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in --help) printf 'usage: fixture guard\n'; exit 0 ;; esac
d="${GUARD_RENDEZVOUS_DIR:?GUARD_RENDEZVOUS_DIR unset}"
: > "$d/two"
i=0
while [ "$i" -lt 200 ]; do
    [ -e "$d/one" ] && { printf 'two: peer arrived\n'; exit 0; }
    sleep 0.1; i=$((i + 1))
done
printf 'two: peer never arrived -- this dispatcher is serial\n'
exit 1
EOF
commit_fixture "$FB"

RV="$WORK/rendezvous"; mkdir -p "$RV"
outB="$(cd "$FB" && GUARD_RENDEZVOUS_DIR="$RV" bash scripts/guard_tree.sh --no-cargo 2>&1)"; rcB=$?
if [ "$rcB" -eq 0 ] && grep -q '^2 checks, 0 failed$' <<< "$outB"; then
    ok "(e) guards run CONCURRENTLY -- two rendezvous guards both see their peer"
else
    bad "(e) rc=$rcB -- the guards did not overlap (serial dispatch):"
    printf '%s\n' "$outB" | sed 's/^/      | /'
fi

# ---------------------------------------------------------------------------
# Fixture C -- fail-closed. A guard whose process dies without ever reporting
# a verdict (SIGKILL: no exit status a `set -e` could read, no output) is a
# FAILURE, never a skip and never a missing row. A parallel runner that reads
# each worker's verdict from a file gets this wrong by omission, which is the
# quiet way a required check stops being one.
FC="$(new_fixture c)"
plant_guard "$FC/scripts/check_alpha.sh" 0 "alpha ran"
cat > "$FC/scripts/check_selfkill.sh" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in --help) printf 'usage: fixture guard\n'; exit 0 ;; esac
kill -9 $$
EOF
commit_fixture "$FC"

outC="$(cd "$FC" && bash scripts/guard_tree.sh --no-cargo 2>&1)"; rcC=$?
if [ "$rcC" -ne 0 ] \
   && grep -q '^2 checks, 1 failed$' <<< "$outC" \
   && grep -q '^FAIL  scripts/check_selfkill.sh \[run\]$' <<< "$outC" \
   && grep -q '^PASS  scripts/check_alpha.sh \[run\]$' <<< "$outC"; then
    ok "(f) a guard that dies without a verdict is a FAILURE, not a missing row"
else
    bad "(f) rc=$rcC -- fail-closed row wrong:"
    printf '%s\n' "$outC" | sed 's/^/      | /'
fi

# ---------------------------------------------------------------------------
if [ "$fails" -ne 0 ]; then
    printf '\nGUARD_TREE PARALLEL CASE TABLE FAILED\n'
    exit 1
fi
printf '\nGUARD_TREE PARALLEL CASE TABLE PASSED (6/6)\n'
exit 0
