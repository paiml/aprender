#!/usr/bin/env bash
#
# predict_merge_test.sh -- self-test for scripts/predict_merge.sh (BSE-14,
# docs/specifications/build-system-enhancement.md §4 wave 3, infra repo).
#
# Every case builds its OWN throwaway fixture repo with a real, LOCAL bare
# repo as `origin` (created via `git clone --bare`/`git fetch`, never `git
# push` and never a real network address) -- so `git fetch origin main`
# inside predict_merge.sh is exercised for real, including the failure path
# (an origin url pointed at a path that does not exist).
#
# Deliberately NOT `set -e`: this file accumulates PASS/FAIL in $PASS/$FAIL
# via check() rather than aborting the whole run on the first RED case (most
# of the cases below assert something that is expected to fail against a
# mutant). `set -u` and `pipefail` are still on.
set -uo pipefail

REAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
REAL_SCRIPT="$REAL_ROOT/scripts/predict_merge.sh"

PASS=0
FAIL=0

check() {
    local desc="$1" ok="$2"
    if [ "$ok" -eq 0 ]; then
        echo "PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $desc"
        FAIL=$((FAIL + 1))
    fi
}

TD="$(mktemp -d)" || exit 1
cleanup_all() {
    rm -rf "${TD:?TD must be set}"
}
trap cleanup_all EXIT

SEED="$TD/seed"
BARE="$TD/origin.git"
REPO="$TD/repo"
export GUARD_PWD_LOG="$TD/guard-pwd.log"
export GATE_PWD_LOG="$TD/gate-pwd.log"

# --- a fake scripts/guard_tree.sh and scripts/gate_touched_crates.sh that
#     record the $PWD they were invoked from, so the test can assert
#     predict_merge.sh ran them against the MERGE tree, not the working tree.
write_fixture_helpers() {
    local dir="$1"
    mkdir -p "$dir/scripts"
    cat > "$dir/scripts/guard_tree.sh" <<'EOF'
#!/usr/bin/env bash
echo "$PWD" >> "${GUARD_PWD_LOG:?GUARD_PWD_LOG must be set}"
exit 0
EOF
    cat > "$dir/scripts/gate_touched_crates.sh" <<'EOF'
#!/usr/bin/env bash
echo "$PWD" >> "${GATE_PWD_LOG:?GATE_PWD_LOG must be set}"
exit 0
EOF
    chmod +x "$dir/scripts/guard_tree.sh" "$dir/scripts/gate_touched_crates.sh"
}

# --- seed repo: becomes the initial content of the bare "origin".
mkdir -p "$SEED"
git init -q -b main "$SEED" || exit 1
git -C "$SEED" config user.email t@t
git -C "$SEED" config user.name t
git -C "$SEED" config core.hooksPath /dev/null
write_fixture_helpers "$SEED"
echo "root" > "$SEED/f.txt"
# Mirrors the real repo's .gitignore (BSE-14): .predict/ is the script's own
# output, and must not itself register as "working tree is dirty".
printf '/.predict/\n' > "$SEED/.gitignore"
git -C "$SEED" add -A
git -C "$SEED" commit -qm root

git clone --bare -q "$SEED" "$BARE" || exit 1

# advance_origin_main -- moves the BARE origin's main ref forward one commit,
# via `git fetch` run FROM the bare repo (never `git push`): a scratch clone
# makes the new commit, and the bare repo fetches that ref into its own
# refs/heads/main. Genuinely exercises `git fetch origin main` detecting
# real movement, with no real network involved.
advance_origin_main() {
    local adv="$TD/advancer"
    rm -rf "${adv:?adv must be set}"
    git clone -q "$BARE" "$adv" || return 1
    git -C "$adv" config user.email t@t
    git -C "$adv" config user.name t
    git -C "$adv" config core.hooksPath /dev/null
    echo "advanced" >> "$adv/f.txt"
    git -C "$adv" commit -qam "advance origin main" >/dev/null || return 1
    git -C "$BARE" fetch -q "$adv" main:main || return 1
}

# reset_bare_to_seed -- undoes any advance_origin_main from a prior case.
reset_bare_to_seed() {
    rm -rf "${BARE:?BARE must be set}"
    git clone --bare -q "$SEED" "$BARE" || exit 1
}

# fresh_case -- rebuilds $REPO from scratch on branch "topic", one commit
# ahead of a freshly-reset origin/main, clean tree, and a fresh
# non-check predict record. Returns non-zero if the baseline predict itself
# did not succeed (a fixture bug, not a case under test).
fresh_case() {
    reset_bare_to_seed
    rm -rf "${REPO:?REPO must be set}"
    rm -f "$GUARD_PWD_LOG" "$GATE_PWD_LOG"
    git clone -q "$BARE" "$REPO" || return 1
    git -C "$REPO" config user.email t@t
    git -C "$REPO" config user.name t
    git -C "$REPO" config core.hooksPath /dev/null
    git -C "$REPO" checkout -q -b topic || return 1
    echo "topic change" >> "$REPO/f.txt"
    git -C "$REPO" add -A
    git -C "$REPO" commit -qm "topic commit" || return 1
    ( cd "$REPO" && bash "$REAL_SCRIPT" ) >"$TD/baseline.out" 2>&1
    [ $? -eq 0 ]
}

run_check() {
    ( cd "$REPO" && bash "$REAL_SCRIPT" --check )
}

run_check_with() {
    # run_check_with <script-path> -- like run_check but against an
    # arbitrary (e.g. mutant) copy of predict_merge.sh.
    local script="$1"
    ( cd "$REPO" && bash "$script" --check )
}

# ============================================================ Case 1 ======
if fresh_case; then
    out="$(run_check 2>&1)"; rc=$?
    check "fresh record: --check exits 0" "$([ "$rc" -eq 0 ] && echo 0 || echo 1)"
    logged_guard="$(tail -n1 "$GUARD_PWD_LOG" 2>/dev/null || true)"
    [ -n "$logged_guard" ] && [ "$logged_guard" != "$REPO" ]
    check "predict ran guard_tree.sh against the MERGE worktree, not the working tree" $?
    logged_gate="$(tail -n1 "$GATE_PWD_LOG" 2>/dev/null || true)"
    [ -n "$logged_gate" ] && [ "$logged_gate" != "$REPO" ]
    check "predict ran gate_touched_crates.sh against the MERGE worktree, not the working tree" $?
else
    check "fresh_case fixture: baseline predict itself succeeded" 1
    cat "$TD/baseline.out"
fi

# ============================================================ Case 2 ======
if fresh_case; then
    advance_origin_main
    out="$(run_check 2>&1)"; rc=$?
    check "origin/main moved: --check exits 3" "$([ "$rc" -eq 3 ] && echo 0 || echo 1)"
    case "$out" in
        *'origin/main moved'*) check "origin/main moved: reason names the movement" 0 ;;
        *) check "origin/main moved: reason names the movement" 1 ;;
    esac
else
    check "fresh_case fixture (case 2) succeeded" 1
fi

# ============================================================ Case 3 ======
if fresh_case; then
    git -C "$REPO" commit -q --allow-empty -m "extra commit on HEAD"
    out="$(run_check 2>&1)"; rc=$?
    check "commit on HEAD: --check exits 3" "$([ "$rc" -eq 3 ] && echo 0 || echo 1)"
    case "$out" in
        *'HEAD changed'*) check "commit on HEAD: reason names HEAD movement" 0 ;;
        *) check "commit on HEAD: reason names HEAD movement" 1 ;;
    esac
else
    check "fresh_case fixture (case 3) succeeded" 1
fi

# ============================================================ Case 4 ======
if fresh_case; then
    echo "dirty" >> "$REPO/f.txt"
    out="$(run_check 2>&1)"; rc=$?
    check "dirty tree: --check exits 3" "$([ "$rc" -eq 3 ] && echo 0 || echo 1)"
    case "$out" in
        *'dirty'*) check "dirty tree: reason names the dirty tree" 0 ;;
        *) check "dirty tree: reason names the dirty tree" 1 ;;
    esac
else
    check "fresh_case fixture (case 4) succeeded" 1
fi

# ============================================================ Case 5 ======
# Unreachable origin: point origin at a path that does not exist. This must
# fail the very first step (git fetch origin main) with exit 4, distinct
# from every "stale record" refusal (exit 3) above.
reset_bare_to_seed
rm -rf "${REPO:?REPO must be set}"
rm -f "$GUARD_PWD_LOG" "$GATE_PWD_LOG"
git clone -q "$BARE" "$REPO" || exit 1
git -C "$REPO" config user.email t@t
git -C "$REPO" config user.name t
git -C "$REPO" config core.hooksPath /dev/null
git -C "$REPO" checkout -q -b topic
git -C "$REPO" remote set-url origin "$TD/does-not-exist.git"
out="$( ( cd "$REPO" && bash "$REAL_SCRIPT" ) 2>&1 )"; rc=$?
check "unreachable origin: exits 4 (distinct from 3)" "$([ "$rc" -eq 4 ] && echo 0 || echo 1)"

# ============================================================ Case 6 ======
# Two concurrent predict runs must never collide and must never leave a
# temp worktree (or its scratch directory) behind.
if fresh_case; then
    SCRATCH="$TD/tmproot"
    mkdir -p "$SCRATCH"
    ( cd "$REPO" && TMPDIR="$SCRATCH" bash "$REAL_SCRIPT" >"$TD/c1.out" 2>&1 ) &
    p1=$!
    ( cd "$REPO" && TMPDIR="$SCRATCH" bash "$REAL_SCRIPT" >"$TD/c2.out" 2>&1 ) &
    p2=$!
    wait "$p1"; rc1=$?
    wait "$p2"; rc2=$?
    check "two concurrent predicts: both exited 0 or 1 (never crashed)" "$( { [ "$rc1" -eq 0 ] || [ "$rc1" -eq 1 ]; } && { [ "$rc2" -eq 0 ] || [ "$rc2" -eq 1 ]; } && echo 0 || echo 1 )"
    leftover="$(find "$SCRATCH" -mindepth 1 2>/dev/null | wc -l | tr -d ' ')"
    check "two concurrent predicts: no scratch dirs left behind (found $leftover)" "$([ "$leftover" -eq 0 ] && echo 0 || echo 1)"
    extra_worktrees="$(git -C "$REPO" worktree list --porcelain | grep -c '^worktree ')"
    check "two concurrent predicts: only the main worktree remains (found $extra_worktrees)" "$([ "$extra_worktrees" -eq 1 ] && echo 0 || echo 1)"
else
    check "fresh_case fixture (case 6) succeeded" 1
fi

# ============================================================ Mutations ===
# Each of the three freshness checks is independently load-bearing: a mutant
# with exactly ONE removed must let the MATCHING case pass (never exit 3),
# proving that check -- not some other path -- is what refuses that case.
mutant_for() {
    # mutant_for <marker> -- writes a copy of predict_merge.sh with the
    # BEGIN/END <marker> block deleted, prints its path.
    local marker="$1"
    local out="$TD/mutant-${marker}.sh"
    sed "/# BEGIN freshness-check-${marker}/,/# END freshness-check-${marker}/d" \
        "$REAL_SCRIPT" > "$out"
    chmod +x "$out"
    printf '%s' "$out"
}

mutant_main_moved="$(mutant_for main-moved)"
mutant_head_mismatch="$(mutant_for head-mismatch)"
mutant_tree_dirty="$(mutant_for tree-dirty)"

for m in "$mutant_main_moved" "$mutant_head_mismatch" "$mutant_tree_dirty"; do
    grep -q 'freshness-check' "$m"
    check "mutant $(basename "$m") still contains the OTHER two freshness checks" $?
done

if fresh_case; then
    advance_origin_main
    mrc="$(run_check_with "$mutant_main_moved" >/dev/null 2>&1; echo $?)"
    check "mutation (no main-moved check): origin/main-moved case no longer exits 3 (RED, as required)" "$([ "$mrc" -ne 3 ] && echo 0 || echo 1)"
else
    check "fresh_case fixture (mutation 1) succeeded" 1
fi

if fresh_case; then
    git -C "$REPO" commit -q --allow-empty -m "extra commit on HEAD"
    mrc="$(run_check_with "$mutant_head_mismatch" >/dev/null 2>&1; echo $?)"
    check "mutation (no head-mismatch check): commit-on-HEAD case no longer exits 3 (RED, as required)" "$([ "$mrc" -ne 3 ] && echo 0 || echo 1)"
else
    check "fresh_case fixture (mutation 2) succeeded" 1
fi

if fresh_case; then
    echo "dirty" >> "$REPO/f.txt"
    mrc="$(run_check_with "$mutant_tree_dirty" >/dev/null 2>&1; echo $?)"
    check "mutation (no tree-dirty check): dirty-tree case no longer exits 3 (RED, as required)" "$([ "$mrc" -ne 3 ] && echo 0 || echo 1)"
else
    check "fresh_case fixture (mutation 3) succeeded" 1
fi

echo ""
echo "$((PASS + FAIL)) checks, $FAIL failed"
[ "$FAIL" -eq 0 ]
