#!/usr/bin/env bash
#
# guard_tree_test.sh -- acceptance test for scripts/guard_tree.sh (BSE-01,
# PMAT-1062).
#
# Asserts:
#   1. guard_tree.sh runs EVERY guard in a fixture and reports EVERY
#      failure, not just the first (no fail-fast), exiting 1.
#   2. A fail-fast MUTANT of guard_tree.sh (stops at the first failure, the
#      exact regression this script exists to catch) is shown to go RED
#      against the very same fixture and assertion -- proving check #1
#      actually discriminates rather than passing by construction.
#   3. `--list` equals `git ls-files 'scripts/check_*.sh'` over the real
#      repo.
#   4. `--list --no-cargo` has the same count as
#      `git ls-files 'scripts/check_*.sh' | grep -LE '(^|[^a-z_-])cargo '`.
#   5. The summary line reads `N checks, M failed`.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || exit 1
GUARD_TREE="$REPO_ROOT/scripts/guard_tree.sh"

total=0
failed=0

pass_row() {
    total=$((total + 1))
    printf 'PASS  %s\n' "$1"
}

fail_row() {
    total=$((total + 1))
    failed=$((failed + 1))
    printf 'FAIL  %s\n' "$1"
    [ -n "${2:-}" ] && printf '      | %s\n' "$2"
}

cleanup_dirs=""
on_exit() {
    for d in $cleanup_dirs; do
        rm -rf "${d:?on_exit: refusing to rm -rf an empty path}"
    done
}
trap on_exit EXIT

# ---------------------------------------------------------------------------
# Fixture: a throwaway git repo carrying a real copy of guard_tree.sh plus
# one passing and two deliberately failing check_*.sh guards.
# ---------------------------------------------------------------------------
make_fixture() {
    dir="$(mktemp -d)" || exit 1
    git -C "$dir" init -q
    git -C "$dir" config user.email test@example.invalid
    git -C "$dir" config user.name "guard_tree_test"
    mkdir -p "$dir/scripts"
    cp "$GUARD_TREE" "$dir/scripts/guard_tree.sh"
    chmod +x "$dir/scripts/guard_tree.sh"

    cat >"$dir/scripts/check_good.sh" <<'SH'
#!/usr/bin/env bash
exit 0
SH

    cat >"$dir/scripts/check_fail_alpha.sh" <<'SH'
#!/usr/bin/env bash
echo "check_fail_alpha: deliberately red"
exit 1
SH

    cat >"$dir/scripts/check_fail_beta.sh" <<'SH'
#!/usr/bin/env bash
echo "check_fail_beta: deliberately red"
exit 1
SH

    chmod +x "$dir"/scripts/check_*.sh
    git -C "$dir" add -A
    git -C "$dir" commit -q -m fixture
    printf '%s\n' "$dir"
}

# make_fail_fast_mutant SRC DST -- rewrite guard_tree.sh so it exits on the
# FIRST failed guard instead of collecting every result. This is the exact
# shape of the bug guard_tree.sh exists to fix (GitHub Actions steps stopping
# at the first red step), reintroduced deliberately so we can prove the
# assertion above would have caught it.
make_fail_fast_mutant() {
    src="$1"
    dst="$2"
    sed 's/failed=\$((failed + 1))/failed=$((failed + 1)); exit 1/' "$src" >"$dst"
    chmod +x "$dst"
}

# ---------------------------------------------------------------------------
# 1. Run-all / report-all, against the real guard_tree.sh.
# ---------------------------------------------------------------------------
fixture="$(make_fixture)"
cleanup_dirs="$cleanup_dirs $fixture"

out="$(cd "$fixture" && bash scripts/guard_tree.sh 2>&1)"
rc=$?
n_alpha="$(grep -c 'check_fail_alpha.sh' <<<"$out")"
n_beta="$(grep -c 'check_fail_beta.sh' <<<"$out")"

if [ "$rc" -eq 1 ] && [ "${n_alpha:-0}" -gt 0 ] && [ "${n_beta:-0}" -gt 0 ]; then
    pass_row "runs every guard and reports every failure (both names present, exit 1)"
else
    fail_row "runs every guard and reports every failure" \
        "rc=$rc n_alpha=${n_alpha:-0} n_beta=${n_beta:-0}"
fi

# ---------------------------------------------------------------------------
# 2. The fail-fast mutation must go RED against the SAME fixture and the
#    SAME assertion (poka-yoke: prove #1 actually discriminates).
# ---------------------------------------------------------------------------
mutant="$fixture/scripts/guard_tree_failfast.sh"
make_fail_fast_mutant "$fixture/scripts/guard_tree.sh" "$mutant"

mutant_out="$(cd "$fixture" && bash scripts/guard_tree_failfast.sh 2>&1)"
m_alpha="$(grep -c 'check_fail_alpha.sh' <<<"$mutant_out")"
m_beta="$(grep -c 'check_fail_beta.sh' <<<"$mutant_out")"

# The mutant stops at the FIRST failing guard git ls-files hands back, so it
# can never report BOTH fail_alpha and fail_beta -- if it did, the mutation
# failed to reintroduce fail-fast and this row itself must fail loudly.
if [ "${m_alpha:-0}" -gt 0 ] && [ "${m_beta:-0}" -gt 0 ]; then
    fail_row "fail-fast mutation must go RED (mutation did not discriminate)" \
        "mutant reported both names: m_alpha=${m_alpha:-0} m_beta=${m_beta:-0}"
else
    pass_row "fail-fast mutation goes RED (misses at least one failing guard name)"
fi

# ---------------------------------------------------------------------------
# 3. --list equals git ls-files 'scripts/check_*.sh' over the real repo.
# ---------------------------------------------------------------------------
real_list="$(cd "$REPO_ROOT" && git ls-files 'scripts/check_*.sh' | sort)"
guard_list="$(cd "$REPO_ROOT" && bash scripts/guard_tree.sh --list | sort)"

if [ "$real_list" = "$guard_list" ]; then
    pass_row "--list equals git ls-files 'scripts/check_*.sh'"
else
    fail_row "--list equals git ls-files 'scripts/check_*.sh'" "lists differ"
fi

# ---------------------------------------------------------------------------
# 4. --list --no-cargo count matches the grep -LE count, over the real repo.
# ---------------------------------------------------------------------------
expected_no_cargo="$(cd "$REPO_ROOT" && git ls-files 'scripts/check_*.sh' \
    | xargs -r grep -LE '(^|[^a-z_-])cargo ' | wc -l | tr -d ' ')"
actual_no_cargo="$(cd "$REPO_ROOT" && bash scripts/guard_tree.sh --list --no-cargo \
    | wc -l | tr -d ' ')"

if [ "$expected_no_cargo" = "$actual_no_cargo" ]; then
    pass_row "--no-cargo count matches grep -LE '(^|[^a-z_-])cargo ' ($actual_no_cargo)"
else
    fail_row "--no-cargo count matches grep -LE count" \
        "expected=$expected_no_cargo actual=$actual_no_cargo"
fi

# ---------------------------------------------------------------------------
# 5. Summary line reads "N checks, M failed".
# ---------------------------------------------------------------------------
n_summary="$(grep -cE '^[0-9]+ checks, [0-9]+ failed$' <<<"$out")"
if [ "${n_summary:-0}" -gt 0 ]; then
    pass_row 'summary line reads "N checks, M failed"'
else
    fail_row 'summary line reads "N checks, M failed"' "no matching line in: $out"
fi

printf '%d checks, %d failed\n' "$total" "$failed"
if [ "$failed" -gt 0 ]; then
    exit 1
fi
exit 0
