#!/usr/bin/env bash
# pr_light.sh -- `make pr-light`: the cargo-free half of CI's `pr-light` job, run locally BEFORE you push.
#
# Operator C2 (2026-11-03 07:05Z): "checks before review". A PR that is red in pr-light cannot be armed, and the
# usual reds (a new `grep | grep -q` site, a literal count in scripts/release, a guard case table) are found in
# seconds here instead of ~10 minutes in CI. Every step runs and every failure is reported, like guard_tree.sh.
#
#   bash scripts/pr_light.sh              # run every step, exit 1 if any failed
#   bash scripts/pr_light.sh --list       # print the steps and exit 0
#   bash scripts/pr_light.sh --self-test
#   bash scripts/pr_light.sh --mutants
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SECTIONS_FILE="${PR_LIGHT_SECTIONS:-$REPO_ROOT/ci/sections.yml}"

# name|command -- the steps mirror .github/workflows/ci.yml `pr-light` (cheap checks + the cargo-free guard tree).
steps() {
    cat << 'STEPS'
fmt|cargo fmt --all -- --check
fail-fast self-test|python3 scripts/ci/fail_fast.py self-test
falsification ids|python3 scripts/ci/fail_fast.py ids --base origin/main
guard-tree case table|bash scripts/tests/guard_tree_test.sh
guard-tree (cargo-free)|bash scripts/guard_tree.sh --no-cargo
readme contract count|bash scripts/readme_sync.sh --check
readme contract claims|bash scripts/check_readme_claims.sh --claim contract_count
STEPS
}

run_steps() { # run_steps -> prints one row per step; returns 1 if any failed
    local failed=0 n=0 name cmd rc
    while IFS='|' read -r name cmd; do
        [ -n "$name" ] || continue
        n=$((n + 1))
        set +e
        (cd "$REPO_ROOT" && bash -c "$cmd") > /dev/null 2>&1
        rc=$?
        set -e
        if [ "$rc" -eq 0 ]; then printf 'ok    %s\n' "$name"; else printf 'FAIL  %s (rc=%s) -- rerun: %s\n' "$name" "$rc" "$cmd"; failed=1; fi
    done
    [ "$n" -gt 0 ] || { printf 'NO-GO: zero steps ran; that is not a pass\n' >&2; return 2; }
    return "$failed"
}

safe_rm_td() {
    local d="${1:-}"
    case "$d" in
        /tmp/pr-light-*|"${TMPDIR:-/tmp}"/pr-light-*)
            [ -n "$d" ] && [ "$d" != "/" ] && rm -rf -- "$d"
            ;;
    esac
}

ci_readme_step_block() {
    awk '
        /^  guard-tree-steps:/ { injob = 1; next }
        injob && /^  [a-zA-Z0-9_-]+:/ { exit }
        injob && /name: "README CONTRACT_COUNT matches measured contracts/ { instep = 1; next }
        instep && /^      - / { exit }
        instep && /^        run:/ { inrun = 1; next }
        inrun && /^          / { sub(/^          /, ""); print; next }
        inrun { exit }
    ' "$SECTIONS_FILE"
}

ci_readme_step_ok() {
    local block td rc1 rc2
    [ -f "$SECTIONS_FILE" ] || return 1
    block="$(ci_readme_step_block)"
    [ -n "$block" ] || return 1
    [ "$(printf '%s\n' "$block" | grep -c 'scripts/readme_sync\.sh --check')" -eq 1 ] || return 1
    [ "$(printf '%s\n' "$block" | grep -c 'scripts/check_readme_claims\.sh --claim contract_count')" -eq 1 ] || return 1
    td="$(mktemp -d "${TMPDIR:-/tmp}/pr-light-ci.XXXXXX")" || return 1
    mkdir -p "$td/scripts"
    printf '#!/usr/bin/env bash\nexit 1\n' > "$td/scripts/readme_sync.sh"
    printf '#!/usr/bin/env bash\nexit 0\n' > "$td/scripts/check_readme_claims.sh"
    set +e
    (cd "$td" && bash -c "$block") > /dev/null 2>&1
    rc1=$?
    printf '#!/usr/bin/env bash\nexit 0\n' > "$td/scripts/readme_sync.sh"
    printf '#!/usr/bin/env bash\nexit 1\n' > "$td/scripts/check_readme_claims.sh"
    (cd "$td" && bash -c "$block") > /dev/null 2>&1
    rc2=$?
    set -e
    safe_rm_td "$td"
    [ "$rc1" -ne 0 ] && [ "$rc2" -ne 0 ]
}

self_test_cases() {
    local pass=0 fail=0 out rc ci_rc=0
    row() { if [ "$2" = "$3" ]; then pass=$((pass + 1)); printf '  ok    %s\n' "$1"; else fail=$((fail + 1)); printf '  FAIL  %s want=%s got=%s\n' "$1" "$3" "$2"; fi; }
    row "the guard tree is a step" "$(steps | grep -c 'guard_tree.sh --no-cargo')" 1
    row "fmt is a step" "$(steps | grep -c '^fmt|')" 1
    row "readme_sync --check is a step" "$(steps | grep -c 'readme_sync\.sh --check')" 1
    row "check_readme_claims --claim contract_count is a step" "$(steps | grep -c 'check_readme_claims\.sh --claim contract_count')" 1
    if ci_readme_step_ok; then ci_rc=0; else ci_rc=1; fi
    row "ci/sections.yml runs both readme checks fail-fast" "$ci_rc" 0
    set +e; out="$(printf 'good|true\nbad|false\nlast|true\n' | run_steps)"; rc=$?; set -e
    row "one failing step makes the run fail" "$rc" 1
    row "a failure does not stop later steps" "$(printf '%s\n' "$out" | grep -c '^ok    last')" 1
    row "the failing step is named" "$(printf '%s\n' "$out" | grep -c '^FAIL  bad')" 1
    set +e; printf 'good|true\n' | run_steps > /dev/null; rc=$?; set -e
    row "all green exits 0" "$rc" 0
    set +e; printf '' | run_steps > /dev/null; rc=$?; set -e
    row "zero steps is not a verdict" "$rc" 2
    printf 'pr_light self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

mutants() {
    local td m_self m_sec bad=0 count=0
    td="$(mktemp -d "${TMPDIR:-/tmp}/pr-light-mut.XXXXXX")" || return 2
    m_self="$td/pr_light.sh"
    m_sec="$td/sections.yml"
    run_mut() {
        local label="$1" script="$2" sec="$3" rc=0
        count=$((count + 1))
        set +e
        PR_LIGHT_SKIP_MUTANTS=1 PR_LIGHT_SECTIONS="$sec" bash "$script" --self-test > /dev/null 2>&1
        rc=$?
        set -e
        if [ "$rc" -ne 0 ]; then
            printf '  ok    mutant killed: %s\n' "$label"
        else
            printf '  FAIL  mutant survived: %s\n' "$label"
            bad=1
        fi
    }
    check_applied() {
        if cmp -s "$1" "$2"; then
            printf '  FAIL  mutant changed nothing: %s\n' "$3"
            bad=1
            return 1
        fi
        return 0
    }

    sed '/^readme contract count|/d' "$0" > "$m_self"
    check_applied "$0" "$m_self" "pr-light-drop-readme-sync" && run_mut "pr-light-drop-readme-sync" "$m_self" "$SECTIONS_FILE"

    sed '/^readme contract claims|/d' "$0" > "$m_self"
    check_applied "$0" "$m_self" "pr-light-drop-readme-claims" && run_mut "pr-light-drop-readme-claims" "$m_self" "$SECTIONS_FILE"

    sed 's/failed=1/failed=0/' "$0" > "$m_self"
    check_applied "$0" "$m_self" "pr-light-swallow-failure" && run_mut "pr-light-swallow-failure" "$m_self" "$SECTIONS_FILE"

    sed 's/setsid --wait bash scripts\/readme_sync\.sh --check && //' "$SECTIONS_FILE" > "$m_sec"
    check_applied "$SECTIONS_FILE" "$m_sec" "ci-drop-readme-sync" && run_mut "ci-drop-readme-sync" "$0" "$m_sec"

    sed 's/ && setsid --wait bash scripts\/check_readme_claims\.sh --claim contract_count//' "$SECTIONS_FILE" > "$m_sec"
    check_applied "$SECTIONS_FILE" "$m_sec" "ci-drop-readme-claims" && run_mut "ci-drop-readme-claims" "$0" "$m_sec"

    sed -e '/set -euo pipefail/{N;s/set -euo pipefail\n\( *setsid --wait bash scripts\/readme_sync\.sh --check\) && \(setsid --wait bash scripts\/check_readme_claims\.sh --claim contract_count\)/\1\n          \2/;}' "$SECTIONS_FILE" > "$m_sec"
    check_applied "$SECTIONS_FILE" "$m_sec" "ci-drop-set-e-and-and" && run_mut "ci-drop-set-e-and-and" "$0" "$m_sec"

    safe_rm_td "$td"
    printf 'pr_light mutants: %s planted, %s survived\n' "$count" "$bad"
    [ "$bad" -eq 0 ]
}

self_test() {
    self_test_cases || return 1
    if [ "${PR_LIGHT_SKIP_MUTANTS:-0}" != 1 ]; then
        mutants || return 1
    fi
}

case "${1:-}" in
    --self-test) self_test ;;
    --mutants) mutants ;;
    --list) steps | cut -d'|' -f1 ;;
    '') printf '== pr-light (local, cargo-free half of CI) ==\n'; steps | run_steps ;;
    *) printf 'usage: %s [--list|--self-test|--mutants]\n' "$(basename "$0")" >&2; exit 2 ;;
esac
