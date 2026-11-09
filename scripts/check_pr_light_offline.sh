#!/usr/bin/env bash
# check_pr_light_offline.sh -- pr-light installs and fetches nothing (GATE-JUNK-001 C6, infra#3711).
#
# A PR gate makes 0 network fetches and uses a pinned pre-provisioned toolchain. This reads the `pr-light:` job of
# .github/workflows/ci.yml and scripts/ci/pr_light_fast.sh, drops comment lines, and goes RED on any line that
# downloads or installs (curl, wget, cargo install, rustup install/update, pip, apt, npm, docker pull, the
# install_pinned_toolchain step), that gives a check a third state (continue-on-error, `|| true`), or that hands
# the guard-tree section back to the python fat_driver. The one network read the job keeps is `gh api` of THIS pull
# request's own body and file list (check_pr_closes_issue / check_pr_ont_delta), which is PR metadata, not a tool.
#
#   bash scripts/check_pr_light_offline.sh               # check the tree
#   bash scripts/check_pr_light_offline.sh --self-test   # every banned spelling must turn RED, the clean text GREEN
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BANNED='(^|[^[:alnum:]_.-])(curl|wget)([[:space:]]|$)|cargo[[:space:]]+(binstall|install)|rustup[[:space:]]+(toolchain[[:space:]]+install|install|update|self|component[[:space:]]+add|target[[:space:]]+add)|pip3?[[:space:]]+install|apt(-get)?[[:space:]]+(install|update)|npm[[:space:]]+(i|install|ci)([[:space:]]|$)|docker[[:space:]]+pull|install_pinned_toolchain\.sh|continue-on-error|\|\|[[:space:]]*(true|:)|--sections[[:space:]]+.[^ ]*guard-tree'

# pr_light_block <ci.yml> -- the lines of the pr-light job with comment lines removed
pr_light_block() {
    awk '/^  pr-light:/{on=1;next} on&&/^  [A-Za-z0-9_-]+:/{on=0} on' "$1" | grep -vE '^[[:space:]]*#'
}

# scan <label> -- stdin is the text; prints each banned line, returns 1 if there is one
scan() {
    local hits
    hits="$(grep -nE "$BANNED" | grep -vE '^[0-9]+:[[:space:]]*#' || true)"
    [ -z "$hits" ] && return 0
    printf '%s\n' "$hits" | sed "s|^|FAIL  $1: |"
    return 1
}

check_tree() {
    local rc=0 blk
    blk="$(pr_light_block "$REPO_ROOT/.github/workflows/ci.yml")"
    [ -n "$blk" ] || { echo "FAIL  the pr-light job is not in ci.yml; a gate that cannot be read is not checked" >&2; return 2; }
    printf '%s\n' "$blk" | scan "ci.yml pr-light" || rc=1
    grep -vE '^[[:space:]]*#' "$REPO_ROOT/scripts/ci/pr_light_fast.sh" | scan "scripts/ci/pr_light_fast.sh" || rc=1
    [ "$rc" -eq 0 ] && echo "ok    pr-light installs and fetches nothing"
    return "$rc"
}

self_test() {
    local pass=0 fail=0 sp
    row() { # row <label> <text> <want rc>
        local got=0
        printf '%s\n' "$2" | scan t > /dev/null || got=1
        if [ "$got" = "$3" ]; then pass=$((pass + 1)); printf '  ok    %s\n' "$1"; else fail=$((fail + 1)); printf '  FAIL  %s want=%s got=%s\n' "$1" "$3" "$got"; fi
    }
    for sp in 'curl -fsSL -o jq https://x/jq' 'wget https://x' 'cargo install ripgrep' 'cargo binstall foo' 'rustup toolchain install 1.93.0' \
        'rustup update' 'pip install yq' 'pip3 install yq' 'apt-get install jq' 'npm ci' 'docker pull img' 'bash scripts/ci/install_pinned_toolchain.sh' \
        'continue-on-error: true' 'bash x.sh || true' "python3 scripts/ci/fat_driver.py run --sections 'guard-tree,pr-review-shadow'"; do
        row "RED on: $sp" "$sp" 1
    done
    row "GREEN on a plain check" 'bash scripts/ci/pr_light_fast.sh' 0
    row "GREEN on the review sections' driver call" "python3 scripts/ci/fat_driver.py run --sections 'pr-review-shadow,pr-review-sign'" 0
    row "GREEN on gh api of this PR" 'gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}" --jq .body' 0
    printf 'check_pr_light_offline self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    '') check_tree ;;
    *) echo "usage: $(basename "$0") [--self-test]" >&2; exit 2 ;;
esac
