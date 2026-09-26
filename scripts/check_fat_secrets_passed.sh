#!/usr/bin/env bash
# check_fat_secrets_passed.sh - every secret a section reads reaches the section.
#
# WHY (#4441 follow-up, found on #4431 2026-09-26). Since #4433 the job bodies live
# in ci/sections.yml and run inside ci.yml's fat jobs. A section's `secrets.NAME`
# is served by scripts/ci/fat_driver.py from the fat job's env var
# FAT_SECRET_NAME, which ci.yml must declare by hand as
#     FAT_SECRET_NAME: ${{ secrets.NAME }}
# #4441 declared FAT_SECRET_PR_REVIEW_SIGNING_KEY_B from secrets.PR_REVIEW_SIGNING_KEY_B
# - the real secret is ..._B64 - so `pr-review-sign` read an empty key, refused
# every unsigned receipt, and Arm 4 went RED on every new PR. Nothing checked
# that the name a section reads is the name the fat job passes.
#
# RULES
#   S1  every secrets.NAME on a non-comment line of ci/sections.yml (except GITHUB_TOKEN, which the driver
#       serves from the job token) has a line `FAT_SECRET_NAME: ${{ secrets.NAME }}`
#       in .github/workflows/ci.yml.
#   S2  every FAT_SECRET_KEY line in ci.yml reads secrets.KEY - the same name.
#
# Usage: check_fat_secrets_passed.sh [SECTIONS_YML CI_YML] | --self-test
set -euo pipefail

check() {
    local sections="$1" ci="$2" bad=0 name line key val
    while IFS= read -r name; do
        [ "$name" = GITHUB_TOKEN ] && continue
        if ! grep -qE "^[[:space:]]*FAT_SECRET_${name}:[[:space:]]*\\\$\\{\\{[[:space:]]*secrets\\.${name}[[:space:]]*\\}\\}[[:space:]]*$" "$ci"; then
            echo "FAIL S1: $sections reads secrets.$name but $ci passes no 'FAT_SECRET_$name: \${{ secrets.$name }}'"
            bad=1
        fi
    done < <(grep -vE '^[[:space:]]*#' "$sections" | grep -oE 'secrets\.[A-Za-z0-9_]+' | sed 's/^secrets\.//' | sort -u)
    while IFS= read -r line; do
        key=$(printf '%s\n' "$line" | sed -E 's/^[[:space:]]*FAT_SECRET_([A-Za-z0-9_]+):.*/\1/')
        val=$(printf '%s\n' "$line" | sed -nE 's/.*secrets\.([A-Za-z0-9_]+).*/\1/p')
        if [ "$key" != "$val" ]; then
            echo "FAIL S2: FAT_SECRET_$key reads secrets.${val:-<none>} - the section asking for secrets.$key gets another value"
            bad=1
        fi
    done < <(grep -E '^[[:space:]]*FAT_SECRET_[A-Za-z0-9_]+:' "$ci" || true)
    return "$bad"
}

self_test() {
    local fails=0 rc
    t=$(mktemp -d)  # global: the EXIT trap runs after this function returns
    trap 'rm -rf "${t:?}"' EXIT
    printf '      env:\n        K: ${{ secrets.PR_REVIEW_SIGNING_KEY_B64 }}\n        T: ${{ secrets.GITHUB_TOKEN }}\n' > "$t/sections.yml"
    # name | ci.yml body | expected rc
    row() {
        printf '%b' "$2" > "$t/ci.yml"
        set +e; check "$t/sections.yml" "$t/ci.yml" > "$t/out" 2>&1; rc=$?; set -e
        if [ "$rc" -eq "$3" ]; then echo "  ok    $1 (rc=$rc)"; else echo "  FAIL  $1 rc=$rc want $3"; sed 's/^/        /' "$t/out"; fails=1; fi
    }
    row passed_by_its_own_name   '      FAT_SECRET_PR_REVIEW_SIGNING_KEY_B64: ${{ secrets.PR_REVIEW_SIGNING_KEY_B64 }}\n' 0
    row the_4441_truncation      '      FAT_SECRET_PR_REVIEW_SIGNING_KEY_B: ${{ secrets.PR_REVIEW_SIGNING_KEY_B }}\n' 1
    row not_passed_at_all        '      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n' 1
    row key_and_value_disagree   '      FAT_SECRET_PR_REVIEW_SIGNING_KEY_B64: ${{ secrets.OTHER }}\n' 1
    row extra_mismatched_line    '      FAT_SECRET_PR_REVIEW_SIGNING_KEY_B64: ${{ secrets.PR_REVIEW_SIGNING_KEY_B64 }}\n      FAT_SECRET_X: ${{ secrets.Y }}\n' 1
    printf '        # secrets.ONLY_IN_A_COMMENT is never read\n        T: ${{ secrets.GITHUB_TOKEN }}\n' > "$t/sections.yml"
    row token_and_comment_need_none '      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}\n' 0
    if [ "$fails" -eq 0 ]; then echo "PASS: 6/6 rows"; else echo "FAIL: case table"; return 1; fi
}

if [ "${1:-}" = --self-test ]; then self_test; exit $?; fi
root=$(git rev-parse --show-toplevel)
check "${1:-$root/ci/sections.yml}" "${2:-$root/.github/workflows/ci.yml}" && echo "OK: every section secret is passed to the fat job under its own name"
