#!/usr/bin/env bash
#
# check_hardcoded_paths.sh — no contract may name a machine-specific path.
#
# WHY THIS EXISTS (#2532)
# -----------------------
# `pmat analyze hardcoded-paths -p . --fail-on-shipped` reported 324 findings in
# SHIPPED code on origin/main @ 5c08e771f — and 46 of them were in contracts/
# itself, the tier whose whole job is to make a defect impossible.
#
# The issue argued "all of these resolve on this host, which is the defect".
# Measured, it is worse than that: of the 91 distinct shipped paths only 31
# still exist here. `/home/<user>/src/aprender-worktrees/crux-spec` — the root of
# 17 crux golden-set paths — is gone. Those 17 falsification tests open with
#
#     GOLD=/home/<user>/src/aprender-worktrees/crux-spec/evidence/crux/…json
#     [ -f "$GOLD" ] || { echo "golden set absent"; exit 2; }
#
# so they self-skip on EVERY machine including the author's, while `pv validate`
# and `pv lint contracts/` both report PASS. A contract that cannot execute
# anywhere is not enforcement; it is a claim that reads like enforcement. That
# is the same failure mode as check_test_fixture_paths.sh's tests-gated-on-a-
# sibling-checkout, one tier up.
#
# WHAT IT CHECKS
# --------------
# Default mode: ZERO machine-specific absolute paths under contracts/. Not a
# ratchet — the tier is at zero as of this commit and there is no defensible
# number above zero for it, so there is deliberately no baseline file to raise.
# A path rooted in a named user's home is portable to exactly one machine.
#
#   PORTABLE, and used by the contracts this guard cleaned:
#     ${APR_CRUX_GOLDENS:-evidence/crux}/…   repo-relative with an override
#     ${APR_MODELS:?}/qwen….gguf             loud failure when unset
#     ${APR_LEADERBOARD_ROOT:?}/…            loud failure when unset
#     $HOME/.cache/…  ~/models/…             the invoking user's home
#     target/release/apr                     workspace-relative
#
# `--full` mode: the whole-tree ratchet over pmat's SHIPPED tier. Detection is
# pmat's — this script only holds the number. See "WHY TWO MODES".
#
# WHY TWO MODES / WHY --full IS NOT WIRED INTO CI
# -----------------------------------------------
# pmat owns this detector (pmat#1017) and re-implementing its tiering would be
# muda, so --full shells out to it and compares `.shipped_count`. But MEASURED
# on 2026-08-20: the clean-room pool that runs the blocking guards
# (16 × intel-clean-room, the only runners carrying the `clean-room` label)
# has pmat 3.31.0, in which `pmat analyze hardcoded-paths` does not exist:
#
#   $ ssh mac-server 'pmat --version; pmat analyze hardcoded-paths --help'
#   pmat 3.31.0
#   error: unrecognized subcommand 'hardcoded-paths'
#
# Wiring --full into the required gate today would red main on every PR. The
# alternatives are worse: `cargo install pmat || true` (book.yml:90 does this —
# a gate that cannot fail), or a cold `cargo install` inside a timeout-boxed
# job (the cargo-audit failure mode that evicted the merge queue).
#
# So the default mode is deliberately narrow, pmat-free, and BLOCKING; --full is
# the authoritative view and runs from `make tier3` and by hand. PROMOTE --full
# INTO ci.yml AS SOON AS THE CLEAN-ROOM FLEET CARRIES pmat >= 3.32.0 — that is
# one line next to the default-mode step, and this comment is the trigger.
#
#   bash scripts/check_hardcoded_paths.sh              # blocking check (contracts/)
#   bash scripts/check_hardcoded_paths.sh --self-test  # case table
#   bash scripts/check_hardcoded_paths.sh --full       # pmat shipped-tier ratchet
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SHIPPED_BASELINE="${REPO_ROOT}/scripts/hardcoded_path_shipped_baseline.txt"

# Scanned tier. Overridable so the self-test can point at a fixture tree.
CONTRACT_DIR="${CONTRACT_DIR:-${REPO_ROOT}/contracts}"
# Vacuity floor: 1778 contract files today. A scan that examined almost nothing
# must go RED, not print the same OK as a scan that examined everything.
MIN_CONTRACT_FILES="${MIN_CONTRACT_FILES:-1000}"
# Vacuity floor for --full: pmat reports files_scanned=14192 on this tree.
MIN_FILES_SCANNED="${MIN_FILES_SCANNED:-14000}"

# An absolute path rooted in a NAMED user's home. `$HOME/...`, `~/...`,
# `${VAR}/...`, `/tmp/...`, `/usr/...` and workspace-relative paths all pass:
# none of them names a machine.
PATTERN='/(home|Users)/[A-Za-z0-9_][A-Za-z0-9_.-]*/'

# This file quotes the defect shapes in its own header, so it must exclude
# itself and its fixtures the way check_pass_grep_anchored.sh does.
scan() {
    local dir="$1"
    find "$dir" -type f \( -name '*.yaml' -o -name '*.yml' \) 2>/dev/null \
        | LC_ALL=C sort \
        | while IFS= read -r f; do
              grep -HnoE "$PATTERN" "$f" 2>/dev/null
          done \
        | sed "s|^${REPO_ROOT}/||"
}

count_files() {
    find "$1" -type f \( -name '*.yaml' -o -name '*.yml' \) 2>/dev/null | grep -c . || true
}

# ---------------------------------------------------------------------------
# Case table. Fixtures live in scripts/lib/ rather than inline heredocs:
# bashrs parses an embedded heredoc as shell (the reason
# fixture_path_selftest_*.rs.txt were moved out of line).
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TD="$(mktemp -d)" || { printf 'FAIL: no temp dir\n' >&2; exit 1; }
    trap 'rm -rf "${TD:?}"' EXIT
    mkdir -p "$TD/contracts"
    cp "${REPO_ROOT}/scripts/lib/hardcoded_path_selftest_bad.yaml.txt"  "$TD/contracts/bad.yaml"
    cp "${REPO_ROOT}/scripts/lib/hardcoded_path_selftest_good.yaml.txt" "$TD/contracts/good.yaml"

    fails=0
    bad="$(scan "$TD/contracts" | grep -c 'bad.yaml' || true)"
    good="$(scan "$TD/contracts" | grep -c 'good.yaml' || true)"

    if [ "$bad" -eq 4 ]; then
        printf 'ok    4/4 defect shapes flagged (named home, other user, /Users, comment)\n'
    else
        printf 'FAIL  flagged %s of 4 defect shapes - guard is blind\n' "$bad"; fails=1
    fi

    # Non-vacuity control: the good fixture is dense with path-shaped text
    # (${VAR}, $HOME, ~, /tmp, /dev, /usr, repo-relative). If it were flagged,
    # "0 hits on contracts/" would mean nothing.
    if [ "$good" -eq 0 ]; then
        printf 'ok    0 false positives on 9 portable path shapes\n'
    else
        printf 'FAIL  %s false positive(s) on portable paths\n' "$good"; fails=1
    fi

    # The check must turn RED on the bad fixture, not merely report hits.
    if CONTRACT_DIR="$TD/contracts" MIN_CONTRACT_FILES=1 \
        bash "${BASH_SOURCE[0]}" >/dev/null 2>&1; then
        printf 'FAIL  guard exited 0 on a tree containing 4 machine-specific paths\n'; fails=1
    else
        printf 'ok    guard exits non-zero on a polluted tree\n'
    fi

    # And it must PASS on a tree that has only the good fixture: a guard that
    # always reds is as useless as one that never does.
    rm -f "$TD/contracts/bad.yaml"
    if CONTRACT_DIR="$TD/contracts" MIN_CONTRACT_FILES=1 \
        bash "${BASH_SOURCE[0]}" >/dev/null 2>&1; then
        printf 'ok    guard exits 0 on a clean tree\n'
    else
        printf 'FAIL  guard reds on a clean tree - it can never be satisfied\n'; fails=1
    fi

    # Fail-closed: a scan that measured nothing must not pass.
    if CONTRACT_DIR="$TD/contracts" MIN_CONTRACT_FILES=9999 \
        bash "${BASH_SOURCE[0]}" >/dev/null 2>&1; then
        printf 'FAIL  guard passed while scanning fewer files than its floor\n'; fails=1
    else
        printf 'ok    guard fails closed below the vacuity floor\n'
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED\n'; exit 0
fi

# ---------------------------------------------------------------------------
# --full: pmat's shipped tier, ratcheted. Detection is pmat's; this holds the
# number. Fails hard when pmat cannot do it, never silently.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--full" ]; then
    printf '=== shipped-tier ratchet via pmat (check_hardcoded_paths.sh --full) ===\n'
    command -v pmat >/dev/null 2>&1 || {
        printf 'FAIL: pmat not found. This mode is a thin wrapper over\n'
        printf '  pmat analyze hardcoded-paths (pmat#1017); it does not re-detect.\n'
        exit 1
    }
    command -v jq >/dev/null 2>&1 || { printf 'FAIL: jq not found.\n'; exit 1; }

    TD="$(mktemp -d)" || { printf 'FAIL: no temp dir\n' >&2; exit 1; }
    trap 'rm -rf "${TD:?}"' EXIT
    # Never read $? through a pipe (Verification Discipline #1).
    ( cd "$REPO_ROOT" && pmat analyze hardcoded-paths -p . -f json ) \
        > "$TD/out.json" 2> "$TD/err.txt"
    rc=$?
    if [ "$rc" -ne 0 ] || [ ! -s "$TD/out.json" ]; then
        printf 'FAIL: pmat analyze hardcoded-paths did not produce JSON (rc=%s).\n' "$rc"
        printf 'Needs pmat >= 3.32.0; 3.31.0 has no such subcommand.\n'
        sed 's|^|  |' "$TD/err.txt" | head -5
        exit 1
    fi

    shipped="$(jq -r '.shipped_count' "$TD/out.json")"
    files="$(jq -r '.files_scanned' "$TD/out.json")"
    case "$shipped$files" in ''|*[!0-9]*) printf 'FAIL: unparseable pmat JSON\n'; exit 1 ;; esac

    if [ "$files" -lt "$MIN_FILES_SCANNED" ]; then
        printf '\nFAIL (vacuity): pmat scanned only %s file(s), floor %s.\n' "$files" "$MIN_FILES_SCANNED"
        printf 'Fix the scan, not this number.\n'
        exit 1
    fi
    [ -f "$SHIPPED_BASELINE" ] || { printf 'FAIL: %s missing.\n' "$SHIPPED_BASELINE"; exit 1; }
    baseline="$(tr -d '[:space:]' < "$SHIPPED_BASELINE")"

    printf 'pmat scanned %s file(s); %s shipped finding(s), baseline %s\n' "$files" "$shipped" "$baseline"
    if [ "$shipped" -gt "$baseline" ]; then
        printf '\nFAIL: shipped machine-specific paths grew %s -> %s.\n' "$baseline" "$shipped"
        ( cd "$REPO_ROOT" && pmat analyze hardcoded-paths -p . --fail-on-shipped 2>&1 | head -40 )
        exit 1
    fi
    if [ "$shipped" -lt "$baseline" ]; then
        printf '\nImproved: %s -> %s. Lower %s to record it.\n' "$baseline" "$shipped" "$SHIPPED_BASELINE"
    fi
    printf 'PASS\n'
    exit 0
fi

# ---------------------------------------------------------------------------
# Default: contracts/ must be at ZERO.
# ---------------------------------------------------------------------------
printf '=== no contract may name a machine-specific path (check_hardcoded_paths.sh) ===\n'

scanned="$(count_files "$CONTRACT_DIR")"
if [ "$scanned" -lt "$MIN_CONTRACT_FILES" ]; then
    printf '\nFAIL (vacuity): scanned only %s contract file(s), floor %s.\n' "$scanned" "$MIN_CONTRACT_FILES"
    printf 'Fix the scan, not this number.\n'
    exit 1
fi

hits="$(scan "$CONTRACT_DIR")"
count="$(printf '%s' "$hits" | grep -c . || true)"

printf 'scanned %s contract file(s); %s machine-specific path(s), allowed 0\n' "$scanned" "$count"

if [ "$count" -gt 0 ]; then
    printf '\nFAIL: a contract names a path that exists on at most one machine.\n'
    printf 'Such a contract self-skips everywhere else while pv reports PASS.\n'
    printf 'Use ${APR_MODELS:?}, ${APR_LEADERBOARD_ROOT:?},\n'
    printf '${APR_CRUX_GOLDENS:-evidence/crux}, $HOME/... or a repo-relative path.\n'
    printf 'There is no baseline to raise (#2532).\n\n'
    printf '%s\n' "$hits" | sed 's|^|  |'
    exit 1
fi

printf 'PASS\n'
exit 0
