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
# WHY THREE MODES / HOW --full ARMS ITSELF
# ----------------------------------------
# pmat owns this detector (pmat#1017) and re-implementing its tiering would be
# muda, so --full shells out to it and compares `.shipped_count`.
#
# The clean-room pool that runs the blocking guards (the only runners carrying
# the `clean-room` label) still cannot do it. RE-MEASURED 2026-08-28, unchanged
# from the 2026-08-20 reading:
#
#   $ ssh mac-server 'pmat --version; pmat analyze hardcoded-paths --help'
#   pmat 3.31.0
#   error: unrecognized subcommand 'hardcoded-paths'
#
# Wiring a bare --full into the required gate today would red main on every PR.
# The alternatives are worse: `cargo install pmat || true` (book.yml:90 does
# this — a gate that cannot fail), or a cold `cargo install` inside a
# timeout-boxed job (the cargo-audit failure mode that evicted the merge queue).
#
# WHAT WENT WRONG WITH LEAVING IT UNWIRED (#2706 / PERF-032)
# ---------------------------------------------------------
# The previous version of this header said "PROMOTE --full INTO ci.yml AS SOON
# AS THE CLEAN-ROOM FLEET CARRIES pmat >= 3.32.0 — this comment is the trigger."
# A comment is not a trigger. Nobody re-reads it, and nothing re-evaluates the
# condition, so the mode that actually catches shipped machine-specific paths
# gated nothing for as long as it took 20 of them to land (299 vs a baseline of
# 278, measured on origin/main 62d23d8d1). The same header also claimed --full
# "runs from `make tier3`"; the Makefile has never invoked this script at all.
#
# So the promotion is now MECHANICAL rather than editorial. --full-if-capable
# probes for the subcommand at run time and:
#   * runs the full ratchet and PROPAGATES ITS EXIT STATUS when pmat can do it;
#   * otherwise PROVES the capability is absent and skips, printing the version
#     and the actual refusal text.
# The day the fleet carries pmat >= 3.32.0 the gate arms itself, with no edit
# and nobody having to remember. This is not `|| true`: `|| true` discards a
# verdict that was actually produced, whereas the skip here is taken only when
# no verdict CAN be produced, and it says so in the log. The residual blind
# spot — a runner silently losing pmat — is the price of not cold-installing a
# toolchain inside a required job, and it is stated here rather than hidden.
#
# THE 6 THAT STAYED, AND WHY (classified, NOT exempted)
# -----------------------------------------------------
# Of the 20 that landed, 14 were fixed outright. Six remain in the tree and are
# still COUNTED -- there is no allowlist here and no path is excused. The
# ratchet came back under its baseline because eight OTHER, pre-existing
# findings were fixed to pay for them (299 -> 277 against a baseline of 278,
# now lowered to 277). Naming them so the next reader does not re-litigate:
#
#   evidence/dogfood/0.64.0/{gx10,intel,mini}.json  (1 each)
#   evidence/dogfood/0.64.0/lambda.json             (2)
#     `path_resolved_apr` records WHICH binary a bare `apr` resolved to on that
#     host. The host-specific path IS the measurement -- on lambda it is
#     /home/noah/.local/bin/apr shadowing ~/.cargo/bin, the #2384/#2361 defect
#     the receipt exists to document. Redacting it would not make the repo more
#     portable; it would delete the evidence and fabricate a cleaner history.
#     A receipt naming the machine it was measured on is the epic's whole point.
#
#   .github/workflows/ci.yml  /home/noah/data/sccache:/sccache  (a 4th copy)
#     A real portability defect: the fleet's sccache mount is one user's home,
#     repeated inline four times. NOT fixed here on purpose -- collapsing it to
#     a single definition changes a docker mount on 16 clean-room runners, and
#     the `env` context is not reliably available in a job-level `container:`/
#     volume position, so a wrong guess silently unshares the cache fleet-wide
#     (see the 'shared-cache cap is CORRECTNESS' lesson). It needs its own PR
#     that can actually observe a CI run. Deliberately left visible in the count.
#
#   bash scripts/check_hardcoded_paths.sh                    # blocking (contracts/)
#   bash scripts/check_hardcoded_paths.sh --self-test        # case table
#   bash scripts/check_hardcoded_paths.sh --full             # shipped-tier ratchet
#   bash scripts/check_hardcoded_paths.sh --full-if-capable  # ratchet, self-arming
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_ROOT="$REPO_ROOT"                      # the checkout that carries the libs
REPO_ROOT="${HP_REPO_ROOT:-$REPO_ROOT}"   # the tree under test (the self-test points this at a fixture repo)
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

    # ---- the pinned-instrument ratchet: a fixture analyser + a fixture repo (PMAT-1059) ----
    pin=$(sed -nE 's/^PMAT_PIN="([0-9.]+)"$/\1/p' "${SELF_ROOT}/scripts/pmat_bin.sh" | head -n1)
    FAKE="$TD/fake-analyser"
    printf '#!/usr/bin/env bash\ncase "$1" in --version) echo "%s %s"; exit 0;; esac\nd=.; while [ $# -gt 0 ]; do case "$1" in -p) d=$2; shift;; --help) echo ok; exit 0;; esac; shift; done\ncat "$d/.fake-scan.json"\n' "pmat" "$pin" > "$FAKE"
    printf '#!/usr/bin/env bash\ncase "$1" in --version) echo "%s 3.0.0"; exit 0;; esac\necho ok\n' "pmat" > "$TD/off-pin"
    chmod +x "$FAKE" "$TD/off-pin"
    scanjson() { local n=$1 i s=""; for ((i = 0; i < n; i++)); do s="$s{\"site\":\"shipped\",\"file\":\"src/f$i.rs\",\"path\":\"fixture://p$i\"},"; done; printf '{"shipped_count":%s,"files_scanned":5,"findings":[%s]}\n' "$n" "${s%,}"; }
    bl() { printf 'count: %s\npmat_version: %s\nbasis: $PMAT analyze hardcoded-paths -p . -f json | jq .shipped_count (fixture)\n' "$1" "$2"; }
    R="$TD/repo"; mkdir -p "$R/scripts"
    ( cd "$R" && git init -q . && git config user.email t@t && git config user.name t && git config core.hooksPath /dev/null && git commit -q --allow-empty -m root )
    hp_row() { # hp_row <want rc> <label> <base n> <base baseline> <head n> <head baseline> [<wanted text>] [<env>]
        local want=$1 label=$2 bn=$3 bbl=$4 hn=$5 hbl=$6 sub=${7:-} env=${8:-} rc=0 out
        ( cd "$R" && git checkout -q --detach && git reset -q --hard "$(git rev-list --max-parents=0 HEAD)" ) || return 1
        mkdir -p "$R/scripts"
        scanjson "$bn" > "$R/.fake-scan.json"; printf '%s\n' "$bbl" > "$R/scripts/hardcoded_path_shipped_baseline.txt"
        ( cd "$R" && git add -A && git commit -qm base && git update-ref refs/remotes/origin/main HEAD && git checkout -q -B row )
        scanjson "$hn" > "$R/.fake-scan.json"; printf '%s\n' "$hbl" > "$R/scripts/hardcoded_path_shipped_baseline.txt"
        ( cd "$R" && git add -A && git commit -qm head --allow-empty )
        [ "$env" = no-base ] && git -C "$R" update-ref -d refs/remotes/origin/main
        out=$(HP_REPO_ROOT="$R" PMAT_BIN_OVERRIDE="${FAKE_BIN:-$FAKE}" PMAT_BIN_NO_FALLBACK=1 MIN_FILES_SCANNED=1 bash "${BASH_SOURCE[0]}" --full-if-capable 2>&1) || rc=$?
        if [ "$rc" = "$want" ] && { [ -z "$sub" ] || printf '%s' "$out" | grep -qF -- "$sub"; }; then printf 'ok    ratchet %s (rc=%s)\n' "$label" "$rc"
        else printf 'FAIL  ratchet %s (rc=%s, wanted %s%s)\n' "$label" "$rc" "$want" "${sub:+; wanted text: $sub}"; printf '%s\n' "$out" | tail -6 | sed 's|^|        |'; fails=1; fi
    }
    hp_row 0 "R1 stamp == pin, 8 <= 8: PASS by the absolute compare"                   8 "$(bl 8 "$pin")" 8 "$(bl 8 "$pin")" "matches the pin"
    hp_row 1 "R2 stamp == pin, +1 path: RED by the absolute compare"                    8 "$(bl 8 "$pin")" 9 "$(bl 8 "$pin")" "grew 8 -> 9"
    hp_row 0 "R3 stale stamp 3.36.0 (count 5), a newer pin widens: REPORT + differential PASS" 8 "$(bl 5 3.36.0)" 8 "$(bl 5 3.36.0)" "BASELINE-STALE{old=3.36.0,new=$pin}"
    hp_row 1 "R4 stale stamp, +1 path vs base: RED by the differential, path named"     8 "$(bl 5 3.36.0)" 9 "$(bl 5 3.36.0)" "src/f8.rs|fixture://p8"
    hp_row 0 "R5 INVALID stamp (no version): REPORT + differential PASS"                8 "$(bl 8 INVALID)" 8 "$(bl 8 INVALID)" "BASELINE-INVALID"
    hp_row 1 "R6 stamp bumped 3.36.0 -> pin while count: stands: refused"              8 "$(bl 8 3.36.0)" 8 "$(bl 8 "$pin")" "A stamp is not a measurement"
    hp_row 1 "R7 stale stamp and no base to name: RED, never the branch against itself" 8 "$(bl 5 3.36.0)" 8 "$(bl 5 3.36.0)" "" no-base
    FAKE_BIN="$TD/off-pin" hp_row 1 "R8 no analyser at the pin: FAIL (ENV), not a pass" 8 "$(bl 8 "$pin")" 8 "$(bl 8 "$pin")" "FAIL (ENV)"
    unset FAKE_BIN

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED\n'; exit 0
fi

# ---------------------------------------------------------------------------
# --full-if-capable / --full: the shipped tier, ratcheted under ONE pinned
# instrument (PMAT-1059, DAG row G-10, #2999).
#
# THE INSTRUMENT IS PART OF THE NUMBER. 277 was recorded 2026-09-05 with no
# version named; 3.37.0 and 3.38.0 both count 317 on that same tree. The day
# paiml/infra pinned the fleet at 3.37.0 (forjar.yaml, PMAT-231) the self-arming
# guard armed and every PR went red for a defect no PR introduced. So:
#   * the binary is scripts/pmat_bin.sh's pin, never PATH. "--full-if-capable"
#     no longer skips: a runner without the pin is an ENV failure (exit 1),
#     because an unanswered ratchet is not a pass;
#   * the baseline carries count:, pmat_version:, basis:. Any of the three
#     missing or unparseable => INVALID, and INVALID is not a number;
#   * the absolute compare (shipped <= count) runs ONLY when the stamp equals
#     the binary's version. Otherwise the guard REPORTs BASELINE-STALE{old,new}
#     (or BASELINE-INVALID) and decides by HEAD vs merge-base under the same
#     binary: delta <= 0 PASS, delta > 0 FAIL naming the new paths;
#   * re-baselining is its own ticket, re-measured, stamped, never a raise: a
#     stamp that moves while count: stands still is refused.
# The base is named by scripts/lib/resolve_base.sh (G-6's resolver, one case
# table for both guards) and materialised as a detached worktree (the analyser
# enumerates with `git ls-files`), so a depth-1 checkout that fetched origin/main
# can answer.
# ---------------------------------------------------------------------------
hp_read_baseline() { # hp_read_baseline <file> -> HP_COUNT HP_VER HP_BASIS HP_VALID HP_WHY
    HP_COUNT=$(sed -nE 's/^count:[[:space:]]*([0-9]+)[[:space:]]*(#.*)?$/\1/p' "$1" | head -n1)
    HP_VER=$(sed -nE 's/^pmat_version:[[:space:]]*([0-9]+\.[0-9]+\.[0-9]+)[[:space:]]*(#.*)?$/\1/p' "$1" | head -n1)
    HP_BASIS=$(sed -nE 's/^basis:[[:space:]]*(\$PMAT analyze hardcoded-paths.*)$/\1/p' "$1" | head -n1)
    HP_VALID=1; HP_WHY=""
    [ -n "$HP_COUNT" ] || { HP_VALID=0; HP_WHY="no 'count: N' line"; }
    [ -n "$HP_VER" ]   || { HP_VALID=0; HP_WHY="${HP_WHY:+$HP_WHY; }no 'pmat_version: X.Y.Z' line"; }
    [ -n "$HP_BASIS" ] || { HP_VALID=0; HP_WHY="${HP_WHY:+$HP_WHY; }no 'basis: \$PMAT analyze hardcoded-paths ...' line"; }
}
hp_scan() { # hp_scan <dir> <out.json>; the analyser's own rc, never $? through a pipe
    ( cd "$1" && "$PMAT" analyze hardcoded-paths -p . -f json ) > "$2" 2> "$2.err"
}
hp_paths() { # hp_paths <json> -> sorted "file|path" lines of the shipped tier
    jq -r '.findings[] | select(.site=="shipped") | "\(.file)|\(.path)"' "$1" | sed 's|^\./||' | LC_ALL=C sort
}
if [ "${1:-}" = "--full-if-capable" ] || [ "${1:-}" = "--full" ]; then
    printf '=== shipped-tier ratchet under the pinned analyser (check_hardcoded_paths.sh %s) ===\n' "$1"
    # shellcheck source=scripts/pmat_bin.sh
    if ! . "${SELF_ROOT}/scripts/pmat_bin.sh"; then
        printf 'FAIL (ENV): scripts/pmat_bin.sh found no analyser at its pin. A runner without the pin cannot answer, and an unanswered ratchet is not a pass.\n'
        exit 1
    fi
    command -v jq >/dev/null 2>&1 || { printf 'FAIL (ENV): jq not found.\n'; exit 1; }
    printf 'armed under the pin %s (%s)\n' "$PMAT_VERSION" "$PMAT"
    TD="$(mktemp -d)" || { printf 'FAIL: no temp dir\n' >&2; exit 1; }
    trap 'rm -rf "${TD:?}"' EXIT
    if ! hp_scan "$REPO_ROOT" "$TD/head.json" || [ ! -s "$TD/head.json" ]; then
        printf 'FAIL: the analyser produced no JSON for HEAD.\n'; sed 's|^|  |' "$TD/head.json.err" | head -5; exit 1
    fi
    shipped="$(jq -r '.shipped_count' "$TD/head.json")"
    files="$(jq -r '.files_scanned' "$TD/head.json")"
    case "$shipped$files" in ''|*[!0-9]*) printf 'FAIL: unparseable analyser JSON\n'; exit 1 ;; esac
    if [ "$files" -lt "$MIN_FILES_SCANNED" ]; then
        printf '\nFAIL (vacuity): the analyser scanned only %s file(s), floor %s. Fix the scan, not this number.\n' "$files" "$MIN_FILES_SCANNED"; exit 1
    fi
    [ -f "$SHIPPED_BASELINE" ] || { printf 'FAIL: %s missing.\n' "$SHIPPED_BASELINE"; exit 1; }
    hp_read_baseline "$SHIPPED_BASELINE"
    # The file itself is shrink-only against a ref this branch cannot rewrite.
    # shellcheck source=scripts/lib_baseline_ratchet.sh
    . "${SELF_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
    baseline_ratchet_check "${REPO_ROOT}" scripts/hardcoded_path_shipped_baseline.txt count || exit 1
    # shellcheck source=scripts/lib/resolve_base.sh
    PROG=check_hardcoded_paths . "${SELF_ROOT}/scripts/lib/resolve_base.sh" || exit 1
    base_ok=0; BASE_REF=""; BASE_HOW=""
    if resolve_base HEAD > "$TD/resolve.txt" 2>&1; then base_ok=1; fi
    if [ "$base_ok" = 1 ] && git -C "$REPO_ROOT" cat-file -e "${BASE_REF}:scripts/hardcoded_path_shipped_baseline.txt" 2>/dev/null; then
        git -C "$REPO_ROOT" show "${BASE_REF}:scripts/hardcoded_path_shipped_baseline.txt" > "$TD/base_baseline.txt"
        bver=$(sed -nE 's/^pmat_version:[[:space:]]*([0-9]+\.[0-9]+\.[0-9]+).*$/\1/p' "$TD/base_baseline.txt" | head -n1)
        bcount=$(sed -nE 's/^count:[[:space:]]*([0-9]+).*$/\1/p' "$TD/base_baseline.txt" | head -n1)
        if [ -n "$HP_VER" ] && [ "$HP_VER" != "${bver:-}" ] && [ -n "$bcount" ] && [ "${HP_COUNT:-}" = "$bcount" ]; then
            printf '\nFAIL: pmat_version moved %s -> %s in %s while count: stayed %s. A stamp is not a measurement: re-baselining is its own ticket, re-measured under the new pin, never a raise.\n' \
                "${bver:-none}" "$HP_VER" "${SHIPPED_BASELINE#"$REPO_ROOT"/}" "$HP_COUNT"
            exit 1
        fi
    fi
    if [ "$HP_VALID" = 1 ] && [ "$HP_VER" = "$PMAT_VERSION" ]; then
        printf 'baseline: count %s, stamped %s (matches the pin); basis: %s\n' "$HP_COUNT" "$HP_VER" "$HP_BASIS"
        printf 'scanned %s file(s); %s shipped finding(s), baseline %s\n' "$files" "$shipped" "$HP_COUNT"
        if [ "$shipped" -gt "$HP_COUNT" ]; then
            printf '\nFAIL: shipped machine-specific paths grew %s -> %s under the same instrument. Fix the paths; never raise the number.\n' "$HP_COUNT" "$shipped"
            printf 'All %s shipped finding(s), by file:\n' "$shipped"
            hp_paths "$TD/head.json" | cut -d'|' -f1 | uniq -c | sort -rn | head -40 | sed 's|^|  |'
            exit 1
        fi
        if [ "$shipped" -lt "$HP_COUNT" ]; then
            printf '\nImproved: %s -> %s. Re-baseline under its own ticket (stamped) to record it.\n' "$HP_COUNT" "$shipped"
        fi
        printf 'PASS\n'; exit 0
    fi
    # ---- instrument mismatch, or an INVALID stamp: the count is not compared ----
    if [ "$HP_VALID" = 1 ]; then
        printf 'REPORT BASELINE-STALE{old=%s,new=%s}: the recorded count %s was measured by another instrument and is not compared.\n' "$HP_VER" "$PMAT_VERSION" "$HP_COUNT"
    else
        printf 'REPORT BASELINE-INVALID{stamp=%s,binary=%s}: %s; the recorded number is not a baseline and is not compared.\n' "${HP_VER:-none}" "$PMAT_VERSION" "$HP_WHY"
    fi
    printf 'Re-baselining is its own ticket (stamped, never a raise). Deciding by HEAD vs merge-base under the same pin.\n'
    if [ "$base_ok" != 1 ]; then
        printf '\nFAIL: no base can be named, so the differential is UNMEASURED and this run cannot pass:\n'; sed 's|^|  |' "$TD/resolve.txt" | head -4; exit 1
    fi
    # The analyser enumerates with `git ls-files`, so the base must be a checkout, not an archive.
    if ! git -C "$REPO_ROOT" worktree add -q --detach "$TD/base" "$BASE_REF" 2> "$TD/wt.err"; then
        printf '\nFAIL: cannot materialise the base tree %s as a worktree:\n' "$BASE_REF"; sed 's|^|  |' "$TD/wt.err" | head -3; exit 1
    fi
    trap 'git -C "$REPO_ROOT" worktree remove --force "$TD/base" >/dev/null 2>&1; rm -rf "${TD:?}"' EXIT
    if ! hp_scan "$TD/base" "$TD/base.json" || [ ! -s "$TD/base.json" ]; then
        printf '\nFAIL: the analyser produced no JSON for the base tree.\n'; sed 's|^|  |' "$TD/base.json.err" | head -5; exit 1
    fi
    base_shipped=$(jq -r '.shipped_count' "$TD/base.json")
    case "$base_shipped" in ''|*[!0-9]*) printf 'FAIL: unparseable base JSON\n'; exit 1 ;; esac
    delta=$((shipped - base_shipped))
    printf 'differential: base %.9s (%s) = %s shipped; HEAD = %s shipped; delta %+d\n' "$BASE_REF" "$BASE_HOW" "$base_shipped" "$shipped" "$delta"
    if [ "$delta" -gt 0 ]; then
        hp_paths "$TD/head.json" > "$TD/head.txt"; hp_paths "$TD/base.json" > "$TD/base.txt"
        printf '\nFAIL: this change adds %s shipped machine-specific path(s) (on HEAD, absent at the base):\n' "$delta"
        comm -13 "$TD/base.txt" "$TD/head.txt" | head -40 | sed 's|^|  |'
        exit 1
    fi
    printf 'PASS (differential, delta %+d)\n' "$delta"; exit 0
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
