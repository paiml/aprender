#!/usr/bin/env bash
# check_ci_train_only.sh -- the heavy tier runs once per merge train; a PR lands only via the train (#7001).
#
# THE RULE (quorum receipt QUORUM-GATE-7001-1113, verdict b). In .github/workflows/ci.yml:
#   TRAIN = pull_request AND head repo == this repo AND head ref starts with `train/aprender-` AND PR author == the fleet login.
#   1. x86-main, determinism and workspace-test-shard run on a pull_request only when all of the last three TRAIN conjuncts hold.
#   2. workspace-test, ci-gate and gate (the verdict jobs) compute TRAIN with all four conjuncts, and their PR-LIGHT-VERDICT
#      block, run as shipped on a non-train pull_request with pr-light green, exits non-zero with the exact refusal message.
#   3. the same block refuses a train that is not on main's tip (behind_by) and reads a failed read as a failure.
#   4. the train verdict needs every heavy job: gate needs x86-main, determinism, workspace-test; workspace-test needs
#      workspace-test-shard; ci-gate needs x86-main. A heavy job missing from `needs` could be skipped unseen.
#   5. nothing in a verdict job is continue-on-error (no third state), and no `exit 0` precedes the block but the draft one.
#
# Usage:  bash scripts/check_ci_train_only.sh [--self-test] [WORKFLOW_FILE]
# Wired:  ci/explicit-test-commands.d/847-ci-train-only.cmd (--self-test) and guard_tree.sh (no args).
set -euo pipefail

MSG='not mergeable alone: lands only via train/aprender-* (heavy tier runs once per train, #7001)'
# has PATTERN -> rc 0 when stdin has a matching line; reads ALL of stdin (grep -q + pipefail reads a present line as absent).
has() { /usr/bin/grep "$@" > /dev/null; }

job_block() { # FILE JOB -> the lines of `  JOB:` up to the next two-space-indented key
    awk -v j="$2" '
        $0 ~ "^  " j ":[[:space:]]*$" { on = 1; print; next }
        on && /^  [A-Za-z0-9_-]+:/ { exit }
        on { print }' "$1"
}

needs_of() { # BLOCK -> the words of the job's `needs: [..]` line
    printf '%s\n' "$1" | sed -n 's/^    needs: *\[\(.*\)\].*/\1/p' | tr ',' ' '
}

verdict_block() { # BLOCK -> the PR-LIGHT-VERDICT block, de-indented
    printf '%s\n' "$1" | awk '/PR-LIGHT-VERDICT-BEGIN/{on=1;next} /PR-LIGHT-VERDICT-END/{on=0} on{sub(/^          /,""); print}'
}

check() { # FILE -> one FAIL line per violated condition; rc 1 when any
    local f="$1" bad=0 j blk n w scr td out rc env_line
    td="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/train-only.XXXXXX")"
    mkdir -p "$td/bin"
    printf '#!/bin/sh\necho 0\n' > "$td/bin/gh"; chmod +x "$td/bin/gh"
    for j in x86-main determinism workspace-test-shard; do
        blk="$(job_block "$f" "$j")"
        if [ -z "$blk" ]; then echo "FAIL  heavy job $j not found"; bad=1; continue; fi
        env_line="$(printf '%s\n' "$blk" | /usr/bin/grep -E '^    if:' || true)"
        for w in "github.event_name != 'pull_request' ||" 'head.repo.full_name == github.repository' \
            "startsWith(github.event.pull_request.head.ref, 'train/aprender-')" 'user.login == '; do
            if ! printf '%s\n' "$env_line" | has -F -- "$w"; then
                echo "FAIL  heavy job $j: TRAIN lacks the conjunct  $w"; bad=1
            fi
        done
    done
    for j in workspace-test ci-gate gate; do
        blk="$(job_block "$f" "$j")"
        if [ -z "$blk" ]; then echo "FAIL  verdict job $j not found"; bad=1; continue; fi
        env_line="$(printf '%s\n' "$blk" | /usr/bin/grep -E '^      TRAIN:' || true)"
        for w in "github.event_name == 'pull_request'" 'head.repo.full_name == github.repository' \
            "startsWith(github.event.pull_request.head.ref, 'train/aprender-')" 'user.login == '; do
            if ! printf '%s\n' "$env_line" | has -F -- "$w"; then
                echo "FAIL  verdict job $j: TRAIN lacks the conjunct  $w"; bad=1
            fi
        done
        if printf '%s\n' "$blk" | has 'continue-on-error'; then echo "FAIL  verdict job $j uses continue-on-error (no third state)"; bad=1; fi
        # nothing before the block may exit 0 except the draft line
        if printf '%s\n' "$blk" | awk '/PR-LIGHT-VERDICT-BEGIN/{exit} /exit 0/ && !/DRAFT/{f=1} END{exit !f}'; then
            echo "FAIL  verdict job $j: an exit 0 other than the draft one precedes the PR-LIGHT block"; bad=1
        fi
        scr="$(verdict_block "$blk")"
        if [ -z "$scr" ]; then echo "FAIL  verdict job $j has no PR-LIGHT-VERDICT block"; bad=1; continue; fi
        # run the block as shipped: a non-train pull_request, pr-light green, a gh that says "on main's tip" -> must refuse
        for n in unset 0 ; do
            rc=0
            if [ "$n" = unset ]; then
                out="$(env -u TRAIN PATH="$td/bin:$PATH" EVT=pull_request PRL=success REPO=x/y BASE_REF=main HEAD_SHA=abc bash -c "$scr" 2>&1)" || rc=$?
            else
                out="$(env TRAIN=0 PATH="$td/bin:$PATH" EVT=pull_request PRL=success REPO=x/y BASE_REF=main HEAD_SHA=abc bash -c "$scr" 2>&1)" || rc=$?
            fi
            if [ "$rc" = 0 ]; then echo "FAIL  verdict job $j can exit 0 on a non-train pull_request (TRAIN=$n)"; bad=1
            elif ! printf '%s\n' "$out" | has -F -- "$MSG"; then echo "FAIL  verdict job $j refuses a non-train PR without the exact message"; bad=1; fi
        done
        # a train that is behind main's tip, and a failed tip read, are failures
        rc=0; printf '#!/bin/sh\necho 4\n' > "$td/bin/gh"
        env TRAIN=1 PATH="$td/bin:$PATH" EVT=pull_request PRL=success REPO=x/y BASE_REF=main HEAD_SHA=abc bash -c "$scr" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then echo "FAIL  verdict job $j passes a train that is behind main's tip"; bad=1; fi
        rc=0; printf '#!/bin/sh\nexit 1\n' > "$td/bin/gh"
        env TRAIN=1 PATH="$td/bin:$PATH" EVT=pull_request PRL=success REPO=x/y BASE_REF=main HEAD_SHA=abc bash -c "$scr" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then echo "FAIL  verdict job $j passes a train whose tip read failed"; bad=1; fi
        rc=0; printf '#!/bin/sh\necho 0\n' > "$td/bin/gh"
        env TRAIN=1 PATH="$td/bin:$PATH" EVT=pull_request PRL=failure REPO=x/y BASE_REF=main HEAD_SHA=abc bash -c "$scr" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = 0 ]; then echo "FAIL  verdict job $j passes a train whose pr-light is red"; bad=1; fi
        rc=0
        env TRAIN=1 PATH="$td/bin:$PATH" EVT=pull_request PRL=success REPO=x/y BASE_REF=main HEAD_SHA=abc bash -c "$scr" > /dev/null 2>&1 || rc=$?
        if [ "$rc" != 0 ]; then echo "FAIL  verdict job $j refuses a train on main's tip with pr-light green (the block must fall through to the heavy checks)"; bad=1; fi
    done
    rm -rf "${td:?}"
    for w in "gate:x86-main" "gate:determinism" "gate:workspace-test" "workspace-test:workspace-test-shard" "ci-gate:x86-main"; do
        j="${w%%:*}"; n="${w#*:}"
        blk="$(job_block "$f" "$j")"
        if ! needs_of "$blk" | tr ' ' '\n' | has -x -F -- "$n"; then echo "FAIL  job $j does not need heavy job $n"; bad=1; fi
    done
    return "$bad"
}

self_test() {
    local src="$1" td fails=0 got name
    td="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/train-only-st.XXXXXX")"; trap 'rm -rf "${td:?}"' RETURN
    if got="$(check "$src")"; then echo "ok    green the workflow as committed"; else echo "FAIL  green: the workflow as committed is RED: $got"; fails=1; fi
    # a long job block (past one pipe buffer) must stay GREEN
    awk '{print} /^  ci-gate:[[:space:]]*$/{on=1} on && /BASE_REF: /{for(i=0;i<20000;i++) print "      # pad pad pad pad pad pad"; on=0}' "$src" > "$td/long.yml"
    if got="$(check "$td/long.yml")"; then echo "ok    green a job block longer than one pipe buffer"; else echo "FAIL  green: a long job block is RED: $got"; fails=1; fi
    mutant() { # NAME SED_SCRIPT -> the mutated workflow must be RED
        name="$1"; sed -E "$2" "$src" > "$td/m.yml"
        if cmp -s "$src" "$td/m.yml"; then echo "FAIL  mutant $name changed nothing"; fails=1; return; fi
        if got="$(check "$td/m.yml")"; then echo "FAIL  mutant $name stayed GREEN"; fails=1; else echo "ok    red   mutant $name"; fi
    }
    # TRAIN conjuncts: each removed from each place
    mutant x86-main-no-ref-prefix  '/^  x86-main:$/,/runs-on:/{s/ && startsWith\(github\.event\.pull_request\.head\.ref, .train\/aprender-.\)//}'
    mutant x86-main-no-same-repo   '/^  x86-main:$/,/runs-on:/{s/github\.event\.pull_request\.head\.repo\.full_name == github\.repository && //}'
    mutant x86-main-no-author      '/^  x86-main:$/,/runs-on:/{s/ && github\.event\.pull_request\.user\.login == .noahgift.//}'
    mutant x86-main-runs-on-any-pr '/^  x86-main:$/,/runs-on:/{s/github\.event_name != .pull_request. \|\| //}'
    mutant det-no-author           '/^  determinism:$/,/runs-on:/{s/ && github\.event\.pull_request\.user\.login == .noahgift.//}'
    mutant det-no-ref-prefix       '/^  determinism:$/,/runs-on:/{s/ && startsWith\(github\.event\.pull_request\.head\.ref, .train\/aprender-.\)//}'
    mutant shard-no-same-repo      '/^  workspace-test-shard:$/,/runs-on:/{s/github\.event\.pull_request\.head\.repo\.full_name == github\.repository && //}'
    mutant shard-no-ref-prefix     '/^  workspace-test-shard:$/,/runs-on:/{s/ && startsWith\(github\.event\.pull_request\.head\.ref, .train\/aprender-.\)//}'
    for jb in workspace-test ci-gate gate; do
        mutant "$jb-env-no-author"  "/^  $jb:\$/,/^    steps:/{s/ && github\.event\.pull_request\.user\.login == .noahgift.//}"
        mutant "$jb-env-no-prefix"  "/^  $jb:\$/,/^    steps:/{s/ && startsWith\(github\.event\.pull_request\.head\.ref, .train\/aprender-.\)//}"
        mutant "$jb-env-no-samerepo" "/^  $jb:\$/,/^    steps:/{s/ && github\.event\.pull_request\.head\.repo\.full_name == github\.repository//}"
        mutant "$jb-env-no-event"   "/^  $jb:\$/,/^    steps:/{s/TRAIN: \\$\{\{ \(github\.event_name == .pull_request. && /TRAIN: \${{ (/}"
        mutant "$jb-non-train-passes" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/if \[ \"\\$\{TRAIN:-0\}\" != 1 \]; then/if false; then/}"
        mutant "$jb-message-changed" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/not mergeable alone/mergeable alone/}"
        mutant "$jb-tip-check-gone"  "/^  $jb:\$/,/^  [a-z-]+:\$/{/behind_by=\\$/d}"
        mutant "$jb-unread-tip-passes" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/\|\| behind=unread/|| behind=0/}"
        mutant "$jb-prl-ignored"     "/^  $jb:\$/,/^  [a-z-]+:\$/{s/\[ \"\\$\{PRL:-\}\" = success \] \|\| /true || /}"
        mutant "$jb-exit0-before-block" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/^( *)# --- PR-LIGHT-VERDICT-BEGIN/\1[ \"\\$\{EVT:-\}\" != pull_request ] || exit 0\n&/}"
        mutant "$jb-continue-on-error" "s/^  $jb:\$/  $jb:\n    continue-on-error: true/"
    done
    # needs
    mutant gate-drops-x86-main     '/^  gate:$/,/^    if:/{s/needs: \[x86-main, /needs: [/}'
    mutant gate-drops-determinism  '/^  gate:$/,/^    if:/{s/ determinism,//}'
    mutant gate-drops-workspace-test '/^  gate:$/,/^    if:/{s/ workspace-test,//}'
    mutant wt-drops-shard          '/^  workspace-test:$/,/^    if:/{s/needs: \[workspace-test-shard, /needs: [/}'
    mutant ci-gate-drops-x86-main  '/^  ci-gate:$/,/^    if:/{s/needs: \[x86-main, /needs: [/}'
    return "$fails"
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1  # script-relative: no repo query (PMAT-6479)
case "${1:-}" in
    --self-test) self_test "${2:-$root/.github/workflows/ci.yml}"; exit $? ;;
    *)
        out="$(check "${1:-$root/.github/workflows/ci.yml}")" && { echo "ok    a pull_request lands only via the train; the heavy tier is train-only"; exit 0; }
        printf '%s\n' "$out"; exit 1 ;;
esac
