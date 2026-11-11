#!/usr/bin/env bash
# pr_light_fast.sh -- the pr-light gate body as ONE bash driver: every lane concurrent, none of them a download (GATE-JUNK-001 C6).
#
# Measured on 19 green PR runs (2026-11-09): pr-light p50 1365 s, p95 2035 s. Two steps were the whole job:
#   "Cheap checks" ~140 s (17 checks run one after another) and the fat_driver guard-tree section ~1200 s
#   (scripts/ci_guards.sh guard-tree: ~45 independent guard steps run strictly serially). Toolchain install and
#   the jq fetch measured 0 s and <1 s; they leave because they are network, not because they were slow.
#
# Rule: a PR gate is O(diff) or O(1), makes no network fetch, uses a pinned pre-provisioned tool, can fail, and gives the
# same verdict twice on one sha. This driver adds no check and drops none: it runs the SAME commands the old steps ran,
#   * "Cheap checks"  -> the lanes in cheap_lanes() (same command lines), and
#   * the fat_driver guard-tree section -> pin_build_base + `ci_guards.sh guard-tree --shard I/K` for I in 1..K + the
#     R-2 and 11.1 PR-body checks (the section's own steps 2, 3, 4, 5; steps 0 and 1 are a docker chown and a checkout).
# `ci_guards.sh --shard` partitions the manifest exactly once (#6255), so K shards run every guard step once.
# Each lane runs in its OWN `git worktree` of HEAD: a case table that plants a mutant never touches another lane's tree.
#
# A failing lane fails the driver (rc 1). A lane that cannot start, a shard that ran nothing (ci_guards rc 2), or zero
# lanes is rc 2/1, never a pass. No `|| true`, no continue-on-error: every lane's status is read.
#
#   bash scripts/ci/pr_light_fast.sh                  # all lanes (cheap + guard-tree shards + PR-body checks)
#   bash scripts/ci/pr_light_fast.sh --list           # print the lane names
#   bash scripts/ci/pr_light_fast.sh --list-commands  # print each lane's command (read by check_guards_are_wired.sh)
#   bash scripts/ci/pr_light_fast.sh --self-test      # case table (can-fail, order-independence, determinism)
#   PR_LIGHT_SHARDS=8   guard-tree shard count (default 8)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# fmt runs on the host's pinned toolchain (rust-toolchain.toml, asserted by require_pinned_tools.sh) with RUSTUP_AUTO_INSTALL=0:
# the sovereign-ci image carries 1.95.0 only, so `cargo fmt` in it made rustup download 1.93.0 (5 components, ~24 s) every run.

# GATE PROOFS (Noah 10:38Z): a self-test / case table / mutant suite OF A GATE runs only when a gate's own script or recipe
# changed (gate_surface_changed), and nightly; never because product code changed. Product-code PRs run the gates, not their proofs.
GATE_SURFACE_ERE='^(scripts/|ci/|\.github/|Makefile$|[^/]*\.mk$)'
# Which guard-tree step is a proof is decided by the step's COMMANDS (ci_guard_steps.sh step_is_proof), never by its name: every
# command must be a self-test, case table or mutant suite. A step with one live gate command in it always runs.
# NIGHTLY: guards-nightly.yml runs this script with PR_LIGHT_PROOFS=all on main, so every proof a product-code PR skipped runs once a day.

# name|command -- the same command lines the old "Cheap checks" step ran, one lane each.
cheap_lanes() { gate_lanes; proof_lanes; }

# the gates themselves: O(1)/O(diff), run on every PR.
gate_lanes() {
    cat << 'LANES'
falsification-ids|python3 scripts/ci/fail_fast.py ids --base origin/main
fmt|RUSTUP_AUTO_INSTALL=0 cargo fmt --all -- --check
no-pull-request-target-guard|sh scripts/guards/no-pull-request-target-guard.sh
exec-bit-guard|bash scripts/release/exec_bit_guard.sh
fetch-p6-wiring|setsid --wait bash scripts/ci/fetch_p6.sh --check-wiring
pr-light-offline|setsid --wait bash scripts/check_pr_light_offline.sh
LANES
}

# the proofs of the gates (self-tests, case tables, mutant suites): run only when gate_surface_changed, and nightly.
proof_lanes() {
    cat << 'LANES'
fail-fast-self-test|python3 scripts/ci/fail_fast.py self-test
guard-tree-case-table|bash scripts/tests/guard_tree_test.sh
bump-version-self-test|bash scripts/bump-version.sh --self-test
falsify-scratch-tmpdir|bash scripts/release/falsify_scratch_tmpdir.sh --mutants
falsify-bump-tmpdir|bash scripts/release/falsify_bump_tmpdir.sh --mutants
no-pull-request-target-guard-test|sh scripts/guards/test-no-pull-request-target-guard.sh
falsify-exec-bit-guard|bash scripts/release/falsify_exec_bit_guard.sh --mutants
falsify-one-command|bash scripts/release/falsify_one_command.sh
pr-light-self-test|bash scripts/pr_light.sh --self-test
claim-literals|setsid --wait bash scripts/check_no_claim_literals.sh --selftest
hardcoded-paths|setsid --wait bash scripts/check_hardcoded_paths.sh --self-test
pr-review-wiring|setsid --wait bash scripts/check_pr_review_wiring.sh --self-test
fixture-quiet|setsid --wait bash scripts/ci/fixture_quiet.sh --self-test
fetch-p6-self-test|setsid --wait bash scripts/ci/fetch_p6.sh --self-test
pr-light-offline-self-test|setsid --wait bash scripts/check_pr_light_offline.sh --self-test
pr-light-fast-self-test|bash scripts/ci/pr_light_fast.sh --self-test && bash scripts/ci/require_pinned_tools.sh --self-test
LANES
}

# the PR-body lanes (the guard-tree section's steps 4 and 5): one metadata read of THIS pull request each, no other network.
pr_body_lanes() {
    [ "${GITHUB_EVENT_NAME:-}" = pull_request ] || return 0
    [ -n "${PR_NUMBER:-}" ] || { echo "pr-body-closes|echo 'PR_NUMBER is unset on a pull_request event' >&2; exit 1"; return 0; }
    cat << 'LANES'
pr-body-closes|set -euo pipefail; setsid --wait bash scripts/check_pr_closes_issue.sh --self-test; gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}" --jq '.body // ""' > "$RUNNER_TEMP/pr-body.txt"; setsid --wait bash scripts/check_pr_closes_issue.sh --body "$RUNNER_TEMP/pr-body.txt"
pr-body-ont-delta|set -euo pipefail; gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}" --jq '.body // ""' > "$RUNNER_TEMP/ont-body.txt"; gh api "repos/${GITHUB_REPOSITORY}/pulls/${PR_NUMBER}/files" --paginate --jq '.[].filename' > "$RUNNER_TEMP/ont-files.txt"; setsid --wait bash scripts/check_pr_ont_delta.sh --body "$RUNNER_TEMP/ont-body.txt" --changed "$RUNNER_TEMP/ont-files.txt"
LANES
}

shard_lanes() {
    local k=${PR_LIGHT_SHARDS:-8} i
    [[ $k =~ ^[1-9][0-9]*$ ]] || { echo "pr_light_fast: PR_LIGHT_SHARDS wants a positive integer, got '$k'" >&2; return 2; }
    for ((i = 1; i <= k; i++)); do printf 'guard-tree-shard-%s-of-%s|bash scripts/ci_guards.sh guard-tree --shard %s/%s\n' "$i" "$k" "$i" "$k"; done
}

# add_worktree <dir>: concurrent `git worktree add` calls can collide on git's own locks; a lane retries, it never skips.
add_worktree() {
    local t
    for t in 1 2 3 4 5; do
        git -C "$REPO_ROOT" worktree add -q --detach "$1" HEAD 2> /dev/null && return 0
        rm -rf "${1:?}"; sleep "$t"
    done
    return 1
}

# run_lanes <scratch> -- reads "name|cmd" rows on stdin; every row runs concurrently in its own worktree of HEAD.
# Prints one row per lane in INPUT order (so two runs of one sha print the same lines) and returns 0 iff every lane exited 0.
run_lanes() {
    local scratch=$1 name cmd n=0 i bad=0 rc
    local -a names=() pids=()
    mkdir -p "$scratch" || return 2
    while IFS='|' read -r name cmd; do
        [ -n "$name" ] || continue
        names+=("$name"); n=$((n + 1))
        (
            wt="$scratch/wt-$n"
            add_worktree "$wt" || { echo "pr_light_fast: cannot make a worktree for $name" >> "$scratch/log-$n"; exit 2; }
            cd "$wt" || exit 2
            export GITHUB_WORKSPACE="$wt" TMPDIR="$scratch/tmp-$n" RUNNER_TEMP="$scratch/rt-$n"
            mkdir -p "$TMPDIR" "$RUNNER_TEMP"
            bash -c "$cmd" >> "$scratch/log-$n" 2>&1
        ) &
        pids+=("$!")
    done
    [ "$n" -gt 0 ] || { echo "NO-GO: zero lanes ran; that is not a pass" >&2; return 2; }
    trap 'kill "${pids[@]}" 2> /dev/null; wait; return 130' TERM INT
    for ((i = 0; i < n; i++)); do
        wait "${pids[$i]}"; rc=$?
        if [ "$rc" -eq 0 ]; then
            printf 'ok    %s\n' "${names[$i]}"
        else
            bad=1
            printf '::group::FAIL  %s (rc=%s)\n' "${names[$i]}" "$rc"
            tail -n 200 "$scratch/log-$((i + 1))"
            printf '::endgroup::\n::error::pr-light lane %s failed (rc=%s)\n' "${names[$i]}" "$rc"
        fi
    done
    trap - TERM INT
    return "$bad"
}

drop_worktrees() { # drop_worktrees <scratch>
    local w
    for w in "$1"/wt-*; do [ -e "$w" ] && git -C "$REPO_ROOT" worktree remove --force "$w" > /dev/null 2>&1; done
    git -C "$REPO_ROOT" worktree prune > /dev/null 2>&1
    rm -rf "${1:?}"
}

# require_event_env: pin_build_base reads PR_HEAD_SHA (pull_request) / MG_BASE_SHA (merge_group); an unset one is a RED, never a skipped pin.
require_event_env() {
    case "${GITHUB_EVENT_NAME:-}" in
        pull_request) [[ ${PR_HEAD_SHA:-} =~ ^[0-9a-f]{40}$ ]] || { echo "::error::pr_light_fast: pull_request without PR_HEAD_SHA (set it from github.event.pull_request.head.sha)" >&2; return 1; } ;;
        merge_group) [[ ${MG_BASE_SHA:-} =~ ^[0-9a-f]{40}$ ]] || { echo "::error::pr_light_fast: merge_group without MG_BASE_SHA" >&2; return 1; } ;;
    esac
}

# gate_surface_changed <changed-paths file>: 0 iff a gate's own script or recipe is among them.
gate_surface_changed() { grep -qE "$GATE_SURFACE_ERE" "$1"; }

# proofs_due: prints 1 (run the gate proofs) or 0. PR_LIGHT_PROOFS=all forces 1 (nightly); an unreadable diff is an error, never 0.
proofs_due() {
    local f rc=0
    [ "${PR_LIGHT_PROOFS:-}" = all ] && { echo 1; return 0; }
    f="$(mktemp "${TMPDIR:-/tmp}/prlf-changed.XXXXXX")" || return 2
    git -C "$REPO_ROOT" diff --name-only "${PR_LIGHT_BASE:-origin/main}...HEAD" > "$f" || rc=2
    if [ "$rc" = 0 ]; then if gate_surface_changed "$f"; then echo 1; else echo 0; fi; fi
    rm -f "${f:?}"
    return "$rc"
}

main_run() {
    cd "$REPO_ROOT" || exit 2
    command -v jq > /dev/null 2>&1 || { echo "::error::pr_light_fast: jq is not provisioned on this runner (forjar machines/*/forjar.yaml pins jq 1.8.x); this gate does not download it" >&2; exit 2; }
    local scratch rc
    require_event_env || exit 1
    scratch="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/pr-light-fast.XXXXXX")" || exit 2
    trap 'drop_worktrees "$scratch"' EXIT
    local proofs t0=$SECONDS
    proofs="$(proofs_due)" || { echo "::error::pr_light_fast: cannot list this PR's changed paths vs ${PR_LIGHT_BASE:-origin/main}; that is not a pass" >&2; exit 2; }
    echo "pr_light_fast: gate proofs ${proofs} (1 = a gate script/recipe changed, or PR_LIGHT_PROOFS=all)"
    export GUARD_TREE_SELF_TESTS="$proofs"
    [ "$proofs" = 1 ] || export CI_GUARDS_SKIP=proofs
    # the guard-tree section's step 2: the comparand every ratchet reads. Serial and first, like the section ran it.
    if [ "$proofs" = 1 ]; then bash scripts/ci/pin_build_base.sh --self-test || { echo "::error::pin_build_base self-test failed"; exit 1; }; fi
    bash scripts/ci/pin_build_base.sh || { echo "::error::pin_build_base failed"; exit 1; }
    { gate_lanes; if [ "$proofs" = 1 ]; then proof_lanes; fi; pr_body_lanes; shard_lanes; } | run_lanes "$scratch"
    rc=$?
    echo "pr_light_fast: lanes done, rc=$rc, wall ${SECONDS}s total, $((SECONDS - t0))s in lanes"
    return "$rc"
}

self_test() {
    local pass=0 fail=0 d out1 out2 rc
    row() { if [ "$2" = "$3" ]; then pass=$((pass + 1)); printf '  ok    %s\n' "$1"; else fail=$((fail + 1)); printf '  FAIL  %s want=%s got=%s\n' "$1" "$3" "$2"; fi; }
    d="$(mktemp -d "${TMPDIR:-/tmp}/prlf-test.XXXXXX")" || return 2
    set +e
    out1="$(printf 'a|true\nb|false\nc|true\n' | run_lanes "$d/1" 2> /dev/null)"; rc=$?
    row "one failing lane fails the driver" "$rc" 1
    row "the failing lane is named" "$(printf '%s\n' "$out1" | grep -c 'FAIL  b (rc=1)')" 1
    row "a failure does not hide the other lanes" "$(printf '%s\n' "$out1" | grep -c '^ok    [ac]$')" 2
    out2="$(printf 'a|true\nc|true\n' | run_lanes "$d/2" 2> /dev/null)"; rc=$?
    row "all green exits 0" "$rc" 0
    out1="$(printf 'a|true\nc|true\n' | run_lanes "$d/3" 2> /dev/null)"
    row "determinism: two runs of one lane set print identical lines" "$([ "$out1" = "$out2" ] && echo same || echo differ)" same
    printf '' | run_lanes "$d/4" > /dev/null 2>&1; rc=$?
    row "zero lanes is not a pass" "$rc" 2
    out1="$(printf 'slow|sleep 1; true\nfast|true\n' | run_lanes "$d/5" 2> /dev/null)"
    row "lanes print in input order, not finish order" "$(printf '%s\n' "$out1" | head -n 1)" 'ok    slow'
    out1="$(printf 'x|exit 7\n' | run_lanes "$d/6" 2> /dev/null)"
    row "a lane's own rc is reported" "$(printf '%s\n' "$out1" | grep -c 'rc=7')" 2
    row "the lane table has every old cheap check plus the offline falsifier and the self-tests (22 lanes)" "$(cheap_lanes | grep -c '|')" 22
    row "no proof lane is a gate lane" "$(gate_lanes | grep -cE -- '--self-test|--mutants|--selftest|_test\.sh|test-no-')" 0
    printf 'crates/aprender-core/src/lib.rs\ndocs/x.md\n' > "$d/prod"; printf 'crates/a.rs\nscripts/check_x.sh\n' > "$d/gate"
    row "a product-code diff does not run the gate proofs" "$(gate_surface_changed "$d/prod" && echo y || echo n)" n
    row "a diff touching a gate script runs them" "$(gate_surface_changed "$d/gate" && echo y || echo n)" y
    printf '.github/workflows/ci.yml\n' > "$d/wf"; printf 'Makefile\n' > "$d/mk"
    row "a workflow change runs them" "$(gate_surface_changed "$d/wf" && echo y || echo n)" y
    row "a Makefile change runs them" "$(gate_surface_changed "$d/mk" && echo y || echo n)" y
    row "PR_LIGHT_PROOFS=all forces them (nightly)" "$(PR_LIGHT_PROOFS=all proofs_due)" 1
    # proof or gate is read off the step's commands (#6859 review: two live gates were skipped by their NAME)
    isp() { printf '%b' "$1" | bash "$REPO_ROOT/scripts/lib/ci_guard_steps.sh" is-proof > /dev/null 2>&1 && echo proof || echo gate; }
    row "a --self-test command is a proof" "$(isp 'setsid --wait bash scripts/check_x.sh --self-test')" proof
    row "a --selftest command is a proof" "$(isp 'bash scripts/x.sh --selftest')" proof
    row "a python --self-test is a proof" "$(isp 'python3 scripts/x.py --self-test')" proof
    row "a _test.sh case table is a proof" "$(isp 'bash scripts/tests/guard_tree_test.sh')" proof
    row "a mutate_ suite is a proof" "$(isp 'bash scripts/mutate_x_guard.sh')" proof
    row "a block of self-tests with comments is a proof" "$(isp 'set -euo pipefail\nbash scripts/a.sh --selftest\npython3 scripts/b.py --self-test   # why\n')" proof
    row "two proofs joined by && are a proof" "$(isp 'bash scripts/a.sh --self-test && bash scripts/b.sh --self-test')" proof
    row "a guard run bare is a gate" "$(isp 'setsid --wait bash scripts/check_mutants_pool.sh')" gate
    row "a gate and its RED proof in one step is a gate" "$(isp 'bash tests/spec/pp066_v16_defects.sh && bash tests/spec/pp066_v16_defects.sh --v15-red')" gate
    row "a gate after a proof on one line is a gate" "$(isp 'bash scripts/x.sh --self-test; bash scripts/x.sh')" gate
    row "a gate on a second line is a gate" "$(isp 'bash scripts/x.sh --self-test\nbash scripts/x.sh\n')" gate
    row "a proof with a tail is a gate" "$(isp 'bash scripts/x.sh --self-test || true')" gate
    row "a --check is a gate" "$(isp 'bash scripts/bump-version.sh --check')" gate
    row "a shell construct is a gate" "$(isp 'if true; then bash scripts/x.sh --self-test; fi')" gate
    row "an empty step is a gate" "$(isp '')" gate
    out1="$(bash "$REPO_ROOT/scripts/lib/ci_guard_steps.sh" proofs guard-tree 2> /dev/null)"
    row "guard-tree: the mutants-pool gate is never skipped" "$(printf '%s\n' "$out1" | awk -F '\t' '$4 == "setsid --wait bash scripts/check_mutants_pool.sh" { print $1 }')" gate
    row "guard-tree: the PP-066 defect table is never skipped" "$(printf '%s\n' "$out1" | awk -F '\t' '$4 ~ /^bash tests\/spec\/pp066_v16_defects\.sh / { print $1 }')" gate
    row "guard-tree: the guard runner itself is never skipped" "$(printf '%s\n' "$out1" | awk -F '\t' '$4 == "bash scripts/guard_tree.sh --no-cargo" { print $1 }')" gate
    row "guard-tree: some steps are proofs" "$([ "$(printf '%s\n' "$out1" | grep -c '^proof')" -ge 10 ] && echo y || echo n)" y
    # every skipped step, command by command: none may be a command without a proof marker (a second, looser reading of the same steps)
    row "guard-tree: no skipped step holds a command that is not a proof" "$(printf '%s\n' "$out1" | awk -F '\t' '$1 == "proof" { n = split($4, c, /;| && /); for (i = 1; i <= n; i++) { x = c[i]; sub(/[ ]+#.*$/, "", x); gsub(/^ +| +$/, "", x); if (x == "" || x == "set -euo pipefail") continue; if (x !~ /--self-test$|--selftest$|--mutants$|--v15-red$|_test\.sh$|\/mutate_[a-z_]+\.sh$/) bad++ } } END { print bad + 0 }')" 0
    # the live skip, on the real manifest: of the mutants-pool pair, the gate (m11) runs and only its RED proof (m12) is skipped
    out2="$(cd "$REPO_ROOT" && CI_GUARDS_SKIP=proofs bash scripts/lib/ci_guard_steps.sh run guard-tree --no-stream --only 'mutants pool \(C108\)|mutants-pool guard can still go RED' 2>&1)"
    row "run: the mutants-pool proof is skipped" "$(printf '%s\n' "$out2" | grep -c '^PROOF-SKIP guard-tree#m12 ')" 1
    row "run: the mutants-pool gate is not skipped" "$(printf '%s\n' "$out2" | grep -c '^PROOF-SKIP guard-tree#m11 ')" 0
    row "run: exactly one step is skipped of the two" "$(printf '%s\n' "$out2" | grep -c '^PROOF-SKIP ')" 1
    row "an unknown CI_GUARDS_SKIP is refused, not read as nothing" "$(CI_GUARDS_SKIP=all bash "$REPO_ROOT/scripts/lib/ci_guard_steps.sh" run guard-tree --shard 1/64 > /dev/null 2>&1; echo $?)" 2
    row "the nightly runs every proof (guards-nightly.yml, PR_LIGHT_PROOFS=all)" "$(awk '/PR_LIGHT_PROOFS: all/ { e = 1 } e && /run: setsid --wait bash scripts\/ci\/pr_light_fast\.sh$/ { print "wired"; exit }' "$REPO_ROOT/.github/workflows/guards-nightly.yml")" wired
    row "shard lanes partition 1..K" "$(PR_LIGHT_SHARDS=3 shard_lanes | cut -d'|' -f1 | paste -sd,)" 'guard-tree-shard-1-of-3,guard-tree-shard-2-of-3,guard-tree-shard-3-of-3'
    row "a bad shard count is refused" "$(PR_LIGHT_SHARDS=0 shard_lanes 2> /dev/null; echo $?)" 2
    local sha=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
    row "pull_request without PR_HEAD_SHA is RED" "$(GITHUB_EVENT_NAME=pull_request PR_HEAD_SHA= require_event_env 2>/dev/null; echo $?)" 1
    row "pull_request with a non-sha PR_HEAD_SHA is RED" "$(GITHUB_EVENT_NAME=pull_request PR_HEAD_SHA=main require_event_env 2>/dev/null; echo $?)" 1
    row "pull_request with PR_HEAD_SHA passes" "$(GITHUB_EVENT_NAME=pull_request PR_HEAD_SHA=$sha require_event_env; echo $?)" 0
    row "merge_group without MG_BASE_SHA is RED" "$(GITHUB_EVENT_NAME=merge_group MG_BASE_SHA= require_event_env 2>/dev/null; echo $?)" 1
    row "a local run needs neither" "$(GITHUB_EVENT_NAME=local require_event_env; echo $?)" 0
    for w in "$d"/*/wt-*; do [ -e "$w" ] && git -C "$REPO_ROOT" worktree remove --force "$w" > /dev/null 2>&1; done
    rm -rf "${d:?}"
    git -C "$REPO_ROOT" worktree prune > /dev/null 2>&1
    set -e
    printf 'pr_light_fast self-test: %s passed, %s failed\n' "$pass" "$fail"
    [ "$fail" -eq 0 ]
}

case "${1:-}" in
    --self-test) self_test ;;
    --list) { cheap_lanes; GITHUB_EVENT_NAME=pull_request PR_NUMBER=0 pr_body_lanes; shard_lanes; } | cut -d'|' -f1 ;;
    # the command of every lane, one per line: what check_guards_are_wired.sh reads to see which guards the lanes run.
    --list-commands) { cheap_lanes; GITHUB_EVENT_NAME=pull_request PR_NUMBER=0 pr_body_lanes; shard_lanes; } | cut -d'|' -f2- ;;
    '') main_run ;;
    *) echo "usage: $(basename "$0") [--list|--list-commands|--self-test]" >&2; exit 2 ;;
esac
