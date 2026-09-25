#!/usr/bin/env bash
# check_guard_steps_run_all.sh -- a guard step may not hide behind another
# guard's failure (#4415, lever 0 of the RC-speed report).
#
# WHY THIS EXISTS
# ---------------
# GitHub Actions skips every later step of a job once one step fails, unless
# the later step's `if:` names a status function (`!cancelled()`, `always()`,
# `failure()`). On the 0.69.4 RC PR #4318 every red guard job reported
# exactly ONE failure. guard-cargo skipped 15-71 of its 77 steps (median 55),
# so a fix-push-wait loop of ~35-50 min per cycle surfaced the guards one at
# a time: 02d46180e -> cef1b1241 -> 696576edb -> 6281d9fbe.
#
# THE FIX (operator amendment A4, 2026-09-25). A guard job J runs ONE step,
# `bash scripts/ci_guards.sh J`, which runs every step of the manifest job
# `J-steps` (`if: false`, never run by GitHub) and keeps going past a red one.
# `make guards-local` runs the same script. This guard keeps that wiring true:
#   - fail-fast steps after J's setup step are a SHRINK-ONLY ratchet
#     (scripts/guard_fail_fast_baseline.txt, both rows 0);
#   - J-steps must be `if: false`, or GitHub would run it a second time;
#   - a manifest step is name/run/env only, and env may use only the
#     ${{ }} the runner can resolve (github.token, runner.temp);
#   - J must have a step that runs `scripts/ci_guards.sh J`, and a step with
#     `id: guard-setup`: the runner's `if:` reads its outcome, and without it
#     the runner step is skipped and the job goes green having run nothing;
#   - no manifest without its job.
#
# The step list is read from ci.yml by scripts/lib/ci_guard_steps.py, the
# runner itself. There is no second list.
#
#   bash scripts/check_guard_steps_run_all.sh              # check
#   bash scripts/check_guard_steps_run_all.sh --self-test  # case table
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${REPO_ROOT}/scripts/lib/ci_guard_steps.py"

usage() {
    sed -n '2,/^set -uo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

fixture() { # fixture <file> <kind>... ; kinds: plain cancelled always event event_cancelled setup nosetup pass fail
    local f=$1 k
    shift
    {
        printf 'on: push\njobs:\n  guard-x:\n    runs-on: ubuntu-latest\n    steps:\n'
        printf '      - uses: actions/checkout@v7\n'
        # the setup step carries `id: guard-setup` unless a later `setup` kind does, or `nosetup`
        case " $* " in *" setup "*|*" nosetup "*) printf '      - name: fetch\n' ;; *) printf '      - id: guard-setup\n        name: fetch\n' ;; esac
        printf '        run: "true"\n'
        for k in "$@"; do
            [ "$k" = nosetup ] && continue
            printf '      - name: %s\n' "$k"
            case "$k" in
                cancelled) printf '        if: ${{ !cancelled() && steps.guard-setup.outcome == %s }}\n' "'success'" ;;
                always) printf '        if: always()\n' ;;
                event) printf '        if: github.event_name == %s\n' "'pull_request'" ;;
                event_cancelled) printf '        if: "!cancelled() && github.event_name == %s"\n' "'pull_request'" ;;
                setup) printf '        id: guard-setup\n' ;;
            esac
            case "$k" in
                fail) printf '        run: "exit 3"\n' ;;
                *) printf '        run: "true"\n' ;;
            esac
        done
    } > "$f"
}

self_test() {
    local d fail=0 n=0
    d=$(mktemp -d) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf -- '${d:?}'" RETURN
    printf 'guard-x 2\n' > "$d/base2"
    printf 'guard-x 0\n' > "$d/base0"
    printf 'other 5\n' > "$d/base_missing"

    case_row() { # case_row <label> <want_rc> <workflow> <baseline> [job]
        local got
        python3 "$LIB" --workflow "$3" check-run-all --baseline "$4" "${5:-guard-x}" > "$d/out" 2>&1
        got=$?
        n=$((n + 1))
        if [ "$got" -eq "$2" ]; then
            printf 'ok   %-58s rc=%s\n' "$1" "$got"
        else
            printf 'FAIL %-58s rc=%s want %s\n' "$1" "$got" "$2"
            sed 's/^/     | /' "$d/out"
            fail=1
        fi
    }

    fixture "$d/w1.yml" plain plain
    case_row "2 fail-fast steps, baseline 2 -> pass" 0 "$d/w1.yml" "$d/base2"
    case_row "2 fail-fast steps, baseline 0 -> RED" 1 "$d/w1.yml" "$d/base0"
    fixture "$d/w2.yml" plain plain plain
    case_row "3rd fail-fast step over baseline 2 -> RED" 1 "$d/w2.yml" "$d/base2"
    fixture "$d/w3.yml" cancelled always event_cancelled
    case_row "!cancelled()/always()/!cancelled()&&event: 0 fail-fast" 0 "$d/w3.yml" "$d/base0"
    fixture "$d/w4.yml" event
    case_row "event-only if: is still fail-fast -> RED at baseline 0" 1 "$d/w4.yml" "$d/base0"
    fixture "$d/w5.yml" setup plain plain
    case_row "id: guard-setup is setup, the 2 after it count" 0 "$d/w5.yml" "$d/base2"
    fixture "$d/w6.yml" nosetup
    case_row "no id: guard-setup -> RED (the runner's if: would skip it)" 1 "$d/w6.yml" "$d/base2"
    case_row "missing job fails closed (rc 2)" 2 "$d/w1.yml" "$d/base2" no-such-job
    case_row "job with no baseline row -> RED" 1 "$d/w1.yml" "$d/base_missing"
    case_row "unreadable workflow fails closed (rc 2)" 2 "$d/absent.yml" "$d/base2"

    # The runner half (#4416) must RUN PAST a failure and report every one.
    local repo="$d/repo"
    mkdir -p "$repo" && git -C "$repo" init -q && mkdir -p "$repo/.github/workflows"
    fixture "$repo/.github/workflows/ci.yml" pass fail pass fail
    local got
    ( cd "$repo" && CI_GUARDS_SCRATCH="$d/scratch" python3 "$LIB" run guard-x ) > "$d/out" 2>/dev/null
    got=$?
    n=$((n + 1))
    if [ "$got" -eq 1 ] && grep -q '^SUMMARY: 2 failed / 4 ran / 0 skipped$' "$d/out"; then
        printf 'ok   %-58s rc=%s\n' "runner reports BOTH failures of 4 steps in one pass" "$got"
    else
        printf 'FAIL %-58s rc=%s\n' "runner reports BOTH failures of 4 steps in one pass" "$got"
        sed 's/^/     | /' "$d/out"
        fail=1
    fi
    fixture "$repo/.github/workflows/ci.yml" event
    ( cd "$repo" && CI_GUARDS_SCRATCH="$d/scratch" python3 "$LIB" run guard-x ) > "$d/out" 2>/dev/null
    got=$?
    n=$((n + 1))
    if [ "$got" -eq 2 ]; then
        printf 'ok   %-58s rc=%s\n' "runner with nothing runnable refuses a vacuous pass" "$got"
    else
        printf 'FAIL %-58s rc=%s want 2\n' "runner with nothing runnable refuses a vacuous pass" "$got"
        fail=1
    fi

    manifest_rows || fail=1

    if [ "$fail" -ne 0 ]; then
        echo "check_guard_steps_run_all self-test: FAILED"
        return 1
    fi
    echo "check_guard_steps_run_all self-test: ${n}/${n} cases pass"
}

# mfixture <file> <variant> [step-run...] -- a guard job + its `-steps` manifest.
# variants: good no_if_false step_if bad_expr run_expr no_runner orphan no_manifest
# (no_manifest: the -steps job is deleted and a step is left behind the runner.)
mfixture() {
    local f=$1 v=$2 r
    shift 2
    {
        printf 'on: push\njobs:\n  guard-x:\n    runs-on: ubuntu-latest\n    steps:\n'
        printf '      - id: guard-setup\n        name: fetch\n        run: "true"\n'
        if [ "$v" != no_runner ]; then
            printf '      - name: run all\n        if: ${{ !cancelled() }}\n        run: bash scripts/ci_guards.sh guard-x\n'
        fi
        if [ "$v" = no_manifest ]; then
            printf '      - name: left behind\n        if: ${{ !cancelled() }}\n        run: "true"\n'
            return 0
        fi
        printf '  guard-x-steps:\n'
        [ "$v" = no_if_false ] || printf '    if: false\n'
        printf '    runs-on: ubuntu-latest\n    steps:\n'
        printf '      - name: tokened\n        env:\n          TOK: ${{ %s }}\n' \
            "$([ "$v" = bad_expr ] && echo github.event.pull_request.body || echo github.token)"
        printf '        run: test "$TOK" = tok-123\n'
        [ "$v" = step_if ] && printf '      - name: gated\n        if: always()\n        run: "true"\n'
        [ "$v" = run_expr ] && printf '      - name: templated\n        run: echo "${{ github.sha }}"\n'
        for r in "$@"; do printf '      - name: "step %s"\n        run: |\n          %s\n' "${#r}" "$r"; done
        if [ "$v" = orphan ]; then
            printf '  guard-y-steps:\n    if: false\n    runs-on: ubuntu-latest\n    steps:\n      - name: a\n        run: "true"\n'
        fi
    } > "$f"
}

# The A4 rows. Uses self_test's $d, $n, case_row.
manifest_rows() {
    local v bad=0 got repo="$d/mrepo" t0 t1
    printf 'guard-x 0\n' > "$d/mbase"
    mfixture "$d/m_good.yml" good "true"
    case_row "manifest if:false + runner step -> pass" 0 "$d/m_good.yml" "$d/mbase"
    for v in no_if_false step_if bad_expr run_expr no_runner orphan no_manifest; do
        mfixture "$d/m_$v.yml" "$v" "true"
        case_row "manifest defect '$v' -> RED" 1 "$d/m_$v.yml" "$d/mbase"
    done

    mrun() { # mrun <label> <want_rc> <grep-E pattern that must match> [runner args] -- CI mode
        local label=$1 want=$2 pat=$3
        shift 3
        : > "$d/msummary"
        ( cd "$repo" && GITHUB_ACTIONS=true GITHUB_TOKEN=tok-123 GITHUB_STEP_SUMMARY="$d/msummary" \
            CI_GUARDS_SCRATCH="$d/mscratch" python3 "$LIB" run "$@" guard-x ) > "$d/out" 2>&1
        got=$?
        n=$((n + 1))
        if [ "$got" -eq "$want" ] && grep -Eq "$pat" "$d/out"; then
            printf 'ok   %-58s rc=%s\n' "$label" "$got"
        else
            printf 'FAIL %-58s rc=%s want %s /%s/\n' "$label" "$got" "$want" "$pat"
            sed 's/^/     | /' "$d/out"
            bad=1
        fi
    }
    mkdir -p "$repo/.github/workflows" && git -C "$repo" init -q
    mfixture "$repo/.github/workflows/ci.yml" good "exit 3" "true" "exit 4"
    mrun "CI: all run past red, both failures reported" 1 '^SUMMARY: 2 failed / 4 ran / 0 skipped$'
    n=$((n + 1))
    if [ "$(grep -c '^::error title=guard-x' "$d/out")" = 2 ] && grep -q '| FAIL |' "$d/msummary"; then
        printf 'ok   %-58s\n' "CI: one ::error per failure + a step-summary table"
    else printf 'FAIL %-58s\n' "CI: one ::error per failure + a step-summary table"; bad=1; fi
    # shellcheck disable=SC2016 # expanded by the step's bash, not here
    mfixture "$repo/.github/workflows/ci.yml" good \
        'b="$RUNNER_TEMP/b"; mkdir -p "$b"; printf "exit 0\n" > "$b/probe_tool"; chmod +x "$b/probe_tool"; echo "$b" >> "$GITHUB_PATH"; echo FOO=bar >> "$GITHUB_ENV"' \
        'probe_tool' 'test "$FOO" = bar'
    RUNNER_TEMP="$d/rt" mrun "CI: GITHUB_PATH / GITHUB_ENV reach later steps" 0 '^SUMMARY: 0 failed / 4 ran'
    # shellcheck disable=SC2016
    mfixture "$repo/.github/workflows/ci.yml" good 'sleep 30 & echo $! > "$RUNNER_TEMP/child.pid"; sleep 30'
    t0=$SECONDS
    mkdir -p "$d/rt"
    RUNNER_TEMP="$d/rt" mrun "CI: a hung step TIMES OUT and goes red" 1 'TIMEOUT' --step-timeout 1
    t1=$SECONDS
    n=$((n + 1))
    sleep 0.2
    if [ -s "$d/rt/child.pid" ] && ! kill -0 "$(cat "$d/rt/child.pid")" 2> /dev/null; then
        printf 'ok   %-58s\n' "the timed-out step's background child is dead too"
    else printf 'FAIL %-58s\n' "the timed-out step's background child is dead too"; bad=1; fi
    n=$((n + 1))
    if [ $((t1 - t0)) -lt 15 ]; then printf 'ok   %-58s %ss\n' "timeout returned promptly" $((t1 - t0))
    else printf 'FAIL %-58s %ss\n' "timeout returned promptly" $((t1 - t0)); bad=1; fi
    n=$((n + 1))
    if [ "$(cd "$repo" && python3 "$LIB" sha guard-x)" = "$(cd "$repo" && GITHUB_ACTIONS=true python3 "$LIB" sha guard-x)" ] \
        && grep -q '^ci_guards: sha256 [0-9a-f]\{16\} manifest [0-9a-f]\{16\}' "$d/out"; then
        printf 'ok   %-58s\n' "the sha line is the same in CI and locally, and is printed"
    else printf 'FAIL %-58s\n' "the sha line is the same in CI and locally, and is printed"; bad=1; fi
    return "$bad"
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --self-test) self_test; exit $? ;;
    "") ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

command -v python3 > /dev/null 2>&1 || { echo "check_guard_steps_run_all: python3 missing" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2
rc=0
python3 "$LIB" check-run-all || rc=$?
# Every guard-shaped step in ci.yml is in a manifest or acknowledged (#4415 lane b):
# run here so CI enforces it, not only make guards-local / pre-push.
python3 "$LIB" check-coverage || { r=$?; [ "$rc" -ge "$r" ] || rc=$r; }
exit "$rc"
