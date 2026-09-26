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
# The step list is read from ci/sections.yml by scripts/lib/ci_guard_steps.sh, the
# runner itself. There is no second list.
#
#   bash scripts/check_guard_steps_run_all.sh              # check
#   bash scripts/check_guard_steps_run_all.sh --self-test  # case table
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${REPO_ROOT}/scripts/lib/ci_guard_steps.sh"

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
        bash "$LIB" --workflow "$3" check-run-all --baseline "$4" "${5:-guard-x}" > "$d/out" 2>&1
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
    mkdir -p "$repo" && git -C "$repo" init -q && mkdir -p "$repo/ci"
    fixture "$repo/ci/sections.yml" pass fail pass fail
    local got
    ( cd "$repo" && CI_GUARDS_SCRATCH="$d/scratch" bash "$LIB" run guard-x ) > "$d/out" 2>/dev/null
    got=$?
    n=$((n + 1))
    if [ "$got" -eq 1 ] && grep -q '^SUMMARY: 2 failed / 4 ran / 0 skipped$' "$d/out"; then
        printf 'ok   %-58s rc=%s\n' "runner reports BOTH failures of 4 steps in one pass" "$got"
    else
        printf 'FAIL %-58s rc=%s\n' "runner reports BOTH failures of 4 steps in one pass" "$got"
        sed 's/^/     | /' "$d/out"
        fail=1
    fi
    fixture "$repo/ci/sections.yml" event
    ( cd "$repo" && CI_GUARDS_SCRATCH="$d/scratch" bash "$LIB" run guard-x ) > "$d/out" 2>/dev/null
    got=$?
    n=$((n + 1))
    if [ "$got" -eq 2 ]; then
        printf 'ok   %-58s rc=%s\n' "runner with nothing runnable refuses a vacuous pass" "$got"
    else
        printf 'FAIL %-58s rc=%s want 2\n' "runner with nothing runnable refuses a vacuous pass" "$got"
        fail=1
    fi

    manifest_rows || fail=1
    reader_rows || fail=1

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
            CI_GUARDS_SCRATCH="$d/mscratch" bash "$LIB" run "$@" guard-x ) > "$d/out" 2>&1
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
    mkdir -p "$repo/ci" && git -C "$repo" init -q
    mfixture "$repo/ci/sections.yml" good "exit 3" "true" "exit 4"
    mrun "CI: all run past red, both failures reported" 1 '^SUMMARY: 2 failed / 4 ran / 0 skipped$'
    n=$((n + 1))
    if [ "$(grep -c '^::error title=guard-x' "$d/out")" = 2 ] && grep -q '| FAIL |' "$d/msummary"; then
        printf 'ok   %-58s\n' "CI: one ::error per failure + a step-summary table"
    else printf 'FAIL %-58s\n' "CI: one ::error per failure + a step-summary table"; bad=1; fi
    # shellcheck disable=SC2016 # expanded by the step's bash, not here
    mfixture "$repo/ci/sections.yml" good \
        'b="$RUNNER_TEMP/b"; mkdir -p "$b"; printf "exit 0\n" > "$b/probe_tool"; chmod +x "$b/probe_tool"; echo "$b" >> "$GITHUB_PATH"; echo FOO=bar >> "$GITHUB_ENV"' \
        'probe_tool' 'test "$FOO" = bar'
    RUNNER_TEMP="$d/rt" mrun "CI: GITHUB_PATH / GITHUB_ENV reach later steps" 0 '^SUMMARY: 0 failed / 4 ran'
    # shellcheck disable=SC2016 # expanded by the step's bash, not here
    mfixture "$repo/ci/sections.yml" good 'trap "touch \"$RUNNER_TEMP/restored\"" EXIT; sleep 30'
    mkdir -p "$d/rt"; rm -f "${d:?}/rt/restored"
    RUNNER_TEMP="$d/rt" mrun "CI: a timed-out step is TIMEOUT, not a crash" 1 'TIMEOUT' --step-timeout 1
    n=$((n + 1))
    if [ -e "$d/rt/restored" ]; then
        printf 'ok   %-58s\n' "a timed-out step still runs its EXIT trap (restores a mutant)"
    else printf 'FAIL %-58s\n' "a timed-out step still runs its EXIT trap (restores a mutant)"; bad=1; fi
    # shellcheck disable=SC2016
    mfixture "$repo/ci/sections.yml" good 'sleep 30 & echo $! > "$RUNNER_TEMP/child.pid"; sleep 30'
    t0=$SECONDS
    mkdir -p "$d/rt"
    RUNNER_TEMP="$d/rt" mrun "CI: a hung step TIMES OUT and goes red" 1 'TIMEOUT' --step-timeout 1
    t1=$SECONDS
    n=$((n + 1))
    # A killed child whose reaper has not collected it yet is a zombie, and kill -0 still
    # answers for a zombie: read its state instead, and give the reaper up to 2 s.
    child_dead=0
    if [ -s "$d/rt/child.pid" ]; then
        cpid="$(cat "$d/rt/child.pid")"
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            cstat="$(ps -o stat= -p "$cpid" 2> /dev/null || true)"
            case "$cstat" in "" | Z*) child_dead=1; break ;; esac
            sleep 0.2
        done
    fi
    if [ "$child_dead" = 1 ]; then
        printf 'ok   %-58s\n' "the timed-out step's background child is dead too"
    else printf 'FAIL %-58s\n' "the timed-out step's background child is dead too"; bad=1; fi
    n=$((n + 1))
    if [ $((t1 - t0)) -lt 15 ]; then printf 'ok   %-58s %ss\n' "timeout returned promptly" $((t1 - t0))
    else printf 'FAIL %-58s %ss\n' "timeout returned promptly" $((t1 - t0)); bad=1; fi
    # A jq that dies mid-stream must stop the run with rc 2, not end the step loop early
    # and report the steps it never read as nothing (the loop reads jq's records from fd 3).
    mkdir -p "$d/mutlib/lib"
    cp "$LIB" "$d/mutlib/lib/" && cp "$(dirname "$LIB")/ci_guard_yaml.awk" "$d/mutlib/lib/" \
        && cp "$(dirname "$LIB")/../ci_guards.sh" "$d/mutlib/"
    sed 's/^def step_record:$/def step_record: if (.idx == "m1" or .idx == "1") then error("injected jq death") else . end |/' \
        "$(dirname "$LIB")/ci_guard_steps.jq" > "$d/mutlib/lib/ci_guard_steps.jq"
    mfixture "$repo/ci/sections.yml" good "true" "true" "true"
    if grep -q 'injected jq death' "$d/mutlib/lib/ci_guard_steps.jq"; then
        LIB="$d/mutlib/lib/$(basename "$LIB")" mrun "CI: a jq that dies mid-stream stops the run (rc 2)" 2 'cannot read the steps of guard-x'
    else n=$((n + 1)); printf 'FAIL %-58s\n' "jq-death mutant: the sed did not apply"; bad=1; fi
    n=$((n + 1))
    local_sha="$(cd "$repo" && bash "$LIB" sha guard-x)"
    ci_sha="$(cd "$repo" && GITHUB_ACTIONS=true bash "$LIB" sha guard-x)"
    if [ "$local_sha" = "$ci_sha" ] && grep -q '^ci_guards: sha256 [0-9a-f]\{16\} manifest [0-9a-f]\{16\}' "$d/out"; then
        printf 'ok   %-58s\n' "the sha line is the same in CI and locally, and is printed"
    else printf 'FAIL %-58s\n' "the sha line is the same in CI and locally, and is printed"; bad=1; fi
    return "$bad"
}

# The YAML reader (#4415, bash port): ci/sections.yml is read by an awk block-YAML
# reader, not PyYAML. It reads the subset the job YAML uses and REFUSES (rc 2)
# everything else -- a shape it misread would be a guard that checks nothing.
# Each refusal row has a control row: the same fixture with a valid step
# appended passes, so the rc 2 is the defect and not the fixture.
# Uses self_test's $d, $n, case_row.
reader_rows() {
    local bad=0 kind got want out
    rfixture() { # rfixture <file> <step text, indented 8, after "- name: extra">
        fixture "$1" plain plain
        printf '      - name: extra\n        if: ${{ !cancelled() }}\n%s\n' "$2" >> "$1"
    }
    # shellcheck disable=SC2016 # literal YAML
    rfixture "$d/r_ok.yml" '        run: |-
          x='"'"'a: b # not a comment'"'"'

          test "$x" = "a: b # not a comment"
        env: {}
        with: [a, '"'"'b, c'"'"', "d"]'
    case_row "reader control: |- block, flow list, {} -> pass" 0 "$d/r_ok.yml" "$d/base2"
    for kind in anchor alias tag folded multiline_plain unclosed_dq duplicate_key tab_indent flow_map colon_in_plain; do
        case "$kind" in
            anchor) out='        run: &a "true"' ;;
            alias) out='        run: *a' ;;
            tag) out='        run: !!str "true"' ;;
            folded) out='        run: >
          true' ;;
            multiline_plain) out='        run: echo a
          echo b' ;;
            unclosed_dq) out='        run: "true' ;;
            duplicate_key) out='        run: "true"
        run: "false"' ;;
            tab_indent) out="$(printf '\t    run: "true"')" ;;
            flow_map) out='        env: {A: b}' ;;
            colon_in_plain) out='        run: echo a: b' ;;
        esac
        rfixture "$d/r_$kind.yml" "$out"
        case_row "reader refuses '$kind' (rc 2, never a guess)" 2 "$d/r_$kind.yml" "$d/base2"
    done

    # Fidelity: quoted scalars decode as YAML says, and a trailing comment is not a name.
    mfixture "$d/r_names.yml" good "true"
    # shellcheck disable=SC2016
    printf '%s\n' "      - name: 'it''s \"q\"'" '        run: "true"' \
        '      - name: "a\tb \"x\" \\ z"' '        run: "true"' \
        '      - name: plain # a comment' '        run: "true"' >> "$d/r_names.yml"
    got="$(bash "$LIB" --workflow "$d/r_names.yml" list guard-x 2>&1)"
    want="$(printf '== guard-x\n  m0  tokened\n  m1  step 4\n  m2  it'"'"'s "q"\n  m3  a\tb "x" \\ z\n  m4  plain')"
    n=$((n + 1))
    if [ "$got" = "$want" ]; then printf 'ok   %-58s\n' "reader: '' / \\\" / \\t / \\\\ / trailing # decode like YAML"
    else printf 'FAIL %-58s\n' "reader: '' / \\\" / \\t / \\\\ / trailing # decode like YAML"; printf '%s\n' "$got" | sed 's/^/     | /'; bad=1; fi

    # car's layout: the guard jobs are the guard-* sections that ci.yml's gate job
    # READS ("X86:guard-x" / res "$X86" guard-y); a section the gate does not
    # read (guard-z, named only by another job) is not a guard job. sections.yml's other
    # top-level keys (matrix-pins' flow maps) are outside the jobs: block and
    # are not read.
    gfixture() { # gfixture <sections> <gate ci.yml> <gate run line>
        { printf 'sovereign-ci:\n  uses: x\nmatrix-pins:\n  - {shard: {any: 1}, shards: 1}\n'
          fixture /dev/stdout plain plain | sed 1d
          printf '  guard-y:\n    runs-on: x\n    steps:\n      - id: guard-setup\n        run: "true"\n      - if: ${{ !cancelled() }}\n        run: "true"\n'
          printf '  guard-z:\n    runs-on: x\n    steps:\n      - run: "exit 9"\n'
        } > "$1"
        printf 'jobs:\n  x86-main:\n    runs-on: x\n    steps:\n      - run: >\n          folded is outside the gate\n  gate:\n    runs-on: x\n    steps:\n      - run: |\n          %s\n  after:\n    runs-on: x\n    steps:\n      - run: echo X86:guard-z\n' "$3" > "$2"
    }
    printf 'guard-x 2\nguard-y 0\n' > "$d/base_xy"
    # shellcheck disable=SC2016 # literal gate text
    gfixture "$d/g_sec.yml" "$d/g_ci.yml" 'for p in X86:guard-x X86:sov.gate; do :; done; GY=$(res "$X86" guard-y)'
    got="$(bash "$LIB" --workflow "$d/g_sec.yml" --gate-workflow "$d/g_ci.yml" check-run-all --baseline "$d/base_xy" 2>&1)"; rc=$?
    n=$((n + 1))
    if [ "$rc" = 0 ] && [ "$(printf '%s\n' "$got" | grep -c '^ok ')" = 2 ] \
        && printf '%s\n' "$got" | grep -q '^ok   guard-x: 2 fail-fast' && printf '%s\n' "$got" | grep -q '^ok   guard-y: 0 fail-fast'; then
        printf 'ok   %-58s\n' "gate reads X86:guard-x + res guard-y -> exactly those 2"
    else printf 'FAIL %-58s rc=%s\n' "gate reads X86:guard-x + res guard-y -> exactly those 2" "$rc"; printf '%s\n' "$got" | sed 's/^/     | /'; bad=1; fi
    gfixture "$d/g_sec.yml" "$d/g_ci.yml" 'echo no guard read here'
    bash "$LIB" --workflow "$d/g_sec.yml" --gate-workflow "$d/g_ci.yml" check-run-all --baseline "$d/base_xy" > "$d/out" 2>&1; rc=$?
    n=$((n + 1))
    if [ "$rc" = 2 ] && grep -q 'reads no guard-\* section' "$d/out"; then printf 'ok   %-58s\n' "a gate that reads no guard section fails closed (rc 2)"
    else printf 'FAIL %-58s rc=%s\n' "a gate that reads no guard section fails closed (rc 2)" "$rc"; sed 's/^/     | /' "$d/out"; bad=1; fi
    bash "$LIB" --workflow "$d/g_sec.yml" --gate-workflow "$d/absent.yml" check-run-all --baseline "$d/base_xy" > "$d/out" 2>&1; rc=$?
    n=$((n + 1))
    if [ "$rc" = 2 ]; then printf 'ok   %-58s\n' "an unreadable gate workflow fails closed (rc 2)"
    else printf 'FAIL %-58s rc=%s\n' "an unreadable gate workflow fails closed (rc 2)" "$rc"; bad=1; fi
    # A flow map INSIDE jobs: is still refused -- only the other top-level keys are skipped.
    { cat "$d/g_sec.yml"; printf '  guard-w:\n    runs-on: x\n    env: {A: b}\n    steps:\n      - run: "true"\n'; } > "$d/g_bad.yml"
    case_row "reader still refuses a flow map inside jobs: (rc 2)" 2 "$d/g_bad.yml" "$d/base2"

    # The port's point: no python anywhere on the path.
    n=$((n + 1))
    if ! grep -n 'python' "$LIB" "${REPO_ROOT}/scripts/ci_guards.sh" > "$d/py" 2>&1 && [ -f "$LIB" ]; then
        printf 'ok   %-58s\n' "no python in the lib or the runner"
    else printf 'FAIL %-58s\n' "no python in the lib or the runner"; sed 's/^/     | /' "$d/py"; bad=1; fi
    return "$bad"
}

case "${1:-}" in
    --help|-h) usage; exit 0 ;;
    --self-test) self_test; exit $? ;;
    "") ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
esac

command -v jq > /dev/null 2>&1 || { echo "check_guard_steps_run_all: jq missing" >&2; exit 2; }
cd "$REPO_ROOT" || exit 2
rc=0
bash "$LIB" check-run-all || rc=$?
# Every guard-shaped step in ci/sections.yml is in a manifest or acknowledged (#4415 lane b):
# run here so CI enforces it, not only make guards-local / pre-push.
bash "$LIB" check-coverage || { r=$?; [ "$rc" -ge "$r" ] || rc=$r; }
exit "$rc"
