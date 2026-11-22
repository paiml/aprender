#!/usr/bin/env bash
# check_nightly_train_pool.sh -- nightly-train.yml must pin the intel X64 clean-room pool
# and run the no-token preflight before judging (#7375).
#
# Incident (#7375, run 44906158865): `train` asked for [self-hosted, Linux, X64, clean-room]
# and landed on `kfocus-mq` (labels self-hosted,Linux,X64,clean-room,tier2,kfocus), whose
# uid carries ~/.cargo/credentials.toml, so nightly_train.sh's strict no_token() guard
# refused to start (exit 2). Pinning `intel` keeps both jobs on the intel clean-room pool
# (like release-gates-nightly.yml and readiness-nightly.yml), and running `--probe-no-token`
# right after checkout gives a fast, named preflight verdict if a mislabelled box ever has
# a reachable registry token.
#
#   bash scripts/check_nightly_train_pool.sh [workflow]   the case table, then the real workflow
#   bash scripts/check_nightly_train_pool.sh --self-test  case table + planted workflow defects
set -uo pipefail
case "${1:-}" in -h|--help) printf '%s\n' 'usage: bash scripts/check_nightly_train_pool.sh [--self-test | WORKFLOW]'; exit 0 ;; esac

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2

job_block() { # job_block <workflow> <job>
    awk -v job="  $2:" '
        $0 == job { f = 1; next }
        f && /^  [A-Za-z0-9_-]+:/ { f = 0 }
        f { print }
    ' "$1"
}

job_runs_on() { # job_runs_on <workflow> <job>
    job_block "$1" "$2" | awk '/^    runs-on:/{sub(/^    runs-on:[ ]*/, ""); sub(/[ ]+#.*$/, ""); print; exit}'
}

has_label() { # has_label <runs-on text> <label> -> 0 iff whole label present (case-insensitive)
    local t l
    t=" $(printf '%s' "$1" | tr '[],"' '    ' | tr '[:upper:]' '[:lower:]') "
    l="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
    case "$t" in *" $l "*) return 0 ;; *) return 1 ;; esac
}

pool_ok() { # 0 iff it pins [self-hosted, Linux, X64, clean-room, intel] and excludes kfocus/mq/tier2/yoga/gpu/arm64
    local low
    low="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
    case "$low" in *kfocus*|*tier2*|*yoga*) return 1 ;; esac
    if has_label "$1" mq || has_label "$1" gpu || has_label "$1" cuda || has_label "$1" arm64; then
        return 1
    fi
    if has_label "$1" self-hosted && has_label "$1" linux && has_label "$1" x64 \
        && has_label "$1" clean-room && has_label "$1" intel; then
        return 0
    fi
    return 1
}

# preflight_check <workflow> -> 0 iff train invokes `nightly_train.sh --probe-no-token` before `nightly_train.sh --out`
preflight_check() {
    local wf="$1" res
    res="$(job_block "$wf" train | awk '
        /^[[:space:]]*-?[[:space:]]*name:/ { next }
        {
            line = $0
            sub(/#.*$/, "", line)
            if (probe == 0 && line ~ /nightly_train\.sh[[:space:]]+--probe-no-token([[:space:]]|$)/) probe = NR
            if (out == 0 && line ~ /nightly_train\.sh[[:space:]]+.*--out([[:space:]]|$)/) out = NR
        }
        END { printf "%d %d\n", probe + 0, out + 0 }
    ')"
    local p="${res%% *}" o="${res##* }"
    if [ "$p" -eq 0 ]; then
        printf 'FAIL  train job has no nightly_train.sh --probe-no-token preflight step\n' >&2
        return 1
    fi
    if [ "$o" -eq 0 ]; then
        printf 'FAIL  train job has no nightly_train.sh --out step\n' >&2
        return 1
    fi
    if [ "$p" -ge "$o" ]; then
        printf 'FAIL  train --probe-no-token (line %s) must run before --out (line %s)\n' "$p" "$o" >&2
        return 1
    fi
    printf 'ok    train --probe-no-token (line %s) precedes --out (line %s)\n' "$p" "$o"
    return 0
}

check_workflow() { # check_workflow <workflow> -> 0 green, 1 policy violation, 2 missing/unreadable/empty
    local wf="$1" bad=0 j text
    if [ ! -f "$wf" ] || [ ! -r "$wf" ]; then
        printf 'ENV   missing or unreadable workflow %s\n' "$wf" >&2
        return 2
    fi
    for j in train andon; do
        text="$(job_runs_on "$wf" "$j")"
        if [ -z "$text" ]; then
            printf 'ENV   no %s runs-on in %s -- cannot judge, not a pass\n' "$j" "$wf" >&2
            return 2
        fi
        if pool_ok "$text"; then
            printf 'ok    %s runs-on %s\n' "$j" "$text"
        else
            printf 'FAIL  %s runs-on %s does not pin [self-hosted, Linux, X64, clean-room, intel] (cannot exclude kfocus-mq)\n' "$j" "$text" >&2
            bad=1
        fi
    done
    if ! preflight_check "$wf"; then
        bad=1
    fi
    return "$bad"
}

run_table() {
    local want text got bad=0
    while IFS='|' read -r want text; do
        got=0
        if ! pool_ok "$text"; then got=1; fi
        if [ "$got" = "$want" ]; then
            printf 'ok    pool %s -> %s\n' "${text:-<empty>}" "$want"
        else
            printf 'FAIL  pool %s wanted %s, got %s\n' "${text:-<empty>}" "$want" "$got" >&2
            bad=1
        fi
    done <<'POOLS'
0|[self-hosted, Linux, X64, clean-room, intel]
0|[self-hosted, Linux, X64, CLEAN-ROOM, Intel]
1|[self-hosted, Linux, X64, clean-room]
1|[self-hosted, Linux, clean-room]
1|[self-hosted, Linux, clean-room, intel]
1|[self-hosted, Linux, X64, clean-room, kfocus]
1|[self-hosted, Linux, X64, clean-room, tier2, kfocus]
1|[self-hosted, Linux, X64, clean-room, mq]
1|[self-hosted, Linux, X64, clean-room, yoga]
1|[self-hosted, Linux, X64, clean-room, intel, kfocus]
1|[self-hosted, Linux, X64, clean-room, intel, tier2]
1|[self-hosted, Linux, X64, clean-room, intel, mq]
1|[self-hosted, Linux, X64, clean-room, intel, yoga]
1|[self-hosted, Linux, X64, clean-room, intel, gpu]
1|[self-hosted, Linux, X64, clean-room, intel, cuda]
1|[self-hosted, Linux, ARM64, clean-room, intel]
1|[self-hosted, Linux, X64, clean-room-2, intel]
1|[self-hosted, Linux, X64, clean-room, intel-2]
1|
POOLS
    return "$bad"
}

self_test() {
    local d bad=0 rc
    d="$(mktemp -d "${TMPDIR:-/tmp}/nightly-train-pool.XXXXXX")" || exit 2
    trap 'rm -rf -- "${d:?}"' EXIT

    if ! run_table; then bad=1; fi

    mk_wf() { # mk_wf <file> <train_runs_on> <andon_runs_on> <mode: ok|noprobe|lateprobe|commentprobe>
        local f="$1" tr="$2" an="$3" mode="$4"
        {
            printf 'jobs:\n'
            printf '  train:\n'
            printf '    runs-on: %s\n' "$tr"
            printf '    steps:\n'
            if [ "$mode" = "ok" ]; then
                printf '      - run: bash scripts/release/nightly_train.sh --probe-no-token\n'
                printf '      - run: bash scripts/release/nightly_train.sh --out %s\n' '"$RUNNER_TEMP/nt"'
            elif [ "$mode" = "noprobe" ]; then
                printf '      - run: bash scripts/release/nightly_train.sh --out %s\n' '"$RUNNER_TEMP/nt"'
            elif [ "$mode" = "lateprobe" ]; then
                printf '      - run: bash scripts/release/nightly_train.sh --out %s\n' '"$RUNNER_TEMP/nt"'
                printf '      - run: bash scripts/release/nightly_train.sh --probe-no-token\n'
            elif [ "$mode" = "commentprobe" ]; then
                printf '      - run: echo skipped # bash scripts/release/nightly_train.sh --probe-no-token\n'
                printf '      - run: bash scripts/release/nightly_train.sh --out %s\n' '"$RUNNER_TEMP/nt"'
            fi
            printf '  andon:\n'
            printf '    runs-on: %s\n' "$an"
        } > "$f"
    }

    mk_wf "$d/good.yml" '[self-hosted, Linux, X64, clean-room, intel]' '[self-hosted, Linux, X64, clean-room, intel]' ok
    if check_workflow "$d/good.yml" > /dev/null 2>&1; then
        printf 'ok    good workflow is GREEN\n'
    else
        printf 'FAIL  good workflow refused\n' >&2; bad=1
    fi

    mk_wf "$d/bare-train.yml" '[self-hosted, Linux, X64, clean-room]' '[self-hosted, Linux, X64, clean-room, intel]' ok
    if check_workflow "$d/bare-train.yml" > /dev/null 2>&1; then
        printf 'FAIL  bare clean-room on train passed (#7375 defect)\n' >&2; bad=1
    else
        printf 'ok    bare clean-room on train is RED (#7375)\n'
    fi

    mk_wf "$d/bare-andon.yml" '[self-hosted, Linux, X64, clean-room, intel]' '[self-hosted, Linux, X64, clean-room]' ok
    if check_workflow "$d/bare-andon.yml" > /dev/null 2>&1; then
        printf 'FAIL  bare clean-room on andon passed\n' >&2; bad=1
    else
        printf 'ok    bare clean-room on andon is RED\n'
    fi

    mk_wf "$d/kfocus.yml" '[self-hosted, Linux, X64, clean-room, intel, kfocus]' '[self-hosted, Linux, X64, clean-room, intel]' ok
    if check_workflow "$d/kfocus.yml" > /dev/null 2>&1; then
        printf 'FAIL  kfocus label passed\n' >&2; bad=1
    else
        printf 'ok    kfocus label is RED\n'
    fi

    mk_wf "$d/noprobe.yml" '[self-hosted, Linux, X64, clean-room, intel]' '[self-hosted, Linux, X64, clean-room, intel]' noprobe
    if check_workflow "$d/noprobe.yml" > /dev/null 2>&1; then
        printf 'FAIL  missing --probe-no-token preflight passed\n' >&2; bad=1
    else
        printf 'ok    missing --probe-no-token preflight is RED\n'
    fi

    mk_wf "$d/lateprobe.yml" '[self-hosted, Linux, X64, clean-room, intel]' '[self-hosted, Linux, X64, clean-room, intel]' lateprobe
    if check_workflow "$d/lateprobe.yml" > /dev/null 2>&1; then
        printf 'FAIL  late --probe-no-token after --out passed\n' >&2; bad=1
    else
        printf 'ok    late --probe-no-token after --out is RED\n'
    fi

    mk_wf "$d/commentprobe.yml" '[self-hosted, Linux, X64, clean-room, intel]' '[self-hosted, Linux, X64, clean-room, intel]' commentprobe
    if check_workflow "$d/commentprobe.yml" > /dev/null 2>&1; then
        printf 'FAIL  commented --probe-no-token passed\n' >&2; bad=1
    else
        printf 'ok    commented --probe-no-token is RED\n'
    fi

    printf 'jobs:\n  train:\n    runs-on:\n  andon:\n    runs-on: [self-hosted, Linux, X64, clean-room, intel]\n' > "$d/empty.yml"
    rc=0; check_workflow "$d/empty.yml" > /dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 2 ]; then
        printf 'ok    empty runs-on is ENV rc=2\n'
    else
        printf 'FAIL  empty runs-on gave rc=%s\n' "$rc" >&2; bad=1
    fi

    printf 'jobs: {}\n' > "$d/none.yml"
    rc=0; check_workflow "$d/none.yml" > /dev/null 2>&1 || rc=$?
    if [ "$rc" -eq 2 ]; then
        printf 'ok    missing job is ENV rc=2\n'
    else
        printf 'FAIL  missing job gave rc=%s\n' "$rc" >&2; bad=1
    fi

    if [ "$bad" -eq 0 ]; then
        printf 'SELF-TEST PASSED\n'
        exit 0
    fi
    printf 'SELF-TEST FAILED\n' >&2
    exit 1
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi

if ! run_table; then
    exit 1
fi
check_workflow "${1:-$ROOT/.github/workflows/nightly-train.yml}"
