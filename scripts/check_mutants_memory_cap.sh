#!/usr/bin/env bash
# check_mutants_memory_cap.sh -- every container that runs cargo-mutants has a memory cap (C137).
#
# A mutant can make a test allocate without bound: realizar's test binary reached 12.3 GB on yoga and
# 35.7 GB on fw16 (2026-09-30) and OOM-froze the host. --timeout bounds time, not memory. A docker
# --memory cap turns that into a kill of one test process: the test fails, so the mutant is CAUGHT.
# It is never skipped and never NOT_MEASURED, because the run still has an outcome for it.
#
# Rule: each `docker run` whose body runs `cargo mutants` or scripts/mutants_diff_gate.sh must carry
# --memory=<N>g and --memory-swap=<N>g with the same N (swap = memory, so the cap is not a swap escape),
# 1 <= N <= 16 (docker reads --memory=0 as unlimited), each flag given exactly once (docker honours the last).
# Known limit: a tripwire against a cap being dropped by accident, not against an author who hides one.
#
#   check_mutants_memory_cap.sh [file...]  default: mutants-nightly.yml, ci/sections.yml and ci.yml (mutants-shard);
#                                          each file must be readable and hold a mutation run, else rc 2
#   check_mutants_memory_cap.sh --self-test
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2

# runs <file> -> one line per mutation docker run: "<lineno> <ok|BAD> <header flags>"
runs() {
    awk '
    function flush() { if (start && mut) { ok = (m != "" && mc == 1 && sc == 1 && s == m && m + 0 >= 1 && m + 0 <= 16); print start, (ok ? "ok" : "BAD"), "memory=" m "g swap=" s "g" } start = 0; mut = 0; m = ""; s = ""; mc = 0; sc = 0 }
    /docker run/ { flush(); start = NR; inhdr = 1 }
    start && inhdr { r = $0
                     while (match(r, /--memory=[0-9]+g/)) { m = substr(r, RSTART + 9, RLENGTH - 10); mc++; r = substr(r, RSTART + RLENGTH) }
                     r = $0
                     while (match(r, /--memory-swap=[0-9]+g/)) { s = substr(r, RSTART + 14, RLENGTH - 15); sc++; r = substr(r, RSTART + RLENGTH) }
                     if ($0 ~ /IMAGE/) inhdr = 0 }
    start && ($0 ~ /cargo mutants / && $0 !~ /--version/ || $0 ~ /mutants_(diff_gate|table_shard)\.sh/) { mut = 1 }
    /^ *- name:/ { flush() }
    END { flush() }' "$1"
}
check() { # check <file...> -> 0 iff every file is readable, holds a mutation run, and all runs are capped
    local f n bad=0 line out
    for f in "$@"; do
        # A file the guard cannot read, or one with no mutation run, was not judged: never a pass.
        [ -f "$f" ] && [ -r "$f" ] || { printf 'ENV   %s is missing or unreadable -- cannot judge, not a pass\n' "$f" >&2; return 2; }
        out=$(runs "$f") || { printf 'ENV   cannot parse %s -- cannot judge, not a pass\n' "$f" >&2; return 2; }
        n=0
        while read -r line; do
            [ -n "$line" ] || continue
            n=$((n + 1))
            case "$line" in *" ok "*) printf 'ok    %s:%s\n' "$f" "$line" ;; *) printf 'FAIL  %s:%s\n' "$f" "$line" >&2; bad=1 ;; esac
        done <<< "$out"
        [ "$n" -gt 0 ] || { printf 'ENV   no mutation docker run found in: %s -- cannot judge, not a pass\n' "$f" >&2; return 2; }
    done
    return "$bad"
}

if [ "${1:-}" = "--self-test" ]; then
    d=$(mktemp -d "${TMPDIR:-/tmp}/mutcap.XXXXXX") || exit 2
    trap 'rm -f "${d:?}"/*.yml; rmdir "${d:?}"' EXIT
    bad=0
    mk() { # mk <name> <docker header flags>
        printf '    steps:\n      - name: m\n        run: |\n          docker run --rm \\\n            %s \\\n            "$IMAGE" \\\n            bash -c '"'"'cargo mutants --in-diff x'"'"'\n' "$2" > "$d/$1.yml"
    }
    row() { # row <want rc> <name> <flags>
        mk "$2" "$3"; local rc=0; check "$d/$2.yml" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = "$1" ]; then printf 'ok    %-14s rc=%s\n' "$2" "$rc"; else printf 'FAIL  %-14s wanted rc=%s got %s\n' "$2" "$1" "$rc"; bad=1; fi
    }
    row 0 capped   '--memory=16g --memory-swap=16g'
    row 0 smaller  '--memory=8g --memory-swap=8g'
    row 1 nocap    '-v /a:/b'
    row 1 nomemory '--memory-swap=16g'
    row 1 noswap   '--memory=16g'
    row 1 swapmore '--memory=16g --memory-swap=32g'
    row 1 toobig   '--memory=64g --memory-swap=64g'
    row 1 zero     '--memory=0g --memory-swap=0g'
    row 1 dupflag  '--memory=16g --memory-swap=16g --memory=128g --memory-swap=128g'
    printf 'x: 1\n' > "$d/none.yml"; rc=0; check "$d/none.yml" > /dev/null 2>&1 || rc=$?
    [ "$rc" = 2 ] && echo "ok    no mutation run is ENV rc=2" || { echo "FAIL  no run gave rc=$rc"; bad=1; }
    # Every listed file is judged on its own: a missing file, or one with no run, is never carried by another.
    mrow() { # mrow <want rc> <name> <file...>
        local want=$1 name=$2 rc=0; shift 2; check "$@" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    %-14s rc=%s\n' "$name" "$rc"; else printf 'FAIL  %-14s wanted rc=%s got %s\n' "$name" "$want" "$rc"; bad=1; fi
    }
    mrow 2 missing     "$d/capped.yml" "$d/gone.yml"
    mrow 2 missingonly "$d/gone.yml"
    mrow 2 onerunless  "$d/capped.yml" "$d/none.yml"
    mrow 0 twocapped   "$d/capped.yml" "$d/smaller.yml"
    # The per-PR shard runs cargo-mutants inside scripts/mutants_table_shard.sh (ci.yml mutants-shard).
    printf '    steps:\n      - name: s\n        run: |\n          docker run --rm \\\n            %s \\\n            -w /workspace "$IMAGE" \\\n            bash -c '"'"'bash scripts/mutants_table_shard.sh pr.diff 1 16 2 out'"'"'\n' '-v /a:/b' > "$d/shard.yml"
    mrow 1 shardnocap  "$d/shard.yml"
    sed -i 's#-v /a:/b#--memory=16g --memory-swap=16g#' "$d/shard.yml"
    mrow 0 shardcapped "$d/shard.yml"
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi
if [ "$#" -gt 0 ]; then check "$@"; else check "$ROOT/.github/workflows/mutants-nightly.yml" "$ROOT/ci/sections.yml" "$ROOT/.github/workflows/ci.yml"; fi
