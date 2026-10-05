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
#   check_mutants_memory_cap.sh [file...]  default: mutants-nightly.yml and ci/sections.yml
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
    start && ($0 ~ /cargo mutants / && $0 !~ /--version/ || $0 ~ /mutants_diff_gate\.sh/) { mut = 1 }
    /^ *- name:/ { flush() }
    END { flush() }' "$1"
}
check() { # check <file...> -> 0 iff at least one mutation run exists across the files and all are capped
    local f n=0 bad=0 line
    for f in "$@"; do
        while read -r line; do
            [ -n "$line" ] || continue
            n=$((n + 1))
            case "$line" in *" ok "*) printf 'ok    %s:%s\n' "$f" "$line" ;; *) printf 'FAIL  %s:%s\n' "$f" "$line" >&2; bad=1 ;; esac
        done < <(runs "$f")
    done
    [ "$n" -gt 0 ] || { printf 'ENV   no mutation docker run found in: %s -- cannot judge, not a pass\n' "$*" >&2; return 2; }
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
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi
if [ "$#" -gt 0 ]; then check "$@"; else check "$ROOT/.github/workflows/mutants-nightly.yml" "$ROOT/ci/sections.yml"; fi
