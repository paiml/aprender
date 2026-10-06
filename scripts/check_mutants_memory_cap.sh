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
# A native run (the mutants-cuda section) is capped instead by `systemd-run --scope -p MemoryMax=<N>G
# -p MemorySwapMax=0` on one line, 1 <= N <= 16, each once. A mutation run with neither cap is a FAIL.
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
    function flush() { if (start && mut) { if (kind == "scope") ok = (m != "" && mc == 1 && sc == 1 && s == "0" && m + 0 >= 1 && m + 0 <= 16)
                                           else ok = (m != "" && mc == 1 && sc == 1 && s == m && m + 0 >= 1 && m + 0 <= 16)
                                           print start, (ok ? "ok" : "BAD"), kind " memory=" m "g swap=" s (kind == "scope" ? "" : "g") }
                       start = 0; mut = 0; m = ""; s = ""; mc = 0; sc = 0; kind = ""; inq = 0 }
    /^[[:space:]]*#/ { next }
    /docker run/ { flush(); start = NR; inhdr = 1; kind = "docker" }
    /systemd-run .*--scope/ { flush(); start = NR; inhdr = 0; kind = "scope"; r = $0
                     while (match(r, /-p MemoryMax=[0-9]+G/)) { m = substr(r, RSTART + 13, RLENGTH - 14); mc++; r = substr(r, RSTART + RLENGTH) }
                     r = $0
                     while (match(r, /-p MemorySwapMax=[0-9]+/)) { s = substr(r, RSTART + 17, RLENGTH - 17); sc++; r = substr(r, RSTART + RLENGTH) } }
    start && inhdr { r = $0
                     while (match(r, /--memory=[0-9]+g/)) { m = substr(r, RSTART + 9, RLENGTH - 10); mc++; r = substr(r, RSTART + RLENGTH) }
                     r = $0
                     while (match(r, /--memory-swap=[0-9]+g/)) { s = substr(r, RSTART + 14, RLENGTH - 15); sc++; r = substr(r, RSTART + RLENGTH) }
                     if ($0 ~ /IMAGE/) inhdr = 0 }
    ($0 ~ /cargo mutants / && $0 !~ /--version/ || $0 ~ /mutants_(diff_gate|table_shard)\.sh/ && $0 !~ /mutants_[a-z_]+\.sh --(exempt-ref|self-test)/) {
                     if (start) mut = 1; else print NR, "BAD", "native: no capped docker run or systemd-run scope" }
    # A scope caps only its own command: it ends at the first line outside a single-quoted span that has no
    # trailing backslash. A mutation run on a later line of the step is then native, not capped by it.
    # Double-quoted strings are dropped first, so a single-quote character inside one opens no span.
    start && kind == "scope" { r = $0; gsub(/"[^"]*"/, "", r); inq = (inq + gsub("\047", "", r)) % 2
                     if (!inq && $0 !~ /\\[[:space:]]*$/) flush() }
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

# The files a bare run judges. The self-test proves every workflow file with a mutation run is listed here.
defaults=("$ROOT/.github/workflows/mutants-nightly.yml" "$ROOT/ci/sections.yml" "$ROOT/.github/workflows/ci.yml")
uncovered() { # uncovered <listed file...> -> each workflow file holding a mutation run that is not listed
    local f l hit
    for f in "$ROOT"/.github/workflows/*.yml "$ROOT/ci/sections.yml"; do
        [ -n "$(runs "$f")" ] || continue
        hit=0; for l in "$@"; do [ "$l" = "$f" ] && hit=1; done
        [ "$hit" = 1 ] || printf '%s\n' "$f"
    done
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
    row 1 dupwithin '--memory=8g --memory-swap=8g --memory=16g --memory-swap=16g'   # each value in range: only the once-only clause fails it
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
    # A native run (no container) is capped by a user scope: MemoryMax 1..16G, MemorySwapMax=0, each once.
    sc() { # sc <name> <systemd-run line>
        printf '    steps:\n      - name: c\n        run: |\n          %s bash -c '"'"'exec "$@"'"'"' cap \\\n            bash scripts/mutants_diff_gate.sh pr.diff --jobs 1\n' "$2" > "$d/$1.yml"
    }
    sc scope     'systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0';       mrow 0 scope     "$d/scope.yml"
    sc scopeswap 'systemd-run --user --scope -q -p MemoryMax=16G';                          mrow 1 scopeswap "$d/scopeswap.yml"
    sc scopebig  'systemd-run --user --scope -q -p MemoryMax=64G -p MemorySwapMax=0';       mrow 1 scopebig  "$d/scopebig.yml"
    sc scopezero 'systemd-run --user --scope -q -p MemoryMax=0G -p MemorySwapMax=0';        mrow 1 scopezero "$d/scopezero.yml"
    sc scopedup  'systemd-run --user --scope -q -p MemoryMax=8G -p MemoryMax=16G -p MemorySwapMax=0'; mrow 1 scopedup "$d/scopedup.yml"
    sc scopeswnz 'systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=8';       mrow 1 scopeswnz "$d/scopeswnz.yml"
    sc native    'nice -n 19';                                                              mrow 1 native    "$d/native.yml"
    # A capped scope around another command does not cap a mutation run on a later line of the same step.
    printf '    steps:\n      - name: o\n        run: |\n          systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 echo hi\n          bash scripts/mutants_diff_gate.sh pr.diff --jobs 1\n' > "$d/scopeother.yml"
    mrow 1 scopeother  "$d/scopeother.yml"
    printf '    steps:\n      - name: a\n        run: |\n          systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 echo "it%ss capped"\n          bash scripts/mutants_diff_gate.sh pr.diff --jobs 1\n' "'" > "$d/scopeapos.yml"
    mrow 1 scopeapos   "$d/scopeapos.yml"
    # Query modes and comments are not mutation runs: a file holding only them has no run to judge.
    printf '    steps:\n      - name: q\n        run: |\n          # bash scripts/mutants_diff_gate.sh pr.diff\n          bash scripts/mutants_diff_gate.sh --exempt-ref "$R" a b\n' > "$d/query.yml"
    mrow 2 queryonly   "$d/query.yml"
    # A file that exists but cannot be read is never a pass. Dropping the -r test is an equivalent mutant (awk then
    # fails and runs's rc gives 2); this row pins the outcome. As root every file reads: not measured.
    cp "$d/capped.yml" "$d/unread.yml"; chmod 000 "$d/unread.yml"
    if [ -r "$d/unread.yml" ]; then echo "NOT_MEASURED unreadable row: this user reads mode 000"; nm=1; else mrow 2 unreadable "$d/unread.yml"; fi
    # The bare run's list covers every workflow file that runs mutants; dropping one is caught.
    u=$(uncovered "${defaults[@]}"); [ -z "$u" ] && echo "ok    defaults cover every mutation file" || { echo "FAIL  not in defaults: $u"; bad=1; }
    u=$(uncovered "${defaults[@]:0:2}"); [ -n "$u" ] && echo "ok    a dropped default is caught" || { echo "FAIL  dropping ci.yml went unseen"; bad=1; }
    [ "$bad" = 0 ] && [ "${nm:-0}" = 1 ] && { echo "SELF-TEST NOT MEASURED (1 row)" >&2; exit 2; }
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi
if [ "$#" -gt 0 ]; then check "$@"; else check "${defaults[@]}"; fi
