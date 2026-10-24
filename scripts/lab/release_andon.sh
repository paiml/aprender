#!/usr/bin/env bash
# release_andon.sh — FLOW-008 A7/F7 andon printer for this repo. LAB: it cannot block a merge or a release,
# so it is wired into a nightly LAB workflow only, never a PR, merge-queue or release job.
#
# Rule printed (A7, first clause): publishable change older than MAX_HOURS (72) with no release since ⇒ ONE line,
# exit 1. No publishable change since the last release is IDLE-OK (exit 0): an idle repo never raises it.
# Not measured here: the second clause (two consecutive non-idle skipped slots) needs a slot ledger that does not
# exist yet; this script says so on every run instead of passing silently.
#
# "Last release" is the newest v[0-9]* tag by creator date, NOT by ancestry: v0.71.4 was cut on a release branch and
# is not an ancestor of main. "Publishable" = crates/, src/ and Cargo.toml outside Markdown.
#
# Usage: scripts/lab/release_andon.sh [--ref main] [--now EPOCH] [--max-hours 72] | --self-test
# Exit: 0 OK or IDLE-OK, 1 ANDON, 2 not_measured (no release tag, unknown ref).
set -euo pipefail

REF=main NOW=$(date +%s) MAX_HOURS=72
PUBLISHABLE=('crates/' 'src/' 'Cargo.toml' ':(exclude)*.md')

print_andon() {
    local last tag oldest sha age
    git rev-parse --verify -q "$REF^{commit}" > /dev/null || { echo "NOT_MEASURED F7: ref $REF not found"; return 2; }
    tag=$(git for-each-ref --sort=-creatordate --count=1 --format='%(refname:short)' 'refs/tags/v[0-9]*')
    [ -n "$tag" ] || { echo "NOT_MEASURED F7: no v* release tag"; return 2; }
    last=$(git for-each-ref --sort=-creatordate --count=1 --format='%(creatordate:unix)' 'refs/tags/v[0-9]*')
    # oldest publishable commit on REF newer than the release tag
    read -r oldest sha < <(git log "$REF" --format='%ct %h' --reverse -- "${PUBLISHABLE[@]}" | awk -v l="$last" '$1 > l {print; exit}') || true
    echo "NOT_MEASURED F7 clause 2: two consecutive non-idle skipped slots (no slot ledger yet)"
    if [ -z "${oldest:-}" ]; then echo "IDLE-OK F7: no publishable change on $REF since $tag"; return 0; fi
    age=$(((NOW - oldest) / 3600))
    if [ "$age" -gt "$MAX_HOURS" ]; then
        echo "ANDON F7: publishable change ${age}h old on $REF (first $sha) with no release since $tag; limit ${MAX_HOURS}h"
        return 1
    fi
    echo "OK F7: oldest unreleased publishable change is ${age}h old (limit ${MAX_HOURS}h) since $tag"
}

self_test() {
    local t f=0 n=0 script
    t=$(mktemp -d); script=$(readlink -f "${BASH_SOURCE[0]}")
    trap 'rm -rf -- "${t:?}"' RETURN
    local H=3600 base=1700000000
    g() { git -C "$t/r" -c user.name=t -c user.email=t@t "$@"; }
    commit() { # FILE HOURS-AFTER-BASE
        mkdir -p "$t/r/$(dirname "$1")"; echo "$RANDOM" >> "$t/r/$1"
        g add -A; GIT_AUTHOR_DATE="@$((base + $2 * H)) +0000" GIT_COMMITTER_DATE="@$((base + $2 * H)) +0000" g commit -qm "$1"
    }
    tagat() { GIT_COMMITTER_DATE="@$((base + $2 * H)) +0000" g tag -a "$1" -m "$1"; }
    run() { # NOW-HOURS [script]
        (cd "$t/r" && bash "${2:-$script}" --ref main --now "$((base + $1 * H))" 2>&1)
    }
    expect() { # NAME WANT-RC WANT-TEXT -- OUTPUT-VAR RC
        n=$((n + 1))
        if [ "$4" = "$2" ] && printf '%s' "$5" | grep -q -- "$3"; then echo "ok   $1"; else echo "BROKE $1 (rc=$4 want $2, text '$3')"; printf '%s\n' "$5" | sed 's/^/     | /'; f=$((f + 1)); fi
    }
    git init -q -b main "$t/r"
    local out rc
    commit docs/a.md 0
    out=$(run 1) && rc=0 || rc=$?; expect "no release tag is not_measured" 2 NOT_MEASURED "$rc" "$out"
    tagat v1.0.0 1
    out=$(run 500) && rc=0 || rc=$?; expect "tag and no publishable change is IDLE-OK however old" 0 IDLE-OK "$rc" "$out"
    commit docs/b.md 2; commit crates/x/NOTES.md 3
    out=$(run 500) && rc=0 || rc=$?; expect "Markdown under crates/ is not publishable" 0 IDLE-OK "$rc" "$out"
    commit crates/x/src/lib.rs 10
    out=$(run 20) && rc=0 || rc=$?; expect "publishable change 10h old is OK" 0 '^OK F7' "$rc" "$out"
    out=$(run 82) && rc=0 || rc=$?; expect "publishable change exactly 72h old is OK (limit is strict)" 0 '^OK F7' "$rc" "$out"
    out=$(run 83) && rc=0 || rc=$?; expect "publishable change 73h old raises the andon" 1 '^ANDON F7' "$rc" "$out"
    commit crates/x/src/more.rs 80
    out=$(run 83) && rc=0 || rc=$?; expect "age is the OLDEST unreleased change, not the newest" 1 '^ANDON F7' "$rc" "$out"
    out=$(run 83) && rc=0 || rc=$?; expect "clause 2 is reported not_measured on every run" 1 'NOT_MEASURED F7 clause 2' "$rc" "$out"
    tagat v1.0.1 90
    out=$(run 500) && rc=0 || rc=$?; expect "a newer release tag clears the andon" 0 IDLE-OK "$rc" "$out"
    out=$(cd "$t/r" && bash "$script" --ref nope --now 1 2>&1) && rc=0 || rc=$?; expect "unknown ref is not_measured" 2 NOT_MEASURED "$rc" "$out"
    # mutants: the copy must go RED against the same rows (SELFTEST_IN_MUTANT stops recursion)
    if [ -z "${SELFTEST_IN_MUTANT:-}" ]; then
        mutant() { # NAME SED-EXPR
            sed "$2" "$script" > "$t/m.sh"
            if cmp -s "$script" "$t/m.sh"; then n=$((n + 1)); echo "BROKE mutant $1 changed nothing"; f=$((f + 1)); return; fi
            n=$((n + 1))
            if SELFTEST_IN_MUTANT=1 bash "$t/m.sh" --self-test > /dev/null 2>&1; then echo "BROKE mutant $1 survived"; f=$((f + 1)); else echo "ok   mutant $1 killed"; fi
        }
        mutant threshold-off 's/-gt "\$MAX_HOURS"/-gt 100000/'
        mutant counts-markdown "s/ ':(exclude)\*.md'//"
        mutant newest-not-oldest 's/--reverse//'
        mutant tag-by-nothing "s/\$1 > l/1/"
    fi
    echo "release_andon self-test: $n rows, $f broke"
    [ "$f" -eq 0 ]
}

while [ $# -gt 0 ]; do
    case "$1" in
        --self-test) self_test; exit $? ;;
        --ref) REF=$2; shift 2 ;;
        --now) NOW=$2; shift 2 ;;
        --max-hours) MAX_HOURS=$2; shift 2 ;;
        *) echo "usage: $0 [--ref main] [--now EPOCH] [--max-hours 72] | --self-test" >&2; exit 64 ;;
    esac
done
print_andon
