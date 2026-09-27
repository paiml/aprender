#!/usr/bin/env bash
# check_no_hosted_runners.sh — no GitHub-hosted runner in any workflow (#3073).
#
# Operator rule (2026-09-10, verbatim): "WE DO NOT USE HOSTED GITHUB RUNNERS. WE USE GX10
# AND LAMBDA-LABS OR YOGA". scripts/check_runner_labels.sh pins a discriminating label on
# every SELF-HOSTED job and, by its own header, exempts hosted ones; nothing refused a
# hosted image. This guard does.
#
# WHAT IT READS. Every non-comment line of .github/workflows/*.yml, not only `runs-on:`
# lines. nightly.yml hid `ubuntu-latest` behind `runs-on: ${{ matrix.runner }}` in a
# matrix value, where a runs-on-only scan cannot see it. A hosted image token is
# (ubuntu|macos|windows)-(latest|<version>[-arm]); `ubuntu:22.04` (a docker image, colon)
# is not one.
#
# THE BASELINE IS A RATCHET, AND IT IS ABSENT. scripts/hosted_runner_baseline.txt, when present,
# names `file:job` entries still allowed to run hosted while a migration is in flight; a hit
# outside it is RED, and so is an entry that no longer matches any hit (stale). Absent means no
# exemption at all, which is the state since every workflow moved to the fleet (#3073). A file
# added later must also be classified in check_baseline_ratchets.sh, or that guard is RED.
#
#   bash scripts/check_no_hosted_runners.sh               # scan .github/workflows
#   bash scripts/check_no_hosted_runners.sh --self-test   # the case table (both polarities)
# Exit: 0 clean · 1 violation or stale baseline · 2 ENV (no workflows to read).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_no_hosted_runners

usage() { printf 'usage: %s [--dir <workflows dir>] [--baseline <file>] | --self-test | --help\n' "$PROG"; }

scan() { # scan <workflows dir> <baseline file> -> prints violations, rc 0/1/2
    # awk, not python (#3697): the fleet is python-free for automation (infra#708), and this is a
    # line scan, so it needs no interpreter. POSIX ERE has no \b, so the token's word boundaries
    # are spelled out: an ASCII word character may not touch it on either side.
    local d=$1 base=$2 f files=()
    for f in "$d"/*.yml "$d"/*.yaml; do [ -f "$f" ] && files+=("$f"); done
    if [ "${#files[@]}" -eq 0 ]; then
        printf 'ENV: no workflow files under %s: a guard with nothing to read is exit 2, never a pass\n' "$d"
        return 2
    fi
    mapfile -t files < <(printf '%s\n' "${files[@]}" | LC_ALL=C sort)
    LC_ALL=C awk -v base="$base" -v nfiles="${#files[@]}" -v pwd="$PWD" '
        BEGIN {
            tok = "(^|[^A-Za-z0-9_])(ubuntu|macos|windows)-(latest|[0-9]+(\\.[0-9]+)*(-arm)?)([^A-Za-z0-9_]|$)"
            while ((getline l < base) > 0) {
                sub(/#.*/, "", l); gsub(/^[[:space:]]+|[[:space:]]+$/, "", l)
                if (l != "") allowed[l] = 1
            }
        }
        FNR == 1 {
            job = "<top>"; bn = FILENAME; sub(/.*\//, "", bn)
            rel = FILENAME; if (index(rel, pwd "/") == 1) rel = substr(rel, length(pwd) + 2)
        }
        /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { job = $0; sub(/^  /, "", job); sub(/:[[:space:]]*$/, "", job) }
        {
            line = $0; sub(/#.*/, "", line)
            if (line !~ tok) next
            key = bn ":" job
            if (key in allowed) { used[key] = 1; next }
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)
            hits[++nh] = rel ":" FNR ": job `" job "` names a GitHub-hosted image: " line
        }
        END {
            for (i = 1; i <= nh; i++) print "VIOLATION " hits[i]
            na = 0; ns = 0
            for (k in allowed) { na++; if (!(k in used)) stale[++ns] = k }
            for (i = 2; i <= ns; i++) {          # sorted, as the report always was
                k = stale[i]
                for (j = i - 1; j >= 1 && stale[j] > k; j--) stale[j + 1] = stale[j]
                stale[j + 1] = k
            }
            for (i = 1; i <= ns; i++) print "STALE baseline entry `" stale[i] "` matches no hosted line any more: delete it from " base
            if (nh || ns) exit 1
            printf "OK: %d workflow files, no GitHub-hosted image outside %d baseline entr%s\n", nfiles, na, (na == 1 ? "y" : "ies")
        }' "${files[@]}"
}

if [ "${1:-}" = "--help" ]; then usage; printf 'modes: default scan, --self-test (case table)\n'; exit 0; fi

if [ "${1:-}" = "--self-test" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/nohosted.XXXXXX")
    if [ -z "$TD" ] || [ "$TD" = "/" ]; then printf '%s: refusing temp dir "%s"\n' "$PROG" "$TD" >&2; exit 2; fi
    trap 'rm -rf -- "$TD"' EXIT
    n=0; red=0
    # row <want rc> <label> <workflow body> [baseline body]
    row() {
        local want=$1 label=$2 body=$3 bl=${4:-} rc=0 out
        n=$((n + 1)); local w="$TD/w$n" b="$TD/bl$n"; mkdir -p "$w"
        [ -n "$body" ] && printf '%b' "$body" > "$w/ci.yml"
        printf '%b' "$bl" > "$b"
        out=$(scan "$w" "$b" 2>&1) || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else red=$((red + 1)); printf 'BROKE row %-2s rc=%s want=%s  %s | %s\n' "$n" "$rc" "$want" "$label" "$(printf '%s' "$out" | tail -1)"; fi
    }
    J='jobs:\n  build:\n'
    row 1 "inline runs-on: ubuntu-latest is RED"                 "${J}    runs-on: ubuntu-latest\n"
    row 1 "a versioned image in a list is RED"                   "${J}    runs-on: [ubuntu-24.04]\n"
    row 1 "the arm image is RED"                                 "${J}    runs-on: ubuntu-24.04-arm\n"
    row 1 "a hosted image hidden in a matrix value is RED"       "${J}    strategy:\n      matrix:\n        include:\n          - runner: macos-latest\n    runs-on: \${{ matrix.runner }}\n"
    row 1 "a hosted image in JSON matrix labels is RED"          "${J}    strategy:\n      matrix:\n        include:\n          - labels: '[\"windows-latest\"]'\n    runs-on: \${{ fromJSON(matrix.labels) }}\n"
    row 0 "a self-hosted clean-room job is GREEN"                "${J}    runs-on: [self-hosted, Linux, X64, clean-room]\n"
    row 0 "a comment that names ubuntu-latest is GREEN"          "${J}    # was: runs-on: ubuntu-latest\n    runs-on: [self-hosted, Linux, X64, clean-room]\n"
    row 0 "a docker image ubuntu:22.04 (colon) is GREEN"         "${J}    runs-on: [self-hosted, Linux, X64, clean-room]\n    container: ubuntu:22.04\n"
    row 0 "a hosted job named in the baseline is GREEN"          "${J}    runs-on: ubuntu-latest\n" "ci.yml:build\n"
    row 1 "a baseline entry for ANOTHER job does not excuse it"  "${J}    runs-on: ubuntu-latest\n" "ci.yml:deploy\n"
    row 1 "a stale baseline entry (no hosted line left) is RED"  "${J}    runs-on: [self-hosted, Linux, X64, clean-room]\n" "ci.yml:build\n"
    row 2 "no workflow files at all is ENV, never a pass"        ""
    # the token's word boundaries, spelled out for ERE (#3697): an ASCII word character touching it
    # makes it another word; any other byte (punctuation, a non-ASCII letter) does not
    row 0 "a word character BEFORE the token is GREEN (xubuntu-latest is not an image)"   "${J}    runs-on: [self-hosted, xubuntu-latest]\n"
    row 0 "a word character AFTER the token is GREEN (ubuntu-latest_2 is not an image)"   "${J}    runs-on: [self-hosted, ubuntu-latest_2]\n"
    row 1 "a non-ASCII letter touching the token is RED (the boundary is ASCII)"           "${J}    runs-on: ubuntu-latest\xc3\xa9\n"
    printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi

DIR="$ROOT/.github/workflows"; BASE="$ROOT/scripts/hosted_runner_baseline.txt"
while [ $# -gt 0 ]; do case "$1" in --dir) DIR=$2; shift 2 ;; --baseline) BASE=$2; shift 2 ;; *) usage >&2; exit 2 ;; esac; done
scan "$DIR" "$BASE"
