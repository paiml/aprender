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
# THE BASELINE IS A RATCHET. scripts/hosted_runner_baseline.txt names `file:job` entries
# still allowed to run hosted while their migration is in flight. A hit outside it is RED,
# and so is an entry that no longer matches any hit (stale), so the list can only shrink.
#
#   bash scripts/check_no_hosted_runners.sh               # scan .github/workflows
#   bash scripts/check_no_hosted_runners.sh --self-test   # the case table (both polarities)
# Exit: 0 clean · 1 violation or stale baseline · 2 ENV (no workflows to read).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_no_hosted_runners

usage() { printf 'usage: %s [--dir <workflows dir>] [--baseline <file>] | --self-test | --help\n' "$PROG"; }

scan() { # scan <workflows dir> <baseline file> -> prints violations, rc 0/1/2
    python3 - "$1" "$2" <<'PY'
import glob, os, re, sys
d, base = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(d, "*.yml")) + glob.glob(os.path.join(d, "*.yaml")))
if not files:
    print(f"ENV: no workflow files under {d}: a guard with nothing to read is exit 2, never a pass")
    sys.exit(2)
tok = re.compile(r"\b(?:ubuntu|macos|windows)-(?:latest|\d+(?:\.\d+)*(?:-arm)?)\b")
allowed = set()
if os.path.exists(base):
    for l in open(base, encoding="utf-8"):
        l = l.split("#", 1)[0].strip()
        if l:
            allowed.add(l)
hits, used = [], set()
for f in files:
    job = "<top>"
    for n, raw in enumerate(open(f, encoding="utf-8"), 1):
        m = re.match(r"^  ([A-Za-z0-9_-]+):\s*$", raw)
        if m:
            job = m.group(1)
        line = raw.split("#", 1)[0]
        if not tok.search(line):
            continue
        key = f"{os.path.basename(f)}:{job}"
        if key in allowed:
            used.add(key)
            continue
        hits.append(f"{os.path.relpath(f)}:{n}: job `{job}` names a GitHub-hosted image: {line.strip()}")
stale = sorted(allowed - used)
for h in hits:
    print(f"VIOLATION {h}")
for s in stale:
    print(f"STALE baseline entry `{s}` matches no hosted line any more: delete it from {base}")
if hits or stale:
    sys.exit(1)
print(f"OK: {len(files)} workflow files, no GitHub-hosted image outside {len(allowed)} baseline entr{'y' if len(allowed) == 1 else 'ies'}")
PY
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
    printf '%s/%s rows\n' "$((n - red))" "$n"; [ "$red" = 0 ] || exit 1; exit 0
fi

DIR="$ROOT/.github/workflows"; BASE="$ROOT/scripts/hosted_runner_baseline.txt"
while [ $# -gt 0 ]; do case "$1" in --dir) DIR=$2; shift 2 ;; --baseline) BASE=$2; shift 2 ;; *) usage >&2; exit 2 ;; esac; done
scan "$DIR" "$BASE"
