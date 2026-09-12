#!/usr/bin/env bash
# check_release_assets.sh — a tagged release carries the SIXTEEN assets it owes,
# or this exits non-zero (row 67-A1, PMAT-1098, issue #3082).
#
# WHY THIS EXISTS
# ---------------
# v0.66.0 shipped eight `pv` tarballs and ZERO `apr` binaries, and nothing was red.
# `verify-cuda-assets` in binary-release.yml asked about the two CUDA assets only,
# inline, in a loop that lived in the workflow — so it could not be run on release
# day by a human, by the dogfood, or by the release-criteria table. No gate asserted
# the ASSET SET. The operator's rule (2026-09-10) is four `apr` binaries on every
# tag: {cuda, cpu} x {x86_64, aarch64}, each with a .sha256.
#
# This script is that assertion, in ONE place, so the workflow and the release-day
# protocol share one checker.
#
#   bash scripts/check_release_assets.sh v0.67.0            # 0 complete · 1 missing · 2 ENV
#   bash scripts/check_release_assets.sh v0.67.0 --assets-from list.txt   # offline seam
#   bash scripts/check_release_assets.sh --selftest         # case table, no network
#
# EXIT CODES — 2 IS NEVER A PASS
#   0  every expected asset is on the release
#   1  at least one is MISSING (each one named)
#   2  ENV: the tag/release could not be read at all (no gh, no token, no network,
#      unreadable fixture). A checker that cannot read the release must not report
#      `ok` — that is the failure mode this whole file exists to refuse.
#
# HOW THE RELEASE IS READ, in order:
#   1. --assets-from FILE (or RELEASE_ASSETS_FIXTURE=FILE) — one asset name per
#      line. The offline seam: the self-test, the release-criteria row and the
#      dogfood row all use it, so none of them needs the network to prove polarity.
#   2. `gh release view <tag> --json assets`, when gh is on PATH.
#   3. The REST API with GITHUB_TOKEN/GH_TOKEN via curl + python3 — the fleet
#      boxes (gx10, yoga) have NO `gh`: run 34448908554 died on
#      `gh: command not found` after a 40-minute GPU build.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROG=check_release_assets

usage() {
    cat <<'USAGE'
usage: check_release_assets.sh <tag> [--assets-from FILE]
       check_release_assets.sh --selftest
       check_release_assets.sh --list <tag>
  exit 0 = every expected asset present · 1 = one or more MISSING · 2 = ENV (unreadable)
USAGE
}

# expected_assets TAG — the sixteen names, derived from the two matrices, printed
# one per line. The apr set is the operator's hard requirement; the pv set is what
# the `build` lane of binary-release.yml has always produced.
expected_assets() {
    local tag="$1" arch flavour libc
    for arch in x86_64 aarch64; do
        for flavour in cuda cpu; do
            printf 'apr-%s-%s-unknown-linux-gnu-%s.tar.gz\n' "$tag" "$arch" "$flavour"
            printf 'apr-%s-%s-unknown-linux-gnu-%s.tar.gz.sha256\n' "$tag" "$arch" "$flavour"
        done
    done
    for arch in x86_64 aarch64; do
        for libc in musl gnu; do
            printf 'pv-%s-%s-unknown-linux-%s.tar.gz\n' "$tag" "$arch" "$libc"
            printf 'pv-%s-%s-unknown-linux-%s.tar.gz.sha256\n' "$tag" "$arch" "$libc"
        done
    done
}

# read_assets TAG — the asset names actually on the release, one per line.
# Returns 2 when the release cannot be read at all. Never invents an empty list:
# an empty stdout with rc=0 means "the release exists and carries nothing", which
# is a MISSING verdict (1), not an ENV one.
read_assets() {
    local tag="$1" json rc
    if [ -n "$ASSETS_FROM" ]; then
        [ -r "$ASSETS_FROM" ] || {
            printf '%s: ENV — asset list %s is not readable\n' "$PROG" "$ASSETS_FROM" >&2
            return 2
        }
        grep -vE '^[[:space:]]*(#|$)' "$ASSETS_FROM" || true
        return 0
    fi
    if command -v gh > /dev/null 2>&1; then
        json=$(gh release view "$tag" --json assets 2>/dev/null); rc=$?
        if [ "$rc" -eq 0 ] && [ -n "$json" ]; then
            printf '%s' "$json" | python3 -c 'import json,sys; print("\n".join(a["name"] for a in json.load(sys.stdin)["assets"]))' 2>/dev/null && return 0
        fi
        printf '%s: gh could not read release %s — falling back to the REST API\n' "$PROG" "$tag" >&2
    fi
    local token="${GITHUB_TOKEN:-${GH_TOKEN:-}}"
    local repo="${GITHUB_REPOSITORY:-paiml/aprender}"
    [ -n "$token" ] || {
        printf '%s: ENV — no gh and no GITHUB_TOKEN/GH_TOKEN; the release cannot be read (this is NOT a pass)\n' "$PROG" >&2
        return 2
    }
    json=$(curl -sSf -H "Authorization: Bearer $token" -H "Accept: application/vnd.github+json" \
        "https://api.github.com/repos/${repo}/releases/tags/${tag}" 2>/dev/null); rc=$?
    [ "$rc" -eq 0 ] && [ -n "$json" ] || {
        printf '%s: ENV — REST read of %s release %s failed (rc=%s)\n' "$PROG" "$repo" "$tag" "$rc" >&2
        return 2
    }
    printf '%s' "$json" | python3 -c 'import json,sys; print("\n".join(a["name"] for a in json.load(sys.stdin)["assets"]))' 2>/dev/null || {
        printf '%s: ENV — the release payload for %s did not parse as JSON assets\n' "$PROG" "$tag" >&2
        return 2
    }
    return 0
}

check_tag() { # check_tag TAG -> 0 complete · 1 missing · 2 ENV
    local tag="$1" have rc=0 miss=0 want
    have=$(read_assets "$tag") || return 2
    while IFS= read -r want; do
        [ -n "$want" ] || continue
        if printf '%s\n' "$have" | grep -qxF "$want"; then
            printf 'ok      %s\n' "$want"
        else
            printf 'MISSING %s\n' "$want"
            miss=$((miss + 1))
            rc=1
        fi
    done <<< "$(expected_assets "$tag")"
    if [ "$rc" -eq 0 ]; then
        printf '%s: %s carries all 16 expected assets (4 apr + 4 sha256 + 8 pv)\n' "$PROG" "$tag"
    else
        printf '%s: %s is MISSING %s expected asset(s) — a release without its four apr binaries is not done (operator rule 2026-09-10)\n' "$PROG" "$tag" "$miss" >&2
    fi
    return "$rc"
}

selftest() {
    local work n=0 red=0 tag=v9.9.9 out
    work=$(mktemp -d) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf '$work'" RETURN
    expected_assets "$tag" > "$work/complete.txt"
    grep -vx "apr-$tag-aarch64-unknown-linux-gnu-cpu.tar.gz" "$work/complete.txt" > "$work/mutant.txt"
    grep -vx "apr-$tag-x86_64-unknown-linux-gnu-cuda.tar.gz.sha256" "$work/complete.txt" > "$work/nosha.txt"
    grep -v '^pv-' "$work/complete.txt" > "$work/nopv.txt"
    : > "$work/empty.txt"

    row() { # row <want-rc> <label> <cmd...>
        local want=$1 label=$2; shift 2
        local rc=0
        n=$((n + 1))
        "$@" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then
            printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else
            printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"
            red=1
        fi
    }

    row 0 "the complete asset set is credited"                bash "$0" "$tag" --assets-from "$work/complete.txt"
    row 1 "MUTATION: apr-<tag>-aarch64-...-cpu.tar.gz removed" bash "$0" "$tag" --assets-from "$work/mutant.txt"
    row 1 "a missing .sha256 is as fatal as a missing tarball" bash "$0" "$tag" --assets-from "$work/nosha.txt"
    row 1 "the eight pv assets are required too"               bash "$0" "$tag" --assets-from "$work/nopv.txt"
    row 1 "an empty release is MISSING (1), not ENV"           bash "$0" "$tag" --assets-from "$work/empty.txt"
    row 2 "an unreadable asset list is ENV (2), never a pass"  bash "$0" "$tag" --assets-from "$work/nope.txt"
    row 2 "no tag at all is a usage error (2)"                 bash "$0"
    row 0 "RELEASE_ASSETS_FIXTURE is the same seam as --assets-from" \
        env RELEASE_ASSETS_FIXTURE="$work/complete.txt" bash "$0" "$tag"
    row 1 "RELEASE_ASSETS_FIXTURE carries the mutation too" \
        env RELEASE_ASSETS_FIXTURE="$work/mutant.txt" bash "$0" "$tag"
    # The verdict must NAME the asset: a count is not an instruction.
    out=$(bash "$0" "$tag" --assets-from "$work/mutant.txt" 2>&1)
    row 0 "the missing asset is named in the output" \
        grep -q "MISSING apr-$tag-aarch64-unknown-linux-gnu-cpu.tar.gz" <<< "$out"
    # Sixteen, not "some": a table that expected four would pass the rows above.
    row 0 "sixteen assets are expected, and four of them are apr tarballs" \
        bash -c "[ \$(bash '$0' --list '$tag' | grep -c .) -eq 16 ] && [ \$(bash '$0' --list '$tag' | grep -c '^apr-.*tar.gz\$') -eq 4 ]"

    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || return 1
    return 0
}

cd "$ROOT" || exit 2
ASSETS_FROM="${RELEASE_ASSETS_FIXTURE:-}"
TAG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --selftest|--self-test) selftest; exit $? ;;
        --list)
            [ $# -ge 2 ] || { usage >&2; exit 2; }
            expected_assets "$2"; exit 0 ;;
        --assets-from)
            [ $# -ge 2 ] || { usage >&2; exit 2; }
            ASSETS_FROM="$2"; shift 2; continue ;;
        --assets-from=*) ASSETS_FROM="${1#--assets-from=}"; shift; continue ;;
        -h|--help) usage; exit 0 ;;
        -*) printf '%s: unknown option %s\n' "$PROG" "$1" >&2; usage >&2; exit 2 ;;
        *) TAG="$1"; shift; continue ;;
    esac
done
[ -n "$TAG" ] || { printf '%s: no tag given — a checker with no release to read is exit 2, never a pass\n' "$PROG" >&2; usage >&2; exit 2; }
check_tag "$TAG"
exit $?
