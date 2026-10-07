#!/usr/bin/env bash
# Case table for scripts/lib/release_policy.sh, the reader of the standing CRUX-smoke release policy
# (`ladder.release_policy`). Each RED row is planted on a copy of the real ladder and asserts its
# reason, so a row cannot pass for the wrong cause. Needs bash and awk only: no yq, no cargo.
# RELEASE_POLICY_LIB points the table at a mutated copy of the library (mutation runs only).
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/.." && pwd)
. "${RELEASE_POLICY_LIB:-$here/lib/release_policy.sh}" || exit 1
ladder="$root/contracts/model-capability-ladder-v1.yaml"
tmp=$(mktemp -d)
export TMPDIR="$tmp"
trap 'rm -rf "${tmp:?}"' EXIT
fail=0 n=0

# row NAME WANT_RC WANT_APPLIES LADDER VERSION PATTERN
#   rc 0: PATTERN must match a line of the effective ladder; rc 1/2: PATTERN must match RP_WHY.
row() {
    local name="$1" want="$2" wapp="$3" lad="$4" ver="$5" pat="$6" out rc
    n=$((n + 1))
    out=$(release_policy_ladder "$lad" "$ver" 2>&1) && rc=0 || rc=$?
    release_policy_ladder "$lad" "$ver" > /dev/null 2>&1 || true # this shell gets RP_APPLIES/RP_WHY
    if [ "$rc" != "$want" ] || [ "$RP_APPLIES" != "$wapp" ]; then
        printf 'FAIL %s: rc %s (want %s), applies %s (want %s), why: %s\n' "$name" "$rc" "$want" "$RP_APPLIES" "$wapp" "$RP_WHY"
        fail=$((fail + 1)); return 0
    fi
    if [ "$rc" = 0 ]; then
        if [ ! -f "$out" ] || ! grep -qE -- "$pat" "$out"; then
            printf 'FAIL %s: effective ladder %s has no line /%s/\n' "$name" "$out" "$pat"
            fail=$((fail + 1)); return 0
        fi
    elif ! printf '%s\n' "$RP_WHY" | grep -qE -- "$pat"; then
        printf 'FAIL %s: reason "%s" is not /%s/\n' "$name" "$RP_WHY" "$pat"
        fail=$((fail + 1)); return 0
    fi
    printf 'ok   %s (rc %s%s)\n' "$name" "$rc" "${RP_WHY:+: $RP_WHY}"
}

plant() { # plant NAME SED_SCRIPT -> a ladder copy with SED_SCRIPT applied; refuses a no-op plant
    local p="$tmp/$1.yaml"
    sed -e "$2" "$ladder" > "$p"
    if cmp -s "$p" "$ladder"; then echo "PLANT $1 changed nothing" >&2; exit 2; fi
    printf '%s\n' "$p"
}

two="$tmp/two.yaml" # a second, complete policy block placed just before emergency_scopes
awk '/^  release_policy:$/ { inb = 1 } inb && /^  [a-z_]+:/ && !/^  release_policy:$/ { inb = 0 }
     inb && !/^ *#/ { blk = blk $0 "\n" } /^  emergency_scopes:$/ { printf "%s", blk } { print }' "$ladder" > "$two"

# The real ladder: a covered version gets a synthesized entry; an earlier one keeps the ladder as-is.
row real-0.71.0           0 1 "$ladder" 0.71.0      '^      release: "0\.71\.0"$'
row real-0.71.0-rc.1      0 1 "$ladder" 0.71.0-rc.1 '^      release: "0\.71\.0-rc\.1"$'
row real-0.72.3-name      0 1 "$ladder" 0.72.3      '^    - name: crux-smoke$'
row real-thinking         0 1 "$ladder" 0.72.3      '^      thinking: \["off", "on"\]$'
row real-quote-verbatim   0 1 "$ladder" 0.71.0      '^      quote: .*"from 0\.71 on, a release ships on CRUX smoke on lambda and gx10 GPU\.'
row real-1.0.0            0 1 "$ladder" 1.0.0       '^      release: "1\.0\.0"$'
row real-0.70.2-uncovered 0 0 "$ladder" 0.70.2      '^  release_policy:$'
row real-0.70.10-numeric  0 0 "$ladder" 0.70.10     '^  release_policy:$'
row real-0.9.99-numeric   0 0 "$ladder" 0.9.99      '^  release_policy:$'
# Unjudgeable input is rc 2 (not measured), never "no policy".
row version-not-semver    2 0 "$ladder" v0.71        'not semver'
row version-empty         2 0 "$ladder" ""           'not semver'
row ladder-missing        2 0 "$tmp/none.yaml" 0.71.0 'cannot read the ladder'
row no-policy-block       0 0 "$(plant nopol '/^  release_policy:/,/^    release_notes:/d')" 0.71.0 '^  emergency_scopes:$'
# One release, one ruling.
row per-release-entry     1 0 "$(plant dup 's/^      release: "0\.70\.1"$/      release: "0.71.0"/')" 0.71.0 'one release takes one ruling'
row per-release-rc-entry  1 0 "$(plant duprc 's/^      release: "0\.70\.1"$/      release: "0.72.0-rc.1"/')" 0.72.0-rc.1 'one release takes one ruling'
# A block the strict reader cannot read is rc 2 with its reason.
row since-not-xyz         2 0 "$(plant since 's/^    since: "0\.71\.0"$/    since: "0.71"/')" 0.71.0 "since '0\.71' is not X\.Y\.Z"
row unknown-key           2 0 "$(plant unk 's/^    larger_rows: nightly$/    larger_rowz: nightly/')" 0.71.0 'unknown key in release_policy: larger_rowz'
row missing-key           2 0 "$(plant miss '/^    red_row_needs: ticket$/d')" 0.71.0 'release_policy has no red_row_needs'
row duplicate-key         2 0 "$(plant dkey 's/^    larger_rows: nightly$/    larger_rows: nightly\n    larger_rows: release/')" 0.71.0 'duplicate key in release_policy: larger_rows'
row empty-value           2 0 "$(plant empty 's/^    hosts: .*$/    hosts: /')" 0.71.0 'empty value for release_policy\.hosts'
row unreadable-line       2 0 "$(plant unread 's/^    hosts: .*$/    - hosts: [lambda]/')" 0.71.0 'unreadable line in release_policy'
row two-blocks            2 0 "$two" 0.71.0 '2 release_policy blocks'
row no-emergency-list     2 0 "$(plant noel 's/^  emergency_scopes:$/  emergency_scopez:/')" 0.71.0 "no top-level 'emergency_scopes:' list"
row hosts-carried         0 1 "$(plant h 's/^    hosts: .*$/    hosts: [gx10]/')" 0.71.0 '^      hosts: \[gx10\]$'
row backslash-kept        0 1 "$(plant bs 's/^    quote: .*$/    quote: "a\\\\nb"/')" 0.71.0 '^      quote: "a\\\\nb"$'

echo "release_policy self-test: $n rows, $fail failed"
[ "$fail" = 0 ]
