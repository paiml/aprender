#!/usr/bin/env bash
# check_crux_receipt_binary_sha.sh - an R7 CRUX receipt names the sha256 of the apr it measured (#6679).
#
# The version line names a COMMIT, not a binary. This runs scripts/lib/crux_apr_sha.sh over a
# case table and checks that the two receipt producers call it: the driver stamps meta.json
# before the judge reads it, and the shard merge refuses shards measured on different binaries.
# It records only. Comparing the sha with a fresh build of the candidate is NOT done here.
#
#   bash scripts/check_crux_receipt_binary_sha.sh
#
# EXIT 0 every row lands; 1 a row or a wiring check failed; 2 cannot run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
LIB="${CRUX_APR_SHA_LIB:-$ROOT/scripts/lib/crux_apr_sha.sh}"
command -v jq > /dev/null 2>&1 || { echo "jq is not installed" >&2; exit 2; }
[ -f "$LIB" ] || { echo "FAIL  no $LIB"; exit 1; }
# shellcheck source=lib/crux_apr_sha.sh
. "$LIB" || exit 2
T=$(mktemp -d "${TMPDIR:-/tmp}/crux-apr-sha.XXXXXX") || exit 2
trap 'rm -rf -- "${T:?}"' EXIT
bad=0
row() { # row <label> <want rc> <cmd...>
    local label=$1 want=$2 got; shift 2
    "$@" > "$T/out" 2>&1; got=$?
    if [ "$got" = "$want" ]; then echo "ok    $label"; else echo "FAIL  $label: rc=$got want=$want: $(tail -n 1 "$T/out")"; bad=1; fi
}
meta() { printf '%s\n' "$2" > "$T/$1"; }

printf 'binary one\n' > "$T/apr"; chmod +x "$T/apr"
want=$(sha256sum "$T/apr"); want=${want%% *}
row "the hash is the binary's sha256" 0 test "$(crux_apr_sha256 "$T/apr")" = "$want"
row "an unresolvable binary is refused" 1 crux_apr_sha256 "$T/no-such-apr"

meta m.json '{"apr":{"version_line":"apr 1.2.3 (abc)"},"models":[]}'
row "stamp writes the pinned sha" 0 crux_stamp_apr_sha "$T/m.json" "$T/apr" "$want"
row "the stamped meta carries sha and keeps the version line" 0 jq -e --arg s "$want" \
    '.apr.binary_sha256 == $s and .apr.version_line == "apr 1.2.3 (abc)" and .models == []' "$T/m.json"
meta noapr.json '{"models":[]}'
row "a meta with no apr object is refused" 1 crux_stamp_apr_sha "$T/noapr.json" "$T/apr" "$want"
row "an empty pinned sha is refused" 1 crux_stamp_apr_sha "$T/m.json" "$T/apr" ""
printf 'binary two\n' > "$T/apr"
meta swapped.json '{"apr":{"version_line":"v"}}'
row "a binary replaced during the run is refused" 1 crux_stamp_apr_sha "$T/swapped.json" "$T/apr" "$want"
row "the refused meta was not stamped" 0 jq -e '.apr | has("binary_sha256") | not' "$T/swapped.json"

a=$(printf 'a%.0s' {1..64}); b=$(printf 'b%.0s' {1..64})
meta a1.json "{\"apr\":{\"binary_sha256\":\"$a\"}}"; meta a2.json "{\"apr\":{\"binary_sha256\":\"$a\"}}"
meta b1.json "{\"apr\":{\"binary_sha256\":\"$b\"}}"; meta none.json '{"apr":{}}'; meta junk.json 'not json'
row "shards of one binary merge" 0 crux_one_apr_sha "$T/a1.json" "$T/a2.json"
row "shards of two binaries are refused" 1 crux_one_apr_sha "$T/a1.json" "$T/b1.json"
row "a stamped shard beside an unstamped one is refused" 1 crux_one_apr_sha "$T/a1.json" "$T/none.json"
row "shards that all predate the stamp still re-merge (the receipt names no sha)" 0 crux_one_apr_sha "$T/none.json" "$T/none.json"
row "an unreadable shard meta is refused" 1 crux_one_apr_sha "$T/a1.json" "$T/junk.json"
row "no shard meta at all is refused" 1 crux_one_apr_sha

# Wiring. line_of FILE PATTERN -> the first non-comment line holding PATTERN.
line_of() { awk -v p="$2" '!/^[ \t]*#/ && index($0, p) { print NR; exit }' "$1"; }
D="$ROOT/scripts/crux_inference_dogfood.sh"; S="$ROOT/scripts/crux_sweep_shards.sh"
ordered() { # ordered <file> <pattern>... -- every pattern present, in this order
    local f=$1 p n last=0; shift
    for p in "$@"; do
        n=$(line_of "$f" "$p")
        [ -n "$n" ] && [ "$n" -gt "$last" ] || { echo "missing or out of order in $(basename "$f"): $p"; return 1; }
        last=$n
    done
}
# the patterns are literal shell text in the two producers, not expansions of this script's variables
# shellcheck disable=SC2016
row "the driver hashes at the pin, stamps meta.json, then judges" 0 ordered "$D" \
    'APR_BINARY_SHA256=$(crux_apr_sha256 "$APR")' \
    'crux_stamp_apr_sha "$WORK/meta.json" "$APR" "$APR_BINARY_SHA256" || decline' \
    'crux_inference_judge.py collect'
# shellcheck disable=SC2016
row "the shard merge refuses mixed binaries before the judge" 0 ordered "$S" \
    'crux_one_apr_sha "${shard_metas[@]}" || die' 'crux_inference_judge.py collect'

[ "$bad" = 0 ] && { echo "PASS  R7 receipts record the measured apr binary's sha256"; exit 0; }
exit 1
