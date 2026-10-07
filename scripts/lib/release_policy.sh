# release_policy.sh -- the STANDING release policy (`ladder.release_policy` in
# contracts/model-capability-ladder-v1.yaml), read with awk only.
#
# From `since` on, every release is judged on CRUX smoke, the scope a per-release
# `emergency_scopes` entry used to grant one release at a time. The existing judge
# (scripts/lib/crux_smoke_scope.py) is not changed: for a covered version this file
# writes a copy of the ladder with that version's entry synthesized from the policy,
# and the caller points the judge at the copy. No yq: it is not installed on every
# host that runs a release gate, so the block is read by a strict awk reader
# (release_policy_block.awk) that refuses anything it does not recognise.
#
# SOURCED library: option-neutral (no `set`), fails by return status. Callers:
#   . scripts/lib/release_policy.sh || exit 1
#
# release_policy_ladder LADDER VERSION
#   stdout: the ladder the judge must read (LADDER itself when no policy covers VERSION).
#           A synthesized copy lives under ${TMPDIR:-/tmp}; the caller removes it.
#   rc 0  ok; RP_APPLIES=1 when the policy covers VERSION, else 0
#   rc 1  RED: a per-release emergency_scopes entry names a version the policy covers
#         (one release, one ruling)
#   rc 2  not measured: the policy block, VERSION or the ladder could not be read.
#         Never a pass, never "no policy".
#   RP_WHY holds the reason on rc 1 and rc 2. Call it in the current shell, not in $( ),
#   when you need RP_APPLIES or RP_WHY.
#
# The nightly must judge the FULL ladder: callers that run nightly pass --scope none
# and never call this.

RP_KEYS="name since date quote hosts thinking larger_rows red_row_needs release_notes"
# RP_AWK_DIR: where the .awk programs live. Overridable only so a mutation run can point a
# mutated copy of this file at the real programs.
RP_AWK_DIR="${RP_AWK_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# rp_block LADDER -> sets RP_BLK to "key<TAB>raw value" per key of the policy block; rc 2 + RP_WHY on
# any line it cannot read, a duplicate or unknown key, or more than one block. rc 0 and an empty
# RP_BLK when the ladder has no policy block. Sets globals, not stdout, so RP_WHY survives.
rp_block() {
    local out rc
    out=$(awk -v keys="$RP_KEYS" -f "$RP_AWK_DIR/release_policy_block.awk" "$1" 2>&1)
    rc=$?
    RP_BLK=""
    if [ "$rc" != 0 ]; then
        RP_WHY=$(printf '%s\n' "$out" | awk -F '\t' '$1 == "ERR" { print $2; exit }')
        [ -n "$RP_WHY" ] || RP_WHY="the ladder $1 could not be read"
        return 2
    fi
    RP_BLK="$out"
}

# rp_core VERSION -> X.Y.Z with any pre-release or build suffix removed; rc 2 when it is not semver.
# 0.71.0-rc.1 is a 0.71.0 cut, so it is covered from since 0.71.0 on.
rp_core() {
    local c="${1%%[-+]*}"
    [[ "$c" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || return 2
    printf '%s\n' "$c"
}

# rp_ge A B -> rc 0 when X.Y.Z A >= B (numeric per field)
rp_ge() {
    local a1 a2 a3 b1 b2 b3
    IFS=. read -r a1 a2 a3 <<< "$1"
    IFS=. read -r b1 b2 b3 <<< "$2"
    [ "$a1" -ne "$b1" ] && { [ "$a1" -gt "$b1" ]; return; }
    [ "$a2" -ne "$b2" ] && { [ "$a2" -gt "$b2" ]; return; }
    [ "$a3" -ge "$b3" ]
}

# rp_get KEY -> the raw value of KEY in the block release_policy_ladder read ($blk, dynamic scope)
rp_get() { printf '%s\n' "$blk" | awk -F '\t' -v k="$1" '$1 == k { print $2; exit }'; }

release_policy_ladder() {
    local ladder="$1" version="$2" blk since core copy
    RP_APPLIES=0 RP_WHY=""
    if [ ! -r "$ladder" ]; then RP_WHY="cannot read the ladder $ladder"; return 2; fi
    rp_block "$ladder" || return 2
    blk="$RP_BLK"
    if [ -z "$blk" ]; then printf '%s\n' "$ladder"; return 0; fi
    since=$(rp_get since); since="${since#\"}"; since="${since%\"}"
    if ! [[ "$since" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        RP_WHY="release_policy.since '$since' is not X.Y.Z"; return 2
    fi
    if ! core=$(rp_core "$version"); then
        RP_WHY="version '$version' is not semver, so the policy's coverage cannot be judged"; return 2
    fi
    if ! rp_ge "$core" "$since"; then printf '%s\n' "$ladder"; return 0; fi
    # One release, one ruling: a per-release entry for a covered version is RED, never merged.
    if awk -v v="$version" -f "$RP_AWK_DIR/release_policy_has_entry.awk" "$ladder"; then
        RP_WHY="emergency_scopes records release $version, which release_policy (since $since) already covers -- one release takes one ruling"
        return 1
    fi
    copy=$(mktemp "${TMPDIR:-/tmp}/ladder-policy.XXXXXX.yaml") || { RP_WHY="mktemp failed"; return 2; }
    if ! RP_N="$(rp_get name)" RP_V="$version" RP_D="$(rp_get date)" RP_Q="$(rp_get quote)" \
            RP_H="$(rp_get hosts)" RP_T="$(rp_get thinking)" \
            awk -f "$RP_AWK_DIR/release_policy_entry.awk" "$ladder" > "$copy"; then
        rm -f "${copy:?}"
        RP_WHY="the ladder has no top-level 'emergency_scopes:' list to carry the policy's entry"
        return 2
    fi
    RP_APPLIES=1
    printf '%s\n' "$copy"
}
