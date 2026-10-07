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
row missing-owner         2 0 "$(plant miso '/^    ticket_owner: /d')" 0.71.0 'release_policy has no ticket_owner'
row duplicate-key         2 0 "$(plant dkey 's/^    larger_rows: nightly$/    larger_rows: nightly\n    larger_rows: release/')" 0.71.0 'duplicate key in release_policy: larger_rows'
row empty-value           2 0 "$(plant empty 's/^    hosts: .*$/    hosts: /')" 0.71.0 'empty value for release_policy\.hosts'
row unreadable-line       2 0 "$(plant unread 's/^    hosts: .*$/    - hosts: [lambda]/')" 0.71.0 'unreadable line in release_policy'
row two-blocks            2 0 "$two" 0.71.0 '2 release_policy blocks'
row header-comment        2 0 "$(plant hc 's/^  release_policy:$/  release_policy: # note/')" 0.71.0 'unreadable release_policy header'
row header-indent         2 0 "$(plant hi 's/^  release_policy:$/    release_policy:/')" 0.71.0 'unreadable release_policy header'
row key-after-blank-line  2 0 "$(plant bl 's/^    larger_rows: nightly$/\n    larger_rowz: nightly/')" 0.71.0 'unknown key in release_policy: larger_rowz'
row no-emergency-list     2 0 "$(plant noel 's/^  emergency_scopes:$/  emergency_scopez:/')" 0.71.0 "no top-level 'emergency_scopes:' list"
row hosts-carried         0 1 "$(plant h 's/^    hosts: .*$/    hosts: [gx10]/')" 0.71.0 '^      hosts: \[gx10\]$'
row backslash-kept        0 1 "$(plant bs 's/^    quote: .*$/    quote: "a\\\\nb"/')" 0.71.0 '^      quote: "a\\\\nb"$'
# A synthesized copy that cannot be written is rc 2, never the uncovered ladder.
TMPDIR="$tmp/no-such-dir" row mktemp-failed 2 0 "$ladder" 0.71.0 'mktemp failed'

# rp_known_failures: the release notes' known-failures list. A gh stub answers the nightly issue search.
cat > "$tmp/gh" <<'GH'
#!/usr/bin/env bash
[ "${KF_FAIL:-}" != 1 ] || exit 1
printf '%s\n' "$KF_JSON"
GH
chmod +x "$tmp/gh"
# kf NAME WANT_RC LADDER ISSUES-JSON FAIL PATTERN [FORBID]
kf() {
    local out rc
    n=$((n + 1))
    out=$(RP_GH="$tmp/gh" KF_JSON="$4" KF_FAIL="$5" rp_known_failures "$3" o/r 2>&1) && rc=0 || rc=$?
    if [ "$rc" != "$2" ] || ! printf '%s\n' "$out" | grep -qE -- "$6" || { [ -n "${7:-}" ] && printf '%s\n' "$out" | grep -qE -- "$7"; }; then
        printf 'FAIL %s: rc %s (want %s), output:\n%s\n' "$1" "$rc" "$2" "$out"; fail=$((fail + 1)); return 0
    fi
    printf 'ok   %s (rc %s)\n' "$1" "$rc"
}
kf kf-known-red-listed      0 "$ladder" '[]' "" '^- qwen35-0\.8b-q4km: known red \(ladder\), #4030$' 'not measured'
kf kf-nightly-rows-listed   0 "$ladder" '[{"number":12,"title":"models-nightly red: fx-1 on gx10"},{"number":9,"title":"models-nightly red: lane on all"}]' "" \
    '^- lane on all: red in the models nightly, #9$'
kf kf-nightly-sorted        0 "$ladder" '[{"number":12,"title":"models-nightly red: fx-1 on gx10"},{"number":9,"title":"models-nightly red: lane on all"}]' "" \
    '^- fx-1 on gx10: red in the models nightly, #12$'
kf kf-other-issue-not-listed 0 "$ladder" '[{"number":3,"title":"something about models-nightly red: x"}]' "" '^- qwen35' '#3$'
kf kf-unread-said           0 "$ladder" '[]' 1 'could not be read when this release was cut \(not measured\)' '^- none$'
kf kf-none                  0 "$(plant nokr '/^  known_red:$/,/^      ruling:/d')" '[]' "" '^- none$' 'known red'
kf kf-ladder-missing        2 "$tmp/none.yaml" '[]' "" '^$|.*' 'Known failures'
# A search that fills gh's 1000-row ceiling may have been cut: not measured, never a short list.
kf kf-full-search-unread    0 "$ladder" "$(jq -nc '[range(1000) | {number: (. + 1), title: "models-nightly red: r\(.) on gx10"}]')" "" \
    'could not be read when this release was cut \(not measured\)' 'red in the models nightly'
kf kf-999-rows-listed       0 "$ladder" "$(jq -nc '[range(999) | {number: (. + 1), title: "models-nightly red: r\(.) on gx10"}]')" "" \
    '^- r998 on gx10: red in the models nightly, #999$' 'not measured'

# rp_ticket_owner: the one owner every nightly red-row ticket names. No owner, no ticket.
# own NAME WANT_RC LADDER PATTERN  (rc 0: PATTERN matches stdout; rc 2: PATTERN matches RP_WHY)
own() {
    local out rc
    n=$((n + 1))
    out=$(rp_ticket_owner "$3" 2>&1) && rc=0 || rc=$?
    rp_ticket_owner "$3" > /dev/null 2>&1 || true # this shell gets RP_WHY
    if [ "$rc" != "$2" ] || ! { [ "$rc" = 0 ] && printf '%s\n' "$out" || printf '%s\n' "$RP_WHY"; } | grep -qE -- "$4"; then
        printf 'FAIL %s: rc %s (want %s), out "%s", why "%s"\n' "$1" "$rc" "$2" "$out" "$RP_WHY"; fail=$((fail + 1)); return 0
    fi
    printf 'ok   %s (rc %s%s)\n' "$1" "$rc" "${RP_WHY:+: $RP_WHY}"
}
own owner-real            0 "$ladder" '^#3598$'
own owner-name            0 "$(plant oname 's/^    ticket_owner: .*$/    ticket_owner: models-team/')" '^models-team$'
own owner-missing         2 "$(plant omiss '/^    ticket_owner: /d')" 'release_policy has no ticket_owner'
own owner-empty-issue     2 "$(plant ozero 's/^    ticket_owner: .*$/    ticket_owner: "#0"/')" "ticket_owner '#0' is neither"
own owner-two-words       2 "$(plant otwo 's/^    ticket_owner: .*$/    ticket_owner: "Some One"/')" "ticket_owner 'Some One' is neither"
own owner-no-block        2 "$(plant onopol '/^  release_policy:/,/^    release_notes:/d')" 'no release_policy block, so no ticket owner'
own owner-ladder-missing  2 "$tmp/none.yaml" 'cannot read the ladder'

echo "release_policy self-test: $n rows, $fail failed"
[ "$fail" = 0 ]
