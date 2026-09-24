#!/usr/bin/env bash
# check_nightly_pin.sh — the case table for scripts/nightly_pin.sh (#4186).
#
#     bash scripts/check_nightly_pin.sh              # run the table
#     bash scripts/check_nightly_pin.sh --self-test  # table + mutants
#
# nightly_pin.sh decides, on a fleet host, whether the apr/pv about to run is
# THE binary the arbiter's nightly manifest names. Every row below is a
# condition that must refuse, except the rows that are the nightly (or an
# explicit override that widens nothing). Refusal rows assert the REASON, not
# just a non-zero rc: a manifest check deleted outright still refuses, because
# a later check trips over the missing data, so a rc-only table would pass it.
#
# --self-test then re-runs the table against mutants of nightly_pin.sh, one
# per check. Every mutant must turn at least one row RED; a mutant the table
# cannot see is a check nothing tests (CLAUDE.md, verification discipline 4/7).
set -euo pipefail

ROOT=$(git rev-parse --show-toplevel)
LIB_REAL="$ROOT/scripts/nightly_pin.sh"
T=$(mktemp -d)
case "$T" in /tmp/* | "${TMPDIR:-/tmp}"/*) ;; *) echo "check_nightly_pin: unexpected mktemp dir '$T'" >&2; exit 2 ;; esac
trap 'rm -rf -- "${T:?}"' EXIT

# HERMETIC: every knob nightly_pin.sh reads is scrubbed here, once, so a row
# sees only what it sets. The refusal text itself advises exporting
# APR_BIN_REQUIRE=head, and a fleet host carries a real marker under $HOME;
# either leaked into the "marker makes nightly the default" row and failed a
# correct tree (PMAT-4186 quorum). The marker baseline points at nothing.
unset APR_BIN PV_BIN APR_BIN_REQUIRE PV_BIN_REQUIRE APR_NIGHTLY_MANIFEST APR_NIGHTLY_MAX_AGE_H GITHUB_ACTIONS GITHUB_RUN_ID
export APR_FLEET_MARKER="$T/nomarker"

case "$(uname -m)" in
    x86_64) TRIPLE=x86_64-unknown-linux-gnu ;;
    aarch64 | arm64) TRIPLE=aarch64-unknown-linux-gnu ;;
    *) echo "check_nightly_pin: unsupported host $(uname -m)" >&2; exit 2 ;;
esac
G=0123456789abcdef0123456789abcdef01234567  # the green nightly sha
S=fedcba9876543210fedcba9876543210fedcba98  # some older main sha

mkbin() {  # mkbin PATH VERSION-LINE
    mkdir -p "$(dirname "$1")"
    printf '#!/bin/sh\n# %s\necho "%s"\n' "$1" "$2" >"$1"
    chmod +x "$1"
}
mkbin "$T/nightly/apr" "apr 0.70.0 (${G:0:9})"
mkbin "$T/cuda/apr" "apr 0.70.0 (${G:0:9})"
mkbin "$T/nightly/pv" "pv 0.70.0 (aprender provable-contracts verifier)"
mkbin "$T/stale/apr" "apr 0.70.0 (${S:0:9})"
mkbin "$T/tagged/apr" "apr 0.69.1 (v0.69.1+no-git)"
mkbin "$T/cratesio/pv" "pv 0.69.1 (aprender provable-contracts verifier)"
mkbin "$T/liar/apr" "apr 0.70.0 (${S:0:9})"
mkbin "$T/nosha/apr" "apr 0.70.0 (v0.70.0+no-git)"
h() { sha256sum "$1" | awk '{print $1}'; }
HA=$(h "$T/nightly/apr"); HC=$(h "$T/cuda/apr")

NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)  # bashrs disable-line=DET002 (fixture timestamps relative to now)
OLD=$(date -u -d '40 hours ago' +%Y-%m-%dT%H:%M:%SZ)  # bashrs disable-line=DET002 (fixture timestamps relative to now)
FUT=$(date -u -d '2 hours' +%Y-%m-%dT%H:%M:%SZ)  # bashrs disable-line=DET002 (fixture timestamps relative to now)
jq -n --arg now "$NOW" --arg t "$TRIPLE" --arg g "$G" \
    --arg a "$(h "$T/nightly/apr")" --arg c "$(h "$T/cuda/apr")" --arg p "$(h "$T/nightly/pv")" \
    --arg l "$(h "$T/liar/apr")" --arg n "$(h "$T/nosha/apr")" '
    {schema: "aprender-nightly-manifest/v1", generated_at: $now, trigger_sha: $g,
     decision: "built", denylist: [],
     targets: {($t): {status: "green", green_sha: $g, built_at: $now, red: null,
        tools: {apr: {asset: "apr.tar.gz", sha256: "x", bin_sha256: $a, version_sha: $g, version_output: "apr"},
                "apr@cuda": {asset: "apr-cuda.tar.gz", sha256: "x", bin_sha256: $c, version_sha: $g, version_output: "apr"},
                pv: {asset: "pv.tar.gz", sha256: "x", bin_sha256: $p, version_sha: null, version_output: "pv"},
                "apr@liar": {bin_sha256: $l, version_sha: $g},
                "apr@nosha": {bin_sha256: $n, version_sha: $g}}}}}' >"$T/base.json"

# ROWS: name | manifest jq filter ("-" = no file, "!" = not JSON) | env | tool | binary | expect | reason
ROWS=$(cat <<EOF
nightly apr accepted|.| |apr|$T/nightly/apr|accept|
nightly apr@cuda variant accepted|.| |apr|$T/cuda/apr|accept|
nightly pv accepted on bin_sha256 (no sha in --version)|.| |pv|$T/nightly/pv|accept|
stale main build refused|.| |apr|$T/stale/apr|refuse|NOT THE NIGHTLY
tagged release (+no-git) not named by manifest refused|.| |apr|$T/tagged/apr|refuse|NOT THE NIGHTLY
crates.io pv refused|.| |pv|$T/cratesio/pv|refuse|NOT THE NIGHTLY
missing manifest refused (fail closed)|-| |apr|$T/nightly/apr|refuse|MISSING MANIFEST
malformed manifest refused|!| |apr|$T/nightly/apr|refuse|MALFORMED MANIFEST (not a JSON
unknown schema refused|.schema="aprender-nightly-manifest/v2"| |apr|$T/nightly/apr|refuse|UNKNOWN SCHEMA
manifest older than max age refused|.generated_at="$OLD"| |apr|$T/nightly/apr|refuse|STALE MANIFEST
max age override widens only by env|.generated_at="$OLD"|APR_NIGHTLY_MAX_AGE_H=48|apr|$T/nightly/apr|accept|
manifest from the future refused|.generated_at="$FUT"| |apr|$T/nightly/apr|refuse|FROM THE FUTURE
unparseable generated_at refused|.generated_at="yesterday-ish"| |apr|$T/nightly/apr|refuse|MALFORMED MANIFEST (generated_at
no target for this host refused|del(.targets["$TRIPLE"])| |apr|$T/nightly/apr|refuse|NO GREEN NIGHTLY
short green_sha refused|.targets["$TRIPLE"].green_sha="0123456"| |apr|$T/nightly/apr|refuse|NO GREEN NIGHTLY
tool absent from manifest refused|del(.targets["$TRIPLE"].tools.pv)| |pv|$T/nightly/pv|refuse|no nightly 'pv'
tool denylisted by name refused|.denylist=["pv"]| |pv|$T/nightly/pv|refuse|DENYLISTED
green sha denylisted refused|.denylist=["$G"]| |apr|$T/nightly/apr|refuse|DENYLISTED
variant denylisted by object refused|.denylist=[{"tool":"apr@cuda"}]| |apr|$T/cuda/apr|refuse|DENYLISTED
binary denylisted by its sha256 (object .bin) refused|.denylist=[{"bin":"$HA"}]| |apr|$T/nightly/apr|refuse|DENYLISTED
binary denylisted by its sha256 (string) refused|.denylist=["$HA"]| |apr|$T/nightly/apr|refuse|DENYLISTED
every field of a denylist object counts|.denylist=[{"tool":"nothing-here","sha":"$G"}]| |apr|$T/nightly/apr|refuse|DENYLISTED
another binary's sha256 denylisted -> accept|.denylist=["$HC"]| |apr|$T/nightly/apr|accept|
hash matches but --version sha is not green refused|.| |apr|$T/liar/apr|refuse|is not green_sha
version_sha set but binary prints none refused|.| |apr|$T/nosha/apr|refuse|carries no sha
version_sha != green_sha refused|.targets["$TRIPLE"].tools.apr.version_sha="$S"| |apr|$T/nightly/apr|refuse|MALFORMED MANIFEST (tools
missing bin_sha256 refused|del(.targets["$TRIPLE"].tools.pv.bin_sha256)| |pv|$T/nightly/pv|refuse|NOT THE NIGHTLY
non-numeric max age refused|.|APR_NIGHTLY_MAX_AGE_H=forever|apr|$T/nightly/apr|refuse|not a whole number
EOF
)

# MODE ROWS: name | env | expected nightly_pin_mode rc
MODES=$(cat <<EOF
unset, no fleet marker -> HEAD provenance|APR_FLEET_MARKER=$T/nomarker|1
explicit nightly -> nightly|APR_BIN_REQUIRE=nightly APR_FLEET_MARKER=$T/nomarker|0
typo mode refused, never guessed|APR_BIN_REQUIRE=nighty APR_FLEET_MARKER=$T/nomarker|2
fleet marker, unset -> nightly (fleet default)|APR_FLEET_MARKER=$T/marker|0
fleet marker, explicit head -> HEAD provenance|APR_BIN_REQUIRE=head APR_FLEET_MARKER=$T/marker|1
fleet marker inside GitHub Actions -> HEAD provenance|GITHUB_ACTIONS=true GITHUB_RUN_ID=31631488466 APR_FLEET_MARKER=$T/marker|1
fleet marker, bare GITHUB_ACTIONS=true (no run id) -> nightly|GITHUB_ACTIONS=true APR_FLEET_MARKER=$T/marker|0
fleet marker, non-numeric run id -> nightly|GITHUB_ACTIONS=true GITHUB_RUN_ID=x APR_FLEET_MARKER=$T/marker|0
fleet marker, nightly job in Actions -> nightly|GITHUB_ACTIONS=true APR_BIN_REQUIRE=nightly APR_FLEET_MARKER=$T/marker|0
EOF
)
: >"$T/marker"

run_table() {  # run_table LIB -> prints one line per row, returns # of failures
    local lib="$1" fails=0 n=0 name filt envs tool bin expect reason m rc err verdict
    while IFS='|' read -r name filt envs tool bin expect reason; do
        [ -n "$name" ] || continue
        n=$((n + 1)); m="$T/m-$n.json"
        case "$filt" in
            -) m="$T/does-not-exist.json" ;;
            !) printf '{not json' >"$m" ;;
            *) jq "$filt" "$T/base.json" >"$m" ;;
        esac
        rc=0
        # shellcheck disable=SC2086  # envs is a deliberate word list
        err=$(env -u APR_BIN_REQUIRE -u PV_BIN_REQUIRE -u GITHUB_ACTIONS APR_NIGHTLY_MANIFEST="$m" $envs \
            bash -c '. "$1" && nightly_pin_check "$2" "$3" APR_BIN_REQUIRE' _ "$lib" "$tool" "$bin" 2>&1 >/dev/null) || rc=$?
        verdict=FAIL
        if [ "$expect" = accept ] && [ "$rc" -eq 0 ]; then verdict=ok; fi
        if [ "$expect" = refuse ] && [ "$rc" -ne 0 ]; then
            case "$err" in *"$reason"*) verdict=ok ;; esac
        fi
        [ "$verdict" = ok ] || fails=$((fails + 1))
        printf '  %-4s %-58s %s rc=%s\n' "$verdict" "$name" "$expect" "$rc"
        [ "$verdict" = ok ] || printf '       stderr: %s\n' "$(printf '%s' "$err" | head -n 1)"
    done <<<"$ROWS"
    while IFS='|' read -r name envs expect; do
        [ -n "$name" ] || continue
        rc=0
        # shellcheck disable=SC2086
        env -u APR_BIN_REQUIRE -u GITHUB_ACTIONS $envs bash -c '. "$1" && nightly_pin_mode APR_BIN_REQUIRE' _ "$lib" >/dev/null 2>&1 || rc=$?
        verdict=FAIL
        [ "$rc" = "$expect" ] && verdict=ok
        [ "$verdict" = ok ] || fails=$((fails + 1))
        printf '  %-4s mode: %-52s rc=%s want %s\n' "$verdict" "$name" "$rc" "$expect"
    done <<<"$MODES"
    return "$fails"
}

# END-TO-END: the resolvers themselves, as a gate sources them.
run_e2e() {
    local fails=0 out rc v m="$T/e2e.json"
    cp "$T/base.json" "$m"
    e2e() {  # e2e NAME EXPECT(accept|refuse) WANT CMD...
        local name="$1" expect="$2" want="$3"; shift 3
        rc=0
        out=$(cd "$ROOT" && env -u APR_BIN -u PV_BIN -u GITHUB_ACTIONS APR_NIGHTLY_MANIFEST="$m" "$@" 2>&1) || rc=$?
        local v=FAIL
        if [ "$expect" = accept ] && [ "$rc" -eq 0 ] && [ "$out" = "$want" ]; then v=ok; fi
        if [ "$expect" = refuse ] && [ "$rc" -ne 0 ]; then case "$out" in *"$want"*) v=ok ;; esac; fi
        [ "$v" = ok ] || fails=$((fails + 1))
        printf '  %-4s e2e: %-52s rc=%s\n' "$v" "$name" "$rc"
        [ "$v" = ok ] || printf '       out: %s\n' "$(printf '%s' "$out" | tail -n 2)"
    }
    e2e "apr_bin.sh nightly mode resolves the nightly" accept "$T/nightly/apr" \
        env APR_BIN_REQUIRE=nightly PATH="$T/nightly:$PATH" bash -c '. scripts/apr_bin.sh || exit 1; printf %s "$APR"'
    e2e "apr_bin.sh nightly mode refuses a stale PATH apr" refuse "NOT THE NIGHTLY" \
        env APR_BIN_REQUIRE=nightly PATH="$T/stale:$PATH" bash -c '. scripts/apr_bin.sh || exit 1; printf %s "$APR"'
    e2e "apr_bin.sh APR_BIN override is still checked" refuse "NOT THE NIGHTLY" \
        env APR_BIN_REQUIRE=nightly APR_BIN="$T/tagged/apr" bash -c '. scripts/apr_bin.sh || exit 1; printf %s "$APR"'
    e2e "apr_bin.sh relative APR_BIN override refused" refuse "is not an absolute path" \
        env APR_BIN_REQUIRE=nightly APR_BIN=scripts/apr_bin.sh bash -c '. scripts/apr_bin.sh || exit 1; printf %s "$APR"'
    e2e "apr_bin.sh typo mode refused" refuse "not a known mode" \
        env APR_BIN_REQUIRE=nighty bash -c '. scripts/apr_bin.sh || exit 1; printf %s "$APR"'
    e2e "apr_bin.sh fleet marker makes nightly the default" refuse "NOT THE NIGHTLY" \
        env APR_FLEET_MARKER="$T/marker" PATH="$T/stale:$PATH" bash -c '. scripts/apr_bin.sh || exit 1; printf %s "$APR"'
    e2e "pv_bin.sh nightly mode resolves the nightly (no cargo build)" accept "$T/nightly/pv" \
        env PV_BIN_REQUIRE=nightly PATH="$T/nightly:$PATH" bash -c '. scripts/pv_bin.sh || exit 1; printf %s "$PV"'
    e2e "pv_bin.sh nightly mode refuses a crates.io pv" refuse "NOT THE NIGHTLY" \
        env PV_BIN_REQUIRE=nightly PATH="$T/cratesio:$PATH" bash -c '. scripts/pv_bin.sh || exit 1; printf %s "$PV"'
    e2e "apr_bin.sh Actions skip of the marker is never silent" refuse "fleet marker ignored inside GitHub Actions" \
        env GITHUB_ACTIONS=true GITHUB_RUN_ID=7 APR_FLEET_MARKER="$T/marker" APR_BIN="$T/stale/apr" bash -c '. scripts/apr_bin.sh || exit 1; printf %s "$APR"'
    e2e "apr_bin.sh accepts under the caller's set -euo pipefail" accept "$T/nightly/apr" \
        env APR_BIN_REQUIRE=nightly PATH="$T/nightly:$PATH" bash -c 'set -euo pipefail; . scripts/apr_bin.sh; printf %s "$APR"'
    e2e "apr_bin.sh refuses under the caller's set -euo pipefail" refuse "NOT THE NIGHTLY" \
        env APR_BIN_REQUIRE=nightly PATH="$T/stale:$PATH" bash -c 'set -euo pipefail; . scripts/apr_bin.sh; printf %s "$APR"'
    e2e "pv_bin.sh fleet marker makes nightly the default" refuse "NOT THE NIGHTLY" \
        env APR_FLEET_MARKER="$T/marker" PATH="$T/cratesio:$PATH" bash -c '. scripts/pv_bin.sh || exit 1; printf %s "$PV"'
    e2e "pv_bin.sh missing manifest refused" refuse "MISSING MANIFEST" \
        env PV_BIN_REQUIRE=nightly APR_NIGHTLY_MANIFEST="$T/none.json" PATH="$T/nightly:$PATH" bash -c '. scripts/pv_bin.sh || exit 1; printf %s "$PV"'
    # Outside any checkout the rule itself is unreachable; with the fleet
    # marker present that must refuse, not quietly fall back to HEAD mode.
    rc=0
    out=$(cd "$T" && env -u APR_BIN_REQUIRE -u GITHUB_ACTIONS APR_FLEET_MARKER="$T/marker" \
        bash -c '. "$1" || exit 1; printf %s "$APR"' _ "$ROOT/scripts/apr_bin.sh" 2>&1) || rc=$?
    v=FAIL
    if [ "$rc" -ne 0 ]; then case "$out" in *"nightly_pin.sh is not in this checkout"*) v=ok ;; esac; fi
    [ "$v" = ok ] || fails=$((fails + 1))
    printf '  %-4s e2e: %-52s rc=%s\n' "$v" "fleet marker outside a checkout refused" "$rc"
    return "$fails"
}

echo "check_nightly_pin: case table against scripts/nightly_pin.sh ($TRIPLE)"
rc=0
run_table "$LIB_REAL" || rc=$?
run_e2e || rc=$((rc + $?))
if [ "$rc" -ne 0 ]; then
    echo "check_nightly_pin: FAIL ($rc rows)"
    exit 1
fi

if [ "${1:-}" = "--self-test" ]; then
    # MUTANTS: each disables one check. Format: name | sed expression.
    MUTANTS=$(cat <<'EOF'
accept any hash|s/and \.value\.bin_sha256 == \$h)/)/
never stale|s/-gt \$((np_max_h \* 3600))/-gt 999999999/
no future check|s/-gt \$((np_now + 300))/-gt \$((np_now + 999999999))/
no missing-manifest check|s/\[ -f "\$np_m" \] \&\& \[ -r "\$np_m" \] ||/true ||/
no schema check|s/\[ "\$np_schema" = "\$NIGHTLY_PIN_SCHEMA" \] ||/true ||/
no tool/commit denylist check|s/\[ "\$np_denied" = "0" \] || { nightly_pin_refuse "\$np_tool" "DENYLISTED/true || { nightly_pin_refuse "$np_tool" "DENYLISTED/
no key/binary-hash denylist check|s/\[ "\$np_denied" = "0" \] || { nightly_pin_refuse "\$np_tool" "'\$np_key'/true || { nightly_pin_refuse "$np_tool" "'$np_key'/
denylist object reads one field|s/(\.tool, \.bin, \.sha | select(\. != null))/(.tool \/\/ .bin \/\/ .sha \/\/ empty)/g
no version-sha check|s/"\$np_bsha"\*) ;;/*) ;;/
no version_sha==green check|s/\[ "\$np_vsha" = "\$np_green" \] ||/true ||/
typo mode falls back to HEAD|s/return 2 ;;/return 1 ;;/
marker honoured in Actions|s/\[ "\${GITHUB_ACTIONS:-}" = "true" \]/[ "${GITHUB_ACTIONS:-}" = "never" ]/
bare GITHUB_ACTIONS=true qualifies|s/'' | \*\[!0-9\]\*) ;;/'' | *[!0-9]*) np_actions=1 ;;/
marker ignored|s/np_mode=nightly$/np_mode=/
EOF
)
    echo "check_nightly_pin: mutants"
    dead=0
    while IFS='|' read -r mname expr; do
        [ -n "$mname" ] || continue
        cp "$LIB_REAL" "$T/mut.sh"
        sed -i "$expr" "$T/mut.sh"
        if cmp -s "$LIB_REAL" "$T/mut.sh"; then
            printf '  FAIL mutant %-40s did not apply (pattern drifted)\n' "$mname"
            dead=$((dead + 1)); continue
        fi
        mrc=0
        run_table "$T/mut.sh" >/dev/null || mrc=$?
        if [ "$mrc" -gt 0 ]; then
            printf '  ok   mutant %-40s killed by %s row(s)\n' "$mname" "$mrc"
        else
            printf '  FAIL mutant %-40s SURVIVED\n' "$mname"
            dead=$((dead + 1))
        fi
    done <<<"$MUTANTS"
    if [ "$dead" -ne 0 ]; then
        echo "check_nightly_pin: FAIL ($dead mutants survived or did not apply)"
        exit 1
    fi
fi
echo "check_nightly_pin: PASS"
