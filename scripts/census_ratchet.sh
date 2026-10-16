#!/usr/bin/env bash
# census_ratchet.sh -- the pv census ratchet (#4709, C303): the bound share of contracts/ only rises.
#
# Measured at both ends with ONE pv in ONE run, never against a stored number:
#
#   bound    = by_anchoring.class + by_anchoring.instance   (pv census --format json)
#   unbound  = by_anchoring.unanchored
#   missing  = .no_falsification_tests (pv census; its release caps are a pv lib test, not here)
#
# PR mode (default): the comparand is scripts/lib/resolve_base.sh's base, extracted with `git archive`.
#   RED if bound falls, if the unbound SHARE rises (integer cross-multiplication, never a float),
#   or if missing rises.
# --release <ver> --commit <sha>: the comparand is the previous v* tag before <sha>.
#   RED if bound(sha) < bound(prev) + min(100, unbound(prev)), or if missing rises. The per-release
#   caps on missing live in pv (corpus_no_test_debt_within_release_cap), never here, and so do their numbers.
#
# NEVER VACUOUS: pv missing or without `census`, a non-zero exit, unparseable JSON, a census without
# no_falsification_tests (a pv before #5295), 0 contracts, a parse error, or anchoring counts that do not
# sum to n_parsed is NOT_MEASURED, rc 2, never a pass.
#
# This is a LAB check (C324): it runs nightly and on demand, never on a PR, merge-queue or release
# path, and it is deliberately not named check_*.sh, so guard-tree does not pick it up.
#
#   bash scripts/census_ratchet.sh                                   # base -> working tree
#   CENSUS_BASE_REF=<ref> bash scripts/census_ratchet.sh             # an explicit comparand (the nightly)
#   bash scripts/census_ratchet.sh --print [dir]                   # one measurement
#   bash scripts/census_ratchet.sh --release 0.72.0 --commit <sha>
#   bash scripts/census_ratchet.sh --self-test                       # the real pv on fixture trees
#
# rc: 0 GREEN, 1 RED, 2 NOT_MEASURED.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
PROG=census_ratchet

# resolve_pv -> sets PV; a PV= from the caller wins (the self-test's stub), else scripts/pv_bin.sh, never PATH
resolve_pv() {
    if [ -n "${PV:-}" ]; then return 0; fi
    # shellcheck source=scripts/pv_bin.sh
    # pv_bin.sh leaves PV set to a binary it REJECTED (stale, wrong tree) and says so only by its
    # status: a failed source is no pv, never a measurement by an unproven one
    . "$REPO_ROOT/scripts/pv_bin.sh" > /dev/null 2>&1 || { unset PV; return 1; }
    [ -n "${PV:-}" ]
}

# measure <tree> -> prints "n bound unbound missing", or NOT_MEASURED <reason> on stderr and rc 2
measure() {
    local tree=$1 out rc line n errs un cl inst miss help
    [ -d "$tree/contracts" ] || { printf 'NOT_MEASURED no contracts/ under %s\n' "$tree" >&2; return 2; }
    # a here-string, never a pipe into a quiet grep: grep -q exits on the first match, pv takes SIGPIPE, and
    # under pipefail a pv WITH census would read as one without it (check_ont_ratchet.sh, #3679)
    help=$("$PV" --help 2> /dev/null) || help=""
    grep -q census <<< "$help" || { printf 'NOT_MEASURED pv %s has no census\n' "$PV" >&2; return 2; }
    rc=0; out=$("$PV" census --format json "$tree/contracts" 2> /dev/null) || rc=$?
    [ "$rc" = 0 ] || { printf 'NOT_MEASURED pv census exited %s on %s\n' "$rc" "$tree" >&2; return 2; }
    line=$(printf '%s' "$out" | jq -r '[.n_parsed, .n_parse_errors, .by_anchoring.unanchored, .by_anchoring.class, .by_anchoring.instance, .no_falsification_tests] | map(if type == "number" then . else error("not a number") end) | @tsv' 2> /dev/null) \
        || { printf 'NOT_MEASURED pv census printed no usable JSON on %s\n' "$tree" >&2; return 2; }
    read -r n errs un cl inst miss <<< "$line"
    [ "$n" -gt 0 ] || { printf 'NOT_MEASURED 0 contracts parsed on %s\n' "$tree" >&2; return 2; }
    [ "$errs" = 0 ] || { printf 'NOT_MEASURED %s parse errors on %s\n' "$errs" "$tree" >&2; return 2; }
    [ $((un + cl + inst)) = "$n" ] || { printf 'NOT_MEASURED anchoring %s+%s+%s != n_parsed %s on %s\n' "$un" "$cl" "$inst" "$n" "$tree" >&2; return 2; }
    printf '%s %s %s %s\n' "$n" $((cl + inst)) "$un" "$miss"
}

# extract <ref> <dir> -> contracts/ at <ref> into <dir>
extract() {
    mkdir -p "$2" && git -C "$REPO_ROOT" archive --format=tar "$1" -- contracts | tar -xf - -C "$2"
}

# judge <base "n b u m"> <head "n b u m"> <mode pr|release> [ver] -> rc 0 GREEN / 1 RED, one line per rule
judge() {
    local bn bb bu bm hn hb hu hm need rc=0
    read -r bn bb bu bm <<< "$1"; read -r hn hb hu hm <<< "$2"
    printf '  base n=%s bound=%s unbound=%s missing_ft=%s\n  head n=%s bound=%s unbound=%s missing_ft=%s\n' \
        "$bn" "$bb" "$bu" "$bm" "$hn" "$hb" "$hu" "$hm"
    if [ "$3" = release ]; then
        need=$((bb + (bu < 100 ? bu : 100)))
        [ "$hb" -ge "$need" ] || { printf 'RED   bound %s < %s (prev %s + step min(100, %s))\n' "$hb" "$need" "$bb" "$bu"; rc=1; }
    else
        [ "$hb" -ge "$bb" ] || { printf 'RED   bound fell %s -> %s\n' "$bb" "$hb"; rc=1; }
        # unbound share rose: hu/hn > bu/bn  <=>  hu*bn > bu*hn
        [ $((hu * bn)) -le $((bu * hn)) ] || { printf 'RED   unbound share rose %s/%s -> %s/%s\n' "$bu" "$bn" "$hu" "$hn"; rc=1; }
    fi
    [ "$hm" -le "$bm" ] || { printf 'RED   missing_ft rose %s -> %s\n' "$bm" "$hm"; rc=1; }
    [ "$rc" = 0 ] && printf 'GREEN\n'
    return "$rc"
}

# run_pr <head tree> -> the working tree against resolve_base's base
run_pr() {
    local tmp b h rc
    # shellcheck source=scripts/lib/resolve_base.sh
    . "$REPO_ROOT/scripts/lib/resolve_base.sh" || return 2
    if [ -n "${CENSUS_BASE_REF:-}" ]; then BASE_REF=$CENSUS_BASE_REF; BASE_HOW="CENSUS_BASE_REF, the nightly 24 h comparand"
    else resolve_base HEAD || { printf 'NOT_MEASURED no base to compare against\n' >&2; return 2; }; fi
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/census-ratchet.XXXXXX") || return 2
    rc=0
    if extract "$BASE_REF" "$tmp"; then
        printf '  comparand %s (%s)\n' "$(git -C "$REPO_ROOT" rev-parse --short "$BASE_REF")" "$BASE_HOW"
        b=$(measure "$tmp") || rc=2
        [ "$rc" = 0 ] && { h=$(measure "$1") || rc=2; }
        [ "$rc" = 0 ] && { judge "$b" "$h" pr || rc=1; }
    else rc=2; printf 'NOT_MEASURED could not extract contracts/ at %s\n' "$BASE_REF" >&2; fi
    rm -rf "${tmp:?}"
    return "$rc"
}

# run_release <ver> <sha>
run_release() {
    local prev tmp b h rc=0
    prev=$(git -C "$REPO_ROOT" describe --tags --abbrev=0 --match 'v[0-9]*' "$2^" 2> /dev/null) \
        || { printf 'NOT_MEASURED no v* tag before %s\n' "$2" >&2; return 2; }
    tmp=$(mktemp -d "${TMPDIR:-/tmp}/census-ratchet.XXXXXX") || return 2
    if extract "$prev" "$tmp/prev" && extract "$2" "$tmp/head"; then
        printf '  release %s at %s, previous tag %s\n' "$1" "$(git -C "$REPO_ROOT" rev-parse --short "$2")" "$prev"
        b=$(measure "$tmp/prev") || rc=2
        [ "$rc" = 0 ] && { h=$(measure "$tmp/head") || rc=2; }
        [ "$rc" = 0 ] && { judge "$b" "$h" release "$1" || rc=1; }
    else rc=2; printf 'NOT_MEASURED could not extract %s or %s\n' "$prev" "$2" >&2; fi
    rm -rf "${tmp:?}"
    return "$rc"
}

# ── self-test: the real pv on fixture trees ───────────────────────────────────
fx_contract() { # fx_contract <file> <bound 0|1> <ft 0|1>
    printf 'metadata:\n  version: "1.0.0"\n  description: fixture\nequations: {}\n' > "$1"
    if [ "$3" = 1 ]; then printf 'falsification_tests:\n  - id: F-1\n' >> "$1"; else printf 'falsification_tests: []\n' >> "$1"; fi
    [ "$2" = 0 ] || printf 'entity:\n  type: kernel\n' >> "$1"
}
self_test() {
    local d bad=0 base head got
    resolve_pv || { echo "NOT_MEASURED no pv for the self-test"; return 2; }
    d=$(mktemp -d "${TMPDIR:-/tmp}/census-ratchet-st.XXXXXX") || return 2
    row() { # row <name> <want rc> <got rc>
        if [ "$2" = "$3" ]; then printf 'ok    %s (rc %s)\n' "$1" "$3"; else printf 'FAIL  %s: wanted rc %s, got %s\n' "$1" "$2" "$3"; bad=1; fi
    }
    pr_case() { # pr_case <name> <want> -> measure $d/base vs $d/head, judge in PR mode
        local b h rc=0
        b=$(measure "$d/base" 2> /dev/null) || rc=2
        [ "$rc" = 0 ] && { h=$(measure "$d/head" 2> /dev/null) || rc=2; }
        [ "$rc" = 0 ] && { judge "$b" "$h" pr > /dev/null || rc=1; }
        row "$1" "$2" "$rc"
    }
    reset() { rm -rf "${d:?}/base" "${d:?}/head"; mkdir -p "$d/base/contracts" "$d/head/contracts"
        fx_contract "$d/base/contracts/a.yaml" 1 1; fx_contract "$d/base/contracts/b.yaml" 0 1
        cp "$d"/base/contracts/*.yaml "$d/head/contracts/"; }
    reset; pr_case "unchanged tree: GREEN" 0
    reset; fx_contract "$d/head/contracts/a.yaml" 0 1; pr_case "unbind one: RED" 1
    reset; fx_contract "$d/head/contracts/c.yaml" 0 1; pr_case "add an unbound contract: RED (share rose)" 1
    reset; fx_contract "$d/head/contracts/b.yaml" 1 1; pr_case "bind one: GREEN" 0
    reset; fx_contract "$d/head/contracts/c.yaml" 1 1; pr_case "add a bound contract: GREEN" 0
    reset; fx_contract "$d/head/contracts/b.yaml" 0 0; pr_case "drop falsification_tests: RED" 1
    # bound falls while the share holds (2/4 -> 1/2): only the bound rule can see it
    reset; fx_contract "$d/base/contracts/c.yaml" 1 1; fx_contract "$d/base/contracts/e.yaml" 0 1
    cp "$d"/base/contracts/*.yaml "$d/head/contracts/"; rm -f "${d:?}/head/contracts/a.yaml" "${d:?}/head/contracts/b.yaml"
    pr_case "drop a bound and an unbound contract: RED (bound fell)" 1
    reset; printf 'not: [valid\n' > "$d/head/contracts/z.yaml"; pr_case "a parse error: NOT_MEASURED" 2
    reset; rm -f "${d:?}"/head/contracts/*.yaml; pr_case "0 contracts: NOT_MEASURED" 2
    # release rows: judge only (the step arithmetic), no tags needed
    got=0; judge "100 10 90 50" "120 20 100 50" release 0.71.9 > /dev/null || got=$?; row "release short of the step: RED" 1 "$got"
    got=0; judge "100 10 90 50" "120 100 20 50" release 0.71.9 > /dev/null || got=$?; row "release meets the step: GREEN" 0 "$got"
    got=0; judge "100 10 5 50" "100 15 0 50" release 0.71.9 > /dev/null || got=$?; row "step capped at unbound(prev): GREEN" 0 "$got"
    got=0; judge "100 10 90 50" "120 100 20 51" release 0.71.9 > /dev/null || got=$?; row "release with missing risen: RED" 1 "$got"
    # stub pv rows: each refusal of the instrument is NOT_MEASURED
    reset
    stub() { printf '#!/usr/bin/env bash\ncase "$1" in --help) echo "census lint";; *) %s;; esac\n' "$1" > "$d/pv-stub"; chmod +x "$d/pv-stub"; }
    stub_case() { local rc=0 save=$PV; PV=$d/pv-stub; measure "$d/head" > /dev/null 2>&1 || rc=$?; PV=$save; row "$1" 2 "$rc"; }
    stub 'echo garbage'; stub_case "stub pv printing garbage: NOT_MEASURED"
    stub 'exit 3'; stub_case "stub pv exiting 3: NOT_MEASURED"
    stub "echo '{\"n_parsed\":2,\"n_parse_errors\":0,\"by_anchoring\":{\"unanchored\":1,\"class\":0,\"instance\":0},\"no_falsification_tests\":0}'"; stub_case "anchoring short of n_parsed: NOT_MEASURED"
    stub "echo '{\"n_parsed\":2,\"n_parse_errors\":1,\"by_anchoring\":{\"unanchored\":2,\"class\":0,\"instance\":0},\"no_falsification_tests\":0}'"; stub_case "a parse error reported with exit 0: NOT_MEASURED"
    stub "echo '{\"n_parsed\":0,\"n_parse_errors\":0,\"by_anchoring\":{\"unanchored\":0,\"class\":0,\"instance\":0},\"no_falsification_tests\":0}'"; stub_case "0 contracts reported with exit 0: NOT_MEASURED"
    stub "echo '{\"n_parsed\":1,\"n_parse_errors\":0,\"by_anchoring\":{\"unanchored\":1,\"class\":0,\"instance\":0},\"no_falsification_tests\":0}'; exit 3"; stub_case "good JSON but exit 3: NOT_MEASURED"
    stub "echo '{\"n_parsed\":1,\"n_parse_errors\":0,\"by_anchoring\":{\"unanchored\":1,\"class\":0,\"instance\":0}}'"; stub_case "a pv without no_falsification_tests (pre-#5295): NOT_MEASURED"
    printf '#!/usr/bin/env bash\necho lint\n' > "$d/pv-stub"; stub_case "pv without census: NOT_MEASURED"
    printf '#!/usr/bin/env bash\ncase "$1" in --help) echo lint;; *) echo '"'"'{"n_parsed":1,"n_parse_errors":0,"by_anchoring":{"unanchored":1,"class":0,"instance":0},"no_falsification_tests":0}'"'"';; esac\n' > "$d/pv-stub"; stub_case "census JSON from a pv whose --help lists no census: NOT_MEASURED"
    # a pv_bin.sh that rejects its binary but leaves PV set to it: resolve_pv must refuse it, not measure with it
    stub "echo '{\"n_parsed\":1,\"n_parse_errors\":0,\"by_anchoring\":{\"unanchored\":0,\"class\":1,\"instance\":0},\"no_falsification_tests\":0}'"
    mkdir -p "$d/rr/scripts"; printf 'PV=%s\nreturn 1\n' "$d/pv-stub" > "$d/rr/scripts/pv_bin.sh"
    got=0; (unset PV; REPO_ROOT=$d/rr; resolve_pv) || got=$?; row "pv_bin.sh rejecting its pv: no pv" 1 "$got"
    rm -rf "${d:?}"
    [ "$bad" = 0 ] && { echo "SELF-TEST PASSED"; return 0; }
    echo "SELF-TEST FAILED"; return 1
}

case "${1:-}" in
    --self-test) self_test; exit $? ;;
    --print) resolve_pv || { echo "NOT_MEASURED no pv" >&2; exit 2; }
        printf 'n bound unbound missing_ft\n'; measure "${2:-$REPO_ROOT}"; exit $? ;;
    --release)
        [ $# = 4 ] && [ "$3" = --commit ] || { printf 'usage: %s --release <ver> --commit <sha>\n' "$PROG" >&2; exit 2; }
        resolve_pv || { echo "NOT_MEASURED no pv" >&2; exit 2; }
        run_release "$2" "$4"; exit $? ;;
    "") resolve_pv || { echo "NOT_MEASURED no pv" >&2; exit 2; }
        run_pr "$REPO_ROOT"; exit $? ;;
    *) printf 'usage: %s [--print [dir] | --release <ver> --commit <sha> | --self-test]\n' "$PROG" >&2; exit 2 ;;
esac
