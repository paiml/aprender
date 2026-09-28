#!/usr/bin/env bash
# check_gate_tool_pins_released.sh — N-1 RULE: every gate-tool pin in this repo
# names a version that was ALREADY on crates.io at the PR's base.
#
# WHY THIS EXISTS
# ----------------
# Operator, 2026-09-28 17:15Z: "N-1 RULE: a gate may only use tool features
# that are ALREADY released. New gate needs a new tool feature -> the tool
# ships one release earlier; the gate lands in the next. Target: gate-tool
# releases on a release's critical path = 0 (today: 2)."
#
# A pin that names an unpublished version (or one published inside the same
# PR) puts a tool release on the critical path of the aprender release that
# carries it. This guard makes that RED at review time, before the release.
#
# WHAT IS A PIN
# -------------
#   tools.toml            every [section] with `version = "X"`
#   scripts/*.sh          a line starting  PMAT_PIN= / BASHRS_PIN= / PV_PIN= / FORJAR_PIN=
#   scripts/*.txt         a baseline header line  "# tool_version=<tool> <ver>"
#                         ("none" means no versioned analyser and is skipped)
#
# Tool -> crate: pmat -> pmat, bashrs -> bashrs, pv -> aprender-contracts-cli,
# forjar -> forjar. A pinned tool with no crate mapping is RED: the guard
# cannot prove it is released, so it does not pretend to.
#
# VERDICT PER PIN (crates.io /api/v1/crates/<crate>/<version>)
#   PASS        exists, not yanked, created_at < committer date of BASE
#   FAIL        404 (not on crates.io)          -- the planted RED is this
#   FAIL        yanked
#   FAIL        created_at >= BASE date         -- released inside this PR: not N-1
#   FAIL        not exact X.Y.Z (ranges, -rc, +dev are not "released")
#   UNMEASURED  registry unreachable / non-200-non-404 / unparseable reply.
#               Printed, never counted as PASS; exit 0 only if nothing FAILed.
#
# BASE = $GATE_PIN_BASE, else HEAD^1 when HEAD is a merge commit (the
# pull_request checkout), else the merge-base with origin/main, else HEAD.
#
#   bash scripts/check_gate_tool_pins_released.sh              # check
#   bash scripts/check_gate_tool_pins_released.sh --self-test  # case table (offline)
#
# Env: GATE_PIN_ROOT (tree to scan, default repo root), GATE_PIN_BASE (commit),
# GATE_PIN_FIXTURES (dir of <crate>-<ver>.json / .404 replies; offline mode).

set -uo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
UA="aprender-gate-pin-check (https://github.com/paiml/aprender)"

usage() {
    printf 'usage: %s [--self-test]\n' "$(basename "$0")" >&2
}

# crate_for TOOL -- crates.io crate that ships TOOL; prints nothing if unmapped.
crate_for() { # crate_for TOOL
    case "$1" in
        pmat) printf 'pmat\n' ;;
        bashrs) printf 'bashrs\n' ;;
        pv) printf 'aprender-contracts-cli\n' ;;
        forjar) printf 'forjar\n' ;;
        *) return 0 ;;
    esac
}

# collect_pins ROOT -- prints "source<TAB>tool<TAB>version", one per pin.
collect_pins() { # collect_pins ROOT
    local root=$1 f
    if [ -f "$root/tools.toml" ]; then
        awk '
            /^\[[^]]+\][[:space:]]*$/ { sec = $0; gsub(/[][[:space:]]/, "", sec); next }
            sec != "" && $1 == "version" {
                v = $0; sub(/^[^"]*"/, "", v); sub(/".*$/, "", v)
                printf "tools.toml:%d\t%s\t%s\n", NR, sec, v
            }
        ' "$root/tools.toml"
    fi
    while IFS= read -r f; do
        awk -v src="${f#"$root"/}" '
            /^(PMAT|BASHRS|PV|FORJAR)_PIN=/ {
                t = $0; sub(/_PIN=.*/, "", t); t = tolower(t)
                v = $0; sub(/^[^=]*=/, "", v); gsub(/["'"'"']/, "", v); sub(/[[:space:]].*$/, "", v)
                # ${ENV:-default}: a digit default is the version; any other default is a
                # pin FILE the gate reads (the fleet pv.pin), resolved by run_check.
                if (v ~ /^\$\{[A-Za-z_][A-Za-z0-9_]*:-.*\}$/) {
                    e = v; sub(/^\$\{/, "", e); sub(/:-.*/, "", e)
                    d = v; sub(/^[^:]*:-/, "", d); sub(/\}$/, "", d)
                    v = (d ~ /^[0-9]/) ? d : "file:" e ":" d
                }
                printf "%s:%d\t%s\t%s\n", src, NR, t, v
            }
        ' "$f"
    done < <(find "$root/scripts" -maxdepth 1 -type f -name '*.sh' 2> /dev/null | LC_ALL=C sort)
    while IFS= read -r f; do
        awk -v src="${f#"$root"/}" '
            /^# tool_version=/ {
                s = $0; sub(/^# tool_version=/, "", s)
                split(s, w, /[[:space:]]+/)
                if (w[1] == "none") next
                printf "%s:%d\t%s\t%s\n", src, NR, w[1], w[2]
            }
        ' "$f"
    done < <(find "$root/scripts" -maxdepth 1 -type f -name '*.txt' 2> /dev/null | LC_ALL=C sort)
}

# fetch CRATE VERSION -- prints "<http_code>\n<body>". http_code 000 = unreachable.
fetch() { # fetch CRATE VERSION
    local crate=$1 ver=$2 fx
    if [ -n "${GATE_PIN_FIXTURES:-}" ]; then
        fx="${GATE_PIN_FIXTURES}/${crate}-${ver}"
        if [ -f "${fx}.json" ]; then
            printf '200\n'
            cat "${fx}.json"
        elif [ -f "${fx}.404" ]; then
            printf '404\n'
        else
            printf '000\n'
        fi
        return 0
    fi
    local body code
    body="$(curl -sS --max-time 20 -A "$UA" -w '\n%{http_code}' \
        "https://crates.io/api/v1/crates/${crate}/${ver}" 2> /dev/null)" || body=$'\n000'
    code="${body##*$'\n'}"
    printf '%s\n%s\n' "$code" "${body%$'\n'*}"
}

# judge TOOL VERSION BASE_EPOCH -- one verdict line; rc 0 PASS, 1 FAIL, 2 UNMEASURED.
judge() { # judge TOOL VERSION BASE_EPOCH
    local tool=$1 ver=$2 base=$3 crate reply code body fields yanked created epoch
    crate="$(crate_for "$tool")"
    if [ -z "$crate" ]; then
        printf 'FAIL: %s %s -- no crates.io crate mapping for gate tool "%s"\n' "$tool" "$ver" "$tool"
        return 1
    fi
    if ! [[ "$ver" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        printf 'FAIL: %s "%s" -- not an exact released X.Y.Z (ranges, -rc and +dev are not releases)\n' "$tool" "$ver"
        return 1
    fi
    reply="$(fetch "$crate" "$ver")"
    code="${reply%%$'\n'*}"
    body="${reply#*$'\n'}"
    case "$code" in
        404)
            printf 'FAIL: %s %s -- %s %s is NOT on crates.io\n' "$tool" "$ver" "$crate" "$ver"
            return 1
            ;;
        200) ;;
        *)
            printf 'UNMEASURED: %s %s -- crates.io reply %s (not_measured, not PASS)\n' "$tool" "$ver" "$code"
            return 2
            ;;
    esac
    fields="$(printf '%s' "$body" | python3 -c 'import json,sys
v=json.load(sys.stdin)["version"]
print(str(bool(v["yanked"])).lower(), v["created_at"])' 2> /dev/null)" || fields=""
    yanked="${fields%% *}"
    created="${fields#* }"
    if [ -z "$fields" ] || ! epoch="$(date -u -d "$created" +%s 2> /dev/null)"; then
        printf 'UNMEASURED: %s %s -- unparseable crates.io reply (not_measured, not PASS)\n' "$tool" "$ver"
        return 2
    fi
    if [ "$yanked" = "true" ]; then
        printf 'FAIL: %s %s -- %s %s is YANKED\n' "$tool" "$ver" "$crate" "$ver"
        return 1
    fi
    if [ "$epoch" -ge "$base" ]; then
        printf 'FAIL: %s %s -- published %s, not before the base (%s): released inside this change, not N-1\n' \
            "$tool" "$ver" "$created" "$(date -u -d "@$base" +%Y-%m-%dT%H:%M:%SZ)"
        return 1
    fi
    printf 'PASS: %s %s -- %s published %s, before the base\n' "$tool" "$ver" "$crate" "$created"
    return 0
}

# base_epoch ROOT -- committer date (epoch) of the base commit; empty if not a git tree.
base_epoch() { # base_epoch ROOT
    local root=$1 base
    if [ -n "${GATE_PIN_BASE:-}" ]; then
        base="$GATE_PIN_BASE"
    elif git -C "$root" rev-parse -q --verify 'HEAD^2' > /dev/null 2>&1; then
        base='HEAD^1'
    elif ! base="$(git -C "$root" merge-base HEAD origin/main 2> /dev/null)"; then
        base='HEAD'
    fi
    git -C "$root" log -1 --format=%ct "$base" 2> /dev/null
}

# resolve_pin_file file:ENV:PATH -- prints the version in the pin file the gate reads
# ($ENV if set, else PATH with a leading $HOME expanded; nothing is eval'd). On failure
# prints the path it tried and returns 1.
resolve_pin_file() { # resolve_pin_file file:ENV:PATH
    local spec=${1#file:} env path v
    env=${spec%%:*}
    path=${spec#*:}
    [[ "$path" == '$HOME'* ]] && path="${HOME}${path#'$HOME'}"
    [ -n "${!env:-}" ] && path=${!env}
    if [ -r "$path" ] && v="$(head -n 1 -- "$path" | tr -d '[:space:]')" && [ -n "$v" ]; then
        printf '%s\n' "$v"
        return 0
    fi
    printf '%s\n' "$path"
    return 1
}

# run_check ROOT BASE_EPOCH -- rc 0 (no FAIL), 1 (a FAIL).
run_check() { # run_check ROOT BASE_EPOCH
    local root=$1 base=$2 src tool ver n=0 fail=0 unm=0 rc
    while IFS=$'\t' read -r src tool ver; do
        n=$((n + 1))
        if [[ "$ver" == file:* ]]; then
            if ! ver="$(resolve_pin_file "$ver")"; then
                printf '%s: UNMEASURED: %s pin file %s unreadable on this host (not_measured, not PASS)\n' "$src" "$tool" "$ver"
                unm=$((unm + 1))
                continue
            fi
            src="$src (via pin file)"
        fi
        printf '%s: ' "$src"
        judge "$tool" "$ver" "$base"
        rc=$?
        [ "$rc" -eq 1 ] && fail=$((fail + 1))
        [ "$rc" -eq 2 ] && unm=$((unm + 1))
    done < <(collect_pins "$root")
    if [ "$n" -eq 0 ]; then
        printf 'FAIL: no gate-tool pins found under %s -- the collector is blind, not the tree clean\n' "$root"
        return 1
    fi
    printf 'SUMMARY: %d pin(s): %d FAIL, %d UNMEASURED\n' "$n" "$fail" "$unm"
    [ "$fail" -eq 0 ]
}

self_test() {
    local tmp pass=0 bad=0 out rc base
    tmp="$(mktemp -d)" || return 1
    # shellcheck disable=SC2064
    trap "rm -rf -- '${tmp:?}'" EXIT
    mkdir -p "$tmp/fx" "$tmp/tree/scripts"
    local old='2026-09-01T00:00:00.000000Z' new='2026-10-01T00:00:00.000000Z'
    rel() { printf '{"version":{"num":"%s","yanked":%s,"created_at":"%s"}}' "$2" "$3" "$4" > "$tmp/fx/$1-$2.json"; }
    rel pmat 3.41.1 false "$old"
    rel pmat 3.37.0 false "$old"
    rel bashrs 7.4.1 false "$old"
    rel aprender-contracts-cli 0.69.2 false "$old"
    rel forjar 1.33.0 false "$old"
    rel pmat 3.40.0 true "$old"
    rel pmat 3.42.0 false "$new"
    printf 'not json' > "$tmp/fx/bashrs-7.0.0.json"
    : > "$tmp/fx/pmat-99.0.0.404"
    base="$(date -u -d '2026-09-28T12:00:00Z' +%s)"
    export GATE_PIN_FIXTURES="$tmp/fx"

    # tool | version | expected rc | expected prefix | why
    local cases=(
        "pmat|3.41.1|0|PASS|published before base"
        "bashrs|7.4.1|0|PASS|published before base"
        "pv|0.69.2|0|PASS|pv maps to aprender-contracts-cli"
        "forjar|1.33.0|0|PASS|forjar maps to forjar"
        "pmat|99.0.0|1|FAIL|PLANTED RED: not on crates.io"
        "pmat|3.42.0|1|FAIL|published after base (same-PR release)"
        "pmat|3.40.0|1|FAIL|yanked"
        "ruff|1.0.0|1|FAIL|unmapped gate tool"
        "pmat|3.41|1|FAIL|not X.Y.Z"
        "pmat|^3.41.1|1|FAIL|a range is not a release"
        "pmat|3.43.0-rc.1|1|FAIL|prerelease is not released"
        "pv|0.70.0+dev.abc|1|FAIL|dev build is not released"
        "pmat||1|FAIL|empty version"
        "forjar|9.9.9|2|UNMEASURED|registry unreachable"
        "bashrs|7.0.0|2|UNMEASURED|unparseable reply"
    )
    local c tool ver want wpre why
    for c in "${cases[@]}"; do
        IFS='|' read -r tool ver want wpre why <<< "$c"
        out="$(judge "$tool" "$ver" "$base")"
        rc=$?
        if [ "$rc" -eq "$want" ] && [[ "$out" == "$wpre:"* ]]; then
            pass=$((pass + 1))
        else
            bad=$((bad + 1))
            printf 'self-test FAIL [%s %s: %s]: want rc=%s %s, got rc=%s: %s\n' "$tool" "$ver" "$why" "$want" "$wpre" "$rc" "$out"
        fi
    done

    # Collector: every pin surface is read; `none` headers and non-pin lines are not.
    printf '[pmat]\nversion = "3.41.1"\n\n[bashrs]\nversion = "7.4.1"\n' > "$tmp/tree/tools.toml"
    printf '#!/usr/bin/env bash\nPMAT_PIN="3.37.0"\n  PMAT_PIN="0.0.1"\necho PV_PIN=0.0.2\n' > "$tmp/tree/scripts/pmat_bin.sh"
    printf 'PV_PIN="${FLEET_PV_PIN:-$HOME/.config/fleet/pv.pin}"\nFORJAR_PIN="${X_FORJAR:-1.33.0}"\n' > "$tmp/tree/scripts/pv_gate.sh"
    printf '# tool_version=pmat 3.41.1\nx\n' > "$tmp/tree/scripts/a_baseline.txt"
    printf '# tool_version=none (grep)\n' > "$tmp/tree/scripts/b_baseline.txt"
    printf '# tool_version=bashrs 7.4.1\n' > "$tmp/tree/scripts/c_baseline.txt"
    local want_pins got_pins
    want_pins=$'tools.toml:2\tpmat\t3.41.1\ntools.toml:5\tbashrs\t7.4.1\nscripts/pmat_bin.sh:2\tpmat\t3.37.0\nscripts/pv_gate.sh:1\tpv\tfile:FLEET_PV_PIN:$HOME/.config/fleet/pv.pin\nscripts/pv_gate.sh:2\tforjar\t1.33.0\nscripts/a_baseline.txt:1\tpmat\t3.41.1\nscripts/c_baseline.txt:1\tbashrs\t7.4.1'
    got_pins="$(collect_pins "$tmp/tree")"
    if [ "$got_pins" = "$want_pins" ]; then
        pass=$((pass + 1))
    else
        bad=$((bad + 1))
        printf 'self-test FAIL [collector]: want\n%s\ngot\n%s\n' "$want_pins" "$got_pins"
    fi

    # Pin files: the env override is read; a planted unpublished pv in it is RED;
    # an unreadable one is UNMEASURED, never PASS.
    printf '0.69.2\n' > "$tmp/pv.pin"
    export FLEET_PV_PIN="$tmp/pv.pin"
    # End to end: the clean tree is GREEN; the planted RED in each surface turns it RED.
    local surf
    if run_check "$tmp/tree" "$base" > /dev/null; then pass=$((pass + 1)); else
        bad=$((bad + 1))
        printf 'self-test FAIL [e2e clean tree]: expected GREEN\n'
    fi
    for surf in tools.toml scripts/pmat_bin.sh scripts/a_baseline.txt; do
        cp -a "$tmp/tree" "$tmp/red"
        case "$surf" in
            tools.toml) sed -i 's/3\.41\.1/99.0.0/' "$tmp/red/$surf" ;;
            scripts/pmat_bin.sh) sed -i 's/^PMAT_PIN="3\.37\.0"/PMAT_PIN="99.0.0"/' "$tmp/red/$surf" ;;
            *) sed -i 's/pmat 3\.41\.1/pmat 99.0.0/' "$tmp/red/$surf" ;;
        esac
        out="$(run_check "$tmp/red" "$base")"
        rc=$?
        if [ "$rc" -eq 1 ] && grep -q "^${surf}:[0-9]*: FAIL: pmat 99.0.0 -- pmat 99.0.0 is NOT on crates.io" <<< "$out"; then
            pass=$((pass + 1))
        else
            bad=$((bad + 1))
            printf 'self-test FAIL [planted RED in %s]: rc=%s\n%s\n' "$surf" "$rc" "$out"
        fi
        rm -rf -- "${tmp:?}/red"
    done
    : > "$tmp/fx/aprender-contracts-cli-0.70.0.404"
    printf '0.70.0\n' > "$tmp/pv.pin"
    out="$(run_check "$tmp/tree" "$base")"
    rc=$?
    if [ "$rc" -eq 1 ] && grep -q '^scripts/pv_gate.sh:1 (via pin file): FAIL: pv 0.70.0 -- aprender-contracts-cli 0.70.0 is NOT on crates.io' <<< "$out"; then
        pass=$((pass + 1))
    else
        bad=$((bad + 1))
        printf 'self-test FAIL [planted RED in pv pin file]: rc=%s\n%s\n' "$rc" "$out"
    fi
    export FLEET_PV_PIN="$tmp/absent.pin"
    out="$(run_check "$tmp/tree" "$base")"
    rc=$?
    if [ "$rc" -eq 0 ] && grep -q '^scripts/pv_gate.sh:1: UNMEASURED: pv pin file .*absent.pin unreadable' <<< "$out"; then
        pass=$((pass + 1))
    else
        bad=$((bad + 1))
        printf 'self-test FAIL [absent pv pin file]: rc=%s\n%s\n' "$rc" "$out"
    fi
    unset FLEET_PV_PIN
    # A tree with no pins is RED, never a vacuous GREEN.
    mkdir -p "$tmp/empty/scripts"
    if run_check "$tmp/empty" "$base" > /dev/null; then
        bad=$((bad + 1))
        printf 'self-test FAIL [empty tree]: expected RED\n'
    else pass=$((pass + 1)); fi
    # UNMEASURED alone does not fail the run, and is printed.
    printf '[forjar]\nversion = "9.9.9"\n' > "$tmp/empty/tools.toml"
    out="$(run_check "$tmp/empty" "$base")"
    rc=$?
    if [ "$rc" -eq 0 ] && grep -q '^SUMMARY: 1 pin(s): 0 FAIL, 1 UNMEASURED$' <<< "$out"; then
        pass=$((pass + 1))
    else
        bad=$((bad + 1))
        printf 'self-test FAIL [unmeasured-only]: rc=%s\n%s\n' "$rc" "$out"
    fi

    unset GATE_PIN_FIXTURES
    if [ "$bad" -ne 0 ]; then
        printf 'self-test FAILED: %d of %d case(s)\n' "$bad" "$((pass + bad))"
        return 1
    fi
    printf 'self-test OK: %d case(s).\n' "$pass"
}

case "${1:-}" in
    --self-test)
        self_test
        exit $?
        ;;
    -h | --help)
        usage
        printf '  --self-test   run the offline case table (includes the planted RED pmat 99.0.0)\n' >&2
        exit 0
        ;;
    "") ;;
    *)
        usage
        exit 2
        ;;
esac

command -v python3 > /dev/null 2>&1 || {
    printf 'FAIL: python3 not on PATH -- cannot read crates.io replies\n'
    exit 1
}
ROOT="${GATE_PIN_ROOT:-$REPO_ROOT}"
BASE_EPOCH="$(base_epoch "$ROOT")"
if [ -z "$BASE_EPOCH" ]; then
    printf 'FAIL: cannot resolve the base commit under %s (GATE_PIN_BASE=%s)\n' "$ROOT" "${GATE_PIN_BASE:-}"
    exit 1
fi
run_check "$ROOT" "$BASE_EPOCH"
