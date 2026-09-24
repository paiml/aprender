#!/usr/bin/env bash
# rc_publish_gate.sh — can this release candidate reach crates.io? (#4287, RC-DOGFOOD-001 §4.5)
#
#   bash scripts/release/rc_publish_gate.sh ROOT     # judge the checked-out rc tree at ROOT
#   bash scripts/release/rc_publish_gate.sh --self-test
#
# WHY AT THE RC. v0.68.0 and v0.69.0 never reached crates.io. v0.68.0 went GitHub-only
# when clean-room B2 went red AT PUBLISH: aprender-core's lib tests named `entrenar`, a
# path-only dev-dep that `cargo publish` deletes (#3425). The fix was a whole hotfix
# cycle. The same defects are visible on the commit, long before the tag, so the rc cut
# (rc-cut.yml, #4285) runs this before it writes the tag: red here means no rc.
#
# THREE CHECKS, each the tool that already owns its rule, pointed at ROOT:
#   graph     check_publish_preflight.sh --graph-only: R6, no versioned sibling
#             dev-dependency on a cycle (PMAT-955, #3468). R1/R3/R4/R5/R7 describe the
#             upload itself and stay at publish.
#   pathonly  check_pathonly_devdeps_unused_in_src.sh --rc-gate ROOT: src/ uses no
#             dev-dep the publish strips, judged against THIS checkout's triage (the
#             default branch), never the rc tree's own baseline, which masked #3425.
#   package   `cargo package --no-verify --locked` over every publishable member in one
#             invocation, so unpublished sibling versions resolve through cargo's local
#             overlay: the cascade's manifests, includes and version pins, without an
#             upload. Every selected crate must print a Packaged line.
#
# ROOT is a clean checkout of the rc commit. The checks run from THIS file's checkout,
# so a release branch cannot weaken the gate that judges it.
#
# EXIT  0 all three green · 1 a check found a publish defect · 2 a check could not measure
#       (a defect found outranks a check that could not run: 1 wins over 2).
set -uo pipefail
PROG=rc_publish_gate
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS="$(cd -- "$HERE/.." && pwd)"

# Pure: the verdict of the three check exit codes. Prints `green`, `red` or `env`.
rpg_verdict() {
    local rc red=0 env=0
    for rc in "$@"; do
        case "$rc" in
            0) ;;
            1) red=1 ;;
            *) env=1 ;;
        esac
    done
    if [ "$red" = 1 ]; then echo red
    elif [ "$env" = 1 ] || [ "$#" -ne 3 ]; then echo env
    else echo green
    fi
}

self_test() {
    local fail=0 got want args why
    echo "$PROG self-test: case table"
    while IFS='|' read -r want args why; do
        # shellcheck disable=SC2086  # args is a space-separated list of exit codes on purpose
        got=$(rpg_verdict $args)
        if [ "$got" = "$want" ]; then echo "  ok   $why"; else echo "  FAIL $why: wanted $want, got $got"; fail=1; fi
    done <<'EOF'
green|0 0 0|all three green
red|1 0 0|the graph refused
red|0 1 0|the path-only rule refused (the #3425 shape)
red|0 0 1|the package walk refused
env|2 0 0|a check that could not measure is not a pass
red|1 2 0|a defect found outranks a check that could not run
env|0 0 101|an unexpected exit code is not a pass
env|0 0|a missing check is not a pass
EOF
    if [ "$fail" -eq 0 ]; then echo "$PROG self-test: PASS"; return 0; fi
    echo "$PROG self-test: FAIL"; return 1
}

# package_walk ROOT -> 0 every publishable member packaged, 1 cargo refused, 2 cannot measure
package_walk() {
    local root=$1 log n got rc=0
    local -a sel=()
    mapfile -t sel < <(cargo metadata --no-deps --offline --format-version 1 --manifest-path "$root/Cargo.toml" 2>/dev/null \
        | python3 -c 'import json,sys; print("\n".join(sorted(map(lambda p: p["name"], filter(lambda p: p.get("publish") != [], json.load(sys.stdin)["packages"])))))')
    n=${#sel[@]}
    if [ "$n" -eq 0 ] || [ -z "${sel[0]}" ]; then
        echo "ENV   package: cargo metadata names no publishable member in $root"
        return 2
    fi
    local -a args=()
    local c
    for c in "${sel[@]}"; do args+=(-p "$c"); done
    log=$(mktemp) || return 2
    (cd -- "$root" && cargo package "${args[@]}" --no-verify --locked) > "$log" 2>&1 || rc=$?
    got=$(grep -cE '^ +Packaged ' "$log")
    if [ "$rc" -eq 0 ] && [ "$got" -eq "$n" ]; then
        echo "ok    package: $got/$n publishable members packaged (cargo package --no-verify --locked)"
        rm -f -- "$log"; return 0
    fi
    echo "FAIL  package: cargo package exited $rc with $got/$n members packaged:"
    grep -E '^(error|Caused by)' -A 4 "$log" | head -n 20 | sed 's/^/        /'
    rm -f -- "$log"; return 1
}

main() {
    case "${1:-}" in
        --self-test) self_test; return ;;
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; return 0 ;;
        '') echo "$PROG: usage: ROOT | --self-test" >&2; return 2 ;;
    esac
    local root=$1 g p k verdict
    [ -f "$root/Cargo.toml" ] || { echo "$PROG: $root has no Cargo.toml" >&2; return 2; }
    root="$(cd -- "$root" && pwd)"
    echo "=== $PROG: can $root reach crates.io? ==="
    PUBLISH_PREFLIGHT_ROOT="$root" bash "$SCRIPTS/check_publish_preflight.sh" --graph-only; g=$?
    bash "$SCRIPTS/check_pathonly_devdeps_unused_in_src.sh" --rc-gate "$root"; p=$?
    package_walk "$root"; k=$?
    verdict=$(rpg_verdict "$g" "$p" "$k")
    echo "$PROG: graph=$g pathonly=$p package=$k -> $verdict"
    case "$verdict" in
        green) return 0 ;;
        red) return 1 ;;
        *) return 2 ;;
    esac
}

main "$@"
