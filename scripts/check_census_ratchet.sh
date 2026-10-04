#!/usr/bin/env bash
# check_census_ratchet.sh — CENSUS-RATCHET (#4709, operator C303): entity-bound contracts
# never drop, and rise every release.
#
# Operator C303, verbatim: "you need to include a ratchet for every release to close the -
# pv census: counts contracts. Only 48 of 1,889 are tied to a real thing so far, which is
# the biggest gap."
#
# THE COUNT. `pv census --format json` (Rust, crates/aprender-contracts-cli/src/commands/
# census.rs). bound = by_anchoring.class + by_anchoring.instance (type-wide + one named
# thing); unbound = by_anchoring.unanchored. Baseline: 48 of 1889 bound at origin/main
# 316dee2cd4 (pv 0.70.1, 2026-10-04).
#
# TWO RULES, BOTH MEASURED AT BOTH ENDS, NEVER STORED (#3569 / never-worse = head vs base):
#   PR       head (working tree) vs base (scripts/lib/resolve_base.sh, extracted by git
#            archive), ONE pv binary in ONE run:  bound may not fall, and the unbound SHARE
#            may not rise (integer cross-multiplication, no float).
#   RELEASE  --release V --commit SHA: bound(SHA) >= bound(previous release tag) + STEP,
#            STEP = min(100, unbound(previous tag)). 0.70.2 is the baseline and pays no step;
#            STEP_FROM below is a DECISION (the operator's), not a measurement.
#            Called by scripts/release/autopilot.sh step `readiness`, before any tag exists.
#
# NEVER VACUOUS. No pv / no `census`, a non-zero exit, JSON that does not parse, 0 contracts,
# parse errors, or anchoring that does not sum to n_parsed: NOT_MEASURED, exit 2. Not a pass.
#
#   bash scripts/check_census_ratchet.sh                       # PR rule, base resolved
#   bash scripts/check_census_ratchet.sh --base REF            # PR rule, explicit base
#   bash scripts/check_census_ratchet.sh --release V --commit SHA [--prev TAG]
#   bash scripts/check_census_ratchet.sh --print [REF]         # one measurement
#   bash scripts/check_census_ratchet.sh --self-test           # red/green on the real pv
#
# EXIT 0 pass · 1 ratchet RED · 2 NOT_MEASURED · 3 caller error
# Pmat-Ticket: PMAT-4709
set -euo pipefail

PROG=check_census_ratchet
REPO_ROOT="${CENSUS_RATCHET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_FROM=0.71.0   # operator C303 via #4709: 0.70.2 is baseline only
STEP_MAX=100
PV_IDENTITY='(aprender provable-contracts verifier)'
TMPS=()
cleanup() { local d; for d in "${TMPS[@]}"; do rm -rf "${d:?}"; done; }
trap cleanup EXIT

usage() { sed -n '26,31p' "${BASH_SOURCE[0]}" >&2; exit 3; }
nm() { printf 'NOT_MEASURED  %s\n' "$*"; exit 2; }

# resolve_pv -> a pv that IS the aprender verifier and lists `census`. Fleet paths first
# (the released pin, N-1; never a bare PATH pv). The SAME binary measures both ends.
resolve_pv() {
    local c ver help_out cands
    cands="${CENSUS_RATCHET_PV:-${FLEET_PV_BIN:-/opt/fleet-bin/bin/pv:$HOME/.cargo/bin/pv}}"
    local IFS=:
    for c in $cands; do
        [ -n "$c" ] && [ -x "$c" ] || continue
        ver="$("$c" --version 2>/dev/null || true)"
        [[ "$ver" == *"$PV_IDENTITY"* ]] || continue
        help_out="$("$c" --help 2>&1 || true)"
        grep -qE '^[[:space:]]+census[[:space:]]' <<<"$help_out" || continue
        PV="$c"; PV_VERSION="${ver%%$'\n'*}"; return 0
    done
    return 1
}

# measure DIR -> sets N UNB BOUND, or NOT_MEASURED (exit 2) naming why.
measure() {
    local dir=$1 out rc vals pe q cls ins v
    rc=0; out="$("$PV" census --format json "$dir" 2>&1)" || rc=$?
    [ "$rc" -eq 0 ] || nm "pv census $dir exited $rc: ${out:0:200}"
    vals="$(jq -er '[.n_parsed, .n_parse_errors, (.quarantined_n // 0), .by_anchoring.unanchored, .by_anchoring.class, .by_anchoring.instance] | map(tostring) | join(" ")' <<<"$out" 2>/dev/null)" \
        || nm "pv census $dir: output is not a census document: ${out:0:200}"
    read -r N pe q UNB cls ins <<<"$vals"
    for v in "$N" "$pe" "$q" "$UNB" "$cls" "$ins"; do
        [[ "$v" =~ ^[0-9]+$ ]] || nm "pv census $dir: non-integer field '$v'"
    done
    [ "$N" -gt 0 ] || nm "pv census $dir counted 0 contracts"
    [ "$pe" -eq 0 ] || nm "pv census $dir: $pe parse error(s), the count is partial"
    BOUND=$((cls + ins))
    [ $((UNB + BOUND)) -eq "$N" ] || nm "pv census $dir: unanchored+class+instance=$((UNB + BOUND)) != n_parsed=$N"
}

# measure_ref REF -> the contracts/ of a commit, via git archive into a temp dir.
measure_ref() {
    local ref=$1 t
    t="$(mktemp -d "${TMPDIR:-/tmp}/census-ratchet.XXXXXX")"; TMPS+=("$t")
    git -C "$REPO_ROOT" archive "$ref" contracts 2>/dev/null | tar -x -C "$t" 2>/dev/null \
        || nm "git archive $ref contracts failed"
    measure "$t/contracts"
}

row() { printf 'CENSUS %-8s %-14s n=%s bound=%s unbound=%s\n' "$1" "$2" "$N" "$BOUND" "$UNB"; }

pr_rule() {
    local base=$1 bn bb bu hn hb hu rc=0
    measure_ref "$base"; bn=$N; bb=$BOUND; bu=$UNB; row base "${base:0:12}"
    measure "$REPO_ROOT/contracts"; hn=$N; hb=$BOUND; hu=$UNB; row head worktree
    printf 'pv: %s (%s)\n' "$PV" "$PV_VERSION"
    if [ "$hb" -lt "$bb" ]; then
        printf 'FAIL  bound FELL %s -> %s: a contract lost its entity: (CENSUS-RATCHET #4709)\n' "$bb" "$hb"; rc=1
    fi
    if [ $((hu * bn)) -gt $((bu * hn)) ]; then
        printf 'FAIL  unbound share ROSE %s/%s -> %s/%s: a new contract needs entity: (type-wide or ref), or bind another with it (#4709)\n' "$bu" "$bn" "$hu" "$hn"; rc=1
    fi
    [ "$rc" -eq 0 ] && printf 'PASS  bound %s -> %s, unbound share %s/%s -> %s/%s\n' "$bb" "$hb" "$bu" "$bn" "$hu" "$hn"
    return "$rc"
}

# prev_tag V -> the greatest final release tag vX.Y.Z below V.
prev_tag() {
    { git -C "$REPO_ROOT" tag -l 'v[0-9]*.[0-9]*.[0-9]*' | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true; printf 'v%s\n' "$1"; } \
        | sort -uV | awk -v v="v$1" '$0 == v { print p; exit } { p = $0 }'
}
version_ge() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -n 1)" = "$2" ]; }

release_rule() {
    local v=$1 commit=$2 prev=$3 pb pu step need note=""
    [ -n "$prev" ] || prev="$(prev_tag "$v")"
    [ -n "$prev" ] || nm "no release tag below v$v to compare against"
    git -C "$REPO_ROOT" rev-parse -q --verify "$prev^{commit}" >/dev/null || nm "previous tag $prev is not fetched"
    measure_ref "$prev"; pb=$BOUND; pu=$UNB; row prev "$prev"
    measure_ref "$commit"; row release "v$v@${commit:0:9}"
    printf 'pv: %s (%s)\n' "$PV" "$PV_VERSION"
    step=0
    if version_ge "$v" "$STEP_FROM"; then step=$(( pu < STEP_MAX ? pu : STEP_MAX )); else note=" [baseline train: the step starts at $STEP_FROM]"; fi
    need=$((pb + step))
    if [ "$BOUND" -lt "$need" ]; then
        printf 'FAIL  v%s bound=%s < %s (bound(%s)=%s + STEP %s = min(%s, unbound %s)): bind %s more contract(s) before the tag\n' \
            "$v" "$BOUND" "$need" "$prev" "$pb" "$step" "$STEP_MAX" "$pu" "$((need - BOUND))"
        return 1
    fi
    printf 'PASS  v%s bound=%s >= %s (bound(%s)=%s + STEP %s)%s\n' "$v" "$BOUND" "$need" "$prev" "$pb" "$step" "$note"
}

self_test() {
    local t r fails=0 got stub c1 c2 c3 case
    resolve_pv || nm "self-test needs a pv with census (none of: ${CENSUS_RATCHET_PV:-/opt/fleet-bin/bin/pv, \$HOME/.cargo/bin/pv})"
    t="$(mktemp -d "${TMPDIR:-/tmp}/census-ratchet-st.XXXXXX")"; r="$t/repo"; TMPS+=("$t")
    mkdir -p "$r/contracts"; git -C "$r" init -q; git -C "$r" config user.email st@x; git -C "$r" config user.name st; git -C "$r" config core.hooksPath /dev/null
    mk() { printf 'metadata:\n  version: "1.0.0"\n  description: %s\n%s' "$1" "${2:-}" > "$r/contracts/$1.yaml"; }
    local ent_i=$'entity:\n  type: binary\n  ref: apr\n' ent_c=$'entity:\n  type: kernel\n'
    mk a "$ent_i"; mk b "$ent_c"; mk c; mk d
    git -C "$r" add -A; git -C "$r" commit -qm base; git -C "$r" tag v0.70.1
    expect() { # expect <label> <want-rc> <args...>
        local label=$1 want=$2; shift 2
        got=0; CENSUS_RATCHET_ROOT="$r" CENSUS_RATCHET_PV="${CENSUS_RATCHET_PV:-$PV}" bash "$SCRIPT_DIR/$PROG.sh" "$@" >"$t/out" 2>&1 || got=$?
        if [ "$got" = "$want" ]; then printf 'ok    %-60s rc=%s\n' "$label" "$got"
        else printf 'BAD   %-60s rc=%s want %s\n' "$label" "$got" "$want"; sed 's/^/      /' "$t/out"; fails=$((fails + 1)); fi
    }
    expect "unchanged tree: GREEN" 0 --base HEAD
    # THE MUTATION (#4709 red/green): unbind ONE contract.
    mk a; expect "MUTANT unbind one contract: RED" 1 --base HEAD
    mk a "$ent_i"; mk e; expect "add an unbound contract (share rises): RED" 1 --base HEAD
    mk e "$ent_c"; expect "add a bound contract: GREEN" 0 --base HEAD
    rm "${r:?}/contracts/e.yaml"; mk c "$ent_i"; expect "bind an existing contract: GREEN" 0 --base HEAD
    mk c; rm "${r:?}/contracts/b.yaml"; expect "delete a bound contract: RED" 1 --base HEAD
    git -C "$r" checkout -q -- contracts
    # Release rule. v0.70.2 is baseline (step 0); v0.71.0 owes min(100, unbound 2) = 2.
    git -C "$r" commit -q --allow-empty -m rel; c1="$(git -C "$r" rev-parse HEAD)"
    expect "release 0.70.2 equal to prev (baseline, step 0): GREEN" 0 --release 0.70.2 --commit "$c1"
    git -C "$r" tag v0.70.2
    mk c "$ent_i"; git -C "$r" commit -qam bind1; c2="$(git -C "$r" rev-parse HEAD)"
    expect "release 0.71.0 short of the step (+1 of 2): RED" 1 --release 0.71.0 --commit "$c2"
    mk d "$ent_c"; git -C "$r" commit -qam bind2; c3="$(git -C "$r" rev-parse HEAD)"
    expect "release 0.71.0 pays the step (+2 = min(100, 2)): GREEN" 0 --release 0.71.0 --commit "$c3"
    expect "release with no tag below it: NOT_MEASURED" 2 --release 0.0.1 --commit "$c3"
    # Never vacuous: stub pvs that carry the identity and census, and lie.
    for case in garbage zero exit parse_err sum; do
        stub="$t/pv-$case"
        {
            printf '#!/usr/bin/env bash\n'
            printf 'case "$1" in --version) echo "pv 9 %s";; --help) printf "  census  x\\n";; *)\n' "$PV_IDENTITY"
            case $case in
                garbage)   printf 'echo "not json";;\n' ;;
                zero)      printf 'echo "{\\"n_parsed\\":0,\\"n_parse_errors\\":0,\\"by_anchoring\\":{\\"unanchored\\":0,\\"class\\":0,\\"instance\\":0}}";;\n' ;;
                exit)      printf 'echo "decline: 0 contracts"; exit 2;;\n' ;;
                parse_err) printf 'echo "{\\"n_parsed\\":3,\\"n_parse_errors\\":1,\\"by_anchoring\\":{\\"unanchored\\":1,\\"class\\":1,\\"instance\\":1}}";;\n' ;;
                sum)       printf 'echo "{\\"n_parsed\\":9,\\"n_parse_errors\\":0,\\"by_anchoring\\":{\\"unanchored\\":1,\\"class\\":1,\\"instance\\":1}}";;\n' ;;
            esac
            printf 'esac\n'
        } > "$stub"; chmod +x "$stub"
        CENSUS_RATCHET_PV="$stub" expect "stub pv '$case': NOT_MEASURED (never a pass)" 2 --base HEAD
    done
    CENSUS_RATCHET_PV="$t/absent" expect "no pv at all: NOT_MEASURED" 2 --base HEAD
    printf '%s self-test: %s\n' "$PROG" "$([ "$fails" = 0 ] && echo PASS || echo "FAIL ($fails)")"
    [ "$fails" = 0 ]
}

mode=pr base="" version="" commit="" prev="" print_ref=""
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test) self_test; exit $? ;;
        -h|--help) usage ;;
        --base) base="${2:?}"; shift ;;
        --release) mode=release; version="${2:?}"; shift ;;
        --commit) commit="${2:?}"; shift ;;
        --prev) prev="${2:?}"; shift ;;
        --print) mode=print; if [ -n "${2:-}" ] && [ "${2:0:1}" != - ]; then print_ref=$2; shift; fi ;;
        *) usage ;;
    esac
    shift
done

resolve_pv || nm "no pv with a census subcommand (looked at: ${CENSUS_RATCHET_PV:-/opt/fleet-bin/bin/pv, \$HOME/.cargo/bin/pv})"
case "$mode" in
    print)
        if [ -n "$print_ref" ]; then measure_ref "$print_ref"; row print "${print_ref:0:12}"; else measure "$REPO_ROOT/contracts"; row print worktree; fi
        printf 'pv: %s (%s)\n' "$PV" "$PV_VERSION" ;;
    release)
        [[ "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { printf '%s: --release wants X.Y.Z, got %s\n' "$PROG" "$version" >&2; exit 3; }
        [ -n "$commit" ] || { printf '%s: --release needs --commit SHA\n' "$PROG" >&2; exit 3; }
        release_rule "$version" "$commit" "$prev" ;;
    pr)
        if [ -z "$base" ]; then
            # shellcheck source=scripts/lib/resolve_base.sh
            . "$SCRIPT_DIR/lib/resolve_base.sh" || exit 2
            resolve_base HEAD || nm "no comparand could be named for HEAD (never the tree against itself)"
            base="$BASE_REF"; printf 'base: %s (%s)\n' "${BASE_REF:0:12}" "$BASE_HOW"
        fi
        pr_rule "$base" ;;
esac
