#!/usr/bin/env bash
# check_bin_names_aprender.sh -- every shipped [[bin]] is named aprender-* (#4430).
#
# WHY. The 0.70.0 all-binaries gate ships every workspace [[bin]] (the nightly
# derives the list from `cargo metadata`, #4189). Before #4430, eleven of them
# carried pre-monorepo names -- alimentar, presentar, ptop, score, simular,
# trueno-rag, trueno-zram, verificar, apr-qa, apr-qa-readme-sync,
# apr-corpus-ingest -- so a fleet box's PATH mixed three naming schemes, and
# `score` and `ptop` claimed generic names that other tools also use.
#
# THE RULE. A bin target, in the root workspace OR the `exclude`d facade
# workspace (crates/facades), passes only if it is:
#   - `apr` or `pv`, the two user-facing names (apr-mono-binary-rule-v1); or
#   - aprender-<word>[-<word>...], lowercase; or
#   - listed with its package in scripts/bin_names_pending_fold.txt, meaning it
#     is being folded into `apr` and will not be renamed.
# A pending-fold row whose bin is gone is RED too, so that list only shrinks.
#
# Bins come from `cargo metadata`, never from a grep of [[bin]]: an
# auto-discovered src/bin/*.rs has no stanza (verificar was one).
#
#   bash scripts/check_bin_names_aprender.sh              # check the tree
#   bash scripts/check_bin_names_aprender.sh --list       # the shipped bin set (asset list)
#   bash scripts/check_bin_names_aprender.sh --self-test  # case table + mutants
#
# Exit: 0 pass · 1 a bin is misnamed, a row is stale, or the scan is vacuous · 2 ENV.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="${REPO_ROOT}/scripts/lib/bin_name_prefix.py"
PENDING="${REPO_ROOT}/scripts/bin_names_pending_fold.txt"
FACADE_WS="${REPO_ROOT}/crates/facades"

# meta <file> <bin:package>... -- a minimal cargo-metadata document.
meta() {
    local f=$1 b sep=''
    shift
    {
        printf '{"packages":['
        for b in "$@"; do
            printf '%s{"name":"%s","manifest_path":"/x/%s/Cargo.toml","targets":[{"name":"%s","kind":["bin"]},{"name":"lib","kind":["lib"]}]}' \
                "$sep" "${b#*:}" "${b#*:}" "${b%%:*}"
            sep=,
        done
        printf ']}\n'
    } > "$f"
}

self_test() {
    local d n=0 fail=0 lib=$LIB
    d=$(mktemp -d) || return 2
    # shellcheck disable=SC2064
    trap "rm -rf \"${d:?}\"" RETURN
    printf 'apr-qa aprender-qa-cli\n' > "$d/pend"
    : > "$d/pend0"
    meta "$d/fac.json" pv:provable-contracts-cli
    meta "$d/good.json" apr:apr-cli apr:aprender pv:aprender-contracts-cli aprender-data:aprender-data apr-qa:aprender-qa-cli
    meta "$d/rogue.json" apr:apr-cli alimentar:aprender-data apr-qa:aprender-qa-cli
    meta "$d/aprfoo.json" apr:apr-cli apr-foo:apr-cli apr-qa:aprender-qa-cli
    meta "$d/bare.json" apr:apr-cli aprender-:x apr-qa:aprender-qa-cli
    meta "$d/nohyph.json" apr:apr-cli aprenderx:x apr-qa:aprender-qa-cli
    meta "$d/upper.json" apr:apr-cli Aprender-Data:x apr-qa:aprender-qa-cli
    meta "$d/moved.json" apr:apr-cli apr-qa:some-other-crate
    meta "$d/nofold.json" apr:apr-cli aprender-data:aprender-data
    meta "$d/fac_rogue.json" pv:provable-contracts-cli trueno-rag:facade
    meta "$d/noapr.json" aprender-data:aprender-data apr-qa:aprender-qa-cli
    printf 'not json\n' > "$d/junk.json"

    row() { # row <want rc> <needle|-> <label> <pending> <label=doc>...
        local want=$1 needle=$2 label=$3 pend=$4 out rc
        shift 4
        n=$((n + 1))
        out=$(python3 "$lib" "$pend" "$@" 2>&1); rc=$?
        if [ "$rc" != "$want" ]; then
            printf 'FAIL %-66s rc=%s want %s\n%s\n' "$label" "$rc" "$want" "$out"; fail=1
        elif [ "$needle" != - ] && ! grep -qF -- "$needle" <<< "$out"; then
            printf 'FAIL %-66s did not name %s\n%s\n' "$label" "$needle" "$out"; fail=1
        else printf 'ok   %s\n' "$label"; fi
    }
    rows() {
        row 0 - 'aprender-*, apr, pv and a pending-fold bin pass' "$d/pend" "root=$d/good.json" "facades=$d/fac.json"
        row 1 '`alimentar`' 'a pre-monorepo name (alimentar) is RED' "$d/pend" "root=$d/rogue.json" "facades=$d/fac.json"
        row 1 '`apr-foo`' 'apr-<x> is not the apr exemption' "$d/pend" "root=$d/aprfoo.json" "facades=$d/fac.json"
        row 1 '`aprender-`' 'a bare `aprender-` is RED' "$d/pend" "root=$d/bare.json" "facades=$d/fac.json"
        row 1 '`aprenderx`' 'aprender without the hyphen is RED' "$d/pend" "root=$d/nohyph.json" "facades=$d/fac.json"
        row 1 '`Aprender-Data`' 'upper case is RED' "$d/pend" "root=$d/upper.json" "facades=$d/fac.json"
        row 1 '`apr-qa` (some-other-crate,' 'a pending-fold name in ANOTHER package is RED' "$d/pend" "root=$d/moved.json" "facades=$d/fac.json"
        row 1 'P  pending-fold row `apr-qa' 'a stale pending-fold row (fold landed) is RED' "$d/pend" "root=$d/nofold.json" "facades=$d/fac.json"
        row 1 '`apr-qa`' 'without its pending-fold row, apr-qa is RED' "$d/pend0" "root=$d/good.json" "facades=$d/fac.json"
        row 1 '`trueno-rag`' 'a rogue bin in the FACADE workspace is RED' "$d/pend" "root=$d/good.json" "facades=$d/fac_rogue.json"
        row 1 'W1 facade' 'a scan without the facade workspace is RED' "$d/pend" "root=$d/good.json"
        row 1 'W2' 'a scan with no `apr` bin is RED (not this tree)' "$d/pend" "root=$d/noapr.json" "facades=$d/fac.json"
        row 2 'ENV' 'unreadable metadata is ENV (2), never a pass' "$d/pend" "root=$d/junk.json" "facades=$d/fac.json"
        row 2 'ENV' 'a missing pending-fold file is ENV (2)' "$d/absent" "root=$d/good.json" "facades=$d/fac.json"
    }
    rows

    # MUTANTS: each must turn at least one row above red.
    local m got
    while IFS= read -r m; do
        n=$((n + 1))
        sed "$m" "$LIB" > "$d/mut.py"
        if cmp -s "$LIB" "$d/mut.py"; then printf 'FAIL mutant did not apply: %s\n' "$m"; fail=1; continue; fi
        got=$(lib="$d/mut.py"; rows 2>&1)
        if grep -q '^FAIL' <<< "$got"; then printf 'ok   mutant killed: %s\n' "$m"
        else printf 'FAIL mutant SURVIVED: %s\n' "$m"; fail=1; fi
    done <<'MUTANTS'
s/if name in EXEMPT or NAME.match(name):/if True:/
s/NAME = re.compile(r"^aprender-\[a-z0-9\]+(-\[a-z0-9\]+)\*\$")/NAME = re.compile(r"^apr")/
s/if pending.get(name) == pkg:/if name in pending:/
s/if pkg not in seen.get(name, set()):/if False:/
s/if "facades" not in labels:/if False:/
MUTANTS
    if [ "$fail" = 0 ]; then echo "check_bin_names_aprender self-test: ${n}/${n} pass"; return 0; fi
    echo "check_bin_names_aprender self-test: FAIL"; return 1
}

tree_meta() { # writes $ROOT_MD and $FAC_MD, or returns 2
    ROOT_MD=$(mktemp) && FAC_MD=$(mktemp) || return 2
    trap 'rm -f "${ROOT_MD:?}" "${FAC_MD:?}"' EXIT
    (cd "$REPO_ROOT" && cargo metadata --no-deps --format-version 1) > "$ROOT_MD" 2> /dev/null \
        && (cd "$FACADE_WS" && cargo metadata --no-deps --format-version 1) > "$FAC_MD" 2> /dev/null \
        || { echo "check_bin_names_aprender: ENV cargo metadata failed" >&2; return 2; }
}

case "${1:-}" in
    -h|--help) sed -n '2,26p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --self-test) self_test; exit $? ;;
    --list) tree_meta || exit 2; python3 "$LIB" --list "root=$ROOT_MD" "facades=$FAC_MD"; exit $? ;;
    "") ;;
    *) echo "check_bin_names_aprender: unknown argument $1" >&2; exit 2 ;;
esac

command -v python3 > /dev/null 2>&1 || { echo "check_bin_names_aprender: ENV python3 missing" >&2; exit 2; }
tree_meta || exit 2
echo "=== every shipped [[bin]] is aprender-* (check_bin_names_aprender.sh, #4430) ==="
python3 "$LIB" "$PENDING" "root=$ROOT_MD" "facades=$FAC_MD"
