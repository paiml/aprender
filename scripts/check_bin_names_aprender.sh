#!/usr/bin/env bash
# check_bin_names_aprender.sh -- every shipped [[bin]] is named aprender-* (#4430).
#
# WHY. The 0.70.0 all-binaries gate ships every workspace [[bin]] (the nightly
# derives the list from the workspace metadata, #4189). Before #4430, eleven of them
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
# Bins are read from the Cargo.toml manifests with the toolchain's own target
# inference (scripts/lib/bin_name_prefix.py --tree), never from a grep of
# [[bin]]: an auto-discovered src/bin/*.rs has no stanza (verificar was one).
# Needing no toolchain keeps this guard in the cargo-free subset that
# guard_tree.sh runs in the guard-tree job, so it is wired by existing.
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
    # tree <dir> -- a two-workspace fixture for --tree: root (members, one
    # excluded), facades, an auto-discovered src/bin/*.rs, a src/bin/<x>/main.rs,
    # an explicit [[bin]] that claims an inferred path, and autobins = false.
    tree() {
        local t=$1
        mkdir -p "$t/src/bin" "$t/crates/a/src/bin/aprender-dir" "$t/crates/b/src/bin" \
            "$t/crates/c/src/bin" "$t/crates/x/src" "$t/crates/facades/pc/src/bin"
        printf '[package]\nname = "aprender"\n[workspace]\nmembers = [".", "crates/a", "crates/b", "crates/c"]\nexclude = ["crates/x", "crates/facades"]\n' > "$t/Cargo.toml"
        : > "$t/src/bin/apr.rs"
        printf '[package]\nname = "a"\n' > "$t/crates/a/Cargo.toml"
        : > "$t/crates/a/src/bin/aprender-a.rs"; : > "$t/crates/a/src/bin/aprender-dir/main.rs"
        printf '[package]\nname = "b"\n[[bin]]\nname = "aprender-b"\npath = "src/bin/old.rs"\n' > "$t/crates/b/Cargo.toml"
        : > "$t/crates/b/src/bin/old.rs"
        printf '[package]\nname = "c"\nautobins = false\n' > "$t/crates/c/Cargo.toml"
        : > "$t/crates/c/src/bin/rogue.rs"
        printf '[package]\nname = "x"\n' > "$t/crates/x/Cargo.toml"; : > "$t/crates/x/src/main.rs"
        printf '[workspace]\nmembers = ["pc"]\n' > "$t/crates/facades/Cargo.toml"
        printf '[package]\nname = "pc"\n' > "$t/crates/facades/pc/Cargo.toml"; : > "$t/crates/facades/pc/src/bin/pv.rs"
    }
    tree "$d/t_good"
    tree "$d/t_auto"; : > "$d/t_auto/crates/a/src/bin/trueno-rag.rs"
    tree "$d/t_dir"; mkdir -p "$d/t_dir/crates/a/src/bin/simular"; : > "$d/t_dir/crates/a/src/bin/simular/main.rs"
    tree "$d/t_main"; : > "$d/t_main/crates/a/src/main.rs"
    tree "$d/t_fac"; : > "$d/t_fac/crates/facades/pc/src/bin/presentar.rs"
    tree "$d/t_glob"; sed -i 's|"crates/a", "crates/b", "crates/c"|"crates/*"|' "$d/t_glob/Cargo.toml"
    tree "$d/t_globx"; sed -i 's|"crates/a", "crates/b", "crates/c"|"crates/*"|; s|"crates/x", ||' "$d/t_globx/Cargo.toml"
    tree "$d/t_nows"; printf '[package]\nname = "aprender"\n' > "$d/t_nows/Cargo.toml"
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
        # --tree: the manifests read the way the toolchain infers targets.
        row 0 '5 bin name(s)' 'tree: explicit, auto, dir, apr, pv; excluded + autobins=false unseen' "$d/pend0" --tree "$d/t_good"
        row 1 '`trueno-rag`' 'tree: an auto-discovered src/bin/*.rs is RED' "$d/pend0" --tree "$d/t_auto"
        row 1 '`simular`' 'tree: an auto-discovered src/bin/<x>/main.rs is RED' "$d/pend0" --tree "$d/t_dir"
        row 1 '`a`' 'tree: src/main.rs ships as the package name -- RED' "$d/pend0" --tree "$d/t_main"
        row 1 '`presentar`' 'tree: a rogue bin in the facade workspace is RED' "$d/pend0" --tree "$d/t_fac"
        row 0 '5 bin name(s)' 'tree: a glob member list honours exclude' "$d/pend0" --tree "$d/t_glob"
        row 1 '`x`' 'tree: the same glob without the exclude scans x -- RED' "$d/pend0" --tree "$d/t_globx"
        row 2 'ENV' 'tree: a root with no [workspace] is ENV (2)' "$d/pend0" --tree "$d/t_nows"
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
s/        if d not in excl:/        if True:/
s/    if pkg.get("autobins", True):/    if False:/
s/                cands.append((e, os.path.join("src", "bin", e, "main.rs")))/                pass/
s/        cands = \[(name, os.path.join("src", "main.rs"))\]/        cands = []/
s/and os.path.normpath(rel) not in paths)/)/
s/hits = sorted(glob.glob(pat)) if any(c in m for c in "\*?\[") else \[pat\]/hits = [pat]/
s/("facades", workspace_bins(os.path.join(root, "crates", "facades")))\]/]/
MUTANTS
    if [ "$fail" = 0 ]; then echo "check_bin_names_aprender self-test: ${n}/${n} pass"; return 0; fi
    echo "check_bin_names_aprender self-test: FAIL"; return 1
}

case "${1:-}" in
    -h|--help) sed -n '2,/^set -uo pipefail/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    --self-test) self_test; exit $? ;;
    --list) python3 "$LIB" --list --tree "$REPO_ROOT"; exit $? ;;
    "") ;;
    *) echo "check_bin_names_aprender: unknown argument $1" >&2; exit 2 ;;
esac

command -v python3 > /dev/null 2>&1 || { echo "check_bin_names_aprender: ENV python3 missing" >&2; exit 2; }
echo "=== every shipped [[bin]] is aprender-* (check_bin_names_aprender.sh, #4430) ==="
python3 "$LIB" "$PENDING" --tree "$REPO_ROOT"
