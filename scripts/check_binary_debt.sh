#!/usr/bin/env bash
# check_binary_debt.sh - every binary the workspace ships is in the binary-debt
# ledger, and the ledger's two debt counters never exceed their ceilings (#4058,
# EPIC #4057).
#
# THE UNIVERSE IS DERIVED, NEVER LISTED
# -------------------------------------
# Every package manifest among: the root Cargo.toml, crates/*/Cargo.toml, and
# every path the root [workspace] names in `members` (globs expanded) OR in
# `exclude`. Cargo's own metadata sees members only, so it misses every
# excluded crate's binary: aprender-viz-ttop, aprender-train-canary,
# trueno-ublk (crates/aprender-zram/bins/) and ccpa-sft-export (tools/). The
# #4057 census scanned crates/*/ only and missed the last two.
# A package's binaries are its [[bin]] tables plus, unless autobins = false,
# src/main.rs (named after the package), src/bin/*.rs and src/bin/*/main.rs.
# A [[bin]] whose path is an auto-discovered file replaces that auto entry.
# A package marked [package.metadata] cargo-fuzz = true is a fuzz harness and is
# never shipped; it is left out BY THAT MARKER, not by name.
#
# THE FINDINGS
# ------------
#   NEW      a binary in the universe with no ledger row            -> RED
#   STALE    a ledger row whose binary is gone (delete the row)     -> RED
#   CLASS    a row whose class is outside the contract's classes    -> RED
#   CEILING  BINARY_DEBT or LEGACY_NAMES above its enforced ceiling -> RED
# STALE is also the vacuity guard: a universe scan that collapsed would read
# every row as STALE, never as a clean tree.
#
# BINARY_DEBT  = rows whose class is not KEEP (KEEP + RENAME counts: it awaits
#                the rename).
# LEGACY_NAMES = legacy crates.io names with no recorded sunset release.
# The enforced ceiling is `ceilings.current`, the measured value, lowered as the
# debt falls. A `releases` row with armed: true also binds once the workspace
# version reaches it; the #4057 table is recorded unarmed until the cop arms it.
#
# Usage: check_binary_debt.sh [--root DIR] [--ledger FILE] | --self-test | --help
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SELF="$ROOT/scripts/check_binary_debt.sh"
# shellcheck source=lib/python_fleet_state.sh
. "$ROOT/scripts/lib/python_fleet_state.sh" || exit 2

usage() {
    sed -n '2,36p' "$SELF" | sed 's/^# \{0,1\}//'
    printf '\n--self-test runs the case table over throwaway workspaces.\n'
}

judge() {
    local root=$1 ledger=$2 pyrc=0
    py_fleet_state check_binary_debt yaml tomllib || pyrc=$?
    [ "$pyrc" -eq 0 ] || { [ "$pyrc" -eq 3 ] && return 0; return "$pyrc"; }
    "${PY_FLEET_PYTHON:-python3}" - "$root" "$ledger" <<'PY'
import glob, os, sys, tomllib, yaml

root, ledger_path = sys.argv[1], sys.argv[2]

def load_toml(p):
    with open(p, "rb") as fh:
        return tomllib.load(fh)

def manifests():
    top = load_toml(os.path.join(root, "Cargo.toml"))
    ws = top.get("workspace", {})
    dirs = {"."} | {os.path.dirname(os.path.relpath(m, root))
                    for m in glob.glob(os.path.join(root, "crates", "*", "Cargo.toml"))}
    for entry in ws.get("members", []) + ws.get("exclude", []):
        for d in glob.glob(os.path.join(root, entry)):
            dirs.add(os.path.relpath(d, root))
    return sorted(d for d in dirs if os.path.isfile(os.path.join(root, d, "Cargo.toml")))

def package_bins(rel):
    d = os.path.join(root, rel)
    t = load_toml(os.path.join(d, "Cargo.toml"))
    pkg = t.get("package")
    if not pkg or (pkg.get("metadata") or {}).get("cargo-fuzz") is True:
        return None, {}
    out = {}
    if pkg.get("autobins", True):
        if os.path.isfile(os.path.join(d, "src", "main.rs")):
            out[pkg["name"]] = "src/main.rs"
        for f in glob.glob(os.path.join(d, "src", "bin", "*.rs")):
            out[os.path.basename(f)[:-3]] = os.path.relpath(f, d)
        for f in glob.glob(os.path.join(d, "src", "bin", "*", "main.rs")):
            out[os.path.basename(os.path.dirname(f))] = os.path.relpath(f, d)
    for b in t.get("bin", []):
        path = b.get("path")
        for k in [k for k, v in out.items() if path and v == path]:
            del out[k]
        out[b["name"]] = path or out.get(b["name"], "?")
    return pkg["name"], out

universe = {}
for rel in manifests():
    name, bins = package_bins(rel)
    for b in bins:
        universe[(name, b)] = rel

with open(ledger_path) as fh:
    c = yaml.safe_load(fh)
classes = set(c["classes"])
rows = c["binaries"]
keyed = {}
bad = []
for r in rows:
    k = (r["crate"], r["bin"])
    if k in keyed:
        bad.append("DUP      %s/%s has two ledger rows" % k)
    keyed[k] = r
    if r["class"] not in classes:
        bad.append("CLASS    %s/%s: class %r is not one of %s" % (k + (r["class"], sorted(classes))))
for k in sorted(set(universe) - set(keyed)):
    bad.append("NEW      %s/%s (%s) ships but has no row in the ledger: classify it" % (k + (universe[k],)))
for k in sorted(set(keyed) - set(universe)):
    bad.append("STALE    %s/%s has a ledger row but no longer ships: delete the row" % k)

debt = sum(1 for r in rows if r["class"] != "KEEP")
legacy = sum(1 for r in c.get("legacy_names", []) if not r.get("sunset"))
ceil_debt = c["ceilings"]["current"]["binary_debt"]
ceil_legacy = c["ceilings"]["current"]["legacy_names"]
top = load_toml(os.path.join(root, "Cargo.toml"))
ver = (top.get("workspace", {}).get("package", {}).get("version")
       or top.get("package", {}).get("version") or "0.0.0")
vkey = tuple(int(x) for x in ver.split("-")[0].split(".")[:3])
bound = "current"
for rel in c["ceilings"].get("releases", []):
    rkey = tuple(int(x) for x in str(rel["release"]).split(".")[:3])
    if rel.get("armed") is True and vkey >= rkey:
        if rel["binary_debt"] < ceil_debt or rel["legacy_names"] < ceil_legacy:
            bound = rel["release"]
        ceil_debt = min(ceil_debt, rel["binary_debt"])
        ceil_legacy = min(ceil_legacy, rel["legacy_names"])
if debt > ceil_debt:
    bad.append("CEILING  BINARY_DEBT %d > %d (ceiling: %s)" % (debt, ceil_debt, bound))
if legacy > ceil_legacy:
    bad.append("CEILING  LEGACY_NAMES %d > %d (ceiling: %s)" % (legacy, ceil_legacy, bound))

for line in bad:
    print("FAIL  " + line)
print("%s  binary debt: %d binaries in the universe, %d ledger rows; BINARY_DEBT %d/%d, LEGACY_NAMES %d/%d (version %s)"
      % ("FAIL" if bad else "PASS", len(universe), len(rows), debt, ceil_debt, legacy, ceil_legacy, ver))
sys.exit(1 if bad else 0)
PY
}

# ---- the case table ---------------------------------------------------------
mkcrate() {  # mkcrate <ws> <reldir> <pkg> [extra-toml] -> a package with src/main.rs
    mkdir -p "$1/$2/src"
    printf '[package]\nname = "%s"\nversion = "0.1.0"\n%s\n' "$3" "${4:-}" > "$1/$2/Cargo.toml"
    printf 'fn main() {}\n' > "$1/$2/src/main.rs"
}

fixture() {  # a workspace: two members, an excluded crate, a fuzz harness, a [[bin]] override
    local ws=$1
    mkdir -p "$ws/src/bin"
    printf '[workspace]\nmembers = [".", "crates/*"]\nexclude = ["tools/extra", "fuzz"]\n\n[package]\nname = "top"\nversion = "0.69.3"\n' > "$ws/Cargo.toml"
    printf 'fn main() {}\n' > "$ws/src/bin/tool.rs"
    mkcrate "$ws" crates/alpha alpha
    mkcrate "$ws" crates/beta beta '[[bin]]
name = "beta-cli"
path = "src/main.rs"'
    mkdir -p "$ws/crates/beta/src/bin/sub"
    printf 'fn main() {}\n' > "$ws/crates/beta/src/bin/sub/main.rs"
    mkdir -p "$ws/crates/shell"
    printf '[workspace]\n' > "$ws/crates/shell/Cargo.toml"
    mkcrate "$ws" tools/extra extra
    mkcrate "$ws" fuzz fuzzer '[package.metadata]
cargo-fuzz = true'
    cat > "$ws/ledger.yaml" <<'YML'
classes: [KEEP, KEEP + RENAME, MERGE-INTO-apr, DEPRECATE→DELETE, DECIDE]
ceilings:
  current: {binary_debt: 2, legacy_names: 1}
  releases:
    - {release: 0.70.0, binary_debt: 1, legacy_names: 1, armed: false}
legacy_names:
  - {name: oldname, sunset: null}
  - {name: gone, sunset: 0.69.0}
binaries:
  - {crate: top, bin: tool, class: KEEP}
  - {crate: alpha, bin: alpha, class: DECIDE}
  - {crate: beta, bin: beta-cli, class: KEEP}
  - {crate: beta, bin: sub, class: MERGE-INTO-apr}
  - {crate: extra, bin: extra, class: KEEP}
YML
}

# SEC011: guarded delete, the repo idiom (check_cascade_converges.sh::_rm). A
# global, not a local: the EXIT trap runs after self_test's scope is gone.
BD_TD=
_rm_td() {
    local v="${BD_TD:-}"
    case "$v" in */binary-debt-selftest.?*) ;; *) return 0 ;; esac
    [ -n "$v" ] && [ "$v" != "/" ] && rm -rf -- "$v" || :
}

self_test() {
    local td fails=0 rows=0 ws rc out
    td=$(mktemp -d "${TMPDIR:-/tmp}/binary-debt-selftest.XXXXXX")
    BD_TD=$td
    trap _rm_td EXIT
    py_fleet_state check_binary_debt yaml tomllib || { rc=$?; [ "$rc" -eq 3 ] && return 0; return "$rc"; }
    # row <want 0|1> <must-print-or-empty> <label> <mutation...>
    row() {
        local want=$1 needle=$2 label=$3; shift 3
        rows=$((rows + 1)); ws="$td/ws$rows"; fixture "$ws"
        (cd "$ws" && eval "$*") || { printf 'BROKE  %s: the mutation itself failed\n' "$label"; fails=$((fails + 1)); return; }
        rc=0; out=$(judge "$ws" "$ws/ledger.yaml" 2>&1) || rc=$?
        if [ "$rc" -eq "$want" ] && { [ -z "$needle" ] || [[ "$out" == *"$needle"* ]]; }; then
            printf 'ok     %s\n' "$label"
        else
            printf 'BROKE  %s: want rc %s%s, got rc %s\n%s\n' "$label" "$want" "${needle:+ + \"$needle\"}" "$rc" "$out"
            fails=$((fails + 1))
        fi
    }
    row 0 "5 binaries in the universe" "the clean fixture passes (root src/bin, [[bin]] override, src/bin/*/main.rs, excluded crate; fuzz and a package-less shell left out)" true
    row 1 "NEW      alpha/newbin" "an unledgered new binary is NEW" "mkdir -p crates/alpha/src/bin; printf 'fn main(){}\n' > crates/alpha/src/bin/newbin.rs"
    row 1 "NEW      extra/extra" "an EXCLUDED crate's binary is in the universe" "sed -i '/crate: extra/d' ledger.yaml"
    row 1 "NEW      gamma/gamma" "a crate added under crates/ is seen" "mkcrate . crates/gamma gamma"
    row 1 "STALE    alpha/alpha" "a row whose binary is gone is STALE" "rm crates/alpha/src/main.rs"
    row 1 "STALE    extra/extra" "dropping a crate from exclude hides it: STALE, never a quiet pass" "sed -i 's/\"tools\\/extra\", //' Cargo.toml"
    row 1 "CLASS    alpha/alpha" "a class outside the contract's set" "sed -i 's/class: DECIDE/class: MAYBE/' ledger.yaml"
    row 1 "CEILING  BINARY_DEBT 3 > 2" "BINARY_DEBT above its ceiling" "sed -i 's/bin: tool, class: KEEP/bin: tool, class: DECIDE/' ledger.yaml"
    row 0 "BINARY_DEBT 2/2" "BINARY_DEBT AT its ceiling is not over it (near miss)" true
    row 1 "CEILING  LEGACY_NAMES 2 > 1" "LEGACY_NAMES above its ceiling" "sed -i 's/sunset: 0.69.0/sunset: null/' ledger.yaml"
    row 0 "" "an UNARMED release ceiling does not bind" true
    row 1 "CEILING  BINARY_DEBT 2 > 1 (ceiling: 0.70.0)" "an ARMED release ceiling binds once the version reaches it" "sed -i 's/armed: false/armed: true/' ledger.yaml; sed -i 's/version = \"0.69.3\"/version = \"0.70.0\"/' Cargo.toml"
    row 0 "" "an ARMED release ceiling does not bind BEFORE its version" "sed -i 's/armed: false/armed: true/' ledger.yaml"
    row 1 "NEW      fuzzer/fuzzer" "only the cargo-fuzz MARKER keeps a harness out" "sed -i '/cargo-fuzz/d' fuzz/Cargo.toml"
    row 1 "DUP      alpha/alpha" "a binary with two rows" "printf '  - {crate: alpha, bin: alpha, class: KEEP}\n' >> ledger.yaml"
    printf '%s  check_binary_debt self-test: %d rows, %d broke\n' "$([ "$fails" -eq 0 ] && echo PASS || echo FAIL)" "$rows" "$fails"
    [ "$fails" -eq 0 ]
}

case "${1:-}" in
    --help|-h) usage ;;
    --self-test) self_test ;;
    *)
        ledger="$ROOT/contracts/binary-debt-v1.yaml" root="$ROOT"
        while [ $# -gt 0 ]; do
            case $1 in
                --root) root=$2; shift 2 ;;
                --ledger) ledger=$2; shift 2 ;;
                *) printf 'unknown argument %s\n' "$1" >&2; exit 2 ;;
            esac
        done
        judge "$root" "$ledger" ;;
esac
