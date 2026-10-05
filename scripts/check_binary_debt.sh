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
MANIFEST_AWK="$ROOT/scripts/lib/cargo_manifest_rows.awk"
LEDGER_AWK="$ROOT/scripts/lib/binary_debt_ledger_rows.awk"

usage() {
    sed -n '2,36p' "$SELF" | sed 's/^# \{0,1\}//'
    printf '\n--self-test runs the case table over throwaway workspaces.\n'
}

# rel_of ROOT PATH -> PATH relative to ROOT, lexically ("." for ROOT itself), as python's relpath.
rel_of() { realpath -ms --relative-to="$1" "$2"; }

# semver_ok V -> 0 when V (before any "-") is MAJOR.MINOR.PATCH, all digits.
semver_ok() { [[ "${1%%-*}" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; }

# semver_ge A B -> 0 when A >= B (both already semver_ok).
semver_ge() {
    local a b i
    IFS=. read -r -a a <<< "${1%%-*}"; IFS=. read -r -a b <<< "${2%%-*}"
    for i in 0 1 2; do
        if ((10#${a[i]} != 10#${b[i]})); then ((10#${a[i]} > 10#${b[i]})); return; fi
    done
    return 0
}

# package_bins ROOT REL -> adds REL's shipped binaries to the caller's `universe`; rc 2 on a
# manifest form the reader refuses. A cargo-fuzz harness and a package-less manifest add none.
package_bins() {
    local root=$1 rel=$2 d="$1/$2" rows k a b c pkg="" haspkg=0 fuzz=false autobins=true f n nb=0 path name
    local -A found=() bname=() bpath=()
    rows=$(awk -f "$MANIFEST_AWK" "$d/Cargo.toml") || return 2
    while IFS=$'\t' read -r k a b c; do
        case "$k" in
            haspkg) haspkg=1 ;;
            name) pkg="$a" ;;
            autobins) autobins="$a" ;;
            fuzz) fuzz="$a" ;;
            bintable) nb="$a" ;;
            bin)
                if [ "$b" = name ]; then
                    bname["$a"]="$c"
                else
                    bpath["$a"]="$c"
                fi ;;
        esac
    done <<< "$rows"
    [ "$haspkg" -eq 1 ] && [ "$fuzz" != true ] || return 0
    if [ -z "$pkg" ]; then printf 'REFUSE %s/Cargo.toml: [package] has no name\n' "$rel" >&2; return 2; fi
    if [ "$autobins" = true ]; then
        if [ -f "$d/src/main.rs" ]; then
            found["$pkg"]=src/main.rs
        fi
        for f in "$d"/src/bin/*.rs; do
            [ -e "$f" ] || continue
            n="${f##*/}"
            found["${n%.rs}"]="src/bin/$n"
        done
        for f in "$d"/src/bin/*/main.rs; do
            [ -e "$f" ] || continue
            n="${f%/main.rs}"
            n="${n##*/}"
            found["$n"]="src/bin/$n/main.rs"
        done
    fi
    # A [[bin]] whose path is an auto-discovered file replaces that auto entry.
    for ((n = 1; n <= nb; n++)); do
        if [ -z "${bname[$n]+set}" ]; then printf 'REFUSE %s/Cargo.toml: [[bin]] #%d has no name\n' "$rel" "$n" >&2; return 2; fi
        name="${bname[$n]}" path="${bpath[$n]-}"
        if [ -n "$path" ]; then
            for k in "${!found[@]}"; do [ "${found[$k]}" = "$path" ] && unset 'found[$k]'; done
            found["$name"]="$path"
        else
            found["$name"]="${found[$name]-?}"
        fi
    done
    for k in "${!found[@]}"; do universe["$pkg"$'\t'"$k"]="$rel"; done
}

# judge ROOT LEDGER -> rc 0 PASS, 1 a finding, 2 a manifest or ledger form the readers refuse.
# bash + awk only. The python reader this replaced needed tomllib; on a runner without it the
# guard printed UNMEASURED and exited 0, so CI never saw it judge a tree.
judge() {
    local root=$1 ledger=$2 rows rel m entry k a b c d ver="" pkgver="" rel_ver
    local debt=0 legacy=0 nrows=0 ceil_debt="" ceil_legacy="" bound=current classes_txt
    local -A dirs=([.]=1) universe=() keyed=() classes=()
    local -a bad=() releases=() order=()
    [ -f "$root/Cargo.toml" ] || { printf 'ENV   check_binary_debt: no Cargo.toml under %s\n' "$root" >&2; return 2; }
    rel_of / / > /dev/null 2>&1 || { printf 'ENV   check_binary_debt: realpath -m --relative-to (GNU coreutils) is required\n' >&2; return 2; }
    rows=$(awk -f "$MANIFEST_AWK" "$root/Cargo.toml") || return 2
    for m in "$root"/crates/*/Cargo.toml; do
        [ -f "$m" ] || continue
        dirs["$(rel_of "$root" "${m%/Cargo.toml}")"]=1
    done
    while IFS=$'\t' read -r k a; do
        case "$k" in
            members|exclude) while IFS= read -r m; do dirs[$(rel_of "$root" "$m")]=1; done < <(compgen -G "$root/$a" || :) ;;
            wsversion) ver="$a" ;;
            pkgversion) pkgver="$a" ;;
            pkgversion_other) pkgver=OTHER ;;
        esac
    done <<< "$rows"
    if [ -z "$ver" ] && [ "$pkgver" = OTHER ]; then
        printf 'REFUSE %s/Cargo.toml: [package] version is not a string and no [workspace.package] version is set\n' "$root" >&2
        return 2
    fi
    ver="${ver:-${pkgver:-0.0.0}}"
    while IFS= read -r rel; do
        [ -f "$root/$rel/Cargo.toml" ] || continue
        package_bins "$root" "$rel" || return 2
    done < <(printf '%s\n' "${!dirs[@]}" | LC_ALL=C sort)

    rows=$(awk -f "$LEDGER_AWK" "$ledger") || return 2
    while IFS=$'\t' read -r k a b c d; do
        case "$k" in
            class) classes["$a"]=1 ;;
            current) ceil_debt="$a" ceil_legacy="$b" ;;
            release) releases+=("$a"$'\t'"$b"$'\t'"$c"$'\t'"$d") ;;
            legacy) [ -n "$b" ] || legacy=$((legacy + 1)) ;;
            row) order+=("$a"$'\t'"$b"$'\t'"$c") ;;
        esac
    done <<< "$rows"
    classes_txt=$(printf '%s\n' "${!classes[@]}" | LC_ALL=C sort | sed "s/.*/'&'/" | paste -sd, - | sed 's/,/, /g')
    for m in "${order[@]}"; do
        IFS=$'\t' read -r a b c <<< "$m"
        nrows=$((nrows + 1))
        [ "$c" = KEEP ] || debt=$((debt + 1))
        [ -z "${keyed[$a$'\t'$b]+set}" ] || bad+=("DUP      $a/$b has two ledger rows")
        keyed["$a"$'\t'"$b"]=1
        [ -n "${classes[$c]+set}" ] || bad+=("CLASS    $a/$b: class '$c' is not one of [$classes_txt]")
    done
    while IFS=$'\t' read -r a b; do
        [ -n "$a" ] && [ -z "${keyed[$a$'\t'$b]+set}" ] && bad+=("NEW      $a/$b (${universe[$a$'\t'$b]}) ships but has no row in the ledger: classify it")
    done < <(printf '%s\n' "${!universe[@]}" | LC_ALL=C sort)
    while IFS=$'\t' read -r a b; do
        [ -n "$a" ] && [ -z "${universe[$a$'\t'$b]+set}" ] && bad+=("STALE    $a/$b has a ledger row but no longer ships: delete the row")
    done < <(printf '%s\n' "${!keyed[@]}" | LC_ALL=C sort)

    # MAJOR.MINOR.PATCH exactly: a 4th component is refused loudly, never dropped.
    semver_ok "$ver" || { printf "FAIL: workspace version '%s' is not MAJOR.MINOR.PATCH\n" "$ver" >&2; return 1; }
    for m in "${releases[@]}"; do
        IFS=$'\t' read -r rel_ver b c d <<< "$m"
        semver_ok "$rel_ver" || { printf "FAIL: ceiling release '%s' is not MAJOR.MINOR.PATCH\n" "$rel_ver" >&2; return 1; }
        if [ "$d" = true ] && semver_ge "$ver" "$rel_ver"; then
            { [ "$b" -lt "$ceil_debt" ] || [ "$c" -lt "$ceil_legacy" ]; } && bound="$rel_ver"
            [ "$b" -lt "$ceil_debt" ] && ceil_debt="$b"
            [ "$c" -lt "$ceil_legacy" ] && ceil_legacy="$c"
        fi
    done
    [ "$debt" -le "$ceil_debt" ] || bad+=("CEILING  BINARY_DEBT $debt > $ceil_debt (ceiling: $bound)")
    [ "$legacy" -le "$ceil_legacy" ] || bad+=("CEILING  LEGACY_NAMES $legacy > $ceil_legacy (ceiling: $bound)")

    for m in "${bad[@]}"; do printf 'FAIL  %s\n' "$m"; done
    printf '%s  binary debt: %d binaries in the universe, %d ledger rows; BINARY_DEBT %d/%d, LEGACY_NAMES %d/%d (version %s)\n' \
        "$([ "${#bad[@]}" -eq 0 ] && echo PASS || echo FAIL)" "${#universe[@]}" "$nrows" "$debt" "$ceil_debt" "$legacy" "$ceil_legacy" "$ver"
    [ "${#bad[@]}" -eq 0 ]
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
    # Mutations that need a redirect or two steps are functions, so `row` runs
    # its argv directly (bashrs SEC001: no eval).
    mut_newbin() { mkdir -p crates/alpha/src/bin && printf 'fn main(){}\n' > crates/alpha/src/bin/newbin.rs; }
    mut_drop_alpha_main() { rm -f -- crates/alpha/src/main.rs; }
    mut_arm_and_bump() { sed -i 's/armed: false/armed: true/' ledger.yaml && sed -i 's/version = "0.69.3"/version = "0.70.0"/' Cargo.toml; }
    mut_dup_alpha() { printf '  - {crate: alpha, bin: alpha, class: KEEP}\n' >> ledger.yaml; }
    # The readers are a TOML/YAML subset: a form outside it is refused (rc 2), never guessed.
    mut_ghost_bin() { printf 'description = """\n[[bin]]\nname = "ghost"\n"""\n' >> crates/alpha/Cargo.toml; }
    mut_ml_exclude() { sed -i 's|^exclude = .*|exclude = [\n    "tools/extra",  # a crate outside the members\n    "fuzz",\n]|' Cargo.toml; }
    mut_inline_fuzz() { sed -i '/^version/a metadata = { cargo-fuzz = true }' crates/alpha/Cargo.toml; }
    mut_nameless_bin() { printf '[[bin]]\npath = "src/main.rs"\n' >> crates/alpha/Cargo.toml; }
    mut_block_row() { printf '  - crate: alpha\n    bin: alpha\n' >> ledger.yaml; }
    mut_pathless_bin() { printf '[[bin]]\nname = "pathless"\n' >> crates/alpha/Cargo.toml; }
    mut_ws_version() { printf '\n[workspace.package]\nversion = "0.70.0"\n' >> Cargo.toml && sed -i 's/armed: false/armed: true/' ledger.yaml; }
    mut_top_dotted() { sed -i '1i package.autobins = false' crates/alpha/Cargo.toml; }
    mut_no_binaries() { sed -i '/^binaries:/,$d' ledger.yaml; }
    mut_classless_row() { printf '  - {crate: alpha, bin: alpha}\n' >> ledger.yaml; }
    mut_release_lowers_legacy() { mut_arm_and_bump && sed -i "s/legacy_names: 1, armed/legacy_names: 0, armed/" ledger.yaml; }
    mut_zero_pad_version() { mut_arm_and_bump && sed -i 's/version = "0.70.0"/version = "0.070.0"/' Cargo.toml; }
    mut_quoted_armed() { mut_arm_and_bump && sed -i 's/armed: true/armed: "true"/' ledger.yaml; }
    mut_root_version_ws() { sed -i 's/^version = "0.69.3"/version.workspace = true/' Cargo.toml; }
    mut_ws_inline_package() { sed -i 's/^\[workspace\]/[workspace]\npackage = { version = "0.70.0" }/' Cargo.toml; }
    mut_dotted_bin_path() { printf '[[bin]]\nname = "dotted"\npath.x = "src/main.rs"\n' >> crates/alpha/Cargo.toml; }
    # row <want rc> <must-print-or-empty> <label> <mutation command + args...>
    row() {
        local want=$1 needle=$2 label=$3; shift 3
        rows=$((rows + 1)); ws="$td/ws$rows"; fixture "$ws"
        (cd "$ws" && "$@") || { printf 'BROKE  %s: the mutation itself failed\n' "$label"; fails=$((fails + 1)); return; }
        rc=0; out=$(judge "$ws" "$ws/ledger.yaml" 2>&1) || rc=$?
        if [ "$rc" -eq "$want" ] && { [ -z "$needle" ] || [[ "$out" == *"$needle"* ]]; }; then
            printf 'ok     %s\n' "$label"
        else
            printf 'BROKE  %s: want rc %s%s, got rc %s\n%s\n' "$label" "$want" "${needle:+ + \"$needle\"}" "$rc" "$out"
            fails=$((fails + 1))
        fi
    }
    row 0 "5 binaries in the universe" "the clean fixture passes (root src/bin, [[bin]] override, src/bin/*/main.rs, excluded crate; fuzz and a package-less shell left out)" true
    row 1 "NEW      alpha/newbin" "an unledgered new binary is NEW" mut_newbin
    row 1 "NEW      extra/extra" "an EXCLUDED crate's binary is in the universe" sed -i '/crate: extra/d' ledger.yaml
    row 1 "NEW      gamma/gamma" "a crate added under crates/ is seen" mkcrate . crates/gamma gamma
    row 1 "STALE    alpha/alpha" "a row whose binary is gone is STALE" mut_drop_alpha_main
    row 1 "STALE    extra/extra" "dropping a crate from exclude hides it: STALE, never a quiet pass" sed -i 's/"tools\/extra", //' Cargo.toml
    row 1 "CLASS    alpha/alpha" "a class outside the contract's set" sed -i 's/class: DECIDE/class: MAYBE/' ledger.yaml
    row 1 "CEILING  BINARY_DEBT 3 > 2" "BINARY_DEBT above its ceiling" sed -i 's/bin: tool, class: KEEP/bin: tool, class: DECIDE/' ledger.yaml
    row 0 "BINARY_DEBT 2/2" "BINARY_DEBT AT its ceiling is not over it (near miss)" true
    row 1 "CEILING  LEGACY_NAMES 2 > 1" "LEGACY_NAMES above its ceiling" sed -i 's/sunset: 0.69.0/sunset: null/' ledger.yaml
    row 0 "" "an UNARMED release ceiling does not bind" true
    row 1 "CEILING  BINARY_DEBT 2 > 1 (ceiling: 0.70.0)" "an ARMED release ceiling binds once the version reaches it" mut_arm_and_bump
    row 0 "" "an ARMED release ceiling does not bind BEFORE its version" sed -i 's/armed: false/armed: true/' ledger.yaml
    row 1 "NEW      fuzzer/fuzzer" "only the cargo-fuzz MARKER keeps a harness out" sed -i '/cargo-fuzz/d' fuzz/Cargo.toml
    row 1 "DUP      alpha/alpha" "a binary with two rows" mut_dup_alpha
    row 0 "5 binaries in the universe" "a [[bin]] line inside a multi-line string is not a table" mut_ghost_bin
    row 0 "5 binaries in the universe" "a multi-line exclude array with comments is read whole" mut_ml_exclude
    row 0 "5 binaries in the universe" "a comma inside a quoted ledger value is not a field separator" sed -i 's/class: DECIDE}/class: DECIDE, why: "a, b: c"}/' ledger.yaml
    row 2 "REFUSE" "a dotted package.name (name.workspace) is refused" sed -i 's/^name = "alpha"/name.workspace = true/' crates/alpha/Cargo.toml
    row 2 "REFUSE" "cargo-fuzz in an inline metadata table is refused" mut_inline_fuzz
    row 2 "REFUSE" "a non-string workspace member is refused" sed -i 's|"crates/\*"\]|"crates/*", 3]|' Cargo.toml
    row 2 "REFUSE" "a [[bin]] with no name is refused" mut_nameless_bin
    row 2 "REFUSE" "a block-style ledger row is refused" mut_block_row
    row 1 "NEW      alpha/pathless (crates/alpha)" "a [[bin]] with a name and no path ships" mut_pathless_bin
    row 1 "(ceiling: 0.70.0)" "[workspace.package] version wins over [package] version" mut_ws_version
    row 1 "CEILING  LEGACY_NAMES 1 > 0 (ceiling: 0.70.0)" "an ARMED release lowers the LEGACY_NAMES ceiling too" mut_release_lowers_legacy
    row 2 "REFUSE" "a ledger row with an empty class is refused" sed -i "s/class: DECIDE}/class: \"\"}/" ledger.yaml
    row 2 "REFUSE" "a legacy_names row with no name is refused" sed -i "s/{name: oldname, sunset: null}/{sunset: null}/" ledger.yaml
    row 1 "is not MAJOR.MINOR.PATCH" "a version that is not MAJOR.MINOR.PATCH fails loudly" sed -i 's/version = "0.69.3"/version = "0.69"/' Cargo.toml
    row 2 "REFUSE" "a quoted table header is refused" sed -i 's/^\[package.metadata\]/["package".metadata]/' fuzz/Cargo.toml
    row 2 "REFUSE" "an inline-table package.name is refused" sed -i 's/^name = "alpha"/name = { workspace = true }/' crates/alpha/Cargo.toml
    row 2 "REFUSE" "a string cargo-fuzz is refused, never read as truthy" sed -i 's/cargo-fuzz = true/cargo-fuzz = "true"/' fuzz/Cargo.toml
    row 2 "REFUSE" "a top-level dotted package key is refused" mut_top_dotted
    row 2 "REFUSE" "a ledger with no binaries section is refused" mut_no_binaries
    row 2 "REFUSE" "a ledger row with no class is refused" mut_classless_row
    row 2 "REFUSE" "a ceiling that is not an integer is refused" sed -i 's/current: {binary_debt: 2,/current: {binary_debt: two,/' ledger.yaml
    row 1 "CEILING  BINARY_DEBT 2 > 1 (ceiling: 0.70.0)" "a zero-padded version component is decimal, never octal" mut_zero_pad_version
    row 0 "" "a QUOTED armed: \"true\" is a string, not a bool: the ceiling stays unarmed" mut_quoted_armed
    row 1 "CEILING  LEGACY_NAMES 2 > 1" "sunset: false is falsy: the name still counts" sed -i 's/sunset: 0.69.0/sunset: false/' ledger.yaml
    row 0 "LEGACY_NAMES 0/1" "a QUOTED sunset: \"null\" is a string: the name is sunset" sed -i 's/sunset: null/sunset: "null"/' ledger.yaml
    row 2 "REFUSE" "a sunset YAML could read as a number is refused" sed -i 's/sunset: 0.69.0/sunset: 0/' ledger.yaml
    row 2 "REFUSE" "an empty classes list is refused" sed -i 's/^classes: .*/classes: []/' ledger.yaml
    row 2 "REFUSE" "a root [package] version that is not a string, with no workspace version, is refused" mut_root_version_ws
    row 2 "REFUSE" "a [workspace] package key (inline or dotted) is refused" mut_ws_inline_package
    row 2 "REFUSE" "a quoted key in a package header is refused" sed -i 's/^\[package.metadata\]/[package."metadata"]/' fuzz/Cargo.toml
    row 2 "REFUSE" "a single-quoted cargo-fuzz key is refused" sed -i "s/^cargo-fuzz = true/'cargo-fuzz' = true/" fuzz/Cargo.toml
    row 2 "REFUSE" "a dotted [[bin]] path key is refused, never read as a pathless bin" mut_dotted_bin_path
    row 2 "REFUSE" "a quoted \"autobins\" key is refused, never skipped" sed -i 's/^name = "alpha"/name = "alpha"\n"autobins" = false/' crates/alpha/Cargo.toml
    row 2 "REFUSE" "an inline-table root [package] version, with no workspace version, is refused" sed -i 's/^version = "0.69.3"/version = { workspace = true }/' Cargo.toml
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
