#!/usr/bin/env bash
#
# check_workspace_siblings_pathed.sh — an in-tree sibling may never be declared
# as a crates.io dependency.
#
# THE CLASS
# ---------
# APR-MONO consolidated 20 repos into this one. Every sibling — trueno, realizar,
# entrenar, pacha, simular, alimentar, renacer, … — now lives under `crates/` and
# is consumed through a `[workspace.dependencies]` path alias:
#
#     # root Cargo.toml
#     trueno = { path = "crates/aprender-compute", version = "0.63.0", package = "aprender-compute" }
#     # member Cargo.toml
#     trueno = { workspace = true }
#
# Writing `trueno = "0.16"` instead does NOT fail to build. Cargo happily resolves
# the crates.io copy ALONGSIDE the in-tree one, so the tree compiles two different
# `trueno`s whose types are mutually incompatible, and the published-crate
# dependency cycle the consolidation removed comes back. Nothing goes red. It is
# invisible until someone reads `cargo tree --duplicates`.
#
# It is not hypothetical. At 88791ff55 (2026-08-14) the lockfile carried FOUR
# registry copies of `trueno` (0.11.0, 0.14.6, 0.15.0, 0.17.5) beside the path
# one, plus registry copies of provable-contracts, provable-contracts-macros,
# jugar-probar, jugar-probar-derive, presentar-core, presentar-terminal and
# trueno-ublk — every one of them seeded by a declaration this guard names.
#
# CLAUDE.md has forbidden this in prose since the consolidation. Prose is not a
# gate; that is the whole finding.
#
# THE RULE
# --------
# For every dependency declaration in every tracked manifest, if the crate named
# is served in-tree, the declaration must resolve to the local source:
#
#   legal    trueno = { workspace = true }
#   legal    trueno = { path = "../aprender-compute" }
#   legal    trueno = { path = "…", version = "0.63.0" }   # path + version is what
#                                                          # `cargo publish` needs
#   FAIL     trueno = "0.16"                               # registry
#   FAIL     trueno = { version = "0.16.5", features = ["gpu"] }
#   FAIL     trueno = { git = "…" }                        # also not the in-tree copy
#
# A `version` beside a `path` is fine — cargo builds from the path and reads the
# version only when publishing. The defect is a version with NO local source.
#
# THE NAME SET — WHY `[lib] name` IS LOAD-BEARING
# -----------------------------------------------
# Package name and lib name DIVERGE throughout this tree, and the divergence is
# exactly where the motivating bug lives:
#
#     crates/aprender-compute   [package] aprender-compute   [lib] trueno
#     crates/aprender-db        [package] aprender-db        [lib] trueno_db
#     crates/aprender-serve     [package] aprender-serve     [lib] realizar
#
# A guard built from `[package] name` alone would hold "aprender-compute" and
# would sail straight past `trueno = "0.16"` — the one declaration it exists to
# stop. So the set is package names UNION lib names, each in both `-` and `_`
# spelling (a lib is `trueno_db`; the crate it shadows publishes as `trueno-db`),
# UNION every root `[workspace.dependencies]` key carrying a `path =`.
#
# SCOPE
# -----
# NAMES come from the root manifest and every manifest under `crates/`.
# SCANNING is wider — every tracked `Cargo.toml`. A violation can be written
# anywhere, and one of the live ones was in `crates/aprender-db/wasm-pkg/`, a
# nested demo manifest no crate-list glob would reach.
#
# VACUITY GUARD
# -------------
# A guard that matches nothing must not report clean — "found no violations" and
# "looked in the wrong place" print identically. Three defences:
#   1. it refuses to pass with fewer than MIN_CRATES in-tree names,
#   2. it refuses to pass having scanned fewer than MIN_MANIFESTS manifests,
#   3. before printing PASS it runs a POSITIVE CONTROL — a synthetic
#      `trueno = "0.16"` through the real scanner with the real name set — and
#      aborts if that is not flagged. The absence is only reported once the
#      search is proved able to succeed.
#
# SELF-TEST
# ---------
#   bash scripts/check_workspace_siblings_pathed.sh --self-test
# runs the must-fail / must-pass case table, twelve whole-fragment cases, and an
# end-to-end mutation probe on a throwaway tree (Verification Discipline #7:
# re-run the table, never re-read the pattern).

set -euo pipefail

SELF_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB_DIR="${REPO_ROOT}/scripts/lib"
NAME_AWK="${LIB_DIR}/workspace_sibling_names.awk"
SCAN_AWK="${LIB_DIR}/workspace_siblings_pathed.awk"
CASES="${LIB_DIR}/workspace_siblings_cases.txt"

MIN_CRATES="${MIN_CRATES:-70}"
MIN_MANIFESTS="${MIN_MANIFESTS:-60}"

for prog in "$NAME_AWK" "$SCAN_AWK"; do
    if [ ! -f "$prog" ]; then
        printf 'ERROR: %s is missing - the scanner cannot run.\n' "$prog" >&2
        exit 2
    fi
done

# ---------------------------------------------------------------------------
# File discovery.
# ---------------------------------------------------------------------------
list_manifests() {
    local root="$1"
    if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
        git -C "$root" ls-files -z '*Cargo.toml' | tr '\0' '\n' | sed "s|^|${root}/|"
    else
        find "$root" -name Cargo.toml -type f -not -path '*/target/*' | sort
    fi
}

# The manifests that DEFINE in-tree crate names: the root facade plus everything
# under crates/, at any depth.
list_name_sources() {
    local root="$1" f
    [ -f "$root/Cargo.toml" ] && printf '%s\n' "$root/Cargo.toml"
    while IFS= read -r f; do
        case "$f" in "$root"/crates/*) printf '%s\n' "$f" ;; esac
    done < <(list_manifests "$root")
}

intree_names() {
    local root="$1" f
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        if [ "$f" = "$root/Cargo.toml" ]; then
            awk -v ROOT=1 -f "$NAME_AWK" "$f"
        else
            awk -v ROOT=0 -f "$NAME_AWK" "$f"
        fi
    done < <(list_name_sources "$root") | sort -u
}

# scan_manifest <file-on-disk> <name-for-report> <namefile>
scan_manifest() {
    awk -v FILE="$2" -v NAMEFILE="$3" -f "$SCAN_AWK" "$1"
}

# ---------------------------------------------------------------------------
# --self-test
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    TD="$(mktemp -d)"
    if [ -z "${TD:-}" ] || [ ! -d "$TD" ]; then
        printf 'FAIL: could not create a temp dir for the case table.\n' >&2
        exit 1
    fi
    trap 'rm -rf "${TD:?}"' EXIT

    # The name set the table runs against: the real divergent trio, in both
    # spellings, so the table exercises package name, lib name, and alias.
    printf '%s\n' trueno trueno-db trueno_db realizar aprender-compute aprender-db \
        > "$TD/names.txt"

    fails=0
    ncases=0

    if [ ! -f "$CASES" ]; then
        printf 'FAIL: the case table %s is missing.\n' "$CASES" >&2
        exit 1
    fi

    # ---- the case table, driven from scripts/lib/workspace_siblings_cases.txt
    #
    # `LINE <fail|pass> <decl>` is wrapped in a [dependencies] header and run
    # through the REAL scanner. `FRAG <flag|clean> <label>` … `END` is a whole
    # manifest fragment, verbatim.
    probe_line() {
        printf '[dependencies]\n%s\n' "$1" > "$TD/probe.toml"
        scan_manifest "$TD/probe.toml" 'probe.toml' "$TD/names.txt"
    }
    eval_line() {
        local want="$1" text="$2" got
        ncases=$((ncases + 1))
        got="$(probe_line "$text")"
        if [ "$want" = fail ] && [ -z "$got" ]; then
            printf 'CASE-TABLE FAIL [must-fail] not flagged: %s\n' "$text" >&2
            fails=$((fails + 1))
        elif [ "$want" = pass ] && [ -n "$got" ]; then
            printf 'CASE-TABLE FAIL [must-pass] false positive: %s\n         -> %s\n' "$text" "$got" >&2
            fails=$((fails + 1))
        fi
    }
    eval_frag() {
        local want="$1" label="$2" got
        ncases=$((ncases + 1))
        got="$(scan_manifest "$TD/frag.toml" 'frag.toml' "$TD/names.txt")"
        if [ "$want" = flag ] && [ -z "$got" ]; then
            printf 'FRAGMENT FAIL [%s] expected a finding, got none\n' "$label" >&2
            fails=$((fails + 1))
        elif [ "$want" = clean ] && [ -n "$got" ]; then
            printf 'FRAGMENT FAIL [%s] false positive: %s\n' "$label" "$got" >&2
            fails=$((fails + 1))
        fi
    }

    # Strip leading whitespace into $LTRIM. A bash-regex capture rather than the
    # nested `${t#"${t%%[![:space:]]*}"}` idiom: same result, and no nested
    # parameter expansion for bashrs to trip over (SC2296).
    ltrim() {
        [[ $1 =~ ^[[:space:]]*(.*)$ ]]
        LTRIM="${BASH_REMATCH[1]}"
    }

    mode=out
    want=""
    label=""
    while IFS= read -r raw || [ -n "$raw" ]; do
        if [ "$mode" = frag ]; then
            if [ "$raw" = "END" ]; then
                mode=out
                eval_frag "$want" "$label"
            else
                printf '%s\n' "$raw" >> "$TD/frag.toml"
            fi
            continue
        fi
        case "$raw" in ''|'#'*) continue ;; esac
        kw="${raw%%[[:space:]]*}"
        ltrim "${raw#"$kw"}"; rest="$LTRIM"
        want="${rest%%[[:space:]]*}"
        ltrim "${rest#"$want"}"; arg="$LTRIM"
        case "$kw" in
            LINE) eval_line "$want" "$arg" ;;
            FRAG) label="$arg"; : > "$TD/frag.toml"; mode=frag ;;
            *)
                printf 'CASE-TABLE FAIL: unknown keyword %s in %s\n' "$kw" "$CASES" >&2
                fails=$((fails + 1))
                ;;
        esac
    done < "$CASES"

    if [ "$mode" = frag ]; then
        printf 'CASE-TABLE FAIL: unterminated FRAG %s (missing END)\n' "$label" >&2
        fails=$((fails + 1))
    fi
    # Vacuity, applied to the table itself: a table that ran no cases proves
    # nothing, and would print the same "OK" as one that ran them all.
    if [ "$ncases" -lt 20 ]; then
        printf 'CASE-TABLE FAIL: only %s cases parsed from %s - the table did not load.\n' \
            "$ncases" "$CASES" >&2
        fails=$((fails + 1))
    fi

    # ---- end-to-end mutation probe on a throwaway tree --------------------
    # A passing regex table does not prove the driver wires up: extending a
    # guard's scope requires re-mutating IN that scope. Build a miniature repo,
    # inject the violation, assert RED; remove it, assert GREEN.
    mk_tree() {
        local d
        d="$(mktemp -d "$TD/tree.XXXXXX")"
        mkdir -p "$d/scripts/lib" "$d/crates/aprender-compute" "$d/crates/aprender-db"
        cp "$SELF_PATH" "$d/scripts/check_workspace_siblings_pathed.sh"
        cp "$NAME_AWK" "$SCAN_AWK" "$d/scripts/lib/"
        printf '[workspace]\nmembers = ["crates/aprender-compute"]\n\n[package]\nname = "aprender"\nversion = "0.1.0"\n\n[lib]\nname = "aprender"\n\n[workspace.dependencies]\ntrueno = { path = "crates/aprender-compute", version = "0.1.0", package = "aprender-compute" }\n' \
            > "$d/Cargo.toml"
        printf '[package]\nname = "aprender-compute"\nversion = "0.1.0"\n\n[lib]\nname = "trueno"\n' \
            > "$d/crates/aprender-compute/Cargo.toml"
        printf '%s' "$d"
    }
    # Rewrite crates/aprender-db's [dependencies] block. The block arrives on ONE
    # line with `\n` escapes, expanded by `printf %b`: a multi-line literal
    # (heredoc or quoted) would be identical to bash but unparseable to bashrs,
    # which reads TOML's `name = "value"` as a malformed shell assignment and
    # then desyncs over the rest of the file.
    write_db() {
        {
            printf '[package]\nname = "aprender-db"\nversion = "0.1.0"\n\n'
            printf '[lib]\nname = "trueno_db"\n\n'
            printf '[dependencies]\n'
            printf '%b' "$2"
        } > "$1/crates/aprender-db/Cargo.toml"
    }
    run_tree() {
        (cd "$1" && MIN_CRATES=4 MIN_MANIFESTS=3 \
            bash scripts/check_workspace_siblings_pathed.sh >/dev/null 2>&1)
    }

    t="$(mk_tree)"
    write_db "$t" 'trueno = { workspace = true }\nserde = "1"\n'
    if ! run_tree "$t"; then
        printf 'MUTATION FAIL: the clean baseline tree is already RED\n' >&2
        fails=$((fails + 1))
    fi

    write_db "$t" 'trueno = "0.16"\nserde = "1"\n'
    if run_tree "$t"; then
        printf 'MUTATION FAIL: stayed GREEN with `trueno = "0.16"` in a member manifest\n' >&2
        fails=$((fails + 1))
    fi

    # The `[lib] name` half specifically: a package-name-only guard passes here.
    write_db "$t" 'trueno-db = "0.3"\n'
    if run_tree "$t"; then
        printf 'MUTATION FAIL: stayed GREEN with `trueno-db = "0.3"` (lib-name shadow)\n' >&2
        fails=$((fails + 1))
    fi

    # A violation in a NESTED manifest, which the name globs never reach but the
    # scan scope must.
    write_db "$t" 'trueno = { workspace = true }\n'
    mkdir -p "$t/crates/aprender-db/wasm-pkg"
    printf '[package]\nname = "trueno-db-wasm"\nversion = "0.1.0"\n\n[dependencies]\ntrueno = "0.7.1"\n' \
        > "$t/crates/aprender-db/wasm-pkg/Cargo.toml"
    if run_tree "$t"; then
        printf 'MUTATION FAIL: stayed GREEN with a violation in a NESTED manifest\n' >&2
        fails=$((fails + 1))
    fi
    rm -rf "${t:?}/crates/aprender-db/wasm-pkg"
    if ! run_tree "$t"; then
        printf 'MUTATION FAIL: did not return GREEN after the mutation was removed\n' >&2
        fails=$((fails + 1))
    fi

    # Vacuity: a tree with no in-tree crates must FAIL, not pass empty-handed.
    v="$(mktemp -d "$TD/vac.XXXXXX")"
    mkdir -p "$v/scripts/lib"
    cp "$SELF_PATH" "$v/scripts/check_workspace_siblings_pathed.sh"
    cp "$NAME_AWK" "$SCAN_AWK" "$v/scripts/lib/"
    printf '[package]\nname = "x"\nversion = "0.1.0"\n' > "$v/Cargo.toml"
    if (cd "$v" && bash scripts/check_workspace_siblings_pathed.sh >/dev/null 2>&1); then
        printf 'VACUITY FAIL: reported clean on a tree with no workspace-local crates\n' >&2
        fails=$((fails + 1))
    fi

    if [ "$fails" -ne 0 ]; then
        printf '\nSELF-TEST FAILED (%s cases wrong)\n' "$fails" >&2
        exit 1
    fi
    printf 'self-test OK: %s cases from %s, plus 6 end-to-end tree probes.\n' \
        "$ncases" "${CASES#"$REPO_ROOT"/}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Normal mode.
# ---------------------------------------------------------------------------
printf '=== workspace-local siblings must be pathed, never crates.io (check_workspace_siblings_pathed.sh) ===\n'

NAMEFILE="$(mktemp)"
FINDINGS="$(mktemp)"
PROBE_DIR="$(mktemp -d)"
cleanup() {
    rm -f "$NAMEFILE" "$FINDINGS"
    if [ -n "${PROBE_DIR:-}" ] && [ "$PROBE_DIR" != / ] && [ -d "$PROBE_DIR" ]; then
        rm -rf "${PROBE_DIR:?}"
    fi
}
trap cleanup EXIT

intree_names "$REPO_ROOT" > "$NAMEFILE"
ncrates="$(grep -c . "$NAMEFILE" || true)"

nmanifests=0
while IFS= read -r m; do
    [ -n "$m" ] || continue
    [ -f "$m" ] || continue
    nmanifests=$((nmanifests + 1))
    scan_manifest "$m" "${m#"$REPO_ROOT"/}" "$NAMEFILE" >> "$FINDINGS"
done < <(list_manifests "$REPO_ROOT")

nfind="$(grep -c . "$FINDINGS" || true)"

# --- vacuity 1 + 2: did the search have any chance at all? -----------------
if [ "$ncrates" -lt "$MIN_CRATES" ]; then
    printf 'FAIL (vacuity): enumerated %s workspace-local crate name(s), expected >= %s.\n' \
        "$ncrates" "$MIN_CRATES" >&2
    printf '  The name extraction has gone blind. Fix the extraction, not this floor.\n' >&2
    exit 1
fi
if [ "$nmanifests" -lt "$MIN_MANIFESTS" ]; then
    printf 'FAIL (vacuity): scanned %s manifest(s), expected >= %s.\n' \
        "$nmanifests" "$MIN_MANIFESTS" >&2
    printf '  The file discovery has gone blind. Fix the discovery, not this floor.\n' >&2
    exit 1
fi

# --- vacuity 3: positive control ------------------------------------------
# Prove the scanner can still find the thing, before believing it found nothing.
printf '[dependencies]\ntrueno = "0.16"\n' > "$PROBE_DIR/Cargo.toml"
probe_out="$(scan_manifest "$PROBE_DIR/Cargo.toml" 'POSITIVE-CONTROL' "$NAMEFILE")"
if [ -z "$probe_out" ]; then
    printf 'FAIL (vacuity): the positive control was NOT flagged.\n' >&2
    printf '  A synthetic `trueno = "0.16"` went unseen, so a clean verdict from this\n' >&2
    printf '  run would mean nothing. The scanner or the name set is broken.\n' >&2
    exit 1
fi

if [ "$nfind" -gt 0 ]; then
    printf '\n%s workspace-local sibling(s) declared as crates.io dependencies:\n\n' "$nfind" >&2
    while IFS=$'\t' read -r file lineno name cls text; do
        [ -n "$file" ] || continue
        printf '  %s %s:%s\n' "$cls" "$file" "$lineno" >&2
        printf '           %s\n' "$text" >&2
        printf '           `%s` is served from this workspace; this pulls a second copy from the registry.\n' "$name" >&2
    done < "$FINDINGS"
    printf '\nEach of these compiles a SECOND, type-incompatible copy of a crate the\n' >&2
    printf 'workspace already builds from source, and reintroduces the published-crate\n' >&2
    printf 'dependency cycle APR-MONO removed. Consume the path alias instead:\n' >&2
    printf '  trueno = { workspace = true }     # in a member manifest\n' >&2
    printf '  trueno = { path = "crates/aprender-compute", version = "...", package = "aprender-compute" }\n' >&2
    printf 'Confirm with: cargo tree --workspace --duplicates\n' >&2
    exit 1
fi

printf 'PASS: %s manifest(s) scanned against %s workspace-local crate name(s); every\n' \
    "$nmanifests" "$ncrates"
printf '      declaration of a workspace-local crate resolves to a path or the workspace alias.\n'
printf '      (positive control flagged, so this absence is a measurement, not a blind spot)\n'
exit 0
