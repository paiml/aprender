#!/usr/bin/env bash
#
# bump-version.sh — bump the release version across EVERY workspace, including
# the one `cargo set-version --workspace` cannot reach.
#
# WHY THIS EXISTS (aprender#2559)
# -------------------------------
# The release bump was `cargo set-version --workspace <v>` plus a written-down
# reminder that three more files have to ride in the same commit. `--workspace`
# means "every member of THIS workspace"; `crates/facades/` is `exclude`d from
# the root, so cargo does not see it and does not warn about it. The three files
# it cannot touch are:
#
#   crates/facades/provable-contracts/Cargo.toml         upstream version pin
#   crates/facades/provable-contracts-macros/Cargo.toml  upstream version pin
#   crates/facades/Cargo.lock                            resolved versions
#
# Omitting them does not fail quietly — it fails LOUDLY and in the worst place:
# the bump commit reds its own CI, because check_facade_compat.sh rows R3 and R5
# compare those exact files against the workspace version. A reminder in prose is
# not a mechanism; this script is, and the existing R3/R5 rows stay as the
# backstop that catches a hand-edited bump that skipped it.
#
# WHAT IS DELIBERATELY *NOT* BUMPED
# ---------------------------------
# `crates/facades/Cargo.toml`'s own `[workspace.package] version` (0.4.0). The
# facade crate NAMES version independently of the aprender version line, and
# that independence is deliberate and documented (aprender#2546): 0.3.1 -> 0.4.0
# signals "something changed" without claiming a 0.63.0 history these names do
# not have. This script asserts that value is unchanged after it runs, so a
# future edit cannot quietly couple the two lines.
#
# What DOES track the aprender version is the facades' `upstream = { version =
# "..." }` requirement — the version of `aprender-contracts*` a PUBLISHED facade
# resolves from the registry. Two different version lines in the same directory,
# which is exactly why this is scripted rather than remembered.
#
#   bash scripts/bump-version.sh 0.64.0
#   bash scripts/bump-version.sh --check          # are all workspaces consistent NOW?
#   bash scripts/bump-version.sh --self-test      # case table
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# The facade manifests whose `upstream` pin tracks the aprender version. Derived,
# not listed: a fourth facade added later is picked up without editing this file,
# and the signpost facade (no `upstream` line at all) is skipped by the same rule
# that makes it publishable at any point in the cascade.
# aprender#2628, FOURTH INSTANCE — and the only one on the WRITE side.
#
# This enumeration used `grep -q '^upstream *=.*version *= *"'`, which matches the
# INLINE spelling only. Cargo treats the multi-line `[dependencies.upstream]`
# table as IDENTICAL, and a manifest written that way vanished from this list.
# The consequence is worse than a missed check, because BOTH the writer and its
# own validator read this function: `set_facade_pins` never rewrote the file,
# `--check` never inspected it, and `--check` then reported rc=0 "all consistent"
# with a stale pin. Ask what the loop iterates, and whether the defect removes
# items from it.
#
# Reading now goes through scripts/lib/facade_pin.py, which parses the manifest
# as TOML. See that file for why tomllib rather than `cargo metadata`.
facade_pinned_manifests() {  # root
    local f
    for f in "$1"/crates/facades/*/Cargo.toml; do
        [ -f "$f" ] || continue
        if [ -n "$(facade_pin "$f")" ]; then
            printf '%s\n' "$f"
        fi
    done
    return 0
}

facade_pin() {  # manifest
    python3 "$REPO_ROOT/scripts/lib/facade_pin.py" read "$1"
}

# The facades' own package version — the one that must NOT move.
facade_own_version() {  # root
    sed -n 's/^version *= *"\([^"]*\)".*/\1/p' "$1/crates/facades/Cargo.toml" | head -1
}

root_version() {  # root
    grep -E '^version = "' "$1/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/'
}

# Rewrite every `upstream = { ..., version = "X", ... }` pin to $2. Anchored at
# line start and scoped to the version field so it cannot touch the `path`, the
# `package` rename, or any other key on the line.
set_facade_pins() {  # root new_version
    local f n=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        # Both spellings (#2628). The enumeration was taught to SEE the
        # multi-line table before this was taught to REWRITE it; self-test row 8
        # caught that half-fix, which is exactly what the row exists for.
        python3 "$REPO_ROOT/scripts/lib/facade_pin.py" write "$f" "$2" || {
            printf 'bump-version: no upstream pin rewritten in %s\n' "$f" >&2
            return 1
        }
        n=$((n + 1))
    done < <(facade_pinned_manifests "$1")
    printf '%s\n' "$n"
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--self-test" ]; then
    fails=0
    TD="$(mktemp -d)" || exit 1
    case "$TD" in /tmp/*|/var/folders/*) : ;; *) printf 'bad tmp\n'; exit 1 ;; esac
    trap 'rm -rf "${TD:?}"' EXIT

    mkdir -p "$TD/crates/facades/reexport" "$TD/crates/facades/signpost"
    printf '[workspace]\n[workspace.package]\nversion = "0.4.0"\n' \
        > "$TD/crates/facades/Cargo.toml"
    printf '[dependencies]\nupstream = { path = "../../up", version = "0.63.0", package = "up" }\n' \
        > "$TD/crates/facades/reexport/Cargo.toml"
    printf '[package]\nname = "signpost"\n# NO [dependencies].\n' \
        > "$TD/crates/facades/signpost/Cargo.toml"

    # Row 1: the pinned facade is rewritten.
    n="$(set_facade_pins "$TD" 0.64.0)"
    if [ "$(facade_pin "$TD/crates/facades/reexport/Cargo.toml")" = "0.64.0" ]; then
        printf 'ok    row 1 a re-export facade pin is bumped\n'
    else
        printf 'FAIL  row 1 pin not bumped\n'; fails=1
    fi

    # Row 2: exactly one manifest was eligible. Without this, row 1 passes even
    # if the rewrite sprayed every file in the tree.
    if [ "$n" = "1" ]; then
        printf 'ok    row 2 exactly the 1 pinned manifest was rewritten (signpost skipped)\n'
    else
        printf 'FAIL  row 2 rewrote %s manifest(s), expected 1\n' "$n"; fails=1
    fi

    # Row 3: THE INDEPENDENCE. The facades' own version must be untouched. This
    # is the row that stops a future "simplification" from coupling the two
    # version lines (#2546).
    if [ "$(facade_own_version "$TD")" = "0.4.0" ]; then
        printf 'ok    row 3 crates/facades/Cargo.toml version = 0.4.0 is UNTOUCHED\n'
    else
        printf 'FAIL  row 3 the facades own version moved to %s -- #2546 says it must not\n' \
            "$(facade_own_version "$TD")"; fails=1
    fi

    # Row 4: the signpost facade, which has no upstream at all, is left alone
    # rather than acquiring a pin.
    if ! grep -q 'upstream' "$TD/crates/facades/signpost/Cargo.toml"; then
        printf 'ok    row 4 the signpost facade did not acquire an upstream pin\n'
    else
        printf 'FAIL  row 4 an upstream pin appeared in the signpost facade\n'; fails=1
    fi

    # Row 5: the rewrite preserves the rest of the dependency line. A pin that
    # bumps the version but eats `path` or `package` breaks the build in a way
    # the version check would still call correct.
    if grep -q 'path = "../../up"' "$TD/crates/facades/reexport/Cargo.toml" \
       && grep -q 'package = "up"' "$TD/crates/facades/reexport/Cargo.toml"; then
        printf 'ok    row 5 path and package survive the rewrite\n'
    else
        printf 'FAIL  row 5 the rewrite damaged the dependency line:\n'
        sed 's/^/        /' "$TD/crates/facades/reexport/Cargo.toml"; fails=1
    fi

    # Row 6: idempotence. A bump re-run must be a no-op, not a corruption.
    before="$(cat "$TD/crates/facades/reexport/Cargo.toml")"
    set_facade_pins "$TD" 0.64.0 >/dev/null
    if [ "$before" = "$(cat "$TD/crates/facades/reexport/Cargo.toml")" ]; then
        printf 'ok    row 6 re-running the same bump is a no-op\n'
    else
        printf 'FAIL  row 6 re-running the bump changed the file again\n'; fails=1
    fi

    # rows 7-8 — aprender#2628, the falsifier for the defect this file had.
    # The multi-line dependency table cargo treats as IDENTICAL to the inline
    # form. The old grep enumeration could not SEE it (row 7), and after that was
    # fixed the old sed still could not REWRITE it (row 8) -- row 8 caught that
    # half-fix on the first run.
    mkdir -p "$TD/crates/facades/multiline"
    printf '[package]\nname = "multiline"\nversion = "0.4.0"\n\n[dependencies.upstream]\npath = "../../aprender-contracts"\nversion = "0.63.0"\npackage = "aprender-contracts"\n' \
        > "$TD/crates/facades/multiline/Cargo.toml"
    # Row 7 must exercise the ENUMERATION, not the reader — those are different
    # functions and it was the enumeration that was blind. An earlier version of
    # this row called facade_pin directly and stayed GREEN under a mutation that
    # reverted the enumeration to its grep, i.e. it did not test what its label
    # claimed.
    # NOTE the here-string. `facade_pinned_manifests "$TD" | grep -q ...` returns
    # 141 under `set -o pipefail` even when grep MATCHES: grep -q exits on the
    # first hit, the producer takes SIGPIPE, and pipefail hands the pipeline the
    # producer's status. It is input-size dependent, so it goes green on a
    # one-line fixture and red on a two-line one. Capture first, then match.
    _enum="$(facade_pinned_manifests "$TD")"
    if grep -q '/multiline/Cargo.toml$' <<< "$_enum"; then
        printf 'ok    row 7 a multi-line [dependencies.upstream] table is ENUMERATED (#2628)\n'
    else
        printf 'FAIL  row 7 a multi-line table is invisible -- it would be neither bumped nor\n'
        printf '      checked, and --check would report ok with a stale pin\n'; fails=1
    fi
    set_facade_pins "$TD" 0.64.0 >/dev/null
    if grep -q 'version = "0.64.0"' "$TD/crates/facades/multiline/Cargo.toml"; then
        printf 'ok    row 8 the multi-line pin is actually REWRITTEN by the bump\n'
    else
        printf 'FAIL  row 8 the multi-line pin was enumerated but not rewritten\n'; fails=1
    fi
    if grep -q 'package = "aprender-contracts"' "$TD/crates/facades/multiline/Cargo.toml" \
       && grep -q 'path = "../../aprender-contracts"' "$TD/crates/facades/multiline/Cargo.toml"; then
        printf 'ok    row 9 the multi-line rewrite left path and package intact\n'
    else
        printf 'FAIL  row 9 the multi-line rewrite damaged a sibling key\n'; fails=1
    fi

    [ "$fails" -eq 0 ] || { printf '\nSELF-TEST FAILED\n'; exit 1; }
    printf '\nSELF-TEST PASSED (9/9)\n'
    exit 0
fi

# ---------------------------------------------------------------------------
if [ "${1:-}" = "--check" ]; then
    WS="$(root_version "$REPO_ROOT")"
    printf '=== version consistency across workspaces ===\n'
    printf 'root workspace:            %s\n' "$WS"
    printf 'crates/facades own version: %s  (INDEPENDENT by design, #2546)\n' \
        "$(facade_own_version "$REPO_ROOT")"
    rc=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        got="$(facade_pin "$f")"
        if [ "$got" = "$WS" ]; then
            printf 'ok    %s upstream pin %s\n' "${f#"$REPO_ROOT"/}" "$got"
        else
            printf 'FAIL  %s upstream pin %s, workspace is %s\n' \
                "${f#"$REPO_ROOT"/}" "$got" "$WS"
            rc=1
        fi
    done < <(facade_pinned_manifests "$REPO_ROOT")
    if ( cd "$REPO_ROOT/crates/facades" && cargo metadata --format-version 1 --locked ) >/dev/null 2>&1; then
        printf 'ok    crates/facades/Cargo.lock matches its manifests (--locked)\n'
    else
        printf 'FAIL  crates/facades/Cargo.lock is stale\n'; rc=1
    fi
    exit "$rc"
fi

NEW="${1:-}"
if ! printf '%s' "$NEW" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
    printf 'Usage: %s <x.y.z> | --check | --self-test\n' "$0" >&2
    exit 2
fi

OLD_FACADE_OWN="$(facade_own_version "$REPO_ROOT")"
printf '=== bumping to %s ===\n' "$NEW"

printf -- '--- root workspace (cargo set-version --workspace) ---\n'
if ! ( cd "$REPO_ROOT" && cargo set-version --workspace "$NEW" ); then
    printf 'FAIL  cargo set-version failed (is cargo-edit installed?)\n' >&2
    exit 1
fi

# THE PART --workspace CANNOT REACH.
printf -- '--- crates/facades (excluded from the root workspace) ---\n'
N="$(set_facade_pins "$REPO_ROOT" "$NEW")"
printf 'rewrote %s facade upstream pin(s)\n' "$N"
# Vacuity: zero rewrites means the pattern stopped matching, not that there is
# nothing to do. Silently bumping nothing is the exact failure being fixed.
if [ "${N:-0}" -lt 1 ]; then
    printf 'FAIL (vacuity): no facade upstream pin matched. The REWRITE is broken,\n' >&2
    printf '      not the tree -- crates/facades still needs bumping by hand.\n' >&2
    exit 1
fi

# Regenerate the second lockfile. `cargo metadata` resolves and writes it; no
# build, about a second. check_lockfile_current.sh resolves the ROOT workspace
# only and cannot see this file.
if ! ( cd "$REPO_ROOT/crates/facades" && cargo metadata --format-version 1 ) >/dev/null; then
    printf 'FAIL  could not regenerate crates/facades/Cargo.lock\n' >&2
    exit 1
fi
printf 'regenerated crates/facades/Cargo.lock\n'

# THE INDEPENDENCE ASSERTION. `cargo set-version --workspace` run from inside
# crates/facades would move this, and so would a plausible "fix" to the rewrite
# above. #2546 says it must not move.
NOW_FACADE_OWN="$(facade_own_version "$REPO_ROOT")"
if [ "$NOW_FACADE_OWN" != "$OLD_FACADE_OWN" ]; then
    printf 'FAIL  crates/facades/Cargo.toml version moved %s -> %s. The facade crate\n' \
        "$OLD_FACADE_OWN" "$NOW_FACADE_OWN" >&2
    printf '      NAMES version independently of the aprender line (#2546) -- these\n' >&2
    printf '      names have no %s history. Revert that file.\n' "$NEW" >&2
    exit 1
fi
printf 'crates/facades/Cargo.toml version left at %s (independent, #2546)\n' "$NOW_FACADE_OWN"

printf -- '\n--- files that MUST ride in the bump commit ---\n'
( cd "$REPO_ROOT" && git status --porcelain -- crates/facades ) | sed 's/^/  /'

printf -- '\n--- verifying ---\n'
exec bash "$REPO_ROOT/scripts/bump-version.sh" --check
