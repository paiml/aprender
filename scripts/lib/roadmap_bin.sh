# roadmap_bin.sh -- resolve THE pv-roadmap binary (#5902), built from this tree.
#
# Source it, never execute it. Sourcing only DEFINES functions; it runs nothing and builds
# nothing. A caller resolves where it first needs the binary, after its cheap early exits:
#     . "$REPO_ROOT/scripts/lib/roadmap_bin.sh" || exit 1
#     roadmap_bin_resolve || exit 2
#     "$ROADMAP_BIN" diff check --base A --head B
#
# pv-roadmap replaced scripts/lib/roadmap_{fragments,diff,merge}.py and scripts/roadmap_trim.py.
# It is its own bin of aprender-contracts-cli (not a `pv` subcommand, so it adds nothing to the
# `pv` surface gate).
#
# RESOLUTION ORDER
#   1. $ROADMAP_BIN_OVERRIDE, if set: the case tables that run a scratch repo export it so the
#      scratch checkout needs no cargo workspace. It is a variable only those harnesses set; the
#      resolved path is NEVER exported as $ROADMAP_BIN, so a developer shell cannot inherit a
#      stale binary from an earlier run (Verification Discipline items 3 and 8).
#   2. The cache: ${PV_ROADMAP_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/aprender/pv-roadmap}/<key>,
#      where <key> is a hash of the CONTENT of every file the binary is built from. A hit needs
#      no cargo and no build, so the pre-commit hook and a bare host stay fast once one build
#      has run; a source edit changes the key, so a hit can never be stale.
#   3. `cargo build`, then the artifact is copied into the cache under its key.
#
# OPTION-NEUTRAL: sets no shell options (this file is sourced). Failure is the RETURN STATUS.

roadmap_bin_root() {
    rb_here="${BASH_SOURCE[0]:-$0}"
    ( cd -- "$(dirname -- "$rb_here")/../.." && pwd )
}

# The cache key: the content hash of every tracked file of the two crates pv-roadmap is built
# from, plus the lockfile. Prints nothing and returns 1 outside a git checkout.
roadmap_bin_key() {
    rb_k_root="$1"
    rb_k_files=$( cd "$rb_k_root" && git ls-files -- crates/aprender-contracts-cli crates/aprender-contracts Cargo.lock 2>/dev/null ) || return 1
    [ -n "$rb_k_files" ] || return 1
    # git hash-object reads the WORKING-TREE bytes (one process for every path), so an
    # uncommitted edit changes the key. A tracked path missing on disk fails it: a miss, a build.
    rb_k_sums=$( cd "$rb_k_root" && printf '%s\n' "$rb_k_files" | git hash-object --stdin-paths 2>/dev/null ) || return 1
    printf '%s\n%s\n' "$rb_k_files" "$rb_k_sums" | sha256sum | cut -c1-24
}

roadmap_bin_resolve() {
    if [ -n "${ROADMAP_BIN_OVERRIDE:-}" ]; then
        if [ ! -x "$ROADMAP_BIN_OVERRIDE" ]; then
            printf 'roadmap_bin.sh: ROADMAP_BIN_OVERRIDE=%s is not executable\n' "$ROADMAP_BIN_OVERRIDE" >&2
            return 1
        fi
        ROADMAP_BIN="$ROADMAP_BIN_OVERRIDE"
        return 0
    fi
    rb_root=$(roadmap_bin_root) || return 1
    rb_cache="${PV_ROADMAP_CACHE:-${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/aprender/pv-roadmap}"
    rb_key=$(roadmap_bin_key "$rb_root") || rb_key=""
    if [ -n "$rb_key" ] && [ -x "$rb_cache/$rb_key" ]; then
        ROADMAP_BIN="$rb_cache/$rb_key"
        return 0
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        printf 'roadmap_bin.sh: no cached pv-roadmap for this tree and cargo not found; cannot build it\n' >&2
        return 1
    fi
    rb_log=$(mktemp) || return 1
    # `cargo build --message-format=json` names the artifact it just produced, so the path comes
    # from the build, never from a guess about the target dir (the dev shell and a script launch
    # land in different ones, see pv_bin.sh).
    if ! ( cd "$rb_root" && cargo build -q -p aprender-contracts-cli --bin pv-roadmap --message-format=json ) >"$rb_log" 2>/dev/null; then
        printf 'roadmap_bin.sh: cargo build of pv-roadmap failed in %s\n' "$rb_root" >&2
        ( cd "$rb_root" && cargo build -q -p aprender-contracts-cli --bin pv-roadmap ) >&2 || true
        rm -f -- "${rb_log:?}"
        return 1
    fi
    rb_exe=$(sed -n 's/.*"executable":"\([^"]*pv-roadmap\)".*/\1/p' "$rb_log" | tail -n 1)
    rm -f -- "${rb_log:?}"
    if [ -z "$rb_exe" ] || [ ! -x "$rb_exe" ]; then
        # An up-to-date build prints no compiler-artifact line for a cached unit; ask cargo where.
        rb_tgt=$( cd "$rb_root" && cargo metadata --no-deps --format-version 1 2>/dev/null | sed -n 's/.*"target_directory":"\([^"]*\)".*/\1/p' )
        rb_exe="$rb_tgt/debug/pv-roadmap"
    fi
    if [ ! -x "$rb_exe" ]; then
        printf 'roadmap_bin.sh: built pv-roadmap but found no executable (%s)\n' "$rb_exe" >&2
        return 1
    fi
    ROADMAP_BIN="$rb_exe"
    # Best effort: a read-only cache dir only costs the next caller a build.
    if [ -n "$rb_key" ] && mkdir -p "$rb_cache" 2>/dev/null; then
        rb_tmp="$rb_cache/.$rb_key.$$"
        if cp -- "$rb_exe" "$rb_tmp" 2>/dev/null && mv -f -- "$rb_tmp" "$rb_cache/$rb_key" 2>/dev/null; then
            ROADMAP_BIN="$rb_cache/$rb_key"
        else
            rm -f -- "$rb_tmp"
        fi
    fi
    return 0
}

return 0 2>/dev/null || true
