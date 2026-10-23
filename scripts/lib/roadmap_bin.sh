# roadmap_bin.sh -- resolve THE pv-roadmap binary (#5902), built from this tree.
#
# Source it, never execute it:
#     . "$REPO_ROOT/scripts/lib/roadmap_bin.sh" || exit 1
#     "$ROADMAP_BIN" diff check --base A --head B
#
# pv-roadmap replaced scripts/lib/roadmap_{fragments,diff,merge}.py and scripts/roadmap_trim.py.
# It is its own bin of aprender-contracts-cli (not a `pv` subcommand, so it adds nothing to the
# `pv` surface gate), found the way pv_bin.sh finds pv-sat: cargo is the freshness authority.
# `cargo build --message-format=json` names the artifact it just produced, so the path comes from
# the build, never from a guess about the target dir (the dev shell and a script launch land in
# different ones, see pv_bin.sh).
#
# ROADMAP_BIN already set and executable wins without a build: the case tables that run a
# scratch repo export it so the scratch checkout needs no cargo workspace.
#
# OPTION-NEUTRAL: sets no shell options (this file is sourced). Failure is the RETURN STATUS.

roadmap_bin_root() {
    rb_here="${BASH_SOURCE[0]:-$0}"
    ( cd -- "$(dirname -- "$rb_here")/../.." && pwd )
}

roadmap_bin_resolve() {
    if [ -n "${ROADMAP_BIN:-}" ] && [ -x "$ROADMAP_BIN" ]; then
        return 0
    fi
    rb_root=$(roadmap_bin_root) || return 1
    if ! command -v cargo >/dev/null 2>&1; then
        printf 'roadmap_bin.sh: cargo not found; cannot build pv-roadmap\n' >&2
        return 1
    fi
    rb_log=$(mktemp) || return 1
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
    export ROADMAP_BIN
    return 0
}

roadmap_bin_resolve
