#!/usr/bin/env bash
# crux_judge_bin.sh - resolve the CRUX judge binary (tools/aprender-crux-judge)
# and export it as $CRUX_JUDGE.
#
# Sourceable:  . scripts/lib/crux_judge_bin.sh || exit 2
#
# The judge, the answer oracles and the prompt certifier are one Rust binary.
# It replaced crux_inference_judge.py, crux_oracles.py and crux_prompt_certify.py
# byte for byte (tools/aprender-crux-judge/README.md lists where it differs).
#
# BUILT FROM THIS TREE, EVERY TIME. The default is `cargo build --release
# --locked` with an explicit --target-dir inside this checkout. The flag
# overrides any CARGO_TARGET_DIR the shell exports, so no other worktree's build
# can be served. Cargo's freshness check then makes the binary this tree's
# sources: a no-op when nothing changed, a rebuild when anything did.
#
# CRUX_JUDGE_BIN=<path> skips the build for a host that must not compile. It is
# used as given, and the line on stderr says so, because nothing here can prove
# which tree built it.
#
# Option-neutral (scripts/check_sourced_libs_option_neutral.sh): no `set` here.
# A failure returns 2 (ENV), the exit code every caller gives a missing tool.

crux_judge_resolve() {
    local root crate target
    if [ -n "${CRUX_JUDGE_BIN:-}" ]; then
        if [ ! -x "$CRUX_JUDGE_BIN" ]; then
            printf 'crux_judge_bin: CRUX_JUDGE_BIN=%s is not an executable file\n' "$CRUX_JUDGE_BIN" >&2
            return 2
        fi
        printf 'crux_judge_bin: using CRUX_JUDGE_BIN=%s as given (not built from this tree)\n' "$CRUX_JUDGE_BIN" >&2
        CRUX_JUDGE=$CRUX_JUDGE_BIN
        export CRUX_JUDGE
        return 0
    fi
    root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd) || return 2
    crate="$root/tools/aprender-crux-judge"
    target="$crate/target"
    if ! command -v cargo >/dev/null 2>&1; then
        printf 'crux_judge_bin: cargo is missing, and CRUX_JUDGE_BIN is unset\n' >&2
        return 2
    fi
    if ! cargo build --quiet --release --locked --manifest-path "$crate/Cargo.toml" \
        --target-dir "$target" >&2; then
        printf 'crux_judge_bin: cargo build of %s failed\n' "$crate" >&2
        return 2
    fi
    CRUX_JUDGE="$target/release/aprender-crux-judge"
    if [ ! -x "$CRUX_JUDGE" ]; then
        printf 'crux_judge_bin: the build left no binary at %s\n' "$CRUX_JUDGE" >&2
        return 2
    fi
    export CRUX_JUDGE
}

crux_judge_resolve
