#!/usr/bin/env bash
# check_clippy_current_stable.sh - the CEILING gate (aprender#2370).
#
# `scripts/check_msrv.sh` proves the workspace still builds on its declared FLOOR.
# Nothing proved the ceiling: every gate we own - `make tier1/2/3`, the sovereign-ci
# `lint` job - runs through `rust-toolchain.toml`, which pins one exact release. So
# clippy findings introduced by NEWER clippy releases accumulate with nothing to
# report them, and the first person to see them is whoever's toolchain the pin does
# not reach. That is aprender#2370: a fresh mbp ran `make`, got clippy 1.96 (a
# homebrew rustc is not rustup-managed, so the pin is silently inert), and 28 errors
# came out of `cargo clippy -- -D warnings` in `tier2` - none of which any gate here
# had ever seen. The pin is not the bug; the pin being the ONLY thing we ever lint
# under is the bug.
#
# This script closes that by linting on whatever stable is CURRENT, and failing.
#
# It refuses to pass vacuously. Three ways this gate could report a meaningless
# green, all of them checked before any linting happens:
#   1. `stable` resolves to something OLDER than the pin (a stale local rustup that
#      never self-updated) - then "clean on current stable" is a lie about a
#      toolchain we already lint under. FAIL.
#   2. clippy is not installed for that toolchain - `cargo clippy` would error, but
#      we say so in one line instead of a cargo backtrace. FAIL.
#   3. the version comparator is wrong. Shell string comparison says 1.9 > 1.10 and
#      1.99 > 1.100, which would make check (1) pass on exactly the drift it exists
#      to catch. The comparator therefore ships a case table and runs it on EVERY
#      invocation (`--self-test` runs only the table and exits).
#
# What it does NOT prove: clippy's lint set is not monotonic, so "clean on current
# stable" does not imply "clean on every release between the pin and it". Measured
# while fixing #2370: the tree is clean on 1.93 (the pin), 1.96 and 1.97 (then
# current), and 1.95 alone reported 8 `collapsible_match` findings that neither
# neighbour emits. So a green here still leaves an intermediate release able to go
# red; that is a smaller hole than the one this closes, and it shrinks on its own as
# people upgrade. Do not read a pass as "every toolchain is clean".
#
# RED-turning mutation (verified): revert any fix from aprender#2370 - e.g. put
# `.map(std::num::NonZero::get).unwrap_or(1)` back in crates/aprender-common/src/sys.rs
# - and this exits 1 with `error: called map(<f>).unwrap_or(<a>) on a Result value`.
# Mutating the comparator (`ver_ge` returning 0 unconditionally) turns the self-test
# RED without needing a toolchain at all.
#
# Usage:
#   bash scripts/check_clippy_current_stable.sh              # self-test, then lint
#   bash scripts/check_clippy_current_stable.sh --self-test  # comparator table only
set -euo pipefail
cd "$(dirname "$0")/.." || exit 2

# ── version comparator ──────────────────────────────────────────────
# ver_ge A B  → success when version A >= version B, numerically, per component.
# Missing components read as 0, so "1.93" == "1.93.0".
ver_ge() {
    local a="$1" b="$2" i a_i b_i
    local -a av bv
    IFS='.' read -r -a av <<<"$a"
    IFS='.' read -r -a bv <<<"$b"
    for i in 0 1 2; do
        a_i="${av[i]:-0}"
        b_i="${bv[i]:-0}"
        # strip any pre-release/suffix noise, keep leading digits only
        a_i="${a_i%%[!0-9]*}"
        b_i="${b_i%%[!0-9]*}"
        a_i="${a_i:-0}"
        b_i="${b_i:-0}"
        if [ "$a_i" -gt "$b_i" ]; then return 0; fi
        if [ "$a_i" -lt "$b_i" ]; then return 1; fi
    done
    return 0
}

# ── the case table ──────────────────────────────────────────────────
# Every row is "A B expected", expected ∈ {ge, lt}. The 1.9/1.10 and 1.99/1.100
# rows are the lexicographic traps: a string comparator gets exactly those wrong,
# and getting them wrong is what makes the vacuity check pass when it must fail.
self_test() {
    local -a cases=(
        "1.97.1 1.93.0 ge"   # newer stable than the pin - the normal case
        "1.93.0 1.97.1 lt"   # stale rustup - MUST be caught
        "1.93.0 1.93.0 ge"   # pin already IS current stable - still a real run
        "1.93 1.93.0 ge"     # missing patch component reads as 0
        "1.93.0 1.93 ge"
        "1.94.0 1.93.9 ge"   # minor outranks patch
        "1.10.0 1.9.0 ge"    # LEXICOGRAPHIC TRAP: "1.10" < "1.9" as strings
        "1.9.0 1.10.0 lt"
        "1.100.0 1.99.0 ge"  # LEXICOGRAPHIC TRAP: three-digit minor
        "1.99.0 1.100.0 lt"
        "2.0.0 1.999.999 ge" # major outranks everything
        "1.999.999 2.0.0 lt"
    )
    local row a b want got fails=0
    for row in "${cases[@]}"; do
        read -r a b want <<<"$row"
        if ver_ge "$a" "$b"; then got=ge; else got=lt; fi
        if [ "$got" != "$want" ]; then
            echo "  ✗ ver_ge $a $b → $got (expected $want)" >&2
            fails=$((fails + 1))
        fi
    done
    if [ "$fails" -ne 0 ]; then
        echo "check_clippy_current_stable: comparator self-test FAILED ($fails/${#cases[@]} rows)" >&2
        echo "  The vacuity check below relies on this comparator; refusing to lint." >&2
        return 1
    fi
    echo "✓ comparator self-test: ${#cases[@]}/${#cases[@]} rows"
    return 0
}

self_test || exit 1
if [ "${1:-}" = "--self-test" ]; then
    exit 0
fi

# ── resolve the pin and the current stable ──────────────────────────
PINNED="$(grep -m1 '^channel' rust-toolchain.toml | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' || true)"
if [ -z "$PINNED" ]; then
    echo "check_clippy_current_stable: rust-toolchain.toml pins a non-numeric channel;" >&2
    echo "  nothing to compare against - treating that as an unpinned tree is not safe." >&2
    exit 2
fi

echo "Pinned channel (rust-toolchain.toml): $PINNED"
echo "→ rustup toolchain install stable (clippy component)"
rustup toolchain install stable --profile minimal --component clippy --no-self-update

STABLE_RUSTC="$(rustup run stable rustc --version)"
STABLE_VER="$(printf '%s' "$STABLE_RUSTC" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)"
[ -n "$STABLE_VER" ] || {
    echo "check_clippy_current_stable: could not parse a version out of: $STABLE_RUSTC" >&2
    exit 2
}

# Prove clippy is really there - a missing component must not read as "no findings".
STABLE_CLIPPY="$(rustup run stable cargo clippy --version)" || {
    echo "check_clippy_current_stable: clippy is not installed for the stable toolchain." >&2
    echo "  Fix: rustup component add clippy --toolchain stable" >&2
    exit 2
}

echo "Current stable: $STABLE_RUSTC / $STABLE_CLIPPY"

if ! ver_ge "$STABLE_VER" "$PINNED"; then
    echo "✗ VACUOUS RUN REFUSED: 'stable' resolved to $STABLE_VER, which is OLDER than" >&2
    echo "  the pinned $PINNED. Linting under it would prove nothing this repo does not" >&2
    echo "  already prove on every PR. Fix: rustup self update && rustup update stable" >&2
    exit 2
fi

# ── the actual gate ─────────────────────────────────────────────────
# The sovereign-ci `lint` job's command, on current stable instead of the pin.
#
# THIS COMMENT USED TO CLAIM A SUPERSET THAT DOES NOT EXIST. It said the scope was
# "same packages, plus the root package's test/bench/example targets" -- but the
# root package HAS no test/bench/example targets. The root manifest is both
# [workspace] and [package] and declares no `default-members`, so
# `workspace_default_members` is 1 (the `aprender` facade alone), and that package
# declares exactly two selectable targets: `lib aprender` and `bin apr`. There is
# no root benches/ or examples/, and root tests/ holds one YAML fixture and no
# .rs. So `--all-targets` selects nothing that a bare `cargo clippy` would not,
# and this command's scope is EQUAL to `make tier2`'s, not a superset of it.
# Measured with `cargo metadata --no-deps` on 50d2bc2bb; aprender#2734.
#
# `--all-targets` stays, because it is the lint job's command verbatim and it is
# what makes this gate widen automatically the day a root target appears. What
# changes is the claim: do not read a pass here as covering targets the facade
# does not have. The 78 non-facade members' 1,782 test/bench/example/bin targets
# (981 examples, 665 tests, 108 benches, 28 bins) are outside BOTH forms: clippy
# reaches a member's LIB only, as a path dependency of the facade compiled under
# RUSTC_WORKSPACE_WRAPPER, and never its other targets. Corroborated by #2721,
# whose 71 findings from THIS command span 60+ member crates -- all in lib code.
#
# Do NOT read the status through a pipe (see CLAUDE.md "Verification Discipline" #1).
echo "→ cargo +stable clippy --all-targets -- -D warnings   (pin $PINNED → stable $STABLE_VER)"
if rustup run stable cargo clippy --all-targets -- -D warnings; then
    echo "✓ clean under current stable clippy ($STABLE_VER); the pin is $PINNED"
else
    rc=$?
    echo "✗ TOOLCHAIN CEILING DRIFT: the tree does NOT pass clippy on current stable" >&2
    echo "  ($STABLE_VER). It still passes on the pinned $PINNED, which is why no PR" >&2
    echo "  gate caught this - see aprender#2370." >&2
    echo "  Fix the findings above. Do not relax the lint; if one is genuinely wrong" >&2
    echo "  for this codebase, #[allow(...)] it with a comment giving the reason." >&2
    exit "$rc"
fi
