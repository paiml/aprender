#!/usr/bin/env bash
# check_dogfood_llm_probe_armed.sh - the deepest probe in the surface sweep must
# be able to RUN, and must be able to FAIL for a server reason rather than a
# harness reason.
#
# `scripts/dogfood_surfaces.sh` ends in a live probe that drives real generation
# through /v1/chat/completions via `probar llm test`. Two independent gaps meant
# that probe never executed a single time, and neither was visible from its
# output -- it reported a tidy `skip` or a confident `FAIL` either way:
#
#   1. It built aprender-test-cli WITHOUT `--features llm`. `Commands::Llm` is
#      declared in commands.rs with no `#[cfg]`, so clap advertises `llm` and
#      renders its `--help` in every build; only the HANDLER in main.rs is
#      gated. A featureless binary therefore PARSES `llm test` and then returns
#      "LLM features not enabled", which the sweep reported as
#      `FAIL probar llm test FAILED` -- blaming the server for a gap in the
#      harness. That `--help` renders fine without the feature is exactly why
#      this was invisible: `aprender-test-cli llm --help` proves nothing.
#
#   2. It skipped whenever DOGFOOD_PROBAR_CONFIG was unset, on the stated
#      grounds that no committed config existed. One did.
#
# Both are the anti-theater class this repo keeps re-learning: a probe that
# never runs is not a probe, and a gate nobody has seen turn RED is
# indistinguishable from a gate that cannot.
#
# METHOD: check the DECISION, not the prose. The feature flag is read off the
# actual cargo invocation the sweep runs, and the default config path is read
# off the actual parameter expansion and then required to EXIST on disk -- so
# deleting or renaming the fixture turns this RED even though the script text
# is untouched.
#
# Exit 0 = the probe is armed.
# Exit 1 = the probe is disarmed (prints which of the two gaps reopened).

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

cd "$(dirname "$0")/.." || exit 1

SWEEP="scripts/dogfood_surfaces.sh"

# Run the two assertions against a given repo root. Prints failures, returns
# non-zero if any fired. Used both for the real tree and for --self-test.
check_tree() {
    local root="$1" sweep="$1/scripts/dogfood_surfaces.sh" fails=0 build_line cfg_path

    if [ ! -f "$sweep" ]; then
        printf 'FAIL: %s does not exist\n' "$sweep"
        return 1
    fi

    # (1) The cargo invocation that produces the probe binary must request the
    # llm feature. Join continuation lines first: the invocation is wrapped
    # across three physical lines with trailing backslashes.
    build_line=$(sed -e :a -e '/\\$/N; s/\\\n//; ta' "$sweep" \
                 | grep -E 'cargo build .*-p aprender-test-cli' | head -1)
    if [ -z "$build_line" ]; then
        printf 'FAIL: no `cargo build -p aprender-test-cli` invocation found in %s\n' "$sweep"
        fails=$((fails + 1))
    elif ! grep -qE -- '--features[= ]+[^ ]*llm' <<< "$build_line" ; then
        printf 'FAIL: the probe binary is built WITHOUT --features llm.\n'
        printf '      clap still advertises `llm`, so the probe will parse and then\n'
        printf '      return "LLM features not enabled" -- reported as a server failure.\n'
        printf '      invocation: %s\n' "$build_line"
        fails=$((fails + 1))
    fi

    # (2) The probe must default to a committed config rather than skipping,
    # and that config must actually be there.
    cfg_path=$(grep -oE ':[[:space:]]+"\$\{DOGFOOD_PROBAR_CONFIG:=\$\{REPO_ROOT\}/[^"}]+\}"' "$sweep" \
               | head -1 | sed -E 's|.*\$\{REPO_ROOT\}/||; s/\}"$//')
    if [ -z "$cfg_path" ]; then
        printf 'FAIL: the probe does not default DOGFOOD_PROBAR_CONFIG to a committed\n'
        printf '      config, so it skips unless an operator exports one by hand.\n'
        fails=$((fails + 1))
    elif [ ! -f "$root/$cfg_path" ]; then
        printf 'FAIL: the probe defaults DOGFOOD_PROBAR_CONFIG to %s, which does not exist.\n' "$cfg_path"
        printf '      The default silently empties and the probe skips forever.\n'
        fails=$((fails + 1))
    fi

    return "$fails"
}

# --self-test: prove this checker turns RED on each gap independently, using the
# pre-fix text verbatim. A guard that has never been seen to fail is theater.
if [ "${1:-}" = "--self-test" ]; then
    TMP=$(mktemp -d)
    cleanup_selftest() {
        if [ -n "${TMP:-}" ] && [ "$TMP" != / ]; then
            rm -rf "$TMP"
        fi
    }
    trap cleanup_selftest EXIT

    rc_selftest=0

    # Case A: the real tree must be GREEN.
    mkdir -p "$TMP/a/scripts"
    cp "$SWEEP" "$TMP/a/scripts/"
    cfg_a=$(grep -oE ':[[:space:]]+"\$\{DOGFOOD_PROBAR_CONFIG:=\$\{REPO_ROOT\}/[^"}]+\}"' "$SWEEP" \
            | head -1 | sed -E 's|.*\$\{REPO_ROOT\}/||; s/\}"$//')
    mkdir -p "$TMP/a/$(dirname "$cfg_a")"
    touch "$TMP/a/$cfg_a"
    if check_tree "$TMP/a" > "$TMP/a.out" 2>&1; then
        printf 'self-test A (unmutated tree is GREEN): PASS\n'
    else
        printf 'self-test A (unmutated tree is GREEN): FAIL -- the guard reds on a good tree\n'
        cat "$TMP/a.out"
        rc_selftest=1
    fi

    # Case B: drop `--features llm` from the build -- gap (1), verbatim pre-fix.
    mkdir -p "$TMP/b/scripts"
    sed -e '/--features llm \\/d' "$SWEEP" > "$TMP/b/scripts/dogfood_surfaces.sh"
    mkdir -p "$TMP/b/$(dirname "$cfg_a")"
    touch "$TMP/b/$cfg_a"
    if check_tree "$TMP/b" > "$TMP/b.out" 2>&1; then
        printf 'self-test B (missing --features llm turns RED): FAIL -- guard stayed green\n'
        rc_selftest=1
    else
        grep -q 'WITHOUT --features llm' "$TMP/b.out" \
            && printf 'self-test B (missing --features llm turns RED): PASS\n' \
            || { printf 'self-test B: RED for the WRONG reason:\n'; cat "$TMP/b.out"; rc_selftest=1; }
    fi

    # Case C: the defaulted fixture is missing -- gap (2), the rename/delete case.
    mkdir -p "$TMP/c/scripts"
    cp "$SWEEP" "$TMP/c/scripts/"
    # deliberately do NOT create the fixture
    if check_tree "$TMP/c" > "$TMP/c.out" 2>&1; then
        printf 'self-test C (absent fixture turns RED): FAIL -- guard stayed green\n'
        rc_selftest=1
    else
        grep -q 'does not exist' "$TMP/c.out" \
            && printf 'self-test C (absent fixture turns RED): PASS\n' \
            || { printf 'self-test C: RED for the WRONG reason:\n'; cat "$TMP/c.out"; rc_selftest=1; }
    fi

    # Case D: the default expansion removed entirely -- the pre-fix skip-forever
    # form, where the probe was gated on an env var nobody sets in CI.
    mkdir -p "$TMP/d/scripts"
    grep -v 'DOGFOOD_PROBAR_CONFIG:=' "$SWEEP" > "$TMP/d/scripts/dogfood_surfaces.sh"
    mkdir -p "$TMP/d/$(dirname "$cfg_a")"
    touch "$TMP/d/$cfg_a"
    if check_tree "$TMP/d" > "$TMP/d.out" 2>&1; then
        printf 'self-test D (no committed default turns RED): FAIL -- guard stayed green\n'
        rc_selftest=1
    else
        grep -q 'does not default DOGFOOD_PROBAR_CONFIG' "$TMP/d.out" \
            && printf 'self-test D (no committed default turns RED): PASS\n' \
            || { printf 'self-test D: RED for the WRONG reason:\n'; cat "$TMP/d.out"; rc_selftest=1; }
    fi

    exit "$rc_selftest"
fi

if check_tree "$PWD"; then
    printf 'dogfood live LLM probe is armed (--features llm, committed config present)\n'
    exit 0
fi
exit 1
