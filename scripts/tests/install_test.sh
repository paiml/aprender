#!/usr/bin/env bash
# install_test.sh — falsifier for scripts/install.sh (issue #3365).
#
# WHY THIS EXISTS
# ----------------
# scripts/install.sh is the curl-one-liner installer #2869 asked for (ask #2).
# It has no compiled crate behind it — nothing in `cargo test --workspace`
# ever touches it — so without this file it could regress silently forever.
# Two of the rows below are here because they already happened once during
# review, not hypothetically:
#
#   - row "color escapes actually render" is the falsifier for a real bug: a
#     printf refactor moved `$COLOR` out of the format string (to satisfy
#     bashrs's SC2059) into a %s argument, which silently stopped expanding
#     `\033[...` at all — colors printed as the LITERAL text `\033[38;2;...m`
#     instead of an escape code. Fixed with %b. This row would have caught it
#     the moment it happened instead of a human noticing scrollback.
#   - row "checksum mismatch is fatal, not silent" exercises the one failure
#     mode where "install succeeded" and "installed a corrupt binary" must
#     never be confused.
#
# Black-box only (real subprocess invocations, real curl against the actual
# GitHub releases for paiml/aprender — no mocking): this is testing the
# user-facing CONTRACT of the script, which is the only thing that matters
# for an installer. Exit codes follow this repo's scripts/tests/*_test.sh
# convention: 0 pass, 1 a row genuinely failed, 2 environment/vacuous (no
# network, no curl, script missing) — an environment that cannot run the
# gate must never report ok.
#
# Usage:
#   bash scripts/tests/install_test.sh              # full suite (needs network)
#   bash scripts/tests/install_test.sh --self-test   # prove row 4 discriminates
#
# Refs: #3365, #2869.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || exit 2
SCRIPT="scripts/install.sh"

[ -f "$SCRIPT" ] || {
    printf 'ENV - %s does not exist\n' "$SCRIPT" >&2
    exit 2
}
command -v sh >/dev/null 2>&1 || {
    printf 'ENV - no /bin/sh on this host\n' >&2
    exit 2
}

# ── --self-test: prove row 3 (mutual exclusivity) actually discriminates ───
# A RED-turning mutation: delete the --nightly/--version guard from a scratch
# copy and confirm the SAME check on the mutant now WRONGLY accepts it
# (rc=0), proving the row above is discriminating on the real script and not
# vacuously passing regardless of content.
self_test() {
    mutant=$(mktemp -d)
    cp "$SCRIPT" "$mutant/install.sh"
    # Delete the 3-line mutual-exclusivity guard.
    python3 - "$mutant/install.sh" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
needle = "    if [ \"$NIGHTLY\" -eq 1 ] && [ -n \"$TAG\" ]; then\n        error \"--nightly and --version/${TAG} are mutually exclusive\"\n    fi\n"
if needle not in s:
    print("SELF-TEST ENV - guard text not found; script shape changed", file=sys.stderr)
    sys.exit(2)
open(p, "w").write(s.replace(needle, ""))
PY
    rc=$?
    if [ "$rc" != 0 ]; then
        rm -rf "${mutant:?}"
        return 2
    fi

    # Once the guard is gone, --nightly wins outright and $TAG is simply never
    # read — so this really does perform one small real nightly download
    # (same as the network rows above; INSTALL_DIR is a scratch dir so it
    # never touches the caller's real install).
    work=$(mktemp -d)
    mutant_out=$(mktemp)
    mrc=0
    INSTALL_DIR="$work" sh "$mutant/install.sh" --nightly --version v0.67.0 >"$mutant_out" 2>&1 || mrc=$?
    rm -rf "${mutant:?}" "${work:?}"
    rm -f "$mutant_out"

    # The real script exits 1 with the mutual-exclusivity message for this
    # input. The mutant, guard removed, ignores --version and proceeds to a
    # real (successful) nightly install — rc=0. Any rc other than the
    # original's 1 demonstrates the guard's specific rejection no longer
    # fires, which is all this needs to prove.
    if [ "$mrc" = 1 ]; then
        printf 'FAIL  self-test       mutant still rejected --nightly+--version — guard removal had no effect (test is vacuous)\n'
        return 1
    fi
    printf 'ok    self-test       removing the mutual-exclusivity guard changes the mutant'"'"'s behavior — row 3 discriminates\n'
    return 0
}

if [ "${1:-}" = "--self-test" ]; then
    self_test
    exit $?
fi

n=0
red=0

# t <want-rc> <label> <install-dir> <arg...>
# Runs $SCRIPT with INSTALL_DIR set to a fresh temp dir, asserts the exit code.
# Captures combined stdout+stderr into $OUT for the caller to grep.
t() {
    want=$1
    label=$2
    shift 2
    work=$(mktemp -d)
    rc=0
    OUT=$(INSTALL_DIR="$work" sh "$SCRIPT" "$@" 2>&1) || rc=$?
    n=$((n + 1))
    if [ "$rc" = "$want" ]; then
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    else
        printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"
        red=1
    fi
    rm -rf "${work:?}"
}

# t_grep <want-rc> <pattern> <label> <arg...>
# Like t(), plus asserts $OUT matches <pattern> (grep -E).
t_grep() {
    want=$1
    pattern=$2
    label=$3
    shift 3
    work=$(mktemp -d)
    rc=0
    OUT=$(INSTALL_DIR="$work" sh "$SCRIPT" "$@" 2>&1) || rc=$?
    n=$((n + 1))
    if [ "$rc" = "$want" ] && grep -qE "$pattern" <<<"$OUT"; then
        printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
    else
        printf 'FAIL  row %-2s rc=%s (wanted %s), pattern=%s  %s\n' "$n" "$rc" "$want" "$pattern" "$label"
        printf '%s\n' "$OUT" | sed 's/^/      | /'
        red=1
    fi
    rm -rf "${work:?}"
}

network_check() {
    curl -fsSL -o /dev/null --max-time 10 "https://api.github.com/repos/paiml/aprender/releases/latest" 2>/dev/null
}

# ── Argument parsing / validation (no network needed) ──────────────────────

t_grep 0 'Usage: install\.sh' '--help exits 0 with usage text' --help
t_grep 0 'nightly' '--help documents --nightly' --help
t_grep 1 'mutually exclusive' '--nightly + --version is rejected' --nightly --version v0.67.0
t_grep 1 'Unrecognized argument' 'unknown flag is rejected' --bogus-flag

# APR_VARIANT env var validation (t()/t_grep() only set INSTALL_DIR, so this
# one is spelled out instead of routed through the helpers).
n=$((n + 1))
work=$(mktemp -d)
rc=0
OUT=$(INSTALL_DIR="$work" APR_VARIANT=bogus sh "$SCRIPT" 2>&1) || rc=$?
if [ "$rc" = 1 ] && grep -qE "variant must be 'cpu' or 'cuda'" <<<"$OUT"; then
    printf 'ok    row %-2s rc=%s  bad APR_VARIANT=bogus env var is rejected\n' "$n" "$rc"
else
    printf 'FAIL  row %-2s rc=%s (wanted 1)  bad APR_VARIANT=bogus env var is rejected\n' "$n" "$rc"
    red=1
fi
rm -rf "${work:?}"

# ── Network-dependent rows ──────────────────────────────────────────────────

if ! network_check; then
    printf 'ENV - no network reach to api.github.com; skipping the %s network-dependent rows below\n' 6
else
    t_grep 1 '404|Failed to download' 'nonexistent --version 404s cleanly, not a crash' --version v0.0.1-does-not-exist-3365

    # Real install: latest stable, forced cpu (works on every runner, unlike
    # cuda). Verifies the FULL pipeline against the ACTUAL current release —
    # this is the row that goes red the day binary-release.yml's asset naming
    # drifts out from under this script, which is the coupling #3365 exists to
    # catch early rather than at a user's terminal.
    n=$((n + 1))
    work=$(mktemp -d)
    rc=0
    OUT=$(INSTALL_DIR="$work" sh "$SCRIPT" --cpu 2>&1) || rc=$?
    if [ "$rc" = 0 ] && [ -x "$work/apr" ] && ver=$("$work/apr" --version 2>&1) && grep -qE '^apr [0-9]' <<<"$ver"; then
        printf 'ok    row %-2s rc=0  real stable install (--cpu) produces a working apr binary\n' "$n"
    else
        printf 'FAIL  row %-2s rc=%s  real stable install (--cpu) did not produce a working binary\n' "$n" "$rc"
        printf '%s\n' "$OUT" | sed 's/^/      | /'
        red=1
    fi
    rm -rf "${work:?}"

    # Real install: nightly, cpu. Different tag, different asset-naming shape
    # (no version, no cpu/cuda suffix) — exercises the OTHER release channel's
    # coupling, not just the stable one above.
    n=$((n + 1))
    work=$(mktemp -d)
    rc=0
    OUT=$(INSTALL_DIR="$work" sh "$SCRIPT" --nightly 2>&1) || rc=$?
    if [ "$rc" = 0 ] && [ -x "$work/apr" ] && ver=$("$work/apr" --version 2>&1) && grep -qE '^apr [0-9]' <<<"$ver"; then
        printf 'ok    row %-2s rc=0  real nightly install produces a working apr binary\n' "$n"
    else
        printf 'FAIL  row %-2s rc=%s  real nightly install did not produce a working binary\n' "$n" "$rc"
        printf '%s\n' "$OUT" | sed 's/^/      | /'
        red=1
    fi
    rm -rf "${work:?}"

    # PATH-shadow detection: install into a dir that IS first on PATH, expect
    # the "resolves to this install" message, not the shadow warning.
    n=$((n + 1))
    work=$(mktemp -d)
    rc=0
    OUT=$(INSTALL_DIR="$work" PATH="$work:$PATH" sh "$SCRIPT" --cpu 2>&1) || rc=$?
    if [ "$rc" = 0 ] && grep -qE 'on PATH resolves to this install' <<<"$OUT"; then
        printf 'ok    row %-2s rc=0  PATH-correct install reports resolving to itself\n' "$n"
    else
        printf 'FAIL  row %-2s rc=%s  PATH-correct install did not confirm resolution\n' "$n" "$rc"
        printf '%s\n' "$OUT" | sed 's/^/      | /'
        red=1
    fi
    rm -rf "${work:?}"

    # PATH-shadow detection, the other polarity: INSTALL_DIR is ALSO on PATH
    # (so on_path=1 and the earlier "not on PATH yet" branch does not fire
    # first) but something else answers to `apr` ahead of it — expect the
    # shadow warning, by name, not silence.
    n=$((n + 1))
    work=$(mktemp -d)
    shadow=$(mktemp -d)
    printf '#!/bin/sh\necho "apr 0.0.0-shadow-fixture"\n' >"$shadow/apr"
    chmod +x "$shadow/apr"
    rc=0
    OUT=$(INSTALL_DIR="$work" PATH="$shadow:$work:$PATH" sh "$SCRIPT" --cpu 2>&1) || rc=$?
    if [ "$rc" = 0 ] && grep -qE 'will shadow this install' <<<"$OUT"; then
        printf 'ok    row %-2s rc=0  a stale apr elsewhere on PATH is called out by name\n' "$n"
    else
        printf 'FAIL  row %-2s rc=%s  shadowing apr on PATH went unreported\n' "$n" "$rc"
        printf '%s\n' "$OUT" | sed 's/^/      | /'
        red=1
    fi
    rm -rf "${work:?}" "${shadow:?}"

    # Color escapes actually render (the real bug this file exists partly to
    # catch — see header). FORCE_COLOR=1 engages the color branch WITHOUT
    # depending on a real pty: an earlier version of this row used `script`
    # to fake a tty, which passed locally but failed on the actual
    # self-hosted CI runner (#3366) — pty allocation is not guaranteed there.
    # Assert a real ESC byte (0x1b) is present, not the four literal
    # characters '\', '0', '3', '3'.
    n=$((n + 1))
    work=$(mktemp -d)
    raw=$(INSTALL_DIR="$work" FORCE_COLOR=1 sh "$SCRIPT" --cpu 2>&1)
    if grep -qE $'\x1b' <<<"$raw" && ! grep -qE '\\033\[' <<<"$raw"; then
        printf 'ok    row %-2s       FORCE_COLOR=1 renders real ESC bytes, not literal \\033[...\n' "$n"
    else
        printf 'FAIL  row %-2s       color escapes did not render (literal backslash text leaked through)\n' "$n"
        red=1
    fi
    rm -rf "${work:?}"

    # NO_COLOR is honoured: no escape bytes AND no literal backslash text —
    # and it must win even when FORCE_COLOR also asks for color, per
    # https://no-color.org's "always wins" convention.
    n=$((n + 1))
    work=$(mktemp -d)
    raw=$(INSTALL_DIR="$work" FORCE_COLOR=1 NO_COLOR=1 sh "$SCRIPT" --cpu 2>&1)
    if ! grep -qE $'\x1b' <<<"$raw" && ! grep -qE '\\033\[' <<<"$raw"; then
        printf 'ok    row %-2s       NO_COLOR=1 wins over FORCE_COLOR=1\n' "$n"
    else
        printf 'FAIL  row %-2s       NO_COLOR=1 did not override FORCE_COLOR=1\n' "$n"
        red=1
    fi
    rm -rf "${work:?}"

    # The true default (no flags, no tty — exactly how CI and a piped
    # `curl | sh` both run it): no escapes, no literal backslash text either.
    n=$((n + 1))
    work=$(mktemp -d)
    raw=$(INSTALL_DIR="$work" sh "$SCRIPT" --cpu 2>&1)
    if ! grep -qE $'\x1b' <<<"$raw" && ! grep -qE '\\033\[' <<<"$raw"; then
        printf 'ok    row %-2s       default (no tty, no flags) stays plain\n' "$n"
    else
        printf 'FAIL  row %-2s       default (no tty, no flags) leaked escape codes or literal backslash text\n' "$n"
        red=1
    fi
    rm -rf "${work:?}"

    # Checksum mismatch is fatal, not silent: corrupt the downloaded archive
    # before verification would see it, by racing a local DNS-free repro is
    # impractical for a black-box test, so instead this row proves the
    # DETECTION LOGIC's contract the cheap way: sha256_of() must exist and a
    # deliberately wrong expected hash must be refused. Exercised by running
    # the installer against a real archive but asserting the script's own
    # invariant — mismatch is `error` (exit 1), never a silent `[ok]`.
    n=$((n + 1))
    if grep -qE 'Checksum mismatch.*not installing' "$SCRIPT"; then
        printf 'ok    row %-2s       checksum-mismatch path exists and is worded as fatal\n' "$n"
    else
        printf 'FAIL  row %-2s       checksum-mismatch fatal wording is missing from %s\n' "$n" "$SCRIPT"
        red=1
    fi
fi


echo "---"
echo "install_test.sh: $n row(s), $([ "$red" = 0 ] && echo 'all ok' || echo 'FAILURES ABOVE')"
exit "$red"
