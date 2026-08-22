#!/usr/bin/env bash
# check_pv_bin_resolution.sh - scripts/pv_bin.sh must hand back the `pv` THIS
# checkout builds, and must REFUSE anything else.
#
# WHY A SECOND GUARD, WHEN check_apr_bin_pinned.sh ALREADY COVERS pv.
# That guard is a text scan: since #2552 it carries BARE_PV/PATHRES_PV classes
# and proves no surface SPELLS a bare `pv`. It says nothing about whether the
# resolver every surface now depends on actually works. A repo can be 100%
# pinned to a resolver that silently returns the wrong binary and every text
# scan stays green.
#
# The repo already draws exactly this line for `apr` -- check_apr_bin_pinned.sh
# scans, check_apr_bin_resolution.sh runs the resolver -- and pv had only the
# scanning half. pv_bin.sh is now on the release-certification path
# (scripts/dogfood_surfaces.sh, Makefile `contracts`), so the half that runs it
# is the half that matters: a stale pv reported 253 test refs / 51 missing where
# the HEAD build reported 371 / 27 on the same tree in the same second (#2552).
#
# Every assertion is paired with a control in the opposite direction. A refusal
# check that refuses everything is worse than none, so each REFUSE case is
# followed by an ACCEPT case differing in exactly the property under test.
#
# The stale binary is SYNTHESIZED, never found. Writing this against the dev
# box's real ~/.cargo/bin/pv (0.49.0 today) would pass here and vacuously pass
# everywhere else, CI included, where no stale pv exists.

set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

fails=0
TMP=$(mktemp -d)
cleanup() {
    if [ -n "${TMP:-}" ] && [ "$TMP" != / ] && [ -d "$TMP" ]; then
        rm -rf "$TMP"
    fi
}
trap cleanup EXIT

note() { printf 'OK  %s\n' "$*"; }
bad()  { printf 'FAIL: %s\n' "$*" >&2; fails=$((fails + 1)); }

printf '=== pv_bin.sh must resolve the HEAD-built pv (check_pv_bin_resolution.sh) ===\n'

# ---------------------------------------------------------------------------
# 1. It resolves at all -- under `set -euo pipefail`, deliberately.
#
#    A resolver is sourced by scripts that set their own options;
#    scripts/dogfood-book.sh runs under `set -u`. A resolution test run beneath
#    a lax shell would report success for a library that dies on an unbound
#    variable in its caller.
# Split across lines rather than `if ! ( ...; [ -x "$PV" ] )`: bashrs reads a
# `(` on the same line as a `[` as SC1028, and this repo's shell-lint ratchet is
# shrink-only.
resolve_rc=0
(
    set -euo pipefail
    . scripts/pv_bin.sh || exit 1
    test -x "$PV"
) >"$TMP/resolve.log" 2>&1 || resolve_rc=$?
if [ "$resolve_rc" -ne 0 ]; then
    bad "pv_bin.sh did not resolve an executable pv under 'set -euo pipefail'"
    sed 's/^/      /' "$TMP/resolve.log" >&2
else
    note "resolves under set -euo pipefail"
fi

RESOLVED=$( set -euo pipefail; . scripts/pv_bin.sh >/dev/null 2>&1 || exit 1; printf '%s' "$PV" ) || RESOLVED=''
[ -n "$RESOLVED" ] || bad "pv_bin.sh exported no \$PV"

# The version comes from cargo, never hardcoded, or this file becomes one more
# thing to bump on every release.
CRATE_VERSION=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
    | jq -r '.packages[] | select(.name=="aprender-contracts-cli") | .version')
if [ -z "$CRATE_VERSION" ] || [ "$CRATE_VERSION" = null ]; then
    bad "cargo metadata reported no version for aprender-contracts-cli"
    CRATE_VERSION='<unknown>'
fi

if [ -n "$RESOLVED" ]; then
    got=$("$RESOLVED" --version 2>&1 | head -1)
    if [ "$got" = "pv $CRATE_VERSION" ]; then
        note "\$PV reports 'pv $CRATE_VERSION', matching the crate"
    else
        bad "\$PV reports '$got' but the crate is $CRATE_VERSION"
    fi
fi

# 1b. It must be cargo's own output for THIS workspace, not something off PATH.
#     This is the assertion that a PATH fallback would fail: ~/.cargo/bin/pv is
#     not under any target_directory.
TARGET_DIR=$(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.target_directory')
if [ -n "$RESOLVED" ]; then
    case "$RESOLVED" in
        "$TARGET_DIR"/*) note "\$PV lives under the target dir of this workspace" ;;
        *) bad "\$PV ($RESOLVED) is not under the cargo target_directory ($TARGET_DIR)" ;;
    esac
fi

# ---------------------------------------------------------------------------
# 2. MUTATION: a stale pv must be REFUSED, and the refusal must name the
#    mismatch. `pv 0.0.1` is a three-line script, so this tests the same
#    decision on a machine that has never had a stale pv.
mkdir -p "$TMP/stale"
printf '#!/usr/bin/env bash\nprintf "pv 0.0.1\\n"\n' > "$TMP/stale/pv"
chmod +x "$TMP/stale/pv"

if ( set -euo pipefail; PV_BIN="$TMP/stale/pv" . scripts/pv_bin.sh ) >"$TMP/stale.log" 2>&1; then
    bad "pv_bin.sh ACCEPTED a pv reporting 0.0.1 (expected refusal)"
elif grep -q '0\.0\.1' "$TMP/stale.log"; then
    note "refuses a stale pv, and names the version it saw"
else
    bad "refused the stale pv but did not say what it found"
    sed 's/^/      /' "$TMP/stale.log" >&2
fi

# 2b. CONTROL: the SAME override path with a correct binary must be ACCEPTED.
#     Without this, a pv_bin.sh that refused every PV_BIN would pass check 2
#     while being useless.
if [ -n "$RESOLVED" ]; then
    if ( set -euo pipefail; PV_BIN="$RESOLVED" . scripts/pv_bin.sh ) >"$TMP/good.log" 2>&1; then
        note "accepts a correctly-versioned pv through the same override path"
    else
        bad "pv_bin.sh refused a pv reporting the crate version"
        sed 's/^/      /' "$TMP/good.log" >&2
    fi
fi

# ---------------------------------------------------------------------------
# 3. The library must be OPTION-NEUTRAL. `set` in a SOURCED file mutates the
#    CALLER's shell; apr_bin.sh shipped that bug and killed the nightly six
#    lines in (CLAUDE.md). check_sourced_libs_option_neutral.sh checks the TEXT;
#    this checks the BEHAVIOUR, which is what actually breaks.
opts_before=$( set -o | LC_ALL=C sort | md5sum )
opts_after=$( . scripts/pv_bin.sh >/dev/null 2>&1 || true; set -o | LC_ALL=C sort | md5sum )
if [ "$opts_before" = "$opts_after" ]; then
    note "sourcing pv_bin.sh leaves the shell options of its caller untouched"
else
    bad "sourcing pv_bin.sh MUTATED the shell options of its caller"
fi

# 3b. CONTROL for 3: prove the fingerprint can detect a change at all, or
#     check 3 is satisfied by a broken measurement rather than a well-behaved
#     library. `pipefail` is a trap here -- this file's own `set -euo pipefail`
#     has already set it, so flipping it changes nothing and the control would
#     be vacuous. `noglob` is off, so flipping it is a real change.
opts_mutated=$( set -o noglob; set -o | LC_ALL=C sort | md5sum )
if [ "$opts_before" = "$opts_mutated" ]; then
    bad "the option fingerprint cannot detect 'set -o noglob' - check 3 is vacuous"
else
    note "the option fingerprint detects a deliberate 'set -o noglob'"
fi

printf '\n'
if [ "$fails" -gt 0 ]; then
    printf '%s assertion(s) failed.\n' "$fails" >&2
    exit 1
fi
printf 'PASS: pv_bin.sh resolves the HEAD-built pv, refuses a stale one, and is option-neutral.\n'
