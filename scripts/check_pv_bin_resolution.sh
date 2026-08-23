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

# The identity marker `pv --version` must carry (#2559). Not decoration: the
# operator settled 2026-08-21 that this binary KEEPS the name `pv` even though
# pv(1) the pipe viewer, the crates.io `pv` crate and (until #2553) the aprender
# facade all claim it, which makes this string the whole mitigation.
PV_IDENTITY='(aprender provable-contracts verifier)'

# 0. The marker this file tests must be the one the binary prints and the one
#    pv_bin.sh decides on -- retyping a literal into a guard is how a guard ends
#    up proving a string nobody emits. Same defence as EXTRACTOR_MISMATCH in
#    scripts/check_pv_version_parse.sh.
if ! grep -qF -- "$PV_IDENTITY" crates/aprender-contracts-cli/src/lib.rs; then
    bad "IDENTITY_MISMATCH: this file tests '$PV_IDENTITY' but aprender-contracts-cli does not print it"
fi
if ! grep -v '^[[:space:]]*#' scripts/pv_bin.sh | grep -qF -- "$PV_IDENTITY"; then
    bad "IDENTITY_MISMATCH: pv_bin.sh does not decide on '$PV_IDENTITY' -- the resolver is back to semver-only"
fi

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
    # ONE invocation, then every read off the SAME captured text: the semver and
    # the identity have to describe the same binary and the same run.
    got_all=$("$RESOLVED" --version 2>&1)
    got_first=$(awk 'NR==1{print; exit}' <<< "$got_all")
    got_semver=$(awk 'NR==1{print $2; exit}' <<< "$got_all")

    # SEMVER, POSITIONALLY -- not whole-line equality against "pv $CRATE_VERSION".
    # This block used to do exactly that, and it was correct only while the
    # version line WAS the bare name and a semver. #2559 appended an identity
    # suffix on purpose (four things claim the name `pv`, and the operator
    # settled that this binary keeps it, so the version line IS the
    # disambiguation mechanism), the suffix rode along into the compared string,
    # and this guard went RED against a perfectly fresh HEAD build:
    #     FAIL: $PV reports 'pv 0.63.0 (aprender provable-contracts verifier)'
    #           but the crate is 0.63.0
    # Same class as the last-field extractor #2559 already had to replace in
    # pv_bin.sh (scripts/check_pv_version_parse.sh): a guard's parser pinned to a
    # string that another change deliberately rewrote. Position 2 of line 1 is
    # the shape pinned from the other side by
    # crates/aprender-contracts-cli/tests/version_identity.rs
    # (`semver_stays_the_second_field_of_the_first_line`).
    if [ "$got_semver" = "$CRATE_VERSION" ]; then
        note "\$PV reports semver '$got_semver', matching the crate"
    else
        bad "\$PV reports semver '$got_semver' but the crate is $CRATE_VERSION (first line: $got_first)"
    fi

    # IDENTITY, on the same line -- the property #2559 exists to provide, and
    # therefore the property this guard has to assert. A semver match alone is
    # satisfied by pv(1) the pipe viewer, whose versions are 1.x today but whose
    # numbering shares the whole 0.x/1.x space; asserting only the number would
    # bless any binary that happened to collide.
    case "$got_first" in
        *"$PV_IDENTITY"*)
            note "\$PV identifies itself: '$got_first'" ;;
        *)
            bad "\$PV does not carry the identity marker '$PV_IDENTITY'. First --version line: '$got_first'" ;;
    esac
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
# 2c. MUTATION, IDENTITY: a pv at the RIGHT version that prints the bare
#     `pv <semver>` must be REFUSED.
#
#     This is the case the semver assertion cannot see, and it is not
#     hypothetical: `pv 0.63.0` is byte-for-byte what pv(1) the pipe viewer and
#     the crates.io `pv` crate print, which is exactly why #2559 added an
#     identity. A resolver that proved freshness from the number alone would
#     hand the release gate a pipe viewer the moment the two numbers collided,
#     and every version assertion in this file would stay green.
#
#     The version is taken from cargo, so the ONLY difference between this case
#     and its 2d control is the identity suffix -- the property under test.
mkdir -p "$TMP/bare"
printf '#!/usr/bin/env bash\nprintf "pv %s\\n"\n' "$CRATE_VERSION" > "$TMP/bare/pv"
chmod +x "$TMP/bare/pv"

if ( set -euo pipefail; PV_BIN="$TMP/bare/pv" . scripts/pv_bin.sh ) >"$TMP/bare.log" 2>&1; then
    bad "pv_bin.sh ACCEPTED a bare 'pv $CRATE_VERSION' with no identity (that is what pv(1) prints)"
elif grep -qi 'identif' "$TMP/bare.log"; then
    note "refuses a correctly-versioned pv that does not identify itself, and says so"
else
    bad "refused the unidentified pv, but not on identity grounds -- the message never says what was wrong"
    sed 's/^/      /' "$TMP/bare.log" >&2
fi

# 2d. CONTROL for 2c: the SAME synthesized script, the SAME version, plus the
#     identity suffix, must be ACCEPTED. Without it, 2c is also satisfied by a
#     pv_bin.sh that refuses every synthesized binary for some unrelated reason.
mkdir -p "$TMP/identified"
printf '#!/usr/bin/env bash\nprintf "pv %s %s\\n"\n' "$CRATE_VERSION" "$PV_IDENTITY" \
    > "$TMP/identified/pv"
chmod +x "$TMP/identified/pv"

if ( set -euo pipefail; PV_BIN="$TMP/identified/pv" . scripts/pv_bin.sh ) >"$TMP/identified.log" 2>&1; then
    note "accepts the same version once it carries the identity marker"
else
    bad "pv_bin.sh refused a pv at the crate version carrying '$PV_IDENTITY'"
    sed 's/^/      /' "$TMP/identified.log" >&2
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
