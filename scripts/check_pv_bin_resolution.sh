#!/usr/bin/env bash
# check_pv_bin_resolution.sh - scripts/pv_bin.sh must resolve the binary THIS
# checkout builds, and must REFUSE anything else.
#
# WHY A SECOND GUARD. check_pv_bin_pinned.sh is a text scan: it proves no
# surface spells a bare `pv`. It says nothing about whether the resolver those
# surfaces now depend on actually works. A repo can be 100% pinned to a resolver
# that silently hands back the wrong binary, and every scanner would stay green.
# (This is the apr split too: check_apr_bin_pinned.sh scans, and
# check_apr_bin_resolution.sh runs the resolver.)
#
# Every assertion here is paired with a control in the opposite direction. An
# assertion that cannot fail is not an assertion, and a refusal check that
# refuses everything is worse than none - so each REFUSE case is followed by an
# ACCEPT case that differs in exactly the property under test.
#
# The stale binary is SYNTHESIZED, not found. Writing this against
# /home/noah/.cargo/bin/pv (which really is 0.49.0 today) would make the guard
# pass on this box and vacuously pass everywhere else, including CI, where no
# stale pv exists. A three-line fake that prints `pv 0.0.1` tests the same
# decision on every machine.

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

note() { printf '%s\n' "$*"; }
bad()  { printf 'FAIL: %s\n' "$*" >&2; fails=$((fails + 1)); }

printf '=== pv_bin.sh must resolve the HEAD-built binary (check_pv_bin_resolution.sh) ===\n'

# ---------------------------------------------------------------------------
# 1. Plain resolution. Run under `set -euo pipefail` deliberately: the first
#    draft of pv_bin.sh assigned PV via `PV=$(pv_bin_resolve)`, which runs the
#    function in a command-substitution SUBSHELL, so PV_CRATE_VERSION never
#    reached the parent. Under a caller with `set -u` -- scripts/dogfood-book.sh
#    is exactly that -- the sourcing died with `PV_CRATE_VERSION: unbound
#    variable` instead of resolving. A resolution test run under a lax shell
#    would have reported success.
if ! ( set -euo pipefail; . scripts/pv_bin.sh || exit 1; [ -x "$PV" ] ) >"$TMP/resolve.log" 2>&1; then
    bad "pv_bin.sh did not resolve an executable pv under 'set -euo pipefail'"
    sed 's/^/      /' "$TMP/resolve.log" >&2
else
    note "OK  resolves under set -euo pipefail"
fi

# The version the CHECKOUT defines, from cargo - never hardcoded, or this file
# becomes one more thing to bump on every release.
CRATE_VERSION=$(cargo metadata --no-deps --format-version 1 2>/dev/null \
    | jq -r '.packages[] | select(.name=="aprender-contracts-cli") | .version')
if [ -z "$CRATE_VERSION" ] || [ "$CRATE_VERSION" = null ]; then
    bad "cargo metadata did not report a version for aprender-contracts-cli"
    CRATE_VERSION='<unknown>'
fi

RESOLVED=$( set -euo pipefail; . scripts/pv_bin.sh >/dev/null 2>&1 || exit 1; printf '%s' "$PV" ) || RESOLVED=''
if [ -z "$RESOLVED" ]; then
    bad "pv_bin.sh exported no \$PV"
else
    got=$("$RESOLVED" --version 2>&1 | head -1)
    if [ "$got" = "pv $CRATE_VERSION" ]; then
        note "OK  \$PV reports 'pv $CRATE_VERSION', matching the crate"
    else
        bad "\$PV reports '$got' but the crate is $CRATE_VERSION"
    fi
fi

# It must be cargo's own output for THIS workspace, not something off PATH.
TARGET_DIR=$(cargo metadata --no-deps --format-version 1 2>/dev/null | jq -r '.target_directory')
case "$RESOLVED" in
    "$TARGET_DIR"/*) note "OK  \$PV lives under this workspace's target dir" ;;
    *) bad "\$PV ($RESOLVED) is not under cargo's target_directory ($TARGET_DIR)" ;;
esac

# ---------------------------------------------------------------------------
# 2. MUTATION: a stale pv must be REFUSED.
mkdir -p "$TMP/stale"
cat > "$TMP/stale/pv" <<'STALE'
#!/usr/bin/env bash
printf 'pv 0.0.1\n'
STALE
chmod +x "$TMP/stale/pv"

if ( set -euo pipefail; PV_BIN="$TMP/stale/pv" . scripts/pv_bin.sh ) >"$TMP/stale.log" 2>&1; then
    bad "pv_bin.sh ACCEPTED a pv reporting 0.0.1 (expected refusal)"
else
    if grep -q 'STALE pv BINARY' "$TMP/stale.log"; then
        note "OK  refuses a stale pv, and says why"
    else
        bad "refused the stale pv but printed no 'STALE pv BINARY' diagnosis"
    fi
fi

# 2b. CONTROL: the same override mechanism with a CORRECT binary must be
#     accepted. Without this, a pv_bin.sh that refused every PV_BIN would pass
#     2 while being useless.
if [ -n "$RESOLVED" ]; then
    if ( set -euo pipefail; PV_BIN="$RESOLVED" . scripts/pv_bin.sh ) >"$TMP/good.log" 2>&1; then
        note "OK  accepts a correctly-versioned pv via the same override path"
    else
        bad "pv_bin.sh refused a pv reporting the crate version"
        sed 's/^/      /' "$TMP/good.log" >&2
    fi
fi

# ---------------------------------------------------------------------------
# 3. MUTATION: pv_bin_assert_unchanged must notice the artifact moving.
#
#    Operates on a COPY. Every checkout on this machine shares one cargo target
#    dir, so mutating the real artifact to test a guard would corrupt whatever
#    else is building - the precise hazard this function exists to detect.
if [ -n "$RESOLVED" ]; then
    cp "$RESOLVED" "$TMP/copy-pv"
    unchanged_rc=0
    ( set -euo pipefail
      PV_BIN="$TMP/copy-pv" . scripts/pv_bin.sh || exit 1
      pv_bin_assert_unchanged || exit 2
      printf 'tamper' >> "$PV"
      pv_bin_assert_unchanged && exit 3
      exit 0
    ) >"$TMP/unchanged.log" 2>&1 || unchanged_rc=$?
    case "$unchanged_rc" in
        0) note "OK  assert_unchanged is green before and RED after the binary moves" ;;
        2) bad "pv_bin_assert_unchanged was RED on an untouched binary" ;;
        3) bad "pv_bin_assert_unchanged stayed GREEN after the binary changed" ;;
        *) bad "assert_unchanged probe failed to run (rc=$unchanged_rc)"
           sed 's/^/      /' "$TMP/unchanged.log" >&2 ;;
    esac
fi

# ---------------------------------------------------------------------------
# 4. The library must be OPTION-NEUTRAL. A sourced `set -e` once killed the
#    nightly story six lines in (CLAUDE.md). check_sourced_libs_option_neutral.sh
#    checks the TEXT; this checks the BEHAVIOUR, which is what actually matters.
#    Each fingerprint is taken INSIDE a subshell, which is the honest scope: a
#    sourced file can only mutate the shell that sources it, and that is the
#    subshell here.
opts_before=$( set -o | LC_ALL=C sort | md5sum )
opts_after=$( . scripts/pv_bin.sh >/dev/null 2>&1 || true; set -o | LC_ALL=C sort | md5sum )
if [ "$opts_before" = "$opts_after" ]; then
    note "OK  sourcing pv_bin.sh leaves the caller's shell options untouched"
else
    bad "sourcing pv_bin.sh MUTATED the caller's shell options"
fi

# 4b. CONTROL for 4: prove the comparison can actually detect a change, or the
#     assertion above is satisfied by a broken measurement rather than by a
#     well-behaved library. This control has already earned its keep: the first
#     draft flipped `pipefail`, which this file's own `set -euo pipefail` has
#     ALREADY set, so the fingerprint was identical and check 4 was vacuous --
#     reported as OK. `noglob` is off here, so flipping it is a real change.
opts_mutated=$( set -o noglob; set -o | LC_ALL=C sort | md5sum )
if [ "$opts_before" = "$opts_mutated" ]; then
    bad "the shell-option fingerprint cannot detect 'set -o noglob' - check 4 is vacuous"
else
    note "OK  the option fingerprint detects a deliberate 'set -o noglob'"
fi

printf '\n'
if [ "$fails" -gt 0 ]; then
    printf '%s assertion(s) failed.\n' "$fails" >&2
    exit 1
fi
printf 'PASS: pv_bin.sh resolves the HEAD-built pv, refuses a stale one, and is option-neutral.\n'
