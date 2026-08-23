# verifier_pin.sh — THE rule that says which binary a release gate is allowed to
# believe, and the two pins that implement it.
#
# Source it, never execute it:
#     . "$SKILL_DIR/verifier_pin.sh" || exit 2
#     verifier_pin_pmat "$CRATE" "$BINPATH"   # sets PMAT_BIN
#     verifier_pin_pv                         # sets PV; 0 pinned, 1 broken, 2 unpinned
#
# ── THE RULE ────────────────────────────────────────────────────────────────
#
#   A gate that decides a release must not resolve its verifier through PATH.
#   Where the repo pins the tool, use the pin; where it does not, REPORT rather
#   than fall back — a skipped gate that says so beats a green one measured with
#   an unknown binary.
#
# It is stated HERE, once, and nowhere else. Every other file that needs it
# points at this one. #2640 exists because the same rule had been rediscovered
# FIVE times, each time as a local fix that did not know about the other four:
#
#   1  PMAT_BIN (user-scope dogfood runner)  a gate ran a pmat that predated
#      CB-200 becoming a ratchet, and Failed a tree the shipped code passes
#   2  scripts/pv_bin.sh                     PATH pv 0.49.0 vs in-tree 0.63.0
#      disagreed on the gate that DECIDES the release
#   3  scripts/apr_bin.sh                    four `apr` binaries coexisted; a
#      bare `apr` resolved to a 26-day-old copy
#   4  aprender#2384                         MCP spawned a bare `apr`, ran
#      0.60.0 while reporting 0.63.0
#   5  APR-BENCH-RFC-001                     unpinned llama.cpp comparator
#
# Five rediscoveries is the evidence that a rule merely STATED is documentation.
# It is therefore also ENFORCED, by scripts/check_verifier_pinning.sh, which
# fails on any bare pv/pmat/apr in command position in the runner — and which
# proves the two pins BEHAVE, not merely that they are present.
#
# OPTION-NEUTRAL BY CONSTRUCTION: this file sets no shell options. `set -euo
# pipefail` in a SOURCED file mutates the CALLER's shell — that leak once killed
# the nightly six lines in (CLAUDE.md; scripts/check_sourced_libs_option_neutral.sh).
# Failure is signalled by RETURN STATUS only.

# ── WHICH pmat runs the pmat gates ──────────────────────────────────────────
#
# `pmat verify` and `pmat comply check` are normally the INSTALLED pmat applied
# to whatever crate is under test — that is the point of a fleet quality tool.
# But when the crate under test IS pmat, that is the one case where it is wrong:
# the gate then measures a DIFFERENT BUILD than the one being released.
#
# Measured, 2026-08-22, releasing pmat 3.32.0:
#   installed ~/.cargo/bin/pmat   version 3.32.0   commit 8134bb373
#   built from the release tree   version 3.32.0   commit 7a7409e03
# Both print "3.32.0". Only the commit line differs, and nothing in the receipt
# showed it. The installed binary predated CB-200 becoming a ratchet — the
# string "recorded baseline" occurs 0 times in it and 4 times in the new one —
# so the release gate ran the OLD zero-tolerance check and returned Fail against
# a tree the shipped code passes. A gate validating a release with a stale copy
# of the thing being released is the same defect class this protocol exists to
# find, sitting inside the protocol.
#
# So: for pmat, the pmat gates use the artifact just built. For every other
# crate, PATH is correct and is what still happens.
verifier_pin_pmat() {
    verifier_pin_crate="${1:-}"
    verifier_pin_built="${2:-}"
    PMAT_BIN=pmat
    if [ "$verifier_pin_crate" = "pmat" ] \
       && [ -n "$verifier_pin_built" ] && [ -x "$verifier_pin_built" ]; then
        PMAT_BIN="$verifier_pin_built"
    fi
    return 0
}

# ── WHICH pv validates the contracts ────────────────────────────────────────
#
# pv is PINNED, never PATH-resolved. scripts/pv_bin.sh records why: a PATH `pv`
# was 0.49.0 while the in-tree crate was 0.63.0, and the two disagreed on the
# gate that decides the release -- strict-test-binding reported 253 refs / 51
# missing under the stale binary and 371 / 27 under HEAD. The dogfood runner IS
# a surface where the release decision is made, so it must not be the one
# holding the stale binary.
#
# This protocol runs against any repo in the fleet, and not all of them carry
# pv_bin.sh. Where it is absent, pv is left UNRESOLVED and the contract gates
# REPORT that rather than falling back to whatever PATH offers. A skipped gate
# that says so beats a green one measured with an unknown binary.
#
# The subshell is load-bearing and is the one deviation from the copy this was
# merged from. pv_bin.sh ends in `PV=$(...) || return 1 2>/dev/null || exit 1`.
# Sourced at the top level of a function body, `return` would unwind the CALLER;
# sourced inside a subshell, a failure exits only the subshell and this function
# still gets to report it. The pin is thereby usable from a function, which is
# what lets check_verifier_pinning.sh exercise it in isolation instead of taking
# its presence on trust.
#
# Returns: 0 = pinned and executable · 1 = pin present but FAILED to resolve
#          2 = this repo ships no pin (report, never fall back to PATH)
verifier_pin_pv() {
    PV=""
    [ -f scripts/pv_bin.sh ] || return 2
    PV=$( . ./scripts/pv_bin.sh >/dev/null 2>&1 && printf '%s' "$PV" )
    if [ -n "$PV" ] && [ -x "$PV" ]; then
        return 0
    fi
    PV=""
    return 1
}
