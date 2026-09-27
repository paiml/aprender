# comparand_tree.sh - extract the comparand tree `pv lint`'s ratchets are measured over (#3569 part 2).
#
# Source it (option-neutral: no `set` here; failure is the return status):
#     REPO_ROOT=<toplevel>; PROG=<caller>; . scripts/lib/comparand_tree.sh || exit 1
#     comparand_tree "$scratch" && PV_LINT_COMPARAND="$COMPARAND_CONTRACTS" "$PV" lint contracts/
#
# A shrink-only debt count is judged head vs base, same scanner, same run (operator, 2026-09-27): the
# base is a TREE, never a number stored in contracts/lint-baseline.json that the PR under test can
# restamp. The base is whatever scripts/lib/resolve_base.sh names (merge-base with origin/main; the first
# parent on a push to main; a refusal otherwise), extracted with `git archive` from the object store, so
# the working tree's edits cannot leak into it.
#
# Extracted: every path a comparand-measuring gate reads that exists at the base -- contracts/ (all of
# them), book/, lean/ and the staging crate's lean/, book/ and discharge-summary.json (theorem-pairing,
# proved-is-derived). contracts/ missing at the base is a refusal: an empty comparand would measure 0
# and turn every ratchet into "no rise".
#
# Sets COMPARAND_CONTRACTS (the extracted contracts dir), BASE_REF and BASE_HOW. Fails CLOSED: any
# failure returns 1 with COMPARAND_CONTRACTS empty, and the caller refuses to run the gate.

# shellcheck source=scripts/lib/resolve_base.sh
. "$(dirname "${BASH_SOURCE[0]}")/resolve_base.sh" || return 1

COMPARAND_PATHS="contracts book lean crates/aprender-contracts-staging/lean crates/aprender-contracts-staging/book crates/aprender-contracts-staging/discharge-summary.json"

comparand_tree() { # comparand_tree SCRATCH_DIR
    local tree="${1:-}" ref p
    local -a have=()
    local PROG="${PROG:-comparand_tree}"
    COMPARAND_CONTRACTS=""
    BASE_REF="" BASE_HOW=""
    [ -n "$tree" ] && [ -d "$tree" ] || { printf '%s: no scratch dir to extract the comparand into\n' "$PROG" >&2; return 1; }
    if ! resolve_base HEAD || [ -z "$BASE_REF" ]; then
        printf '%s: no comparand could be named for HEAD, so the ratchets are UNMEASURED at the base; refused (not "unchanged").\n' "$PROG" >&2
        printf '    In CI: git fetch --no-tags --depth=2 origin +refs/heads/main:refs/remotes/origin/main\n' >&2
        return 1
    fi
    ref="$BASE_REF"
    for p in $COMPARAND_PATHS; do
        git -C "$REPO_ROOT" cat-file -e "$ref:$p" 2>/dev/null && have+=("$p")
    done
    case " ${have[*]} " in
        *" contracts "*) ;;
        *) printf '%s: %s carries no contracts/; refused (an empty comparand measures 0)\n' "$PROG" "$ref" >&2; return 1 ;;
    esac
    if ! git -C "$REPO_ROOT" archive --format=tar "$ref" -- "${have[@]}" | tar -xf - -C "$tree"; then
        printf '%s: could not extract the comparand at %s\n' "$PROG" "$ref" >&2
        return 1
    fi
    [ -d "$tree/contracts" ] || { printf '%s: extraction at %s left no contracts/\n' "$PROG" "$ref" >&2; return 1; }
    COMPARAND_CONTRACTS="$tree/contracts"
    printf '%s: comparand %s (%s)\n' "$PROG" "$(git -C "$REPO_ROOT" rev-parse --short "$ref" 2>/dev/null || printf '%s' "$ref")" "$BASE_HOW" >&2
    return 0
}
