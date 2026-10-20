. scripts/pv_bin.sh || exit 1
PV_SAT=$(dirname "$PV")/pv-sat   # never an inherited PV_SAT: it may be another tree's or a stale build

# A stale pv regenerates shapes.ttl in an old form and the gate then goes red (#5776 lost ~25 min). pv_bin.sh
# proves the tree; this proves the BUILD: pv and pv-sat must be newer than every tracked source of the verifier
# crates in THIS checkout, so a binary built before the merge just done is refused (same idea as apr_bin.sh).
stale_pv=$(for b in "$PV" "$PV_SAT"; do
    [ -x "$b" ] || { echo "$b (missing)"; continue; }
    git ls-files -z -- crates/aprender-contracts crates/aprender-contracts-cli Cargo.lock | xargs -0 -r find -L -newer "$b" -type f 2>/dev/null | head -1 | sed "s|^|$b older than |"
done) || true   # set -e: the loops end on a false test
if [ -n "$stale_pv" ]; then
    printf 'contracts_resolve: REFUSED, pv is not built from this HEAD:\n%s\nrebuild: cargo build -p aprender-contracts-cli --bin pv --bin pv-sat\n' "$stale_pv" >&2
    exit 1
fi

bash scripts/contracts_gate.sh regen
"$PV_SAT" contracts#!/usr/bin/env bash
# contracts_resolve.sh -- finish a `git merge origin/main` that stopped on the GENERATED contract files.
#
# contracts/contracts.nt, contracts/shapes.ttl and contracts/witness/*.json are derived from contracts/*.yaml
# (scripts/contracts_gate.sh GENERATED=, pv-sat); README.md carries a derived count (scripts/readme_sync.sh). Two PRs that each add a contract conflict on them, and
# neither side's copy is right: the merged tree needs a fresh one. This takes no side. It drops the
# generated paths from the conflict, regenerates them from the merged sources with the gate's own regen
# step and pv-sat, and stages the result. It never touches a source file, and it refuses to run if any
# conflict is outside the generated set, so a real conflict is never resolved by regeneration.
#
#   git merge origin/main        # stops on contracts.nt / witness
#   bash scripts/contracts_resolve.sh
#   bash scripts/contracts_gate.sh   # the gate stays the judge
#
# Pmat-Ticket: #5762
set -euo pipefail

GENERATED_RE='^contracts/(contracts\.nt|shapes\.ttl|witness/.*\.json)$'

conflicted=$(git diff --name-only --diff-filter=U)
[ -n "$conflicted" ] || echo "contracts_resolve: no conflict; regenerating the generated files from the merged sources"
other=$(printf '%s\n' "$conflicted" | grep -vE "$GENERATED_RE" || true)
if [ -n "$other" ]; then
    printf 'contracts_resolve: REFUSED, conflicts outside the generated set:\n%s\n' "$other" >&2
    exit 1
fi

. scripts/pv_bin.sh || exit 1
PV_SAT=$(dirname "$PV")/pv-sat   # never an inherited PV_SAT: it may be another tree's, or a stale build

# A stale pv regenerates shapes.ttl in an old form and the gate then goes red (#5776 lost ~25 min). pv_bin.sh
# proves the TREE; this proves the BUILD: pv and pv-sat must be newer than every tracked source of the verifier
# crates in this checkout, so a binary built before the merge just done is refused (same idea as apr_bin.sh).
# Checked BEFORE the conflicted files are dropped, so a refusal leaves the merge exactly as it was.
stale_pv=$(for b in "$PV" "$PV_SAT"; do
    [ -x "$b" ] || { echo "$b (missing)"; continue; }
    git ls-files -z -- crates/aprender-contracts crates/aprender-contracts-cli Cargo.lock \
        | while IFS= read -r -d "" s; do if [ "$s" -nt "$b" ]; then echo "$b is older than $s"; break; fi; done
done) || true   # set -e: the loops end on a false test
if [ -n "$stale_pv" ]; then
    printf 'contracts_resolve: REFUSED, pv is not built from this HEAD:\n%s\nrebuild: cargo build -p aprender-contracts-cli --bin pv --bin pv-sat\n' "$stale_pv" >&2
    exit 1
fi

# Drop both sides of every generated path; the regeneration below recreates what the merged sources produce.
printf '%s\n' "$conflicted" | while IFS= read -r f; do
    [ -n "$f" ] || continue
    git rm -q --cached --ignore-unmatch -- "$f"
    rm -f -- "${f:?}"
done

bash scripts/contracts_gate.sh regen
"$PV_SAT" contracts
bash scripts/readme_sync.sh --write   # README.md carries a derived contract count: also stale after a merge
git add -A -- contracts/contracts.nt contracts/shapes.ttl contracts/witness README.md
left=$(git diff --name-only --diff-filter=U)
[ -z "$left" ] || { printf 'contracts_resolve: still conflicted:\n%s\n' "$left" >&2; exit 1; }
echo "contracts_resolve: regenerated and staged; run scripts/contracts_gate.sh, then commit the merge"
