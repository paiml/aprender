#!/usr/bin/env bash
# surface_audit_bins_gate.sh -- the surface ledger covers EVERY shipped binary, N/N.
#
# #4476 ONT-4g. docs/audits/surface_audit.csv is the ledger the dogfood coverage
# ratio is computed over. Two ways it can lie, and both read as green today:
#
#   EMPTY   the review that opened #4476 was handed a 0-row copy of the file. A
#           ratio over an empty ledger is 0/0, and a producer that writes 0 rows
#           and exits 0 hands the next step exactly that.
#   PARTIAL a binary that ships with no ledger row is outside every coverage
#           number. Nobody sees it go missing, because the ratio only counts
#           the rows that are there.
#
# So, at the RELEASE sha:
#   rows    > 0, or RED
#   binaries in the ledger == [[bin]] targets in `cargo metadata`, printed as N/N,
#           and ANY mismatch in EITHER direction is RED by name
#
# The [[bin]] universe comes from cargo metadata, never from a list or a glob:
# auto-discovered binaries have no [[bin]] stanza, and `autobins = false` removes
# ones that do (the same rule scripts/dogfood_surfaces.sh follows).
#
# WHERE IT RUNS: the release pre-publish dogfood, via [package.metadata.dogfood]
# gates in the root Cargo.toml. That is where the decision is made and where
# cargo is present. The PR-time half is scripts/check_surface_audit_bins.sh, the
# cargo-free case table that guard_tree.sh dispatches. It drives this script
# through --metadata fixtures and proves an empty ledger turns it RED.
#
#   bash scripts/surface_audit_bins_gate.sh                     # the release check
#   bash scripts/surface_audit_bins_gate.sh --csv F --metadata M.json
#   producer | bash scripts/surface_audit_bins_gate.sh --require-rows
#                  # pass stdin through; exit 1 if it held 0 feature rows
#
# Exit: 0 GREEN, 1 RED, 2 usage or a missing tool.

set -uo pipefail

CSV="docs/audits/surface_audit.csv"
META=""
MODE="gate"

while [ $# -gt 0 ]; do
    case "$1" in
        --csv)          CSV="${2:?--csv needs a path}"; shift 2 ;;
        --metadata)     META="${2:?--metadata needs a path}"; shift 2 ;;
        --require-rows) MODE="require-rows"; shift ;;
        -h|--help)      sed -n '2,33p' "$0"; exit 0 ;;
        *) printf 'surface_audit_bins_gate: unknown argument %s\n' "$1" >&2; exit 2 ;;
    esac
done

# --- producer side: 0 emitted rows is a failure, never an empty success -------
#
# A feature row is "<binary>\t<feature>". FEATURESET and UNPROBED are the
# emitter's own bookkeeping lines. They are not surface, so a stream made only of
# them is still an empty stream.
if [ "$MODE" = "require-rows" ]; then
    buf=$(cat)
    [ -z "$buf" ] || printf '%s\n' "$buf"
    n=$(printf '%s\n' "$buf" | awk -F'\t' 'NF>=2 && $1!="FEATURESET" && $1!="UNPROBED" && $1!="" {c++} END{print c+0}')
    if [ "$n" -eq 0 ]; then
        printf 'RED  surface producer emitted 0 feature rows. An empty ledger is 0/0, never a pass.\n' >&2
        exit 1
    fi
    printf 'surface producer: %s feature rows\n' "$n" >&2
    exit 0
fi

command -v jq >/dev/null 2>&1 || { printf 'surface_audit_bins_gate: jq is required\n' >&2; exit 2; }

if [ ! -f "$CSV" ]; then
    printf 'RED  %s does not exist. The ledger is absent, not empty.\n' "$CSV"
    exit 1
fi

header=$(head -n 1 "$CSV")
if [ "${header%%,*}" != "binary" ]; then
    printf 'RED  %s: column 1 is "%s", expected "binary". The ledger shape changed; fix this gate with it, never around it.\n' \
        "$CSV" "${header%%,*}"
    exit 1
fi

rows=$(awk 'NR>1 && NF {c++} END{print c+0}' "$CSV")
if [ "$rows" -eq 0 ]; then
    printf 'RED  %s has 0 rows. Coverage over an empty ledger is 0/0, never a pass.\n' "$CSV"
    exit 1
fi

tmp=$(mktemp -d) || exit 2
trap 'rm -rf "${tmp:?}"' EXIT

if [ -z "$META" ]; then
    META="$tmp/metadata.json"
    if ! cargo metadata --no-deps --format-version 1 > "$META" 2> "$tmp/metadata.err"; then
        printf 'RED  cargo metadata failed, so the [[bin]] universe is unknown:\n'
        tail -n 5 "$tmp/metadata.err"
        exit 1
    fi
fi

if ! jq -r '.packages[].targets[] | select(.kind | index("bin")) | .name' "$META" \
        2> "$tmp/jq.err" | LC_ALL=C sort -u > "$tmp/declared"; then
    printf 'RED  could not read [[bin]] targets from %s\n' "$META"
    exit 1
fi
awk -F, 'NR>1 && NF {print $1}' "$CSV" | LC_ALL=C sort -u > "$tmp/ledger"

declared=$(grep -c . "$tmp/declared")
if [ "$declared" -eq 0 ]; then
    printf 'RED  cargo metadata declares 0 binary targets. The enumeration is broken, not the workspace.\n'
    exit 1
fi

covered=$(LC_ALL=C comm -12 "$tmp/declared" "$tmp/ledger" | grep -c .)
missing=$(LC_ALL=C comm -23 "$tmp/declared" "$tmp/ledger")
stray=$(LC_ALL=C comm -13 "$tmp/declared" "$tmp/ledger")

printf 'surface ledger: %s rows, binaries %s/%s\n' "$rows" "$covered" "$declared"
rc=0
if [ -n "$missing" ]; then
    rc=1
    while IFS= read -r b; do
        printf 'RED  [[bin]] %s ships with no row in %s\n' "$b" "$CSV"
    done <<< "$missing"
fi
if [ -n "$stray" ]; then
    rc=1
    while IFS= read -r b; do
        printf 'RED  ledger binary %s is not a [[bin]] target in cargo metadata (renamed or removed)\n' "$b"
    done <<< "$stray"
fi
[ "$rc" -eq 0 ] && printf 'GREEN %s/%s\n' "$covered" "$declared"
exit "$rc"
