#!/usr/bin/env bash
# mutate_spec_conformance.sh - the mutation set for the PP-9 ledger rules (L1/L2/L3)
# in scripts/lib/spec_conformance.py, run against scripts/spec_conformance.sh --selftest.
#
# PP-LLAMA-001 v3.1 §6 PP-9, PMAT-930/PMAT-931. A condition the scanner states that no
# fixture can remove is a condition nobody knows is load-bearing: the first L2 carried
# three shape conditions and one mutant, and the review quorum refuted the rule four ways.
# Every mutant below is a text replacement that MUST occur exactly once (a mutant that
# changed nothing reports a kill it never earned), the scanner is restored after each,
# and a mutant is KILLED iff the case table prints at least one BROKE row.
#
# USAGE
#   scripts/mutate_spec_conformance.sh          run the set, print the table, exit 1 on a survivor
#   scripts/mutate_spec_conformance.sh --list   print the catalogue and exit
set -euo pipefail
PROG=$(basename "$0")
REPO_ROOT=$(git rev-parse --show-toplevel)
TARGET="$REPO_ROOT/scripts/lib/spec_conformance.py"
SELFTEST="$REPO_ROOT/scripts/spec_conformance.sh"
BACKUP=$(mktemp)
trap 'cp "$BACKUP" "$TARGET"; rm -f "$BACKUP"' EXIT
cp "$TARGET" "$BACKUP"

# id | old text | new text  (tab-separated; each old text occurs EXACTLY ONCE)
CATALOGUE=$(cat <<'CAT'
l2-emit-removed	        _emit_ledger_split(table_end, outside)	        pass  # MUTANT
backtick-strip-removed	    rid = cells[0].strip().strip("`").strip()	    rid = cells[0].strip()
leading-pipe-required	    if line.count("|") < 2:	    if not line.lstrip().startswith("|"):
superseded-cutoff-removed	(i for i, line in enumerate(lines) if SUPERSEDED_HEAD.match(line.strip())),	(i for i, line in enumerate(lines) if False),
l2-relabelled-l1	    emit("VIOLATION", "L2", " ".join(rid for _, rid, _ in outside),	    emit("VIOLATION", "L1", " ".join(rid for _, rid, _ in outside),
l3-removed	    _check_ledger_shapes(rows, header)	    pass  # MUTANT
outside-rows-not-spent	    return rows + [c for _, _, c in outside]	    return rows
header-skip-removed	    if first is None or not _row_id(first):	    if first is None:
CAT
)

if [ "${1:-}" = "--list" ]; then
    printf '%s\n' "$CATALOGUE" | cut -f1
    exit 0
fi

attempted=0; killed=0; survivors=""
while IFS=$'\t' read -r id old new; do
    [ -n "$id" ] || continue
    attempted=$((attempted + 1))
    cp "$BACKUP" "$TARGET"
    n=$(python3 - "$TARGET" "$old" "$new" <<'PY'
import sys
p, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p, encoding="utf-8").read()
n = s.count(old)
if n == 1:
    open(p, "w", encoding="utf-8").write(s.replace(old, new))
print(n)
PY
)
    if [ "$n" != 1 ]; then
        printf '  BROKE %-28s the mutation site occurs %s time(s), not once -- the set is stale\n' "$id" "$n"
        survivors="$survivors $id(stale)"
        continue
    fi
    if cmp -s "$BACKUP" "$TARGET"; then
        printf '  BROKE %-28s the mutant changed nothing\n' "$id"
        survivors="$survivors $id(noop)"
        continue
    fi
    out=$(bash "$SELFTEST" --selftest 2>&1 || true)
    broke=$(printf '%s\n' "$out" | grep -c 'BROKE' || true)
    if [ "$broke" -ge 1 ]; then
        printf '  killed %-28s %s case(s) BROKE\n' "$id" "$broke"
        killed=$((killed + 1))
    else
        printf '  SURVIVED %-25s the case table stayed green\n' "$id"
        survivors="$survivors $id"
    fi
done <<< "$CATALOGUE"
cp "$BACKUP" "$TARGET"

printf '%s: attempted=%s killed=%s survivors=%s\n' "$PROG" "$attempted" "$killed" "${survivors:- none}"
if [ "$killed" -ne "$attempted" ]; then
    printf 'FAIL  %s mutant(s) survived: a condition the scanner states and no fixture tests\n' "$((attempted - killed))"
    exit 1
fi
printf 'PASS  every mutant of the PP-9 ledger rules is killed by a named fixture\n'
