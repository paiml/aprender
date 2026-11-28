#!/usr/bin/env bash
# mutate_spec_conformance.sh - the mutation set for the PP-9 ledger rules (L0/L1/L2/L3)
# in crates/aprender-ci-tools/src/spec_conformance_ledger.rs, run against
# scripts/spec_conformance.sh --selftest with the aprender-ci-tools binary rebuilt per mutant.
#
# PP-LLAMA-001 v3.1 §6 PP-9, PMAT-930/PMAT-931. A condition the scanner states that no
# fixture can remove is a condition nobody knows is load-bearing: the first L2 carried
# three shape conditions and one mutant, and the review quorum refuted the rule four ways.
# Every mutant below is a text replacement that MUST occur exactly once (a mutant that
# changed nothing reports a kill it never earned), the ledger module is restored after each,
# and a mutant is KILLED iff the case table prints at least one BROKE row. A mutant that
# does not compile is not a kill: it is named as a survivor.
#
# The mutant set was written against the Python parser (scripts/lib/spec_conformance.py,
# PMAT-7534 slice 2d removed it). Each id keeps its original rule, with one equivalent Rust
# mutant per id.
#
# USAGE
#   scripts/mutate_spec_conformance.sh          run the set, print the table, exit 1 on a survivor
#   scripts/mutate_spec_conformance.sh --list   print the catalogue and exit
#
# ENVIRONMENT
#   MUTATE_TARGET_DIR  the cargo target dir the aprender-ci-tools binary is built into
#                      (default: ${TMPDIR:-/tmp}/mutate-spec-conformance-target)
set -euo pipefail
PROG=$(basename "$0")
REPO_ROOT=$(git rev-parse --show-toplevel)
TARGET="$REPO_ROOT/crates/aprender-ci-tools/src/spec_conformance_ledger.rs"
SELFTEST="$REPO_ROOT/scripts/spec_conformance.sh"
MUTATE_TARGET_DIR="${MUTATE_TARGET_DIR:-${TMPDIR:-/tmp}/mutate-spec-conformance-target}"
BIN="$MUTATE_TARGET_DIR/debug/aprender-ci-tools"
BUILD_LOG="$MUTATE_TARGET_DIR/build.log"
BACKUP=$(mktemp)
# SEC010: validate before cp/rm -- TARGET, BACKUP, the target dir and the binary are
# REPO_ROOT plus a literal, mktemp's answer, or the operator's override; refuse '..' in any.
case "$TARGET$BACKUP$MUTATE_TARGET_DIR$BIN" in
    *..*) echo "ERROR: TARGET/BACKUP/MUTATE_TARGET_DIR must not contain '..'" >&2; exit 1 ;;
esac
trap 'cp "$BACKUP" "$TARGET"; rm -f "$BACKUP"' EXIT
cp "$TARGET" "$BACKUP"

# id | old text | new text  (tab-separated; each old text occurs EXACTLY ONCE)
CATALOGUE=$(cat <<'CAT'
l2-emit-removed	out.push(split_record(table_end, &outside));	let _ = (table_end, &outside); // MUTANT
leading-pipe-required	if line.matches('|').count() < 2 {	if !line.trim_start().starts_with('|') {
l2-relabelled-l1	"L2",	"L1",
l3-removed	out.extend(check_shapes(rows, header.len()));	let _ = (rows, header.len()); // MUTANT
outside-rows-not-spent	.chain(outside.into_iter().map(|o| o.cells))	.chain(std::iter::empty())
tier-claim-check-removed	|| !claims_spend(&cells) {	|| false {
id-required-reintroduced	|| !claims_spend(&cells) {	|| !claims_spend(&cells) || row_id(&cells).is_empty() {
tier-backtick-strip-removed	cells.iter().any(|c| spend_tier(&norm(c)))	cells.iter().any(|c| spend_tier(c.trim()))
conformant-tier-dropped	["RECORDED", "CONFORMANT"]	["RECORDED", "RECORDED"]
l1-respend-check-removed	if !seen.insert(key.clone()) {	if false {
l1-tier-filter-narrowed	if !spend_tier(&cell(cells, cols.conformance)) {	if !cell(cells, cols.conformance).to_uppercase().starts_with("RECORDED") {
wrap-emphasis-strip-removed	const WRAP: [char; 5] = ['`', '*', '_', ' ', '\t'];	const WRAP: [char; 3] = ['`', ' ', '\t'];
key-normaliser-bypassed	.map(|c| norm(c))	.map(|c| c.trim().trim_matches('`').to_owned())
key-casefold-removed	casefold(&cell(cells, *idx))	cell(cells, *idx)
l0-missing-columns-ignored	.filter(|(_, idx)| idx.is_none())	.filter(|_| false)
CAT
)

if [ "${1:-}" = "--list" ]; then
    printf '%s\n' "$CATALOGUE" | cut -f1
    exit 0
fi

# build_ci_tools: the binary from this tree, in the form the host's cargo wrapper admits
# (`nice -n 10 cargo build -j 8`, with the release profile overrides cleared). The log is kept.
build_ci_tools() {
    (cd "$REPO_ROOT" && env -u CARGO_PROFILE_RELEASE_CODEGEN_UNITS -u CARGO_PROFILE_RELEASE_LTO \
        CARGO_TARGET_DIR="$MUTATE_TARGET_DIR" nice -n 10 cargo build -j 8 -p aprender-ci-tools) \
        >"$BUILD_LOG" 2>&1
}

# count_occurrences <file> <literal>: how many times the literal occurs in the file.
count_occurrences() {
    local s rest n=0
    s=$(cat -- "$1"; printf x)
    s=${s%x}
    rest=$s
    while [[ $rest == *"$2"* ]]; do
        n=$((n + 1))
        rest=${rest#*"$2"}
    done
    printf '%s\n' "$n"
}

# replace_once <file> <old> <new>: rewrite the file with the first occurrence of old replaced.
# The replacement is quoted, so a '&' in it is literal (bash 5.2 patsub_replacement).
replace_once() {
    local s
    s=$(cat -- "$1"; printf x)
    s=${s%x}
    printf '%s' "${s/"$2"/"$3"}" >"$1"
}

# THE BASELINE FIRST. A mutant is killed by a case that was GREEN without it; a
# case table that is already red kills every mutant for free (PMAT-932).
build_ci_tools || { printf 'FAIL  baseline build of aprender-ci-tools failed (%s)\n' "$BUILD_LOG"; exit 1; }
if base=$(CI_TOOLS_BIN="$BIN" bash "$SELFTEST" --selftest 2>&1); then base_rc=0; else base_rc=$?; fi
if [ "$base_rc" -ne 0 ] || [ "$(printf '%s\n' "$base" | grep -c 'BROKE')" -ne 0 ]; then
    printf 'FAIL  the unmutated case table is not green (rc=%s); nothing below could be a kill\n' "$base_rc"
    exit 1
fi
printf '  baseline green: %s\n' "$(printf '%s\n' "$base" | tail -1 | sed 's/^ *//')"

attempted=0; killed=0; unviable=0; survivors=""
while IFS=$'\t' read -r id old new; do
    [ -n "$id" ] || continue
    attempted=$((attempted + 1))
    cp "$BACKUP" "$TARGET"
    n=$(count_occurrences "$TARGET" "$old")
    if [ "$n" != 1 ]; then
        printf '  BROKE %-28s the mutation site occurs %s time(s), not once -- the set is stale\n' "$id" "$n"
        survivors="$survivors $id(stale)"
        continue
    fi
    replace_once "$TARGET" "$old" "$new"
    if cmp -s "$BACKUP" "$TARGET"; then
        printf '  BROKE %-28s the mutant changed nothing\n' "$id"
        survivors="$survivors $id(noop)"
        cp "$BACKUP" "$TARGET"
        continue
    fi
    rm -f "$BIN"
    if ! build_ci_tools; then
        printf '  SURVIVED %-25s the mutant does not compile (not a kill; see %s)\n' "$id" "$BUILD_LOG"
        survivors="$survivors $id(nobuild)"
        cp "$BACKUP" "$TARGET"
        continue
    fi
    if out=$(CI_TOOLS_BIN="$BIN" bash "$SELFTEST" --selftest 2>&1); then :; else :; fi
    cp "$BACKUP" "$TARGET"
    # A mutant that CRASHES the scanner turns every case BROKE at once; that is
    # not a fixture discriminating a condition, so it is UNVIABLE, never a kill.
    crashed=$(printf '%s\n' "$out" | grep -c 'scanner errored' || true)
    broke=$(printf '%s\n' "$out" | grep -c 'BROKE' || true)
    if [ "$crashed" -ge 1 ]; then
        printf '  UNVIABLE %-25s the scanner errored under this mutant (%s case(s))\n' "$id" "$crashed"
        unviable=$((unviable + 1))
        survivors="$survivors $id(unviable)"
    elif [ "$broke" -ge 1 ]; then
        printf '  KILLED %-28s %s case(s) BROKE\n' "$id" "$broke"
        killed=$((killed + 1))
    else
        printf '  SURVIVED %-25s the case table stayed green\n' "$id"
        survivors="$survivors $id"
    fi
done <<< "$CATALOGUE"
cp "$BACKUP" "$TARGET"
# Leave the binary built from the restored tree, not from the last mutant.
build_ci_tools || { printf 'FAIL  rebuild of the restored tree failed (%s)\n' "$BUILD_LOG"; exit 1; }

printf '%s: attempted=%s killed=%s unviable=%s survivors=%s\n' "$PROG" "$attempted" "$killed" "$unviable" "${survivors:- none}"
if [ "$killed" -ne "$attempted" ]; then
    printf 'FAIL  %s mutant(s) survived: a condition the scanner states and no fixture tests\n' "$((attempted - killed))"
    exit 1
fi
printf 'PASS  every mutant of the PP-9 ledger rules is killed by a named fixture\n'
