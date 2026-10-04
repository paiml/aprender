#!/usr/bin/env bash
# release_ladders_nightly_table.sh -- planted rows for scripts/release/release_ladders_nightly.sh (RQ-2).
# Copies the real script into a throwaway repo, swaps autopilot for a stub that exits and writes STATUS
# the way autopilot's die/say do, and checks the index row, the receipt selection and the exit. Then
# plants mutants (no version gate, newest-mtime receipt, every failure mapped to "not measured") and
# requires a row to catch each.
#
# EXIT 0 every row holds and every mutant is caught · 1 otherwise
#
#   bash scripts/release/release_ladders_nightly_table.sh
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SUT="$ROOT/scripts/release/release_ladders_nightly.sh"
TD="$(mktemp -d)"
trap 'rm -rf "${TD:?}"' EXIT

# The stub autopilot: STUB_<step>="<exit>|<STOP text or empty>"; dogfood writes a receipt per STUB_RECEIPTS
# ("<name>:<commit>" ...), each with that commit, into $RELEASE_AP/wt/.dogfood.
cat > "$TD/autopilot.sh" <<'STUB'
#!/usr/bin/env bash
[ "$1" = --ladders ] || exit 64
v="$2" s="$3" step="$4"; var="STUB_$step"; spec="${!var:-0|}"
echo "$step" >> "$RELEASE_AP/calls"
if [ "$step" = dogfood ]; then
    mkdir -p "$RELEASE_AP/wt/.dogfood"
    for p in ${STUB_RECEIPTS:-}; do
        c="${p#*:}"; [ "$c" = HEAD ] && c="$s"
        printf '{"commit":"%s","version":"%s","phase":"pre-publish","tag":"%s"}\n' "$c" "$v" "${p%%:*}" > "$RELEASE_AP/wt/.dogfood/receipt-${p%%:*}.json"
    done
fi
[ -n "${spec#*|}" ] && echo "2026-01-01T00:00:00Z STOP ${spec#*|}" >> "$RELEASE_AP/STATUS"
exit "${spec%%|*}"
STUB

fail=0
row() { if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1: got '$2', want '$3'"; fail=1; fi; }

n=0
# repo <version> [tag] -> a fresh repo carrying the script under test at scripts/release/; sets G, R, S
repo() {
    n=$((n + 1)); G="$TD/g$n"; R="$TD/root$n"
    mkdir -p "$G/scripts/release"
    cp "$SUTX" "$G/scripts/release/release_ladders_nightly.sh"
    printf '[workspace]\nmembers = []\n\n[workspace.package]\nversion = "%s"\nedition = "2021"\n' "$1" > "$G/Cargo.toml"
    git -C "$G" init -q
    git -C "$G" add -A
    git -C "$G" -c core.hooksPath=/dev/null -c user.name=t -c user.email=t@t commit -qm c
    [ -z "${2:-}" ] || git -C "$G" tag "$2"
    S=$(git -C "$G" rev-parse HEAD)
}
# go -> "<exit>|<index row minus the epoch, or none>"
go() {
    local rc=0
    RELEASE_LADDERS_ROOT="$R" RELEASE_LADDERS_AUTOPILOT="$TD/autopilot.sh" RELEASE_LADDERS_REF=HEAD \
        bash "$G/scripts/release/release_ladders_nightly.sh" > "$TD/out" 2>&1 || rc=$?
    echo "$rc|$( { [ -f "$R/index.tsv" ] && tail -n 1 "$R/index.tsv" | cut -f1-5 | tr '\t' ' ' | sed "s/$S/S/"; } || echo none)"
}

table() { # table <sut> <label>
    SUTX="$1"; local l="$2"
    repo 9.9.0 v9.9.0
    row "$l version already tagged: no row, exit 0" "$(go)" "0|none"
    repo 9.9.0 v9.8.0
    row "$l bumped + untagged, all GO -> row 0 0 0, exit 0" "$(STUB_RECEIPTS=a:HEAD go)" "0|S 9.9.0 0 0 0"
    row "$l   ... all three steps ran" "$(tr '\n' ' ' < "$R/$S/calls")" "deep dogfood models "
    row "$l   ... the receipt was copied" "$(jq -r .commit "$R/$S/dogfood/receipt.json")" "$S"
    go > /dev/null
    row "$l same sha a second night: no second row" "$(wc -l < "$R/index.tsv" | tr -d ' ')" 1
    repo 9.9.0
    row "$l models RED (NO-GO rc=1) -> 1, exit 1" "$(STUB_models='1|T-1 model matrix NO-GO rc=1: x' go)" "1|S 9.9.0 0 0 1"
    repo 9.9.0
    row "$l models lock busy (NO-GO rc=75) -> 2" "$(STUB_models='1|T-1 model matrix NO-GO rc=75: x' go)" "1|S 9.9.0 0 0 2"
    repo 9.9.0
    row "$l dogfood declines (NO-GO rc=2) -> 2" "$(STUB_dogfood='1|dogfood pre-publish NO-GO rc=2 (x)' go)" "1|S 9.9.0 0 2 0"
    repo 9.9.0
    row "$l deep RED with prose and no rc -> 1 (fail closed)" "$(STUB_deep='1|T-1 doctests RED (x)' go)" "1|S 9.9.0 1 0 0"
    repo 9.9.0
    row "$l autopilot usage/ENV exit 2 -> 2" "$(STUB_deep='2|' go)" "1|S 9.9.0 2 0 0"
    repo 9.9.0
    STUB_RECEIPTS="20260101:0000000000000000000000000000000000000000 20250101:HEAD 20270101:1111111111111111111111111111111111111111" go > /dev/null
    row "$l receipt picked by commit, not newest name" "$(jq -r .tag "$R/$S/dogfood/receipt.json" 2> /dev/null || echo none)" "20250101"
    repo 9.9.0
    STUB_RECEIPTS="20270101:1111111111111111111111111111111111111111" go > /dev/null
    row "$l only a foreign-commit receipt: none copied" "$([ -e "$R/$S/dogfood/receipt.json" ] && echo copied || echo none)" "none"
}

table "$SUT" real

# autopilot --ladders refuses a malformed call before any params, gh or git (exit 2)
AP_SUT="$ROOT/scripts/release/autopilot.sh"
apx() { local rc=0; "$@" > /dev/null 2>&1 || rc=$?; echo "$rc"; }
row "autopilot --ladders without RELEASE_AP -> 2" "$(apx env -u RELEASE_AP bash "$AP_SUT" --ladders 9.9.0 "$(printf 'a%.0s' {1..40})" deep)" 2
row "autopilot --ladders step 'tag' -> 2" "$(apx env RELEASE_AP="$TD/x" bash "$AP_SUT" --ladders 9.9.0 "$(printf 'a%.0s' {1..40})" tag)" 2
row "autopilot --ladders short sha -> 2" "$(apx env RELEASE_AP="$TD/x" bash "$AP_SUT" --ladders 9.9.0 abc123 deep)" 2
row "autopilot --ladders wrong arg count -> 2" "$(apx env RELEASE_AP="$TD/x" bash "$AP_SUT" --ladders 9.9.0 deep)" 2
row "autopilot --ladders created no state dir" "$([ -e "$TD/x" ] && echo created || echo none)" none

mutant() { # mutant <label> <sed expr>
    local m="$TD/mutant.sh" c
    sed "$2" "$SUT" > "$m"
    if cmp -s "$SUT" "$m"; then echo "FAIL  mutant $1: the sed did not change the script (pattern drifted)"; fail=1; return; fi
    c=$(table "$m" "  [$1]" 2>&1 | grep -c '^FAIL' || true)
    if [ "$c" -gt 0 ]; then echo "ok    mutant $1 caught by $c row(s)"; else echo "FAIL  mutant $1 survived every row"; fail=1; fi
}
mutant no-version-gate      's/rev-parse -q --verify "refs\/tags\/v\$V"/rev-parse -q --verify refs\/tags\/never-a-tag/'
mutant receipt-by-mtime     's/if \[ "\$(jq -r .\.commit \/\/ empty. -- "\$r" 2> \/dev\/null)" = "\$S" \]; then/if true; then/'
mutant red-is-not-measured  's/^        \*) echo 1 ;;$/        *) echo 2 ;;/'
mutant no-rerun-guard       's/cut -f1 "\$ROOT\/index.tsv" | grep -qx "\$S"/false/'

exit "$fail"
