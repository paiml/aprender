#!/usr/bin/env bash
# check_surface_audit_bins.sh -- the case table for scripts/surface_audit_bins_gate.sh.
#
# #4476 ONT-4g. The gate itself runs at release, where the [[bin]] universe can be
# read from the workspace metadata of the release sha (it is declared in the root
# Cargo.toml [package.metadata.dogfood] gates). This file is its PR-time half: it
# drives the gate through fixture metadata JSON, with no build tool, so guard_tree.sh
# dispatches it on every PR, including a docs-only one.
#
# The row that matters most is the mutation the ticket names: an EMPTY ledger, and
# an EMPTY producer stream, must each turn the gate RED. A gate that goes green on
# 0/0 is the defect #4476 was opened for, since the review was handed a 0-row file.
#
#   bash scripts/check_surface_audit_bins.sh
#
# Exit: 0 when every row holds, 1 when any row fails (all rows always run).

set -uo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GATE="$ROOT/scripts/surface_audit_bins_gate.sh"

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    sed -n '2,16p' "$0"
    exit 0
fi

[ -f "$GATE" ] || { printf 'FAIL: %s is missing -- the gate this table exercises was deleted\n' "$GATE"; exit 1; }
command -v jq >/dev/null 2>&1 || { printf 'FAIL: jq is required\n'; exit 1; }

T=$(mktemp -d) || exit 1
trap 'rm -rf "${T:?}"' EXIT

HDR='binary,feature,quality_1_10,verified_hardware,top_competitor,in_dogfood_skill,cluster_id,cluster_label,evidence_path,confidence'

# Two packages: one [[bin]] each, plus a lib target that must NOT count.
cat > "$T/meta2.json" <<'EOF'
{"packages":[
 {"name":"a","targets":[{"name":"aprender-alpha","kind":["bin"]},{"name":"a","kind":["lib"]}]},
 {"name":"b","targets":[{"name":"aprender-beta","kind":["bin"]}]}
]}
EOF
printf '{"packages":[{"name":"a","targets":[{"name":"a","kind":["lib"]}]}]}\n' > "$T/meta0.json"

{ printf '%s\n' "$HDR"
  printf 'aprender-alpha,aprender-alpha run,6,UNKNOWN,x,no,1,c,src/a.rs:1,high\n'
  printf 'aprender-alpha,aprender-alpha list,6,UNKNOWN,x,no,1,c,src/a.rs:2,high\n'
  printf 'aprender-beta,aprender-beta go,6,UNKNOWN,x,no,1,c,src/b.rs:1,high\n'
} > "$T/full.csv"
: > "$T/empty.csv"
printf '%s\n' "$HDR" > "$T/header_only.csv"
grep -v '^aprender-beta,' "$T/full.csv" > "$T/missing.csv"
{ cat "$T/full.csv"; printf 'trueno-rag,trueno-rag eval,6,UNKNOWN,x,no,1,c,src/r.rs:1,high\n'; } > "$T/stray.csv"
sed '1s/^binary,/bin,/' "$T/full.csv" > "$T/badhdr.csv"

pass=0; fail=0
row() {
    # $1 label  $2 expected rc  $3 substring the output must contain  $4.. the command
    local label="$1" want="$2" needle="$3" out rc; shift 3
    out=$("$@" 2>&1); rc=$?
    if [ "$rc" -eq "$want" ] && [ "$(printf '%s' "$out" | grep -cF -- "$needle")" -gt 0 ]; then
        pass=$((pass+1)); printf 'ok    %s\n' "$label"
    else
        fail=$((fail+1)); printf 'FAIL  %s (rc=%s want %s; wanted "%s")\n' "$label" "$rc" "$want" "$needle"
        printf '%s\n' "$out" | sed 's/^/        /' | tail -n 6
    fi
}
feed() { local input="$1"; shift; printf '%b' "$input" | "$@"; }

row " 1 full ledger, every [[bin]] present -> GREEN 2/2" 0 "GREEN 2/2" \
    bash "$GATE" --csv "$T/full.csv" --metadata "$T/meta2.json"
row " 2 MUTATION empty ledger (0 bytes) -> RED" 1 "RED" \
    bash "$GATE" --csv "$T/empty.csv" --metadata "$T/meta2.json"
row " 3 MUTATION header only, 0 rows -> RED 0 rows" 1 "has 0 rows" \
    bash "$GATE" --csv "$T/header_only.csv" --metadata "$T/meta2.json"
row " 4 a shipped [[bin]] has no row -> RED by name, 1/2" 1 "RED  [[bin]] aprender-beta ships with no row" \
    bash "$GATE" --csv "$T/missing.csv" --metadata "$T/meta2.json"
row " 4b ... and the count is printed N/N" 1 "binaries 1/2" \
    bash "$GATE" --csv "$T/missing.csv" --metadata "$T/meta2.json"
row " 5 ledger names a binary the workspace does not ship -> RED by name" 1 "ledger binary trueno-rag is not a [[bin]]" \
    bash "$GATE" --csv "$T/stray.csv" --metadata "$T/meta2.json"
row " 6 ledger file absent -> RED" 1 "does not exist" \
    bash "$GATE" --csv "$T/nope.csv" --metadata "$T/meta2.json"
row " 7 column 1 renamed -> RED, never a silent reparse" 1 'expected "binary"' \
    bash "$GATE" --csv "$T/badhdr.csv" --metadata "$T/meta2.json"
row " 8 metadata declares 0 binaries -> RED (vacuous universe)" 1 "declares 0 binary targets" \
    bash "$GATE" --csv "$T/full.csv" --metadata "$T/meta0.json"
row " 9 MUTATION producer emits nothing -> RED" 1 "emitted 0 feature rows" \
    feed '' bash "$GATE" --require-rows
row "10 MUTATION producer emits only bookkeeping lines -> RED" 1 "emitted 0 feature rows" \
    feed 'FEATURESET\t<default>\nUNPROBED\tptop (declared, not built)\n' bash "$GATE" --require-rows
row "11 producer emits a feature row -> passes it through, rc 0" 0 "apr	apr run" \
    feed 'FEATURESET\t<default>\napr\tapr run\n' bash "$GATE" --require-rows

printf '\n%s/%s rows hold\n' "$pass" "$((pass+fail))"
[ "$fail" -eq 0 ]
