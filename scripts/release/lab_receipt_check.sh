#!/usr/bin/env bash
# lab_receipt_check.sh — judge one nightly lab receipt against contracts/nightly-lab-receipt-v1.yaml (#7155, epic #7149 R1)
#
# A receipt is the JSON a LAB lane writes each night, one entry per cell. A cell is MEASURED or NOT_MEASURED and
# nothing else; a NOT_MEASURED cell names its reason from a closed enum; a check is absent or {kind, limit, ok}.
# Anything else is invalid. This script holds the shape; the contract names it; --self-test is the case table and
# --mutants plants each wrong shape rule into a copy of the judge and requires the table to go red.
#
# USAGE  lab_receipt_check.sh FILE      judge one receipt
#        lab_receipt_check.sh --self-test   the case table
#        lab_receipt_check.sh --mutants     every planted mutant must turn the case table red
# EXIT   0 valid / table held / all mutants killed · 1 invalid / row broke / mutant survived
#        2 not_measured (jq missing) · 3 caller error
set -uo pipefail

# The judge: reads one receipt, prints one line per defect, nothing when valid.
read -r -d '' PROG <<'JQ' || true
def top: ["schema_version","lane","sha","runner_name","started","finished","device","cells"];
def dev: ["model","mem_mb","temp_c_start","temp_c_end"];
def cellk: ["id","state","value","unit","n","median","spread","baseline","delta_pct","reason","check"];
def reasons: ["runner_not_started","step_timeout","build_failed","tool_missing","run_failed"];
def kinds: ["tolerance","golden","baseline","size","loss","drift"];
def missing($req): . as $o | $req[] | select(. as $k | $o | has($k) | not) | "missing field: \(.)";
def extra($ok): . as $o | keys[] | select(. as $k | $ok | index($k) | not) | "unknown field: \(.)";
def cell_errs($i):
  if type != "object" then "cells[\($i)] is not an object" else
  ((.id // "?") | tostring) as $id
  | ((missing(cellk[:10]) , extra(cellk)) as $e | "cells[\($i)] \($id): \($e)"),
    (if has("state") and (.state != "MEASURED" and .state != "NOT_MEASURED")
       then "cells[\($i)] \($id): state is not MEASURED or NOT_MEASURED" else empty end),
    (if has("reason") and .reason != null and (.reason | IN(reasons[]) | not)
       then "cells[\($i)] \($id): reason is not in the enum" else empty end),
    (if .state == "NOT_MEASURED" and (has("reason") and .reason == null)
       then "cells[\($i)] \($id): NOT_MEASURED without a reason" else empty end),
    (if .state == "MEASURED" and (has("reason") and .reason != null)
       then "cells[\($i)] \($id): MEASURED with a reason" else empty end),
    (if has("check") then
       (.check | if type != "object" then "cells[\($i)] \($id): check is not an object" else
          ((missing(["kind","limit","ok"]), extra(["kind","limit","ok"])) as $c | "cells[\($i)] \($id): check \($c)")
        end),
       (.check | if type == "object" and has("kind") and (.kind | IN(kinds[]) | not)
          then "cells[\($i)] \($id): check kind is not in the enum" else empty end),
       (.check | if type == "object" and has("ok") and (.ok | type != "boolean")
          then "cells[\($i)] \($id): check ok is not a boolean" else empty end)
     else empty end)
  end;
if type != "object" then "receipt is not an object" else
  missing(top), extra(top),
  (if has("device") then (.device | if type != "object" then "device is not an object"
     else (missing(dev), extra(dev)) end) else empty end),
  (if has("cells") then (.cells | if type != "array" then "cells is not an array"
     else to_entries[] | .key as $i | .value | cell_errs($i) end) else empty end)
end
JQ

judge() { # judge PROG FILE -> defects on stdout; status 0 valid, 1 invalid, 3 unreadable
    local prog=$1 out
    out=$(jq -r "$prog" "$2" 2>&1) || { printf '%s\n' "$out"; return 3; }
    [ -z "$out" ] && return 0
    printf '%s\n' "$out"; return 1
}

# a valid receipt: one measured cell with a check, one not-measured cell
good() {
    cat <<'J'
{"schema_version":1,"lane":"wasm","sha":"aaaa","runner_name":"r1","started":"2026-11-17T00:00:00Z","finished":"2026-11-17T00:10:00Z",
 "device":{"model":"x86","mem_mb":64000,"temp_c_start":null,"temp_c_end":null},
 "cells":[
  {"id":"wasm.viz.size_gzip","state":"MEASURED","value":1200,"unit":"bytes","n":5,"median":1200,"spread":3,"baseline":1190,"delta_pct":0.8,"reason":null,"check":{"kind":"size","limit":1309,"ok":true}},
  {"id":"wasm.core.build","state":"NOT_MEASURED","value":null,"unit":null,"n":0,"median":null,"spread":null,"baseline":null,"delta_pct":null,"reason":"build_failed"}]}
J
}

# rows: "NAME|EXPECT|JQ-EDIT" applied to the good receipt
ROWS=(
  "good|valid|."
  "no schema_version|invalid|del(.schema_version)"
  "no lane|invalid|del(.lane)"
  "no sha|invalid|del(.sha)"
  "no runner_name|invalid|del(.runner_name)"
  "no started|invalid|del(.started)"
  "no finished|invalid|del(.finished)"
  "no device|invalid|del(.device)"
  "no cells|invalid|del(.cells)"
  "device field missing|invalid|del(.device.temp_c_end)"
  "unknown top field|invalid|.extra=1"
  "unknown device field|invalid|.device.gpu=1"
  "cell field missing|invalid|del(.cells[0].median)"
  "cell reason missing|invalid|del(.cells[1].reason)"
  "unknown cell field|invalid|.cells[0].note=1"
  "third state|invalid|.cells[0].state=\"SKIPPED\""
  "lowercase state|invalid|.cells[0].state=\"measured\""
  "reason outside enum|invalid|.cells[1].reason=\"offline\""
  "not measured without reason|invalid|.cells[1].reason=null"
  "measured with reason|invalid|.cells[0].reason=\"run_failed\""
  "check kind outside enum|invalid|.cells[0].check.kind=\"vibes\""
  "check ok not boolean|invalid|.cells[0].check.ok=\"yes\""
  "check field missing|invalid|del(.cells[0].check.limit)"
  "check extra field|invalid|.cells[0].check.why=1"
  "check not an object|invalid|.cells[0].check=true"
  "cells not an array|invalid|.cells={}"
  "cell not an object|invalid|.cells[0]=1"
  "check absent|valid|del(.cells[0].check)"
  "step_timeout reason|valid|.cells[1].reason=\"step_timeout\""
)

table() { # table PROG -> prints FAIL rows; status 1 if any
    local prog=$1 bad=0 row name expect edit tmp got
    tmp=$(mktemp) || return 3
    for row in "${ROWS[@]}"; do
        IFS='|' read -r name expect edit <<< "$row"
        good | jq -c "$edit" > "$tmp"
        got=valid; judge "$prog" "$tmp" > /dev/null || got=invalid
        if [ "$got" != "$expect" ]; then printf 'FAIL %s: want %s got %s\n' "$name" "$expect" "$got"; bad=1; fi
    done
    rm -f "${tmp:?}"; return $bad
}

# mutants: "OLD|NEW" substitutions in PROG; each must turn the table red
MUTANTS=(
  'missing($req): . as $o|missing($req): empty | . as $o'
  'IN(reasons[]) | not|IN(reasons[], "offline") | not'
  '.state != "MEASURED" and .state != "NOT_MEASURED"|.state != "MEASURED"'
  'extra($ok): . as $o|extra($ok): empty | . as $o'
  '.state == "NOT_MEASURED" and (has("reason") and .reason == null)|false'
  '.state == "MEASURED" and (has("reason") and .reason != null)|false'
  'IN(kinds[]) | not|IN(kinds[], "vibes") | not'
  '.ok | type != "boolean"|false'
  'missing(cellk[:10])|missing(cellk[:9])'
)

mutants() {
    local m old new mut survived=0 n=0
    for m in "${MUTANTS[@]}"; do
        old=${m%%|*}; new=${m#*|}; n=$((n + 1))
        mut=${PROG/"$old"/"$new"}
        if [ "$mut" = "$PROG" ]; then printf 'FAIL mutant %d changes nothing: %s\n' "$n" "$old"; survived=1; continue; fi
        if table "$mut" > /dev/null; then printf 'FAIL mutant %d survived: %s\n' "$n" "$old"; survived=1; fi
    done
    [ "$survived" -eq 0 ] && printf 'mutants: all %d killed\n' "$n"
    return $survived
}

command -v jq > /dev/null 2>&1 || { printf 'NOT_MEASURED: jq is missing\n'; exit 2; }
case "${1:-}" in
    --self-test) if table "$PROG"; then printf 'self-test: PASS (%d rows)\n' "${#ROWS[@]}"; else printf 'self-test: FAIL\n'; exit 1; fi ;;
    --mutants) mutants || exit 1 ;;
    -*|"") printf 'usage: lab_receipt_check.sh FILE | --self-test | --mutants\n' >&2; exit 3 ;;
    *) [ -r "$1" ] || { printf 'caller error: cannot read %s\n' "$1" >&2; exit 3; }
       judge "$PROG" "$1"; rc=$?; exit "$rc" ;;
esac
