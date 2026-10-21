#!/usr/bin/env bash
# flow010_receipt_toolchain.sh — FLOW-010 P3 PR-A (F10-4): a receipt whose toolchain differs from the
# fleet pin is invalid (R.P != P_fleet => invalid).
#
#   scripts/flow010_receipt_toolchain.sh <receipt.json> <fleet-pin.json>
#   scripts/flow010_receipt_toolchain.sh --selftest
#
# Exit: 0 valid · 1 invalid (the first offending tool is named on stderr) · 2 usage.
# Fail-closed: an unreadable file, bad JSON, a missing/non-object toolchain or an empty pin is invalid (rc 1),
# never valid; "nothing was compared" cannot pass.
#
# STUB (swap later): the receipt layout is FLOW-009 flow009.receipt/1 (infra#1884 receipt.sh @ a03047ca6) and
# its `toolchain` shape is not confirmed, so the whole receipt format sits behind ONE const, RECEIPT_TOOLCHAIN_JQ:
# a jq filter that must yield an object {tool: version}. The fleet pin is the same shape. The signature (RC-5 `sig`)
# is NOT checked here: that is a separate PR, once .flow009/batcher.pub exists in a repo.
set -euo pipefail

RECEIPT_TOOLCHAIN_JQ='.toolchain'

# toolchain_diff <receipt.json> <pin.json> -> prints one "tool: receipt=X pin=Y" line per mismatch; rc 1 if any/unreadable
toolchain_diff() {
  local r p out
  r="$(jq -c "$RECEIPT_TOOLCHAIN_JQ" "$1" 2>/dev/null)" || { echo "unreadable receipt: $1"; return 1; }
  p="$(jq -c '.' "$2" 2>/dev/null)" || { echo "unreadable fleet pin: $2"; return 1; }
  out="$(jq -rn --argjson r "$r" --argjson p "$p" '
    if ($r | type) != "object" then "receipt toolchain is not an object"
    elif ($p | type) != "object" or ($p | length) == 0 then "fleet pin is empty or not an object"
    else $p | to_entries[] | . as $e | ($r | .[$e.key]) as $got | select($got != $e.value)
         | "\($e.key): receipt=\($got // "absent") pin=\($e.value)" end')" || { echo "jq failed"; return 1; }
  [ -z "$out" ] || { printf '%s\n' "$out"; return 1; }
}

selftest() {
  local d fail=0 n=0
  d="$(mktemp -d)"; [ -n "$d" ] && [ -d "$d" ] || return 1
  trap 'rm -rf "${d:?}"' RETURN
  printf '{"toolchain":{"pv":"0.70.2","rustc":"1.93.0"}}' >"$d/ok.json"
  printf '{"toolchain":{"pv":"0.69.1","rustc":"1.93.0"}}' >"$d/oldpv.json"
  printf '{"toolchain":{"rustc":"1.93.0"}}' >"$d/missing.json"
  printf '{"toolchain":{"pv":"0.70.2","rustc":"1.93.0","extra":"1"}}' >"$d/extra.json"
  printf '{"toolchain":"pv 0.70.2"}' >"$d/str.json"
  printf '{"schema":"flow009.receipt/1"}' >"$d/none.json"
  printf 'not json' >"$d/bad.json"
  printf '{"pv":"0.70.2","rustc":"1.93.0"}' >"$d/pin.json"
  printf '{}' >"$d/emptypin.json"
  # case: file  pin  expected-rc  label
  while read -r f pin want label; do
    n=$((n + 1))
    local rc=0
    toolchain_diff "$d/$f" "$d/$pin" >"$d/out" 2>&1 || rc=$?
    if [ "$rc" != "$want" ]; then echo "FAIL $label: rc=$rc want=$want ($(cat "$d/out"))"; fail=$((fail + 1)); fi
  done <<EOF
ok.json pin.json 0 matching-toolchain-valid
oldpv.json pin.json 1 F10-4-pv-0.69.1-vs-fleet-0.70.2-rejected
missing.json pin.json 1 pinned-tool-absent-from-receipt
extra.json pin.json 0 extra-receipt-tool-not-in-pin-ok
str.json pin.json 1 toolchain-not-an-object
none.json pin.json 1 no-toolchain-field
bad.json pin.json 1 unparseable-receipt
nonexistent.json pin.json 1 missing-receipt-file
ok.json emptypin.json 1 empty-pin-compares-nothing
ok.json bad.json 1 unparseable-pin
EOF
  # the offending tool is named
  n=$((n + 1))
  toolchain_diff "$d/oldpv.json" "$d/pin.json" >"$d/out" 2>&1 || true
  grep -q '^pv: receipt=0.69.1 pin=0.70.2$' "$d/out" \
    || { echo "FAIL names-offending-tool"; fail=$((fail + 1)); }
  echo "flow010_receipt_toolchain selftest: $n cases, $fail failed"
  [ "$fail" = 0 ]
}

case "${1:-}" in
  --selftest) selftest ;;
  '' | -*) echo "usage: $0 <receipt.json> <fleet-pin.json> | --selftest" >&2; exit 2 ;;
  *)
    [ $# -eq 2 ] || { echo "usage: $0 <receipt.json> <fleet-pin.json> | --selftest" >&2; exit 2; }
    if diff="$(toolchain_diff "$1" "$2")"; then echo "valid: toolchain matches fleet pin"; else
      printf 'INVALID receipt toolchain: %s\n' "$diff" >&2; exit 1
    fi ;;
esac
