#!/usr/bin/env bash
# dogfood_jq_parsers_test.sh — case table for the two jq readers of the
# dogfood declarations (#4377: both were Python until then):
#   scripts/lib/dogfood_gates.jq       [package.metadata.dogfood] gates
#   scripts/lib/dogfood_transports.jq  [package.metadata.transports]
#
# Each row is <reader> <fixture JSON> <expected output>, where the expected
# output is every line joined by `|`, or ERROR when jq must fail (the caller maps
# a failure to META_ERROR). The gates rows were first checked against the
# Python they replaced, on the same fixtures, before it was deleted.
#
# Usage: bash scripts/tests/dogfood_jq_parsers_test.sh     exit 0 all rows pass, 1 a row
#        failed, 2 the test could not run (no jq, a reader file missing).
set -uo pipefail

ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null) || { echo "ENV  not inside a git checkout" >&2; exit 2; }
command -v jq >/dev/null 2>&1 || { echo "ENV  jq is not installed" >&2; exit 2; }
G="$ROOT/scripts/lib/dogfood_gates.jq"
T="$ROOT/scripts/lib/dogfood_transports.jq"
[ -f "$G" ] && [ -f "$T" ] || { echo "ENV  a reader is missing ($G, $T)" >&2; exit 2; }

fails=0 rows=0
row() { # row <name> <gates|transports> <json> <expected>
  local name=$1 reader=$2 json=$3 want=$4 got out rc=0 f
  case "$reader" in gates) f=$G ;; transports) f=$T ;; esac
  out=$(printf '%s' "$json" | jq -r --arg crate foo -f "$f" 2>/dev/null) || rc=$?
  got=$(printf '%s' "$out" | paste -sd'|' -)
  # A failed read must also print NOTHING: half a plan beside the error is the bug.
  if [ "$rc" -ne 0 ]; then got="ERROR${out:+ with output: $got}"; fi
  rows=$((rows + 1))
  if [ "$got" = "$want" ]; then
    printf 'ok    %-34s %s\n' "$name" "$want"
  else
    printf 'FAIL  %-34s want [%s] got [%s]\n' "$name" "$want" "$got"
    fails=$((fails + 1))
  fi
}

# ── gates ───────────────────────────────────────────────────────────────────
row gates-two-pkgs-no-match gates '{"packages":[{"name":"x","metadata":{"dogfood":{"gates":["a.sh"]}}},{"name":"y"}]}' NOPKG
row gates-one-pkg-other-name gates '{"packages":[{"name":"only","metadata":{"dogfood":{"gates":["g.sh"]}}}]}' 'GATE g.sh'
row gates-named-pkg-trimmed gates '{"packages":[{"name":"y"},{"name":"foo","metadata":{"dogfood":{"gates":[" scripts/a.sh ","scripts/check_noext"]}}}]}' 'GATE scripts/a.sh|GATE scripts/check_noext'
row gates-null-metadata gates '{"packages":[{"name":"foo","metadata":null}]}' NODECL
row gates-dogfood-not-table gates '{"packages":[{"name":"foo","metadata":{"dogfood":[1]}}]}' NODECL
row gates-not-a-list gates '{"packages":[{"name":"foo","metadata":{"dogfood":{"gates":"g.sh"}}}]}' BADSHAPE
row gates-empty-list gates '{"packages":[{"name":"foo","metadata":{"dogfood":{"gates":[]}}}]}' EMPTY
row gates-blank-entry gates '{"packages":[{"name":"foo","metadata":{"dogfood":{"gates":["a.sh","  "]}}}]}' BADSHAPE
row gates-non-string-entry gates '{"packages":[{"name":"foo","metadata":{"dogfood":{"gates":["a.sh",3]}}}]}' BADSHAPE
row gates-no-packages gates '{}' NOPKG

# ── transports ──────────────────────────────────────────────────────────────
TGT='"targets":[{"name":"foo","kind":["lib"],"src_path":"src/lib.rs"},{"name":"mcp_e2e","kind":["test"],"src_path":"tests/mcp_e2e.rs"}]'
row tp-no-pkg transports '{"packages":[{"name":"b"},{"name":"a"}]}' 'NOPKG a,b'
row tp-no-table transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":null}]}" NODECL
row tp-empty-table transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":{\"transports\":{}}}]}" NODECL
row tp-decls-sorted transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":{\"transports\":{\"mcp\":{\"e2e\":\"mcp_e2e\",\"features\":[\"mcp\",\"x\"]},\"cli\":{}}}}]}" 'DECL cli - - -|DECL mcp mcp_e2e mcp,x tests/mcp_e2e.rs'
row tp-e2e-not-a-target transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":{\"transports\":{\"http\":{\"e2e\":\"nope\"}}}}]}" 'DECL http nope - -'
row tp-manifest transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":{\"transports\":{\"manifest\":{\"path\":\"m.json\"},\"cli\":{}}}}]}" 'MANIFEST m.json -|DECL cli - - -'
row tp-bare-boolean transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":{\"transports\":{\"cli\":true}}}]}" 'BADSHAPE cli'
row tp-transports-not-table transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":{\"transports\":[1]}}]}" ERROR
row tp-transport-not-table transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":{\"transports\":{\"cli\":\"yes\"}}}]}" ERROR
row tp-features-not-strings transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":{\"transports\":{\"cli\":{\"features\":[1]}}}}]}" ERROR
row tp-missing-targets transports '{"packages":[{"name":"foo","metadata":{"transports":{"cli":{}}}}]}' ERROR
# all or nothing: a bad transport after a good one prints nothing, not half a plan
row tp-error-prints-nothing transports "{\"packages\":[{\"name\":\"foo\",$TGT,\"metadata\":{\"transports\":{\"a\":{},\"b\":\"bad\"}}}]}" ERROR

if [ "$fails" -ne 0 ]; then
  printf 'FAIL  %d of %d rows\n' "$fails" "$rows"
  exit 1
fi
printf 'PASS  %d rows\n' "$rows"
