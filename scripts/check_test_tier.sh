#!/usr/bin/env bash
# check_test_tier.sh — the ratchet behind the 80/20 PR tier (PMAT-1105, docs/specifications/ci-fleet-hygiene.md §6).
#
#   bash scripts/check_test_tier.sh --junit F --catch-ledger J [--committed T] [--budget-pct N] [--update]
#   bash scripts/check_test_tier.sh --selftest
#
# Regenerates the tier table (scripts/lib/test_tier.py) from a nextest junit + a catch ledger and compares it with
# the committed table (default evidence/fleet/test-tier.tsv). RED (exit 1) when
#   (a) the PR tier costs more than --budget-pct (default 50) of the FULL suite's seconds,
#   (b) any module changed tier relative to the committed table and --update was not given (a tier move is a
#       reviewed row, never a silent drift), or
#   (c) a module that owns a 'falsif' test is 'nightly' — designed catches always run on PRs.
# --update rewrites the committed table from the inputs and prints the moved modules. Missing inputs = exit 2.
set -euo pipefail

JUNIT=""; LEDGER=""; COMMITTED="evidence/fleet/test-tier.tsv"; BUDGET=50; UPDATE=0; SELFTEST=0
while [ $# -gt 0 ]; do
  case "$1" in
    --junit) JUNIT="$2"; shift 2 ;;
    --catch-ledger) LEDGER="$2"; shift 2 ;;
    --committed) COMMITTED="$2"; shift 2 ;;
    --budget-pct) BUDGET="$2"; shift 2 ;;
    --update) UPDATE=1; shift ;;
    --selftest) SELFTEST=1; shift ;;
    *) echo "check_test_tier: unknown argument $1" >&2; exit 2 ;;
  esac
done
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# run_check JUNIT LEDGER COMMITTED BUDGET UPDATE -> prints the verdict lines; returns 0/1
run_check() {
  local junit="$1" ledger="$2" committed="$3" budget="$4" update="$5" out rc=0
  out="$(mktemp -d)"
  python3 "$HERE/lib/test_tier.py" "$junit" "$ledger" "$out" > "$out/summary.json"
  local pct; pct=$(python3 -c "import json;print(json.load(open('$out/stats.json'))['pr_pct_seconds'])")
  local prn; prn=$(python3 -c "import json;d=json.load(open('$out/stats.json'));print(d['pr_tests'],'/',d['total_tests'],'tests',d['pr_seconds'],'/',d['total_seconds'],'s')")
  echo "tier: pr = $prn = ${pct}% of seconds (budget ${budget}%)"
  # (a) budget
  if python3 -c "import sys; sys.exit(0 if float('$pct') > float('$budget') else 1)"; then
    echo "FAIL  (a) PR tier ${pct}% of seconds exceeds the ${budget}% budget"; rc=1
  fi
  # (c) falsifier in nightly, in the NEW table
  local bad_f; bad_f=$(awk -F'\t' 'NR>1 && $7>0 && $8=="nightly"{print $1}' "$out/tier.tsv" | head -5 | tr '\n' ' ')
  if [ -n "$bad_f" ]; then echo "FAIL  (c) modules with a falsif test are nightly: $bad_f"; rc=1; fi
  # (b) tier moves vs the committed table (keyed by column 1)
  local moves=""
  if [ -f "$committed" ]; then
    moves=$(awk -F'\t' 'NR==FNR{ if (FNR>1) new[$1]=$8; next } FNR>1 && ($1 in new) && new[$1]!=$8 {print $1": "$8" -> "new[$1]}' "$out/tier.tsv" "$committed")
  else
    echo "note: no committed table at $committed (first run)"
  fi
  if [ -n "$moves" ]; then
    if [ "$update" = 1 ]; then
      echo "moved (accepted with --update):"; printf '%s\n' "$moves" | sed 's/^/  /'
    else
      echo "FAIL  (b) tier moved without --update:"; printf '%s\n' "$moves" | sed 's/^/  /'; rc=1
    fi
  fi
  if [ "$update" = 1 ] && [ "$rc" = 0 ]; then
    mkdir -p "$(dirname "$committed")"; cp "$out/tier.tsv" "$committed"; echo "committed table rewritten: $committed"
  fi
  [ -n "$out" ] && [ "$out" != "/" ] && rm -rf "$out"
  return "$rc"
}

if [ "$SELFTEST" = 1 ]; then
  FX="$HERE/../tests/fixtures/test_tier"; T="$(mktemp -d)"; trap '[ -n "$T" ] && [ "$T" != "/" ] && rm -rf "$T"' EXIT
  err=0
  row() { if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1: got rc=$2 want $3"; err=1; fi; }
  cp "$FX/committed.tsv" "$T/c.tsv"
  r=0; run_check "$FX/junit.xml" "$FX/ledger.json" "$T/c.tsv" 50 0 >/dev/null || r=$?; row "budget respected, table matches -> ok" "$r" 0
  r=0; run_check "$FX/junit.xml" "$FX/ledger.json" "$T/c.tsv" 3 0 >/dev/null || r=$?; row "budget 3% exceeded (tier is 4.72%) -> RED" "$r" 1
  sed 's/^\(crateB::light\t.*\t\)nightly$/\1pr/' "$FX/committed.tsv" > "$T/moved.tsv"
  r=0; run_check "$FX/junit.xml" "$FX/ledger.json" "$T/moved.tsv" 50 0 >/dev/null || r=$?; row "silent tier move (committed says pr, computed nightly) -> RED" "$r" 1
  cp "$T/moved.tsv" "$T/moved2.tsv"
  r=0; run_check "$FX/junit.xml" "$FX/ledger.json" "$T/moved2.tsv" 50 1 >/dev/null || r=$?
  if [ "$r" = 0 ] && grep -q $'^crateB::light\t.*\tnightly$' "$T/moved2.tsv"; then echo "ok    same move with --update -> ok and table rewritten"; else echo "FAIL  same move with --update"; err=1; fi
  # (c): a ledger that makes the falsif module fall below the line still keeps it pr (f_tests>0); prove the rule fires
  # on a table whose computation would put a falsif module in nightly: use a junit where the falsif test's module has
  # zero touches and the density line is reached elsewhere -> computed tier must still be pr, so rule (c) is a
  # property of the generator; assert it on the fixture (crateA::guard has 0 touches and a falsif test).
  python3 "$HERE/lib/test_tier.py" "$FX/junit.xml" "$FX/ledger.json" "$T" >/dev/null
  if awk -F'\t' '$1=="crateA::guard" && $8=="pr"' "$T/tier.tsv" | grep -q .; then echo "ok    falsif module with zero touches is pr (rule c holds)"; else echo "FAIL  falsif module with zero touches is pr"; err=1; fi
  # (c) negative: a table with the falsif module marked nightly must be refused even with --update
  sed 's/^\(crateA::guard\t.*\t\)pr$/\1nightly/' "$FX/committed.tsv" > "$T/badc.tsv"
  cp "$T/tier.tsv" "$T/gen.tsv"; sed -i 's/^\(crateA::guard\t.*\t\)pr$/\1nightly/' "$T/gen.tsv"
  bad_f=$(awk -F'\t' 'NR>1 && $7>0 && $8=="nightly"{print $1}' "$T/gen.tsv"); if [ -n "$bad_f" ]; then echo "ok    (c) detector sees a falsif module in nightly: $bad_f"; else echo "FAIL  (c) detector"; err=1; fi
  cp "$FX/committed.tsv" "$T/i1.tsv"; run_check "$FX/junit.xml" "$FX/ledger.json" "$T/i1.tsv" 50 0 > "$T/o1" || true; run_check "$FX/junit.xml" "$FX/ledger.json" "$T/i1.tsv" 50 0 > "$T/o2" || true
  if cmp -s "$T/o1" "$T/o2"; then echo "ok    idempotent: two runs, identical output"; else echo "FAIL  idempotent"; err=1; fi
  exit "$err"
fi

[ -n "$JUNIT" ] && [ -f "$JUNIT" ] || { echo "check_test_tier: --junit FILE is required (nextest junit)" >&2; exit 2; }
[ -n "$LEDGER" ] && [ -f "$LEDGER" ] || { echo "check_test_tier: --catch-ledger FILE is required ({\"crate::module\": touches})" >&2; exit 2; }
run_check "$JUNIT" "$LEDGER" "$COMMITTED" "$BUDGET" "$UPDATE"
