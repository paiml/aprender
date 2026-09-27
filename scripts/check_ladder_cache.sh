#!/usr/bin/env bash
# check_ladder_cache.sh — the ladder's receipt cache reuses ONLY an exact green (#4520 step 5).
#
# WHY. A rerun of one binary re-read every held model off the disk (17 GB per 27B cell per verb) to
# re-prove what the same bytes had already proved. The cache stops that -- and a cache is the easiest
# place in a gate to launder a stale green, so every field of its key has a must-MISS row here.
#
# TWO LAYERS.
#   1. scripts/lib/ladder_cache.py over a case table: one exact hit, then one row per key field
#      (host, apr_sha, binary bytes, contract, model sha, backends, "unknown") and per row state
#      (red, budget-violated, absent) that must MISS; a copied row keeps the date that MEASURED it.
#   2. THE SHIPPED measure() (lifted from model_ladder.sh) against a planted previous receipt, with
#      every measuring step stubbed to write a marker: a hit must append the copied row and measure
#      NOTHING; a miss, --no-cache and --only must each reach the measuring path.
# --self-test plants a lookup that ignores the binary's bytes and requires this check to go RED.
#
# Exit: 0 all as expected · 1 a case landed wrong · 2 could not check.
set -uo pipefail
SELF_TEST=0
case "${1:-}" in --self-test) SELF_TEST=1 ;; "") ;; *) echo "unknown argument '$1'" >&2; exit 2 ;; esac
ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "  cannot check: not in a repository" >&2; exit 2; }
cd "$ROOT" || exit 2

T=$(mktemp -d) || exit 2
trap 'rm -rf -- "${T:?}"' EXIT
mkdir -p "$T/lib"
cp scripts/lib/ladder_cache.py "$T/lib/ladder_cache.py"
if [ "$SELF_TEST" = 1 ]; then   # the mutant: a rebuilt binary at the same HEAD still hits
  sed -i 's/("apr_bin_sha256", bin_sha),//' "$T/lib/ladder_cache.py"
  grep -q '"apr_bin_sha256", bin_sha' "$T/lib/ladder_cache.py" && { echo "  cannot check: self-test mutation did not apply" >&2; exit 2; }
fi
fails=0
ok()  { printf '  ok    %s\n' "$1"; }
bad() { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }

# ── layer 1: the lookup's case table ──────────────────────────────────────────
python3 - "$T/lib" > "$T/l1.out" <<'PY'
import copy, sys
sys.path.insert(0, sys.argv[1]); import ladder_cache as C
row = {"id": "r1", "sha256": "m" * 64, "green": True, "backends": {"cpu": {}, "cuda": {}}}
prev = {"host": "lambda", "apr_sha": "a" * 40, "apr_bin_sha256": "b" * 64, "contract_sha256": "c" * 64,
        "date": "2026-09-28T01:00Z", "rungs": [row]}
key = ["lambda", "a" * 40, "b" * 64, "c" * 64, "r1", "m" * 64, "cpu,cuda"]
def case(name, want_hit, p=prev, k=key):
    got, why = C.lookup(p, *k)
    good = (got is not None) == want_hit
    print(("ok   " if good else "FAIL ") + f" {name}: want {'hit' if want_hit else 'miss'}, got {'hit' if got else 'miss (' + str(why) + ')'}")
    return got
hit = case("exact-key-hit", True)
print(("ok   " if hit and hit.get("cached_from", {}).get("date") == prev["date"] else "FAIL ")
      + " hit-names-its-source: cached_from.date is the measuring run's date")
for i, name in enumerate(["host", "apr-sha", "binary-bytes", "contract", "cell-id", "model-sha", "backends"]):
    k = list(key); k[i] = "cpu" if name == "backends" else "x"
    case(f"{name}-differs", False, k=k)
for i in (1, 2, 3):
    k = list(key); k[i] = "unknown"; p = dict(prev); p[["", "apr_sha", "apr_bin_sha256", "contract_sha256"][i]] = "unknown"
    case(f"unknown-{i}-never-hits", False, p=p, k=k)
p = dict(prev); p.pop("apr_bin_sha256"); case("pre-cache-receipt-misses", False, p=p)
for name, mut in [("red-row", {"green": False}), ("budget-violated", {"budget_violations": [{"budget": "x"}]})]:
    p = copy.deepcopy(prev); p["rungs"][0].update(mut); case(f"{name}-misses", False, p=p)
p = copy.deepcopy(prev); p["rungs"] = []; case("absent-cell-misses", False, p=p)
p = copy.deepcopy(prev); p["date"] = "2026-09-29T00:00Z"; p["rungs"][0]["cached_from"] = {"date": "2026-09-01T00:00Z"}
got = case("chained-copy-hits", True, p=p)
print(("ok   " if got and got["cached_from"]["date"] == "2026-09-01T00:00Z" else "FAIL ")
      + " chained-copy-keeps-the-measuring-date")
PY
[ $? = 0 ] || { echo "  cannot check: layer 1 crashed" >&2; exit 2; }
while IFS= read -r l; do case "$l" in ok*) ok "${l#ok    }" ;; *) bad "${l#FAIL  }" ;; esac; done < "$T/l1.out"

# ── layer 2: the shipped measure() against a planted previous receipt ─────────
body=$(sed -n '/^measure() {/,/^# ---- 1\. the ladder/p' scripts/model_ladder.sh | sed '$d')
[ -n "$body" ] || { echo "  cannot check: measure() not found in model_ladder.sh" >&2; exit 2; }
# the lifted body calls scripts/lib/ladder_cache.py by repo path: point it at the (maybe mutated) copy
body=${body//scripts\/lib\/ladder_cache.py/$T\/lib\/ladder_cache.py}
eval "$body"
declare -F measure >/dev/null || { echo "  cannot check: measure() did not load" >&2; exit 2; }
# every measuring step after the cache is replaced by a marker: reaching it means "measured"
ladder_disk_probe() { :; }
ladder_append() { printf '%s\n' "$2" >> "$1"; }
fit_verdict() { echo MEASURED >> "$T/measured"; echo '{"verdict": "does-not-fit", "reason": "stub"}'; }
ladder_write_decline() { echo "decline: $*" >&2; exit 2; }
OUT_DIR="$T/out"; HOST=lambda; mkdir -p "$OUT_DIR"; WORK="$T/work"; mkdir -p "$WORK"
APR_SHA=$(printf 'a%.0s' {1..40}); APR_BIN_SHA=$(printf 'b%.0s' {1..64}); CONTRACT_SHA=$(printf 'c%.0s' {1..64})
MSHA=$(printf 'm%.0s' {1..64}); MODEL="$T/model.gguf"; : > "$MODEL"
python3 - "$OUT_DIR/lambda.json" "$APR_SHA" "$APR_BIN_SHA" "$CONTRACT_SHA" "$MSHA" <<'PY'
import json, sys
json.dump({"host": "lambda", "apr_sha": sys.argv[2], "apr_bin_sha256": sys.argv[3], "contract_sha256": sys.argv[4],
           "date": "2026-09-28T01:00Z", "rungs": [{"id": "r1", "sha256": sys.argv[5], "green": True,
                                                  "backends": {"cpu": {}, "cuda": {}}}]}, open(sys.argv[1], "w"))
PY
run_cell() { # <no_cache> <only> <apr_bin_sha> -> sets hit=1 when nothing was measured and one cached row appended
  NO_CACHE=$1; ONLY=$2; APR_BIN_SHA=$3; EXECUTED=0; RED=0; ROWS="$T/rows"; : > "$ROWS"; : > "$T/measured"
  ( measure r1 model.gguf "$MODEL" "$MSHA" cpu,cuda 1 0 ) > "$T/cell.out" 2>&1
  if [ ! -s "$T/measured" ] && grep -q '"cached_from"' "$ROWS"; then hit=1; else hit=0; fi
}
run_cell 0 "" "$APR_BIN_SHA"
[ "$hit" = 1 ] && ok "measure: exact key copies the row and measures nothing" \
               || bad "measure: exact key was re-measured or not copied ($(head -c 200 "$T/cell.out"))"
run_cell 0 "" "$(printf 'd%.0s' {1..64})"
[ "$hit" = 0 ] && [ -s "$T/measured" ] && ok "measure: a rebuilt binary (other bytes) is re-measured" \
               || bad "measure: a rebuilt binary at the same HEAD reused the old green"
run_cell 1 "" "$(printf 'b%.0s' {1..64})"
[ "$hit" = 0 ] && [ -s "$T/measured" ] && ok "measure: --no-cache re-measures" || bad "measure: --no-cache still copied"
run_cell 0 r1 "$(printf 'b%.0s' {1..64})"
[ "$hit" = 0 ] && [ -s "$T/measured" ] && ok "measure: --only re-measures (a targeted rerun is for separating a flake)" \
               || bad "measure: --only copied a cached row"

if [ "$SELF_TEST" = 1 ]; then
  [ "$fails" -gt 0 ] && { echo "self-test: the planted bytes-blind cache turned this RED ($fails case(s)) -- good"; exit 0; }
  echo "self-test: FAIL -- a cache that ignores the binary's bytes passed every case"; exit 1
fi
[ "$fails" = 0 ] && { echo "ladder cache: all cases as expected"; exit 0; }
echo "ladder cache: $fails case(s) wrong"; exit 1
