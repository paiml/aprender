#!/usr/bin/env bash
# check_ladder_cells_producer.sh — the cells[] PRODUCER (#3712 row B) against its own JUDGE.
#
# scripts/lib/model_ladder_cells_produce.py writes the receipt's `cells[]`;
# scripts/lib/model_ladder_cells.py judges them. This runs the producer end to end
# against a fake `apr` (scripts/lib/model_ladder_cells_produce_cases/fake_apr.py,
# one defect per FAKE_APR_MODE), hands the rows to the judge, and asserts the
# verdict each defect must draw. No GPU, no model: seconds.
#
# Then every producer rule is deleted in a copy and the case that proves it must
# turn RED — a case table that stays green under its own mutant proves nothing.
#
#   bash scripts/check_ladder_cells_producer.sh     # 0 all cases + all mutants as expected · 1 otherwise
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
WORK=$(mktemp -d) || exit 2
trap 'rm -rf "${WORK:?}"' EXIT
FAKE="$ROOT/scripts/lib/model_ladder_cells_produce_cases/fake_apr.py"

cat > "$WORK/ladder.yaml" <<'EOF'
ladder:
  hosts: [{id: t, required: true}]
  cells:
    verbs: [run, chat, serve, code]
    long_rungs_for: {families: [qwen2], representatives: {}}
EOF
cat > "$WORK/rungs.json" <<'EOF'
{"schema": "apr-release-context-rungs/v1",
 "consumers": [{"consumer": "c", "max_prompt_tokens": 7000}],
 "rungs": [{"id": "4k", "tokens": 4096}, {"id": "8k", "tokens": 8192}]}
EOF
printf '{"file":"m.gguf","sha256":"%064d","bytes":1000}\n' 0 > "$WORK/inventory.jsonl"
printf 'm.gguf|%s/m.gguf\n' "$WORK" > "$WORK/models.txt"
: > "$WORK/m.gguf"

# run_case <lib-dir> <mode> -> prints the case's verdict lines (the harness below)
run_case() {
  local lib=$1 mode=$2 d
  d=$(mktemp -d "$WORK/$2.XXXXXX") || { echo "MKTEMP-FAIL"; return; }
  FAKE_APR_MODE=$mode python3 "$lib/model_ladder_cells_produce.py" enrich --apr "$FAKE" \
      --inventory "$WORK/inventory.jsonl" --models "$WORK/models.txt" --out "$d/inv.jsonl" > /dev/null 2>&1 \
      || { echo "ENRICH-CRASH"; return; }
  FAKE_APR_MODE=$mode python3 "$lib/model_ladder_cells_produce.py" measure --apr "$FAKE" \
      --inventory "$d/inv.jsonl" --models "$WORK/models.txt" --ladder "$WORK/ladder.yaml" \
      --rungs "$WORK/rungs.json" --work "$d" --timeout 60 --serve-ceiling 30 --out "$d/cells.json" > /dev/null 2>&1 \
      || { echo "MEASURE-CRASH"; return; }
  python3 - "$lib" "$WORK/ladder.yaml" "$WORK/rungs.json" "$d/inv.jsonl" "$d/cells.json" <<'PY'
import json, sys, yaml
sys.path.insert(0, sys.argv[1])
import model_ladder_cells as J
L = yaml.safe_load(open(sys.argv[2]))["ladder"]
inv = [json.loads(l) for l in open(sys.argv[4]) if l.strip()]
cells = json.load(open(sys.argv[5]))
R = {"inventory": inv, "cells": cells, "gpu_mem_total_bytes": 24564 * 1024 * 1024}
lines = []
J.judge(L, {"t": R}, json.load(open(sys.argv[3])), lines.append)
for l in lines:
    print("JUDGE " + l)
v = lambda verdict, verb=None: sum(c["verdict"] == verdict and (verb is None or c["verb"] == verb) for c in cells)
print(f"ROWS {len(cells)} PASS {v('pass')} REFUSED {v('refused')} RUNPASS {v('pass', 'run')} FELLBACK {sum(c.get('fallback') is True for c in cells)}")
print("MINPT " + str(min((c["prompt_tokens"] or 0) - (4096 if c["context"] == "4k" else 8192) for c in cells)))
print("INV " + json.dumps({k: inv[0].get(k) for k in ("arch", "context_length", "thinking_modes", "kv_bytes_per_token")}))
PY
}

# expect <lib> <label> <mode> <python-bool over `out` (the case's printed lines, one string)>
bad=0
SEQ=0
expect() {
  local lib=$1 label=$2 mode=$3 pred=$4 out
  out=$(run_case "$lib" "$mode")
  if python3 -c 'import sys; out = sys.stdin.read(); sys.exit(0 if eval(sys.argv[1]) else 1)' "$pred" <<< "$out"; then
    printf 'ok    %s\n' "$label"; return 0
  fi
  printf 'FAIL  %s\n%s\n' "$label" "$(sed 's/^/        /' <<< "$out" | head -12)"; return 1
}

# measure_rc <lib> <label> <want-rc> <extra measure args...> -> the quorum-2 guards (#3715): a --only that names
# no inventory model, and a rungs file the judge FAILs, must each refuse -- never write "0 row(s)" at rc 0.
measure_rc() {
  local lib=$1 label=$2 want=$3 d rc; shift 3
  SEQ=$((SEQ + 1)); d="$WORK/rc.$SEQ"; mkdir -p "$d" || return 1
  FAKE_APR_MODE=good python3 "$lib/model_ladder_cells_produce.py" enrich --apr "$FAKE" \
      --inventory "$WORK/inventory.jsonl" --models "$WORK/models.txt" --out "$d/inv.jsonl" > /dev/null 2>&1
  FAKE_APR_MODE=good python3 "$lib/model_ladder_cells_produce.py" measure --apr "$FAKE" \
      --inventory "$d/inv.jsonl" --models "$WORK/models.txt" --ladder "$WORK/ladder.yaml" \
      --work "$d" --timeout 60 --serve-ceiling 30 --out "$d/cells.json" "$@" > "$d/log" 2>&1
  rc=$?
  if [ "$rc" = "$want" ] && { [ "$want" = 0 ] || grep -q 'cells: refused' "$d/log"; }; then
    printf 'ok    %s\n' "$label"; return 0
  fi
  printf 'FAIL  %s (rc %s, want %s)\n%s\n' "$label" "$rc" "$want" "$(sed -n '$p' < "$d/log" | sed 's/^/        /')"; return 1
}

# offmode <lib> -> an off-mode prompt is sized by the OFF budget: with a thinking budget that leaves the on-mode
# prompt below the 4k rung, every off-mode row still reaches its rung's count (quorum 2 finding 3, #3715).
offmode() {
  local lib=$1 d
  SEQ=$((SEQ + 1)); d="$WORK/off.$SEQ"; mkdir -p "$d" || return 1
  FAKE_APR_MODE=good python3 "$lib/model_ladder_cells_produce.py" enrich --apr "$FAKE" \
      --inventory "$WORK/inventory.jsonl" --models "$WORK/models.txt" --out "$d/inv.jsonl" > /dev/null 2>&1
  FAKE_APR_MODE=good python3 "$lib/model_ladder_cells_produce.py" measure --apr "$FAKE" \
      --inventory "$d/inv.jsonl" --models "$WORK/models.txt" --ladder "$WORK/ladder.yaml" \
      --rungs "$WORK/rungs.json" --work "$d" --timeout 60 --serve-ceiling 30 --max-tokens-thinking 30000 \
      --out "$d/cells.json" > /dev/null 2>&1
  if python3 - "$d/cells.json" <<'PY2'
import json, sys
rows = [c for c in json.load(open(sys.argv[1])) if c["thinking"] == "off" and c["verb"] == "run"]
need = {"4k": 4096, "8k": 8192}
sys.exit(0 if rows and all((c["prompt_tokens"] or 0) >= need[c["context"]] for c in rows) else 1)
PY2
  then printf 'ok    %s\n' "offmode: an off-mode prompt leaves room for the off budget only, and reaches its rung"; return 0; fi
  printf 'FAIL  %s\n' "offmode: an off-mode prompt was sized by the thinking budget and fell below its rung"; return 1
}

cases() { # cases <lib> -> 0 all as expected
  local lib=$1 r=0
  expect "$lib" "enrich derives the owed-set terms from the header (arch, context, both thinking modes, KV)" good \
    '"\"arch\": \"qwen2\"" in out and "\"context_length\": 32768" in out and "[\"off\", \"on\"]" in out and "\"kv_bytes_per_token\": 57344" in out' || r=1
  expect "$lib" "good: every owed cell has ONE row (2 rungs x 4 verbs x 2 modes), all but code pass" good \
    '"cells: 16 owed, 12 pass" in out and "ROWS 16 " in out and not [l for l in out.splitlines() if l.startswith("JUDGE FAIL") and "code/" not in l and "PASSES on no" not in l]' || r=1
  expect "$lib" "good: apr code reports no backend, and the judge names it (apr's gap, visible)" good \
    '"code/on/4k '"'"'pass'"'"' is not a pass: backend None" in out or "backend None" in out' || r=1
  expect "$lib" "fellback: a cpu run after asking for gpu is not a pass" fellback '"fell back" in out and "FELLBACK 12" in out' || r=1
  expect "$lib" "noneedle: an answer without the token-0 needle is not a pass" noneedle '"RUNPASS 0" in out' || r=1
  expect "$lib" "noclose: a think block that never closes is not a pass" noclose '"thinking never closed" in out' || r=1
  expect "$lib" "refuse: a pre-load capacity refusal is a refused row with apr's arithmetic" refuse '"REFUSED 8" in out and "REFUSED without" not in out' || r=1
  expect "$lib" "undercount: a denser tokenizer than guessed still reaches every rung's count" undercount \
    'int(out.split("MINPT ")[1].split()[0]) >= 0' || r=1
  local rj="$WORK/rungs.json"
  measure_rc "$lib" "only-rung: --only naming a rung id (no inventory model) refuses, rc 2" 2 --rungs "$rj" --only 4k || r=1
  measure_rc "$lib" "only-model: --only naming the inventory model measures it, rc 0" 0 --rungs "$rj" --only inv:m.gguf || r=1
  printf '{"schema": "apr-release-context-rungs/v1", "rungs": []}\n' > "$WORK/rungs-empty.json"
  measure_rc "$lib" "rungs-empty: a rungs file the judge FAILs refuses by name, never 0 rows" 1 --rungs "$WORK/rungs-empty.json" || r=1
  offmode "$lib" || r=1
  return $r
}

echo "== cases"
cases "$ROOT/scripts/lib" || bad=1

echo "== mutants (each must turn a case RED)"
mutant() { # mutant <name> <sed-expr>
  local md="$WORK/mut-$1"
  mkdir -p "$md"
  cp scripts/lib/model_ladder_cells.py "$md/"
  sed "$2" scripts/lib/model_ladder_cells_produce.py > "$md/model_ladder_cells_produce.py"
  if cmp -s scripts/lib/model_ladder_cells_produce.py "$md/model_ladder_cells_produce.py"; then
    echo "FAIL  mutant $1 did not apply -- it proves nothing"; bad=1; return
  fi
  if cases "$md" > "$md/out" 2>&1; then
    echo "FAIL  mutant $1 SURVIVED every case"; bad=1
  else
    echo "ok    mutant $1 killed by: $(grep -m1 '^FAIL' "$md/out" | cut -c7-)"
  fi
}
mutant needle     's/elif NEEDLE_WORD not in answer:/elif False:/'
mutant fallback   's/row\["fallback"\] = be.get("fell_back")/row["fallback"] = False/'
mutant retry      's/for _attempt in range(3):/for _attempt in range(1):/'
mutant owed-set   's/for rid, tok in J.owed_rungs(item, rungs, C.get("long_rungs_for") or {}, consumer_max):/for (rid, tok), _one in zip(J.owed_rungs(item, rungs, C.get("long_rungs_for") or {}, consumer_max), range(1)):/'
mutant refusal    's/if rc != 0 and ref is not None:/if False:/'
mutant think      's/if "<\/think>" in text:/if False:/'
mutant only-vacuity 's/if a.only and not matched:/if False:/'
mutant rungs-silent 's/^    if rc:$/    if False:/'
mutant offmode-budget 's/target = min(tok, int(ctx) - budget - 1)/target = min(tok, int(ctx) - a.max_tokens_thinking - 1)/'

[ "$bad" = 0 ] && { echo "check_ladder_cells_producer: all cases and mutants as expected"; exit 0; }
exit 1
