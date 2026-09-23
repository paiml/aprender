#!/usr/bin/env bash
# check_ont_4c5_probe.sh — ONT-4c5's own probe (paiml/infra paiml-ontology.md v4.12 §5, 948ae923), run on this
# tree, and then the row's mutations, each of which MUST turn the probe RED.
#
# The probe is the row's jq predicate, verbatim: verdict Pass, capability-cells armed, pc_shapes fired, no
# NotRun cell, and the 14 ids of the baseline set B inside the domain. A probe that stays green under a mutation
# it names is theater, so every mutation below is expected RED and the script fails if one is not.
#
# Mutations run in a scratch copy: contracts/ and evidence/ are COPIED (they are what a mutation edits); every
# other top-level entry is symlinked, so the extractors that read crates/ or lean/ see the real tree. The
# checkout is never edited.
#
# The row's fifth mutation — "fold a missing cell into Fail → the plant passes" — is a CODE mutation; it is
# held by `cargo test -p aprender-contracts --lib lint::capability_cells_gate` and the receipt's mutant log,
# not here.
#
# Exit: 0 probe green and every mutation RED · 1 a case disagreed · 2 usage (pv not resolvable, jq missing).
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
. scripts/pv_bin.sh || exit 2
command -v jq >/dev/null || { echo "check_ont_4c5_probe: jq not found" >&2; exit 2; }

B='["qwen2-1.5b-q4km@lambda","qwen2-1.5b-q4km@gx10","qwen3-1.7b-q4km@lambda","qwen3-1.7b-q4km@gx10","qwen35-0.8b-q4km@lambda","qwen35-0.8b-q4km@gx10","qwen35-2b-q4km@lambda","qwen35-2b-q4km@gx10","qwen35-4b-q4km@lambda","qwen35-4b-q4km@gx10","qwen35-9b-q4km@lambda","qwen35-9b-q4km@gx10","qwen35-27b-q4km@lambda","qwen35-27b-q4km@gx10"]'
PROBE='.verdict=="Pass" and (.armed_shapes|index("capability-cells")!=null) and .pc_shapes["capability-cells"]=="fired" and (.capability_cells.not_run|length)==0 and ('"$B"' - .capability_cells.domain)==[]'

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
fails=0

# probe <root> <label> <expect GREEN|RED> [<must-name>]
probe() {
    local root="$1" label="$2" expect="$3" name="${4:-}" rc=0 got
    (cd "$root" && "$PV" lint contracts/ --gate shapes --format json) > "$TMP/out.json" 2> "$TMP/err.txt" || rc=$?
    if [ -s "$TMP/out.json" ] && jq -e "$PROBE" "$TMP/out.json" > /dev/null 2>&1; then
        got=GREEN
    else
        got=RED
    fi
    if [ "$got" != "$expect" ]; then
        printf 'FAIL %-44s expected %s, got %s (pv rc=%s)\n' "$label" "$expect" "$got" "$rc"
        tail -3 "$TMP/err.txt" | sed 's/^/     /'
        fails=$((fails + 1))
        return 0
    fi
    if [ -n "$name" ] && ! grep -qF -- "$name" "$TMP/out.json" "$TMP/err.txt"; then
        printf 'FAIL %-44s %s as expected, but nothing names %s\n' "$label" "$got" "$name"
        fails=$((fails + 1))
        return 0
    fi
    printf 'ok   %-44s %s (pv rc=%s)%s\n' "$label" "$got" "$rc" "${name:+ naming $name}"
}

# scratch <dir>: contracts/ and evidence/ copied, everything else symlinked
scratch() {
    local d="$1" e
    mkdir -p "$d"
    for e in * .[!.]*; do
        [ -e "$e" ] || continue
        case "$e" in
            contracts|evidence) cp -R "$e" "$d/$e" ;;
            .git|.pv|target) ;;
            *) ln -s "$PWD/$e" "$d/$e" ;;
        esac
    done
}

probe . "baseline (this tree)" GREEN

# M1 — drop capability-cells from armed_shapes
scratch "$TMP/m1"
python3 - "$TMP/m1/contracts/lint-baseline.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
assert "capability-cells" in d["armed_shapes"], "mutation anchor missing"
d["armed_shapes"].remove("capability-cells"); json.dump(d, open(p, "w"), indent=2)
PY
probe "$TMP/m1" "M1 capability-cells disarmed" RED

# M2 — flip one required rung (qwen35-2b-q4km) to required: false → B ⊄ D
scratch "$TMP/m2"
python3 - "$TMP/m2/contracts/model-capability-ladder-v1.yaml" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
m = re.search(r"- id: qwen35-2b-q4km\b", s)
assert m, "mutation anchor missing"
tail = s[m.end():]
r = re.search(r"required: true", tail)
nxt = re.search(r"\n\s*- id: ", tail)
assert r and (nxt is None or r.start() < nxt.start()), "rung has no required: true of its own"
s = s[:m.end()] + tail[:r.start()] + "required: false" + tail[r.end():]
open(p, "w").write(s)
PY
probe "$TMP/m2" "M2 qwen35-2b-q4km required: false" RED

# M3..M5 — label one required V* row DEFER / MANUAL / NO-VERDICT → exit 1 naming the cell
for label in DEFER MANUAL NO-VERDICT; do
    d="$TMP/m-$label"
    scratch "$d"
    python3 - "$d/evidence/dogfood/models/0.69.1/lambda.json" "$label" <<'PY'
import json, sys
p, label = sys.argv[1], sys.argv[2]; d = json.load(open(p))
rows = [r for r in d["rungs"] if r.get("id") == "qwen35-4b-q4km"]
assert len(rows) == 1, "mutation anchor missing"
rows[0]["verdict"] = label; json.dump(d, open(p, "w"), indent=2)
PY
    probe "$d" "M $label on qwen35-4b-q4km@lambda" RED "qwen35-4b-q4km@lambda"
done

# M6 — delete one required V* row → exit 1 naming the cell (absent is NotRun, never Fail)
scratch "$TMP/m6"
python3 - "$TMP/m6/evidence/dogfood/models/0.69.1/gx10.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
n = len(d["rungs"]); d["rungs"] = [r for r in d["rungs"] if r.get("id") != "qwen3-1.7b-q4km"]
assert len(d["rungs"]) == n - 1, "mutation anchor missing"
json.dump(d, open(p, "w"), indent=2)
PY
probe "$TMP/m6" "M6 qwen3-1.7b-q4km row deleted on gx10" RED "qwen3-1.7b-q4km@gx10"

# M7 — a Fail cell is ADMITTED: flip one V* row to green: false → capability-cells still has no NotRun
scratch "$TMP/m7"
python3 - "$TMP/m7/evidence/dogfood/models/0.69.1/lambda.json" <<'PY'
import json, sys
p = sys.argv[1]; d = json.load(open(p))
rows = [r for r in d["rungs"] if r.get("id") == "qwen35-9b-q4km"]
assert len(rows) == 1, "mutation anchor missing"
rows[0]["green"] = False; json.dump(d, open(p, "w"), indent=2)
PY
(cd "$TMP/m7" && "$PV" lint contracts/ --gate shapes --format json) > "$TMP/m7.json" 2>/dev/null || true
if jq -e '(.capability_cells.not_run|length)==0 and .capability_cells.domain != null' "$TMP/m7.json" > /dev/null; then
    printf 'ok   %-44s Fail admitted by capability-cells (not_run empty)\n' "M7 qwen35-9b-q4km@lambda green: false"
else
    printf 'FAIL %-44s a Fail cell was reported NotRun\n' "M7 qwen35-9b-q4km@lambda green: false"
    fails=$((fails + 1))
fi

if [ "$fails" -ne 0 ]; then
    echo "check_ont_4c5_probe: $fails case(s) disagreed"
    exit 1
fi
echo "check_ont_4c5_probe: probe green on this tree; every mutation RED"
