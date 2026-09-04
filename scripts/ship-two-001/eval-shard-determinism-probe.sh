#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# SHIP-TWO-001 — FALSIFY-SHARD-003 determinism probe
# ─────────────────────────────────────────────────────────────
# Contract: contracts/eval-sharding-v1.yaml (FALSIFY-SHARD-003)
# Spec:     docs/specifications/aprender-train/ship-two-models-spec.md §12.6
# AC-EX-007 cannot discharge until this probe either PASSES (byte-identical
# completions across hosts at temp=0) or the contract is amended to record
# the tolerated non-determinism with a looser gate.
#
# Procedure:
#   1. For each task in the probe set, emit the HumanEval prompt as JSONL.
#   2. On each host, run `apr run --batch-jsonl` at temperature=0.0.
#   3. Diff completions task-by-task between host_A and host_B.
#   4. PASS if ∀ tasks: byte-identical. FAIL otherwise. Report per-task delta.
#
# Usage:
#   bash scripts/ship-two-001/eval-shard-determinism-probe.sh \
#       --hosts yoga,gx10 \
#       --model /home/noah/.cache/pacha/models/<hash>.gguf \
#       --probe-tasks 0-15
#
#   Optional env overrides:
#     APR_BIN            orchestrator apr binary (default: apr on PATH)
#     REMOTE_APR_BIN     remote apr binary path (default: 'apr' on remote PATH)
#     TOKENIZER          tokenizer.json path (auto-derived if sibling exists)
#     MAX_TOKENS         default 512
#     REMOTE_WORKDIR     default /tmp/apr-shard-probe
#     LOCALHOST_ALIAS    default yoga (executed without ssh)
#     HUMANEVAL_JSONL    path to HumanEval problems.jsonl (auto-located)
#     DRY_RUN            1 = print plan + exit
#
# Exit codes:
#   0 — PASS (byte-identical across hosts)
#   1 — FAIL (at least one byte-differing task_id)
#   2 — infrastructure error (host unreachable, benchmark missing, etc.)
# ─────────────────────────────────────────────────────────────

set -euo pipefail

# Reproducibility escape hatch (bashrs DET002): SOURCE_DATE_EPOCH, when a
# caller sets it, pins the clock this probe stamps its evidence filenames
# with. Unset -- the normal case for a live probe run -- this falls through
# to the real wall clock, so evidence filenames are unchanged from before.
_stamp_compact() { date -u -d "@${SOURCE_DATE_EPOCH:-$(printf '%(%s)T' -1)}" +%Y%m%d_%H%M%S; }

HOSTS=""
MODEL=""
PROBE_RANGE="0-15"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --hosts)        HOSTS="$2"; shift 2 ;;
        --model)        MODEL="$2"; shift 2 ;;
        --probe-tasks)  PROBE_RANGE="$2"; shift 2 ;;
        -h|--help)
            sed -n '1,40p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

[[ -n "$HOSTS" ]] || { echo "ERROR: --hosts required" >&2; exit 2; }
[[ -n "$MODEL" ]] || { echo "ERROR: --model required" >&2; exit 2; }
[[ -f "$MODEL" ]] || { echo "ERROR: model not found: $MODEL" >&2; exit 2; }

MAX_TOKENS="${MAX_TOKENS:-512}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/tmp/apr-shard-probe}"
LOCALHOST_ALIAS="${LOCALHOST_ALIAS:-yoga}"
APR_BIN="${APR_BIN:-apr}"
REMOTE_APR_BIN="${REMOTE_APR_BIN:-apr}"
DRY_RUN="${DRY_RUN:-0}"

# Locate HumanEval problems.jsonl (same heuristic as eval-shard.sh)
if [[ -z "${HUMANEVAL_JSONL:-}" ]]; then
    for cand in \
        data/benchmarks/humaneval.jsonl \
        "${APR_LEADERBOARD_ROOT:-$HOME/src/apr-leaderboard}/data/benchmarks/humaneval.jsonl" \
        evidence/ship-two-001/humaneval/problems.jsonl ; do
        if [[ -f "$cand" ]]; then HUMANEVAL_JSONL="$cand"; break; fi
    done
fi
[[ -n "${HUMANEVAL_JSONL:-}" && -f "$HUMANEVAL_JSONL" ]] \
    || { echo "ERROR: could not locate HumanEval problems.jsonl; set HUMANEVAL_JSONL=" >&2; exit 2; }

# Parse hosts
IFS=',' read -r -a HOST_ARR <<< "$HOSTS"
[[ "${#HOST_ARR[@]}" -ge 2 ]] || { echo "ERROR: need ≥2 hosts for probe; got: $HOSTS" >&2; exit 2; }
HOST_A="${HOST_ARR[0]}"
HOST_B="${HOST_ARR[1]}"

# Parse probe range
if [[ "$PROBE_RANGE" == *-* ]]; then
    LO="${PROBE_RANGE%-*}"
    HI="${PROBE_RANGE#*-}"
else
    LO="$PROBE_RANGE"; HI="$PROBE_RANGE"
fi
[[ "$LO" =~ ^[0-9]+$ && "$HI" =~ ^[0-9]+$ && "$LO" -le "$HI" ]] \
    || { echo "ERROR: --probe-tasks must be N or N-M, got: $PROBE_RANGE" >&2; exit 2; }

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

EVIDENCE_DIR="evidence/ship-two-001/shard-003-determinism"
mkdir -p "$EVIDENCE_DIR"
STAMP="$(_stamp_compact)"
PROBE_LOG="$EVIDENCE_DIR/probe_${STAMP}.log"
PROBE_JSON="$EVIDENCE_DIR/probe_${STAMP}.json"

echo "[SHARD-003] probe $HOST_A vs $HOST_B, tasks $LO..$HI, temp=0.0" | tee "$PROBE_LOG"
echo "[SHARD-003] model: $MODEL" | tee -a "$PROBE_LOG"
echo "[SHARD-003] humaneval: $HUMANEVAL_JSONL" | tee -a "$PROBE_LOG"

# Build probe JSONL (one line per task_id with prompt field)
PROBE_JSONL="$TMPDIR/probe_shard.jsonl"
python3 -c "
import json, sys
lo, hi = $LO, $HI
with open('$HUMANEVAL_JSONL') as f, open('$PROBE_JSONL', 'w') as out:
    for line in f:
        if not line.strip(): continue
        obj = json.loads(line)
        tid = obj.get('task_id', '')
        m = tid.split('/')[-1]
        try: idx = int(m)
        except ValueError: continue
        if lo <= idx <= hi:
            out.write(json.dumps({'task_id': tid, 'prompt': obj['prompt']}) + '\n')
print(f'[SHARD-003] probe JSONL built: $PROBE_JSONL', file=sys.stderr)
"
LINE_COUNT=$(wc -l < "$PROBE_JSONL")
EXPECTED=$((HI - LO + 1))
[[ "$LINE_COUNT" -eq "$EXPECTED" ]] \
    || { echo "ERROR: probe JSONL lines=$LINE_COUNT expected=$EXPECTED" | tee -a "$PROBE_LOG"; exit 2; }
echo "[SHARD-003] probe tasks: $LINE_COUNT" | tee -a "$PROBE_LOG"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "[SHARD-003] DRY_RUN: exiting before dispatch" | tee -a "$PROBE_LOG"
    exit 0
fi

# Run one host (local or remote) — writes $TMPDIR/<alias>.jsonl
run_host() {
    local alias="$1"
    local out="$TMPDIR/${alias}.jsonl"
    local in="$PROBE_JSONL"
    echo "[SHARD-003] dispatch $alias" | tee -a "$PROBE_LOG"
    if [[ "$alias" == "$LOCALHOST_ALIAS" ]]; then
        "$APR_BIN" run "$MODEL" \
            --batch-jsonl "$in" \
            --max-tokens "$MAX_TOKENS" \
            --temperature 0.0 \
            --top-k 1 \
            > "$out" 2>> "$PROBE_LOG"
    else
        # Ensure remote workdir + stage inputs
        ssh "$alias" "mkdir -p $REMOTE_WORKDIR" >> "$PROBE_LOG" 2>&1
        rsync -azq "$in" "$alias:$REMOTE_WORKDIR/probe_shard.jsonl" >> "$PROBE_LOG" 2>&1
        # Model path on remote: assume same path (user's responsibility to pre-cache)
        ssh "$alias" "$REMOTE_APR_BIN run $MODEL \
            --batch-jsonl $REMOTE_WORKDIR/probe_shard.jsonl \
            --max-tokens $MAX_TOKENS \
            --temperature 0.0 \
            --top-k 1" \
            > "$out" 2>> "$PROBE_LOG"
    fi
    local n
    n=$(wc -l < "$out")
    echo "[SHARD-003] $alias returned $n lines" | tee -a "$PROBE_LOG"
    [[ "$n" -eq "$EXPECTED" ]] || { echo "ERROR: $alias returned $n lines, expected $EXPECTED" | tee -a "$PROBE_LOG"; return 2; }
}

run_host "$HOST_A" || exit $?
run_host "$HOST_B" || exit $?

# Diff per task_id — use Python for robust JSON comparison
VERDICT_FILE="$TMPDIR/verdict.txt"
python3 - "$TMPDIR/${HOST_A}.jsonl" "$TMPDIR/${HOST_B}.jsonl" "$HOST_A" "$HOST_B" \
        "$PROBE_JSON" "$VERDICT_FILE" << 'PYEOF'
import json, sys, os
a_file, b_file, a_name, b_name, out_json, verdict_file = sys.argv[1:7]

def load_jsonl(p):
    out = {}
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            obj = json.loads(line)
            tid = obj.get('task_id') or obj.get('id') or obj.get('prompt','')[:32]
            # completion field may be 'completion' | 'output' | 'generated'
            comp = obj.get('completion', obj.get('output', obj.get('generated','')))
            out[tid] = comp
    return out

a = load_jsonl(a_file)
b = load_jsonl(b_file)
ids = sorted(set(a) | set(b))
diffs = []
identical = 0
only_a = sorted(set(a) - set(b))
only_b = sorted(set(b) - set(a))
for tid in ids:
    ca = a.get(tid); cb = b.get(tid)
    if ca is None or cb is None: continue
    if ca == cb:
        identical += 1
    else:
        diffs.append({
            'task_id': tid,
            'a_len': len(ca), 'b_len': len(cb),
            'first_diff_byte': next((i for i,(x,y) in enumerate(zip(ca,cb)) if x != y), min(len(ca), len(cb))),
            'a_excerpt': ca[:120],
            'b_excerpt': cb[:120],
        })
verdict = 'PASS' if (len(diffs) == 0 and not only_a and not only_b) else 'FAIL'
total = len(ids)
report = {
    'falsification_id': 'FALSIFY-SHARD-003',
    'host_a': a_name, 'host_b': b_name,
    'total_tasks': total, 'identical': identical, 'divergent': len(diffs),
    'only_a': only_a, 'only_b': only_b,
    'verdict': verdict,
    'divergences': diffs,
}
with open(out_json, 'w') as f:
    json.dump(report, f, indent=2)
with open(verdict_file, 'w') as f:
    f.write(verdict)
print(f'[SHARD-003] {a_name} vs {b_name}: identical {identical}/{total}, divergent {len(diffs)}, verdict {verdict}')
if diffs:
    for d in diffs[:5]:
        print(f'  {d["task_id"]}: first_diff_byte={d["first_diff_byte"]} lens={d["a_len"]}/{d["b_len"]}')
PYEOF

VERDICT="$(cat "$VERDICT_FILE")"
echo "[SHARD-003] report: $PROBE_JSON"
echo "[SHARD-003] verdict: $VERDICT"

if [[ "$VERDICT" == "PASS" ]]; then
    exit 0
else
    exit 1
fi
