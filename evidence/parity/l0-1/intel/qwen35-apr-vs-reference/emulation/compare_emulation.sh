#!/usr/bin/env bash
# PMAT-3091 ggml vec_dot emulation: comparisons (lambda), over run_emulation.sh outputs and the committed comparators.
set -uo pipefail
EV="${EV:-/mnt/nvme-raid0/agent-wt/layer-3091/evidence/parity/l0-1/intel/qwen35-apr-vs-reference}"
W="${WORK:-/tmp/emul3091}"
REF="$W/ref"
LW="$EV/layerwise"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"
GGUF_PY="$HOME/src/llama.cpp-d1d3c3396/gguf-py"
LL="${LLAMA_SUB:-/tmp/obs3091/llama}"
EMBD="${APR_EMBD:-/tmp/layer3091/apr}"
O="$W/cmp"
mkdir -p "$O"
echo "host=$(hostname) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
sha256sum "$EV/compare_raw_logits.py" "$LW/compare_layerwise.py" "$LW/layer_steps.py" "$LW/kernel_isolation.py" "$EV/emulation/logits_gap.py" "$REF"/*.bin "$LL/sub-p4.bin" "$LL/sub-p0.bin"

echo "== logits vs llama per-token"
for p in orig p1 p2 p3 p4; do
  for v in off on only-Q4_K only-Q5_K only-Q6_K only-Q8_0; do
    sub="$W/runs/$v-$p.bin"
    timeout 600 python3 "$EV/compare_raw_logits.py" "$REF/$p-per-token.bin" "$sub" --json "$O/logits-$p-$v.json" > "$O/logits-$p-$v.tsv" < /dev/null
    echo "compare_raw_logits $p $v rc=$?"
  done
  pos=""; [ "$p" = p4 ] && pos=0,1,2,3; [ "$p" = orig ] && pos=4,28
  POSITIONS_ENV="$pos" timeout 600 python3 "$EV/emulation/logits_gap.py" "$p" "$REF/$p-per-token.bin" "$W/runs/off-$p.bin" \
    "on=$W/runs/on-$p.bin" "only-Q4_K=$W/runs/only-Q4_K-$p.bin" "only-Q5_K=$W/runs/only-Q5_K-$p.bin" \
    "only-Q6_K=$W/runs/only-Q6_K-$p.bin" "only-Q8_0=$W/runs/only-Q8_0-$p.bin" < /dev/null > "$O/gap-$p.tsv"
  echo "logits_gap $p rc=$?"
done

echo "== sub-layer curves vs llama (OFF and ON dumps)"
for v in off on; do
  timeout 900 python3 "$LW/compare_layerwise.py" p4 "$LL/sub-p4" "$LL/sub-p4.bin" "$EMBD/embd-p4" "$W/runs/$v-dump-p4.bin" 0,1,2,3 "$LW/layer_types.tsv" "$MODEL" "$GGUF_PY" "$W/runs/$v-dump-p4" < /dev/null > "$O/sublayer_p4_$v.tsv" 2> "$O/sublayer_p4_$v.selfcheck.txt"
  echo "compare_layerwise p4 $v rc=$?"
  timeout 900 python3 "$LW/compare_layerwise.py" orig "$LL/sub-p0" "$LL/sub-p0.bin" "$EMBD/embd-p0" "$W/runs/$v-dump-orig.bin" 4,28 "$LW/layer_types.tsv" "$MODEL" "$GGUF_PY" "$W/runs/$v-dump-orig" < /dev/null > "$O/sublayer_orig_$v.tsv" 2> "$O/sublayer_orig_$v.selfcheck.txt"
  echo "compare_layerwise orig $v rc=$?"
  timeout 600 python3 "$LW/layer_steps.py" --ref p4:1 "$O/sublayer_p4_$v.tsv" "$O/sublayer_orig_$v.tsv" < /dev/null > "$O/layer_steps_$v.tsv"
  echo "layer_steps $v rc=$?"
done
cmp "$O/sublayer_p4_off.tsv" "$LW/sublayer_p4.tsv"; echo "sublayer_p4_off vs committed sublayer_p4.tsv cmp rc=$?"
cmp "$O/sublayer_orig_off.tsv" "$LW/sublayer_orig.tsv"; echo "sublayer_orig_off vs committed sublayer_orig.tsv cmp rc=$?"

echo "== kernel isolation, all 24 layers, OFF and ON"
for v in off on; do
  timeout 1800 python3 "$LW/kernel_isolation.py" "$MODEL" "$GGUF_PY" "$LW/layer_types.tsv" 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23 \
    "p4:$LL/sub-p4:$W/runs/$v-dump-p4:0,1,2,3" "orig:$LL/sub-p0:$W/runs/$v-dump-orig:4,28" < /dev/null > "$O/kernel_isolation_$v.tsv"
  echo "kernel_isolation $v rc=$?"
done
sha256sum "$O"/*
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
