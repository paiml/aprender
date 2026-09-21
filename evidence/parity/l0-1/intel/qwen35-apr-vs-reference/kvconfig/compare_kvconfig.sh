#!/usr/bin/env bash
# PMAT-3091 kvconfig comparisons (lambda), committed comparators only.
# refs: A = committed per-token (/tmp/emul3091/ref, re-hashed), B/C = intel kvconfig runs copied to /tmp/kv3091/ref.
# subjects: apr OFF / ON = /tmp/emul3091/runs (EMULATION.md), plus ON-on-intel orig/p4 cmp vs the lambda ON runs.
set -uo pipefail
EV="${EV:-/mnt/nvme-raid0/agent-wt/layer-3091/evidence/parity/l0-1/intel/qwen35-apr-vs-reference}"
E=/tmp/emul3091; W=/tmp/kv3091; O="$W/cmp"; LW="$EV/layerwise"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"; GGUF_PY="$HOME/src/llama.cpp-d1d3c3396/gguf-py"; EMBD=/tmp/layer3091/apr
mkdir -p "$O"
echo "host=$(hostname) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
sha256sum "$EV/compare_raw_logits.py" "$EV/emulation/logits_gap.py" "$LW/compare_layerwise.py" "$LW/layer_steps.py" "$MODEL" \
  "$E"/ref/*-per-token.bin "$W"/ref/*.bin "$E"/runs/off-{orig,p1,p2,p3,p4}.bin "$E"/runs/on-{orig,p1,p2,p3,p4}.bin "$E"/runs/on-dump-{p4,orig}.bin
echo "--- ON intel vs ON lambda"
for p in orig p4; do cmp "$W/ref/apr-on-$p-intel.bin" "$E/runs/on-$p.bin"; echo "apr-on-$p intel vs lambda cmp rc=$?"; done
echo "--- logits"
ref() { case "$1" in A) echo "$E/ref/$2-per-token.bin";; *) echo "$W/ref/$1-$2.bin";; esac; }
for p in orig p1 p2 p3 p4; do for c in A B C; do
  for m in off on; do
    timeout 600 python3 "$EV/compare_raw_logits.py" "$(ref $c $p)" "$E/runs/$m-$p.bin" --json "$O/logits-$p-$m-vs-$c.json" < /dev/null > "$O/logits-$p-$m-vs-$c.tsv"
    echo "compare_raw_logits $p $m vs $c rc=$?"
  done
  timeout 600 python3 "$EV/emulation/logits_gap.py" "$p" "$(ref $c $p)" "$E/runs/off-$p.bin" "on=$E/runs/on-$p.bin" < /dev/null > "$O/gap-$p-vs-$c.tsv"
  echo "logits_gap $p vs $c rc=$?"
done; done
echo "--- sub-layer curves vs config C dumps"
for m in off on; do
  timeout 900 python3 "$LW/compare_layerwise.py" p4 "$W/ref/C-sub-p4" "$W/ref/C-p4.bin" "$EMBD/embd-p4" "$E/runs/$m-dump-p4.bin" 0,1,2,3 "$LW/layer_types.tsv" "$MODEL" "$GGUF_PY" "$E/runs/$m-dump-p4" < /dev/null > "$O/sublayer_p4_${m}_vs_C.tsv" 2> "$O/sublayer_p4_${m}_vs_C.selfcheck.txt"
  echo "compare_layerwise p4 $m vs C rc=$?"
  timeout 900 python3 "$LW/compare_layerwise.py" orig "$W/ref/C-sub-orig" "$W/ref/C-orig.bin" "$EMBD/embd-p0" "$E/runs/$m-dump-orig.bin" 4,28 "$LW/layer_types.tsv" "$MODEL" "$GGUF_PY" "$E/runs/$m-dump-orig" < /dev/null > "$O/sublayer_orig_${m}_vs_C.tsv" 2> "$O/sublayer_orig_${m}_vs_C.selfcheck.txt"
  echo "compare_layerwise orig $m vs C rc=$?"
  timeout 600 python3 "$LW/layer_steps.py" --ref p4:1 "$O/sublayer_p4_${m}_vs_C.tsv" "$O/sublayer_orig_${m}_vs_C.tsv" < /dev/null > "$O/layer_steps_${m}_vs_C.tsv"
  echo "layer_steps $m vs C rc=$?"
done
timeout 300 python3 "$EV/kvconfig/tables.py" "$O" /tmp/emul3091/cmp/sublayer_p4_on.tsv < /dev/null > "$O/tables.md"; echo "tables rc=$?"
sha256sum "$O"/*
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
