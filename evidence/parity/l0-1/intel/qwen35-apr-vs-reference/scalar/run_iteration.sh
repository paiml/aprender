#!/usr/bin/env bash
# PMAT-3091 scalar iteration (lambda): after one SSE2 op port under APR_EMULATE_GGML_VECDOT=scalar —
#  1. switch-OFF invariance (orig f6f79264…, p4 92f1b54d…), 2. `=1` unchanged vs the committed on-{orig,p4} subjects,
#  3. scalar orig/p4 logits + sub-layer dumps (p4 pos 0-3, orig pos 4,28), 4. walk_points.py vs the SCALAR llama config-C
#  dumps, 5. logits compare (compare_raw_logits.py). usage: run_iteration.sh <iter-label>. Outputs under scalar/iter/<label>/.
set -uo pipefail
IT="${1:?iteration label}"
BIN="${BIN:-/mnt/nvme-raid0/targets/obs-3091/release/examples/qwen35_layer_obs}"
EV=/mnt/nvme-raid0/agent-wt/layer-3091/evidence/parity/l0-1/intel/qwen35-apr-vs-reference
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"; W=/tmp/scalar3091; R="$W/ref"; RUN="${RUNROOT:-$W/iter}/$IT"; O="$EV/scalar/iter/$IT"; ENVV=APR_EMULATE_GGML_VECDOT
declare -A IDS=([orig]="$EV/prompt_token_ids.txt" [p4]="$EV/variation/prompt-4.ids")
declare -A SUBJECT=([orig]=f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252 [p4]=92f1b54d504a53f758a56911c4779ce1900207bd89c78d58597e558ebc6d3ee9)
mkdir -p "$RUN" "$O"
echo "iter=$IT obs_code=$(git -C /mnt/nvme-raid0/agent-wt/obs-3091 rev-parse HEAD) dirty_tracked=$(git -C /mnt/nvme-raid0/agent-wt/obs-3091 status --porcelain --untracked-files=no | wc -l) host=$(hostname) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
sha256sum "$BIN"
run() { # label switch mode ids extra...
  local label="$1" sw="$2"; shift 2
  if [ "$sw" = "-" ]; then env -u "$ENVV" timeout 1800 "$BIN" "$1" "$MODEL" "$2" "$RUN/$label.bin" "${@:3}" < /dev/null > "$RUN/$label.log" 2>&1
  else env "$ENVV=$sw" timeout 1800 "$BIN" "$1" "$MODEL" "$2" "$RUN/$label.bin" "${@:3}" < /dev/null > "$RUN/$label.log" 2>&1; fi
  echo "run $label switch=$sw rc=$? $(grep -E 'mask=' "$RUN/$label.log" | sed 's/.*mask=/mask=/') sha256=$(sha256sum "$RUN/$label.bin" 2>/dev/null | cut -c1-16)"
}
for p in orig p4; do run "off-$p" - noop "${IDS[$p]}"; got=$(sha256sum "$RUN/off-$p.bin" | cut -d' ' -f1); echo "invariance_off $p sha_equal=$([ "$got" = "${SUBJECT[$p]}" ] && echo true || echo false)"; done
for p in orig p4; do run "on1-$p" 1 noop "${IDS[$p]}"; cmp -s "$RUN/on1-$p.bin" /tmp/emul3091/runs/on-$p.bin; echo "eq1_unchanged $p cmp_rc=$? (0 = byte-identical to the committed =1 subject)"; done
rm -rf "${RUN:?}/scalar-dump-p4" "${RUN:?}/scalar-dump-orig"
run scalar-dump-p4 scalar dump "${IDS[p4]}" 0,1,2,3 "$RUN/scalar-dump-p4"
run scalar-dump-orig scalar dump "${IDS[orig]}" 4,28 "$RUN/scalar-dump-orig"
for p in p4 orig; do
  timeout 900 python3 "$EV/scalar/walk_points.py" "$p" "$R/SC-sub-$p" "$RUN/scalar-dump-$p" < /dev/null > "$O/walk_$p.tsv" 2> "$O/walk_$p.summary.txt"; echo "walk $p rc=$?"; cat "$O/walk_$p.summary.txt"
  timeout 600 python3 "$EV/compare_raw_logits.py" "$R/SC-$p.bin" "$RUN/scalar-dump-$p.bin" --json "$O/logits-$p-scalar-vs-SC.json" < /dev/null > "$O/logits-$p-scalar-vs-SC.tsv" 2>&1; echo "compare_raw_logits $p rc=$?"
done
for p in orig p4; do echo "argmax $p positions=$(tail -n +2 "$O/logits-$p-scalar-vs-SC.tsv" | awk -F'\t' '$1 ~ /^[0-9]+$/' | wc -l) mismatches=$(tail -n +2 "$O/logits-$p-scalar-vs-SC.tsv" | awk -F'\t' '$1 ~ /^[0-9]+$/ && $6 == "False"' | wc -l) min_cos=$(tail -n +2 "$O/logits-$p-scalar-vs-SC.tsv" | awk -F'\t' '$1 ~ /^[0-9]+$/ {print $3}' | sort -g | head -1)"; done
sha256sum "$O"/*
echo "end_utc=$(date -u +%FT%TZ)"
