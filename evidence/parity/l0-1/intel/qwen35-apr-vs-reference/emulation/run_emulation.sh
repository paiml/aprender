#!/usr/bin/env bash
# PMAT-3091 ggml vec_dot emulation: apr runs (lambda). One job at a time, timeout + </dev/null on every binary.
# 1. INVARIANCE (switch OFF): noop logits for orig/p1-p4 must be byte-equal to the recorded subject logits;
#    orig/p4 are the brief's gate (exit 3 if not). OFF observer dumps are also diffed against the 0543266ee dumps.
# 2. MECHANISM + MEASUREMENT (switch ON = 1): noop logits for orig/p1-p4, observer dumps at p4 0-3 and orig 4,28.
# 3. ATTRIBUTION: switch = one qtype at a time (Q4_K, Q5_K, Q6_K, Q8_0), noop logits for orig/p1-p4.
set -uo pipefail
BIN="${BIN:-/mnt/nvme-raid0/targets/obs-3091/release/examples/qwen35_layer_obs}"
MODEL="$HOME/models/Qwen3.5-0.8B-Q4_K_M.gguf"
EV="${EV:-/mnt/nvme-raid0/agent-wt/layer-3091/evidence/parity/l0-1/intel/qwen35-apr-vs-reference}"
W="${WORK:-/tmp/emul3091}"
OLD_OBS="${OLD_OBS:-/tmp/obs3091/apr}"
ENVV=APR_EMULATE_GGML_VECDOT
declare -A IDS=([orig]="$EV/prompt_token_ids.txt" [p1]="$EV/variation/prompt-1.ids" [p2]="$EV/variation/prompt-2.ids" [p3]="$EV/variation/prompt-3.ids" [p4]="$EV/variation/prompt-4.ids")
declare -A SUBJECT=([orig]=f6f792649014dcef6eec1afaf96852a293cc95c6cfceb1c88d62e16ee36c0252 [p1]=b9815d9be8b14b255db54d22821ff63a306d7dd8ae94d7bdfe4c93328376b7f8 [p2]=d5ffa2b73bc304b93c8461b103e66092c3f8a1d0850d596d3ee9b5006179a7fa [p3]=b2a984d0bacc24c1e5963f16256194c172fa99dd90481f6b364fc22a206e3e91 [p4]=92f1b54d504a53f758a56911c4779ce1900207bd89c78d58597e558ebc6d3ee9)
PROMPTS=(orig p1 p2 p3 p4)
mkdir -p "$W/runs"
echo "obs_code=$(git -C /mnt/nvme-raid0/agent-wt/obs-3091 rev-parse HEAD) dirty_files=$(git -C /mnt/nvme-raid0/agent-wt/obs-3091 status --porcelain | wc -l) host=$(hostname) nproc=$(nproc) start_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
git -C /mnt/nvme-raid0/agent-wt/obs-3091 status --porcelain
sha256sum "$BIN" "$MODEL" "${IDS[@]}"

# run LABEL SWITCH(-=unset) MODE ARGS...
run() {
  local label="$1" sw="$2"; shift 2
  local out="$W/runs/$label.bin"
  echo "run $label switch=$sw start_utc=$(date -u +%H:%M:%SZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
  if [ "$sw" = "-" ]; then
    env -u "$ENVV" timeout 1800 "$BIN" "$1" "$MODEL" "$2" "$out" "${@:3}" < /dev/null > "$W/runs/$label.log" 2>&1
  else
    env "$ENVV=$sw" timeout 1800 "$BIN" "$1" "$MODEL" "$2" "$out" "${@:3}" < /dev/null > "$W/runs/$label.log" 2>&1
  fi
  local rc=$?
  echo "run $label rc=$rc end_utc=$(date -u +%H:%M:%SZ)"
  grep -E '^qwen35_layer_obs:' "$W/runs/$label.log" | sed 's/^/  /'
  sha256sum "$out" 2>/dev/null | sed 's/^/  /'
  return $rc
}

echo "== 1. invariance (switch OFF)"
gate=0
for p in "${PROMPTS[@]}"; do
  run "off-$p" - noop "${IDS[$p]}" || gate=1
  got=$(sha256sum "$W/runs/off-$p.bin" | cut -d' ' -f1)
  if [ "$got" = "${SUBJECT[$p]}" ]; then echo "invariance $p sha_equal=true"; else echo "invariance $p sha_equal=false got=$got want=${SUBJECT[$p]}"; [ "$p" = orig ] || [ "$p" = p4 ] && gate=1; fi
done
if [ "$gate" != 0 ]; then echo "STOP: switch-OFF invariance failed"; exit 3; fi
rm -rf "$W/runs/off-dump-p4" "$W/runs/off-dump-orig" "$W/runs/on-dump-p4" "$W/runs/on-dump-orig"
run off-dump-p4 - dump "${IDS[p4]}" 0,1,2,3 "$W/runs/off-dump-p4"
run off-dump-orig - dump "${IDS[orig]}" 4,28 "$W/runs/off-dump-orig"
cmp "$W/runs/off-dump-p4.bin" "$W/runs/off-p4.bin"; echo "off-dump-p4 logits vs off-p4 cmp rc=$?"
cmp "$W/runs/off-dump-orig.bin" "$W/runs/off-orig.bin"; echo "off-dump-orig logits vs off-orig cmp rc=$?"
diff -rq "$W/runs/off-dump-p4" "$OLD_OBS/obs-p4"; echo "off-dump-p4 vs 0543266ee obs-p4 (every .f32 + manifest) diff rc=$?"
diff -rq "$W/runs/off-dump-orig" "$OLD_OBS/obs-p0"; echo "off-dump-orig vs 0543266ee obs-p0 (every .f32 + manifest) diff rc=$?"

echo "== 2. switch ON (all ported qtypes)"
for p in "${PROMPTS[@]}"; do run "on-$p" 1 noop "${IDS[$p]}"; done
run on-dump-p4 1 dump "${IDS[p4]}" 0,1,2,3 "$W/runs/on-dump-p4"
run on-dump-orig 1 dump "${IDS[orig]}" 4,28 "$W/runs/on-dump-orig"
cmp "$W/runs/on-dump-p4.bin" "$W/runs/on-p4.bin"; echo "on-dump-p4 logits vs on-p4 cmp rc=$?"
cmp "$W/runs/on-dump-orig.bin" "$W/runs/on-orig.bin"; echo "on-dump-orig logits vs on-orig cmp rc=$?"
for p in "${PROMPTS[@]}"; do cmp -s "$W/runs/on-$p.bin" "$W/runs/off-$p.bin"; echo "on-$p vs off-$p cmp rc=$? (1 = the switch changed the logits)"; done
for pos in 0 1 2 3; do cmp -s "$W/runs/on-dump-p4/pos$pos/linear_attn_out-0.f32" "$W/runs/off-dump-p4/pos$pos/linear_attn_out-0.f32"; echo "p4 pos$pos linear_attn_out-0 on vs off cmp rc=$?"; cmp -s "$W/runs/on-dump-p4/pos$pos/final_output-0.f32" "$W/runs/off-dump-p4/pos$pos/final_output-0.f32"; echo "p4 pos$pos final_output-0 on vs off cmp rc=$?"; done

echo "== 3. attribution (one qtype at a time)"
for q in Q4_K Q5_K Q6_K Q8_0; do for p in "${PROMPTS[@]}"; do run "only-$q-$p" "$q" noop "${IDS[$p]}"; done; done

echo "== 4. repeat ON (n=2) for orig/p4"
for p in orig p4; do run "on2-$p" 1 noop "${IDS[$p]}"; cmp "$W/runs/on2-$p.bin" "$W/runs/on-$p.bin"; echo "on2-$p vs on-$p cmp rc=$?"; done
echo "end_utc=$(date -u +%FT%TZ) load=$(cut -d' ' -f1-3 /proc/loadavg)"
