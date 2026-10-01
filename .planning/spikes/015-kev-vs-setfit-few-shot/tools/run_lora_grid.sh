#!/opt/homebrew/bin/bash
# Variant (d): LoRA + head fine-tuned from the released Kev-0.8B on the SetFit selections, two recipes declared
# before any score was read. Run from vendor/kev. Each cell: train, score test, then delete the adapter (disk).
set -uo pipefail
R=../../runs
declare -A RECIPE=([r1]="--epochs 2 --accum 8 --lr 2e-5" [r2]="--epochs 4 --accum 1 --lr 5e-5")
for seed in 13 17 23; do for k in 8 16 64; do for r in r1 r2; do
  out="$R/lora-0.8b-$r-s$k-seed$seed"; [ -f "$out-test.json" ] && continue
  s=$(date +%s)
  uv run python -m kev.train --data "$R/shots/s$k-seed$seed.jsonl" --base Qwen/Qwen3.5-0.8B-Base \
    --base_revision dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 --init_from jaredpalmer/kev-0.8b ${RECIPE[$r]} \
    --batch 1 --device mps --seed "$seed" --out "$out" > "$out.train.log" 2>&1 || { echo "TRAIN-FAIL $out"; continue; }
  e=$(date +%s)
  uv run python ../../tools/kev_eval.py --run "$out" --task stance-abortion --split test --out "$out-test.npz" > "$out.eval.log" 2>&1 || echo "EVAL-FAIL $out"
  python3 - "$out-test.json" "$((e-s))" <<'PY'
import json,sys; m=json.load(open(sys.argv[1])); m["train_wall_s"]=int(sys.argv[2]); json.dump(m,open(sys.argv[1],"w"),indent=1)
PY
  rm -rf "$out"; echo "done $out train=$((e-s))s"
done; done; done
echo LORA-DONE
