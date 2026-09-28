#!/usr/bin/env bash
# A/B: initcheck with FP8 pad-zero + skip-warmup (arm A) vs FP8_PREFILL=0 (arm B)
export TMPDIR=/mnt/nvme-raid0/tmp
R=/mnt/nvme-raid0/tmp/6b-ktest05/lambda
M=/home/noah/models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf
P="What is 2+2? Answer with one number."
for arm in padzero nofp8 padzero_only skipwarm_only; do
  case $arm in
    padzero) E=(APR_FP8_PAD_ZERO=1 APR_SKIP_FP8_WARMUP=1);;
    nofp8) E=(FP8_PREFILL=0 FP8_DECODE=0);;
    padzero_only) E=(APR_FP8_PAD_ZERO=1);;
    skipwarm_only) E=(APR_SKIP_FP8_WARMUP=1);;
  esac
  env BATCHED_PREFILL=1 "${E[@]}" flock /tmp/apr-gpu.lock timeout 3000 /usr/local/cuda/bin/compute-sanitizer --tool initcheck --print-limit 0 --show-backtrace no --log-file $R/san4/$arm.log $R/apr-pinned run $M --prompt "$P" --max-tokens 2 --temperature 0 > $R/san4/$arm.out 2> $R/san4/$arm.err
  echo "$arm rc=$? $(grep -h 'ERROR SUMMARY' $R/san4/$arm.log) skipped=$(grep -ho '[0-9]* errors were skipped' $R/san4/$arm.log) strides=$(grep -o 'Address 0x[0-9a-f]*' $R/san4/$arm.log | wc -l)" >> $R/san4/summary.txt
done
