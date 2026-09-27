#!/usr/bin/env bash
# EXT-27 KL lane: runs the named arms in order; skips an arm whose log exists.
set -uo pipefail
cd /mnt/nvme-raid0/tmp/ext-models/ext27 || exit 1
LP=/mnt/nvme-raid0/tmp/llama-d1d3-build/bin/llama-perplexity
for arm in "$@"; do
  mkdir "claim-$arm" 2>/dev/null || continue
  nice "$LP" -m "$arm.gguf" -f corpus.txt -c 512 -t 12 --kl-divergence --kl-divergence-base ref-logits.kld > "kl-$arm.log" 2>&1
  echo "$arm rc=$?" >> arms.status
done
