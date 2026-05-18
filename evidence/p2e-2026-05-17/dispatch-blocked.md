# P2-E dispatch — BLOCKED on cuda_training_available runtime gate

**Date:** 2026-05-17
**Ticket:** PMAT-690 P2-E (per evidence/p2c-2026-05-17/findings.md §112-118)

## Symptom

`apr pretrain --device cuda --init <imported.apr> ...` exits 1 with:

```
error: Validation failed: --device `cuda` requested but CUDA runtime is
not available on this host (contract gpu-training-backend-v1
GATE-GPUTRAIN-002: no silent CPU fallback). Rebuild with `--features
cuda` or pass `--device cpu` to opt in to the CPU path.
```

## Investigation

- Host: lambda-vector, NVIDIA RTX 4090 (24 GB), CUDA 12.8 driver 570.207
- `nvidia-smi` works
- `apr gpu` correctly reports the 4090
- Binary at `/mnt/nvme-raid0/targets/aprender/release/apr` has 490+
  occurrences of `cuda*` symbols (`strings`-grep)
- `cargo build -p apr-cli --bin apr --release --features cuda` exits 0
  but doesn't update the binary mtime (cache hit) — the existing binary
  has cuda symbols but `cuda_training_available()` still returns false
  at runtime

## Root cause (hypothesis)

`cuda_training_available()` in `crates/aprender-train/src/autograd/cuda_training.rs:259`
is gated by `#[cfg(feature = "cuda")]` on the `aprender-train` (entrenar)
crate's `cuda` feature. apr-cli's `cuda = ["inference", "realizar/cuda",
"entrenar/cuda"]` declares the dependency, but the binary that's currently
canonical at `/mnt/nvme-raid0/targets/aprender/release/apr` was apparently
built without `entrenar/cuda` activated. Force-rebuild attempts haven't
overwritten it because cargo's incremental cache decides "no source
changed".

## Recommended next action (new ticket)

PMAT-691 — investigate the apr-cli `--features cuda` propagation to the
aprender-train compile-time `cuda` feature. Probably needs one of:

1. A `cargo clean -p apr-cli -p aprender-train` followed by a clean
   release build with `--features cuda` (and verify with `strings apr |
   grep cuda_training_available`).
2. A `cuda-batch` feature definition fix if the `entrenar/cuda` indirect
   feature isn't being recognized.
3. A workspace-level `cuda` feature on the root facade `aprender` crate
   so `cargo build --release --features cuda` (no `-p`) builds the root
   `apr` binary with training cuda support.

Once unblocked, retry the P2-E dispatch with the recipe in
`dispatch-params.md`.

## Status

- P2-E params staged: `evidence/p2e-2026-05-17/dispatch-params.md`
- Run dir staged: `/mnt/nvme-raid0/runs/model-2-p2e-tuned-hp-20260517/`
- All inputs verified: dataset shards (qwen-v3), tokenizer (Qwen 151,936
  vocab), init APR (Qwen-0.5B), `--force-under-provisioned` bypass

P2-E is **scope-ready** but **dispatch-blocked** on the cuda-feature
propagation bug. Once PMAT-691 lands, the dispatch is a single shell
command away.
