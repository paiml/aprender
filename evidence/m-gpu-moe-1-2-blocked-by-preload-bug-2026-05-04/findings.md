# M-GPU-MOE-1.2 heavy test blocked by `preload_weights_gpu` MoE-unaware bug

**Date**: 2026-05-04
**Host**: lambda-vector RTX 4090
**Apr binary**: `/mnt/nvme-raid0/targets/aprender/release/apr` (v0.31.2, post-#1485)
**Cached GGUF**: `/home/noah/.cache/pacha/models/2b88b180a790988f.gguf` (Qwen3-Coder-30B-A3B-Instruct-Q4_K_M, 17.3 GB)
**Test**: `crates/aprender-serve/tests/qwen3_moe_gpu_parity.rs::falsify_qw3_moe_gpu_parity_001_cosine_vs_cpu`
**Falsifier**: `FALSIFY-QW3-MOE-GPU-PARITY-001` per `qwen3-moe-forward-gpu-v1` v1.2.0

## Symptom

```
$ cargo test -p aprender-serve --test qwen3_moe_gpu_parity --features cuda \
    --release -- --include-ignored --nocapture

FALSIFY-QW3-MOE-GPU-PARITY-001: cosine vs CPU LAZY-FUSED-MATVEC
  gguf:    /home/noah/.cache/pacha/models/2b88b180a790988f.gguf
  prompt:  [785, 9217, 308]
[BOS-FALLBACK] No tokenizer.ggml.bos_token_id in GGUF — using
  architecture default for 'qwen3moe'
FALSIFY-QW3-MOE-GPU-PARITY-001: running CPU forward on 3 prompt tokens
  (this takes a few minutes)...
[GH-129] Early kernel preload: 49 modules compiled

thread 'falsify_qw3_moe_gpu_parity_001_cosine_vs_cpu' panicked at
  crates/aprender-serve/tests/qwen3_moe_gpu_parity.rs:144:10:

OwnedQuantizedModelCuda::new(model, 0) should succeed on RTX 4090:
  UnsupportedOperation { operation: "preload_weights_gpu",
    reason: "PAR-043: Failed to build indexed weights:
             Invalid launch config: Quantized weight 'blk.0.ffn_gate.weight' not cached" }
```

CPU forward succeeded (took ~20s for 3-token prompt). The CPU
LAZY-FUSED-MATVEC path is correct and produces logits. The failure
is at the **second** `OwnedQuantizedModel::from_mapped(...)` →
`OwnedQuantizedModelCuda::new(...)` chain — specifically inside
`preload_weights_gpu` → `executor.build_indexed_weights(...)` which
demands `blk.0.ffn_gate.weight` exists in the GPU weight cache.

## Root cause (5-whys)

1. **Why does the GPU forward fail to construct?**
   `OwnedQuantizedModelCuda::new` calls `preload_weights_gpu`, which
   calls `executor.build_indexed_weights` with `arch =
   self.model.config.constraints` (an `ArchConstraints`). The
   indexer attempts `get_qweight("blk.0.ffn_gate.weight")` and finds
   nothing.

2. **Why is `blk.0.ffn_gate.weight` missing from the GPU weight cache?**
   The `upload_layer_ffn` helper at
   `crates/aprender-serve/src/gguf/cuda/weights_preload_gpu.rs:348`
   uploads `blk.{i}.ffn_gate.weight` only `if let Some(ref gate) =
   layer.ffn_gate_weight`. For MoE layers, `from_gguf_for_moe` (per
   `crates/aprender-serve/src/gguf/transformer.rs:303-389`,
   specifically `load_quantized_layer_moe_skeleton`) sets
   `ffn_gate_weight: None` because MoE doesn't have a single
   per-layer gate tensor — it has 128 expert gates per layer
   (`blk.{i}.ffn_gate_exps.weight` etc).

3. **Why is `build_indexed_weights` unconditionally requiring
   `ffn_gate.weight`?**
   `crates/aprender-serve/src/cuda/executor/weights.rs:325-373`
   calls `get_qweight(&gate_name)?` (with `?` propagation) for every
   layer regardless of architecture. The fail-fast behaviour was
   designed to catch missing weights in dense models (LLaMA-style
   SwiGLU). It pre-dates MoE arch support in the wrapper type.

4. **Why doesn't M-GPU-MOE-1.1.2 (PR #1477) sidestep this?**
   PR #1477's `forward_qwen3_moe_cuda` body bypasses
   `indexed_layer_weights` for the FFN section (uses
   `moe_layers` parameter directly). But the wrapper construction
   itself still goes through `preload_weights_gpu` → 
   `build_indexed_weights` BEFORE the forward method is ever called.
   The wrapper can't be CONSTRUCTED for an MoE model on CUDA today.

5. **Why didn't unit tests catch this?**
   The lib-only `forward_qwen3_moe_cuda_stub_compiles_with_correct_signature`
   test (PR #1464) exercises only the function signature at compile
   time. The heavy `qwen3_moe_gpu_parity.rs` test (PR #1484) is
   `#[ignore]`d by default and requires both RTX 4090 hardware AND
   the cached 17.3 GB GGUF — neither available in default CI. This
   bug was only discoverable via `--include-ignored` on lambda-vector,
   which this evidence run is the first dogfood of.

## Where the fix needs to land

`crates/aprender-serve/src/cuda/executor/weights.rs:325-373`
(`build_indexed_weights`):

The minimum-viable fix is a 4th boolean parameter `is_moe: bool` (or
equivalent flag derivable from `&ArchConstraints`) that gates the
3 FFN-related lookups:

```rust
let (ffn_gate_ptr, ffn_gate_len) = if is_moe {
    (0u64, 0usize)  // MoE: per-expert weights live in moe_layers param
} else {
    get_qweight(&gate_name)?
};
let (ffn_up_ptr,   ffn_up_len)   = if is_moe { (0, 0) } else { get_qweight(&up_name)? };
let (ffn_down_ptr, ffn_down_len) = if is_moe { (0, 0) } else { get_qweight(&down_name)? };
```

Plus matching qtype resolution at lines 382-384 (use `0` sentinel
for MoE).

The MoE detection itself can be added either:
- as a new field on `ArchConstraints` (set to `true` for `qwen3_moe`
  per `arch-constraints-v1.yaml`), OR
- as an extra parameter passed by the caller (which knows from
  `self.model.moe_layers`)

The first option is cleaner because `build_indexed_weights` already
takes `&ArchConstraints` and the contract YAML is the right place
for arch-level facts.

## Test verification path

After the fix lands:

```
cargo test -p aprender-serve --test qwen3_moe_gpu_parity --features cuda \
    --release -- --include-ignored --nocapture
```

Expected: progresses past `OwnedQuantizedModelCuda::new`, runs the
GPU forward via `forward_qwen3_moe_cuda` (M-GPU-MOE-1.1.2, PR #1477
squash `dc6f94d3b`), computes cosine vs CPU reference, asserts ≥0.99.

If the GPU forward then fails at runtime (e.g., per-expert dispatch
diverges from CPU), that's the SECOND falsifier — different bug,
different fix scope.

## Status

- Bug confirmed live on lambda-vector RTX 4090, 2026-05-04.
- Bug class: **MoE-unaware GPU weight indexing**, pre-existing in
  CudaExecutor, exposed by M-GPU-MOE-1.1.2 + 1.2 cascade.
- Fix scope: small (~30 LOC in `weights.rs` + 1-2 callers + maybe
  arch-constraints-v1.yaml field).
- Severity: **R10-blocking** — without this fix, the M-GPU-MOE-1.1.2
  integration is unusable in production despite the PR #1477 SHIP.

## References

- Contract: `contracts/qwen3-moe-forward-gpu-v1.yaml` v1.2.0
- M-GPU-MOE-1.1.2 PR #1477 squash `dc6f94d3b`
- M-GPU-MOE-1.2 test PR #1484 squash `8cbb7b51e`
- Companion-spec milestones M50, M51, M52, M53, M54
- Companion-spec R10 risk row
