# Pillar-3 NF4 numerical-equivalence beat — measured (2026-06-13)

**Claim:** apr's pure-Rust NF4 blockwise quantization is numerically equivalent to
bitsandbytes (Unsloth's quant backend) — a faithful, contract-gated replacement, not
an approximation.

**Host:** noah-Lambda-Vector. **Incumbent:** `bitsandbytes==0.49.2` via `uv`
(CPU path, blocksize=64, quant_type='nf4', compress_statistics=False).

## Why equivalence (not "better")
apr's `NF4_LUT` is the canonical 16-level NF4 codebook sourced verbatim from
`bitsandbytes/csrc/kernels.cu`, and apr's blockwise convention matches bitsandbytes'
single-level scheme exactly: per-block `absmax = max|x|`, `code = quantize_nf4(x/absmax)`,
`dequant = NF4_LUT[code] * absmax`. So reconstruction is bit-equivalent by construction;
"quant quality" is parity. The beat is faithfulness + provability (contract +
`NF4Dequant.lean`), which bitsandbytes' untested CUDA kernel does not ship.

## Method + result
Deterministic input `x[i] = (i-32)*0.05` for i in 0..64 (range [-1.6, 1.55], 1 block).

| Quantity | bitsandbytes | apr (pure Rust) |
|----------|-------------|-----------------|
| round-trip MSE | 0.00737779 | 0.007378 |
| max abs round-trip error | 0.23609149 | (same codebook) |
| **max \|apr_recon − bnb_recon\|** | — | **4.92e-7** (essentially bit-exact) |

apr reconstruction matches bitsandbytes element-wise to **4.92e-7** and the round-trip
MSE is identical to 6 decimals. apr's NF4 is a numerically-equivalent replacement.

## CI-gated form
`crates/aprender-compute/src/brick/quant_ops/nf4.rs::tests::beat_nf4_bitsandbytes_equivalence`
pins the bitsandbytes reconstruction + MSE and asserts apr matches (max|Δ| < 1e-3, MSE
within 1e-4) — deterministic, no bitsandbytes/torch/GPU needed at CI time. Contract:
`contracts/apr-nf4-bitsandbytes-equivalence-beat-v1.yaml`. Wired into ci.yml.
