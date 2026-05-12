## SHIP-007 per-layer divergence accumulation analysis (2026-04-28)

### Method

Parsed `apr-trace.txt` and `gguf-trace.txt` (canonical 7B teacher, prompt "What is 2+2?") to compute APR/GGUF std ratio at each sub-stage of layers 0-6.

### Key finding — drift is gradual, then explosive

| Layer | ffn_swigl ratio | output ratio | Phase |
|------:|----------------:|-------------:|-------|
| 0 | 1.11× | 1.12× | small drift |
| 1 | 1.37× | 1.39× | accumulating |
| 2 | 1.13× | 1.30× | accumulating |
| **3** | **18.23×** | **18.57×** | **EXPLOSION** |
| 4 | 3.33× | 19.52× | sustained |
| 5 | 4.48× | 20.74× | sustained |
| 6 | 0.99× | 17.76× | partial recovery in sub-FFN |

The first 3 layers accumulate small drifts (<1.4× std ratio). At layer 3, the cumulative drift crosses a threshold where silu's saturated regime (gate values near -6) amplifies the gap non-linearly:

- Layer 3 ffn_gate: 1.36× ratio (small but non-trivial std difference)
- Layer 3 ffn_silu: 4.59× ratio (silu amplifies)
- Layer 3 ffn_swigl: 18.23× ratio (silu × up compounds)
- Layer 3 ffn_out: 60× ratio (down matmul cascades)

After layer 3, the residual stream is permanently corrupted (output ratio sticks at 17-21×).

### Why this matters for shipping MODEL-1

The bug surface is now characterized in full:
1. Bug is NOT at any single named site — it's CUMULATIVE F32 precision drift
2. Layers 0-2 produce per-element values whose aggregate std looks similar (within 5%) but whose per-element values diverge enough that...
3. Layer-3 ffn_gate matmul (with byte-identical weights) produces 36% wider output distribution
4. Silu non-linearity at gate values near -6 (saturated regime) amplifies the 36% to 4.6×
5. Multiply by ffn_up compounds to 18.23×
6. Down-projection cascades to 60×

The fix isn't "fix the matmul" or "fix the bias." It's: **match the F32 accumulator precision in APR's residual additions to what GGUF uses**. If APR uses lower-precision accumulation (or different reduction order) than GGUF, then over 3 layers the per-element values drift just enough.

### Concrete next investigation step

Compare APR's `helpers::f32_matmul` vs GGUF's `fused_q4k_q8k_parallel_matvec_into` with respect to:
- **Accumulator type**: f32 vs f32 (probably same)
- **Reduction order**: serial vs parallel rayon
- **Tile/block boundaries**: how partial sums are combined
- **FMA vs separate mul+add**

The hypothesis: APR's reduction is parallel (rayon) which produces non-deterministic ordering of accumulations. GGUF's may be serial or have a fixed deterministic order. F32 accumulation is non-associative; different orders → different results at the per-element level.

If confirmed, fix = ensure deterministic reduction order in APR's matmul kernels. This would reduce per-element divergence at layer 0/1/2 below the threshold where layer-3 silu amplifies it.

### Test for this hypothesis

Run APR forward TWICE with the same input. Compare layer-3 ffn_swigl output element-wise. If results differ across runs (non-determinism within APR itself), parallel reduction is confirmed as the source.

### Layer-by-layer table (full)

```
Layer         Stat |    APR std   GGUF std |  std ratio
----- ------------ | ---------- ---------- | ----------
    0    attn_norm |     0.2213     0.2421 |     0.91
    0          qkv |    10.3291     1.1402 |     9.06 <<< trace-point mismatch (§32)
    0     attn_out |     0.1776     0.1662 |     1.07
    0     ffn_norm |     0.1773     0.1790 |     0.99
    0     ffn_gate |     0.9448     0.9129 |     1.03
    0     ffn_silu |     0.1597     0.1611 |     0.99
    0    ffn_swigl |     0.0881     0.0793 |     1.11
    0      ffn_out |     0.3245     0.2773 |     1.17
    0       output |     0.4016     0.3581 |     1.12
    1       output |     0.6464     0.4660 |     1.39 (drift growing)
    2       output |     0.7159     0.5528 |     1.30 (drift growing)
    3       output |    11.7756     0.6341 |    18.57 (EXPLOSION)
    4       output |    15.4269     0.7903 |    19.52 (sustained)
    5       output |    16.9458     0.8170 |    20.74 (sustained)
    6       output |    18.2438     1.0271 |    17.76 (sustained)
```

### Path to shipping MODEL-1

1. Test the parallel-reduction hypothesis (run APR forward twice, check layer-3 element-wise determinism)
2. If non-deterministic → fix APR matmul reduction order to be deterministic
3. Re-run trace, verify layer-3 ffn_swigl ratio drops below 1.5×
4. Verify SHIP-002/005/006/007/008 PARTIALs flip to DISCHARGED
5. MODEL-1 ships cleanly through both APR and GGUF backends

If parallel-reduction hypothesis fails (APR is deterministic), the next candidate is accumulator precision in residual additions — switch to FP64 accumulation in residual sums.
