## SHIP-007 layer-3 sub-FFN bisection (2026-04-28)

### Setup

After PR #1082 (sub-FFN populate) + PR #1083 (CLI wiring) merged on 2026-04-28, `apr trace --payload <gguf>` now emits per-layer sub-FFN stats (ffn_gate / ffn_up / ffn_silu / ffn_swigl / ffn_out) for the FIRST TIME on the GGUF side. This unblocks the §32.4 layer-3 deep bisection.

Binary rebuilt: `/mnt/nvme-raid0/targets/aprender/release/apr` with `--features cuda` from main HEAD post-#1083 merge.

### Layer-3 side-by-side comparison

| Stat | APR | GGUF | Ratio |
|------|----:|-----:|------:|
| attn_norm (input to QKV) | std 0.293 | std 0.276 | 1.06× — fine |
| qkv (post matmul + bias) | std 1.953 | std 0.723 | 2.70× — TRACE-POINT MISMATCH per §32 |
| attn_out (post O-proj) | std 0.182 | std 0.199 | 0.91× — fine |
| ffn_norm (input to FFN gate) | std 0.995 | std 1.035 | 0.96× — fine |
| **ffn_gate (post gate matmul)** | **std 1.924** | **std 1.413** | **1.36× — DIVERGENCE STARTS HERE** |
| ffn_up (post up matmul) | std 1.335 | std 1.456 | 0.92× — fine |
| ffn_silu (silu of gate) | std 0.168 | std 0.037 | **4.59× — silu amplifies gate** |
| ffn_swigl (silu × up) | std 1.222 | std 0.067 | **18.23× — multiply compounds** |
| ffn_out (post down matmul) | std 11.459 | std 0.191 | **60.0× — final amplification** |

### Conclusion

**ffn_gate at layer 3 is the FIRST sub-FFN site where APR and GGUF aggregate stats diverge significantly (1.36× std ratio).**

The chain:
1. Layer-3 inputs (attn_norm, ffn_norm) agree within 5-6%
2. Layer-3 ffn_gate weights are **byte-identical** APR ≡ GGUF (verified earlier today via `diag_compare_layer3_ffn.rs`)
3. Yet ffn_gate post-matmul output diverges 36%

This is paradoxical unless:
- (a) The **per-element** values of ffn_norm input differ (despite similar std), and the matmul propagates per-element differences
- (b) The matmul implementation has nondeterminism not visible at the aggregate stat level

(a) is most plausible — cumulative F32 precision drift through layers 0-2 residual connections produces per-element values whose std looks similar to GGUF but whose actual elements differ enough that layer-3 ffn_gate matmul produces 36% wider distribution.

### Next step

To confirm (a), we need element-wise diff of ffn_norm input at layer 3 between APR and GGUF (not just aggregate stats). Currently `apr trace` only emits stats, not raw tensors. The next investigation step is to add a `--save-tensor <stage>` flag to `apr trace` that captures specific layer/sub-stage tensors for byte-level comparison.

Alternatively, a Rust diagnostic example can run both forward passes from the same input and compute per-element diff at each named stage.

### Why this matters for shipping MODEL-1

MODEL-1 (`paiml/qwen2.5-coder-7b-apache-q4k-v1`) is published to HuggingFace but its APR backend produces wrong outputs. SHIP-002 / 005 / 006 / 007 / 008 (5 PARTIALs) all depend on this fix. With this layer-3 sub-FFN bisection:

- Bug surface narrowed from "(layer 3, FFN sub-block)" (§17) to **"(layer 3, ffn_gate matmul output)"** — the gate matmul is the first site where divergence is statistically detectable.
- Weights are byte-identical → fix is NOT in the converter
- Aggregate input stats are similar → fix is in per-element behavior of ffn_norm input or matmul nondeterminism
- Fix scope: investigate element-wise ffn_norm differences between APR and GGUF forward paths

Once the per-element divergence is localized and fixed, the 5 PARTIALs can be promoted to DISCHARGED via verification runs. MODEL-1 ships cleanly through both APR and GGUF backends.

### Evidence files

- `apr-trace.txt` — full `apr trace --payload` on `qwen2.5-coder-7b-instruct-q4k.apr`
- `gguf-trace.txt` — full `apr trace --payload` on `qwen2.5-coder-7b-instruct-q4k.gguf`
- This `findings.md` — analysis
