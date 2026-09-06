# L0-1b — step 1 (bisection by controls) and step 0 (the per-op table), lambda, 2026-09-06

Ticket PMAT-1070, issue #2971, PP-066 row L0-1b. Binary: `apr 0.65.2 (cbb511c55)` built with
`--features cuda` from agent/L0-1b (plan commit on top of agent/L0-1 @ 665b71194); model
`/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf`; prompt = `PROMPT` in
`scripts/check_model_parity.sh` (78 tokens); host lambda (RTX 4090, sm_89). Every number below
is a measurement of that binary on that host; nothing is inferred from a lane verdict (I16).

## Step 1 — arms (each row is one `apr parity <1.5B> --json` run)

| arm | env | positions | min cosine (position, token) | # < 0.98 | argmax ≠ |
|---|---|---|---|---|---|
| A0 baseline | — | 78 | 0.9508 (0, 785) | 1 | 2 |
| A1 graph off | `SKIP_CUDA_GRAPH=1` | 78 | 0.9508 (0, 785) | 1 | 2 |
| A2 FP8 decode off | `FP8_DECODE=0` | 78 | 0.9508 (0, 785) | 1 | 2 |
| A3 FP8 all off | `FP8_PREFILL=0 FP8_DECODE=0` | 78 | 0.9508 (0, 785) | 1 | 2 |
| A4 flash decode off | `FLASH_DECODE=0` | 78 | 0.9508 (0, 785) | 1 | 2 |
| A5 fused gate-up off | `FUSED_GATE_UP=0` | 78 | 0.9508 (0, 785) | 1 | 2 |
| A6 all off | A1+A3+A4+A5 | 78 | 0.9508 (0, 785) | 1 | 2 |
| reference 7B | — | 78 | 0.9986 (0, 785) | 0 | 2 |

Seven arms, one cosine to four decimals: no switchable path (graph capture, FP8 cuBLASLt, flash
decode, fused gate+up) is the mechanism. The divergence lives in the base per-layer kernels and
appears at position 0 only.

## Step 0 — `apr parity <model> --per-op` (281 rows × 78 positions; 9 s on the 1.5B, 18 s on the 7B)

Layers 0–25 of the 1.5B agree at every op (min cosine ≥ 0.9916, most ≥ 0.998). Then, at
position 0:

| layer | op | min cosine | @pos | max \|Δ\| |
|---|---|---|---|---|
| 26 | attn_norm … post_attn_residual | ≥ 0.9962 | — | ≤ 12.85 |
| 26 | ffn_norm | 0.998730 | 21 | 0.31 |
| 26 | ffn_swigl | 0.997669 | 21 | **140.2** |
| 26 | ffn_out | 0.996157 | 23 | **408.5** |
| 26 | post_ffn_residual | **0.660150** | 0 | 395.7 |
| 27 | attn_norm | 0.761651 | 0 | 27.1 |
| 27 | post_attn_residual | 0.645909 | 0 | 394.8 |
| 27 | ffn_swigl | 0.581351 | 0 | 42.0 |
| 27 | post_ffn_residual | 0.559132 | 0 | 399.0 |
| — | final_norm | 0.639776 | 0 | 86.4 |
| — | lm_head | **0.950827** | 0 | 12.0 |

The `lm_head` row is the gate's own number (0.9508): the table closes the loop end to end.
The 7B: every op ≥ 0.98 over 78 positions (worst: layer 23 attn_out 0.9946).

### The element (layer 26, position 0)

| stage | CPU | GPU | note |
|---|---|---|---|
| post_attn_residual dim 408 | −3664.4 | −3677.2 | the residual carries a massive activation (also dims 520/940 ≈ +2233); cosine 0.999999 |
| ffn_norm (input to gate/up) | cos 1.000000, max Δ 0.028 | | dim 408 = −146.68 on both sides — the FFN input is identical |
| ffn_swigl neuron 2908 | −996.4 | −1136.5 | +14 % |
| ffn_swigl neuron 7035 | 573.6 | 616.9 | +7.5 % |
| ffn_out dim 408 | +3675.7 | +4084.2 | the FFN CANCELS the massive activation: CPU residual → +11.3 |
| post_ffn_residual dim 408 | 11.3 | 407.0 | the GPU keeps 407 → layer 27 cosine 0.56–0.76 → logits 0.9508 |

Positions ≥ 1 carry no massive activation and agree (ffn_swigl@26 pos 1: cos 0.999, max Δ 0.7).

### Which side is wrong — the probe (`.pr/L0-1b/step0/probe26.py`, gguf-py dequant, float64)

Recomputing neurons 2908 / 7035 from the dequantised Q4_K weights of `blk.26.ffn_{gate,up}` and the
CPU-tapped `ffn_norm` input:

| computation | neuron 2908 | neuron 7035 |
|---|---|---|
| f64, f32 input (truth for the quantised weights) | **−1142.0** | **618.4** |
| GPU tap | −1136.5 (−0.5 %) | 616.9 (−0.2 %) |
| CPU tap | −996.4 (−12.7 %) | 573.6 (−7.2 %) |
| input Q8-quantised per 32 | −1116.3 (−2.25 %) | 611.9 |
| input Q8-quantised per 256 (Q8_K, one scale per super-block) | **−996.369** | **573.560** |

The CPU value is reproduced to three decimals by quantising the activation vector to int8 with one
scale per 256 elements: the −146.68 outlier at dim 408 sets the scale (1.155) for dims 256–511 and
crushes the other 255 elements to coarse steps; neuron 2908 draws 24.7 of its 68.7 gate pre-activation
from that block. **The CPU reference is the inaccurate side.** The GPU tap is within 0.5 % of the truth; note (quorum lane 2)
that per-32 int8 quantisation of the input gives −1116.3, 2.25 % off — so the GPU's own activation numerics are NOT
established by this table, only that they are closer to the truth than Q8_K's per-256 scale.

**End-to-end falsifier (arm A7, run after the quorum):** `DIRECT_FP32_GEMV=1 apr parity <1.5B> --json` — the CPU
reference uses f32 activations in `fused_q4k_parallel_matvec_into` (parallel_k.rs:258) and nothing else changes —
**min cosine 0.950827 → 0.999896 (position 22), 0 positions < 0.98, 0 argmax mismatches; `--per-op` names no op
(layer 26 ffn_swigl 0.999247, post_ffn_residual 0.999857, lm_head 0.999896).** The CPU's Q8_K activation
quantisation is the mechanism, shown on the shipped binary with one environment variable.

## Consequence for the fix (step 2)

The load-time gate and `apr parity` compare the GPU against a CPU path whose activations are
Q8_K-quantised (256-element scale). On massive-activation tokens (position 0 of Qwen2.5-1.5B;
smaller on the 7B: 0.9986) that reference loses up to 13 % on the outlier neurons, and the "GPU
divergence" is the CPU's error. The fix is on the CPU side: the Q4_K × Q8 dot family
(`fused_q4k_q8k_dot`, `fused_q4k_q8k_ffn_up_gate_into`, `fused_q4k_q8k_parallel_matvec_into`, and
the quantiser feeding them) must stop letting one element set the scale of 255 others. Candidates
measured on this token (`.pr/L0-1b/step0/probe26b.py`; error on neuron 2908 vs the f64 truth):
per-32 scales −2.25 % (the quorum's pick; touches every SIMD variant); f32 activations ≈ 0 but −17 %
CPU decode (PMAT-305); an outlier split (zero |x| > τ·rms, τ ∈ [4, 8] → the same 3 dims, before
Q8_K, then add the outlier terms exactly) −0.06 % at ≈ zero cost when no outlier is present. The
split needs a criterion that separates crushed blocks from ordinary ones — see the table below.
The GPU m=1 stream is untouched by any of these (CF-4 holds by construction on the GPU side).
Revert → the 1.5B per-op table names layer 26 again; 7B GREEN (POP-F-003).

Not done here: the kernel change itself (next session; the receipt carries this table).

### Fallback criterion — measured on this tree (78 positions × 28 layers, CPU dump trees)

Per 256-block, `max|x| / second-largest |x|` (the crushing statistic: how much of the block's int8 range one
element takes):

| matmul input | pos 0 max | pos 0 blocks ≥ 8 | pos ≥ 1 max | pos ≥ 1 p99.9 | pos ≥ 1 blocks ≥ 8 | blocks |
|---|---|---|---|---|---|---|
| attn_norm (→ qkv) | 21.4 | 14 | 4.6 | 3.9 | 0 | 12,936 |
| ffn_norm (→ gate/up) | 20.1 | 27 | 6.0 | 4.7 | 0 | 12,936 |
| final_norm (→ lm_head) | 2.7 | 0 | 7.7 | 7.2 | 0 | 462 |
| attention (→ o_proj) | 1.0 | 0 | 5.2 | 3.9 | 0 | 12,936 |
| ffn_swigl (→ down) | 641.3 | 30 | 89.5 | 21.9 | 553 | 75,460 |

(`max|x|/rms` per block does not separate: it saturates at 16.0 on ordinary blocks too; global `max/rms` does not either —
ffn_swigl's median is 78.)

**Decided step-2 spec (orchestrator, after the quorum; the quorum's per-32 is the fallback plan):** the CPU Q4_K × Q8_K
drivers (`fused_q4k_parallel_matvec_into`, the gate/up driver, the scratch-path twins) run a matmul with f32 activations
(the existing `DIRECT_FP32_GEMV` path, `fused_q4k_dot_simd`) whenever the **normed residual-stream input** (qkv, gate/up,
lm_head) contains a block with `max/second ≥ 8` — basis: the table above (77 non-first positions × 28 layers never exceed
6.0; the first token's crushed blocks are ≥ 20). The down projection keeps Q8_K: spiky SwiGLU inputs are normal and its
error is ≤ 0.4 % cosine at every position (per-op rows). Expected: the f32 path measured end to end as arm A7
(0.999896); cost only on massive-activation tokens. If the down-projection exclusion proves wrong on another manifest model,
per-32 scales (quorum) is the fallback plan.
