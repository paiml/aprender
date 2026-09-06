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

### Which side is wrong — the probe (`scratchpad/probe26.py`, gguf-py dequant, float64)

Recomputing neurons 2908 / 7035 from the dequantised Q4_K weights of `blk.26.ffn_{gate,up}` and the
CPU-tapped `ffn_norm` input:

| computation | neuron 2908 | neuron 7035 |
|---|---|---|
| f64, f32 input (truth for the quantised weights) | **−1142.0** | **618.4** |
| GPU tap | −1136.5 (−0.5 %) | 616.9 (−0.2 %) |
| CPU tap | −996.4 (−12.7 %) | 573.6 (−7.2 %) |
| input Q8-quantised per 32 (GPU-style) | −1116.3 | 611.9 |
| input Q8-quantised per 256 (Q8_K, one scale per super-block) | **−996.369** | **573.560** |

The CPU value is reproduced to three decimals by quantising the activation vector to int8 with one
scale per 256 elements: the −146.68 outlier at dim 408 sets the scale (1.155) for dims 256–511 and
crushes the other 255 elements to coarse steps; neuron 2908 draws 24.7 of its 68.7 gate pre-activation
from that block. **The CPU reference is the inaccurate side.** The GPU (per-32 activation scales in
its DP4A path) is within 0.5 % of the truth.

## Consequence for the fix (step 2)

The load-time gate and `apr parity` compare the GPU against a CPU path whose activations are
Q8_K-quantised (256-element scale). On massive-activation tokens (position 0 of Qwen2.5-1.5B;
smaller on the 7B: 0.9986) that reference loses up to 13 % on the outlier neurons, and the "GPU
divergence" is the CPU's error. The fix is on the CPU side: the Q4_K × Q8 dot family
(`fused_q4k_q8k_dot`, `fused_q4k_q8k_ffn_up_gate_into`, `fused_q4k_q8k_parallel_matvec_into`, and
the quantiser feeding them) takes one activation scale per 32-element sub-block — the sub-block
the Q4_K scales/mins already iterate — instead of one per 256. This matches the GPU's numerics and
is strictly more accurate; the GPU m=1 stream is untouched (CF-4 holds by construction on the GPU
side), the CPU stream changes only where a 256-block held an outlier. Revert → the 1.5B per-op
table names layer 26 again; 7B GREEN.

Not done here: the kernel change itself (next session; the receipt carries this table).
