# #4485 — decode GEMV precision split, PRM-S1 v2 on lambda-cuda

| Field | Value |
|---|---|
| Set | review-replay-v2, 200 items (50 each at 2k/8k/16k/32k), sha `a4910a65…b30632` |
| Cell | lambda-cuda (RTX 4090, sm_89), shared host |
| apr | `apr 0.70.0 (bed98197d4)` = 6cc64fe20 + #4485 policy + GEMV log, sha256 `088bf48f4aa8c317b8048e0e94cf52f5252a1a24dd68175555e8e434e359f5af` |
| llama.cpp | `d1d3c3396aa13a5f239109a822666c4870490ad5`, `-ngl 99 -c 36864 -np 1` |
| Model | Qwen3.5-4B-Q4_K_M.gguf, sha `00fe7986…ef11a4` |
| Engagement | server log per leg: `decode GEMV q4k=HwDp4a q6k=HwDp4a (production, sm_89)` and `q4k=Mwv q6k=Mwv (env override)` |

## Quality falsifier (tolerance stated in advance: the verdict mix holds)

| Pair | Verdict identical | Mix (Fail/Pass) |
|---|---|---|
| DP4A vs float (apr) | 198/200 | 39/161 vs 39/161 — **holds** |
| llama vs DP4A | 194/200 | 37/163 vs 39/161 |

Greedy text is not identical (54/200 output-token-identical, DP4A vs float). That is expected
from Q8_1 activation rounding. The kernel-level bound lives in `forward_qwen35_gemv_policy_tests.rs`:
DP4A graph vs eager stays within the |DP4A − float| band, cos ≥ 0.999 on 0.8B and 4B, and
`tests_dp4a_graph_replay.rs` shows the replayed DP4A GEMV is bit-exact to eager.

## Decode tok/s (both apr passes pooled, n=100 per stratum; llama n=50)

| Stratum | DP4A p50/p75 | float p50/p75 | llama p50/p75 | DP4A/float p75 | DP4A/llama p75 |
|---|---|---|---|---|---|
| 2k | 100.1 / 105.8 | 69.0 / 101.0 | 141.0 / 148.1 | 1.048 | 0.714 |
| 8k | 80.6 / 93.0 | 78.5 / 85.3 | 138.9 / 141.1 | 1.090 | 0.659 |
| 16k | 78.0 / 83.6 | 80.4 / 92.3 | 131.3 / 136.2 | 0.906 | 0.614 |
| 32k | 51.6 / 57.4 | 69.2 / 72.0 | 163.3 / 171.8 | **0.797** | 0.334 |
| all | 79.0 / 95.0 | 75.2 / 91.2 | 137.5 / 147.9 | 1.042 | 0.642 |

p75 is used because contention only makes items slower. An nvidia-smi app logger showed foreign
processes on the GPU without the flock during both passes, and float pass 1 lost its first ~70 items to that.

## Reading

- DP4A wins at 2k/8k by 5–9% (p75). Smoke on one prompt: 127.1 vs 101.5 tok/s.
- **At 32k DP4A is slower in both passes** (48–59 vs 57–81 tok/s). A GEMV has no context length,
  so this points at an interaction with the long-context path (attention / Q8_1 staging), not the
  GEMV itself. Open: needs an in-process nsys A/B at 32k before the DP4A default applies at long context.
- The residual gap to llama is 1.4–1.6× at ≤16k and 3× at 32k. GEMV precision is not the main term
  at long context. The 32k gap is attention-bound work for a separate ticket.
