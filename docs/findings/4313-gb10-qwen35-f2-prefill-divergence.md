# GB10 (sm_121): Qwen3.5 F2 guard falls back to CPU on code prompts, and the fault is in the batched prefill

- **Status:** RED on gx10. Owner: aprender-98. This is a 0.70.0-final gate item. It does not gate 0.69.5-rc.2 (cop ruling, 2026-09-25).
- **Model:** `Qwen3.5-4B-Q4_K_M.gguf` (sha256 `00fe7986ff5f…`).
- **Binary:** `apr 0.69.5-rc.1 (5660b0877)`, build `exe-sha256:d39bb0b6…`. The same build ran on both hosts.
- **Hosts:** gx10 (NVIDIA GB10, sm_121, aarch64) and lambda (RTX 4090, sm_89, x86_64).
- **User-visible effect:** the run is still correct but slow. F2 rejects the GPU, so the run is served on CPU (`fell_back: true`, about 0.2 tok/s on the rc smoke).

## Observation

The rc.1 smoke on gx10 (`p850.txt`, 1006 prompt tokens, `--chat`) failed with:

```
warning: GPU output diverges from CPU at position 63 (argmax 248045 != 248045, cosine 0.8425);
min cosine 0.8425, validated via batched prefill — falling back to CPU
```

Lambda passes F2 on the same prompt, with the same binary and the same model sha.

## Prompt variation (≥4 distinct inputs before naming anything)

Each run used `APR_F2_REVALIDATE=1` and a fresh receipt dir, with `--backend cuda` and `--temperature 0`. F2 judges the last 64 prompt tokens plus 1 decode step.

| Input (distinct probe tail) | Prompt tok | F2 on gx10 |
|---|---|---|
| p850 (code question), `--chat` | 1006 | FAIL, pos 63, cos 0.8425 |
| p850, no `--chat` / head / tail-2500B (same tail) | — | FAIL, pos 63, cos 0.8425 (same probe) |
| Rust source excerpt (`session.rs` head), `--chat` | 813 | FAIL, pos 39, cos 0.7788 |
| Rust excerpt tails of 180/260/360 B (same last 64 tok) | 76/98/131 | FAIL, pos 39, cos 0.7788 (deterministic) |
| `docs/BEATS.md` head, `--chat` | ~700 | pass (65 positions) |
| English prose, repeated, `--chat` | 613 | pass |
| p850 first 450 B, `--chat` | 193 | pass |
| Biology question, `--chat` | short | pass |
| 5 earlier short/medium prompts (13–65 positions) | — | pass |

Two distinct probes fail, and both are code. Every divergence has an **equal argmax** and a low whole-vocabulary cosine. It is deterministic to 4 digits across reruns.

## What it is NOT (each an A/B on the failing Rust probe, t180)

| Knob | cos @ pos 39 | Conclusion |
|---|---|---|
| default (f16 prefill GEMM, #4313) | 0.7788 | — |
| `APR_QWEN35_PREFILL_GEMM=f32` | 0.7782 (p850: 0.8409 vs 0.8425) | **not the #4313 f16 default.** The fault predates it |
| `MWV_Q6K=1` / `DP4A_Q6K=1` (lm_head is tied `token_embd`, Q6_K) | 0.7788 / 0.7788 | not the lm_head GEMV variant |
| `BATCHED_PREFILL=0` | 0.7788 | that knob does not reach the qwen35 F2 probe |
| `APR_QWEN35_PREFILL_ATTENTION=f32` / `flash` | 0.7788 / 0.7775 | not the prefill attention path |

**The CPU reference is not the fault either.** `apr parity` on gx10 compares CPU against the **per-token** GPU forward. On the t180 prompt it passes all 65 positions (min cos 0.9819 @ 45). With `SKIP_PARITY_GATE=1` the gx10 GPU (`ran: gpu`, `fell_back: false`) produces greedy text byte-identical to the gx10 CPU over 24 tokens.

## Where it is

The GPU per-token (decode) path agrees with the CPU. Only the GPU **batched prefill** (`prefill_logits_at`, which F2 and serving both use) diverges. Every GEMM, attention and lm_head knob leaves the error unchanged. That leaves the GDN (DeltaNet) prefill ops that have no knob, in `Qwen35CudaModel::prefill_deltanet`:

- `qwen35_conv1d_rows`
- `qwen35_l2_norm_rows`
- `qwen35_gates_rows`
- `qwen35_delta_rule_scan` (`DeltaRuleChunkScanKernel`, state on chip)
- `gdn_gated_rmsnorm_into`

Also on the list are `batched_rmsnorm_into` and `prefill_ffn_rows` (SwiGLU). One of them misbehaves on sm_121 for content-dependent inputs. The op is **not yet named**.

## Next step (needs a GPU test binary on gx10)

1. Run `qwen35_prefill_equals_per_token` with the failing probe token ids on gx10, extended to compare each GDN op's output between prefill and per-token decode at every row. The first diverging op names the defect.
2. gx10's root disk was at 93% (69 G free) on 2026-09-25, and session rules forbid building there. The test binary has to be built elsewhere for aarch64, or on gx10 once disk is freed and the cop allows it.
3. Once the op is fixed, the gx10 code-prompt F2 row turns green. Until then F2 falling back is the **correct** safe behaviour.

## Raw evidence (gx10)

- `~/f2-4313/b/`: p850 variants, f16 vs f32.
- `~/f2-4313/c/`: 4 distinct long prompts, plus rust f32.
- `~/f2-4313/d/`: Rust tail lengths.
- `~/f2-4313/f/`: CPU vs GPU greedy, `apr parity --json`.
- `~/f2-4313/g/`: Q6K variants and BATCHED_PREFILL.
- `~/f2-4313/h/`: prefill attention f32 vs flash.
- `~/rel-0695-e6/smoke-gx10-a5b5-v0.69.5-rc.1.*`: the original failing smoke.
