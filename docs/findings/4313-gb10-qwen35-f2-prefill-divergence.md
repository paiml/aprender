# GB10 (sm_121): Qwen3.5 F2 guard falls back to CPU on code prompts. No op is defective; the gate's metric is ill-conditioned

- **Status:** RED on gx10 (false reject: see Bisect). Owner: aprender-98. This is a 0.70.0-final gate item. It does not gate 0.69.5-rc.2 (cop ruling, 2026-09-25).
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

## Superseded hypothesis: "the fault is in the batched prefill"

An earlier revision of this file named the GDN prefill ops. The per-layer bisect below refutes that. Tokenization for these runs is the GGUF tokenizer on the raw text (`APR_BISECT_TEXT`), so the probes are close to, but not byte-identical with, `apr run --chat`'s.

## Bisect (2026-09-25): `qwen35_bisect_real_probe_per_layer`

The test is a diagnostic `#[ignore]` test in `forward_qwen35_cuda_prefill_tests.rs`. It reproduces F2's shape: a model sized for 66 positions, a state sized for probe + 1, and `prefill_logits_at` at every position. It compares that against GPU per-token decode **and** the CPU `forward_single_qwen35` oracle, then dumps per-layer conv/ssm/KV state and per-position logits. It was cross-built for aarch64 on lambda, since there are no builds on gx10, and run on gx10 through gpu-q.

1. **The batched prefill is exact.** On every probe (p850, rust, prose, t180, raw and ChatML-wrapped), gx10 batched equals gx10 per-token at every position (cos 1.000000). Every layer's conv, ssm and KV state agrees to rel L∞ ≤ 1e-5 (f32 noise).
2. **The t180 raw probe reproduces F2: gx10 GPU vs gx10 CPU gives cos 0.8315 @ pos 40.** The divergence is **isolated**: pos 39 is 0.9995 and pos 41 is 0.9995. Nothing carried in the recurrent or KV state is corrupted.
3. **Cross-host, the CPUs disagree with each other just as hard.** Same probe ids, same model sha; lambda CPU is x86 AVX2, gx10 CPU is aarch64 NEON:

| pos | lambda CPU ~ gx10 CPU | lambda CPU ~ gx10 GPU | gx10 CPU ~ gx10 GPU |
|---|---|---|---|
| 27 | **0.7266** | 0.7113 | 0.9992 |
| 40 | 0.9985 | 0.8357 | **0.8315** |
| 52 | **0.2710** | 0.2766 | 0.9898 |
| others | ≥ 0.995 | ≥ 0.995 | ≥ 0.995 |

At each spike a *different* backend is the odd one out, and the two CPUs are among them. So this is not a GB10 op.

4. **In probability space the spikes are small.** The top token is identical on all three backends at all three positions. The top-4 lists are equal or differ in the 3rd slot only.
   - pos 40: p1 0.985 (CPU) vs 0.959 (GPU), KL 1.7e-2.
   - pos 27, lambda vs gx10 CPU: KL 6.2e-2.

   The whole-vocab raw-logit cosine collapses because a handful of **tail vocabulary rows** swing by several logits (pos 40: token 53983 +7.6, 166756 +6.2, 107110 -5.9). Token 53983 recurs as a min or max-swing id at pos 27, 40 and 52 across backends.

## Where it is: the F2 metric, not a kernel

F2 (`f2_multi_position_report`, `infer/inference_result.rs`) rejects when **any single position** has a whole-vocab cosine below 0.95. On repetitive code (`self.x … self.y …`) some positions are ill-conditioned: fp32 rounding differences of about 1e-6, which differ legitimately between cuBLAS/PTX, AVX2 and NEON, are amplified into large swings on a few tail logits. The distribution the sampler sees is unchanged. The x86 CPU would "fail" F2 against the aarch64 CPU on the same probe.

- **Named "op":** there is none. The amplifier is the lm_head projection of the final hidden onto a few tail rows of the tied Q6_K `token_embd` (53983 and others), measured by a cosine that those rows dominate.
- **Fix direction (stage/0.70.1, owner aprender-59; needs cop sign-off since it changes a safety gate):** judge F2 in probability space, e.g. KL or top-k agreement plus p1 delta, or require a spike to persist across ≥2 adjacent positions. A real device defect corrupts state and persists, as the pre-#3596 failures did. A transient does not.
- **Until then** F2 falling back on these probes is safe but a **false reject**: the gx10 GPU output is as correct as either CPU.

## Reproduce

```
APR_BISECT_MODEL=~/models/Qwen3.5-4B-Q4_K_M.gguf APR_BISECT_TEXT=t180.txt APR_BISECT_DUMP=out \
  <realizar test bin> --exact gguf::cuda::forward_qwen35_cuda::prefill::prefill_tests::qwen35_bisect_real_probe_per_layer --ignored --nocapture
# APR_BISECT_CPU_ONLY=1 dumps the CPU oracle only (no GPU); APR_BISECT_CHAT=1 wraps in ChatML; APR_BISECT_ROWS sets chunk rows.
```

## Raw evidence (gx10)

- `~/f2-4313/b/`: p850 variants, f16 vs f32.
- `~/f2-4313/c/`: 4 distinct long prompts, plus rust f32.
- `~/f2-4313/d/`: Rust tail lengths.
- `~/f2-4313/f/`: CPU vs GPU greedy, `apr parity --json`.
- `~/f2-4313/g/`: Q6K variants and BATCHED_PREFILL.
- `~/f2-4313/h/`: prefill attention f32 vs flash.
- `~/rel-0695-e6/smoke-gx10-a5b5-v0.69.5-rc.1.*`: the original failing smoke.
- `~/f2-4313/bis/`: bisect logs (`p850/rust/prose.log`, `t180chat/p850chat/t180raw/proschat.log`) and `dump-gx10/` logits. The lambda CPU dump is at `/mnt/nvme-raid0/tmp/aprender-98/xb/dump-lambda/`.
