# 0.72 "Train What You Serve" — top-5 row specs (L2 shaping)

**Status:** PROPOSED (look-ahead L2, APR-LOOKAHEAD-001 §4). Nothing here is minted; the cop mints tickets.
**Epic:** #4000 · **Author:** la-72 · **Date:** 2026-09-27 · **Tree read:** origin/main `aca6f2d7f6`
**Exit criteria (operator ruling 2026-09-27, APR-LOOKAHEAD-001 §2a):** T1 verbs end to end on Qwen 3.5 with 0 refusals ·
T2 fine-tune throughput ≥ 0.8× Unsloth · T3 Prometheus B2 challenger · T4 dogfood model rc on Hugging Face · T5 round-trip cosine ≥ 0.98.
Marks: `[V]` verified in tree · `[C]` computed · `[A]` asserted · `[U]` unmeasured.

## §0 The finding that shapes the train `[V]`

T1 reads "the qwen35 refusals are removed". The training verbs have **no** qwen35 refusal. They mis-model it silently:

| Where | What |
|---|---|
| `crates/aprender-train/src/train/pretrain_real.rs:230` | `"qwen3_5" \| "qwen3.5"` → family `qwen3` (dense) |
| `crates/aprender-train/src/transformer/config.rs:246` | `qwen3_5_9b()` is a dense `Decoder`: no Gated DeltaNet (GDN) layers |
| `crates/aprender-train/src/transformer/config.rs:297-301` | any arch prefixed `qwen3` gets head_dim = 128; the 3.5 preset's own comment says 256 |
| `crates/aprender-train*` | no GDN implementation; GDN exists only in aprender-serve (`forward_qwen35*`, `gdn_ops`) |

So "0 refusals" means **GDN forward and backward in the training crate**. LoRA over a frozen base still back-propagates
through every GDN layer, so no adapter-only shortcut avoids the backward. `apr merge` and `apr quantize` are
tensor-name keyed and architecture-independent `[V]`; they need qwen35 cells, not new modelling.

## §1 Critical path `[C]`

```
R1 honesty gate ─► R2 GDN forward (= serve) ─► R3 GDN backward ─► R4 QLoRA e2e ─┬─► R5 Unsloth baseline ─► R14 perf
                                                                                ├─► R6 distill, R7 merge
                                                  #4418 GGUF name map (0.71) ───┴─► R10 round-trip ─► R13 HF rc
```

## §2 Rows

### R1 — Training-arch honesty gate · contract `train-arch-honesty-v1` · K̂ 30 `[A]`
- **Change:** aprender-train refuses any architecture it doesn't model (`UnsupportedArch{arch, missing}`) before any
  weight is read. Delete the `qwen3_5 → qwen3` alias. head_dim comes from `contracts/model-families/<f>.yaml`, not a prefix match.
- **Falsifiers:** TAH-001 (named refusal at the verb) · TAH-002 planted: re-adding the alias turns TAH-001 RED ·
  TAH-003 head_dim table over every family YAML · TAH-004 refusal precedes I/O.
- **Why first:** it is small, it lands under L1/L2 rules as a normal PR, and it stops wrong training today.
  R2/R3 later remove the refusal. A refusal is the honest interim state, not the goal.

### R2 — GDN forward in aprender-train, identical to serve · contract `qwen35-train-gdn-v1` (QTG-001/002/005) · K̂ 90 `[A]`
- **Change:** add a GDN layer (causal conv1d, gated delta recurrence, gated RMSNorm, per-head L2 norm) and a hybrid
  layer schedule to the training transformer. Reuse the `gated-delta-net-v1` equations; don't re-derive them.
- **Gate:** per-position cosine ≥ 0.9999 and equal argmax between aprender-train and aprender-serve logits on Qwen3.5-4B,
  both loading the same .apr bytes (CPU, dequantized f32). This check runs on CPU, so it may run while train-active.
- **Planted:** the sigmoid-gate mutant turns it RED. The same mutant scores 0.7656 on the serving side (qwen35-hybrid-forward-v1).
- **Spike S-R2 (desk, 2026-09-27) — share the kernel, or run a parallel implementation? Answer: parallel training impl, serve as the oracle. No ruling needed.** `[V]`
  - The crate DAG already allows train → serve. `crates/aprender-train/Cargo.toml:84` has
    `realizar = { workspace = true, optional = true }` (implicit feature `realizar`), and aprender-serve does not depend on
    aprender-train, so there is no cycle. `realizar::gguf::forward_qwen35` is `pub` (`crates/aprender-serve/src/gguf/mod.rs:147`)
    and exports `causal_conv1d`, `delta_rule_recurrence` and `delta_rule_recurrence_gqa` (`forward_qwen35.rs:89,130,221`).
  - Those functions cannot be the training forward. They step one token at a time, update the conv/recurrent state in
    place, and save no activations, so autograd has nothing to differentiate through. A training GDN has to be a
    sequence-level (chunked) forward on the tape that keeps S_t (or recomputes it per chunk).
  - So: aprender-train implements its own sequence GDN forward+backward. The parity test QTG-001 runs under
    `--features realizar` and calls serve's `forward_qwen35` on the same .apr bytes as the ORACLE. "train = serve" holds
    by test, not by shared code, which is the same guarantee without a refactor.
  - Moving the recurrence into aprender-compute is a refactor with no extra guarantee; deferred (not a 0.72 row).
  - `[U]` Build cost: `cargo check -p aprender-train --features realizar` was not run (heavy slots were saturated). The
    first R2 PR must show it green.
- **Evidence (2026-09-27, branch `la-72/r3-backward`, no PR while the repo is over the PR cap)** `[V]`
  - `transformer::Qwen35Model::from_gguf` + `forward` on the real Qwen3.5-0.8B-Q4_K_M: QTG-001 holds, logit for logit,
    against serve's `forward_single_qwen35`. The two sides read the GGUF independently, and dequant agrees to ≤ 1e-6.
  - **Finding:** serve's oracle must run under `realizar::quantize::with_fp32_activations`. By default, serve's Q4_K
    matvec quantises the activation to Q8_K first, which alone moves the 0.8B to cos 0.998 of the f32 forward. That gap
    is serve's arithmetic, not a model mismatch. Recorded as a precondition of `train_serve_forward_identity`.

### R3 — GDN backward + gradcheck · contract `qwen35-train-gdn-v1` (QTG-003/004) · K̂ 120 `[A]`
- **Change:** analytic backward for every GDN parameter, plus the carried state S₀ (chunked training passes state across chunks).
- **Gate:** f64 central-difference gradcheck, rel err ≤ 1e-3, on a tiny layer (2 heads, d = 8, T = 16).
- **Planted:** zeroing the read-out gradient (through r_t = S_tᵀk_t) turns it RED.
- **Risk K1:** no in-tree training-side reference exists; this is the train's largest schedule risk.
- **Evidence (2026-09-27, `la-72/r3-backward` @8bbedb0960, branch only)** `[V]`
  - The backward is split into 4 steps. Each step recomputes its own forward, is generic over f32/f64, and carries its
    own gradcheck and planted mutants:
    - scan (`gated_delta_scan_backward`, dS₀ included; QTG-004 read-out mutant RED at 5.2e-2);
    - GDN mixer (`gdn_mixer_backward`; 5 mutants RED, plus the softplus cut-over from review 6b);
    - attention and block (`gated_attn_backward`, `qwen35_block_backward`; 8 RED);
    - whole LM with next-token CE (`Qwen35LmRef::loss_and_grads`; 7 RED).
  - Rows QTG-006/007/008 added.
  - The f32 path is unchanged: the real 0.8B QTG-001 was re-run green after each generic refactor. On the real 0.8B,
    `loss_and_grads` equals an independent f64 CE of `forward`'s logits, and every gradient is finite (about 2 min on CPU).
  - **Finding (QTG-008):** a plain f32 softmax denominator over the 248k vocabulary drops the tail, which biased the
    0.8B loss 1.2e-4 low (8.880096 vs 8.881130). The denominator is now a Kahan sum.
  - K1 is retired: the training side has its own f64 reference.
- **Spike S-R3b (2026-09-27, `la-72/r3-backward` @9af72ac173) — does the R3 backward train the real 0.8B? Yes.** `[V]`
  - Full-weight normalised descent (step 0.1 in weight space) on one 5-token sentence, CPU release, 32 cores:
    loss 8.881 → 2.679 → 0.962 → 0.823 → 3.097 → 0.286. That is about 7.1 s per `loss_and_grads`, deterministic
    across reruns.
  - The gradient is a strong descent direction: the first step cuts the loss 3.3×.
  - **Finding (R4 input):** a fixed-length step is not monotone near the minimum (it bounced at step 4). R4 must use
    AdamW or a decaying schedule, not fixed normalised SGD. The gate is first-step descent plus final < initial/10.
  - CPU full-weight training of the 0.8B is spike-grade only (7 s per 5 tokens); R4's throughput work stays on CUDA.

### R4 — QLoRA end to end on Qwen3.5-4B (CUDA) · contract `qwen35-qlora-e2e-v1` · K̂ 90 `[A]`
- **Cell:** NF4 base; LoRA r16 on attention, MLP and GDN projections; 1,000-sample pinned set; seed 42; 200 steps; RTX 4090.
- **Gates:** loss(last 10) ≤ 0.9 × loss(first 10), all finite. Served base+adapter equals the training-side merged forward
  (cos ≥ 0.999, equal argmax).
- **Planted:** a zero step (lr = `f32::MIN_POSITIVE`) must FAIL the loss gate with bit-identical losses. Not lr = 0: the
  in-tree AdamW panics on lr = 0, and a crash reads as the falsifier firing (S-R4a). The receipt names the device from a trace line (CLAUDE.md verification rule 2).
- **Spike S-R4a (2026-09-27, `la-72/r3-backward` @045a7edb73) — does LoRA + AdamW over the R3 backward train the real
  0.8B? Yes, monotone.** `[V]`
  - `transformer::Qwen35Lora`: r16, alpha 32, on all 150 attention/GDN/MLP projections (10.2M params, 1.3% of 0.8B).
    Adapter gradients are projected from the full `dW'` (`dA = s·Bᵀ·dW'`, `dB = s·dW'·Aᵀ`), and the in-tree `AdamW`
    steps the adapters.
  - Measured with AdamW at lr 1e-4 on one 5-token sentence (CPU release): loss 8.881 → 5.394 → 3.210 → 1.631 → 0.442 →
    0.053 → 0.011 → 0.004 → 0.002.
  - Step cost is about 7.2 s: forward+backward 6.5–7.3 s, merge 0.1 s, projection+AdamW 0.5 s.
  - This answers S-R3b: AdamW does not bounce the way fixed-step SGD did.
  - Gates: FALSIFY-QTG-009 (finite-difference check per adapter tensor, 2 seeds; base never written; a zero step is
    exact) with 6 planted mutants RED.
  - What R4 on CUDA still needs: an NF4 base, and adapter gradients that never hold all of `dW'` at once (14.3 GB at 4B; per-matrix streaming costs 94 MB, S-R17).
- **Oracle and pre-flight (from S-R4a):** QQE-005 holds the CUDA adapter gradients to the CPU `Qwen35Lora` reference
  (rel ≤ 1e-2 per tensor, 0.8B, B randomised). QQE-006 is the CPU pre-flight (0.8B, 8 AdamW steps, monotone, < ½ start),
  which must be green before GPU time is spent. The target set is pinned by GGUF tensor name.
- **Existing surface `[V]`:** QLoRA path at `crates/apr-cli/src/commands/finetune.rs:270-336`. The contract it cites,
  `qlora-training-loop-v1` (`finetune.rs:349`), has no file (row R16).

### R5 — Unsloth fine-tune throughput harness · contract `beat-unsloth-finetune-throughput-v1` · K̂ 60 `[A]`
- **Today `[V]`:** there is no training-throughput harness. The sibling `beat-unsloth-coldstart-speed-v1` explicitly
  *concedes* GPU in-loop QLoRA throughput to Unsloth. T2 reverses that concession.
- **Harness:** pinned Unsloth (uv lock, versions recorded) vs `apr finetune`, same GPU, same recipe, same data sha.
  Median of 3 runs. Trained tokens/s over 200 timed steps, excluding load.
- **Same-work guard (planted):** trainable-parameter counts must match within 1%. Halving apr's targets must FAIL.
- **First run = baseline.** The 0.8 threshold is fixed by operator ruling. The gap feeds R14.
- **Spike S-R5 (desk, 2026-09-27) `[V, external]`:** Unsloth supports Qwen3.5 fine-tuning (0.8B–122B) with its own Triton
  kernels for the GDN layers, and needs transformers v5. It **advises against QLoRA on Qwen3.5** ("higher than normal
  quantization differences"). Its default targets are `q,k,v,o,gate,up,down`, with no GDN projections. It lists 10 GB
  for 4B bf16 LoRA. Source: unsloth.ai/docs/models/qwen3.5/fine-tune. **Therefore the T2 cell is bf16 LoRA, not QLoRA**,
  with Unsloth's default targets on both sides. The contract is updated to match.
- **Consequence for the ranking:** R15 ("`-m lora` is CPU F32 today", `finetune.rs:280` on main aca6f2d7f6, re-measured by R19) moves onto the critical path,
  because bf16 LoRA on CUDA IS the T2 cell. R4 stays the T1 finetune cell, but NF4 quality on Qwen3.5 is now a known risk
  (K10). R4 adds a gate: QLoRA's final loss is within 5% of bf16 LoRA's on the same cell, or QLoRA is documented as
  unsupported for qwen3.5 (an honest refusal, not a silent quality loss).

### Spike S-R17 — does the 4B fit a 24 GB RTX 4090, and must R4 avoid `dW'`? · `[V]` (2026-09-27, desk, GGUF shapes)
- **Method:** tensor shapes read from `~/models/Qwen3.5-{0.8B,4B}-Q4_K_M.gguf` (python `gguf`). Targets are the GGUF names
  pinned in `qwen35-qlora-e2e-v1`. Cross-check: the 0.8B gives 150 slots and 10.2M r16 adapter params, the same as S-R4a.
- **4B:** 4.206B params, 200 target slots holding 3.565B of them, r16 adapters 30.5M.

  | Plan | Weights-side bytes |
  |---|---|
  | S-R4a CPU approach (f32 base + f32 work copy + full f32 grads) | 16.8 + 16.8 + 16.8 ≈ 50 GB host RAM |
  | all target `dW'` at once, f32 | 14.3 GB |
  | `dW'` streamed one matrix / one layer at a time | 0.094 GB / 0.451 GB |
  | NF4 targets (4 bit + f32 absmax per 64) + non-target bf16 | 2.01 + 1.28 = 3.3 GB |
  | bf16 base (the T2 bf16-LoRA cell) | 8.4 GB |
  | adapters + grad + AdamW m, v (f32) | 0.49 GB |

- **Answers:** (a) R4 QLoRA leaves about 20 GB for activations on a 4090, and the T2 bf16 cell leaves about 15 GB. (b) The
  memory constraint is "never all of `dW'` at once" (14.3 GB), not "never `dW'`": a per-matrix `dW'` costs 94 MB. Streaming
  it costs compute (a rows×cols GEMM per matrix, which the adapter-only path avoids), not memory. So R4 may stream `dW'`
  first and treat the adapter-only backward as an R14 throughput item. (c) The QQE-006 CPU pre-flight stays on the 0.8B
  (about 9 GB), because the S-R4a approach at 4B needs about 50 GB of host RAM.
- **Not measured:** activation memory (it depends on sequence length and checkpointing). R17 owns it.

### Spike S-R4c — can QQE-005 compare a Q4_K CPU base with the NF4 CUDA base? · `[V]` (2026-09-27, measured, CPU)
- **Method:** Qwen3.5-0.8B, r16 alpha 32 over all 150 targets, `B` randomised (scale 0.01, fixed seed), tokens
  `[9707, 11, 1879]`. Adapter grads from `Qwen35Lora::set_grads` on the Q4_K-dequantized base, then on the same base with every
  target round-tripped through `trueno_gpu::kernels::quantize_nf4`/`dequantize_nf4` (the quantizer the CUDA QLoRA path uses).
  Test: `real_model_nf4_base_gradient_gap` (ignored; `QWEN35_GGUF`), branch `la-72/r3-backward` @8a9f4f0104.
- **Result:** loss 9.179 (Q4_K) vs 9.692 (NF4). The per-tensor relative gap over 300 adapter tensors is min 0.332, median 0.710,
  p90 0.872 and max 1.195. **All 300 are above QQE-005's 1e-2.** The control (same base twice) is 0.
- **Answer:** no. The base difference is 30–120× the tolerance, so QQE-005 is meaningful only if the CPU reference loads the NF4
  round-tripped base (the K12 fix, now in the contract). The round trip itself is pinned by
  `nf4_round_trip_base_is_idempotent_and_bounded`: it moves the weights, each 64-block error is at most 0.16·absmax, and a
  second trip is bit-identical. A planted no-op round trip turns it RED.
- **Finding:** the tree has three NF4 quantizers, and they do not agree on nibble order: aprender-gpu `kernels::quantize_nf4`
  (low nibble first), trueno `brick::quant_ops::nf4` (high first) and a private copy in aprender-train `wgpu_nf4.rs`. The CPU
  `QLoRALayer` (`lora/qlora.rs`) uses symmetric int4 (`quant4bit`), which is not NF4. QQE-005 must name the CUDA path's quantizer.

### Spike S-R15 — is CUDA LoRA (the T2 cell) a routing fix or new code? · `[V]` (2026-09-27, desk read at shaping @5826e292bb)
- **Question:** `-m lora --gpu-backend cuda` prints a warning and trains on the CPU (`finetune.rs:280`). Would setting the CUDA init
  gate for plain LoRA be enough?
- **Answer: no. Flipping the gate alone would train nothing on the GPU, silently.**
  - `InstructPipeline` calls `init_cuda` only when `quantize_nf4` is set (`instruct_pipeline/constructors.rs:56,173,293`).
    `init_cuda` has an FP32 branch (`CudaTransformerBlock`, `cuda_init.rs:304`), but the LoRA grad workspace and optimizer states are
    created only for NF4 (`cuda_init.rs:66`), and the block backward runs only for NF4 (`training.rs:309`). With the gate flipped,
    LoRA would get a CUDA forward and no adapter gradient, and the loss would stay flat.
  - The FP32 `CudaTransformerBlock::backward`/`optimizer_step` (`transformer/cuda_block.rs:1276,2248`) is full-weight fine-tuning
    (`cuda_trainer.rs:868`), not LoRA.
  - **The CUDA LoRA path adapts Q and V only:** `constructors.rs:383,397` build two adapters per layer, and
    `CudaLoraGradWorkspace` (`cuda_block.rs:3853`) holds `A_q, B_q, A_v, B_v` plus two norm grads. The T2 cell needs Unsloth's
    seven targets (q, k, v, o, gate, up, down), and R4 needs those plus the GDN projections (Qwen35Lora: 150 slots at 0.8B).
  - The pipeline's `Transformer` has no GDN layer, so Qwen3.5 on CUDA also needs R2/R3's GDN forward and backward as CUDA blocks.
- **Consequences:**
  - R15 is new CUDA work: a non-quantized (bf16) frozen-base variant of the NF4 block's LoRA backward, extended from 2 to 7 (T2)
    and 10 (R4) target kinds. It is not a one-line gate. Its K̂ should be re-sized, and R4 shares the same target extension.
  - `beat-unsloth-finetune-throughput-v1` already requires the same targets on both sides (its planted half-targets
    row), so today's Q+V path would correctly fail SAME-WORK.
  - QQE-005 would have passed vacuously on a Q+V-only CUDA side ("every tensor it emits" matches). The contract now requires
    the same 300-tensor adapter set, with a planted drop-to-Q/V row.
  - R15's own falsifier, for whoever takes it: after N CUDA steps of `-m lora`, the adapters must have changed and the loss
    must have moved. A CUDA forward with frozen adapters is RED.

## §3 Remaining ranked rows (R6–R20)
See the L2 handoff (`docs/lookahead/0.72.md` once LA-00 lands). In brief: R6 distill 27B→4B at batch > 1 · R7 merge cells ·
R8 quantize policy for GDN tensors · R9 #4418 (0.71 dependency) · R10 T5 round-trip gate (none exists `[V]`) ·
R11 B2 wiring (#4367) · R12 training receipts on APR-OBS identity · R13 HF rc · R14 throughput work · R15 LoRA on CUDA ·
R16 dangling `qlora-training-loop-v1` · R17 4B memory plan · R18 teacher/student vocab alignment · R19 ROADMAP PMAT-711 stale ·
R20 declarative recipe (only if pulled from E8 0.75).

## §4 Rulings requested (S-4)
RQ-1 epic #4000 body is still "Agent Ready" · RQ-2 T1 scope = GDN training · RQ-3 T2 vs E8 0.75 (#4002) overlap ·
RQ-4 #4418 stays in 0.71. Full text is in the handoff file.
