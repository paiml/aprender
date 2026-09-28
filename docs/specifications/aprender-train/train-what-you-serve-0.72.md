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

### Spike S-R10 — does a Qwen3.5 model round-trip between .apr, safetensors and GGUF today? · `[V]` (2026-09-28, measured, CPU)
- **Setup:** apr 0.69.3 @574583d382, built in the private target dir. Qwen3.5-0.8B from HF (`Qwen/Qwen3.5-0.8B`, bf16, one
  shard behind `model.safetensors.index.json`) and the local Qwen3.5 GGUFs. Contract `qwen35-format-roundtrip-v1` (QFR-001..005).
- **safetensors → .apr: works.** Importing the directory gives 489 tensors (488 plus a materialised `lm_head`), including
  144 visual and 11 MTP tensors. `apr import hf://Qwen/Qwen3.5-0.8B` fails with a 404: it asks for `model.safetensors`, and the
  repo ships a single-shard index.
- **.apr → safetensors → .apr: bit-identical, with one hole.** `apr diff --values` reports 489/489 identical and max diff 0.
  The export keeps HF names but widens bf16 to F32 (2× size) and writes no config. On its own the output does not re-import
  (it is detected as Qwen2, overridden to Qwen3, and fails the GH-279 completeness gate, rc 5). It works only once the source
  `config.json` is copied beside it (QFR-003).
- **.apr → GGUF: exits 0 with 488/489 tensors unmapped.** Each tensor is "passing through" under its HF name
  (`model.language_model.…`), so llama.cpp cannot load the file. This is the #4418 class at 0.8B (738/739 at 4B), and rc 0
  on an unloadable file is its own defect (QFR-004). `apr diff` of .apr vs that GGUF printed nothing in 17 minutes and was stopped.
- **GGUF → .apr: refused for every real Qwen3.5 GGUF** (0.8B Q4_K_M, 0.8B IQ4_XS, 2B, 4B; rc 5). The reader's config
  allowlist (`gguf/reader_parsing.rs:93`) has `qwen3.` but not `qwen35.`, so `hidden_dim` reads as 0. **That is deliberate**:
  `api_tests_all_keys_3733.rs` pins `qwen35.*` off the config accessors, because tensor evidence relabels the file as Qwen3
  (`[ARCH-EVIDENCE] Override … → Qwen3`) and the importer has no qwen35 name map. Widening the allowlist would import a hybrid
  model under the Qwen3 map and write a silently wrong .apr. The defect is the message: it says "This GGUF file may be malformed"
  instead of naming qwen35 as unsupported (QFR-005).
- **Consequences:**
  - The T5 gate can arm the safetensors ↔ .apr leg now (QFR-001/002, CPU, 0.8B). The GGUF legs wait for #4418 (export) plus a
    qwen35 GGUF import path. **That import path is not a row today**; it belongs with R9 and is at least the size of #4418.
  - R13 (HF rc publish) needs QFR-003: a published safetensors must re-import without a hand-copied config.
  - Until R10 lands, T5 evidence means "HF safetensors → .apr → safetensors", never "GGUF round-trip".

### Spike S-R6 — is CUDA distill's batch = 1 a hard limit, and what does it do at batch > 1 today? · `[V]` (2026-09-28, desk read at origin/main aca6f2d7f6)
- **Setup:** desk read of `apr distill --backend cuda` (`crates/apr-cli/src/commands/distill.rs` `run_cuda_backend`), the
  pipeline (`aprender-train-distill/src/pipeline.rs`), `kd_step.rs` and `CudaStudentProvider` (`student_provider.rs`).
  Contract `distill-batch-honesty-v1` (DBH-001..005).
- **batch > 1 is not refused; it silently trains on 1 of B rows.** The CLI default is `--batch-size 16` and the only
  check is `> 0`. The pipeline draws B rows, runs the teacher and the student on every row, and `kd_step` returns B
  unscaled gradient rows and a loss averaged over B. `CudaStudentProvider` then caches only `input_ids.last()`, and
  `apply_kd_gradient` runs `forward_backward_with_grad` on `gradient.last()` alone. So at the default, 15 of 16 rows are
  thrown away after their forward pass has been paid for, and the logged loss describes 16 rows that were never trained on.
  The learning rate is not diluted, because the rows are not pre-divided by B. The effective batch is simply 1.
- **The per-position path is worse.** `apply_kd_gradient_per_position` flattens to `[B·P][vocab]` and calls
  `apply_kd_gradient`, whose trait doc says it "averages them"; the CUDA override keeps one row of B·P.
- **It is a known limit that nothing enforces.** The code comments say "Phase 2d … batch_size=1 assumption" and "Future
  Phase 2e fuses input_ids + gradient", but no refusal, warning or contract pins it
  (`apr-distill-teacher-backend-selection-v1` even sizes the teacher at batch 32).
- **qwen35 on this path:** `TransformerConfig::from_apr_metadata` matches `architecture.starts_with("qwen3")`, so a
  `qwen3_5` .apr (the HF import stamps `qwen3_5` / `Qwen3_5ForConditionalGeneration`, S-R10) builds a dense Qwen3 config
  with no GDN. On main nothing refuses it. R1 (`la-72/4552-train-arch-honesty`, TAH-001) makes `apr distill` refuse it on the
  teacher and the student before any read. R6 on qwen35 therefore needs R1 and the CUDA GDN trainer (S-R15), in that order.
- **Consequences:**
  - The immediate fix is small and belongs before any R6 run: refuse `--backend cuda` with batch > 1 by name (DBH-002),
    or accumulate every row (DBH-001). Either turns DBH-003 green; the current code is its planted RED.
  - Real batch > 1 needs Phase 2e: a fused `(input_ids, gradient)` step on the trainer, sized with R6.
  - Any CUDA distill loss curve recorded so far at batch > 1 measured effective batch 1. Treat those receipts as batch 1.

### Spike S-R7 — does `apr merge` keep a Qwen3.5 model a Qwen3.5 model? · `[V]` (2026-09-28, measured, CPU)
- **Setup:** apr 0.69.3 @574583d382 (private target dir). Inputs are the S-R10 Qwen3.5-0.8B .apr (489 tensors, bf16,
  2.77 GB) and its bit-identical round-trip copy. Every strategy was run on that identity case: average, weighted 0.7/0.3,
  slerp, and ties/dare with `--base-model` = the input. Each output was compared with `apr diff --values --limit 1000`
  and `apr inspect`. Contract `merge-output-fidelity-v1` (MOF-001..005).
- **Values: correct on the identity case.** All five strategies give 489/489 identical, with no NaN from slerp at angle 0
  and no drift from ties/dare on a zero task vector. Wall time was 127–472 s for 0.8B on CPU.
- **Container: wrong for every strategy.** `-o merged.apr` writes a **SafeTensors** file. The first 8 bytes are a header
  length (`23 eb 00 …`), not `APR\0`. The file has no metadata at all: `architecture`, `hf_architecture`, `model_type` and
  `rope_theta` are gone, and `apr inspect` guesses `llama`. Tensors are widened from bf16 to F32 (4.51 GB). The writer is
  `save_safetensors(output_path, …)` (`converter/ties_merge.rs:306`; `merge.rs` imports the same writer), so this is not
  qwen35-specific. It holds for every architecture.
- **Why it blocks R7:** a merged Qwen3.5 cannot be served, trained or re-imported as Qwen3.5. S-R10 measured that a bare
  safetensors without `config.json` re-imports as Qwen2→Qwen3 and fails the GH-279 gate (QFR-003). The hybrid GDN layout
  survives only as tensor names, and nothing downstream reads it without the config.
- **Not covered by this spike:** distinct inputs. The identity case cannot tell whether weights are normalised or whether
  ties trims by the right density. `lora-merge-forward-equivalence-v1` is pure LoRA algebra with no architecture cell. R7's
  forward-equivalence cell needs a second Qwen3.5 checkpoint (a LoRA-merged 0.8B from R4, or the Base variant).
- **Consequences:**
  - Merge must write APR v2 when the output is `.apr` and carry the first input's metadata (MOF-002/003), and it must keep
    the input dtype unless asked to widen (MOF-004). This comes before any R7 merge receipt, and it is small.
  - Until then, any "merged model" receipt is a safetensors file that only loads with a hand-copied config.

### Spike S-R12 — what does a training run record about itself today? · `[V]` (2026-09-28, desk read at origin/main aca6f2d7f6)
- **Setup:** desk read of the writers behind `apr finetune` (instruct and classify), `apr distill`, `apr pretrain` and
  `apr train`, checked against nine identity fields. Pivotal citations were re-read by hand. Contract `train-run-receipt-v1`
  (TRR-001..006). APR-OBS has no schema in aprender or infra (aprender-84, #4484), so this contract proposes one.
- **No training output records the binary.** The apr version and git sha are absent from every writer. `APR_GIT_SHA`
  is built in (`aprender-build-sha`), and `apr --version`, `bench` and `serve` use it; the `aprender-train-*` binaries use
  it only in `--version`. The one near miss is the classify checkpoint's `provenance.tool`, which carries the train
  crate's version, not apr's, and no sha.
- **Seed: never persisted.** finetune hardcodes `seed: 42` (`finetune.rs:474`, and again for classify). distill has a
  config seed that `training_metadata.json` does not write. pretrain prints its seed to stdout only.
- **Recipe: copied, never hashed.** distill copies a few hyperparameters in plain text. No command hashes the effective
  config after defaults, so a changed default is invisible.
- **Data: hashed in finetune only, and over parsed samples.** Instruct hashes the parsed samples, and classify hashes the
  sorted (input, label) pairs, which ignores order. distill and pretrain record no data hash.
- **Models: paths, never hashes.** The base, teacher and student are recorded as paths or ids. `apr-checkpoint-v1`
  F-CKPT-017 defines a canonical `base_model_hash`, and `TeacherProvenance.hash` exists in apr-format. **No writer emits
  either.**
- **Device and timestamps:** printed, rarely saved. `--backend cuda` distill prints its receipt JSON to stdout only.
  classify's `training_state.json` is the richest record, but it is a live monitor file, not a receipt.
- **Consequences:**
  - Every T2–T4 receipt (R4 QLoRA, R6 distill, R13 publish) would today be unattributable: nothing ties an output to the
    binary, recipe, data or base that produced it. R12 is a prerequisite of their evidence, not a nice-to-have.
  - The fix is one shared writer: a `train_receipt.json` beside every training output with the nine fields, called by all
    four commands. It reuses `APR_GIT_SHA`, the F-CKPT-017 hash and a sha256 of the data file bytes.
  - Decision for R12's owner: whether the data hash covers file bytes (proposed, since it detects a reorder) or parsed
    samples (today's finetune). The contract requires the byte hash and allows the sample hash as an extra key.

### Spike S-R13 — would `apr publish` put a loadable, receipted Qwen3.5 model on HF today? · `[V]` (2026-09-28, measured, `--dry-run --offline`, no upload)
- **Setup:** apr 0.69.3 @574583d382, run with HF tokens unset. Three input directories: the trained-style `.apr` alone
  (the S-R10 import), the bare safetensors export from S-R10, and the original HF source (weights, `config.json`,
  both tokenizer files and the index) as a control. Code read at origin/main aca6f2d7f6 (`commands/publish.rs`).
  Contract `hf-rc-publish-v1` (HRP-001..005).
- **The dry-run plan is not the upload.** For the HF source, `--dry-run` and `--dry-run --json` list one file, the
  weights. The real path (`upload_to_hub_extended`) also sends `find_companion_files`: 3 files here (`config.json` and
  both tokenizer files). They are printed only under `-v` (`publish.rs:482`), and `build_dry_run_plan` never receives
  them (`:529` passes `extra_files`). A reviewer who approves the plan approves a different upload.
- **A trained output publishes weights only.** The `.apr`-only and bare-safetensors directories plan exactly one file.
  An `.apr` alone is not loadable by HF tooling, and the bare export has no `config.json`, which S-R10 found it never
  writes (QFR-003). No path takes a trained `.apr` to a loadable HF repo without hand-copying the config and tokenizer.
  `model.safetensors.index.json` is not a companion; single-shard sources are covered by the `model.safetensors` alias,
  but multi-shard ones would not be (inferred).
- **The generated card is wrong for a Qwen3.5 derivative.** It says `license: mit` by default (Qwen3.5 is Apache-2.0),
  lists `accuracy: N/A` as a metric, has no `base_model`, no receipt, and no git sha (version only). Its usage snippet
  says `Model::load("model.apr")` even when the upload is safetensors-only.
- **Consequences:**
  - R13 (EXT-001 rc publish) needs HRP-001 first: the plan is what a quorum reviews, so it must equal the upload.
  - A trained model needs a publish directory builder: `.apr` → `model.safetensors`, plus the source `config.json`,
    tokenizer and chat template, plus the R12 receipt. That builder is the same fix as QFR-003.
  - The license must come from the base model and must never default for a derivative (HRP-003). Publishing an
    Apache-2.0 derivative under MIT is a licence error, not a style issue.

### Spike S-R11b — can a sealed test item reach an apr training run today? · `[V]` (2026-09-28, desk, origin/main aca6f2d7f6 + origin/rex/001-prm-s1-v2 e1bfade985)
- **Question:** R11 says "0 sealed-test hashes in train data". Does any `apr` training path refuse sealed eval items?
  The verb-refusal half of R11 is aprender-ont's (#3597); this spike covers only data ingress.
  Contract `train-ingress-sealed-refusal-v1` (TIS-001..005).
- **Answer: no, and there is nothing yet for a trainer to consult.**
  - **Trainers:** none of the loaders checks a sealed set. Loaders read: `finetune.rs:504` and
    `instruct_corpus.rs:92` `load_instruct_corpus`, `distill.rs:2304` `read_prompts_jsonl`, `corpus.rs:91` `load_jsonl`,
    `classification.rs:337/393`, `shard_reader.rs:25` (pretrain). Any JSONL given on the command line is trained on.
  - **The sealed set:** `trace-split-guard-v1` defines one (FALSIFY-TSG-004: "a renamed, re-indented sealed diff puts its
    whole component in `sealed`"), and `trace-dedup-v1` defines the near-duplicate clusters it relies on. Both exist
    only as contracts on the unmerged `rex/001*` branches. No crate on either branch implements the `sealed` split.
    The PRM ledger lists C7–C9 as todo (owner aprender-cb).
  - **Existing tools only report:** `apr data decontaminate` (`data.rs:880`, calls `check_contamination` at `:916`)
    returns PASS/FAIL/VACUOUS and rewrites nothing. `apr eval` contamination (`eval/mod.rs:469`, 10-gram overlap at
    `:516`) reports only. Neither is run by a trainer.
  - **Nearby contracts cover other questions:** `crux-B-07` FALSIFY-004 is calibration vs eval (it has Rust tests).
    `apr-data-pipeline-v1` ADP-DET-002, named "No cross-contamination", formally states split determinism, not
    disjointness, and its Lean proof is `sorry`.
- **Consequences:**
  - A TSG split, even once implemented, only helps if the run cannot bypass it. The guard has to sit at trainer
    ingress (finetune, distill prompts, pretrain shards): hash plus near-duplicate against a sealed manifest, refuse and
    name the item, and record the manifest sha in the R12 receipt (TRR). Otherwise "0 sealed hashes in train data" is
    a claim nobody can check.
  - The ingress check should reuse TDD normalisation (re-path, re-indent, rename), not the exact sha. S-R11b
    inference: an exact-sha check misses the perturbations TSG-004 names.
  - Sequencing: TIS needs TSG and TDD merged, or at least their normaliser. Until then, TIS-005 (refuse when no
    manifest is supplied for an rc-bound run) is the only part that can be met.

## §3 Ranking v2, after the spikes (2026-09-28)
This replaces the pre-spike order. Rows move for three reasons:
- **Evidence voids:** a row whose absence makes other rows' receipts worthless moves up. That is R12, and the
  honesty refusals (DBH-001, MOF-002, HRP-001).
- **Sizing:** S-R15 made R15 new kernel-side work, not a flag change.
- **Ownership:** rows already held on a branch drop out of L2's queue.

K̂ is minutes of worker time; `[A]` is an assumption, the rest are carried from §2.

| # | Row | Changed by | Cell to build first | K̂ | State |
|---|---|---|---|---|---|
| 1 | R1 honesty gate | S-R6 (distill takes the dense config for qwen3_5) | TAH-001..003 | 30 | branches `la-72/4552-train-arch-honesty`, `tah-003-head-dim` |
| 2 | R12 training receipts | S-R12: 0/4 verbs record sha, recipe, model hash or seed | one shared `train_receipt.json` writer, TRR-001..006 | 45 `[A]` | contract only |
| 3 | R2 GDN forward | — | QTG-001 parity vs serve | 90 | branches `r2-gdn-forward`, `r2-gated-attn`, `r2-qwen35-model` |
| 4 | R3 GDN backward | — | QTG-003/006 gradcheck | 120 | branch `r3-backward` (dev-dep `features=["cuda"]` note) |
| 5 | R15 CUDA LoRA | S-R15: NF4-only backward, Q+V adapters only | LoRA grad workspace for non-NF4, all projections | 120 `[A]`, re-size before R4 | contract only |
| 6 | R4 QLoRA 4B end to end | S-R17 (fits 24 GB only without `dW'`), S-R4c | QQE-001..006 | 90 | blocked on R2, R3, R15 |
| 7 | R5 Unsloth harness | S-R5 | Unsloth side runs now | 60 | parallel to R2–R4 |
| 8 | R10 round-trip | S-R10: st↔apr bit-identical, but the export has no config (QFR-003); GGUF legs wait on #4418 | QFR-003 self-describing export | 40 `[A]` | contract only |
| 9 | R13 HF rc publish | S-R13: plan ≠ upload, weights only, license mit | HRP-001 (small), then the publish directory builder (= the QFR-003 fix) | 45 `[A]` | needs R10, R12 |
| 10 | R11 sealed ingress | S-R11b: no loader checks; TSG/TDD unmerged | TIS-001/002 at the three loaders | 40 `[A]` + the TDD normaliser (aprender-cb) | verb half: aprender-ont #3597 |
| 11 | R6 distill batch | S-R6: batch B > 1 silently trains the last row | DBH refusal first (15 `[A]`), then real batching (90 `[A]`) | 15 + 90 | contract only |
| 12 | R7 merge | S-R7: `-o *.apr` writes metadata-free F32 safetensors | MOF-002/003 APR writer | 30 `[A]` | contract only |
| 13 | R9 #4418 GGUF name map | still OPEN in 0.71 at 2026-09-28 | — | owned by 0.71 | if it slips, R10's GGUF legs and T4/T5 slip with it |
| 14 | R8 GDN quantize policy | — | — | — | branch `79/r8-gdn-quant-policy` |
| 15 | R17 memory, measured | S-R17 desk plan | peak-memory run on the 4090 | 30 `[A]` | needs GPU; after train-active clears |
| 16 | R18 vocab alignment | — | — | — | branch `76/0.72-r18-vocab-cell` |
| 17 | R16 dangling `qlora-training-loop-v1` | — | — | — | branch `la/r16-qlora-loop-contract` |
| 18 | R19 ROADMAP PMAT-711 stale | done | — | — | shaping @378ec8e920 |
| 19 | R14 throughput work | — | sized from the R5 baseline | — | after R5 |
| 20 | R20 declarative recipe | — | — | — | only if pulled from E8 0.75 (RQ-3) |

**Critical path v2:**
```
R1 ─► R2 ─► R3 ─┐
R15 ────────────┼─► R4 ─┬─► R5 baseline ─► R14
R12 receipts ───┘       ├─► R6 (DBH refusal lands with R1), R7 (MOF)
                        └─► R13 HF rc ◄── R10 QFR-003 export ◄── #4418 (GGUF legs only)
R11 TIS ◄── TDD normaliser (PRM C7–C9) ─────► gates every R4/R6 run counted for 0.72
```

**Cheap refusals first:** DBH-001 (distill B > 1), MOF-002 (merge writes APR), HRP-001 (plan = upload) and TIS-005 (no
manifest, no rc run) are each ≤ 15 `[A]`. Each turns a silent wrong answer into a named refusal, and none depends on
GDN work. They are the best 0.72 value per minute before R2 lands.

## §4 Rulings requested (S-4)
RQ-1 epic #4000 body is still "Agent Ready" · RQ-2 T1 scope = GDN training · RQ-3 T2 vs E8 0.75 (#4002) overlap ·
RQ-4 #4418 stays in 0.71. Full text is in the handoff file.
