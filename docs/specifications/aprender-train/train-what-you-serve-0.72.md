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
This is the shaping-time path. The current one is §3's critical path v3 (2026-10-03): R21 (GDN on CUDA) follows R3 and
feeds both R4 and T2, and so does row 4b (the HF-convention loader, on #4418's transforms). R15a and R12 join before
R4, and T2 needs R15b.

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
- **Served side and merged file `[V]` (read at `316dee2cd4`):**
  - apr serves Qwen3.5 only from a qwen35 GGUF. `run`, `chat` and `serve` load no adapter; a Modelfile `ADAPTER` line
    is printed (`modelfile/mod.rs:122`) and never applied. QQE-003's served side is therefore the merged model exported
    as a GGUF, which needs #4418 (row 14).
  - The merged file comes from `apr finetune merge` (`finetune_display_next_validate.rs:469`). It works on files: each
    adapter pair is added to the base tensor whose name it matches (`adapter_pair_names`, :213), and every other tensor
    is copied. The file keeps the base's HF conventions, provided the adapter is in HF order (QQE-007).
  - The trainer's own merge (`Qwen35Lora::merge_into`) is in memory and in GGUF conventions. Nothing writes it, and
    nothing should: those values under HF names would make the GGUF export apply #4418's transforms a second time.
  - Three wrong merges are written with exit 0, and `verify_merged_runnable` passes each:
    - GDN targets named the GGUF way (`attn_qkv`, `attn_gate`, `ssm_out`) beside HF-named attention and MLP targets.
      `gguf_proj_to_hf` maps only the seven llama projections, so the GDN deltas are dropped.
    - A PEFT-named adapter (`lora_A.weight`) matches nothing, and the zero-match refusal counts only `.lora_a` names,
      so the output is the base.
    - A safetensors adapter with no `lora_alpha` in its header or sidecar is merged at alpha 16. At S-R4a's r16 and
      alpha 32, that halves the delta.
  - QQE-003's cosine of 0.999 on logits may not see a LoRA delta the merge dropped. QQE-009 (PROPOSED) checks the merge
    tensor by tensor and plants all three.
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
- **Spike S-R5 (desk, 2026-09-27) `[V, external]`:** Unsloth supports Qwen3.5 fine-tuning (0.8B–122B), and needs transformers v5.
  *(Corrected 2026-09-29: this spike said "with its own Triton kernels for the GDN layers". The GDN fast path is fla plus
  torch.compile, in draft PR #10744. See the External reference study, U4.)* It **advises against QLoRA on Qwen3.5** ("higher than normal
  quantization differences"). Its default targets are `q,k,v,o,gate,up,down`, with no GDN projections. It lists 10 GB
  for 4B bf16 LoRA. Source: unsloth.ai/docs/models/qwen3.5/fine-tune. **Therefore the T2 cell is bf16 LoRA, not QLoRA**,
  with Unsloth's default targets on both sides. The contract is updated to match.
- **Consequence for the ranking:** R15 ("`-m lora` is CPU F32 today", `finetune.rs:280` on main aca6f2d7f6, re-measured by R19) moves onto the critical path,
  because bf16 LoRA on CUDA IS the T2 cell. R4 stays the T1 finetune cell, but NF4 quality on Qwen3.5 is now a known risk
  (K10). R4 adds a gate: QLoRA's final loss is within 5% of bf16 LoRA's on the same cell, or QLoRA is documented as
  unsupported for qwen3.5 (an honest refusal, not a silent quality loss). *(Amended 2026-10-03, qwen35-qlora-e2e-v1
  1.1.0: the LoRA reference may be bf16 or fp32, as recorded in its receipt, and must run on the same GPU as the QLoRA
  side. At `316dee2cd4`, `-m lora` trains on the CPU, so the reference needs R15a C4.)*

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

### External reference study — Unsloth, for T2 / R5 / R14 · `[V, external]` (2026-09-29, desk, no code copied)
L2 deliverable 2. Each fact carries its source; `[U]` marks one that was inferred or seen only in a search snippet.

| # | Fact | Tag | Source |
|---|---|---|---|
| U1 | Unsloth fine-tunes Qwen3.5 0.8B–122B, including 4B. It needs transformers v5 | `[V]` | unsloth.ai/docs/models/qwen3.5/fine-tune |
| U2 | "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models". bf16 LoRA on 4B is listed at about 10 GB | `[V]` | same page |
| U3 | Default LoRA targets are `q,k,v,o,gate,up,down` only. The GDN `in_proj_*`, `out_proj` and `conv1d` are not targeted | `[V]` | same page |
| U4 | **The Qwen3.5 GDN fast path is not a merged Unsloth kernel.** Draft PR #10744 torch-compiles the GDN eager ops and reuses the fla Triton kernels (`compile_fla_no_autotune` from unsloth_zoo). `causal_conv1d` is optional (falls back). The PR is still a draft | `[V]` | github.com/unslothai/unsloth/pull/10744 |
| U5 | Without fla and causal-conv1d, HF's GDN runs an fp32 torch chunk loop that takes about half the step time | `[U]` snippet | github.com/huggingface/transformers/issues/48718 |
| U6 | Kernels in `unsloth/kernels/`: `fast_lora`, `cross_entropy_loss`, `rope_embedding`, `rms_layernorm`, `swiglu`, `int4_packed`, `fp8`, `moe/`. None is GDN-specific | `[V]` names, `[U]` purposes | GitHub contents API, unsloth/kernels |
| U7 | Headline claims are 2× faster with 70% less VRAM versus HF+FA2 (Llama 8B/70B, H100/Blackwell, QLoRA, r32, batch 2, grad-accum 4). The packing blog reports 1.7–3× tokens/s, and part of that is padding removal. No Qwen3.5 or RTX 4090 number is published | `[V]` | README; docs/basics/unsloth-benchmarks; docs/blog/3x-faster-training-packing |
| U8 | The only Qwen3.5 speed number: Qwen3.5-9B on B200, 829 → 662 ms/step. That is Unsloth versus Unsloth (#10744), not versus HF | `[V]` | PR #10744 |
| U9 | Its benchmarks use `adamw_8bit`, gradient checkpointing `"unsloth"` (activations offloaded to CPU RAM), and about 5 min of torch.compile warmup. Sequences are padded to max length, so padding likely counts in its tokens/s | `[V]`; padding counting `[U]` | docs/basics/unsloth-benchmarks; lora-hyperparameters-guide |
| U10 | The current release is unsloth 2026.9.12 (PyPI, 2026-09-28). Pins: `transformers>=4.51.3,<=5.5.0`, `bitsandbytes>=0.45.5`, `trl>=0.18.2,<=0.24.0`, `peft>=0.18.0`, `triton>=3.0.0`. torch is unpinned | `[V]` | pypi.org/project/unsloth; pyproject.toml on main |

**Consequences (all go into `beat-unsloth-finetune-throughput-v1` 1.1.0):**
- **The incumbent must run its fast path.** U4 and U5 mean an Unsloth install without fla could run about 2× slower on GDN. A ratio measured against that would be a win over a crippled incumbent. The harness records unsloth, unsloth_zoo, fla and causal-conv1d versions. If fla is missing, it refuses with `INCUMBENT_SLOW_PATH` and reports no ratio. This is a new planted falsifier.
- **Same work means the same optimizer, checkpointing and token count.**
  - The optimizer is torch AdamW with fp32 states on both sides; `adamw_8bit` is never used.
  - Gradient checkpointing is off on both sides.
  - Packing is off on both sides.
  - Tokens/s counts non-pad label tokens only, on both sides. U7 and U9 show that padding and packing alone move Unsloth's own numbers by up to 2×.
- **The timed window starts after compile.** The 50 warmup steps are kept. The receipt records when Unsloth's torch.compile finished. A timed window that overlaps compilation is not measured.
- **S-R5's "own Triton kernels for the GDN layers" is corrected by U4**, and the contract text is corrected with it. R14's gap analysis compares against fla's chunked GDN, not an Unsloth GDN kernel.
- The Unsloth versions are pinned from U10 at harness time. U10 is the 2026-09-29 reading, not the lock.

## §3 Ranking v3 (2026-10-03)
v3 replaces v2 (2026-09-28, at `363f9ca810`). Rows still move for v2's three reasons: an evidence void moves a row
up, a new size changes its place, and a row already held on a branch drops out of L2's queue. Changes since v2:
- **R15 is re-sized from 120 to 450 `[A]`** (`r15-cuda-lora-cells.md`; C5 was sized from the code at 150) and split
  by consumer:
  - **R15a** is cells C1–C4 (210). It is on R4's path, because QQE-004's reference run is `-m lora` on CUDA
    (qwen35-qlora-e2e-v1 1.1.0).
  - **R15b** is cells C5–C7 (240). Only T2 needs it. Its largest cell, C5 bf16 (150), is the subject of RQ-5 (§4).
- **R12 has an owner.** la-impl holds `la/r12-train-receipt` @68747b344e, which has the writer and the `apr pretrain`
  wiring, built on TRR 1.0.0. The TRR 1.1.0 delta (device object, recipe with compute_dtype, timed window) is on
  `la-72/r15-receipt-ext`, and la-impl has been told.
- **R5 is done on the desk.** Verdict, orchestrator, incumbent side, pinned data and apr adapter are stacked from
  `r5-verdict` to `r5-apr-adapter` @7aeb557271. What is left is GPU time, plus the apr side, which is R15b.
- **The cheap refusals sit on fold branches:** DBH-001, MOF-002, HRP-001 and TIS-005, on `fold-dbh-a`/`-b`,
  `fold-mof`, `fold-hrp` and `fold-tis`. Each is ≤ 15 `[A]`, turns a silent wrong answer into a named refusal, and
  needs no GDN work. They are the first PRs to open after LIVE 0.70.1.
- **R20 is out of 0.72.** RQ-3 was ruled 2026-09-27: #4002 (E8, 0.75) keeps everything beyond GDN training.
- **R21, GDN on CUDA, is new.** T2, R4 and R6 on qwen35 train Qwen3.5 on CUDA, and 24 of the 4B's 32 layers are
  GDN. R15's cells are dense only, R2/R3's GDN is CPU only (`fold-r2r3`; its four GDN files never mention CUDA), and
  serving's CUDA GDN is forward only. S-R15 named the need at shaping, but neither v2 nor v3 sized a row for it.
  Spike S-R21 (2026-10-03, `r21-cuda-qwen35-hybrid-block.md`) sized it at 400 `[A]` (325–465) in f32, plus 25
  `[A]` for bf16 on T2's path. It also widened the row to the whole hybrid block: the CUDA trainer has neither the
  attention output gate nor the partial RoPE that the 4B's 8 full-attention layers need. Its falsifiers are
  `qwen35-train-cuda-v1` QTC-001..005. The spike also found that GGUF and HF order Qwen3.5's value heads
  differently and that nothing on main converts between them. #4418's branch has the converter. `apr finetune`
  trains from .apr only and `apr import` refuses every real Qwen3.5 GGUF, so R4's and T2's bases are HF-sourced and
  need a load-time conversion (row 4b, QQE-008). R4's PEFT export also needs the permutation back (QQE-007).

State is read from the branch tips on 2026-10-03. origin/main is `316dee2cd4` and no la-72 branch has landed. K̂ is
minutes of worker time still left; `[A]` marks an assumption.

| # | Row | Next cell | K̂ left | State |
|---|---|---|---|---|
| 1 | R1 honesty gate | TAH-001..003 | 30 | `la-72/4552-train-arch-honesty` @1fda81ad6c, `tah-003-head-dim` @57ada988a7 |
| 2 | R12 training receipts | TRR 1.1.0 fields on the shared writer | 45 `[A]` | la-impl, `la/r12-train-receipt` @68747b344e (TRR 1.0.0) |
| 3 | R2 GDN forward | QTG-001 parity vs serve | 90 | `fold-r2r3` @5a837dfa3b; one-line Cargo.toml conflict with main |
| 4 | R3 GDN backward | QTG-003/006 gradcheck | 120 | in `fold-r2r3` (`r3-backward` @8a9f4f0104) |
| 4a | R21 GDN on CUDA (the hybrid block) | QTC-001 training forward = the R2 CPU forward | 400 `[A]`, +25 bf16 on T2's path | S-R21 desk spike done 2026-10-03 (`r21-cuda-qwen35-hybrid-block.md`); no branch; oracle `fold-r2r3`; LoRA wiring needs R15a's C1; cargo after LIVE 0.70.1 |
| 4b | HF-convention loader | QQE-008: the HF copy's step-1 loss = the GGUF copy's, or a refusal | 25 `[A]`, +20 if #4418 has not landed | new 2026-10-03, no branch; calls #4418's `transform_qwen35_tensor` (`m0694/4418-qwen35-gguf-main` @1af0e3cc11); on R4's path and T2's; cargo after LIVE 0.70.1 |
| 5 | R15a CUDA LoRA, C1–C4 | C1 `lora_backward` extraction (FALSIFY-LORA_GRADIENT_FLOW_V1_004) | 210 `[A]` | cells and falsifiers on `la-72/r15-receipt-ext`; cargo after LIVE 0.70.1 |
| 6 | R4 QLoRA 4B end to end | QQE-001..007, QQE-009 | 90 + 15 + 10 `[A]` | blocked on R2, R3, R21, R15a, 4b; QQE-003 also needs #4418 (row 14), because apr serves Qwen3.5 only from a GGUF and loads no adapter; contract 1.1.0 pins the QQE-004 reference's precision and device, adds QQE-007, the export permutation (the +15), and proposes QQE-009, a strict `apr finetune merge` (the +10) |
| 7 | R5 Unsloth harness | Unsloth-side runs (`r5-unsloth-ft-runbook.md`) | GPU only | desk-done, `r5-apr-adapter` @7aeb557271; needs train-idle |
| 8 | R15b CUDA LoRA, C5–C7 | C6 flags and receipt (with R12), C7 timed window; C5 bf16 per RQ-5 | 240 `[A]` | contracts apr-finetune-canonical-task-v1 1.1.0, TRR 1.1.0 |
| 9 | R10 round-trip | QFR-003 self-describing export | 40 `[A]` | `fold-r10-qfr` @6b13da980f (QFR-001/002/003/005); QFR-004 is #4418 |
| 10 | R13 HF rc publish | HRP-001 plan = upload | 45 `[A]` | `fold-hrp` @6c93634c3c; header reader `amr-on-4607` @d34ce7cacd waits on #4607 |
| 11 | R11 sealed ingress | TIS-002 planted perturbed item | 40 `[A]` + the TDD normaliser | `fold-tis` @9fed5c27db (TIS-001/003/004/005); TIS-002 waits on a foreign branch |
| 12 | R6 distill batch | DBH refusal, then real batching | 15 + 90 | `fold-dbh-a` @2cfe655b98, `fold-dbh-b` @df5e4d74f6, GPU halves `gpu-falsifiers` @e890928f4e; 3 Definition-of-Ready tests, plus R18's 3 |
| 13 | R7 merge | MOF-002/003 APR writer | 30 `[A]` | `fold-mof` @17fbe022eb; Definition of Ready: FWD-001's path, FWD-002..004 unwritten |
| 14 | R9 #4418 GGUF name map | — | owned by 0.71 | fix not on main at `316dee2cd4`; if it slips, R10's GGUF legs and T4/T5 slip with it, R4's QQE-003 has no served side, and row 4b copies its transforms (+20 `[A]`) |
| 15 | R8 GDN quantize policy | — | — | `79/r8-gdn-quant-policy` |
| 16 | R17 memory, measured | peak-memory run on the 4090 | 30 `[A]` | needs GPU at train-idle |
| 17 | R18 vocab alignment | — | — | `76/0.72-r18-vocab-cell`; 3 Definition-of-Ready tests shared with R6 |
| 18 | R16 dangling `qlora-training-loop-v1` | — | — | `la/r16-qlora-loop-contract` |
| 19 | R14 throughput work | sized from the R5 gap | — | after R5 and R15b |
| — | R19 ROADMAP PMAT-711 stale | — | done | shaping @378ec8e920 |
| — | R20 declarative recipe | — | out | RQ-3: stays in #4002 (E8, 0.75) |

**Critical path v3:**
```
R1 ─► R2 ─► R3 ─► R21 GDN on CUDA ─┐
R15a C1–C4 ────────────────────────┼─► R4 ─┬─► R6 (DBH refusal lands with R1), R7 (MOF)
#4418 transforms ─► 4b HF loader ──┤       │
R12 receipts ──────────────────────┤       └─► R13 HF rc ◄── R10 QFR-003 export ◄── #4418 (GGUF legs only)
                                   └─► R15b C5–C7 (C5 = RQ-5) ─► R5 T2 verdict ─► R14
R11 TIS ◄── TDD normaliser (PRM C7–C9) ─────► gates every R4/R6 run counted for 0.72
```
T2 trains Qwen3.5-4B, so its apr side needs R2, R3, R21 and 4b as well as R15a and R15b. It does not need R4. R21
(400 + 25 `[A]`) is the largest row on both R4's path and T2's. Its LoRA wiring calls R15a's C1 helper, so C1
lands before R21's projection cell. The value-head work splits in two. The load-time conversion (4b) is on both
paths, because `apr finetune` trains from .apr only and the only Qwen3.5 .apr today is HF-sourced. The export
permutation (QQE-007) is R4's alone, because T2's canonical cell targets no GDN projection.

## §4 Rulings (S-4)
Ruled by the cop on 2026-09-27 at 12:11Z (full text in the handoff file):
- RQ-1: #4000's old "Agent Ready" rows go to the backlog milestone. Done 2026-09-27.
- RQ-2: T1 includes GDN forward and backward training in aprender-train.
- RQ-3: 0.72 owns GDN training for the qwen3.5 family; #4002 (E8, 0.75) keeps everything beyond it.
- RQ-4: #4418 stays in 0.71.

Requested 2026-10-03, not blocking:
- RQ-5: if R15 cell C5 (bf16) misses 0.72, should T2
  - (a) stay bf16 and slip to 0.73, or
  - (b) add a declared second cell, "apr fp32 vs Unsloth fp32"?

  (b) needs the incumbent re-run in fp32 as well; it never allows a cross-precision ratio. Recommendation: (a).
  C1–C4, C6 and C7 are needed either way, and the decision point is when C4 lands. Full text:
  `r15-cuda-lora-cells.md` §Consequences and the handoff file.

## §5 Fold plan: one APR v2 metadata reader `[C]`

Three places read an APR v2 file's metadata by hand:

- `publish_license.rs::apr_metadata` (HRP fold)
- `merge_output.rs::read_apr_metadata` (MOF fold)
- `tensor.rs::read_apr_metadata` / `read_apr_metadata_json` (on main)

Each does the same steps: open, read the 64-byte header, seek to `metadata_offset`, read `metadata_size` bytes up to `MAX_METADATA_SIZE`. They already differ on the error path. The publish copy errors on a corrupt v2 file and returns nothing if the file is not v2. The tensor copy returns `Option` and hides the error.

The open PR for whole-file model reads adds `apr_format::prefix::apr_v2_header_prefix`, a bounded prefix read of header + metadata + tensor index, and moves `tensor.rs` onto it. This fold puts one function beside it:

```rust
/// Header + metadata block only (no tensor index). Not APR v2 → Ok(None);
/// corrupt v2, or metadata past MAX_METADATA_SIZE → Err.
pub fn apr_v2_metadata(path: &Path) -> Result<Option<AprV2Metadata>, String>
```

Callers:
- publish maps the `Err` to `CliError`.
- merge maps the `Err` to `AprenderError`.
- `tensor.rs` keeps its `Option` via `.ok().flatten()`, so its behaviour does not change.

The publish and merge copies are **moved** into this one function. None stays as a fourth copy.

Contract `apr-v2-metadata-reader-v1` (written with the code, not before):

| ID | Falsifier | Planted mutant that must turn it RED |
|---|---|---|
| AMR-001 | Every call site returns the same metadata on a fixture table: valid v2, not v2, truncated header, `metadata_size` past the cap, `metadata_offset` past EOF | Swap `Ok(None)` and `Err` |
| AMR-002 | Bounded read: at most 64 + `metadata_size` bytes read on a sparse 4 GiB file | Drop the size cap |
| AMR-003 | Structure: no `AprV2Header::from_bytes` outside `apr-format` and the v2 internals | Read the metadata at offset 64 and ignore `metadata_offset` |

Order: after the whole-file-reads PR, the HRP fold and the MOF fold are all on main.

## §6 Measurement plan: how each exit criterion is measured `[C]`

L2 deliverable 3 (APR-LOOKAHEAD-001 §4). Written 2026-09-29 from contracts at the branch tips named below, and updated
2026-10-03 for T2 contract 1.3.0 and QQE-004 in qwen35-qlora-e2e-v1 1.1.0. Each criterion names what measures it, where
it runs, what counts as green, and what counts as **NOT MEASURED**. NOT MEASURED is never green (L25). A criterion is
green only when every row under it is green on **one** pinned `apr` binary (version + sha in the receipt), built from the
release commit.

**Rules for every row:**
- The device comes from a trace line in the log, never from a flag or `CUDA_VISIBLE_DEVICES` (CLAUDE.md verification rule 2).
- Every training run writes `train_receipt.json` (`train-run-receipt-v1`, TRR-001..006). A run with no receipt, or a
  receipt missing a key, is NOT MEASURED.
- Every rc-bound training run passes the sealed-ingress check (`train-ingress-sealed-refusal-v1`, TIS-001..005). With no
  manifest it is refused (TIS-005), and never counted as `sealed_hits = 0`.
- Each row needs its planted falsifier RED before its green counts. A planted check that cannot turn RED makes the row NOT MEASURED.
- Hosts: **lambda** = RTX 4090 24 GB, GPU rows only, and only when the train is idle. **intel** = CPU tests, load1 ≤ 32.
  **gx10** = GB10 second GPU, used for the cross-host control only.

### T1 — the four verbs run end to end on Qwen 3.5 with 0 refusals

| Verb | Measured by | Host | Green | NOT MEASURED when |
|---|---|---|---|---|
| all | TAH-001..004 (`train-arch-honesty-v1`): a qwen3_5 config is never silently built as dense | intel | 4/4, planted dense map RED | branch `la-72/4552-train-arch-honesty` not on main |
| finetune | QQE-006 CPU pre-flight first, then QQE-001/002/003 on 4B (`qwen35-qlora-e2e-v1`). QQE-005 holds CUDA adapter gradients to the CPU reference | intel (006), lambda (001–005) | exit 0; loss(last 10) ≤ 0.9 × loss(first 10); served = merged (cos ≥ 0.999, equal argmax); QQE-002 frozen-step RED | QQE-006 not green, since no GPU time is spent before it is |
| finetune (NF4) | QQE-004 (1.1.0): QLoRA's mean loss over the last 10 steps ≤ 1.05 × LoRA's, same cell, seed and data. The QLoRA receipt says `recipe.precision = nf4`, the LoRA one bf16 or fp32, and both carry the same `device.uuid` | lambda | ≤ 1.05, **or** a named refusal for qwen3.5 QLoRA (K10) | only one side ran; a side ran on the CPU (true of `-m lora` at `316dee2cd4`, until R15a C4); the sides ran on different GPUs; any other precision pair |
| distill | `distill-batch-honesty-v1` (DBH) on the fold-dbh branches: batch B > 1 trains every row or refuses by name | intel (refusal), lambda (batched KD) | refusal green on CPU; batched KD matches B single-row steps | DBH-001/006/007/008 GPU halves not run |
| merge | `merge-output-fidelity-v1` (MOF): `-o *.apr` writes an APR with metadata and a qwen3_5 arch | intel | MOF-002/003 green; the planted F32-safetensors writer RED | — |
| quantize | the R8 GDN quantize policy cell, branch `79/r8-gdn-quant-policy` (another session's) | intel | owner's falsifiers green | that branch is not on main; L2 does not measure it |

The count for T1 is **refusals = 0 AND silent-wrong = 0**. A named refusal (DBH-002, QFR-005, QQE-004's escape) still counts
as a refusal. It is honest, but it is not T1-green.

### T2 — fine-tune throughput ≥ 0.8× Unsloth

- **Measured by:** `beat-unsloth-finetune-throughput-v1` 1.3.0. The command is `scripts/bench/unsloth_finetune_throughput.sh
  --model Qwen3.5-4B --gpu 0 --out evidence/beat-unsloth-ft/<version>/`, on the `la-72/r5-*` branches (R5, done on the
  desk). The apr side stays NOT MEASURED until `apr finetune` has the canonical flags and receipt fields (R15b, built on
  R2, R3, R21, 4b and R15a). Before that, the apr adapter exits 4 and lists every missing key.
- **Cell:** the canonical task, the same on both sides:
  - bf16 LoRA, not QLoRA (S-R5). Precision is pinned to bf16 on both sides (1.3.0).
  - r16, alpha 32, on the 7 targets q, k, v, o, gate, up and down.
  - AdamW fp32, checkpointing off, packing off.
  - seq 512, batch 4, grad_accum 1.
  - 200 timed steps, starting after 50 warmup steps and after compile.
  - `label_tokens_timed` = 408800 on the pinned `APR_FT_DATA`.
- **Statistic:** median of 3 runs per side, run interleaved on the same GPU in one session. The ratio is
  apr tokens/s ÷ Unsloth tokens/s, counting non-pad label tokens.
- **Host:** lambda 4090, train idle. The gx10 run is a control only, and never the T2 number.
- **Green:** ratio ≥ 0.80, **and** all three planted runs RED in the same session:
  SAMEWORK (`--planted-half-targets`), INCUMBENT-FASTPATH (`--planted-no-fla`) and SAMEOPT (`--planted-incumbent-adamw8bit`).
  PRECISION, FULL-WINDOW and DATA-PINNED are verdict checks on every run, tested on the CPU.
- **NOT MEASURED when:** fla is missing (INCUMBENT_SLOW_PATH), the versions are not in the receipt, the apr receipt lacks
  a key (adapter exit 4), or fewer than 3 runs completed on either side. A precision or label-token mismatch is a SAME-WORK
  FAIL (exit 1), not NOT MEASURED.
- **Not T2:** after R15a C4, an fp32 apr run done by hand against the bf16 Unsloth baseline gives a sizing number for K2
  and RQ-5. It is never a T2 number: PRECISION makes it a SAME-WORK FAIL by design.
- **First:** Unsloth's side can run the moment the train is idle. It needs no apr work, and it gives R14 its target.

### T3 — Prometheus B2 challenger

- **Measured by:** the PRM-001 §5.4 judgement rule, owned by rex (branch `rex/001-prm-s1-v2`), not this spec. L2 measures only
  the inputs:
  - Teacher: Qwen3.5-27B, local.
  - Student: 4B, trained by `apr distill`. It is distill-green under T1.
  - The training data carries `sealed_hits = 0` against a non-empty manifest (TIS-001..005).
  - The receipt is complete (TRR).
- **NOT MEASURED until:** PRM cluster.rs lands on main (it blocks TIS-002), and DBH batched KD is green on lambda.

### T4 — the first improved dogfood model on Hugging Face as an rc, with receipts

- **Measured by:** `hf-rc-publish-v1` (branch `la-72/fold-hrp`), HRP-001..005:
  - The plan lists every file it uploads.
  - The published repo is loadable: `apr import` of exactly the planned files works.
  - The license is never defaulted to MIT (planted).
  - The card carries the TRR fields and no N/A metrics.
- **"Improved"** means the rc's score beats its base on the T3 judgement, from the same receipt. A publish with no
  comparison is a publish, not T4.
- **Host:** intel for `--dry-run --offline`. The one real upload is a release-path action, done at the cut.
- **NOT MEASURED when:** the upload happened without a dry-run receipt of the same sha, or the T5 check was not run on the
  uploaded files.

### T5 — trained weights round-trip gguf ↔ safetensors ↔ .apr at cosine ≥ 0.98

- **Measured by:** `qwen35-format-roundtrip-v1` (branch `la-72/fold-r10-qfr`):
  - QFR-001 (in tree, pygmy) and the Qwen3.5-0.8B cell: lossless legs are bit-identical.
  - QFR-002, planted: a single transposed tensor turns the gate RED.
  - QFR-003: a bare exported safetensors re-imports.
  - QFR-004: GGUF export maps every tensor.
  - QFR-006 (PROPOSED 2026-10-03): the exported GGUF computes the source's function. An export that maps the names and
    skips #4418's value transforms passes QFR-001..004, so this row compares the export with llama.cpp's own conversion
    of the same snapshot, tensor by tensor and by `apr eval` perplexity.
  - QFR-006 checks the export, not the merge. A merged file that lost the GDN deltas, or carries them at half scale,
    exports and converts identically in apr and in llama.cpp, so QFR-001..006 stay green on it. QQE-009
    (`qwen35-qlora-e2e-v1`) checks the merge.
- **Statistic:** the **minimum** per-tensor cosine over all tensors, never the mean. It uses `apr diff --values
  --limit <|T|>`. The default `--limit 10` samples too few tensors, and QFR-002 guards against that.
- **Model:** the trained 4B from the T1 finetune row, after the 0.8B dev cell is green. The GGUF legs and QFR-006 need
  the 4B: the 0.8B has as many value heads as key heads (16 and 16), so the value-head reorder is the identity there
  and a 0.8B cell cannot catch an export that skips it.
- **Host:** intel, CPU.
- **NOT MEASURED when:** #4418 (qwen35 GGUF name map) is not merged. Without it the GGUF legs cannot run, and QFR-005's
  named refusal is the only honest answer.

### Order when the train goes idle (GPU queue)

1. Unsloth side of T2. It needs no apr work.
2. R17: peak memory on the 4B.
3. QQE-005: CUDA adapter gradients against the CPU reference, on 0.8B.
4. DBH GPU halves.
5. QQE-001..004 on 4B. QQE-004's LoRA reference needs R15a C4.
6. The apr side of T2. It needs R15b.

CPU rows (TAH, TRR, MOF, QFR on 0.8B, HRP dry-run, TIS-001/003/004/005) run on intel at any time, load1 ≤ 32.
