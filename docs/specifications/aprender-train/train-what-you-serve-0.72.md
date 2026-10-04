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
  layer schedule to the training transformer. Take the recurrence from serve's code, not from `gated-delta-net-v1`
  1.0.0, which states a different GDN (K36, row 21). `qwen35-train-gdn-v1` 1.1.0 on `fold-r2r3` states the one serve
  computes.
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
- **Gates:** loss(last 10) ≤ 0.9 × loss(first 10), all finite. Served base+adapter equals the training-side merged forward,
  both on the base `apr finetune merge` reads (cos ≥ 0.999, equal argmax). The merged model's training loss is within 2%
  of the trained model's, which ran on the NF4 base (QQE-010, row 28). Doubling alpha doubles the first step's merged
  delta, on CUDA as on the CPU (QQE-011, row 29).
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
  - QQE-003's cosine of 0.999 on logits may not see a LoRA delta the merge dropped. QQE-009 (PROPOSED) plants all three
    and checks the whole chain tensor by tensor on a 3/6 fixture: the adapter R4 writes, merged by `apr finetune merge`
    and read back through row 4b's loader, must equal `Qwen35Lora::merge_into`. That also catches a writer that drops
    a target kind, which no merge-side check can see.
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
  - The trainer holds a Qwen3.5 in GGUF conventions (`Qwen35Model::from_gguf`). Before the builder writes it under HF
    names, it must undo every value transform #4418's export applies, including −1 on the five zero-centred norms (K38,
    row 23).
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
- **Row 20, K30, is new and off the critical path.** `apr train` writes config.json's `tie_word_embeddings` from
  `TransformerConfig::ties_embeddings()`, which returns `use_bias && vocab_size > 150000` whenever the configured flag
  is false and never reads the model it saved (`helpers.rs:876` and `config.rs:434` in aprender-train at
  `316dee2cd4`). The flag and the weights can disagree either way:
  - An untied checkpoint with attention biases and a vocabulary over 150000 is written as tied. apr's safetensors
    loaders read the flag before the tensor, so apr serves the embedding as the head.
  - A run without weights saves no head but is written as untied, so an HF-convention loader starts a random head.
  Only `apr train` reaches this writer (`train_from_yaml`); `apr finetune`, which R4 and T2 use, does not. The
  contract is `apr-train-output-config-v1` (TOC-001..003) on its own branch, and the fix writes the saved model's
  own state from both save paths.
- **Rows 21 and 22, K36 and K37, make main's GDN contracts state and test the shipped function. Both are off the
  critical path.** R2's gate is QTG-001, which runs serve's code as the oracle, so neither row blocks a training cell.
  Read at `316dee2cd4`:
  - **K36.** `gated-delta-net-v1` 1.0.0 states a decay, a read and an output that neither crate computes: a sigmoid
    decay, a read before the decay, and z applied twice. Its tests run against
    `provable_contracts::kernels::gated_delta_net::gdn_recurrence_scalar`, which is not a delta rule. It computes
    `S ← αS + β k⊗v`, with no `v − Sᵀk` term (`gated_delta_net.rs:52`). The decay test checks the contract's sigmoid
    formula verbatim (`gated_delta_net_contract.rs:80`), and GDN-BND-001's Lean proof is about the same sigmoid
    (`Recurrence.lean:58`). `qwen35-hybrid-forward-v1` 1.0.0 leaves out four things that serve and the training layer
    both compute in attention: the sigmoid output gate, the K norm, partial RoPE and the output projection. Its GDN
    sublayer sends every projection through the conv, but only q, k and v go through it (`forward_qwen35.rs:1508`).
  - **K37.** `pv audit --binding` takes `status: implemented` on trust. Six realizar bindings for these two contracts
    name functions that exist nowhere under `crates/` (`gated_delta_net_{decay,read,write,delta,output,forward}`), and
    the audit still counts them as implemented. Two more realizar bindings and one entrenar binding are the same. The
    strict gate already exists: pv lint's `bindings` gate (PV-ONT-028/029, `lint/bindings_gate.rs`) resolves every
    implemented binding, and `binding-allowlist.json` lists all 9 under #4502. It does not yet protect 0.72's training
    contracts, for two reasons. First, `bindings` is not in `lint-baseline.json`'s `armed_gates`. Second, the allowlist
    misses one ghost, `gated_rmsnorm_oxide::kernels::gated_rmsnorm` (`contracts/binding.yaml:817`), which would turn
    the gate red once armed. K37 is a comment on #4502, not a ticket. The K36 branch removes the 7 allowlist entries
    that its rebinding resolves.
- **Row 23, K38: serve, train and the GGUF export agree on Qwen3.5's RMSNorm weights, but serve's safetensors path
  does not refuse the model.** HF Qwen3.5 stores five norms zero-centred: input, post-attention, q, k and final. It
  applies them as x̂·(1 + w) (transformers 5.3.0 `modeling_qwen3_5.py:808,821`). The gated `linear_attn.norm` is plain
  x̂·w·silu(z). llama.cpp adds 1 to every `norm.weight` except `linear_attn.norm.weight`, so GGUF stores 1 + w.
  On Qwen3.5-4B, measured on CPU, GGUF − HF = 1 exactly for all five, and `ssm_norm` equals `linear_attn.norm`.
  `ssm_a` is −exp(A_log) computed in f32: bit-exact against torch f32, and 3·10⁻³ away from a bf16 exp. So the
  converter that made this file upcast to f32 first, as QFR-006's bitwise prediction assumes. The file does not name
  its llama.cpp version, so the same check at d1d3c3396 stays open until QFR-006's 4B cell runs.
  - **They agree.** Serve's GGUF forward applies the stored 1 + w as x̂·w. The training model loads from the GGUF
    (`fold-r2r3`, `Qwen35Model::from_gguf`) and does the same. #4418's export adds the 1 to the five and not to the
    gated norm, and a unit test pins that (`m0694/4418-qwen35-gguf-main`).
  - **Serve's safetensors path does not refuse it.** `apr run`, `chat` and `serve` load a Qwen3.5 safetensors
    checkpoint. Each GDN layer's `in_proj_qkv` and `out_proj` go into the dense attention slots, and the GDN tensors go
    into fields that no CPU forward reads. The norms get no +1. The GGUF paths ask `hybrid_forward_handles`; the
    safetensors paths do not. This is a desk read and has not been run. The fix is a refusal at conversion. It is off
    0.72's path, because QQE-003's served side is the GGUF.
- **Row 24, K39: `apr finetune` trains on a chat format that serve never sends.** Desk read at `316dee2cd4`; nothing
  run. Ids are from the HF snapshot's `tokenizer_config.json`.
  - **The control tokens are split.** Every `apr finetune` training path ends in `InstructPipeline::from_apr`
    (`finetune.rs:387,461`). Its tokenizer comes from the .apr's own vocabulary first (`constructors.rs:212`), rebuilt
    as tokenizer JSON with `"added_tokens": []` (`:343-350`). `load_from_json` registers special tokens only from that
    list (`qwen2.rs:391,442`). `BpeTokenizer::encode` keeps only registered special tokens whole
    (`qwen2bpe_tokenizer.rs:328,372`). So `<|im_start|>` and `<|im_end|>` reach the model as six BPE pieces each, in
    the prompt and in the trained target (`instruct_trainer.rs:329-332`, `accessors.rs:23`). #3920 fixed this for
    `load_from_vocab_merges`: a vocabulary entry shaped `<|…|>` is registered whole (`qwen2.rs:528-553`). Its doc
    says "one parser and not two". The train loader goes through JSON and never reaches it. Serve matches special
    tokens first (`gguf/byte_level_bpe.rs:384-400`), so the served prompt carries 248045 and 248046 as single tokens.
    The trained target never ends in 248046 (`<|im_end|>`, the eos).
  - **The template differs.** Train renders fixed ChatML with a default system prompt (`instruct_corpus.rs:51-66`).
    Serve renders the model's own template with thinking off unless asked (`chat_template_helpers.rs:209`). There is
    no system turn unless the request sends one, and `<think>\n\n</think>\n\n` follows `<|im_start|>assistant\n` (HF
    `chat_template.jinja:54-64,148-150`). Train has no think block. `<think>` and `</think>` (248068, 248069) are
    added tokens that are not shaped `<|…|>`, so the #3920 rule alone would still split them.
  - **Serve has two Qwen3.5 prompts.** A model with no template of its own gets apr's built-in formatter
    (`chat_template_helpers.rs:206-207`). A name containing `qwen3` picks `Qwen3NoThink`. That formatter ends in
    `<think>\n</think>\n` (`chat_template_qwen3_nothink.rs:44`), while the model's own template ends in
    `<think>\n\n</think>\n\n`. `\n\n` is one token, `ĊĊ`, so the ids differ. Every HF-sourced .apr carries no
    template (below), so `apr run` and `apr chat` on it send the built-in prompt (`:190` names both callers). A
    published GGUF sends the model's own. The GGUF that 0.72 exports does not (next bullet).
  - **The GGUF that 0.72 serves is the one apr exports, and it has no template either.** R4's QQE-003, T4 and T5
    serve the file that `apr export --format gguf` writes from `apr finetune merge`'s output (row 14). The merge
    clones the base's metadata (`finetune_display_next_validate.rs:518`), so an HF-sourced base passes on no template.
    - At `316dee2cd4` the export also writes no eos id and no token types, and it names the pre-tokenizer `default`,
      because `resolve_pre_tokenizer_type` has no Qwen3.5 arm (`export.rs:68-84`). S-R10's export of the .apr below,
      with apr 0.69.3, has exactly that (`hf-rt.gguf`, 18 keys). GGUF serve implements only `qwen2` and `qwen35`
      (`byte_level_bpe.rs:47-53`). On `default` it falls back to greedy longest match, which its own warning says
      "does NOT reproduce the model's tokenization" (`token.rs:165-176,419-426`). With no eos id in the file, the eos
      becomes 248044, `<|endoftext|>` (`config.rs:331,789-791`), while every turn ends in 248046. At this commit the
      file does not load anyway, because its tensor names pass through (QFR-004).
    - #4418's branch (`m0694/4418-qwen35-gguf-main` @1af0e3cc11, not on main) rewrites the tokenizer block for
      Qwen3.5 (`fix_qwen35_tokenizer_metadata`, `qwen35_gguf.rs:655`). It sets pre `qwen35`, types `<|…|>` CONTROL
      and the other added tokens USER_DEFINED, and sets eos to `<|im_end|>` and padding to `<|endoftext|>`. Its test
      is `tokenizer_fixup_types_pads_and_ids`. The template still comes only from the .apr's metadata or from a
      `chat_template.jinja` beside the input (`qwen35_chat_template`, `gguf_export_config.rs:744`). An HF-sourced
      base has neither, so apr serves the exported model with the built-in `Qwen3NoThink` prompt (TSC-003). The
      importer fix in row 24 closes this as well, because the merge passes the base's template on.
  - **Measured on S-R10's HF-sourced .apr** (Qwen3.5-0.8B, the HF directory imported by apr 0.69.3; header read
    only). `tokenizer.vocabulary` has 248,070 entries, with `<|im_start|>`, `<|im_end|>`, `<think>` and `</think>` at
    248045, 248046, 248068 and 248069. The metadata has 17 keys: none is a chat template, a token-type list, an
    added-token list or an eos id. The HF snapshot's `tokenizer_config.json` has both. So the .apr gives train neither
    the model's own template nor which entries are special, and the fix spans the importer as well as train.
  - **Effect on 0.72.** R4 and T2's apr side train through this path, because `apr finetune` trains from .apr only.
    R4's gates cannot see it: the loss gate is met on pieces too, and QQE-003 feeds both forwards the same ids. T4
    would publish a model tuned on a prompt that no apr server sends, which ends its answers in text pieces instead
    of the eos. T2's ratio is not biased by this. Its rows are plain `{"text": …}` Rust that fill 512 tokens on both
    sides, so the count is fixed by shape. If R15 routes plain rows through `format_chat_prompt`, apr's rows would
    carry the system turn and the pieces in place of about 45 tokens of Rust `[A]`.
  - **Earlier work.** `origin/PMAT-3803-trainer-apr-tokenizer` @5116cbc28b (2026-09-22) is not on main, and no PR head
    matches its last 12 commits. It builds a byte-level vocabulary with realizar's canonical BPE, under
    `feature = "realizar"` (#3742). It does not touch the template.
  - **Falsifiers (TSC, PROPOSED in `train-serve-chat-format-v1` on `la-72/k39-k40-contracts` @ae7a75b7f0):**
    - TSC-001: the tokenizer `from_apr` builds from an embedded vocabulary that holds `<|im_start|>`, `<|im_end|>`,
      `<think>` and `</think>` encodes each as one id. It is RED at `316dee2cd4`: on the Qwen3.5 vocabulary
      `<|im_end|>` alone is six pieces and `<think>` three (HF tokenizers). Planted: `"added_tokens": []` restored.
    - TSC-002: for a one-turn sample on the Qwen3.5 vocabulary (CPU), the ids `prepare_samples` gives (prompt, then
      response) equal serve's ids for the same messages: the model's own template with thinking off, then the
      answer, then 248046. It is RED today on the system turn, the think block and the pieces. Planted: the default
      system prompt restored.
    - TSC-003: for Qwen3 and Qwen3.5, the built-in `Qwen3NoThink` renders a two-turn conversation exactly as the
      model's own template does with thinking off. It is RED at `316dee2cd4` on the two newlines. Planted:
      `<think>\n</think>\n` restored.
    - TSC-004: the GGUF that `apr export` writes from `apr finetune merge`'s output, on an HF-sourced Qwen3.5 base,
      carries the model's own template, pre `qwen35`, eos 248046 and CONTROL types on the `<|…|>` tokens. It runs on
      QQE-009's 3/6 fixture. It is RED at `316dee2cd4` on all four (S-R10's export). On #4418's branch it would be
      RED on the template alone (desk read). Planted: the template dropped from the base .apr.
- **Row 25, K40: apr's tokenizers split the same text three ways before BPE.** Desk read at `316dee2cd4`, plus a
  simulation: HF `tokenizers` 0.22.2 on the Qwen3.5 vocabulary and merges, with each split swapped in. It is not the
  Rust code.
  - **The paths.** Qwen3.5's `tokenizer.json` splits text with a regex before BPE: letters, single digits, punctuation
    runs, newline runs, and space runs that leave their last space to the next word. NFC is applied first.
    - GGUF serve implements that regex (`gguf/byte_level_bpe.rs`, since #3772).
    - Train does not. `HfTokenizer` wraps aprender-core's `BpeTokenizer`, whose `pre_tokenize` starts a new word at
      every whitespace character (`qwen2bpe_tokenizer.rs:463-466`: "Future: Use self.config for model-specific
      pre-tokenization rules"). `apr chat` on a .apr builds the same type (`chat.rs:461`).
    - `apr run` and `apr serve` on a .apr use realizar's `apr::BpeTokenizer` (`infer/mod.rs:726,817` through
      `encode_text`, and `serve/handlers.rs:1307`). It splits only at special tokens and runs the merges over the
      whole segment (`apr/tokenizer.rs:125-147,243-262`).
    - None of the three applies NFC.
  - **Measured (simulation).** Five samples: two Python, one Rust, one Markdown and one English. They come to 160
    tokens under the regex, 193 under train's split, and 160 with no split.
    - English is the same under all three.
    - Indented code is not. Four spaces then `if` is `ĠĠĠ`,`Ġif` under the regex; `Ġ`,`Ġ`,`Ġ`,`Ġif` in train; and
      `ĠĠĠĠ`,`if` on .apr serve.
    - A blank line is `ĊĊ` under the regex and `Ċ`,`Ċ` in train. A five-line Python class is 23 tokens under the
      regex and 38 in train.
    - Every split decodes back to the same text, so a round-trip test cannot see the difference.
  - **Effect on 0.72.** R4 and T4 tune through train's split. On code, the model learns indentation as ids that its
    pretraining never produced and that no apr server sends. R4's gates cannot see this: QQE-003 feeds both forwards
    the same ids, and the loss gate is met under any split. T2's ratio is not biased. Every row fills 512 tokens on
    both sides, so `label_tokens_timed` is fixed by shape (`unsloth_ft_data.py` on `la-72/r5-data`). The sides only
    see different ids for the same Rust text, and apr's 512 tokens cover less of it. Separately, .apr serve hands
    even the base model a split it was never trained on. The served GGUF splits with the regex only if the exported
    file names `qwen35`. At `316dee2cd4` it names `default` and serve falls back to greedy longest match (K39);
    #4418's branch names `qwen35`.
  - **Earlier work.**
    - `crux-M-05-v1.yaml` (draft) states the check: `ids_apr == ids_hf` on 128 fixtures that include code. Nothing in
      apr-cli or aprender-serve implements it.
    - `bpe-tokenization-v1.yaml:98-99` says `pre_tokenize` splits at whitespace and punctuation. The code splits at
      whitespace only.
    - PMAT-3803's branch (K39) moves train onto realizar's `apr::BpeTokenizer` (`hf.rs:35,67` there). On that branch
      the .apr tokenizer uses the regex only when `canonical_for_apr` can name a pre-tokenizer: `tokenizer.pre_type`,
      else the architecture, where `for_architecture` knows `qwen35` but not `qwen3_5`. An HF-sourced Qwen3.5 .apr
      carries neither (S-R10's `hf.apr`), so for 0.72's model train would match .apr serve, with no split, and still
      not match the regex. Qwen2 and Qwen3 .apr files would get the regex.
  - **Falsifiers (TPP, PROPOSED in `tokenizer-pretokenize-parity-v1` on the same branch).** Each runs on CPU against
    frozen reference ids.
    The ids come from the shipped `tokenizer.json`, pinned by its sha256. The fixture covers indented Python and
    Rust, blank lines, space runs before a newline, digits, punctuation runs, contractions and combining marks.
    - TPP-001: train's tokenizer, built by `from_apr` from a Qwen3.5 .apr, gives the reference ids. It is RED at
      `316dee2cd4` in the simulation. Planted: the whitespace-only `pre_tokenize` restored.
    - TPP-002: `AprV2Model::encode_text` on the same .apr gives the reference ids. Planted: the whole-segment merge
      restored.
    - TPP-003: GGUF serve gives the reference ids, both on a published GGUF and on the file `apr export` writes
      (TSC-004's chain). On a published GGUF it should be GREEN today, which keeps the reference honest. On the
      exported file it would be RED at `316dee2cd4` (pre `default`, greedy fallback) and GREEN on #4418's branch.
- **Row 26, K41: the HF-sourced Qwen3.5 .apr loses the model's rope base and dimensions.** Desk read at `316dee2cd4`
  and on #4418's branch (@1af0e3cc11), plus header reads of S-R10's `hf.apr` and `hf-rt.gguf` (apr 0.69.3) and of
  the published GGUFs. Nothing was run.
  - **The source.** Qwen3.5's `config.json` nests the text model under `text_config`, and the rope base under
    `text_config.rope_parameters`: `rope_theta` 1e7, `partial_rotary_factor` 0.25, `mrope_section` [11,11,10]. The
    top level has neither the dims nor `rope_theta`. The published 0.8B, 2B, 4B, 9B and 27B GGUFs all say
    `rope.freq_base` 1e7 and `key_length` 256.
  - **At `316dee2cd4`.**
    - Import: `load_model_config_from_json` reads the top level only (`source_load_result.rs:282-303`). `rope_theta`
      takes its default, 10000, and is always `Some`. S-R10's `hf.apr` says rope_theta 10000 and has no hidden size,
      head counts or head_dim.
    - Export: S-R10's `hf-rt.gguf` names arch `qwen3_5` and says freq_base 10000, heads 16/8 and ctx 0, with no
      `key_length` and no `dimension_sections`. The published 0.8B says `qwen35`, 1e7, 8/2, 262144, 256 and
      [11,11,10].
    - Serve takes the base from the file's `rope.freq_base` (`gguf/config.rs:739-742`), and the Qwen3.5 forward
      rotates with it (`forward_qwen35.rs:1663`). Neither fallback gives 1e7. Serve's
      `default_rope_theta_for_architecture` has no qwen35 arm (`config.rs:365`, so 1e4), and core's copy gives 1e6
      to any name containing "qwen" (`export.rs:52`).
    - `apr finetune` refuses, which is loud. The .apr has no dims, and `read_hf_config_file` reads the sibling
      `config.json` at the top level only (`model_config.rs:154-203`; it also drops `head_dim` and defaults rope to
      10000). The last resort, `--model-size`, is wrong too: `qwen3_5_9b()` sets 1e6, "Same 1M theta as Qwen2"
      (`transformer/config.rs:246-264`), and so does the family contract's 9b row
      (`contracts/model-families/qwen3_5.yaml:28`). The published 9B says 1e7. The contract's 27b row also says
      hidden 6144 where the published 27B says 5120, and the contract has no 0.8B, 2B or 4B row.
    - `apr finetune merge` clones the base's metadata (K39). `backfill_arch_dims` fills hidden, vocab, layers and
      intermediate from the tensors, and head counts for qwen2 only (`finetune_display_next_validate.rs:169-178,240`).
      It never touches rope_theta.
  - **On #4418's branch.** The import is fixed: `merge_text_config` lifts `text_config`
    (`source_load_result.rs:288,381`), and the base is read from `rope_parameters` (`:304-310`). A fresh import gets
    1e7 and the dims. An older .apr is not caught. `qwen35_base` takes each field from the .apr first and from the
    `config.json` beside it second (`gguf_export_config.rs:680-709`). Its 1e7 fallback fires only when the .apr has
    no rope_theta (`qwen35_gguf.rs:497`), and an import made before #4418 always has one: 10000. So S-R10's
    `hf.apr`, or a merge of it, exported with the HF `config.json` beside it gets the right dims and freq_base 10000,
    with no warning. Without that `config.json` the export refuses on the missing head count. #4418 changes no
    apr-cli, train or rope-default file, so the finetune reader, the presets and both fallbacks stay as above.
  - **Effect on 0.72.** R4 serves the export of a merged HF-sourced base, and so does T4's judgement if it serves the
    rc from apr's export. If that base was imported before #4418 lands, the served model rotates at 1e4 where it was
    trained at 1e7. A quarter of each head rotates (64 of 256 dims, 32 frequency pairs), and pair i turns by
    base^(−i/32) rad per token. Both bases give 1 rad at i = 0. At i = 16 it is 0.01 against 3.2e-4, about 32 times
    faster. The effect on loss and output is not measured. Most gates cannot see it. QFR-001..005 and T5 check
    values, names and the architecture, not the rope base. QQE-003 compares two forwards that read the same file
    (fold-r2r3's training model takes `rope.freq_base` from the GGUF, `qwen35_model.rs:294`). QFR-006 compares the
    `qwen35.*` keys with llama.cpp's conversion, but on a fresh import only, never on a merge or an older .apr.
  - **Falsifiers (PROPOSED in `qwen35-format-roundtrip-v1` 1.1.0 on `la-72/fold-r10-qfr` @e6ea295728).**
    - QFR-007: the .apr that `apr import` writes from an HF Qwen3.5 snapshot, and the GGUF that `apr export` writes
      from `apr finetune merge`'s output of it, carry the dims and rope base the source config states. In the GGUF
      that is `freq_base`, the head counts, `key_length`, `context_length` and `dimension_sections`. It runs on
      TSC-004's chain, with a fixture config that nests `text_config` as Qwen3.5's does. On the 0.8B the reference is
      the published GGUF's header. It is RED on S-R10's files (measured) and at `316dee2cd4` (desk read, the same
      top-level read). On #4418's branch it would be GREEN for a fresh import. Planted: `merge_text_config` or the
      `rope_parameters` read removed.
    - QFR-008: `apr export` refuses a qwen35 .apr whose rope_theta or a dim disagrees with the `config.json` beside it,
      or with the source config QFR-003 stores, and prints both values. It would be RED on #4418's branch, which
      exports S-R10's `hf.apr` at 1e4 (desk read). Planted: S-R10's `hf.apr` with the HF `config.json` beside it.
    - QFR-009: for qwen35, serve's and core's rope fallbacks return 1e7, and `qwen3_5_9b()` and the family contract
      match the published 9B and 27B headers. It is RED at `316dee2cd4` on all four rope values (1e4, 1e6, 1e6, 1e6)
      and on the 27B's hidden size. Planted: 1e6 restored in the preset. The preset's line fits R1's branch
      (`4552-train-arch-honesty` @1fda81ad6c), which already gates `qwen3_5_9b()` and leaves its rope at 1e6.
  - **Until it lands,** re-import S-R10's base after #4418 merges. Do not reuse it.
- **Row 27, K42: the training window cuts the end of the answer and says nothing.** Desk read at `316dee2cd4`, plus a
  token count of the repo's two SFT corpora on the Qwen3.5 tokenizer (HF tokenizers 0.22.2, imitating each path's
  tokenizer). No apr code was run.
  - **At `316dee2cd4`.**
    - `train_step` joins the prompt and target ids and keeps the first `max_seq_len` (`instruct_pipeline/training.rs:29-33`).
      The default is 512 (`mod.rs:87`), and `apr finetune --max-seq-len` sets it (`finetune.rs:334`). `evaluate` cuts
      the same way (`:488-490`). The target ends in `<|im_end|>` (`instruct_corpus.rs:63`), so a cut takes the stop first.
    - Nothing checks the length first. `prepare_samples` tokenizes and keeps every sample (`instruct_trainer.rs:324-336`).
      F-INST-002, "Total token count (prompt + response) must fit max_seq_len", is stated (`instruct_corpus.rs:9`) and
      not enforced.
    - One case is reported. When the prompt alone fills the window, the CUDA step prints a skip line
      (`training.rs:149-157`, the fix for FALSIFY-CUDA-LOSS-WINDOW-512-001), and the CPU step returns 0 with no message
      (`:87-88`). A cut target is reported nowhere, and the epoch line prints tokens and samples, not cuts.
    - The flag reads as a memory setting: "Maximum sequence length for GPU buffer allocation (lower = less VRAM)"
      (`model_ops_commands.rs:58`). The f32 logits buffer alone is `max_seq_len` × vocab (`cuda_init.rs:453-455`). On
      Qwen3.5's 248,070 entries that is 0.51 GB at 512 and 2.03 GB at 2048, so a small card pushes the window down.
    - A build with `--features wgpu` (in neither the default nor `full`) trains in `train_wgpu_sft` under
      `--gpu-backend wgpu`, or under the default `auto` with QLoRA (`finetune.rs:291-295`, `:442`, `:906`;
      `model_ops_commands.rs:70-72`). That path encodes the raw instruction and response, with no template and no eos
      (`:917-918`), in a fixed 512 window (`:719`) whose overflow `train_step` drops with no message
      (`wgpu_pipeline.rs:768-769`), on q_proj and v_proj only (`:725`). TSC's "every training path ends in `from_apr`"
      holds for default and `full` builds only. No CI job builds `--features wgpu`: the clippy feature matrix excludes
      it as not compiling (#4056, `check_clippy_feature_matrix.sh:20`), and whether it compiles at `316dee2cd4` is not
      measured.
    - Its DPO branch (`:737-740`, `:866`) trains the prompt, chosen and rejected texts raw, with beta fixed at 0.1
      (`:881-884`). No `{prompt, chosen, rejected}` file reaches it: the corpus is parsed as instruction and response
      first (`:409`, `:519`), so such a file fails as invalid JSONL. No `apr finetune` help text, contract or spec
      says it trains DPO.
  - **Measured (simulation).** Each sample is rendered as `format_chat_prompt` renders it and counted two ways: with
    serve's ids (control tokens whole, the regex split), and imitating main's train tokenizer (control tokens in
    pieces, TSC-001; a split at each whitespace, TPP-001).
    - `datasets/apr_code_sft_curated.jsonl`: 124 samples, each ending in `</tool_call>`. With serve's ids they are
      456–480 tokens and all fit in 512. With main's train tokenizer they are 511–535, so 123 are cut: 122 by 2 to 10
      tokens and one by 23. Every cut takes at least the last two of the six pieces of `<|im_end|>`, and 68 samples
      also lose the `>` that closes `</tool_call>`. K39 and K40 add the tokens, and K42 turns them into a cut answer.
    - `datasets/apr_code_sft_balanced.jsonl`: 200 samples with a long system prompt. At 512, 184 prompts fill the
      window on either count, and main's tokenizer cuts the other 16, so no sample is trained whole. With serve's ids,
      1024 skips 86 and cuts 31, and 2048 skips 6 and cuts 2.
  - **Effect on 0.72.** A sample longer than the window teaches an answer that does not end: no eos, and on a tool
    call no closing tag. The served model then runs past its answer. The step's loss looks normal, and validation
    cannot see it, because `evaluate` scores the same cut windows. R4's 200-step cell and every T4 run train on chat
    samples, and how many are cut depends on the corpus and on `--max-seq-len`. T2's canonical cell is fixed by
    shape and is unaffected.
  - **Falsifiers (PROPOSED in `train-serve-chat-format-v1` on `la-72/k39-k40-contracts` @c93f1a5112).**
    - TSC-005: `train_step` and `evaluate` use a sample whole, or refuse it and count it, never a prefix. It is RED
      at `316dee2cd4` by reading, and the counts above say how often it fires. Planted: `full_ids[..max_seq_len]`
      restored.
    - TSC-006: `apr finetune` counts the samples over `--max-seq-len` before step 1, in its output and its receipt,
      and refuses unless told to drop them. It is RED at `316dee2cd4`: nothing counts them. Planted: the count
      removed.
    - TSC-007: the `--features wgpu` route trains on serve's ids, whole within `--max-seq-len`, on the recipe's
      targets, or refuses before the model loads, naming itself and `--gpu-backend cuda`. It is RED at `316dee2cd4` by
      reading on four counts: raw text with no eos, a 512 window, a silent cut and two targets. Planted: `auto`'s
      route to wgpu for `-m qlora` restored with no refusal.
  - **Until it lands,** count the corpus with the model's tokenizer.json before a run, add K39 and K40's inflation
    (about 12% on the curated corpus), and set `--max-seq-len` above the longest sample, or drop the long samples by
    hand.
- **Row 28, K43: a QLoRA run trains on one base, and the user is served another.** Desk read at `316dee2cd4`, plus an
  NF4 round trip in numpy of Qwen3.5-4B's shipped bf16 weights. No apr code was run.
  - **At `316dee2cd4`.**
    - `apr finetune -m qlora` sets `quantize_nf4` (`finetune.rs:336`). The CUDA path then quantizes each frozen weight
      to NF4 at upload (`transformer/cuda_block.rs:2943`): blocks of 64 values, one f32 absmax each, the QLoRA
      codebook (aprender-gpu `kernels/quantize/nf4_cpu.rs:23,120`). The adapter's gradients flow through that base, so
      the adapter learns a correction to NF4(W).
    - `apr finetune merge` adds the adapter to the base file's own tensors (`finetune_display_next_validate.rs:469`),
      so the user is served W + ΔW. Neither the merge nor any receipt names the base the run trained on.
    - The code already keeps the two apart elsewhere. The CUDA parity probe passes the CPU model's weights through the
      same NF4 round trip "so the replay isolates STRUCTURAL divergence from quantization noise"
      (`instruct_pipeline/parity_probe.rs:121-133`), and QQE-005 puts both sides of R4's gradient oracle on the NF4
      base (S-R4c).
  - **Measured.** The round trip with the CUDA path's block layout (64 consecutive values, nearest NF4 code) was run on
    the first 256 rows of each of R4's 10 target kinds: q, k, v and o and the three MLP projections in layer 3, and
    `in_proj_qkv`, `in_proj_z` and `out_proj` in GDN layer 0. It moves each by 9.2–9.5% of its Frobenius norm (median
    9.33%). The trained model and the merged model differ on every matrix the adapter touches.
  - **Effect on 0.72.** R4's loss gates (QQE-001, QQE-004) score NF4(W) + ΔW, the model that was trained. The user
    runs W + ΔW, from the GGUF exported from the merge. QQE-003 named no base. If its train side was the trainer's own
    model, it compared two models that differ for a reason that is not a bug. If its train side was `merge_into` on
    the file's base, it never saw the trained model. Either way, no gate asks whether the served model keeps what the
    run learned. T2's bf16 cell trains and merges on one base and is unaffected, and so is QQE-004's LoRA reference.
    A T4 rc built from a QLoRA adapter is affected.
  - **Falsifiers (PROPOSED in `qwen35-qlora-e2e-v1` on `la-72/r15-receipt-ext` @75f11cd071).**
    - QQE-003 now names its base: both sides are on the base `apr finetune merge` reads. The identity then measures
      layout, scale, naming and export only.
    - QQE-010: after R4's last step, on the first 20 pinned samples, the merged file, read back through row 4b's
      loader, scores within 2% of the trained model on the training loss, and below its own base. The receipt names
      both bases and the three losses. If the 2% clause fails, the merge takes the NF4 base instead and the receipt
      says so. Planted: a merge that drops the adapter fails the second clause. A train side scored on the merge's
      base passes vacuously, so the test first checks that the two bases differ by 5–15% in a QLoRA run.
  - **Until it lands,** when R4 runs, score the merged file on the same 20 samples with the training loss and record
    both losses before calling the run a pass.
- **Row 29, K44: alpha does not reach CUDA QLoRA training.** Desk read at `316dee2cd4`. No code was run.
  - **At `316dee2cd4`.**
    - The NF4 block multiplies B by s = alpha/r once, at upload, "to avoid a separate scale kernel in forward"
      (`transformer/cuda_block.rs:3025-3038`). After that, nothing in training reads s. The forward adds (x·A)·B'
      (`:3272`, `:3326`). The backward computes dB' = (x·A)ᵀ·dq and dA = xᵀ·(dq·B'ᵀ) (`:4433`, `:4456`, `:4594`,
      `:4608`), and AdamW steps B' itself (`:5098`, `:5130`). The comment at `:4434` says the gradient "includes the
      scale factor", but no s appears in it.
    - B starts at 0 on both trainers (`lora/layer/core.rs:109`, `transformer_trainer/cuda_trainer.rs:783`), so
      B' = s·0 = 0 for every alpha. The (A, B') trajectory is the same for every alpha until the save divides B' by s
      (`instruct_pipeline/accessors.rs:84-94`).
    - The CPU path applies s in the forward (`lora/layer/core.rs:278`) and AdamW steps B, as PEFT does and as
      `adapter_gradient_reference` in `qwen35-qlora-e2e-v1` states. AdamW's step does not change when the gradient
      is rescaled (up to ε), so on CUDA B moves 1/s as far per step. `apr finetune` sets alpha = 2·rank
      (`apr-finetune-v1`, `alpha_rank_ratio`), so s = 2 by default. T2's canonical task also says alpha 32 at r16.
    - Two paths are not affected. The merge adds s·(B'/s)·A = B'·A, which is the function CUDA trained. The
      transformer trainer's checkpoint saves and restores B' unchanged (`cuda_trainer.rs:3187-3210`, `:3396-3420`).
  - **Effect on 0.72.**
    - One recipe trains one model on CUDA and another on the CPU, and the CUDA one also differs from what PEFT or
      Unsloth train at the same alpha. No gate sees this. `cuda-nf4-train-loss-parity-v1` compares at B = 0, and
      QQE-004 compares two CUDA runs.
    - R15a's C1 moves the inline Q backward into `lora_backward`, gated on bit-identical gradients, so every CUDA LoRA
      cell built on it inherits the same rule unless C1 changes it. That covers R4's 7 targets, R21's GDN targets and
      T2's bf16 cell.
    - QQE-005 would see this as a backward defect, because the CUDA dB is the reference's dB/s (rel 0.5 at s = 2).
      A test that rescaled by s to pass would hide it.
    - T2's ratio is unaffected, because tokens per second do not depend on alpha.
  - **Falsifiers (PROPOSED in `qwen35-qlora-e2e-v1` on `la-72/r15-receipt-ext` @5956c65452).**
    - QQE-011: two runs that differ only in alpha (16 and 32 at r16) each take one AdamW step from B = 0.
      - The delta the merge adds, read from each written adapter, is twice as large at alpha 32, within 2e-2.
      - In each run, the trainer's own loss after the step equals the loss of its base plus the file's delta, within
        1e-4.
      - The CPU path is the control and is GREEN at `316dee2cd4`. The CUDA path is RED there, with a ratio of 1.
      - Planted: the test refuses equal alphas. An export that drops the division by s passes the ratio check and
        fails the loss check.
    - QQE-005 now compares dB as AdamW receives it, for the tensor the file stores, and never rescales it.
  - **The fix belongs in C1,** which already rewrites this code. Keep B unscaled on the GPU, apply s in the forward
    and in both backward products, and drop the ×s at upload and the ÷s at save. Not built, not measured.
    The wgpu pipeline already follows this rule and is an in-tree reference: its forward adds s·(x·A)·B
    (`finetune/wgpu_pipeline.rs:285`, `:812`), its backward scales both products by s (`:1115-1139`), and its export
    writes the raw B with alpha in the metadata (`:686-756`).
  - **Until it lands,** run CUDA LoRA at alpha = rank, where the two rules agree (s = 1), and record alpha in the
    receipt.

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
| 14 | R9 #4418 GGUF name map | — | owned by 0.71 | fix not on main at `316dee2cd4`; if it slips, R10's GGUF legs and T4/T5 slip with it, R4's QQE-003 has no served side, and row 4b copies its transforms (+20 `[A]`); its branch (@1af0e3cc11) also fixes the exported file's tokenizer block, but takes the template only from the .apr or a `chat_template.jinja` beside it (TSC-004, row 24); its export also prefers the .apr's rope_theta to the `config.json` beside it, so a .apr imported before it exports at 1e4 with no warning (QFR-008, row 26) |
| 15 | R8 GDN quantize policy | — | — | `79/r8-gdn-quant-policy` |
| 16 | R17 memory, measured | peak-memory run on the 4090 | 30 `[A]` | needs GPU at train-idle |
| 17 | R18 vocab alignment | — | — | `76/0.72-r18-vocab-cell`; 3 Definition-of-Ready tests shared with R6 |
| 18 | R16 dangling `qlora-training-loop-v1` | — | — | `la/r16-qlora-loop-contract` |
| 19 | R14 throughput work | sized from the R5 gap | — | after R5 and R15b |
| 20 | K30 `apr train` tie flag | TOC-001/002: config.json says tied exactly when the saved model has no head | 15 `[A]` | `la-72/k30-train-tie-flag` @687554a60a (`apr-train-output-config-v1`); off the critical path; opens with the cheap refusals after LIVE 0.70.1 |
| 21 | K36 GDN contract text | restate `gated-delta-net-v1`'s decay, read and output; point its tests at the shipped decay; re-prove GDN-BND-001 | 60 `[A]` | contracts on `la-72/k36-gdn-contract` @e8834d7711: both at 2.0.0, bindings point at the served fns, allowlist 163→156. The tests and the Lean re-proof come at PR time, after LIVE 0.70.1. `qwen35-train-gdn-v1` @80723cf206 already states the served GDN |
| 22 | K37 phantom bindings | fold into #4502: make `pv audit --binding` agree with the `bindings` gate; arm the gate once the `gated_rmsnorm_oxide` ghost is fixed or allowlisted | 10 `[A]` (was 30; the gate exists) | comment on #4502 posted; the K36 branch fixes 6 of the 9 phantoms at `316dee2cd4` |
| 23 | K38 Qwen3.5 norm convention | serve's safetensors conversion refuses a hybrid `layer_types`; R13's builder inverts #4418's value transforms, norm −1 included | 15 `[A]` + R13's builder | desk read plus a CPU measurement on the 4B; GGUF serve, train and #4418 agree |
| 24 | K39 train/serve chat format | the HF importer writes the added tokens and the chat template into the .apr; TSC-001: `from_apr`'s tokenizer keeps `<\|im_start\|>`, `<\|im_end\|>`, `<think>` and `</think>` whole; TSC-002: train renders the model's own template, thinking off, no default system turn, target ends in the eos id; TSC-003: serve's built-in `Qwen3NoThink` renders as the model's own template does; TSC-004: the GGUF exported from a merged HF-sourced base carries the model's template, pre `qwen35`, the eos and the token types | 110 `[A]` | contract PROPOSED @ae7a75b7f0 (pv 0/0); desk read at `316dee2cd4` plus a header read of S-R10's .apr (17 keys, no template, no added tokens) and of its GGUF export (18 keys: pre `default`, no eos, token types or template); #4418's branch fixes all but the template; PMAT-3803's branch has part of the tokenizer half, unmerged; must be green before R4's 200-step cell and any T4 run |
| 25 | K40 pre-tokenizer split | one regex pre-tokenizer shared by train, `apr chat` and .apr serve; TPP-001/002: train's tokenizer and `encode_text` give the HF reference ids on a frozen code fixture; TPP-003 keeps GGUF serve on them, on the file apr exports too | 80 `[A]` | contract PROPOSED @ae7a75b7f0 (pv 0/0); desk read at `316dee2cd4` plus a simulation on the Qwen3.5 vocabulary: 5 samples are 193 tokens in train against 160 under the regex, and .apr serve has the same count with different ids on indented code; CRUX-M-05 (draft) states the check and nothing implements it; PMAT-3803's branch moves Qwen3.5 training to .apr serve's no-split path, not the regex (the HF-sourced .apr has no `pre_type` and says `qwen3_5`); T2 is unaffected because its count is fixed by shape; must be green before R4's 200-step cell and any T4 run |
| 26 | K41 Qwen3.5 rope base and dims | QFR-007: the imported .apr and the GGUF exported from its merge carry the source config's dims and rope base (1e7); QFR-008: `apr export` refuses a qwen35 .apr that disagrees with its source config; QFR-009: the qwen35 rope fallbacks, the 9B preset and the family contract say 1e7 | 40 `[A]` | QFR-007..009 PROPOSED @e6ea295728; desk read at `316dee2cd4` and on #4418's branch, plus header reads: S-R10's `hf.apr` says 10000 with no dims, and its GGUF says 10000, heads 16/8, ctx 0; the published 0.8B to 27B say 1e7; #4418 fixes a fresh import but exports an older .apr at 1e4 with no warning; QFR-006 checks fresh imports only, and QQE-003 cannot see it; re-import S-R10's base after #4418, never reuse it; must be green before R4's 200-step cell and any T4 run |
| 27 | K42 training window | TSC-005: `train_step` and `evaluate` use a sample whole or refuse and count it, never a prefix; TSC-006: `apr finetune` counts the samples over `--max-seq-len` before step 1 and refuses unless told to drop them; TSC-007: the `--features wgpu` route does the same or refuses by name | 45 + 15 `[A]` | TSC-005/006 PROPOSED @275c009e4a, TSC-007 @c93f1a5112 (pv 0/0); desk read at `316dee2cd4` plus a simulation on the repo's SFT corpora: on main's train tokenizer the default 512 cuts 123 of the 124 curated samples, each losing part of `<\|im_end\|>` and 68 also the `>` that closes `</tool_call>`; with serve's ids all fit; the CUDA step reports only a prompt that fills the window; a `--features wgpu` build trains on raw text at a fixed 512; must be green before R4's 200-step cell and any T4 run |
| 28 | K43 QLoRA merge base | QQE-010: the merged file scores within 2% of the trained model on the training loss, and below its own base; QQE-003 runs both sides on the base `apr finetune merge` reads | 40 `[A]` | QQE-010 PROPOSED @75f11cd071 on `la-72/r15-receipt-ext` (pv 0/0); desk read at `316dee2cd4`: `-m qlora` trains against the NF4 round trip of the base (`cuda_block.rs:2943`), and `apr finetune merge` adds the adapter to the unquantized file; the round trip moves each of R4's 10 target kinds by 9.2–9.5% (Frobenius, numpy on the bf16 weights); measured after R4's last step, so it gates R4's verdict and any T4 rc built from a QLoRA adapter; T2 unaffected |
| 29 | K44 LoRA scale in training | QQE-011: doubling alpha doubles the first step's merged delta on CUDA as on the CPU, and the written adapter carries the trainer's own function; QQE-005 compares dB as AdamW receives it | 30 `[A]` | QQE-011 PROPOSED @5956c65452 on `la-72/r15-receipt-ext` (pv 0/0); desk read at `316dee2cd4`: the NF4 block bakes s into B at upload (`cuda_block.rs:3025-3038`) and never applies it again, so alpha is inert on CUDA and B moves 1/s as far per step as on the CPU; the merge and resume are consistent; the fix goes in R15a's C1; must be green before R4's 200-step cell; T2's ratio unaffected |
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
K39 TSC, K40 TPP, K41 QFR-007..009, K42 ────► gate R4's 200-step cell and every T4 run
K43 QQE-010 ────────────────────────────────► gates R4's verdict and any T4 rc built from a QLoRA adapter
K44 QQE-011, fixed in R15a C1 ──────────────► gates R4's 200-step cell and QQE-005
```
T2 trains Qwen3.5-4B, so its apr side needs R2, R3, R21 and 4b as well as R15a and R15b. It does not need R4. R21
(400 + 25 `[A]`) is the largest row on both R4's path and T2's. Its LoRA wiring calls R15a's C1 helper, so C1
lands before R21's projection cell. The value-head work splits in two. The load-time conversion (4b) is on both
paths, because `apr finetune` trains from .apr only and the only Qwen3.5 .apr today is HF-sourced. The export
permutation (QQE-007) is R4's alone, because T2's canonical cell targets no GDN projection. Rows 24–27 (K39 the chat
format, K40 the pre-tokenizer split, K41 the rope base and dims, K42 the training window) are small, but none of R4's
gates sees them, and each changes what the model learns, is given or computes. They gate R4's 200-step cell and every
T4 run, not T2's ratio. Row 28 (K43, the QLoRA merge base) is measured only after R4's last step, so it gates R4's
verdict rather than the cell's start. It does not touch T2, which trains and merges on one base. Row 29 (K44, the
LoRA scale) is a small change inside R15a's C1, which already rewrites that code. It gates R4's 200-step cell, because
it changes how far B moves per step. It does not change T2's ratio.

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
| finetune (merge) | QQE-010 (PROPOSED, K43): after R4's last step, on the first 20 pinned samples, the merged file read back through row 4b's loader has a training loss ≤ 1.02 × the trained model's (NF4 base + adapter) and < its own base's. The receipt names both bases and the three losses | lambda | ≤ 1.02 and below the base, **or** the merge takes the NF4 base and the receipt says so | the train side was scored on the merge's base (the two bases differ by < 5%); the merged file was not read back through 4b's loader |
| finetune (LoRA scale) | QQE-011 (PROPOSED, K44): two one-step runs at alpha 16 and 32 (r16); the merged delta read from each written adapter doubles, on CUDA as on the CPU, and each trainer's loss equals its base plus the file's delta | lambda (CUDA), intel (CPU control) | ratio 2 ± 2e-2 and loss within 1e-4, on both devices | the two alphas were equal; Δ was read from the GPU buffers instead of the file |
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
3. QQE-005: CUDA adapter gradients against the CPU reference, on 0.8B. QQE-011's two one-step runs on a tiny config
   go first; they take seconds.
4. DBH GPU halves.
5. QQE-001..004 on 4B. QQE-004's LoRA reference needs R15a C4. QQE-010 follows on the same run's adapter.
6. The apr side of T2. It needs R15b.

CPU rows (TAH, TRR, MOF, QFR on 0.8B, HRP dry-run, TIS-001/003/004/005) run on intel at any time, load1 ≤ 32.
