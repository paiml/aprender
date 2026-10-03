# R15 re-size: CUDA LoRA cells (0.72, L2)

Status: desk read 2026-10-03 at origin/main `316dee2cd4`. Nothing here was built or run. `[V]` marks facts read
in the code, with file:line; `[A]` marks estimates. Paths are under `crates/aprender-train/src/` unless they start
with `apr-cli/`. This replaces R15's K̂ of 120 `[A]` in the 0.72 ranking v2 (§3, row 5, "re-size before R4").
Ranking v3 (§3, 2026-10-03) carries the new size, split into R15a (C1–C4) and R15b (C5–C7).

## What the code says `[V]`

S-R15 (2026-09-27, at shaping @5826e292bb) still holds line for line on `316dee2cd4`:

- CUDA is initialised only for NF4: `finetune/instruct_pipeline/constructors.rs:56,173,293`. The LoRA grad
  workspace and optimizer state exist only for NF4 (`instruct_pipeline/cuda_init.rs:66`), and so does the block
  backward (`instruct_pipeline/training.rs:309`). `apr-cli/src/commands/finetune.rs:280` warns and trains on
  the CPU.
- Only Q and V get adapters:
  - `build_lora_layers` (`constructors.rs:358`, pushes at `:383` and `:397`)
  - `inject_adapter_weights` (`:428`, which indexes `idx*2 + !is_q`)
  - `CudaLoraGradWorkspace` (`transformer/cuda_block.rs:3853`)
  - `GpuLoraOptimizerState` (`:3954`)
  - gradient clipping (`:3902-3936`)

  All five are hard-coded to `a_q, b_q, a_v, b_v`.
- The FP32 `CudaTransformerBlock` backward and `optimizer_step` (`cuda_block.rs:1276,2248`) train every weight;
  neither is LoRA.

New in this read:

- **No bf16 in the training block.** All `CudaTransformerBlock` buffers are `GpuBuffer<f32>`.
  - `gemm_forward_bf16` (`autograd/cuda_forward/matmul.rs:685`) is a hand-written PTX kernel with no non-test
    callers.
  - The bf16 cast kernels (`autograd/cuda_forward/bf16_cast.rs`) are exported but unused in `transformer/` or
    `finetune/`.
  - The only reduced-precision GEMM is FP16 cuBLAS (`matmul_f16.rs`), enabled by `FP16_GEMM=1` and only with NF4
    (`cuda_init.rs:363`).
- **The LoRA backward is written out inline, once per projection.**
  - It lives in `backward_nf4_attention` (`cuda_block.rs:4297`): Q at about `:4427-4485`, V at about `:4588-4625`.
  - It runs the same five GEMMs each time: recompute `X·A`, then `dB`, `dInter`, `dA` and `dX +=`.
  - The FFN backward (`backward_nf4_ffn`, `:4103`) has no LoRA code.
  - The AdamW kernel `adamw_step_cuda` (`autograd/cuda_optim.rs:182`) is generic over tensors; its callers are not.
- **No timed-window plumbing.**
  - Device syncs exist: `CudaTrainer::synchronize` (`autograd/cuda_training.rs:98`) and the stream syncs at
    `training.rs:415`.
  - `StepProfiler` (`train/transformer_trainer/step_profiler.rs:108`) times phases with host wall clock.
  - There is no CUDA event timer, no label-token count and no `[TRACE]` device line. The only `[TRACE]` lines are
    dequantisation debug prints.
- **The CPU instruct path has no target list.** The generic `LoRAConfig.target_modules` (`lora/config.rs:16`)
  already knows `q/k/v/o/gate/up/down` and the `all_linear` shorthand, but only `ClassifyPipeline` reads it.

## Cells `[A]`

| # | Cell | Serves | K̂ (min) | Falsifier |
|---|---|---|---|---|
| C1 | Extract `lora_backward(x, a, b, dy, da, db, dx)` from the inline Q block; switch Q and V to it | R4, T2 | 30 | Q/V adapter grads bit-identical before and after on a tiny NF4 run (the refactor changes nothing) |
| C2 | Target list: the workspace, optimizer state, clipping and NF4 adapter fields become a `Vec` keyed by target kind; `build_lora_layers` and `inject_adapter_weights` read `LoRAConfig.target_modules` | R4, T2 | 60 | `--targets q_proj,v_proj` gives today's tensor set; `all_linear` gives 7 adapters per layer; a dropped target is RED (FALSIFY-FT-TASK-003) |
| C3 | Call C1 for k and o in the attention backward, and add LoRA to the FFN backward (gate, up, down) | R4, T2 | 60 | after N CUDA steps every one of the 7 adapter kinds has changed, and the loss moved (S-R15's own falsifier) |
| C4 | Frozen-base non-NF4 LoRA block: the FP32 `CudaTransformerBlock` forward plus C1–C3 backward, base weights frozen (no full-weight AdamW), and `init_cuda` reached for `-m lora` | R4 (QQE-004 reference), T2 | 60 | base weight checksums unchanged after N steps; adapters changed; `finetune.rs:280` no longer falls back to the CPU |
| C5 | bf16 frozen base: bf16 weight storage plus bf16 cuBLAS GEMM (`CUDA_R_16BF`, following `matmul_f16.rs`), fp32 adapters and optimizer state | T2 (R4 only if the QQE-004 reference must be bf16) | 120 | `recipe.precision` reads `bf16` from the loaded weights; loss within tolerance of the C4 fp32 run over 20 steps |
| C6 | `apr finetune` flags (apr-finetune-canonical-task-v1) and the TRR 1.1.0 receipt writer, coordinated with the R12 owner | T2 | 45 | FALSIFY-FT-TASK-001/002, FALSIFY-TRR-007/010 |
| C7 | Timed window: device sync at both edges, label-token count, `after_compile` (no PTX module load inside the window), `[TRACE]` device line with name and UUID | T2 | 45 | FALSIFY-TRR-009 (33 tokens); a planted PTX load inside the window sets `after_compile` false |

**Total: 420 `[A]`, against 120 `[A]` in ranking v2.** R4 needs C1–C3 (150), and C4 (60) as well: QQE-004's
reference run is `-m lora` on CUDA (qwen35-qlora-e2e-v1 1.1.0), which makes R4's share 210. C5, at 120, is needed
only for T2, or for R4 if the QQE-004 reference must be bf16 and not fp32. C6 (45) is needed by T2 and shared with
R12.

## Consequences

- **Order:** C1 → C2 → C3 → C4 first, because R4 needs them anyway (NF4 + 7 targets, plus the GDN projections after
  R2/R3, and C4 for the QQE-004 `-m lora` reference). Then C5 → C6 → C7 for T2. C6 can run in parallel with anything.
- **C5 is the largest single cell, and only T2 needs it** (R4 takes an fp32 reference from C4, recorded as
  `recipe.precision = fp32`). The verdict now pins `precision=bf16` on both sides
  (beat-unsloth-finetune-throughput-v1 1.3.0, FALSIFY-BEAT-UNSLOTH-FT-PRECISION). An R15 built without C5 makes T2
  fail SAME-WORK instead of quietly comparing fp32 apr with bf16 Unsloth. Before 1.3.0 that mismatch would have
  gone into the ratio unseen.
- **Ruling requested (RQ-5, spec §4; not blocking):** if C5 does not fit 0.72, should T2
  - (a) stay bf16 and slip to 0.73, or
  - (b) gain a declared second cell, "apr fp32 vs Unsloth fp32"?

  (b) needs the incumbent run in fp32 as well, never a cross-precision ratio. Recommendation: (a). The beat
  claim is about the configuration Unsloth users run, and a fp32-vs-fp32 cell answers a different question.
