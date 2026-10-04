# R15 re-size: CUDA LoRA cells (0.72, L2)

Status: desk read 2026-10-03 at origin/main `316dee2cd4`. Nothing here was built or run. `[V]` marks facts read
in the code, with file:line; `[A]` marks estimates. Paths are under `crates/aprender-train/src/` unless they start
with `apr-cli/`. This replaces R15's K̂ of 120 `[A]` in the 0.72 ranking v2 (§3, row 5, "re-size before R4").
Ranking v3 (§3, 2026-10-03) carries the new size, split into R15a (C1–C4) and R15b (C5–C7). C5 was re-sized from
the code the same day (§C5, sized from the code), which moves R15 from 420 to 450.
The C1–C7 falsifiers are PROPOSED rows, named in the table: C1–C5 in the existing LoRA contracts, C5–C7 in
train-run-receipt-v1 1.1.0 and apr-finetune-canonical-task-v1 1.1.0. Each test is `[U]` until cargo (and, for the
CUDA rows, a GPU) is allowed after LIVE 0.70.1.

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
    `dInter` is written over the `X·A` scratch (`lora_inter`), so `dB` has to run first. No step applies
    alpha/rank: the scale sits in B, which is uploaded already multiplied by s (K44, spec row 29).
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
| C1 | Extract `lora_backward(x, a, b, dy, da, db, dx)` from the inline Q block; switch Q and V to it | R4, T2 | 30 | Q/V adapter grads of step 2 bit-identical before and after on a tiny NF4 run at alpha = 2·rank (the refactor changes nothing; at step 1, B = 0 makes dA zero on both sides). K44's fix lands after it as its own commit, and the test is then re-run at alpha = rank. FALSIFY-LORA_GRADIENT_FLOW_V1_004 |
| C2 | Target list: the workspace, optimizer state, clipping and NF4 adapter fields become a `Vec` keyed by target kind; `build_lora_layers` and `inject_adapter_weights` read one target list in `InstructConfig` | R4, T2 | 60 | `--targets q_proj,v_proj` gives today's tensor set; `all_linear` gives 7 adapters per layer; a dropped target is RED (FALSIFY-FT-TASK-003). FALSIFY-LORA_TARGET_SELECTION_V1_004 (C2b) and _005 (C2c) |
| C3 | Call C1 for k and o in the attention backward, and add LoRA to the FFN backward (gate, up, down) | R4, T2 | 60 | after N CUDA steps every one of the 7 adapter kinds has changed, and the loss moved (S-R15's own falsifier). FALSIFY-LORA_GRADIENT_FLOW_V1_005 |
| C4 | Frozen-base non-NF4 LoRA block: the FP32 `CudaTransformerBlock` forward plus C1–C3 backward, base weights frozen (no full-weight AdamW), and `init_cuda` reached for `-m lora` | R4 (QQE-004 reference), T2 | 60 | base weight checksums unchanged after N steps; adapters changed; `finetune.rs:280` no longer falls back to the CPU. FALSIFY-LORA-ADAPTER-TRAINS-BASE-FROZEN-CUDA-003 |
| C5 | bf16 frozen base on C4's non-NF4 block: the file's bf16 weights uploaded as bf16, a bf16-input cuBLAS GEMM (`CUDA_R_16BF`, fp32 accumulate, following `matmul_f16.rs`) at the 14 base-weight GEMM sites and the lm_head, and a round-to-nearest-even cast for activations and gradients; fp32 adapters and optimizer state | T2 (R4 only if the QQE-004 reference must be bf16) | 150 | `recipe.precision` reads `bf16` from the loaded weights; loss within tolerance of the C4 fp32 run over 20 steps. FALSIFY-LORA-ADAPTER-TRAINS-BASE-FROZEN-CUDA-BF16-004, FALSIFY-TRR-012 |
| C6 | `apr finetune` flags (apr-finetune-canonical-task-v1) and the TRR 1.1.0 receipt writer, coordinated with the R12 owner | T2 | 45 | FALSIFY-FT-TASK-001/002, FALSIFY-TRR-007/010, the CPU half of FALSIFY-TRR-012 |
| C7 | Timed window: device sync at both edges, label-token count, `after_compile` (no PTX module load inside the window), `[TRACE]` device line with name and UUID | T2 | 45 | FALSIFY-TRR-009 (33 tokens); a planted PTX load inside the window sets `after_compile` false. FALSIFY-TRR-013 |

**C2 lands in three parts.** C2a makes `inject_adapter_weights` route each tensor of a loaded PEFT adapter by its
name, or refuse the whole load, and owns row 003 of `lora-target-selection-v1`. C2b is the host half of the C2 row
above, and row 004: the target list in `InstructConfig`, `build_lora_layers` and `inject_adapter_weights` by one slot
map, and a check in every instruct-pipeline constructor that refuses any list but `q_proj`, `v_proj`, by name. C2c is
the rest, and row 005: the workspace, optimizer state, clipping and NF4 adapter fields keyed by target. The check
relaxes only when the forward, the CUDA blocks, their sync and the backward cover every selected target, which takes
C2c and C3. Until then the control of FALSIFY-FT-TASK-003 is `--targets q_proj,v_proj`.

**Total: 450 `[A]`, against 120 `[A]` in ranking v2.** R4 needs C1–C3 (150), and C4 (60) as well: QQE-004's
reference run is `-m lora` on CUDA (qwen35-qlora-e2e-v1 1.1.0), which makes R4's share 210. C5, at 150 (sized from
the code below; it was 120 until then), is needed only for T2, or for R4 if the QQE-004 reference must be bf16 and
not fp32. C6 (45) is needed by T2 and shared with R12. No cell moves a GDN layer to CUDA (§Consequences, last
bullet).

## C5, sized from the code (2026-10-03)

The 120 in the first version of the table was a guess. A desk read of the code C5 has to change gives 150 `[A]`
(range 130–165). Paths are under `crates/aprender-train/src/`, except the `aprender-gpu` driver files.

- **No bf16-input GEMM exists.** `[V]`
  - `gemm_f16_to_f32` (`aprender-gpu/src/driver/cublas.rs:325`) passes `CUDA_R_16F` for A and B, `CUDA_R_32F` for C
    and `CUBLAS_COMPUTE_32F` (`:353-363`).
  - `CUDA_R_16BF` (14, `cublas_sys.rs:81`) appears only as the output type of the FP8 cuBLASLt path
    (`cublaslt.rs:52`, `:536`).
  - `GpuBuffer<u16>` cannot tell f16 from bf16, so the dtype has to travel with the buffer.
- **The GPU bf16 cast truncates.** `[V]`
  - `cast_f32_to_bf16_gpu` (`autograd/cuda_forward/bf16_cast.rs:122`) keeps the high 16 bits
    (`shr_u32_imm(bits, 16)`, `:59`). The CPU helper (`:210`, `half::bf16::from_f32`) rounds to nearest even.
  - Truncation is the planted mutant of FALSIFY-LORA-ADAPTER-TRAINS-BASE-FROZEN-CUDA-BF16-004 (a), so C5 cannot use
    the kernel as it stands. The fix is `(bits + 0x7FFF + ((bits >> 16) & 1)) >> 16` with a NaN guard, or
    `cvt.rn.bf16.f32` on sm_80 and later `[A]`.
  - Both cast kernels target sm_70 and have no callers.
- **Today's fp16 base is derived from NF4.** `[V]`
  - `set_fp16_weights` (`transformer/cuda_block.rs:3149`) casts the CPU-dequantised NF4 copies (`:2975-3018`), and
    `FP16_GEMM=1` is selected only with NF4 (`instruct_pipeline/cuda_init.rs:363`).
  - So C5 builds on C4's non-NF4 block and uploads the file's bf16 bytes, one layer at a time. A bf16 GEMM on the
    NF4 path would train an NF4-rounded base and has to record `recipe.precision = nf4` (TRR 1.1.0).
- **Fourteen GEMM sites read base weights.** `[V]`
  - Seven forward GEMMs (`cuda_block.rs:818-928`) and seven `gemm_backward_a` calls (`:1353-1927`).
  - The seven `gemm_backward_b` calls go away with the frozen base (C4).
  - Activations and gradients need a cast on the way in, as on the fp16 path (`cuda_block.rs:3246`,
    `autograd/cuda_backward/gemm.rs:284-286`).
- **The lm_head GEMM is fp32 on the GPU** (`instruct_pipeline/cuda_forward.rs:335`, `:520`; backward
  `instruct_pipeline/training.rs:266`). `[V]` Qwen3.5-4B ties the head to its 248320×2560 embedding, so the head is
  part of the bf16 base and C5 runs it in bf16 as well.

| Part | K̂ `[A]` |
|---|---|
| `gemm_bf16_to_f32` in the driver, after `gemm_f16_to_f32` (`cublas.rs:325`) | 15 |
| Training-side wrappers after `matmul_f16.rs:125` and `:79`, with the operand swap at `:143` | 15 |
| Round-to-nearest-even cast, bit-tested against `half::bf16::from_f32` | 20–30 |
| Per-layer bf16 upload with a dtype tag on the buffer | 15–20 |
| The 14 GEMM sites behind one dtype dispatch, plus the activation and gradient casts | 40–55 |
| lm_head in bf16: both forward sites and the backward | 15–20 |
| `--precision` selector, and a refusal on a GPU without bf16 GEMM (below sm_80 `[A]`) | 10 |
| **C5** | **130–165, carried as 150** |

On the NF4 path the same work would be 170–200 and would train an NF4-rounded base, so C5 stays on C4's block. With
C5 at 150, R15 is 450: R15a 210, R15b 240.

## Consequences

- **Order:** C1 → C2 → C3 → C4 first, because R4 needs them anyway (NF4 + 7 targets, and C4 for the QQE-004
  `-m lora` reference). Then C5 → C6 → C7 for T2. C6 can run in parallel with anything. R4's GDN projections are not
  an R15 cell: they need a CUDA GDN block first (last bullet).
- **C5 is the largest single cell, and only T2 needs it** (R4 takes an fp32 reference from C4, recorded as
  `recipe.precision = fp32`). The verdict now pins `precision=bf16` on both sides
  (beat-unsloth-finetune-throughput-v1 1.3.0, FALSIFY-BEAT-UNSLOTH-FT-PRECISION). An R15 built without C5 makes T2
  fail SAME-WORK instead of quietly comparing fp32 apr with bf16 Unsloth. Before 1.3.0 that mismatch would have
  gone into the ratio unseen.
- **The verdict half of C7 already exists.** `check_pins` (`scripts/bench/unsloth_ft_verdict.py:103`) pins
  `timed_after_compile` true on every run of both sides (test case "timed before compile"), so C7 only has to
  measure it: a count of module loads, JITs and graph captures read at both window edges (FALSIFY-TRR-013).
- **Ruling requested (RQ-5, spec §4; not blocking):** if C5 does not fit 0.72, should T2
  - (a) stay bf16 and slip to 0.73, or
  - (b) gain a declared second cell, "apr fp32 vs Unsloth fp32"?

  (b) needs the incumbent run in fp32 as well, never a cross-precision ratio. Recommendation: (a). The beat
  claim is about the configuration Unsloth users run, and a fp32-vs-fp32 cell answers a different question.
- **R15 trains the dense layers only, and no row moves GDN to CUDA.** `[V]`
  - 24 of Qwen3.5-4B's 32 layers are `linear_attention` (GDN); every fourth layer is full attention (HF config).
  - The CUDA trainer loads q/k/v/o and gate/up/down for every layer (`instruct_pipeline/cuda_init.rs:198-214`).
  - aprender-train has no GDN code at `316dee2cd4`. R2/R3's GDN forward and backward (`la-72/fold-r2r3`
    @5a837dfa3b: `transformer/gdn.rs`, `gdn_backward.rs` and their two test files) are CPU only; none of the four
    files mentions CUDA. Serving's CUDA GDN (`aprender-serve/src/cuda/executor/gdn_ops.rs`) is forward only.
  - S-R15 named the need at shaping ("GDN forward and backward as CUDA blocks", spec §2). R4, T2 and R6 on qwen35
    all wait on it. Ranking v2 and v3 sized no row for it.

  Proposed as R21 in spec §3: a correct CUDA GDN block, forward and backward, with LoRA on the GDN projections
  (attn_qkv, attn_gate, ssm_out in qwen35-qlora-e2e-v1), checked against the CPU R3 oracle. K̂ was 360–480 `[A]` before
  spike S-R21, for causal conv1d, the q/k L2 norm, the gates, a sequential delta-rule scan with checkpointed state, and
  the gated RMSNorm, each forward and backward. A chunked kernel fast enough to race fla is R14's job. Running the
  GDN layers on the CPU inside the CUDA pipeline would be correct but slow (S-R4a: 7.2 s a step for 5 tokens at 0.8B
  on the CPU), so the estimate does not assume it. Spike S-R21 (2026-10-03, `r21-cuda-qwen35-hybrid-block.md`)
  sized R21 at 400 `[A]` plus 25 `[A]` for bf16 on T2's path. It widened the row to the gated full-attention
  layers, whose output gate and partial RoPE the CUDA trainer also lacks.
