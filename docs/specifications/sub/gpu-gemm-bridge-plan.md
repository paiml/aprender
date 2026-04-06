# GPU GEMM Bridge Plan: 0.39x → 0.6x+ cuBLAS

**Date**: 2026-04-06
**Status**: RESEARCH COMPLETE — Ready for Implementation
**Ref**: CGP spec P3c, trueno#200, trueno#203

## The Gap

| Metric | trueno PTX | cuBLAS | Ratio | Target |
|--------|-----------|--------|-------|--------|
| 1024 TFLOP/s | 40.5 | 104.9 | 0.39x | 0.6x (63 TFLOP/s) |
| 4096 TFLOP/s | — | 150.0 | — | 0.5x (75 TFLOP/s) |
| FP16 peak % | 12.3% | 31.8% | — | 19%+ |

## Research: What the Best Implementations Do

### CUTLASS 3.x (NVIDIA, SM 8.0+)

From CUTLASS source and documentation [48]:

**Tile hierarchy for SM 8.9 (Ada)**:
- CTA tile: **128×128** or **256×128**
- Warp tile: **64×64** (4 warps per warp group)
- MMA instruction: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`
- K-tile: 32 (two m16n8k16 along K per stage)

**Pipeline**: 3-5 async stages with `cp.async`
- Stage 0: global → shared (cp.async)
- Stage 1: shared → register (ldmatrix)
- Stage 2: compute (mma.sync)
- Stages overlap via cooperative scheduling

**Shared memory**: 64-164 KB (opt-in via `cuFuncSetAttribute`)
- 128×32 A tile × FP16 = 8 KB per stage
- 32×128 B tile × FP16 = 8 KB per stage
- 4 stages × 16 KB = 64 KB total

**Key insight**: `mma.sync.aligned.m16n8k16` processes a 16×8 output
fragment per warp (vs `wmma.mma.sync.m16n16k16` which processes 16×16
but at lower IPC). The m16n8k16 instruction has 2× instruction throughput
because each instruction computes half the output — the hardware can
issue them back-to-back without stalling.

### HGEMM Step-by-Step (xlite-dev/HGEMM, GitHub)

This repo demonstrates the optimization path from naive to near-cuBLAS:

| Stage | Technique | TFLOP/s (A100) | % peak |
|-------|-----------|----------------|--------|
| V1 | Naive | 2.4 | 0.8% |
| V2 | Shared memory tiling | 13.0 | 4.2% |
| V3 | 1D block tiling | 41.2 | 13.2% |
| V4 | 2D block tiling | 94.5 | 30.3% |
| V5 | Register-level mma.sync | 135.4 | 43.4% |
| V6 | Double buffer + swizzle | 171.0 | 54.8% |
| V7 | CUTLASS-3 style | 225.4 | 72.3% |

**V5→V6 (+26%)**: Double-buffering shared memory + swizzle pattern eliminates
bank conflicts. The swizzle permutes shared memory addresses so that warp-level
loads hit different banks.

**V6→V7 (+32%)**: CUTLASS-3 warp specialization — some warps do loads while
others do compute. Requires cooperative launch.

### llama.cpp GPU GEMM Strategy

llama.cpp uses cuBLAS for all GEMM >512 elements. For quantized GEMV (Q4_K),
they have custom CUDA C kernels with:
- One warp per output row
- `__shfl_xor_sync` for warp-level reductions
- Direct dequant-fused dot product (no separate dequant pass)

For the Q4_K GPU kernel, llama.cpp achieves ~500 GFLOPS on RTX 4090 for
large matrices — this is the target for trueno's DP4A Q4K kernel (#175).

### Burn / CubeCL (Rust)

**burn** (tracel-ai/burn): Uses cuBLAS via cubecl for production GEMM.
Their custom GEMM kernel in cubecl achieves ~30-40% of cuBLAS — similar
to our current position. They explicitly fall back to cuBLAS for production.

**cubecl** (tracel-ai/cubecl): Rust GPU compute framework. Has a GEMM
kernel with tiling and shared memory. Not competitive with cuBLAS for
production use. Their approach: compile Rust-like syntax to CUDA/Vulkan.

### Candle (HuggingFace)

Uses cudarc → cuBLAS exclusively for GEMM. No custom kernel. Their finding:
switching from `CUBLAS_COMPUTE_32F` to `CUBLAS_COMPUTE_16F` gives +15%.

### Key arXiv References

**[48] CUTLASS 3.0** (Thakkar et al., 2023): Describes the producer-consumer
warp specialization pattern. "Epilogue fusion" eliminates store-load round
trips for activation functions.

**CUDA-L2** (arXiv:2512.02551): Uses reinforcement learning to generate
CUDA kernels that beat cuBLASLt-AutoTuned on A100 by 11.4%. The key
insight: optimal tile configurations are hardware-specific and cannot be
derived analytically.

**"Outperforming cuBLAS on H100"** (cudaforfun.substack.com): Demonstrates
that hand-written CUDA can beat cuBLAS for specific shapes by using:
- TMA (Tensor Memory Accelerator) — Hopper only
- WGMMA (Warp Group MMA) — Hopper only
- Persistent kernels with cluster support

## Root Cause Analysis: Why 0.39x

### Issue 1: wmma.m16n16k16 vs mma.sync.m16n8k16

Our kernel uses `wmma.mma.sync.aligned.m16n16k16`:
- Processes 16×16 output per warp per instruction
- **One instruction per 16 cycles** on SM 8.9
- 16×16×16 × 2 = 8192 FLOPs per instruction
- Throughput: 8192/16 = **512 FLOP/cycle/warp**

cuBLAS uses `mma.sync.aligned.m16n8k16`:
- Processes 16×8 output per warp per instruction
- **One instruction per 8 cycles** on SM 8.9 (2× issue rate)
- 16×8×16 × 2 = 4096 FLOPs per instruction
- Throughput: 4096/8 = **512 FLOP/cycle/warp**

**Same theoretical throughput!** The difference is practical:
- `mma.sync` fragments are smaller → more flexible scheduling
- `mma.sync` uses 4 registers per fragment vs 8 for `wmma` → less pressure
- `mma.sync` enables `ldmatrix` instruction for efficient shared→register

### Issue 2: 64×64 CTA vs 128×128 CTA

| Metric | 64×64 | 128×128 | Improvement |
|--------|-------|---------|-------------|
| Compute-to-load (FLOP/byte) | 32 | 64 | **2×** |
| WMMAs per K-tile | 16 | 64 | **4×** |
| Smem per stage (A+B) | 4 KB | 16 KB | 4× |
| Smem 4-stage pipeline | 16 KB | 64 KB | 4× |

With 128×128 tiles, we load 4× less data per FLOP → memory latency is
hidden by 4× more compute per loaded tile.

### Issue 3: 2-stage vs 4-stage pipeline

Our kernel: 2-stage cp.async double-buffer (8 KB smem).
CUTLASS: 3-5 stage pipeline (48-80 KB smem).

More stages = more in-flight loads = better latency hiding:
- Global memory latency: ~400 cycles on RTX 4090
- Each cp.async group loads one K-tile
- 2 stages: hide 1 × 400 = 400 cycles
- 4 stages: hide 3 × 400 = 1200 cycles

### Issue 4: No ldmatrix instruction

`ldmatrix` loads 4 matrix fragments from shared memory in one instruction,
perfectly mapped to `mma.sync` operand layout. Without it, we use 8 individual
`ld.shared` instructions per fragment → 8× instruction overhead for shared
memory reads.

## Bridge Plan: 0.39x → 0.6x+

### Phase 1: Add `mma.sync.m16n8k16` to PTX builder (Est: +20%) ✅ DONE

**What**: Add `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16` as new
PtxOp variant. Each `mma.sync` uses:
- 2 registers for A fragment (4 FP16 values)
- 1 register for B fragment (2 FP16 values)
- 4 registers for C/D accumulator (4 FP32 values)

**Why**: Even with same theoretical throughput, `mma.sync` enables:
- `ldmatrix` (efficient shared→register transfer)
- Smaller fragments → more flexible warp scheduling
- Exact register layout control

**Files to modify**:
- `trueno-gpu/src/ptx/instructions/mod.rs` — add `MmaSync` variant to `PtxOp`
- `trueno-gpu/src/ptx/builder/tensor_core.rs` — add `mma_sync_m16n8k16()` method
- `trueno-gpu/src/ptx/builder/emit/wmma.rs` — add `mma.sync` emission

### Phase 2: 128×128 CTA tile kernel (Est: +40%)

**What**: New kernel `build_cta128_mma_fp16()`:
- 128×128 output tile, 256 threads (8 warps)
- Each warp computes 64×8 output (8 `mma.sync.m16n8k16` per K-step)
- K-tile = 32 (two m16n8k16 per stage)
- 4-stage cp.async pipeline

**CUTLASS SM80 FP16 default (from default_gemm_configuration.h)**:
- ThreadblockShape: `GemmShape<128, 256, 64>` — 128×256 CTA with K=64
- WarpShape: `GemmShape<64, 64, 64>` — each warp owns 64×64 output
- InstructionShape: `GemmShape<16, 8, 16>` — `mma.sync.m16n8k16`
- Stages: 3

**Our conservative 128×128 target (Phase 2)**:
- CTA: 128×128, K-tile=32 (two m16n16k16 per stage)
- Warp: 32×32 (4 warps per CTA, vs CUTLASS 4 warps for 64×64)
- Shared memory per stage: A=128×32×2=8KB + B=32×128×2=8KB = 16KB
- 3 stages: 48 KB total (fits in 48 KB static — no cuFuncSetAttribute needed)

**Files to modify**:
- New file: `trueno-gpu/src/kernels/gemm/basic/tensor_core/cta128_mma.rs`
- `trueno-gpu/src/driver/module.rs` — add `cuFuncSetAttribute` for dynamic smem
- `trueno-gpu/src/driver/sys/mod.rs` — add `cuFuncSetAttribute` FFI

### Phase 3: ldmatrix + smem swizzle (Est: +15%)

**What**: Add `ldmatrix.sync.aligned.m8n8.x4` instruction:
- Loads 4 8×8 matrix fragments from shared memory in one instruction
- Each thread loads one row of 8 FP16 values
- 4× fewer shared memory load instructions

**Swizzle pattern**: XOR-based permutation of smem addresses to avoid
bank conflicts when 32 threads load from adjacent rows.

### Phase 4: cuFuncSetAttribute for >48KB smem

**What**: Call `cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes)`
to enable kernels with >48 KB shared memory.

RTX 4090 SM: 100 KB shared memory per SM. With opt-in:
- 4-stage 128×128: 64 KB → fits with CTA occupancy of 1
- 3-stage: 48 KB → static only, CTA occupancy of 2

### Status and Measured Results (2026-04-06)

| Phase | Technique | Status | Result |
|-------|-----------|--------|--------|
| Phase 1 | mma.sync m16n8k16 PTX support | ✅ DONE | Builder methods + emission |
| Phase 2 | 128×128 CTA kernel | ⚠️ **NEGATIVE** | 28.4 vs 40.5 TFLOP/s (occupancy loss) |
| Phase 2b | 3-stage pipeline (64×64) | ⚠️ **NEUTRAL** | +3% at 512, -3% at 1024 |
| Phase 3 | ldmatrix + swizzle | NOT STARTED | PTX ops added, kernel TBD |
| Phase 4 | cuFuncSetAttribute | NOT STARTED | — |

**128×128 negative result root cause**: 24KB smem/CTA → fewer concurrent CTAs.
The 2× compute-to-load ratio improvement is offset by occupancy loss.
CUTLASS compensates with mma.sync + ldmatrix (8× fewer smem load instructions)
+ more pipeline stages.

**Instruction analysis (2026-04-06)**: PTX instruction count reveals the kernel
is 96% address-computation overhead (152/158 instructions are non-compute).
Incremental optimization (stride-based addressing) REGRESSED because ptxas
instruction reordering is affected by register dependencies. Only a full
rewrite using mma.sync+ldmatrix (which fundamentally changes the fragment
load pattern) can improve the ratio.

**Revised estimate**: 0.5x cuBLAS achievable with mma.sync+ldmatrix at 64×64.
0.6x+ requires solving the occupancy/tile-size tradeoff (needs warp specialization).

### Production Path (Available Now)

The cuBLAS optional backend (`--features cuda`) provides **105-150 TFLOP/s**
production throughput via `Matrix::matmul`. Pure-Rust PTX development continues
as a Track 2 research effort toward vendor independence.

## What We Cannot Match (cuBLAS advantages)

1. **SASS-level scheduling**: cuBLAS emits SASS directly, controlling
   dual-issue and instruction pairing. PTX→SASS compilation by `ptxas`
   cannot match hand-tuned SASS.

2. **Persistent kernels with L2 residency**: cuBLAS uses `cudaStreamGetAttribute`
   to pin data in L2. Not available via PTX.

3. **Proprietary heuristics**: cuBLAS auto-tunes tile configurations per
   GPU SKU. 150+ kernel variants per GEMM shape.

## Implementation Priority

1. Phase 2 (128×128 tile) gives the biggest single improvement (+40%)
   and can use existing `wmma.m16n16k16` initially — no new PTX needed.
2. Phase 1 (mma.sync) can be done in parallel as PTX builder extension.
3. Phase 4 (cuFuncSetAttribute) is a small driver addition.
4. Phase 3 (ldmatrix + swizzle) requires the most PTX builder work.

**Recommended start**: Phase 2 with wmma (128×128, 4-stage, 48KB static smem).
This tests the tile size hypothesis without requiring new PTX instructions.
If successful, add mma.sync (Phase 1) and ldmatrix (Phase 3) incrementally.
