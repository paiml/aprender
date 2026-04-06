# Design by Contract in Trueno-GPU

## Overview

Trueno-GPU generates PTX assembly for NVIDIA GPUs at runtime. Contracts ensure
generated kernels target valid compute capabilities and respect hardware limits.

## PTX Module Contracts

### Valid SM Targets

PTX modules specify a compute capability via `PtxModule::target("sm_XX")`. The
`validate_target()` function enforces that only SM 70+ targets are accepted:

- `sm_70` -- Volta
- `sm_75` -- Turing
- `sm_80`, `sm_86`, `sm_89` -- Ampere/Ada
- `sm_90` -- Hopper

Invalid SM targets (e.g., `sm_50`, `sm_61`) are rejected with
`GpuError::InvalidTarget`. Pre-Volta targets are unsupported because the PTX
builder uses instructions that require SM 7.0+.

### PTX Version Validation

`validate_version(major, minor)` rejects any PTX version below 7.0. The default
version is 8.0 (set by `PtxModule::new()`). This contract prevents generating
PTX that references unavailable features.

### Address Size

`PtxModule::address_size()` accepts 32 or 64. The default is 64-bit addressing,
which is required for modern GPUs with >4 GB memory.

### Shared Memory Declaration

`PtxKernel::shared_memory(bytes)` declares a shared memory allocation for the
kernel. The emitter generates `.shared .align 16 .b8 smem[N]` in the PTX output.
The actual hardware limit depends on the SM target (48 KB for Volta, up to
228 KB for Hopper with opt-in).

### Kernel Parameters

Each `PtxKernel::param(ty, name)` adds a typed parameter to the kernel entry
point. The `PtxType` enum ensures only valid PTX types are used (U8 through F64,
plus vector types V2F32 and V4F32). Register allocation respects type-specific
prefixes to avoid CUDA_ERROR_INVALID_PTX (e.g., U32 uses `%r`, S32 uses `%ri`).

## Module-Level Validation

`PtxModule::validate()` combines version and target validation into a single
check. Call this before `emit()` to fail fast on invalid configurations.

## Relationship to Trueno Contracts

Trueno-GPU inherits the row-major layout contract from the parent trueno crate.
All GPU kernels read weight data in row-major order. The column-major GGUF data
is transposed before reaching GPU memory.

## Source of Truth

- **Parent contracts**: `~/src/trueno/src/contracts.rs`
- **PTX validation**: `trueno-gpu/src/ptx/mod.rs` (`validate_target`, `validate_version`)
- **PTX builder**: `trueno-gpu/src/ptx/builder/ptx_module.rs` (`PtxModule`, `PtxKernel`)
- **Contract YAML**: `~/src/aprender/contracts/tensor-layout-v1.yaml`
- **Full DbC spec**: `~/src/aprender/docs/specifications/enforce-provable-DbC.md`
