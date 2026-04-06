# CUDA Context Safety

> **Status**: Implemented (February 2026)
>
> See: `src/cuda/executor/mod.rs`, `src/cuda/executor/core.rs`

## The Problem: Context Poisoning

CUDA GPU contexts are fragile. A single kernel crash (null pointer dereference, shared memory overflow, illegal instruction) returns `CUDA_ERROR_UNKNOWN` (code 700) and **permanently poisons the GPU device for the entire process lifetime**. No API call can recover a poisoned context -- only a full process restart.

In a test suite running 1,196 CUDA tests in one process, a single poisoned context cascades into hundreds of failures.

## Root Causes

Three classes of operations poison CUDA contexts:

### 1. Null Device Pointer Dereference

```rust
// A GpuBuffer with ptr=0 was passed to a kernel
// The kernel reads from address 0x0 → GPU crash → context poisoned
cuLaunchKernel(func, ..., ptr=0x0, ...);
// → CUDA_ERROR_UNKNOWN (700) - permanent
```

### 2. Shared Memory Overflow

```rust
// Kernel accesses shared_mem[head_dim * head_dim] but only
// shared_mem[tile_q * head_dim] was allocated (tile_q < head_dim)
// → Out-of-bounds shared memory access → GPU crash
```

### 3. PTX Compilation with Poisoned Context

```rust
// After context is poisoned, even cuModuleLoadData fails
// → CUDA_ERROR_UNKNOWN propagates to all subsequent operations
```

## Defense Architecture

Realizar uses an Erlang-style **fail-fast** defense with four layers:

### Layer 1: Null Pointer Validation

Every kernel-launching method validates all device pointers before launch:

```rust
fn validate_device_ptr(ptr: u64, name: &str) -> Result<(), GpuError> {
    if ptr == 0 {
        return Err(GpuError::InvalidParameter(format!(
            "{name}: null device pointer (0x0) — refusing to launch \
             kernel to prevent unrecoverable GPU device poisoning"
        )));
    }
    Ok(())
}

// Applied to all 20 kernel methods in q4k.rs, q_basic.rs, layer.rs
```

### Layer 2: Shared Memory Bounds Checking

Flash attention validates that shared memory allocation covers the worst-case access pattern:

```rust
let tile_q = seq_len.min(TILE_SIZE);
let required = tile_q * head_dim;
let accessed = head_dim * head_dim;

if accessed > required {
    return Err(GpuError::InvalidParameter(...));
}
```

### Layer 3: Sync-on-Drop (Fail-Fast)

`CudaExecutor` synchronizes its context on drop. If synchronization fails (indicating a poisoned context), the context and stream are **not** returned to the pool:

```rust
impl Drop for CudaExecutor {
    fn drop(&mut self) {
        let healthy = self.context.synchronize().is_ok();
        if healthy {
            // Return to pool for reuse
            CONTEXT_POOL.lock().unwrap().replace(self.context.take());
            STREAM_POOL.lock().unwrap().replace(self.stream.take());
        }
        // If unhealthy: resources are dropped, never reused
    }
}
```

### Layer 4: CPU Fallback Paths

When GPU constraints aren't met, automatic CPU fallback avoids the risk entirely:

```rust
if seq_len >= head_dim {
    self.flash_attention_multi_head(...)?;  // GPU path
} else {
    // CPU scaled dot-product attention
    // Numerically equivalent, avoids shared memory overflow
}
```

## Context Pool Architecture

```
┌─────────────────────────────────────────────────────┐
│                   CUDA_SENTINEL                      │
│  Global health sentinel — checked on executor init   │
├─────────────────────────────────────────────────────┤
│                   CONTEXT_POOL                       │
│  Mutex<Option<CudaContext>>                          │
│  Only receives healthy contexts (sync-on-drop gate)  │
├─────────────────────────────────────────────────────┤
│                   STREAM_POOL                        │
│  Mutex<Option<CudaStream>>                           │
│  Paired with context lifecycle                       │
└─────────────────────────────────────────────────────┘
```

## Test Results

Before context safety: 578 passed, 411 failed (context poisoning cascade)
After context safety: **1,196 passed, 0 failed**

The 411 failures were caused by:
1. `test_batched_q4k_gemv_into_m12_tiled` launching a kernel with `weight_ptr=0`
2. `test_cov004_flash_attention_cached_*` hitting shared memory overflow with `head_dim=8, seq_len=1`

## See Also

- [Memory Management](./memory-management.md) - GPU buffer pooling
- [Flash Attention](../phases/phase2-flash-attention.md) - Shared memory validation details
