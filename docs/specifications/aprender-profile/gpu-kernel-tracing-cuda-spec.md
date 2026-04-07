# GPU Kernel-Level Tracing: Phase 2 - CUDA Support

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Specification - Design Phase
**Sprint Target:** 38 (CUDA Kernel Tracing via CUPTI)
**GitHub Issue:** #16 (Phase 2)
**Depends On:** Sprint 37 (Phase 1: wgpu support)

## Executive Summary

This specification defines **CUDA kernel-level observability** via **CUPTI (CUDA Profiling Tools Interface)** integrated with **Renacer's** OTLP export infrastructure. Following **Phase 1's** wgpu implementation and **Toyota Way** principles, this spec extends GPU observability from WebGPU applications to NVIDIA CUDA workloads.

**Business Value:**
- **CUDA Bottleneck Identification**: Identify slow CUDA kernels in production
- **Multi-GPU Backend Comparison**: Compare wgpu vs CUDA performance
- **Production ML/HPC Observability**: Trace PyTorch, TensorFlow, JAX CUDA kernels
- **Unified Observability**: Single OTLP backend for wgpu + CUDA + SIMD + syscalls

**Key Principle (Toyota Way):**
> *"Reuse proven patterns, add complexity only where necessary."* - We reuse Phase 1's GpuKernel struct and adaptive sampling, adding CUPTI-specific integration.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 2: CUPTI Activity API Integration](#3-phase-2-cupti-activity-api-integration)
4. [Implementation Plan](#4-implementation-plan)
5. [Dependencies and Requirements](#5-dependencies-and-requirements)
6. [Testing Strategy](#6-testing-strategy)
7. [Comparison: wgpu vs CUDA](#7-comparison-wgpu-vs-cuda)

---

## 1. Background and Motivation

### 1.1 Phase 1 Accomplishments (Sprint 37)

**✅ wgpu Support Complete:**
- GpuKernel struct and record_gpu_kernel() method
- GpuProfilerWrapper integrating wgpu-profiler with OTLP
- Adaptive sampling (100μs threshold)
- 9 integration tests, all passing

**❌ CUDA Not Supported:**
- CUDA kernel executions invisible to Renacer
- ML/HPC workloads (PyTorch, TensorFlow, JAX) untraced
- No multi-backend comparison (wgpu vs CUDA)

### 1.2 Use Case: ML Training Pipeline

**Example application:** PyTorch model training with GPU acceleration

**Current visibility:**
```
syscall:ioctl (CUDA driver call)     - 5μs    ✅ Traced (indirect)
cuda:matrix_multiply_kernel          - 150ms  ❌ NOT traced
cuda:softmax_kernel                  - 45ms   ❌ NOT traced
cuda:backward_pass_kernel            - 200ms  ❌ NOT traced
```

**Desired visibility (Phase 2):**
```
Root Span: "process: python train.py"
├─ Span: "syscall: ioctl" (CUDA driver call) - 5μs                  ✅
├─ Span: "gpu_kernel: sum_aggregation" (wgpu) - 60ms                ✅ Phase 1
└─ Span: "cuda_kernel: matrix_multiply" (CUDA) - 150ms              🎯 NEW Phase 2
    ├─ Attributes:
    │   - gpu.backend: "cuda"
    │   - gpu.kernel: "matrix_multiply"
    │   - gpu.duration_us: 150000
    │   - gpu.cuda.sm_count: 84  (streaming multiprocessors)
    │   - gpu.cuda.occupancy: 0.75
    │   - gpu.is_slow: true
    └─ Status: OK
```

### 1.3 Why CUPTI?

**CUPTI (CUDA Profiling Tools Interface):**
- **Official NVIDIA API** for CUDA profiling
- **Activity API**: Asynchronous kernel timestamp collection
- **Callback API**: Synchronous event notification
- **Production-Ready**: Used by Nsight Systems, NVIDIA Profiler

**Alternatives Considered:**
- ❌ **NVTX**: User-instrumentation only (requires code changes)
- ❌ **cudaEventRecord()**: Requires manual instrumentation
- ✅ **CUPTI Activity API**: Automatic, zero-code-change instrumentation

**Rust Integration:**
- ✅ **cudarc**: Modern Rust CUDA wrapper with CUPTI support
- ✅ **FFI Bindings**: C API can be wrapped with rust-bindgen
- ✅ **Production Use**: Polar Signals uses CUPTI+eBPF in production (2025)

---

## 2. Architecture Overview

### 2.1 Integration Layers

```
┌─────────────────────────────────────────────────────────────┐
│  Observability Backend (Jaeger, Tempo, etc.)                │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ OTLP Protocol
                          │
┌─────────────────────────────────────────────────────────────┐
│  Renacer OTLP Exporter (src/otlp_exporter.rs)               │
│  - Export syscall spans                  ✅ Sprint 30       │
│  - Export SIMD compute blocks            ✅ Sprint 32       │
│  - Export wgpu GPU kernels               ✅ Sprint 37       │
│  - NEW: Export CUDA GPU kernels          🎯 Sprint 38       │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ record_gpu_kernel(GpuKernel)
                          │ (reuse Phase 1 API!)
                          │
┌─────────────────────────────────────────────────────────────┐
│  CUDA Kernel Tracer (NEW: src/cuda_tracer.rs)               │
│  - Wrapper around CUPTI Activity API                        │
│  - Convert CUPTI activities → GpuKernel metadata            │
│  - Adaptive sampling (duration > 100μs)                     │
│  - Export as OTLP spans                                     │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ CUPTI Activity API
                          │
┌─────────────────────────────────────────────────────────────┐
│  CUPTI (CUDA Profiling Tools Interface)                     │
│  - Asynchronous activity recording                          │
│  - Kernel timestamps (HW-based on Blackwell+)               │
│  - Memory transfer tracking                                 │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ Automatic instrumentation
                          │
┌─────────────────────────────────────────────────────────────┐
│  User's CUDA Application (e.g., PyTorch, TensorFlow)        │
│  - CUDA kernels launched via cudaLaunchKernel               │
│  - Zero code changes required                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Span Hierarchy (Unified: wgpu + CUDA + SIMD)

**Complete Trace (syscalls + SIMD + wgpu + CUDA):**
```
Root Span: "process: python train.py"
├─ Span: "syscall: openat" (duration: 15μs)
├─ Span: "compute_block: calculate_statistics" (SIMD) (duration: 227μs)  ← Sprint 32
├─ Span: "gpu_kernel: sum_aggregation" (wgpu) (duration: 60ms)          ← Sprint 37
└─ Span: "cuda_kernel: matrix_multiply_fp16" (CUDA) (duration: 150ms)   ← NEW Sprint 38
    ├─ gpu.backend: "cuda"
    ├─ gpu.kernel: "matrix_multiply_fp16"
    ├─ gpu.duration_us: 150000
    ├─ gpu.cuda.sm_count: 84
    ├─ gpu.cuda.occupancy: 0.75
    ├─ gpu.cuda.grid_dim: "[128,128,1]"
    ├─ gpu.cuda.block_dim: "[16,16,1]"
    ├─ gpu.is_slow: true
    └─ Status: OK
```

**Benefits:**
- ✅ **Cross-backend comparison**: Compare wgpu (60ms) vs CUDA (150ms) for same operation
- ✅ **Bottleneck identification**: CUDA kernel is 2.5x slower than wgpu
- ✅ **Unified timeline**: All GPU backends + SIMD + syscalls in one trace

### 2.3 Span Attributes (Extended for CUDA)

**Resource-Level Attributes (once at startup):**
```json
{
  "resource": {
    "service.name": "renacer",
    "compute.library": "trueno",           // Sprint 32
    "gpu.library.wgpu": "23.0.0",          // Sprint 37
    "gpu.library.cuda": "12.6",            // NEW Sprint 38
    "gpu.tracing.backends": "wgpu,cuda",
    "process.pid": 12345
  }
}
```

**Span-Level Attributes (per CUDA kernel):**
```json
{
  "span.name": "cuda_kernel: matrix_multiply_fp16",
  "span.kind": "INTERNAL",
  "attributes": {
    "gpu.backend": "cuda",
    "gpu.kernel": "matrix_multiply_fp16",
    "gpu.duration_us": 150000,
    "gpu.cuda.device_id": 0,
    "gpu.cuda.context_id": 123456,
    "gpu.cuda.stream_id": 7,
    "gpu.cuda.grid_dim": "[128,128,1]",
    "gpu.cuda.block_dim": "[16,16,1]",
    "gpu.cuda.sm_count": 84,
    "gpu.cuda.occupancy": 0.75,
    "gpu.cuda.shared_mem_bytes": 49152,
    "gpu.cuda.registers_per_thread": 32,
    "gpu.is_slow": true,
    "gpu.threshold_us": 100
  },
  "status": "OK"
}
```

---

## 3. Phase 2: CUPTI Activity API Integration

### 3.1 CUPTI Activity API Overview

**Key Concepts:**
- **Activity**: A recorded event (kernel launch, memory copy, etc.)
- **Activity Buffer**: Circular buffer for asynchronous activity records
- **Activity Kind**: Type of activity (KERNEL, MEMCPY, etc.)

**Workflow:**
1. Initialize CUPTI and register activity buffer callback
2. Enable activity kinds (CUPTI_ACTIVITY_KIND_KERNEL)
3. CUDA application runs (kernels launch automatically)
4. CUPTI writes activity records to buffer asynchronously
5. Buffer callback fires → read activity records
6. Convert activities to GpuKernel → export to OTLP

### 3.2 Reuse Phase 1 GpuKernel Struct

**No changes needed!** Phase 1's `GpuKernel` struct already supports CUDA:

```rust
/// GPU kernel metadata for tracing (Sprint 37 + 38)
///
/// Supports multiple backends: wgpu (Phase 1), CUDA (Phase 2)
#[derive(Debug, Clone)]
pub struct GpuKernel {
    pub kernel: String,           // Works for both wgpu and CUDA
    pub duration_us: u64,         // Works for both
    pub backend: &'static str,    // "wgpu" or "cuda"
    pub workgroup_size: Option<String>, // wgpu: workgroup, CUDA: block_dim
    pub elements: Option<usize>,  // Optional for both
    pub is_slow: bool,            // Adaptive sampling
}
```

**Extension for CUDA-specific attributes:**
- Add optional fields to `GpuKernel` for CUDA metadata
- OR: Use span attributes in `record_gpu_kernel()` for CUDA-specific data

**Design Decision:** Use span attributes (keeps `GpuKernel` simple).

### 3.3 CUDA Tracer Wrapper

**File:** `src/cuda_tracer.rs` (NEW)

**Purpose:**
- Initialize CUPTI Activity API
- Register activity buffer callback
- Convert CUPTI activity records → `GpuKernel` structs
- Apply adaptive sampling (duration > 100μs)
- Export to OTLP via `OtlpExporter::record_gpu_kernel()`

**High-Level Implementation:**

```rust
//! CUDA kernel tracing wrapper for CUPTI (Sprint 38)
//!
//! Integrates CUPTI Activity API with Renacer's OTLP export infrastructure.
//! Follows Sprint 37's wgpu pattern: adaptive sampling, kernel-level tracing.

use anyhow::Result;
use crate::otlp_exporter::{GpuKernel, OtlpExporter};

/// Configuration for CUDA kernel tracing
#[derive(Debug, Clone)]
pub struct CudaTracerConfig {
    /// Minimum duration to trace (default: 100μs, same as wgpu/SIMD)
    pub threshold_us: u64,
    /// Trace all kernels regardless of duration (debug mode)
    pub trace_all: bool,
    /// CUPTI activity buffer size (default: 8MB)
    pub buffer_size: usize,
}

impl Default for CudaTracerConfig {
    fn default() -> Self {
        CudaTracerConfig {
            threshold_us: 100,
            trace_all: false,
            buffer_size: 8 * 1024 * 1024, // 8MB
        }
    }
}

/// Wrapper around CUPTI Activity API that exports to OTLP
#[cfg(feature = "cuda-tracing")]
pub struct CudaTracerWrapper {
    otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
    config: CudaTracerConfig,
    cupti_initialized: bool,
}

#[cfg(feature = "cuda-tracing")]
impl CudaTracerWrapper {
    /// Initialize CUDA tracer with CUPTI Activity API
    ///
    /// # Returns
    ///
    /// Returns `Ok(CudaTracerWrapper)` on success, or error if:
    /// - CUDA runtime not available
    /// - CUPTI library not found
    /// - Activity API initialization fails
    pub fn new(
        otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
        config: CudaTracerConfig,
    ) -> Result<Self> {
        // TODO: Initialize CUPTI Activity API
        // - cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL)
        // - cuptiActivityRegisterCallbacks(buffer_requested, buffer_completed)
        // - Allocate activity buffers

        todo!("Implement CUPTI initialization")
    }

    /// Process CUPTI activity buffer and export to OTLP
    ///
    /// Called from CUPTI buffer completion callback.
    ///
    /// # Adaptive Sampling
    ///
    /// Only kernels with `duration >= threshold_us` (default 100μs) are exported,
    /// unless `config.trace_all = true` (debug mode).
    pub fn process_activity_buffer(&mut self, buffer: &[u8]) {
        // TODO: Parse CUPTI activity records
        // - cuptiActivityGetNextRecord() in loop
        // - For each CUPTI_ACTIVITY_KIND_KERNEL:
        //   - Extract kernel name, duration, grid/block dims
        //   - Convert to GpuKernel struct
        //   - Apply adaptive sampling
        //   - Export via otlp_exporter.record_gpu_kernel()

        todo!("Implement activity buffer processing")
    }

    /// Flush pending CUPTI activities and export to OTLP
    pub fn flush(&mut self) {
        // TODO: cuptiActivityFlushAll() to force buffer completion
        todo!("Implement flush")
    }
}

impl Drop for CudaTracerWrapper {
    fn drop(&mut self) {
        // TODO: Clean up CUPTI resources
        // - cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL)
        // - cuptiFinalize()
    }
}

// Stub implementation when CUDA tracing feature is disabled
#[cfg(not(feature = "cuda-tracing"))]
pub struct CudaTracerWrapper;

#[cfg(not(feature = "cuda-tracing"))]
impl CudaTracerWrapper {
    pub fn new(
        _otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
        _config: CudaTracerConfig,
    ) -> Result<Self> {
        anyhow::bail!("CUDA tracing support not compiled in. Enable the 'cuda-tracing' feature.");
    }
}
```

### 3.4 CUPTI FFI Bindings

**Approach 1: Use cudarc crate**
- ✅ Modern, maintained Rust CUDA wrapper
- ✅ Supports CUDA 11.4-13.0
- ✅ Includes CUPTI support
- ❌ May not expose all CUPTI Activity API functions

**Approach 2: Custom FFI with rust-bindgen**
- ✅ Full control over CUPTI API exposure
- ✅ Can target specific CUDA versions
- ❌ More maintenance overhead
- ❌ Need to handle CUDA version differences

**Recommended:** Start with cudarc, add custom bindings if needed.

**Example FFI Bindings:**

```rust
// File: src/cupti_bindings.rs (if custom FFI needed)

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::os::raw::{c_void, c_char, c_uint};

// CUPTI Activity Kinds
pub const CUPTI_ACTIVITY_KIND_KERNEL: c_uint = 2;
pub const CUPTI_ACTIVITY_KIND_MEMCPY: c_uint = 3;

// CUPTI Activity Record for Kernel
#[repr(C)]
pub struct CUpti_ActivityKernel4 {
    pub kind: c_uint,
    pub start: u64,
    pub end: u64,
    pub device_id: u32,
    pub context_id: u32,
    pub stream_id: u32,
    pub correlation_id: u32,
    pub grid_x: i32,
    pub grid_y: i32,
    pub grid_z: i32,
    pub block_x: i32,
    pub block_y: i32,
    pub block_z: i32,
    pub static_shared_memory: i32,
    pub dynamic_shared_memory: i32,
    pub local_memory_per_thread: u32,
    pub registers_per_thread: i32,
    pub name: *const c_char,
}

extern "C" {
    pub fn cuptiActivityEnable(kind: c_uint) -> c_uint;
    pub fn cuptiActivityDisable(kind: c_uint) -> c_uint;
    pub fn cuptiActivityFlushAll(flag: c_uint) -> c_uint;
    pub fn cuptiActivityGetNextRecord(
        buffer: *mut c_void,
        valid_buffer_size_bytes: usize,
        record: *mut *mut c_void,
    ) -> c_uint;
}
```

---

## 4. Implementation Plan

### 4.1 Sprint 38 Checklist (Phase 2: CUDA)

**RESEARCH Phase:**
- [x] Research CUPTI Activity API
- [x] Research cudarc Rust bindings
- [x] Assess FFI binding requirements
- [ ] Prototype CUPTI initialization
- [ ] Test CUPTI on NVIDIA GPU hardware

**RED Phase (Tests First):**

**File:** `tests/sprint38_cuda_kernel_tracing_tests.rs`

```rust
#[test]
#[cfg(all(feature = "cuda-tracing", feature = "otlp"))]
fn test_cuda_kernel_traced_when_slow() {
    // Test that slow CUDA kernels (>100μs) are traced
    // NOTE: Requires NVIDIA GPU hardware
}

#[test]
#[cfg(all(feature = "cuda-tracing", feature = "otlp"))]
fn test_cuda_kernel_attributes() {
    // Test CUDA-specific span attributes (grid_dim, block_dim, sm_count)
}

#[test]
#[cfg(all(feature = "cuda-tracing", feature = "otlp"))]
fn test_cuda_and_wgpu_unified_trace() {
    // Test that CUDA and wgpu kernels appear in same trace
}
```

**GREEN Phase (Implementation):**
1. Add `cudarc` or custom CUPTI bindings to `Cargo.toml` (~10 lines)
2. Create `src/cuda_tracer.rs` with `CudaTracerWrapper` (~300 lines)
3. Implement CUPTI Activity API initialization (~100 lines)
4. Implement activity buffer callback (~150 lines)
5. Convert CUPTI activities → GpuKernel (~50 lines)
6. Add module exports in `src/lib.rs` (~2 lines)
7. Implement 6+ integration tests (~300 lines)

**REFACTOR Phase:**
1. Add unit tests for edge cases
2. Verify complexity ≤10 for all functions
3. Benchmark overhead (target: <2% like wgpu/SIMD)

**Total Code:** ~900 lines (more complex than wgpu due to FFI)

### 4.2 Cargo.toml Changes

**Add to `[dependencies]`:**
```toml
# CUDA kernel tracing (Sprint 38)
cudarc = { version = "0.11", optional = true, features = ["cupti"] }
# OR custom CUPTI bindings:
# cupti-sys = { version = "0.1", optional = true }
```

**Add to `[features]`:**
```toml
# CUDA kernel-level tracing (Sprint 38)
cuda-tracing = ["dep:cudarc", "otlp"]
```

### 4.3 CLI Flags (Extend Phase 1 pattern)

**Reuse Phase 1 flags:**
```bash
--trace-gpu              # Enable GPU kernel tracing (wgpu + CUDA if available)
--trace-gpu-all          # Debug mode: trace ALL kernels
--trace-gpu-threshold N  # Custom threshold (default: 100μs)
```

**New CUDA-specific flags:**
```bash
--trace-cuda-only        # Trace only CUDA kernels (disable wgpu)
--cuda-buffer-size N     # CUPTI activity buffer size (default: 8MB)
```

---

## 5. Dependencies and Requirements

### 5.1 System Requirements

**MANDATORY:**
- ✅ **NVIDIA GPU**: Compute Capability 3.5+ (Kepler or newer)
- ✅ **CUDA Toolkit**: Version 11.4+ (CUPTI included)
- ✅ **GPU Driver**: Compatible with CUDA toolkit version
- ✅ **libcuda.so / cuda.dll**: CUDA runtime library

**OPTIONAL:**
- ✅ **Multiple GPUs**: For multi-device tracing
- ✅ **Nsight Systems**: For validation (compares CUPTI vs Nsight)

### 5.2 Rust Dependencies

**Approach 1: cudarc (Recommended)**
```toml
cudarc = { version = "0.11", features = ["cupti"] }
```

**Approach 2: Custom Bindings**
```toml
# Create cupti-sys crate with rust-bindgen
bindgen = "0.69"  # build dependency
```

### 5.3 Testing Requirements

**Challenge:** Testing requires NVIDIA GPU hardware

**Solutions:**
1. **CI/CD with GPU Runners**: GitHub Actions with NVIDIA GPUs (expensive)
2. **Conditional Tests**: `#[ignore]` tests unless `CUDA_VISIBLE_DEVICES` set
3. **Mock Tests**: Mock CUPTI API for unit tests
4. **Manual Testing**: Document manual test procedure for developers

**Recommended Strategy:**
- Unit tests with mocked CUPTI API (always run)
- Integration tests marked `#[ignore]` by default
- GitHub Actions with NVIDIA GPU runner (optional)

### 5.4 Deployment Considerations

**Graceful Degradation:**
- If CUDA not available → disable cuda-tracing feature at compile time
- If CUPTI fails to initialize → log warning, continue without CUDA tracing
- wgpu + SIMD tracing still work without CUDA

---

## 6. Testing Strategy

### 6.1 Unit Tests (Mocked CUPTI)

**File:** `src/cuda_tracer.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_tracer_config_defaults() {
        let config = CudaTracerConfig::default();
        assert_eq!(config.threshold_us, 100);
        assert!(!config.trace_all);
        assert_eq!(config.buffer_size, 8 * 1024 * 1024);
    }

    // TODO: Mock CUPTI API for unit tests
}
```

### 6.2 Integration Tests (Requires GPU)

**File:** `tests/sprint38_cuda_kernel_tracing_tests.rs`

```rust
#[test]
#[ignore] // Run manually with: cargo test --features cuda-tracing -- --ignored
#[cfg(all(feature = "cuda-tracing", feature = "otlp"))]
fn test_cuda_matrix_multiply_traced() {
    // Launch CUDA matrix multiply kernel
    // Verify span appears in OTLP with correct attributes
}
```

### 6.3 Manual Testing Procedure

**Prerequisites:**
```bash
# Install CUDA Toolkit
sudo apt install nvidia-cuda-toolkit  # Ubuntu
# OR download from: https://developer.nvidia.com/cuda-downloads

# Verify CUDA available
nvidia-smi
nvcc --version
```

**Test Application:**
```bash
# Build with CUDA tracing
cargo build --features cuda-tracing

# Run simple CUDA test
cargo run --features cuda-tracing --example cuda_matrix_multiply

# Verify spans in Jaeger:
# - Open http://localhost:16686
# - Search for service: "renacer-cuda-test"
# - Verify spans: cuda_kernel:matrix_multiply
```

---

## 7. Comparison: wgpu vs CUDA

| Aspect | wgpu (Phase 1) | CUDA (Phase 2) |
|--------|----------------|----------------|
| **API** | wgpu-profiler (Rust) | CUPTI (C FFI) |
| **Complexity** | Low (managed by wgpu-profiler) | High (manual CUPTI setup) |
| **Hardware** | Any GPU (Vulkan/Metal/DX12) | NVIDIA GPUs only |
| **Dependencies** | wgpu 23.0 + wgpu-profiler 0.18 | CUDA Toolkit + cudarc |
| **Testing** | Easy (wgpu works on CPU backend) | Hard (requires NVIDIA GPU) |
| **Code Lines** | ~250 lines | ~900 lines |
| **Timestamp Source** | GPU timestamp queries | CUPTI Activity API |
| **Overhead** | <1% (wgpu-profiler) | <2% (CUPTI Activity) |
| **Production Use** | ✅ Ready | ✅ Production-proven (Polar Signals) |
| **Adaptive Sampling** | ✅ 100μs threshold | ✅ 100μs threshold (same) |
| **OTLP Export** | ✅ Unified | ✅ Unified (same GpuKernel) |

**Key Insight:** CUDA is more complex but follows same patterns as wgpu.

---

## 8. Success Criteria

### 8.1 Technical Requirements

- ✅ CUDA kernel spans appear in Jaeger/Tempo alongside wgpu/SIMD/syscalls
- ✅ <2% performance overhead with adaptive sampling (100μs threshold)
- ✅ Graceful degradation if CUDA not available
- ✅ 6+ integration tests (manual or CI with GPU runner)
- ✅ Works with CUDA 11.4+ (including 12.x and 13.0)

### 8.2 Business Value

**User can answer:**
- ✅ "Why did my CUDA kernel take 150ms instead of 60ms?"
- ✅ "Is wgpu or CUDA faster for this workload?"
- ✅ "Which CUDA kernel is the bottleneck in my ML pipeline?"

---

## 9. Implementation Status

### 9.1 Current Status: **SPECIFICATION COMPLETE**

- ✅ Research complete (CUPTI, cudarc, FFI)
- ✅ Architecture designed (reuses Phase 1 patterns)
- ✅ Specification complete (this document)
- ⏳ **Implementation BLOCKED**: Requires NVIDIA GPU hardware

### 9.2 Next Steps

**Option 1: Full Implementation (Requires GPU)**
1. Acquire NVIDIA GPU hardware (or cloud instance)
2. Install CUDA Toolkit 12.6+
3. Implement `src/cuda_tracer.rs` with CUPTI FFI
4. Write integration tests
5. Validate with manual testing

**Option 2: Framework Implementation (No GPU Required)**
1. Implement `CudaTracerWrapper` structure with `todo!()` placeholders
2. Write unit tests with mocked CUPTI
3. Document manual test procedure
4. Community contribution: GPU owners can complete implementation

**Option 3: Skip to Phase 3 (ROCm) or Phase 4 (Memory Transfers)**
- Phase 3 (ROCm): Similar complexity to CUDA, requires AMD GPU
- **Phase 4 (Memory Transfers)**: Can implement for wgpu TODAY (no new hardware)

### 9.3 Recommended Path Forward

**Recommendation: Implement Phase 4 (Memory Transfers) next**

**Rationale:**
- ✅ Builds on Phase 1 (wgpu) - no new hardware needed
- ✅ High business value (identifies transfer bottlenecks)
- ✅ Easier to test (same wgpu test infrastructure)
- ✅ Completes "observability trifecta": kernel time + transfer time + SIMD time

**Then:** Return to Phase 2 (CUDA) when GPU hardware available.

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | Claude Code | Initial specification (Issue #16 Phase 2) |

**Status:** ✅ Specification Complete - Implementation Blocked (No GPU)
**Approval Required:** Product Owner (Noah Gift)
**Dependencies:** NVIDIA GPU hardware, CUDA Toolkit 11.4+
**Related Specs:**
- `gpu-kernel-tracing-spec.md` (Sprint 37 - Phase 1: wgpu support)
- `trueno-tracing-integration-spec.md` (Sprint 32 - SIMD compute tracing)
- Issue #16 (GitHub - GPU Kernel-Level Tracing - All Phases)

**Next Recommendation:** **Implement Phase 4 (Memory Transfers for wgpu) - No GPU hardware blocker**
