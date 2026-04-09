# GPU Kernel-Level Tracing Specification for Renacer

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Specification - Ready for Implementation
**Sprint Target:** 37 (GPU Kernel Tracing - Phase 1: wgpu)
**GitHub Issue:** #16

## Executive Summary

This specification defines **GPU kernel-level observability** for **wgpu-based applications** integrated with **Renacer's** OTLP export infrastructure. Following **Sprint 32's** SIMD compute tracing architecture and **Toyota Way** principles, this spec extends observability from CPU compute operations to GPU kernel executions.

**Business Value:**
- **GPU Bottleneck Identification**: Identify slow GPU kernels (compute shaders, memory transfers)
- **Production GPU Debugging**: Understand why GPU-accelerated applications are slow
- **Workload Profiling**: Compare GPU vs SIMD performance for same operations
- **Unified Observability**: Single OTLP backend for syscalls, SIMD compute, AND GPU kernels

**Key Principle (Toyota Way):**
> *"Extend proven patterns, don't reinvent."* - We reuse Sprint 32's block-level tracing architecture, adaptive sampling, and OTLP export for GPU kernels.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 1: wgpu Timestamp Query Integration](#3-phase-1-wgpu-timestamp-query-integration)
4. [Implementation Plan](#4-implementation-plan)
5. [Testing Strategy](#5-testing-strategy)
6. [Performance Impact](#6-performance-impact)
7. [Future Work](#7-future-work)

---

## 1. Background and Motivation

### 1.1 Current State

**✅ Sprint 30-32 Accomplished:**
- Syscall tracing with OTLP export (Sprint 30)
- W3C Trace Context propagation (Sprint 33)
- SIMD compute block tracing via Trueno (Sprint 32)
- Integration tests with Jaeger backend (Sprint 34)

**❌ Not Supported:**
- Direct GPU kernel execution tracing
- wgpu compute shader timing
- GPU memory transfer tracking
- GPU vs SIMD performance comparison

### 1.2 Use Case: trueno-db

**Example application:** `trueno-db` (GPU-first analytics database)
- Uses wgpu for GPU aggregations (SUM, AVG, MIN, MAX on large datasets)
- SIMD fallback via Trueno (already traced via Sprint 32 ✅)
- GPU execution path currently invisible to renacer

**Current visibility:**
```
compute_block:extended_stats (SIMD) - 227μs  ✅ Traced
syscall:ioctl (GPU driver call)     - 5μs    ✅ Traced (indirect)
gpu:sum_aggregation (wgpu kernel)   - 60ms   ❌ NOT traced
```

**Desired visibility (Sprint 37):**
```
Root Span: "process: trueno-db-server"
├─ Span: "compute_block: calculate_statistics" (SIMD fallback) - 227μs  ✅
├─ Span: "syscall: ioctl" (GPU driver call) - 5μs                       ✅
└─ Span: "gpu_kernel: sum_i32_compute_shader" (wgpu) - 60ms             🎯 NEW
    ├─ Attributes:
    │   - gpu.backend: "wgpu"
    │   - gpu.kernel: "sum_i32_compute_shader"
    │   - gpu.duration_us: 60000
    │   - gpu.workgroup_size: [256, 1, 1]
    │   - gpu.elements: 1000000
    │   - gpu.is_slow: true
    └─ Status: OK
```

### 1.3 Why wgpu-profiler?

**Ecosystem Research:**
- **wgpu-profiler**: Community-standard profiling library for wgpu
- **Features**: QuerySet management, nestable scopes, zero-copy results
- **Exports**: Chrome trace JSON, Tracy, Puffin integration
- **Status**: Mature, actively maintained

**Integration Strategy:**
- ✅ **Reuse wgpu-profiler** for timestamp query management
- ✅ **Add OTLP export** as a new backend (alongside Chrome/Tracy/Puffin)
- ✅ **Follow Sprint 32 patterns**: Adaptive sampling, block-level tracing

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
│  - NEW: Export GPU kernel spans          🎯 Sprint 37       │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ record_gpu_kernel(GpuKernel)
                          │
┌─────────────────────────────────────────────────────────────┐
│  GPU Kernel Tracer (NEW: src/gpu_tracer.rs)                 │
│  - Wrapper around wgpu-profiler::GpuProfiler                │
│  - Convert wgpu scopes → GpuKernel metadata                 │
│  - Adaptive sampling (duration > 100μs)                     │
│  - Export as OTLP spans                                     │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ Uses wgpu-profiler API
                          │
┌─────────────────────────────────────────────────────────────┐
│  wgpu-profiler (external crate)                             │
│  - Timestamp query management                               │
│  - Scope creation and nesting                               │
│  - Query result resolution                                  │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ QuerySet / TimestampWrites
                          │
┌─────────────────────────────────────────────────────────────┐
│  User's wgpu Application (e.g., trueno-db)                  │
│  - Compute shaders, render passes                           │
│  - Instruments code with profiler scopes                    │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Span Hierarchy (Unified Observability)

**Complete Trace (syscalls + SIMD + GPU):**
```
Root Span: "process: trueno-db-server"
├─ Span: "syscall: openat" (duration: 15μs)
├─ Span: "syscall: read" (duration: 42μs)
├─ Span: "compute_block: calculate_statistics" (SIMD) (duration: 227μs)  ← Sprint 32
│   ├─ compute.operation: "calculate_statistics"
│   ├─ compute.duration_us: 227
│   ├─ compute.elements: 10000
│   └─ compute.backend: "Unknown"  (Trueno SIMD)
└─ Span: "gpu_kernel: sum_aggregation" (wgpu) (duration: 60ms)  ← NEW Sprint 37
    ├─ gpu.backend: "wgpu"
    ├─ gpu.kernel: "sum_aggregation"
    ├─ gpu.duration_us: 60000
    ├─ gpu.workgroup_size: "[256,1,1]"
    ├─ gpu.elements: 1000000
    ├─ gpu.is_slow: true
    └─ Status: OK
```

**Benefits:**
- ✅ **Unified timeline**: See GPU kernels alongside syscalls and SIMD operations
- ✅ **Performance comparison**: Compare SIMD (227μs) vs GPU (60ms) for same operation
- ✅ **Bottleneck identification**: GPU kernel is 265x slower (indicates small data or transfer overhead)

### 2.3 Span Attributes

**Resource-Level Attributes (once at startup):**
```json
{
  "resource": {
    "service.name": "renacer",
    "compute.library": "trueno",      // Sprint 32
    "gpu.library": "wgpu",            // NEW Sprint 37
    "gpu.library.version": "23.0.0",
    "process.pid": 12345
  }
}
```

**Span-Level Attributes (per GPU kernel):**
```json
{
  "span.name": "gpu_kernel: sum_aggregation",
  "span.kind": "INTERNAL",
  "attributes": {
    "gpu.backend": "wgpu",
    "gpu.kernel": "sum_aggregation",
    "gpu.duration_us": 60000,
    "gpu.workgroup_size": "[256,1,1]",
    "gpu.elements": 1000000,
    "gpu.is_slow": true,
    "gpu.threshold_us": 100
  },
  "status": "OK"
}
```

---

## 3. Phase 1: wgpu Timestamp Query Integration

### 3.1 New Data Structure: `GpuKernel`

**File:** `src/otlp_exporter.rs` (extend existing)

**Add Struct:**
```rust
/// GPU kernel metadata for tracing (Sprint 37)
///
/// Represents a single GPU kernel execution (compute shader, render pass, etc.)
/// captured via wgpu timestamp queries.
#[derive(Debug, Clone)]
pub struct GpuKernel {
    /// Kernel name (e.g., "sum_aggregation", "matrix_multiply")
    pub kernel: String,
    /// Total duration in microseconds
    pub duration_us: u64,
    /// GPU backend (always "wgpu" for Phase 1)
    pub backend: &'static str,
    /// Workgroup size for compute shaders (e.g., "[256,1,1]")
    pub workgroup_size: Option<String>,
    /// Number of elements processed (if known)
    pub elements: Option<usize>,
    /// Whether this kernel exceeded the slow threshold (>100μs)
    pub is_slow: bool,
}
```

### 3.2 OTLP Exporter Extension

**File:** `src/otlp_exporter.rs` (extend existing)

**Add Method:**
```rust
impl OtlpExporter {
    /// Record a GPU kernel execution as a span (Sprint 37)
    ///
    /// Exports GPU kernel timing captured via wgpu-profiler timestamp queries.
    /// Follows Sprint 32's adaptive sampling pattern (only trace if duration > threshold).
    ///
    /// # Arguments
    ///
    /// * `kernel` - Metadata about the GPU kernel execution
    ///
    /// # Adaptive Sampling
    ///
    /// This method should only be called if duration >= threshold (default 100μs).
    /// The caller (GpuProfilerWrapper) handles sampling decisions.
    #[cfg(feature = "gpu-tracing")]
    pub fn record_gpu_kernel(&self, kernel: GpuKernel) {
        let mut span = self
            .tracer
            .span_builder(format!("gpu_kernel: {}", kernel.kernel))
            .with_kind(SpanKind::Internal)
            .with_attributes({
                let mut attrs = vec![
                    KeyValue::new("gpu.backend", kernel.backend.to_string()),
                    KeyValue::new("gpu.kernel", kernel.kernel.clone()),
                    KeyValue::new("gpu.duration_us", kernel.duration_us as i64),
                    KeyValue::new("gpu.is_slow", kernel.is_slow),
                ];

                // Optional attributes
                if let Some(ref wg_size) = kernel.workgroup_size {
                    attrs.push(KeyValue::new("gpu.workgroup_size", wg_size.clone()));
                }
                if let Some(elements) = kernel.elements {
                    attrs.push(KeyValue::new("gpu.elements", elements as i64));
                }

                attrs
            })
            .start(&self.tracer);

        span.set_status(Status::Ok);
        span.end();
    }
}
```

**Update Resource Attributes:**
```rust
impl OtlpExporter {
    pub fn new(config: OtlpConfig, trace_context: Option<TraceContext>) -> Result<Self> {
        // ... existing code ...

        let resource = Resource::builder()
            .with_service_name(config.service_name.clone())
            .with_attributes(vec![
                // Sprint 32: SIMD compute tracing
                KeyValue::new("compute.library", "trueno"),
                KeyValue::new("compute.library.version", "0.4.0"),
                KeyValue::new("compute.tracing.abstraction", "block_level"),

                // NEW Sprint 37: GPU kernel tracing
                #[cfg(feature = "gpu-tracing")]
                KeyValue::new("gpu.library", "wgpu"),
                #[cfg(feature = "gpu-tracing")]
                KeyValue::new("gpu.tracing.abstraction", "kernel_level"),
            ])
            .build();

        // ... rest of setup ...
    }
}
```

### 3.3 GPU Profiler Wrapper

**File:** `src/gpu_tracer.rs` (NEW)

**Purpose:**
- Wrap `wgpu-profiler::GpuProfiler`
- Convert wgpu profiling results → `GpuKernel` structs
- Apply adaptive sampling (duration > 100μs)
- Export to OTLP via `OtlpExporter::record_gpu_kernel()`

**Implementation:**
```rust
//! GPU kernel tracing wrapper for wgpu-profiler (Sprint 37)
//!
//! Integrates wgpu timestamp queries with Renacer's OTLP export infrastructure.
//! Follows Sprint 32's adaptive sampling and block-level tracing patterns.

#[cfg(feature = "gpu-tracing")]
use anyhow::Result;
#[cfg(feature = "gpu-tracing")]
use wgpu_profiler::{GpuProfiler, GpuProfilerSettings};

use crate::otlp_exporter::{GpuKernel, OtlpExporter};

/// Configuration for GPU kernel tracing
#[derive(Debug, Clone)]
pub struct GpuTracerConfig {
    /// Minimum duration to trace (default: 100μs, same as Sprint 32 SIMD tracing)
    pub threshold_us: u64,
    /// Trace all kernels regardless of duration (debug mode)
    pub trace_all: bool,
}

impl Default for GpuTracerConfig {
    fn default() -> Self {
        GpuTracerConfig {
            threshold_us: 100,
            trace_all: false,
        }
    }
}

/// Wrapper around wgpu-profiler that exports to OTLP
#[cfg(feature = "gpu-tracing")]
pub struct GpuProfilerWrapper {
    profiler: GpuProfiler,
    otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
    config: GpuTracerConfig,
}

#[cfg(feature = "gpu-tracing")]
impl GpuProfilerWrapper {
    /// Create a new GPU profiler wrapper
    ///
    /// # Arguments
    ///
    /// * `device` - wgpu Device
    /// * `queue` - wgpu Queue (for timestamp_period)
    /// * `otlp_exporter` - Optional OTLP exporter for trace export
    /// * `config` - Tracing configuration (thresholds, sampling)
    pub fn new(
        device: &wgpu::Device,
        otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
        config: GpuTracerConfig,
    ) -> Result<Self> {
        let settings = GpuProfilerSettings::default();
        let profiler = GpuProfiler::new(&device, settings)?;

        Ok(GpuProfilerWrapper {
            profiler,
            otlp_exporter,
            config,
        })
    }

    /// Get a reference to the underlying wgpu-profiler
    ///
    /// Users instrument their wgpu code with standard wgpu-profiler API:
    /// ```ignore
    /// let mut scope = wrapper.profiler_mut().scope("kernel_name", &mut encoder);
    /// let mut compute_pass = scope.scoped_compute_pass("compute");
    /// // ... GPU commands ...
    /// ```
    pub fn profiler_mut(&mut self) -> &mut GpuProfiler {
        &mut self.profiler
    }

    /// Process finished GPU profiling frame and export to OTLP
    ///
    /// Call this after `queue.submit()` and `profiler.end_frame()`.
    ///
    /// # Arguments
    ///
    /// * `queue` - wgpu Queue (for timestamp_period)
    pub fn export_frame(&mut self, queue: &wgpu::Queue) {
        if let Some(frame_data) = self
            .profiler
            .process_finished_frame(queue.get_timestamp_period())
        {
            // Convert wgpu-profiler results to GpuKernel structs
            for scope in &frame_data {
                let duration_us = (scope.duration * 1_000_000.0) as u64;

                // Adaptive sampling: Only export if duration > threshold OR debug mode
                if self.config.trace_all || duration_us >= self.config.threshold_us {
                    if let Some(ref exporter) = self.otlp_exporter {
                        let kernel = GpuKernel {
                            kernel: scope.label.clone(),
                            duration_us,
                            backend: "wgpu",
                            workgroup_size: None, // TODO: Extract from wgpu metadata
                            elements: None,       // TODO: User-provided via scope metadata
                            is_slow: duration_us > self.config.threshold_us,
                        };

                        exporter.record_gpu_kernel(kernel);
                    }
                }
            }
        }
    }
}

// Stub implementation when GPU tracing feature is disabled
#[cfg(not(feature = "gpu-tracing"))]
pub struct GpuProfilerWrapper;

#[cfg(not(feature = "gpu-tracing"))]
impl GpuProfilerWrapper {
    pub fn new(
        _device: &wgpu::Device,
        _otlp_exporter: Option<std::sync::Arc<OtlpExporter>>,
        _config: GpuTracerConfig,
    ) -> Result<Self> {
        anyhow::bail!("GPU tracing support not compiled in. Enable the 'gpu-tracing' feature.");
    }
}
```

### 3.4 User Integration Example

**File:** User's wgpu application (e.g., `trueno-db/src/gpu_executor.rs`)

```rust
use renacer::{GpuProfilerWrapper, GpuTracerConfig, OtlpConfig, OtlpExporter};
use wgpu;

fn main() {
    // Setup OTLP exporter (same as Sprint 30)
    let otlp_config = OtlpConfig::new(
        "http://localhost:4317".to_string(),
        "trueno-db".to_string(),
    );
    let otlp_exporter = OtlpExporter::new(otlp_config, None).unwrap();
    let otlp_arc = std::sync::Arc::new(otlp_exporter);

    // Setup GPU profiler wrapper
    let mut gpu_tracer = GpuProfilerWrapper::new(
        &device,
        Some(otlp_arc.clone()),
        GpuTracerConfig::default(),
    )
    .unwrap();

    // Instrument GPU code (standard wgpu-profiler API)
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut scope = gpu_tracer.profiler_mut().scope("sum_aggregation", &mut encoder);
        let mut compute_pass = scope.scoped_compute_pass("compute");

        compute_pass.set_pipeline(&pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups, 1, 1);
    }

    gpu_tracer.profiler_mut().resolve_queries(&mut encoder);
    queue.submit(Some(encoder.finish()));
    gpu_tracer.profiler_mut().end_frame().unwrap();

    // Export GPU profiling results to OTLP
    gpu_tracer.export_frame(&queue);
}
```

---

## 4. Implementation Plan

### 4.1 Sprint 37 Checklist (Phase 1: wgpu)

**RED Phase (Tests First):**

**File:** `tests/sprint37_gpu_kernel_tracing_tests.rs`

```rust
#[test]
#[cfg(feature = "gpu-tracing")]
fn test_gpu_kernel_traced_when_slow() {
    // Test that slow GPU kernels (>100μs) are traced
}

#[test]
#[cfg(feature = "gpu-tracing")]
fn test_gpu_kernel_not_traced_when_fast() {
    // Test that fast GPU kernels (<100μs) are NOT traced (adaptive sampling)
}

#[test]
#[cfg(feature = "gpu-tracing")]
fn test_gpu_kernel_attributes() {
    // Test span attributes (backend, kernel, duration, workgroup_size)
}

#[test]
#[cfg(feature = "gpu-tracing")]
fn test_resource_level_gpu_attributes() {
    // Test gpu.library at Resource level, not Span level
}

#[test]
#[cfg(feature = "gpu-tracing")]
fn test_debug_mode_traces_all_kernels() {
    // Test --trace-gpu-all flag bypasses threshold
}

#[test]
#[cfg(feature = "gpu-tracing")]
fn test_gpu_and_simd_unified_trace() {
    // Test that GPU kernels and SIMD compute blocks appear in same trace
}
```

**GREEN Phase (Implementation):**
1. Add `wgpu` and `wgpu-profiler` dependencies to `Cargo.toml` (feature: `gpu-tracing`)
2. Add `GpuKernel` struct to `src/otlp_exporter.rs` (~20 lines)
3. Add `record_gpu_kernel()` method to `OtlpExporter` (~40 lines)
4. Add GPU Resource-level attributes to `OtlpExporter::new()` (~5 lines)
5. Create `src/gpu_tracer.rs` with `GpuProfilerWrapper` (~150 lines)
6. Add module exports in `src/lib.rs`
7. Implement 6+ integration tests

**REFACTOR Phase:**
1. Add unit tests for edge cases
2. Verify complexity ≤10 for all functions
3. Benchmark overhead (target: <2% like Sprint 32)

**Total Code:** ~250 lines (following Sprint 32's minimalist approach)

### 4.2 Cargo.toml Changes

**Add to `[dependencies]`:**
```toml
# GPU kernel tracing (Sprint 37)
wgpu = { version = "23.0", optional = true }
wgpu-profiler = { version = "0.18", optional = true }
```

**Add to `[features]`:**
```toml
# GPU kernel-level tracing (Sprint 37)
gpu-tracing = ["dep:wgpu", "dep:wgpu-profiler", "otlp"]
```

**Update `default`:**
```toml
default = ["otlp"]  # gpu-tracing is opt-in
```

### 4.3 CLI Flags (Extend Sprint 32 pattern)

**New Flags (Sprint 37):**
```bash
--trace-gpu              # Enable GPU kernel tracing (default: adaptive sampling)
--trace-gpu-all          # Debug mode: trace ALL kernels (bypass 100μs threshold)
--trace-gpu-threshold N  # Custom threshold (default: 100μs)
```

**Examples:**
```bash
# Default: Trace only slow GPU kernels (>100μs)
renacer --otlp-endpoint http://localhost:4317 --trace-gpu -- ./trueno-db-server

# Debug mode: Trace all GPU kernels
renacer --otlp-endpoint http://localhost:4317 --trace-gpu-all -- ./app

# Custom threshold: Trace GPU kernels >50μs
renacer --otlp-endpoint http://localhost:4317 --trace-gpu --trace-gpu-threshold 50 -- ./app
```

---

## 5. Testing Strategy

### 5.1 Integration Tests (6+ tests)

**File:** `tests/sprint37_gpu_kernel_tracing_tests.rs`

**Test Matrix:**

| Test Case | Duration | Expected Behavior |
|-----------|----------|-------------------|
| Fast kernel | 50μs | ❌ No span (below threshold) |
| Slow kernel | 5ms | ✅ Span exported |
| Debug mode | 50μs | ✅ Span exported (--trace-gpu-all) |
| Unified trace | varies | ✅ GPU + SIMD + syscall spans in same trace |
| wgpu feature disabled | N/A | ✅ Compile-time error with clear message |

**Total Tests:** 6+ integration tests

### 5.2 Performance Tests

**Benchmark:** `benches/gpu_kernel_overhead.rs`

```rust
fn bench_gpu_compute_no_tracing(c: &mut Criterion) {
    // Baseline: GPU compute without any profiling
}

fn bench_gpu_compute_with_profiler(c: &mut Criterion) {
    // wgpu-profiler overhead only (no OTLP export)
}

fn bench_gpu_compute_with_otlp_export(c: &mut Criterion) {
    // Full overhead: wgpu-profiler + OTLP export + adaptive sampling
}
```

**Target:**
- wgpu-profiler overhead: <1% (per wgpu-profiler docs)
- OTLP export overhead: <1% (adaptive sampling skips fast kernels)
- **Total overhead: <2%** (same as Sprint 32 SIMD tracing)

---

## 6. Performance Impact

### 6.1 Overhead Analysis

| Scenario | Overhead | Spans/sec | Acceptable? |
|----------|----------|-----------|-------------|
| No tracing | 0% | 0 | ✅ Baseline |
| wgpu-profiler only | <1% | 0 | ✅ Negligible |
| OTLP export (adaptive) | <2% | <100 | ✅ **Safe** |
| OTLP export (debug mode) | ~5% | <500 | ✅ Developer use only |

**Toyota Way Compliance:**
- ✅ **Jidoka**: Adaptive sampling prevents DoS on tracing backend
- ✅ **Muda**: Kernel-level tracing (not per-instruction)
- ✅ **Safe by default**: Debug mode requires explicit flag

### 6.2 GPU Timestamp Query Overhead

**Per wgpu-profiler documentation:**
- Timestamp query overhead: **~5-10μs per kernel** (one query at start, one at end)
- Impact: Negligible for kernels >100μs (our adaptive sampling threshold)
- Impact: High for micro-kernels <10μs (which we skip via adaptive sampling)

**Design Decision:**
- ✅ Default threshold (100μs) ensures overhead <10%
- ✅ Users can increase threshold for production (e.g., `--trace-gpu-threshold 1000`)

---

## 7. Future Work

### 7.1 Phase 2: CUDA Support (Sprint 38+)

**CUDA Profiling Tools:**
- **CUPTI** (CUDA Profiling Tools Interface)
- **NVTX** (NVIDIA Tools Extension)
- Similar adaptive sampling pattern

**Example:**
```bash
renacer --otlp-endpoint http://localhost:4317 --trace-gpu-cuda -- ./cuda-app
```

### 7.2 Phase 3: ROCm Support (Sprint 39+)

**AMD GPU Profiling:**
- **ROCProfiler** for AMD GPUs
- HIP runtime integration

### 7.3 Phase 4: Memory Transfer Tracking

**Extend `GpuKernel` struct:**
```rust
pub struct GpuMemoryTransfer {
    pub direction: TransferDirection,  // CPUToGPU, GPUToCPU
    pub bytes: usize,
    pub duration_us: u64,
}
```

**Use case:**
- Identify PCIe bandwidth bottlenecks
- Detect excessive CPU ↔ GPU transfers

### 7.4 Integration with wgpu Features

**Device Features Detection:**
- Check `device.features().contains(Features::TIMESTAMP_QUERY)`
- Graceful degradation if timestamp queries not supported

**Web Support:**
- Timestamp queries are **native-only** (not supported in WebGPU/WASM)
- Feature flag ensures no overhead for web targets

---

## 8. Success Criteria

### 8.1 Technical Requirements

- ✅ GPU kernel spans appear in Jaeger/Tempo alongside syscalls and SIMD compute blocks
- ✅ <2% performance overhead with adaptive sampling (threshold: 100μs)
- ✅ No DoS on tracing backend (max 100 spans/second per process)
- ✅ 6+ integration tests
- ✅ Backward compatible (works without `gpu-tracing` feature)
- ✅ Works with wgpu 23.0+ (latest stable)

### 8.2 Business Value

**User can answer:**
- ✅ "Why did my GPU aggregation take 60ms instead of 5ms?"
- ✅ "Is the GPU kernel slow, or is it CPU ↔ GPU transfer overhead?"
- ✅ "Should I use GPU or SIMD for this dataset size?"

**User CANNOT answer (out of scope for Phase 1):**
- ❌ "What is the occupancy of my GPU?" (requires deeper GPU profiling)
- ❌ "Which CUDA cores executed this kernel?" (CUDA-specific, Phase 2)

---

## 9. Conclusion

### 9.1 Toyota Way Compliance

**v1.0 Specification Principles:**

✅ **Genchi Genbutsu (Go and See):** Reuse proven wgpu-profiler infrastructure. No custom timestamp query management.

✅ **Jidoka (Stop the Line):** Mandatory adaptive sampling (100μs threshold). Cannot DoS tracing backend.

✅ **Muda (Eliminate Waste):** Kernel-level tracing (~250 lines total). No per-instruction overhead.

✅ **Poka-Yoke (Mistake Proofing):** Feature flag prevents accidental overhead. Graceful degradation if device doesn't support timestamp queries.

**Document Status:** ✅ Ready for Implementation

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | Claude Code | Initial specification (Issue #16) |

**Status:** ✅ Ready for Implementation (Sprint 37)
**Approval Required:** Product Owner (Noah Gift)
**Next Review:** Post-Sprint 37 Retrospective
**Related Specs:**
- `trueno-tracing-integration-spec.md` (Sprint 32 - SIMD compute tracing)
- `deep-strace-rust-wasm-binary-spec.md` (Core Renacer spec)
- Issue #16 (GitHub - GPU Kernel-Level Tracing Feature Request)

**Related Issues:**
- Sprint 32: SIMD compute tracing (completed ✅)
- Sprint 33: W3C Trace Context (completed ✅)
- Sprint 34: Integration tests (completed ✅)
- Issue #16: GPU kernel-level tracing (in progress 🎯)
