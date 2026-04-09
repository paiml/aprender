# GPU Memory Transfer Tracking: Phase 4

Version: 1.0
Status: proposed
Date: 2026-04-09

**Version:** 1.0
**Date:** 2025-11-21
**Status:** Specification - Ready for Implementation
**Sprint Target:** 39 (GPU Memory Transfer Tracking)
**GitHub Issue:** #16 (Phase 4)
**Depends On:** Sprint 37 (Phase 1: wgpu kernel tracing)

## Executive Summary

This specification defines **GPU memory transfer observability** for **wgpu applications**, tracking CPU↔GPU data movement to identify PCIe bandwidth bottlenecks. Following **Phase 1's** kernel tracing and **Toyota Way** principles, this spec completes the GPU observability trifecta: **kernel execution + memory transfers + SIMD compute**.

**Business Value:**
- **Transfer Bottleneck Identification**: Identify slow CPU↔GPU transfers (often >10x slower than kernels)
- **PCIe Bandwidth Analysis**: Measure actual vs theoretical bandwidth utilization
- **Memory Optimization**: Guide decisions on buffer sizes, staging strategies
- **Complete GPU Timeline**: See when GPU is computing vs waiting for data

**Key Principle (Toyota Way):**
> *"Make the invisible visible."* - Memory transfers are often the hidden bottleneck. Trace them to find the truth.

---

## Table of Contents

1. [Background and Motivation](#1-background-and-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Phase 4: Memory Transfer Tracking](#3-phase-4-memory-transfer-tracking)
4. [Implementation Plan](#4-implementation-plan)
5. [Testing Strategy](#5-testing-strategy)
6. [Performance Impact](#6-performance-impact)

---

## 1. Background and Motivation

### 1.1 The Hidden Bottleneck Problem

**Common Performance Anti-Pattern:**
```
GPU kernel: 5ms   ✅ Fast!
CPU → GPU transfer: 45ms  ❌ 9x slower (hidden bottleneck)
GPU → CPU transfer: 2ms   ✅ Acceptable
```

**Root Cause:** Developers focus on kernel optimization, miss transfer overhead.

### 1.2 Phase 1-3 Accomplishments

**✅ Phase 1 Complete (Sprint 37):**
- wgpu GPU kernel tracing
- GpuKernel struct + record_gpu_kernel() method
- Adaptive sampling (100μs threshold)
- 9 integration tests passing

**✅ Phase 2 Specified (Sprint 38):**
- CUDA kernel tracing via CUPTI
- Blocked on NVIDIA GPU hardware

**✅ Phase 3 Planned:**
- ROCm (AMD GPU) kernel tracing
- Similar to Phase 2, blocked on AMD hardware

**❌ Memory Transfers Not Tracked:**
- CPU → GPU buffer uploads invisible
- GPU → CPU buffer downloads invisible
- PCIe bandwidth bottlenecks undetected

### 1.3 Use Case: Real-Time Graphics Pipeline

**Example application:** Game rendering with dynamic mesh updates

**Current visibility (Phase 1 only):**
```
Root Span: "process: game_engine"
└─ Span: "gpu_kernel: vertex_shader" - 3ms  ✅ Traced
```

**Hidden bottleneck:**
```
CPU → GPU: Upload mesh data - 25ms     ❌ NOT traced (bottleneck!)
GPU kernel: Process vertices - 3ms      ✅ Traced
GPU → CPU: Readback framebuffer - 1ms  ❌ NOT traced
```

**Desired visibility (Phase 4):**
```
Root Span: "process: game_engine"
├─ Span: "gpu_transfer: mesh_upload" (CPU→GPU) - 25ms           🎯 NEW Phase 4
│   ├─ gpu_transfer.direction: "cpu_to_gpu"
│   ├─ gpu_transfer.bytes: 10485760  (10MB)
│   ├─ gpu_transfer.bandwidth_mbps: 419.4  (25ms for 10MB)
│   └─ gpu_transfer.is_slow: true  (expected <5ms)
├─ Span: "gpu_kernel: vertex_shader" - 3ms                      ✅ Phase 1
└─ Span: "gpu_transfer: framebuffer_readback" (GPU→CPU) - 1ms   🎯 NEW Phase 4
    ├─ gpu_transfer.direction: "gpu_to_cpu"
    ├─ gpu_transfer.bytes: 8294400  (7.9MB)
    └─ gpu_transfer.bandwidth_mbps: 8294.4  (1ms for 8MB)
```

**Insight:** Mesh upload (25ms) is 8.3x slower than kernel execution (3ms) → optimize transfer strategy!

### 1.4 Transfer Types in wgpu

**CPU → GPU (Uploads):**
- `queue.write_buffer()` - Immediate copy to staging, then GPU
- `queue.write_texture()` - Texture uploads
- `encoder.copy_buffer_to_buffer()` - GPU-side copy (fast, already on GPU)

**GPU → CPU (Downloads):**
- `buffer.map_async()` - Asynchronous readback
- `buffer.slice().get_mapped_range()` - Access mapped data

**GPU ↔ GPU (Internal):**
- `encoder.copy_buffer_to_buffer()` - Already tracked by Phase 1 (part of command buffer)

**Phase 4 Scope:** Track CPU ↔ GPU transfers only (the slow ones).

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
│  - NEW: Export GPU memory transfers      🎯 Sprint 39       │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ record_gpu_transfer(GpuMemoryTransfer)
                          │
┌─────────────────────────────────────────────────────────────┐
│  GPU Transfer Tracker (EXTEND: src/gpu_tracer.rs)           │
│  - Wrapper methods: traced_write_buffer(), etc.             │
│  - Wall-clock timing (std::time::Instant)                   │
│  - Convert transfer metadata → GpuMemoryTransfer struct     │
│  - Adaptive sampling (same 100μs threshold)                 │
│  - Export as OTLP spans                                     │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ User calls wrappers instead of direct wgpu
                          │
┌─────────────────────────────────────────────────────────────┐
│  User's wgpu Application                                    │
│  - Replace: queue.write_buffer()                            │
│  - With: transfer_tracker.traced_write_buffer()            │
│  - Minimal code changes                                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Span Hierarchy (Complete GPU Observability)

**Timeline with kernels + transfers:**
```
Root Span: "process: game_engine"
├─ Span: "gpu_transfer: mesh_data_upload" (CPU→GPU) - 25ms
│   ├─ gpu_transfer.direction: "cpu_to_gpu"
│   ├─ gpu_transfer.bytes: 10485760
│   ├─ gpu_transfer.bandwidth_mbps: 419.4
│   └─ gpu_transfer.buffer_usage: "VERTEX"
├─ Span: "gpu_kernel: vertex_shader" - 3ms
│   ├─ gpu.backend: "wgpu"
│   ├─ gpu.kernel: "vertex_shader"
│   └─ gpu.duration_us: 3000
├─ Span: "gpu_kernel: fragment_shader" - 2ms
└─ Span: "gpu_transfer: framebuffer_readback" (GPU→CPU) - 1ms
    ├─ gpu_transfer.direction: "gpu_to_cpu"
    ├─ gpu_transfer.bytes: 8294400
    └─ gpu_transfer.bandwidth_mbps: 8294.4
```

**Benefits:**
- ✅ **Complete GPU timeline**: See kernels AND transfers
- ✅ **Bottleneck identification**: Transfer (25ms) >> kernel (3ms)
- ✅ **Bandwidth analysis**: Actual (419 MB/s) vs theoretical (PCIe 4.0: 32 GB/s)

### 2.3 Span Attributes

**Resource-Level Attributes (once at startup):**
```json
{
  "resource": {
    "service.name": "renacer",
    "gpu.library.wgpu": "23.0.0",
    "gpu.tracing.capabilities": "kernels,transfers"
  }
}
```

**Span-Level Attributes (per transfer):**
```json
{
  "span.name": "gpu_transfer: mesh_data_upload",
  "span.kind": "INTERNAL",
  "attributes": {
    "gpu_transfer.direction": "cpu_to_gpu",
    "gpu_transfer.bytes": 10485760,
    "gpu_transfer.duration_us": 25000,
    "gpu_transfer.bandwidth_mbps": 419.4,
    "gpu_transfer.buffer_usage": "VERTEX",
    "gpu_transfer.is_slow": true,
    "gpu_transfer.threshold_us": 100
  },
  "status": "OK"
}
```

---

## 3. Phase 4: Memory Transfer Tracking

### 3.1 New Data Structure: `GpuMemoryTransfer`

**File:** `src/otlp_exporter.rs` (extend existing)

```rust
/// GPU memory transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    /// CPU → GPU (buffer upload)
    CpuToGpu,
    /// GPU → CPU (buffer download/readback)
    GpuToCpu,
}

impl TransferDirection {
    pub fn as_str(&self) -> &'static str {
        match self {
            TransferDirection::CpuToGpu => "cpu_to_gpu",
            TransferDirection::GpuToCpu => "gpu_to_cpu",
        }
    }
}

/// GPU memory transfer metadata for tracing (Sprint 39 - Phase 4)
///
/// Represents a single CPU↔GPU memory transfer operation captured via wall-clock timing.
#[derive(Debug, Clone)]
pub struct GpuMemoryTransfer {
    /// Transfer name/label (e.g., "mesh_data_upload", "framebuffer_readback")
    pub label: String,
    /// Transfer direction (CPU→GPU or GPU→CPU)
    pub direction: TransferDirection,
    /// Number of bytes transferred
    pub bytes: usize,
    /// Total duration in microseconds
    pub duration_us: u64,
    /// Calculated bandwidth in MB/s
    pub bandwidth_mbps: f64,
    /// Optional buffer usage hint (e.g., "VERTEX", "UNIFORM", "STORAGE")
    pub buffer_usage: Option<String>,
    /// Whether this transfer exceeded the slow threshold (>100μs)
    pub is_slow: bool,
}

impl GpuMemoryTransfer {
    /// Create a new GPU memory transfer record
    ///
    /// Automatically calculates bandwidth from bytes and duration.
    pub fn new(
        label: String,
        direction: TransferDirection,
        bytes: usize,
        duration_us: u64,
        buffer_usage: Option<String>,
        threshold_us: u64,
    ) -> Self {
        // Calculate bandwidth: MB/s = (bytes / 1_000_000) / (duration_us / 1_000_000)
        let bandwidth_mbps = if duration_us > 0 {
            (bytes as f64 * 1_000_000.0) / (duration_us as f64 * 1_048_576.0)
        } else {
            0.0
        };

        GpuMemoryTransfer {
            label,
            direction,
            bytes,
            duration_us,
            bandwidth_mbps,
            buffer_usage,
            is_slow: duration_us > threshold_us,
        }
    }
}
```

### 3.2 OTLP Exporter Extension

**File:** `src/otlp_exporter.rs` (extend existing)

```rust
impl OtlpExporter {
    /// Record a GPU memory transfer as a span (Sprint 39 - Phase 4)
    ///
    /// Exports GPU memory transfer timing (CPU↔GPU) captured via wall-clock measurement.
    /// Follows Sprint 37's adaptive sampling pattern.
    ///
    /// # Arguments
    ///
    /// * `transfer` - Metadata about the GPU memory transfer
    ///
    /// # Adaptive Sampling
    ///
    /// This method should only be called if duration >= threshold (default 100μs).
    /// The caller (transfer tracking wrapper) handles sampling decisions.
    pub fn record_gpu_transfer(&self, transfer: GpuMemoryTransfer) {
        let mut span_attrs = vec![
            KeyValue::new("gpu_transfer.direction", transfer.direction.as_str().to_string()),
            KeyValue::new("gpu_transfer.bytes", transfer.bytes as i64),
            KeyValue::new("gpu_transfer.duration_us", transfer.duration_us as i64),
            KeyValue::new("gpu_transfer.bandwidth_mbps", transfer.bandwidth_mbps),
            KeyValue::new("gpu_transfer.is_slow", transfer.is_slow),
        ];

        // Optional buffer usage
        if let Some(ref usage) = transfer.buffer_usage {
            span_attrs.push(KeyValue::new("gpu_transfer.buffer_usage", usage.clone()));
        }

        let mut span = self
            .tracer
            .span_builder(format!("gpu_transfer: {}", transfer.label))
            .with_kind(SpanKind::Internal)
            .with_attributes(span_attrs)
            .start(&self.tracer);

        span.set_status(Status::Ok);
        span.end();
    }
}
```

### 3.3 Transfer Tracking Wrapper

**File:** `src/gpu_tracer.rs` (extend existing Phase 1 code)

```rust
#[cfg(feature = "gpu-tracing")]
impl GpuProfilerWrapper {
    /// Trace a buffer write operation (CPU → GPU)
    ///
    /// # Arguments
    ///
    /// * `queue` - wgpu Queue
    /// * `buffer` - Target buffer
    /// * `offset` - Byte offset
    /// * `data` - Data to write
    /// * `label` - Transfer label for tracing
    ///
    /// # Example
    ///
    /// ```ignore
    /// gpu_tracer.traced_write_buffer(
    ///     &queue,
    ///     &vertex_buffer,
    ///     0,
    ///     &vertex_data,
    ///     "mesh_upload",
    /// );
    /// ```
    pub fn traced_write_buffer(
        &self,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        offset: wgpu::BufferAddress,
        data: &[u8],
        label: &str,
    ) {
        // Wall-clock timing (simple, accurate enough for transfers)
        let start = std::time::Instant::now();

        // Perform actual write
        queue.write_buffer(buffer, offset, data);

        let duration_us = start.elapsed().as_micros() as u64;
        let bytes = data.len();

        // Adaptive sampling: Only export if duration > threshold OR trace_all
        if self.config.trace_all || duration_us >= self.config.threshold_us {
            if let Some(ref exporter) = self.otlp_exporter {
                let transfer = GpuMemoryTransfer::new(
                    label.to_string(),
                    TransferDirection::CpuToGpu,
                    bytes,
                    duration_us,
                    None, // TODO: Extract buffer usage from buffer descriptor
                    self.config.threshold_us,
                );

                exporter.record_gpu_transfer(transfer);
            }
        }
    }

    /// Trace a buffer map operation (GPU → CPU)
    ///
    /// Returns the mapped buffer slice wrapped with automatic unmap.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let data = gpu_tracer.traced_map_buffer(
    ///     &buffer,
    ///     "framebuffer_readback",
    /// ).await;
    /// // Use data...
    /// // Auto-unmaps when dropped
    /// ```
    pub async fn traced_map_buffer(
        &self,
        buffer: &wgpu::Buffer,
        label: &str,
    ) -> wgpu::BufferView {
        let start = std::time::Instant::now();

        // Start async map operation
        let buffer_slice = buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        // Wait for map to complete (this measures actual transfer time)
        rx.await.unwrap().unwrap();

        let duration_us = start.elapsed().as_micros() as u64;
        let bytes = buffer.size() as usize;

        // Record transfer
        if self.config.trace_all || duration_us >= self.config.threshold_us {
            if let Some(ref exporter) = self.otlp_exporter {
                let transfer = GpuMemoryTransfer::new(
                    label.to_string(),
                    TransferDirection::GpuToCpu,
                    bytes,
                    duration_us,
                    None,
                    self.config.threshold_us,
                );

                exporter.record_gpu_transfer(transfer);
            }
        }

        buffer_slice.get_mapped_range()
    }
}
```

### 3.4 User Integration Example

**Before (Phase 1 - kernels only):**
```rust
// Upload mesh data (no tracing)
queue.write_buffer(&vertex_buffer, 0, &vertex_data);

// Execute kernel (traced ✅)
let mut scope = gpu_tracer.profiler_mut().scope("vertex_shader", &mut encoder);
```

**After (Phase 4 - kernels + transfers):**
```rust
// Upload mesh data (now traced ✅)
gpu_tracer.traced_write_buffer(
    &queue,
    &vertex_buffer,
    0,
    &vertex_data,
    "mesh_data_upload",
);

// Execute kernel (traced ✅)
let mut scope = gpu_tracer.profiler_mut().scope("vertex_shader", &mut encoder);

// Readback results (now traced ✅)
let result_data = gpu_tracer.traced_map_buffer(
    &output_buffer,
    "result_readback",
).await;
```

---

## 4. Implementation Plan

### 4.1 Sprint 39 Checklist (Phase 4: Memory Transfers)

**RED Phase (Tests First):**

**File:** `tests/sprint39_gpu_transfer_tracking_tests.rs`

```rust
#[test]
#[cfg(all(feature = "gpu-tracing", feature = "otlp"))]
fn test_cpu_to_gpu_transfer_traced() {
    // Test that write_buffer is traced with correct attributes
}

#[test]
#[cfg(all(feature = "gpu-tracing", feature = "otlp"))]
fn test_gpu_to_cpu_transfer_traced() {
    // Test that map_async is traced
}

#[test]
#[cfg(all(feature = "gpu-tracing", feature = "otlp"))]
fn test_transfer_bandwidth_calculated() {
    // Test that bandwidth is calculated correctly
}

#[test]
#[cfg(all(feature = "gpu-tracing", feature = "otlp"))]
fn test_kernels_and_transfers_unified_trace() {
    // Test that transfers and kernels appear in same trace
}
```

**GREEN Phase (Implementation):**
1. Add `TransferDirection` enum to `src/otlp_exporter.rs` (~20 lines)
2. Add `GpuMemoryTransfer` struct to `src/otlp_exporter.rs` (~60 lines)
3. Add `record_gpu_transfer()` method to `OtlpExporter` (~40 lines)
4. Add `traced_write_buffer()` to `GpuProfilerWrapper` (~50 lines)
5. Add `traced_map_buffer()` to `GpuProfilerWrapper` (~50 lines)
6. Implement 6+ integration tests (~300 lines)

**Total Code:** ~520 lines

### 4.2 No New Dependencies

**Reuse Phase 1:**
- ✅ wgpu 23.0 (already added in Sprint 37)
- ✅ wgpu-profiler 0.18 (already added in Sprint 37)
- ✅ gpu-tracing feature flag (already defined)

**No new dependencies required!**

---

## 5. Testing Strategy

### 5.1 Integration Tests

**File:** `tests/sprint39_gpu_transfer_tracking_tests.rs`

```rust
#[test]
#[cfg(all(feature = "gpu-tracing", feature = "otlp"))]
fn test_large_transfer_traced() {
    // Upload 10MB buffer, verify:
    // - Span exists with name "gpu_transfer: large_data"
    // - direction = "cpu_to_gpu"
    // - bytes = 10485760
    // - bandwidth_mbps > 0
}

#[test]
#[cfg(all(feature = "gpu-tracing", feature = "otlp"))]
fn test_small_transfer_not_traced() {
    // Upload 100 bytes (fast transfer), verify:
    // - No span exported (adaptive sampling)
}
```

---

## 6. Performance Impact

### 6.1 Overhead Analysis

**Wall-clock timing overhead:** <1μs per transfer (negligible)

**Adaptive sampling:** Same as Phase 1 (100μs threshold)

**Expected overhead:** <0.5% (wall-clock timing is very cheap)

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-11-21 | Claude Code | Initial specification (Issue #16 Phase 4) |

**Status:** ✅ Ready for Implementation
**Dependencies:** Sprint 37 (Phase 1: wgpu kernel tracing)
**Next Steps:** Implement `GpuMemoryTransfer` struct and transfer tracking methods
