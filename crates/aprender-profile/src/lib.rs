//! Renacer - Pure Rust system call tracer with source-aware correlation
//!
//! This library provides the core functionality for tracing system calls
//! in Rust binaries, with support for DWARF debug information, function profiling,
//! and comprehensive filtering.
// Allow unwrap/expect in test code — production code is lint-clean
#![cfg_attr(test, allow(clippy::unwrap_used, clippy::unwrap_in_result))]
// PMAT-139: enforce `// SAFETY:` on every unsafe block (none in lib/bins today).
#![deny(clippy::undocumented_unsafe_blocks)]
// Pedantic lints carried over from pre-monorepo renacer crate.
// Follow-up: fix per-lint in dedicated PRs (out of scope for APR-MONO sibling-alias MUDA fix).
#![allow(
    clippy::redundant_else,
    clippy::implicit_hasher,
    clippy::struct_excessive_bools,
    clippy::doc_link_with_quotes,
    clippy::inline_always,
    clippy::format_collect,
    clippy::needless_continue,
    clippy::ref_option,
    clippy::used_underscore_binding
)]
#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;
pub mod adaptive_backend; // Sprint 40: Adaptive Backend Selection (Specification Section 5.2)
pub mod adaptive_sampler; // Sprint 40: Adaptive Sampling (Specification Section 7.3)
pub mod analysis; // REN-002: Architectural anti-pattern detection (§27)
pub mod anomaly;
pub mod anti_patterns; // Sprint 41: Anti-pattern detection (God Process, Tight Loop, PCIe)
#[cfg(target_os = "linux")]
pub mod arch; // Architecture-specific ptrace register access (x86_64 + aarch64)
pub mod assertion_dsl; // Sprint 44: renacer.toml parser for build-time assertions
pub mod assertion_engine; // Sprint 44: Assertion evaluation engine (Toyota Way: Andon)
pub mod assertion_types; // Sprint 44: Build-time trace assertion types (Toyota Way: Andon)
pub mod autoencoder;
pub mod brick_tracer; // Sprint 58: ComputeBrick-aware tracing for trueno/cbtop integration
pub mod causal_graph; // Sprint 41: Causal graph construction for critical path analysis
pub mod chaos;
#[cfg(target_os = "linux")]
pub mod cli;
pub mod cluster; // Single-Shot Compile Tooling: TOML-based syscall clustering (Section 6.1)
pub mod critical_path; // Sprint 41: Critical path analysis (longest path in DAG)
pub mod csv_output;
pub mod cuda_tracer; // Sprint 38: CUDA kernel-level tracing via CUPTI
pub mod decision_export; // Sprint 49: Decision trace OTLP export (Ticket #19)
pub mod decision_trace;
pub mod depyler_ingest; // Sprint 49: Depyler decision trace ingestion (Ticket #18)
pub mod dwarf;
pub mod experiment_span; // REN-001: Experiment span types for entrenar integration
pub mod filter;
pub mod function_profiler;
pub mod gpu_tracer; // Sprint 37: GPU kernel-level tracing for wgpu
pub mod hpu;
pub mod html_output;
pub mod isolation_forest;
pub mod json_output;
pub mod lamport_clock; // Sprint 40: Lamport logical clocks for causal ordering (Toyota Way: Poka-Yoke)
pub mod lazy_span; // Sprint 36: Lazy span creation for performance
pub mod ml_anomaly;
pub mod ml_pipeline; // Sprint 48: Enhanced ML pipeline (StandardScaler, DBSCAN, LOF, PCA)
pub mod model_persistence; // Sprint 48: .apr model persistence (Toyota Way: Muda elimination)
pub mod otlp_exporter;
pub mod otlp_types;
#[cfg(target_os = "linux")]
pub mod process_tracer; // Sprint 59: Process-level syscall tracing for ptop integration (SPEC-057)
pub mod profiling;
pub mod regression; // Single-Shot Compile Tooling: Statistical regression detection (Section 6.4)
pub mod ring_buffer; // Sprint 40: Lock-free ring buffer for span export (Toyota Way: Heijunka)
pub mod rle_compression; // Sprint 41: Run-length encoding for tight loop compression (Toyota Way: Muda)
pub mod semantic_equivalence; // Sprint 40: Semantic Equivalence (Specification Section 6.3)
pub mod sequence; // Single-Shot Compile Tooling: N-gram sequence mining (Section 6.1.1)
pub mod span_pool; // Sprint 36: Memory pool for span allocations
pub mod span_record; // Sprint 40: Parquet-compatible span schema
#[cfg(target_os = "linux")]
pub mod stack_unwind;
pub mod stats;
pub mod syscalls;
pub mod time_attribution; // Single-Shot Compile Tooling: Time-weighted attribution (Section 6.2)
pub mod trace_context; // Sprint 33: W3C Trace Context propagation
#[cfg(target_os = "linux")]
pub mod tracer;
pub mod transpiler_map;
pub mod trueno_db_storage; // Sprint 40: Trueno-DB Parquet storage for golden thread traces
pub mod unified_trace; // Sprint 40: Unified Trace Model (Specification Section 3.1)
#[cfg(target_os = "linux")]
pub mod validate; // Sprint 50: Golden trace validation (APR Runtime Support)
pub mod validation_engine; // Sprint 40: ValidationEngine for Batuta Integration (Specification Section 5.1)
                           // visualize module removed — was ratatui-only (Sprint 52-57)
pub mod alerting; // Sprint 57: Alert engine (threshold, rate, anomaly-based)
pub mod metrics; // Sprint 56: Counter, Gauge, Histogram with SIMD
