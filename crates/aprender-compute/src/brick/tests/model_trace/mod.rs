//! Model-Level Inference Tracing Tests (Phase 13)
//!
//! Split from monolithic model_trace.rs for maintainability.
//!
//! ## Submodules
//!
//! - `core_traces` - F250-F275: Core model trace tests (tensor stats, attention, logit, quant, kv cache)
//! - `coverage` - F276-F298: Additional coverage tests for Phase 13
//! - `tile_profiling` - F356-F378: Tile profiling tests (TILING-SPEC-001)
//! - `simd_quant_graph` - SIMD-EXP, QUANT-Q5K/Q6K, ExecutionGraph coverage

mod core_traces;
mod coverage;
mod simd_quant_graph;
mod tile_profiling;
