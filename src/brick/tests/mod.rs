//! Brick Module Tests
//!
//! Shattered test monolith for improved maintainability.
//!
//! ## Test Modules
//!
//! - `ops` - Basic operations (TokenBudget, DotOp, AddOp, MatmulOp, SoftmaxOp)
//! - `attention` - AttentionOp tests (PMAT-017)
//! - `budget` - ByteBudget and coverage tests
//! - `fused` - Fused LLM operations tests (PMAT-PERF-009)
//! - `profiler` - BrickProfiler tests (PAR-073, PAR-200, PMAT-451)
//! - `falsification_v2` - PAR-200/201 falsification tests (F101-F120)
//! - `cpa` - Critical Path Analysis tests (Phase 9, F128-F140)
//! - `phases` - Phase 11-12 tests (F150-F175)
//! - `model_trace` - Model-level inference tracing (F250-F270)

mod attention;
mod budget;
mod cpa;
mod falsification_v2;
mod fused;
mod model_trace;
mod ops;
mod phases;
mod profiler;
