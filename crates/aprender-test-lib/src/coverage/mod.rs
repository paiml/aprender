//! WASM Coverage Tooling for Probar
//!
//! Per spec: `docs/specifications/probar-wasm-coverage-tooling.md`
//!
//! This module implements a novel WASM coverage instrumentation framework
//! using the Batuta Sovereign AI Stack primitives.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │  PROBAR COVERAGE ARCHITECTURE                                    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  WASM Module → Block Decomposer → Coverage Executor → Report    │
//! │                     ↓                    ↓                       │
//! │              CFG Analysis         Trueno SIMD Aggregation       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Toyota Way Principles Applied
//!
//! - **Poka-Yoke**: Type-safe BlockId, FunctionId, EdgeId prevent errors
//! - **Muda**: Thread-local buffering eliminates atomic contention
//! - **Jidoka**: Soft Jidoka distinguishes Stop vs LogAndContinue
//! - **Heijunka**: Superblock tiling amortizes scheduling overhead

mod block;
mod collector;
mod executor;
pub mod formatters;
mod hypotheses;
mod jidoka;
mod memory;
mod report;
mod superblock;
mod thread_local;

pub use block::{BlockId, EdgeId, FunctionId};
pub use collector::{CoverageCollector, CoverageConfig, Granularity};
pub use executor::{CoverageExecutor, SuperblockResult};
pub use formatters::{CoberturaFormatter, HtmlFormatter, HtmlReportConfig, LcovFormatter, Theme};
pub use hypotheses::{CoverageHypothesis, NullificationConfig, NullificationResult};
pub use jidoka::{CoverageViolation, JidokaAction, TaintedBlocks};
pub use memory::CoverageMemoryView;
pub use report::{BlockCoverage, CoverageReport, CoverageSummary};
pub use superblock::{Superblock, SuperblockBuilder, SuperblockId};
pub use thread_local::ThreadLocalCounters;

#[cfg(test)]
mod tests;
