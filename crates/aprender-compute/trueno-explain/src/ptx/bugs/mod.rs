//! PTX Bug Detection - Static Analysis for GPU Kernel Bugs
//!
//! Shattered from monolithic bugs.rs into:
//! - `types` - Bug severity, classification, and report types
//! - `analyzer` - `PtxBugAnalyzer` implementation
//! - `coverage` - PTX coverage tracking
//!
//! Bug Classes (from probar `gpu_pixels`):
//! - P0 Critical: `SharedMemU64Addressing`, `LoopBranchToEnd`, `MissingBarrierSync`
//! - P1 High: `NonInPlaceLoopAccumulator`, `RegisterSpills`
//! - P2 Medium: `RedundantMoves`, `UnoptimizedMemoryPattern`

mod analyzer;
mod coverage;
mod types;

pub use analyzer::PtxBugAnalyzer;
pub use coverage::{PtxCoverageReport, PtxCoverageTracker, PtxCoverageTrackerBuilder, PtxFeature};
pub use types::{BugSeverity, PtxBug, PtxBugClass, PtxBugReport};

#[cfg(test)]
mod tests;
