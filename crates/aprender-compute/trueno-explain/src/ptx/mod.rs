//! PTX analysis module
//!
//! Parses and analyzes NVIDIA PTX assembly to detect:
//! - Register pressure (Muda of Transport when spills occur)
//! - Memory access patterns (Muda of Waiting when uncoalesced)
//! - Warp divergence (Heijunka imbalance)
//! - Bug patterns (probar-style static analysis)

mod bugs;
mod parser;

pub use bugs::{
    BugSeverity, PtxBug, PtxBugAnalyzer, PtxBugClass, PtxBugReport, PtxCoverageReport,
    PtxCoverageTracker, PtxCoverageTrackerBuilder, PtxFeature,
};
pub use parser::PtxAnalyzer;
