//! trueno-explain: PTX/SIMD/wgpu Visualization and Tracing CLI
//!
//! A CLI tool that visualizes and traces code generation flows across
//! Trueno's execution targets. Implements the Toyota Way principle of
//! **Genchi Genbutsu** (Go and See) by making invisible compiler
//! transformations visible.
//!
//! # Features
//!
//! - PTX register pressure analysis
//! - Memory access pattern detection
//! - Warp divergence visualization (Heijunka)
//! - Muda (waste) detection and elimination suggestions
//!
//! # Example
//!
//! ```rust
//! use trueno_explain::ptx::PtxAnalyzer;
//! use trueno_explain::analyzer::Analyzer;
//!
//! let ptx = r#"
//!     .entry test() {
//!         .reg .f32 %f<16>;
//!         ret;
//!     }
//! "#;
//!
//! let analyzer = PtxAnalyzer::new();
//! let report = analyzer.analyze(ptx).expect("PTX analysis should succeed");
//! assert_eq!(report.registers.f32_regs, 16);
//! ```

// Allow some pedantic lints for this CLI tool
#![cfg_attr(test, allow(clippy::disallowed_methods))] // Tests use unwrap() freely
#![cfg_attr(test, allow(clippy::float_cmp))] // Tests compare exact float values
#![allow(clippy::cast_precision_loss)] // Acceptable for display percentages
#![allow(clippy::cast_possible_truncation)] // Instruction counts won't exceed u32
#![allow(clippy::format_push_string)] // Performance not critical for CLI
#![allow(clippy::too_many_lines)] // Will refactor in future sprints
#![allow(clippy::unwrap_used)] // Safe for compile-time constant regex
#![allow(clippy::unused_self)] // Methods may need self for future extensibility
#![allow(clippy::map_unwrap_or)] // Style preference
#![allow(clippy::unnecessary_literal_bound)] // Trait signature constraint

pub mod analyzer;
pub mod compare;
pub mod diff;
pub mod error;
pub mod output;
pub mod ptx;
pub mod simd;
pub mod tui;
pub mod wgpu;

pub use analyzer::{AnalysisReport, Analyzer, MudaWarning, RegisterUsage};
pub use compare::{compare_analyses, format_comparison_json, format_comparison_text};
pub use diff::{compare_reports, format_diff_json, format_diff_text, DiffReport, DiffThresholds};
pub use error::{ExplainError, Result};
pub use output::{format_json, format_text, OutputFormat};
pub use ptx::{
    BugSeverity, PtxAnalyzer, PtxBug, PtxBugAnalyzer, PtxBugClass, PtxBugReport, PtxCoverageReport,
    PtxCoverageTracker, PtxCoverageTrackerBuilder, PtxFeature,
};
pub use simd::{SimdAnalyzer, SimdArch, SimdInstructionCounts};
pub use tui::{run_tui, TuiApp};
pub use wgpu::{WgpuAnalyzer, WorkgroupSize};
