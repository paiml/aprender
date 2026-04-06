//! Tools for TUI comparison, verification, and benchmarking.
//!
//! This module contains utilities for pixel-perfect TUI comparison:
//! - CIEDE2000 color difference (ΔE00)
//! - Character-level diff (CLD)
//! - Structural similarity (SSIM)
//!
//! And headless benchmarking tools:
//! - `HeadlessCanvas` for in-memory rendering
//! - `RenderMetrics` for performance statistics
//! - `BenchmarkHarness` for automated benchmarks

pub mod bench;
mod color_diff;
#[cfg(feature = "tui-compare")]
mod tui_compare;

pub use bench::{
    BenchmarkHarness, BenchmarkResult, ComparisonResult, DeterministicContext, FrameTimeStats,
    HeadlessCanvas, MemoryStats, PerformanceTargets, RenderMetrics,
};
pub use color_diff::{average_delta_e, ciede2000, rgb_to_lab, DeltaECategory, Lab, Rgb};
#[cfg(feature = "tui-compare")]
pub use tui_compare::{
    compare_tui, generate_report, DiffCell, PanelResult, PanelThreshold, TuiComparisonConfig,
    TuiComparisonResult,
};
