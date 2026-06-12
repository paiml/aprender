//! # Trueno-Viz
//!
//! SIMD/GPU/WASM-accelerated visualization library for data science and machine learning.
//!
//! Built on the [trueno](https://crates.io/crates/trueno) core library, trueno-viz provides
//! hardware-accelerated rendering of statistical and scientific visualizations with zero
//! JavaScript/HTML dependencies.
//!
//! ## Features
//!
//! - **Pure Rust**: No JavaScript, HTML, or browser dependencies
//! - **Hardware Acceleration**: Automatic dispatch to SIMD (SSE2/AVX2/AVX512/NEON), GPU, or WASM
//! - **Grammar of Graphics**: Declarative, composable visualization API
//! - **Multiple Outputs**: PNG, SVG, and terminal (ASCII/Unicode) rendering
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use trueno_viz::prelude::*;
//!
//! // Create a scatter plot
//! let plot = ScatterPlot::new()
//!     .x(&[1.0, 2.0, 3.0, 4.0, 5.0])
//!     .y(&[2.0, 4.0, 1.0, 5.0, 3.0])
//!     .color(Rgba::BLUE)
//!     .build();
//!
//! // Render to PNG
//! plot.render_to_file("scatter.png")?;
//! ```
//!
//! ## Feature Flags
//!
//! - `gpu`: Enable GPU compute acceleration
//! - `parallel`: Enable parallel processing with rayon
//! - `ml`: Integration with aprender/entrenar ML libraries
//! - `graph`: Integration with trueno-graph
//! - `db`: Integration with trueno-db
//! - `terminal`: Terminal output support
//! - `svg`: SVG output support
//! - `full`: All features enabled
//!
//! ## Academic References
//!
//! This library implements algorithms from peer-reviewed research:
//!
//! - Wilkinson, L. (2005). *The Grammar of Graphics*. Springer.
//! - Wu, X. (1991). "An Efficient Antialiasing Technique." SIGGRAPH '91.
//! - Douglas, D. H., & Peucker, T. K. (1973). Line simplification algorithm.
//! - Fruchterman, T. M. J., & Reingold, E. M. (1991). Force-directed graph layout.
#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
// Allow unwrap() in tests only - banned in production code (Cloudflare incident 2025-11-18)
#![cfg_attr(test, allow(clippy::unwrap_used))]
// Allow common patterns in graphics/visualization code
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::doc_markdown)]
// ============================================================================
// Core Modules
// ============================================================================
/// Color types and color space conversions.
#[macro_use]
#[allow(unused_macros, unused_variables)]
mod generated_contracts;
pub mod color;
/// Core framebuffer for pixel rendering.
pub mod framebuffer;
/// Geometric primitives (points, lines, rectangles).
pub mod geometry;
/// Scale functions for data-to-visual mappings.
pub mod scale;
// ============================================================================
// Visualization Modules
// ============================================================================
/// Grammar of Graphics implementation.
pub mod grammar;
/// High-level plot types (scatter, heatmap, histogram, etc.).
pub mod plots;
// ============================================================================
// Rendering Modules
// ============================================================================
/// SIMD/GPU acceleration layer.
pub mod accel;
/// Output encoders (PNG, SVG, terminal).
pub mod output;
/// Rendering backends and rasterization.
pub mod render;
// ============================================================================
// Optional Integration Modules
// ============================================================================
/// Ecosystem integrations (trueno-db, trueno-graph, aprender).
pub mod interop;
// NOTE: the `monitor` module (ttop-2.0 btop-like TUI) and its `monitor` feature were
// removed in #1979. The module was deleted in the ttop-2.0 refactor; the feature and its
// gated tests/examples that still imported it could no longer compile.
/// Text prompt interface for declarative visualization DSL.
pub mod prompt;
/// WebAssembly bindings for browser usage.
#[cfg(feature = "wasm")]
#[cfg_attr(docsrs, doc(cfg(feature = "wasm")))]
pub mod wasm;
/// Dashboard widgets for experiment tracking and visualization.
pub mod widgets;
// ============================================================================
// Error Types
// ============================================================================
/// Error types for trueno-viz operations.
pub mod error;
pub use error::{Error, Result};
// ============================================================================
// Prelude
// ============================================================================
/// Commonly used types and traits for convenient imports.
///
/// ```rust,ignore
/// use trueno_viz::prelude::*;
/// ```
pub mod prelude {
    pub use crate::color::{Hsla, Rgba};
    pub use crate::error::{Error, Result};
    pub use crate::framebuffer::Framebuffer;
    pub use crate::geometry::{Line, Point, Rect};
    pub use crate::plots::{
        ConfusionMatrix, Heatmap, HeatmapPalette, Histogram, LineChart, LineSeries, LossCurve,
        PrCurve, RocCurve, ScatterPlot,
    };
    pub use crate::scale::{ColorScale, LinearScale, LogScale, Scale};
    pub use crate::widgets::{ResourceBar, RunRow, RunStatus, RunTable, Sparkline, TrendDirection};
    pub use batuta_common::display::WithDimensions;
}
// ============================================================================
// Re-exports
// ============================================================================
/// Re-export trueno for direct access to SIMD operations.
pub use trueno;
// ============================================================================
// Tests
// ============================================================================
#[cfg(test)]
mod tests {
    #[test]
    fn test_library_compiles() {
        // Smoke test to ensure the library compiles
        assert!(true);
    }
}
