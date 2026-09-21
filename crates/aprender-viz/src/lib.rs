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
// APEX-001 EV-2a rule 5. The ban list lives in `crates/aprender-viz/.clippy.toml`, NOT the
// repository root: clippy reads exactly one config, the nearest, so a root entry is read for
// crates that have no config of their own and is ignored here (measured). This deny is what
// turns that list from advice into a build failure — without it the lints are warnings and
// `cargo clippy` exits 0, which is also measured.
#![deny(clippy::disallowed_methods)]
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
pub mod breaks;
pub mod color;
/// Core framebuffer for pixel rendering.
pub mod framebuffer;
/// Geometric primitives (points, lines, rectangles).
pub mod geometry;
/// Content manifest: per-file digests reduced to one root hash (APEX-001 EV-2a rule 4).
pub mod manifest;
/// Scale functions for data-to-visual mappings.
pub mod scale;
/// Text as glyph outlines, from caller-pinned font bytes (APEX-001 EV-2a rule 1).
#[cfg(feature = "text-path")]
pub mod text;
/// Rasterisation through a pinned, font-free `resvg` (APEX-001 EV-2a rule 3, EV-2d).
#[cfg(feature = "raster")]
pub mod raster;

/// The coordinate grid emitted geometry is snapped to, in user units.
///
/// APEX-001 EV-2a rule 5. Deterministic transcendentals stop the *inputs* to layout from
/// differing between hosts; this stops a difference that survives anyway from reaching the
/// bytes. One thousandth of a user unit is far below a device pixel at any plausible scale, so
/// snapping is invisible in the image and decisive in the file.
pub const COORD_GRID: f64 = 1e-3;

/// Snap one coordinate to [`COORD_GRID`].
///
/// Ties round half away from zero, and `-0.0` is normalised to `0.0` so the two zeros cannot
/// print differently.
#[must_use]
pub fn quantise(v: f64) -> f64 {
    if !v.is_finite() {
        return v;
    }
    let snapped = (v / COORD_GRID).round() * COORD_GRID;
    if snapped == 0.0 {
        0.0
    } else {
        snapped
    }
}

/// Quantise every number in an SVG path `d` string.
///
/// Operates on the serialised form because that is the last point before the bytes are fixed:
/// whatever produced the path, what reaches the file is on the grid.
#[must_use]
pub fn quantise_path_data(d: &str) -> String {
    let mut out = String::with_capacity(d.len());
    let mut num = String::new();
    for c in d.chars() {
        if c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+' {
            num.push(c);
        } else {
            flush_number(&mut num, &mut out);
            out.push(c);
        }
    }
    flush_number(&mut num, &mut out);
    out
}

/// Quantise one coordinate and print its shortest form on the grid.
///
/// Every number the SVG writer emits goes through here, so what reaches the file is a function
/// of the grid cell rather than of the last few bits of a float: `1.0000004` and `1.0000001`
/// both print `1`, and `-0.0` prints `0`. A non-finite value prints as Rust formats it, which is
/// never a valid coordinate and is left visible rather than silently replaced.
#[must_use]
pub fn format_coord(v: f64) -> String {
    let q = quantise(v);
    if !q.is_finite() {
        return format!("{q}");
    }
    if (q - q.round()).abs() < f64::EPSILON {
        format!("{}", q.round() as i64)
    } else {
        format!("{q:.3}").trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Parse one accumulated number, quantise it, and append its shortest form.
fn flush_number(num: &mut String, out: &mut String) {
    if num.is_empty() {
        return;
    }
    match num.parse::<f64>() {
        Ok(v) => out.push_str(&format_coord(v)),
        Err(_) => out.push_str(num),
    }
    num.clear();
}
#[cfg(test)]
mod coord_grid_tests {
    use super::{format_coord, quantise, quantise_path_data, COORD_GRID};

    #[test]
    fn format_coord_prints_the_shortest_form_on_the_grid() {
        let cases = [
            (0.0, "0"),
            (-0.0, "0"),
            (1.0, "1"),
            (1.000_000_4, "1"),
            (0.999_999_6, "1"),
            (2.5, "2.5"),
            (1.234_56, "1.235"),
            (0.000_5, "0.001"),
            (-0.000_5, "-0.001"),
            (0.000_4, "0"),
            (-0.000_4, "0"),
            (312.0, "312"),
            (-7.123_456, "-7.123"),
            (1e6 + 0.25, "1000000.25"),
        ];
        for (v, want) in cases {
            assert_eq!(format_coord(v), want, "format_coord({v})");
        }
        assert_eq!(format_coord(f64::NAN), "NaN");
        assert_eq!(format_coord(f64::INFINITY), "inf");
    }

    #[test]
    fn quantise_is_idempotent_and_on_the_grid() {
        for v in [0.0, 0.1, 0.123_456, -3.999_9, 1e-9, 12_345.678_9] {
            let q = quantise(v);
            assert_eq!(quantise(q), q, "quantise({v}) is not a fixed point");
            let cells = q / COORD_GRID;
            assert!((cells - cells.round()).abs() < 1e-6, "quantise({v}) = {q} is off the grid");
        }
    }

    #[test]
    fn path_data_numbers_are_quantised_and_commands_are_kept() {
        assert_eq!(quantise_path_data("M 1.00004 2.0006 L -0.0 3 Z"), "M 1 2.001 L 0 3 Z");
        assert_eq!(
            quantise_path_data("M1.5,2.5C3.33333,4.44444 5,6 7.7777,8.8888Z"),
            "M1.5,2.5C3.333,4.444 5,6 7.778,8.889Z"
        );
    }
}

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
