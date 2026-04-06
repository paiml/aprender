//! Rendering backends and rasterization.
//!
//! Provides SIMD-accelerated rasterization algorithms for geometric primitives.
//!
//! # Algorithms
//!
//! - **Wu's Anti-aliased Line**: Smooth line rendering with sub-pixel accuracy
//! - **Bresenham's Line**: Fast non-antialiased line drawing
//! - **Midpoint Circle**: Filled and outlined circle rendering
//!
//! # References
//!
//! - Wu, X. (1991). "An Efficient Antialiasing Technique." SIGGRAPH '91.
//! - Bresenham, J. E. (1965). "Algorithm for computer control of a digital plotter."

mod primitives;

pub use primitives::{
    draw_circle, draw_circle_outline, draw_line, draw_line_aa, draw_point, draw_rect,
    draw_rect_outline, i32_px, Drawable,
};
