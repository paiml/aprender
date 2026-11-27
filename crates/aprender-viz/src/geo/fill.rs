//! Even-odd polygon fill with holes, into a framebuffer and as SVG path data.
//!
//! A polygon is a list of rings in pixel coordinates. A point is inside when a ray from it
//! crosses an odd number of ring edges, so a ring inside another ring is a hole. The
//! framebuffer fill is a scanline pass that samples each pixel at its centre `(x + 0.5,
//! y + 0.5)`. The SVG form is the same rings as closed subpaths for `fill-rule="evenodd"`.
//!
//! Each ring is closed implicitly from its last vertex back to its first, so a ring may repeat
//! its first vertex at the end without changing the result.

use crate::color::Rgba;
use crate::format_coord;
use crate::framebuffer::Framebuffer;
use std::fmt::Write as FmtWrite;
use std::ops::Range;

/// Why a set of rings was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum FillError {
    /// A vertex has a NaN or infinite coordinate.
    #[error("ring {ring} vertex {vertex} is not finite")]
    NonFinite {
        /// Index of the ring.
        ring: usize,
        /// Index of the vertex within the ring.
        vertex: usize,
    },
    /// A ring has fewer than three vertices, so it encloses no area.
    #[error("ring {ring} has {vertices} vertices; a ring needs at least 3")]
    DegenerateRing {
        /// Index of the ring.
        ring: usize,
        /// Number of vertices the ring has.
        vertices: usize,
    },
    /// An edge crossing overflowed to a non-finite x, which happens only when an edge spans
    /// more than `f64::MAX`.
    #[error("ring {ring}: an edge crossing is not finite")]
    NonFiniteResult {
        /// Index of the ring that owns the edge.
        ring: usize,
    },
}

/// Fill the even-odd region of `rings` in `fb` with `color`.
///
/// Pixels outside the framebuffer are clipped. Input is validated before the first pixel is
/// written. An edge crossing can still overflow while filling, and then the rows painted before
/// that row remain painted.
///
/// # Errors
///
/// Returns [`FillError`] if any ring has fewer than three vertices, any coordinate is not
/// finite, or an edge crossing overflows.
pub fn fill_polygon_even_odd(
    fb: &mut Framebuffer,
    rings: &[Vec<(f64, f64)>],
    color: Rgba,
) -> Result<(), FillError> {
    validate(rings)?;
    let Some((ymin, ymax)) = y_extent(rings) else {
        return Ok(());
    };
    for py in pixel_range(ymin, ymax, fb.height()) {
        let sy = f64::from(py) + 0.5;
        let xs = crossings(rings, sy)?;
        for pair in xs.chunks_exact(2) {
            fill_span(fb, py, pair[0], pair[1], color);
        }
    }
    Ok(())
}

/// SVG path data for `rings`: one closed subpath per ring, subpaths separated by a space.
///
/// Use it with `SvgEncoder::path_even_odd`, which emits `fill-rule="evenodd"`. Numbers are
/// written with [`format_coord`], so the same rings always produce the same bytes.
///
/// # Errors
///
/// Returns [`FillError`] under the same conditions as [`fill_polygon_even_odd`], except that
/// no edge crossing is computed, so [`FillError::NonFiniteResult`] is never returned.
pub fn svg_path_d(rings: &[Vec<(f64, f64)>]) -> Result<String, FillError> {
    validate(rings)?;
    let subpaths: Vec<String> = rings.iter().map(|ring| subpath(ring)).collect();
    Ok(subpaths.join(" "))
}

/// Refuse a ring with fewer than three vertices or a non-finite coordinate.
fn validate(rings: &[Vec<(f64, f64)>]) -> Result<(), FillError> {
    for (ring, points) in rings.iter().enumerate() {
        if points.len() < 3 {
            return Err(FillError::DegenerateRing { ring, vertices: points.len() });
        }
        for (vertex, &(x, y)) in points.iter().enumerate() {
            if !x.is_finite() || !y.is_finite() {
                return Err(FillError::NonFinite { ring, vertex });
            }
        }
    }
    Ok(())
}

/// The lowest and highest y over every vertex, or `None` when there are no rings.
fn y_extent(rings: &[Vec<(f64, f64)>]) -> Option<(f64, f64)> {
    let mut ys = rings.iter().flatten().map(|&(_, y)| y).peekable();
    ys.peek()?;
    Some(ys.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), y| (lo.min(y), hi.max(y))))
}

/// The pixel indices `p` whose centre `p + 0.5` lies in `[lo, hi)`, clipped to `0..limit`.
fn pixel_range(lo: f64, hi: f64, limit: u32) -> Range<u32> {
    let limit = f64::from(limit);
    let start = (lo - 0.5).ceil().clamp(0.0, limit);
    let end = (hi - 0.5).ceil().clamp(0.0, limit);
    (start as u32)..(end as u32)
}

/// Every edge of a ring as a `(from, to)` pair, including the closing edge.
fn edges(ring: &[(f64, f64)]) -> impl Iterator<Item = ((f64, f64), (f64, f64))> + '_ {
    ring.iter().copied().zip(ring.iter().copied().cycle().skip(1))
}

/// The sorted x values where the horizontal line `y = sy` crosses an edge of any ring.
///
/// An edge counts when one endpoint is at or below `sy` and the other is strictly above it.
/// That half-open rule counts a vertex on the line once, so the crossings always pair up.
fn crossings(rings: &[Vec<(f64, f64)>], sy: f64) -> Result<Vec<f64>, FillError> {
    let mut xs = Vec::new();
    for (ring_index, ring) in rings.iter().enumerate() {
        for ((ax, ay), (bx, by)) in edges(ring) {
            if (ay <= sy) != (by <= sy) {
                let t = (sy - ay) / (by - ay);
                let x = ax + t * (bx - ax);
                if !x.is_finite() {
                    return Err(FillError::NonFiniteResult { ring: ring_index });
                }
                xs.push(x);
            }
        }
    }
    xs.sort_by(f64::total_cmp);
    Ok(xs)
}

/// Paint the pixels of row `py` whose centres lie in `[x0, x1)`.
fn fill_span(fb: &mut Framebuffer, py: u32, x0: f64, x1: f64, color: Rgba) {
    for px in pixel_range(x0, x1, fb.width()) {
        fb.set_pixel(px, py, color);
    }
}

/// One closed SVG subpath: `M` to the first vertex, `L` to each later one, then `Z`.
fn subpath(ring: &[(f64, f64)]) -> String {
    let mut d = String::new();
    for (i, &(x, y)) in ring.iter().enumerate() {
        let command = if i == 0 { 'M' } else { 'L' };
        let _ = write!(d, "{command}{} {}", format_coord(x), format_coord(y));
    }
    d.push('Z');
    d
}
