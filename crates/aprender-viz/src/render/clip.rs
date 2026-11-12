//! Segment clipping shared by the line rasterisers.
//!
//! The rasterisers walk one pixel per step along the major axis, so a segment whose endpoints
//! are far off the canvas (a finite point at `1e38` under `xlim(0, 1)`) would walk billions of
//! pixels that are never visible. A segment is therefore clipped, in `f64`, to the canvas grown by
//! [`CLIP_GUARD`] pixels on every side (Liang-Barsky) before any loop runs. A segment that
//! already lies inside that box is returned untouched, bit for bit, so output for ordinary data
//! is the same as without clipping.
//!
//! A segment that crosses the guard is not bit-exact: the cut end is rounded to an integer (so
//! Bresenham restarts its error term) or the f32 walk restarts at the cut (Wu). Measured against
//! the unclipped line, the drawn pixels move by at most 2 px (aliased) and 3 px (antialiased);
//! contract `render-total-v1`, RT-DEV-001, bounded by
//! `guard_crossing_segments_stay_within_the_stated_bound`. Only data with an end more than
//! 1024 px off the canvas is affected.

use crate::framebuffer::Framebuffer;

/// Margin, in pixels, around the canvas inside which a segment is drawn exactly as given.
pub(crate) const CLIP_GUARD: f64 = 1024.0;

/// A segment `(x0, y0, x1, y1)`.
pub(crate) type Segment = (f64, f64, f64, f64);

/// The canvas grown by [`CLIP_GUARD`]: `(xmin, ymin, xmax, ymax)`.
fn guard_box(fb: &Framebuffer) -> Segment {
    (
        -CLIP_GUARD,
        -CLIP_GUARD,
        f64::from(fb.width()) + CLIP_GUARD,
        f64::from(fb.height()) + CLIP_GUARD,
    )
}

fn point_in(x: f64, y: f64, (xmin, ymin, xmax, ymax): Segment) -> bool {
    x >= xmin && x <= xmax && y >= ymin && y <= ymax
}

/// One Liang-Barsky edge: narrow the parameter interval `(t0, t1)` by the edge `p * t <= q`, or
/// `None` when nothing of the segment is left.
fn clip_edge(p: f64, q: f64, (t0, t1): (f64, f64)) -> Option<(f64, f64)> {
    if p == 0.0 {
        // Parallel to the edge: wholly outside it, or untouched by it.
        return (q >= 0.0).then_some((t0, t1));
    }
    let r = q / p;
    if p < 0.0 {
        (r <= t1).then(|| (t0.max(r), t1))
    } else {
        (r >= t0).then(|| (t0, t1.min(r)))
    }
}

/// Liang-Barsky: the part of `s` inside `bounds`, or `None` when none of it is.
///
/// Every input must be finite; the caller checks.
pub(crate) fn clip_segment(s: Segment, bounds: Segment) -> Option<Segment> {
    let (x0, y0, x1, y1) = s;
    let (xmin, ymin, xmax, ymax) = bounds;
    let (dx, dy) = (x1 - x0, y1 - y0);
    let mut t = (0.0_f64, 1.0_f64);
    for (p, q) in [(-dx, x0 - xmin), (dx, xmax - x0), (-dy, y0 - ymin), (dy, ymax - y0)] {
        t = clip_edge(p, q, t)?;
    }
    let (t0, t1) = t;
    // The interpolation can land a few ulps (of the *input* magnitude) outside the box when an
    // endpoint is huge, so the result is clamped back into it.
    let (px, py) =
        (|t: f64| (x0 + t * dx).clamp(xmin, xmax), |t: f64| (y0 + t * dy).clamp(ymin, ymax));
    Some((px(t0), py(t0), px(t1), py(t1)))
}

/// The part of `s` worth drawing on `fb`, or `None` when `s` has a non-finite coordinate or lies
/// wholly outside the canvas plus [`CLIP_GUARD`].
///
/// A segment with both endpoints inside the guard box is returned unchanged.
pub(crate) fn canvas_segment(fb: &Framebuffer, s: Segment) -> Option<Segment> {
    let (x0, y0, x1, y1) = s;
    if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
        return None;
    }
    let bounds = guard_box(fb);
    if point_in(x0, y0, bounds) && point_in(x1, y1, bounds) {
        return Some(s);
    }
    clip_segment(s, bounds)
}

#[cfg(test)]
mod tests {
    use super::*;

    const B: Segment = (0.0, 0.0, 10.0, 10.0);

    #[test]
    fn inside_segment_is_unchanged() {
        assert_eq!(clip_segment((1.0, 2.0, 8.0, 9.0), B), Some((1.0, 2.0, 8.0, 9.0)));
    }

    #[test]
    fn crossing_segment_is_cut_at_the_box() {
        let c = clip_segment((-10.0, 5.0, 20.0, 5.0), B);
        assert_eq!(c, Some((0.0, 5.0, 10.0, 5.0)));
        let d = clip_segment((5.0, -10.0, 5.0, 20.0), B);
        assert_eq!(d, Some((5.0, 0.0, 5.0, 10.0)));
    }

    #[test]
    fn outside_segments_vanish() {
        assert_eq!(clip_segment((-5.0, -5.0, -1.0, 20.0), B), None);
        assert_eq!(clip_segment((20.0, 0.0, 30.0, 10.0), B), None);
        // Parallel to an edge and outside it.
        assert_eq!(clip_segment((0.0, 11.0, 10.0, 11.0), B), None);
        assert_eq!(clip_segment((11.0, 0.0, 11.0, 10.0), B), None);
        // Diagonal that misses the corner.
        assert_eq!(clip_segment((-5.0, 8.0, 3.0, 16.0), B), None);
    }

    #[test]
    fn degenerate_point_segment() {
        assert_eq!(clip_segment((5.0, 5.0, 5.0, 5.0), B), Some((5.0, 5.0, 5.0, 5.0)));
        assert_eq!(clip_segment((50.0, 5.0, 50.0, 5.0), B), None);
    }

    #[test]
    fn extreme_segment_clips_to_finite_ends() {
        let c = clip_segment((-1e38, -1e38, 1e38, 1e38), B).expect("diagonal crosses the box");
        for v in [c.0, c.1, c.2, c.3] {
            assert!(v.is_finite() && (-1e-6..=10.0 + 1e-6).contains(&v), "{c:?}");
        }
    }

    #[test]
    fn canvas_segment_refuses_non_finite() {
        let fb = Framebuffer::new(8, 8).expect("fb");
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(canvas_segment(&fb, (bad, 0.0, 1.0, 1.0)), None);
            assert_eq!(canvas_segment(&fb, (0.0, 0.0, 1.0, bad)), None);
        }
    }

    #[test]
    fn canvas_segment_passes_inside_and_cuts_outside() {
        let fb = Framebuffer::new(8, 8).expect("fb");
        let s = (1.5, 2.5, 7.25, 3.0);
        assert_eq!(canvas_segment(&fb, s), Some(s));
        let far = canvas_segment(&fb, (4.0, 4.0, 1e30, 4.0)).expect("starts inside");
        assert!((far.2 - (8.0 + CLIP_GUARD)).abs() < 1e-6, "{far:?}");
        assert_eq!(canvas_segment(&fb, (1e30, 1e30, 2e30, 3e30)), None);
    }
}
