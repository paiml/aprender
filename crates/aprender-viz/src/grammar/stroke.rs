//! Strokes: what a path-family or reference layer puts on a panel (PLOT-GG-001,
//! paiml/aprender#6909).
//!
//! `geom_line`, `geom_path`, `geom_step`, `geom_hline`, `geom_vline` and `geom_abline` are all
//! drawn the same way by every render target: as polylines in data space, clipped to the visible
//! domain. The layout ([`BuiltLayer`]) already holds the ordered, grouped, per-panel vertices
//! that `ggplot_build()` produces (proven against R in `ggplot2_lines_oracle.rs`); this module
//! only turns them into segments. The terminal, the raster and the SVG targets then map those
//! segments to cells or pixels, so there is one place where "what is drawn" is decided.
//!
//! Rules, each pinned by a test:
//!
//! * A vertex with a `NaN` coordinate ends the run: nothing is drawn to or from it.
//! * A run also ends where the group (or the panel) changes.
//! * A segment is clipped to the visible domain (Liang-Barsky), so a vertex a million times
//!   outside the panel costs one clip, never a long raster loop. A segment wholly outside is
//!   dropped.
//! * A reference line spans the visible domain: `geom_hline` and `geom_vline` the whole panel,
//!   `geom_abline` from the left edge to the right edge at `slope * x + intercept`, clipped.
//! * A domain that is not finite draws nothing.

use crate::color::Rgba;
use crate::framebuffer::Framebuffer;
use crate::output::SvgEncoder;
use crate::render::draw_line_aa;
use crate::scale::{LinearScale, Scale};

use super::build::BuiltLayer;
use super::path::{BuiltPath, ReferenceKind};

/// Pixels per millimetre of `linewidth`: ggplot2's `.pt` (72.27 / 25.4), which turns a
/// `linewidth` into R's `lwd`; one `lwd` unit is one pixel on a 96 dpi screen.
pub const LINEWIDTH_PX_PER_MM: f64 = 72.27 / 25.4;

/// A visible domain: `(x_lo, x_hi)` and `(y_lo, y_hi)`.
pub type Domain = ((f64, f64), (f64, f64));

/// Is this layer drawn as strokes?
pub(super) fn is_stroke(layer: &BuiltLayer) -> bool {
    layer.path.is_some()
}

/// The stroke colour of a layer, with its alpha applied.
pub(super) fn colour(path: &BuiltPath) -> Rgba {
    let c = path.aes.colour;
    match path.aes.alpha {
        Some(a) if a.is_finite() => {
            Rgba::new(c.r, c.g, c.b, (a.clamp(0.0, 1.0) * 255.0).round() as u8)
        }
        _ => c,
    }
}

/// The stroke width of a layer in pixels.
pub(super) fn width_px(path: &BuiltPath) -> f64 {
    path.aes.linewidth * LINEWIDTH_PX_PER_MM
}

/// The polylines of a stroke layer on `panel` (0-based), clipped to `domain`. Empty for a layer
/// that is not a stroke layer, for a panel the layer has no rows in, and for a domain that is
/// not finite.
pub(super) fn polylines(layer: &BuiltLayer, panel: usize, domain: Domain) -> Vec<Vec<(f64, f64)>> {
    let Some(path) = &layer.path else { return Vec::new() };
    let ((x0, x1), (y0, y1)) = domain;
    if ![x0, x1, y0, y1].iter().all(|v| v.is_finite()) {
        return Vec::new();
    }
    let domain = ((x0.min(x1), x0.max(x1)), (y0.min(y1), y0.max(y1)));
    let runs = match (&path.reference, &path.stairs) {
        (Some(_), _) => reference_runs(path, panel, domain),
        (None, Some(s)) => split_runs(&s.x, &s.y, &s.panel, &s.group, panel),
        (None, None) => split_runs(&layer.x, &layer.y, &path.panel, &path.group, panel),
    };
    runs.iter().flat_map(|r| clip_run(r, domain)).collect()
}

/// The vertices of the single segment a reference line is, across the visible domain.
fn reference_runs(
    path: &BuiltPath,
    panel: usize,
    ((xlo, xhi), (ylo, yhi)): Domain,
) -> Vec<Vec<(f64, f64)>> {
    let Some(r) = &path.reference else { return Vec::new() };
    let Some(&c) = r.intercept.get(panel) else { return Vec::new() };
    let run = match r.kind {
        ReferenceKind::Horizontal => vec![(xlo, c), (xhi, c)],
        ReferenceKind::Vertical => vec![(c, ylo), (c, yhi)],
        ReferenceKind::Diagonal => {
            let s = r.slope.get(panel).copied().unwrap_or(f64::NAN);
            vec![(xlo, s.mul_add(xlo, c)), (xhi, s.mul_add(xhi, c))]
        }
    };
    vec![run]
}

/// Split the rows of panel `want` into runs of consecutive finite vertices of one group. Rows are
/// stable-sorted by group first, as `GeomPath$draw_panel` does (`data[order(data$group), ]`), so a
/// group whose rows are interleaved with another's in the data is still one polyline, and
/// polylines come out in ascending group order. A row of another panel is not part of the panel's
/// sequence: it neither joins nor breaks a run.
fn split_runs(
    x: &[f64],
    y: &[f64],
    panel: &[usize],
    group: &[i64],
    want: usize,
) -> Vec<Vec<(f64, f64)>> {
    let mut rows: Vec<usize> =
        (0..x.len().min(y.len())).filter(|&i| panel.get(i) == Some(&want)).collect();
    rows.sort_by_key(|&i| group.get(i).copied());
    let mut runs: Vec<Vec<(f64, f64)>> = Vec::new();
    let mut cur: Vec<(f64, f64)> = Vec::new();
    let mut last_group = None;
    for i in rows {
        let (xv, yv, g) = (x[i], y[i], group.get(i).copied());
        let usable = xv.is_finite() && yv.is_finite();
        if (!usable || last_group != g) && !cur.is_empty() {
            runs.push(std::mem::take(&mut cur));
        }
        if usable {
            last_group = g;
            cur.push((xv, yv));
        }
    }
    if !cur.is_empty() {
        runs.push(cur);
    }
    runs
}

/// Clip every segment of a run to the domain; segments that stay joined stay one polyline.
fn clip_run(run: &[(f64, f64)], domain: Domain) -> Vec<Vec<(f64, f64)>> {
    let mut out: Vec<Vec<(f64, f64)>> = Vec::new();
    for w in run.windows(2) {
        let Some((a, b)) = clip(w[0], w[1], domain) else { continue };
        match out.last_mut() {
            Some(cur) if cur.last() == Some(&a) => cur.push(b),
            _ => out.push(vec![a, b]),
        }
    }
    out
}

/// Liang-Barsky: the part of segment `a`-`b` inside the closed domain, or `None`.
pub(super) fn clip(
    a: (f64, f64),
    b: (f64, f64),
    ((xlo, xhi), (ylo, yhi)): Domain,
) -> Option<((f64, f64), (f64, f64))> {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    if !(dx.is_finite() && dy.is_finite()) {
        return None;
    }
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for (p, q) in [(-dx, a.0 - xlo), (dx, xhi - a.0), (-dy, a.1 - ylo), (dy, yhi - a.1)] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    if t0 > t1 {
        return None;
    }
    let at = |t: f64| {
        if t <= 0.0 {
            a
        } else if t >= 1.0 {
            b
        } else {
            ((a.0 + t * dx).clamp(xlo, xhi), (a.1 + t * dy).clamp(ylo, yhi))
        }
    };
    Some((at(t0), at(t1)))
}

/// The cells a segment crosses between two cells (Bresenham), both ends included.
pub(super) fn cells_between(a: (usize, usize), b: (usize, usize)) -> Vec<(usize, usize)> {
    let (mut x, mut y) = (a.0 as i64, a.1 as i64);
    let (x1, y1) = (b.0 as i64, b.1 as i64);
    let (dx, dy) = ((x1 - x).abs(), -(y1 - y).abs());
    let (sx, sy) = (if x < x1 { 1 } else { -1 }, if y < y1 { 1 } else { -1 });
    let mut err = dx + dy;
    let mut out = Vec::new();
    loop {
        out.push((x as usize, y as usize));
        if x == x1 && y == y1 {
            return out;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

/// Draw a stroke layer's polylines into the framebuffer, anti-aliased, one pixel wide.
pub(super) fn draw_raster(
    fb: &mut Framebuffer,
    layer: &BuiltLayer,
    panel: usize,
    domain: Domain,
    (xs, ys): (&LinearScale, &LinearScale),
) {
    let Some(path) = &layer.path else { return };
    let colour = colour(path);
    for line in polylines(layer, panel, domain) {
        for w in line.windows(2) {
            let (Some(a), Some(b)) = (px(w[0], xs, ys), px(w[1], xs, ys)) else { continue };
            draw_line_aa(fb, a.0, a.1, b.0, b.1, colour);
        }
    }
}

/// Add a stroke layer's polylines to an SVG as `<polyline>` elements at pixel coordinates.
pub(super) fn add_svg(
    mut svg: SvgEncoder,
    layer: &BuiltLayer,
    panel: usize,
    domain: Domain,
    (xs, ys): (&LinearScale, &LinearScale),
) -> SvgEncoder {
    let Some(path) = &layer.path else { return svg };
    let (colour, width) = (colour(path), width_px(path) as f32);
    for line in polylines(layer, panel, domain) {
        // A vertex that maps to no finite pixel (a canvas with no room) drops its polyline: an
        // attribute is never NaN or infinite.
        let Some(points) = line.iter().map(|&p| px(p, xs, ys)).collect::<Option<Vec<_>>>() else {
            continue;
        };
        svg = svg.polyline(&points, colour, width);
    }
    svg
}

/// A data-space point in pixels; `None` when either coordinate maps to no finite pixel.
fn px((x, y): (f64, f64), xs: &LinearScale, ys: &LinearScale) -> Option<(f32, f32)> {
    let (a, b) = (xs.scale(x as f32), ys.scale(y as f32));
    (a.is_finite() && b.is_finite()).then_some((a, b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::path::{Reference, StrokeAes};

    const DOM: Domain = ((0.0, 10.0), (0.0, 10.0));

    fn aes(alpha: Option<f64>) -> StrokeAes {
        StrokeAes { colour: Rgba::new(10, 20, 30, 255), linewidth: 0.5, linetype: 1, alpha }
    }

    fn path(reference: Option<Reference>, n: usize, alpha: Option<f64>) -> BuiltPath {
        BuiltPath {
            panel: vec![0; n],
            group: vec![-1; n],
            aes: aes(alpha),
            stairs: None,
            reference,
        }
    }

    fn layer(x: &[f64], y: &[f64], p: Option<BuiltPath>) -> BuiltLayer {
        BuiltLayer { x: x.to_vec(), y: y.to_vec(), points: None, path: p }
    }

    #[test]
    fn colour_applies_alpha_and_ignores_a_non_finite_one() {
        assert_eq!(colour(&path(None, 0, Some(0.5))).a, 128);
        assert_eq!(colour(&path(None, 0, Some(7.0))).a, 255);
        assert_eq!(colour(&path(None, 0, Some(-1.0))).a, 0);
        assert_eq!(colour(&path(None, 0, Some(f64::NAN))).a, 255);
        assert_eq!(colour(&path(None, 0, None)), Rgba::new(10, 20, 30, 255));
        assert!((width_px(&path(None, 0, None)) - 0.5 * LINEWIDTH_PX_PER_MM).abs() < 1e-12);
    }

    #[test]
    fn polylines_of_a_non_stroke_layer_or_a_bad_domain_are_empty() {
        let plain = layer(&[1.0, 2.0], &[1.0, 2.0], None);
        assert!(!is_stroke(&plain));
        assert!(polylines(&plain, 0, DOM).is_empty());
        let l = layer(&[1.0, 2.0], &[1.0, 2.0], Some(path(None, 2, None)));
        assert!(is_stroke(&l));
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(polylines(&l, 0, ((0.0, bad), (0.0, 10.0))).is_empty());
            assert!(polylines(&l, 0, ((0.0, 10.0), (bad, 10.0))).is_empty());
        }
        // A reversed domain is the same domain; another panel has no rows.
        assert_eq!(polylines(&l, 0, ((10.0, 0.0), (10.0, 0.0))), polylines(&l, 0, DOM));
        assert!(polylines(&l, 1, DOM).is_empty());
    }

    #[test]
    fn runs_break_at_nan_group_and_panel() {
        let n = f64::NAN;
        let x = [0.0, 1.0, n, 3.0, 4.0, 5.0, 6.0];
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let panel = [0, 0, 0, 0, 0, 1, 0];
        let group = [1, 1, 1, 1, 2, 2, 2];
        let runs = split_runs(&x, &y, &panel, &group, 0);
        assert_eq!(
            runs,
            vec![
                vec![(0.0, 0.0), (1.0, 1.0)],
                vec![(3.0, 3.0)],
                // Row 5 is panel 1: it neither joins nor breaks panel 0's group 2.
                vec![(4.0, 4.0), (6.0, 6.0)]
            ]
        );
        // Short panel/group slices are not a panic: no panel, no rows; no group, one group.
        assert!(split_runs(&x, &y, &[], &group, 0).is_empty());
        assert_eq!(
            split_runs(&x, &y, &panel, &[], 0),
            vec![vec![(0.0, 0.0), (1.0, 1.0)], vec![(3.0, 3.0), (4.0, 4.0), (6.0, 6.0)]]
        );
    }

    #[test]
    fn interleaved_groups_are_one_polyline_each_in_group_order() {
        // R: data[order(data$group), ] before drawing (ggplot2 4.0.3 GeomPath).
        let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let panel = [0; 6];
        let group = [3, 1, 2, 3, 1, 2];
        let runs = split_runs(&x, &y, &panel, &group, 0);
        assert_eq!(
            runs,
            vec![
                vec![(1.0, 1.0), (4.0, 4.0)],
                vec![(2.0, 2.0), (5.0, 5.0)],
                vec![(0.0, 0.0), (3.0, 3.0)]
            ]
        );
        // A gap inside one group still breaks it, in sorted order.
        let y = [0.0, 1.0, 2.0, 3.0, f64::NAN, 5.0];
        let runs = split_runs(&x, &y, &panel, &group, 0);
        assert_eq!(runs[0], vec![(1.0, 1.0)]);
    }

    #[test]
    fn split_runs_keeps_data_order_inside_each_group() {
        // R's order() is stable: inside a group the vertices stay in data order. 200 rows in
        // 3 interleaved groups is far past the size below which an unstable sort happens to
        // behave stably, so sort_unstable_by_key turns this RED.
        let n = 200;
        let x: Vec<f64> = (0..n).map(f64::from).collect();
        let y = x.clone();
        let panel = vec![0; n as usize];
        let group: Vec<i64> = (0..n).map(|i| i64::from((i * 7 + i / 5) % 3)).collect();
        let runs = split_runs(&x, &y, &panel, &group, 0);
        assert_eq!(runs.len(), 3);
        for (g, run) in runs.iter().enumerate() {
            let want: Vec<(f64, f64)> = (0..n)
                .filter(|&i| i64::from((i * 7 + i / 5) % 3) == g as i64)
                .map(|i| (f64::from(i), f64::from(i)))
                .collect();
            assert_eq!(run, &want, "group {g}: vertex order is not data order");
        }
    }

    #[test]
    fn reference_runs_span_the_domain() {
        let r = |kind, intercept: &[f64], slope: &[f64]| {
            Some(Reference { kind, intercept: intercept.to_vec(), slope: slope.to_vec() })
        };
        let h = path(r(ReferenceKind::Horizontal, &[3.0], &[]), 0, None);
        assert_eq!(reference_runs(&h, 0, DOM), vec![vec![(0.0, 3.0), (10.0, 3.0)]]);
        let v = path(r(ReferenceKind::Vertical, &[4.0], &[]), 0, None);
        assert_eq!(reference_runs(&v, 0, DOM), vec![vec![(4.0, 0.0), (4.0, 10.0)]]);
        let d = path(r(ReferenceKind::Diagonal, &[1.0], &[2.0]), 0, None);
        assert_eq!(reference_runs(&d, 0, DOM), vec![vec![(0.0, 1.0), (10.0, 21.0)]]);
        // No panel entry, no reference, or a missing slope: nothing to draw.
        assert!(reference_runs(&h, 1, DOM).is_empty());
        assert!(reference_runs(&path(None, 0, None), 0, DOM).is_empty());
        let no_slope = path(r(ReferenceKind::Diagonal, &[1.0], &[]), 0, None);
        let l = layer(&[], &[], Some(no_slope));
        assert!(polylines(&l, 0, DOM).is_empty());
        // And through polylines the diagonal is clipped to the domain.
        let l = layer(&[], &[], Some(d));
        let lines = polylines(&l, 0, DOM);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].first(), Some(&(0.0, 1.0)));
        assert!((lines[0][1].1 - 10.0).abs() < 1e-9 && (lines[0][1].0 - 4.5).abs() < 1e-9);
    }

    #[test]
    fn clip_keeps_the_inside_part_and_drops_the_rest() {
        assert_eq!(clip((1.0, 1.0), (2.0, 2.0), DOM), Some(((1.0, 1.0), (2.0, 2.0))));
        assert_eq!(clip((-5.0, 5.0), (15.0, 5.0), DOM), Some(((0.0, 5.0), (10.0, 5.0))));
        assert_eq!(clip((5.0, -5.0), (5.0, 15.0), DOM), Some(((5.0, 0.0), (5.0, 10.0))));
        // Wholly outside, parallel to an edge and outside, and a diagonal that misses the corner.
        assert_eq!(clip((20.0, 0.0), (20.0, 5.0), DOM), None);
        assert_eq!(clip((0.0, 20.0), (5.0, 20.0), DOM), None);
        assert_eq!(clip((-5.0, 8.0), (2.0, 15.0), DOM), None);
        // A segment whose extent overflows is dropped, not looped over.
        assert_eq!(clip((-f64::MAX, 0.0), (f64::MAX, 0.0), DOM), None);
        // A point on the boundary is kept.
        assert_eq!(clip((10.0, 10.0), (10.0, 10.0), DOM), Some(((10.0, 10.0), (10.0, 10.0))));
    }

    #[test]
    fn clipped_runs_stay_joined_only_where_they_touch() {
        let run = [(-5.0, 5.0), (5.0, 5.0), (5.0, 20.0), (-5.0, 20.0)];
        let out = clip_run(&run, DOM);
        assert_eq!(out, vec![vec![(0.0, 5.0), (5.0, 5.0), (5.0, 10.0)]]);
    }

    #[test]
    fn cells_between_is_bresenham() {
        assert_eq!(cells_between((0, 0), (0, 0)), vec![(0, 0)]);
        assert_eq!(cells_between((0, 0), (3, 0)), vec![(0, 0), (1, 0), (2, 0), (3, 0)]);
        assert_eq!(cells_between((2, 0), (0, 0)), vec![(2, 0), (1, 0), (0, 0)]);
        assert_eq!(cells_between((0, 0), (0, 2)), vec![(0, 0), (0, 1), (0, 2)]);
        assert_eq!(cells_between((0, 3), (3, 0)), vec![(0, 3), (1, 2), (2, 1), (3, 0)]);
        let c = cells_between((0, 0), (6, 2));
        assert_eq!((c.first(), c.last(), c.len()), (Some(&(0, 0)), Some(&(6, 2)), 7));
        // Ties between the two error terms (2 * err == dy or == dx): both steps are taken.
        assert_eq!(cells_between((0, 0), (1, 2)), vec![(0, 0), (1, 1), (1, 2)]);
        assert_eq!(cells_between((0, 0), (2, 1)), vec![(0, 0), (1, 1), (2, 1)]);
        assert_eq!(cells_between((1, 2), (0, 0)), vec![(1, 2), (0, 1), (0, 0)]);
    }

    #[test]
    fn an_unmappable_vertex_draws_nothing_and_writes_no_non_finite_attribute() {
        // An infinite range maps 0 to NaN (0 * inf): no finite pixel exists for that vertex.
        let xs = LinearScale::new((0.0, 10.0), (0.0, f32::INFINITY)).expect("x");
        let ys = LinearScale::new((0.0, 10.0), (100.0, 0.0)).expect("y");
        assert_eq!(px((0.0, 5.0), &xs, &ys), None);
        assert_eq!(px((10.0, 5.0), &xs, &ys), None);
        let l = layer(&[0.0, 10.0], &[5.0, 5.0], Some(path(None, 2, None)));
        let mut fb = Framebuffer::new(100, 100).expect("fb");
        let before = fb.pixels().to_vec();
        draw_raster(&mut fb, &l, 0, DOM, (&xs, &ys));
        assert_eq!(fb.pixels(), &before[..]);
        let svg = add_svg(SvgEncoder::new(100, 100), &l, 0, DOM, (&xs, &ys)).render();
        assert!(!svg.contains("<polyline") && !svg.contains("NaN"), "{svg}");
    }

    #[test]
    fn raster_and_svg_draw_strokes_and_skip_other_layers() {
        let xs = LinearScale::new((0.0, 10.0), (0.0, 100.0)).expect("x");
        let ys = LinearScale::new((0.0, 10.0), (100.0, 0.0)).expect("y");
        let l = layer(&[0.0, 10.0], &[5.0, 5.0], Some(path(None, 2, None)));
        let mut fb = Framebuffer::new(100, 100).expect("fb");
        let before = fb.pixels().to_vec();
        draw_raster(&mut fb, &l, 0, DOM, (&xs, &ys));
        assert_ne!(fb.pixels(), &before[..], "the stroke changed pixels");
        let on_line = fb.get_pixel(50, 50).expect("pixel");
        assert_ne!(on_line, Rgba::new(0, 0, 0, 0));
        let svg = add_svg(SvgEncoder::new(100, 100), &l, 0, DOM, (&xs, &ys)).render();
        assert!(svg.contains("<polyline points=\"0,50 100,50\""), "{svg}");
        // A layer that is not a stroke layer leaves both targets untouched.
        let plain = layer(&[0.0, 10.0], &[5.0, 5.0], None);
        let mut fb2 = Framebuffer::new(100, 100).expect("fb");
        let blank = fb2.pixels().to_vec();
        draw_raster(&mut fb2, &plain, 0, DOM, (&xs, &ys));
        assert_eq!(fb2.pixels(), &blank[..]);
        let none = add_svg(SvgEncoder::new(100, 100), &plain, 0, DOM, (&xs, &ys)).render();
        assert!(!none.contains("<polyline"), "{none}");
    }
}
