//! Waveform widget: a realtime audio waveform drawn as one polyline.
//!
//! This is the pure half of the audio display. [`waveform_points`] maps a
//! buffer of samples to canvas points and is total: any input (NaN, infinity,
//! out-of-range values, empty buffers, degenerate bounds) yields points that
//! lie inside the target rectangle. [`Waveform`] owns a sample buffer and
//! paints that path. Capturing samples (microphone, Web Audio) is a separate
//! concern and not part of this module.
//!
//! Contract: `contracts/present-waveform-path-v1.yaml`.

use presentar_core::{
    widget::{AccessibleRole, LayoutResult},
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Event,
    Point, Rect, Size, TypeId, Widget,
};
use std::any::Any;
use std::time::Duration;

/// Default upper bound on the number of points a [`Waveform`] draws.
pub const DEFAULT_MAX_POINTS: usize = 1024;

/// Treat NaN and infinities as silence; clamp everything else to [-1, 1].
fn sanitize(sample: f32) -> f32 {
    if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// Largest magnitude accepted for any coordinate or extent. Larger finite
/// values are clamped so that `x + width` can never overflow to infinity.
const COORD_LIMIT: f32 = 1.0e15;

/// A finite coordinate: NaN and infinities become 0, huge values are clamped.
fn sane(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(-COORD_LIMIT, COORD_LIMIT)
    } else {
        0.0
    }
}

/// `bounds` made safe to compute with: every component finite and within
/// `COORD_LIMIT`, and the size not negative. Valid bounds in range pass through
/// unchanged.
fn sane_bounds(bounds: Rect) -> Rect {
    Rect::new(
        sane(bounds.x),
        sane(bounds.y),
        sane(bounds.width).max(0.0),
        sane(bounds.height).max(0.0),
    )
}

/// Whether `bounds` is finite with a size that is not negative.
fn valid_bounds(bounds: Rect) -> bool {
    [bounds.x, bounds.y, bounds.width, bounds.height]
        .iter()
        .all(|v| v.is_finite())
        && bounds.width >= 0.0
        && bounds.height >= 0.0
}

/// A usable preferred extent: finite values are clamped into
/// `[0, COORD_LIMIT]`; NaN and infinities count as "not set".
fn sane_extent(value: f32) -> Option<f32> {
    value.is_finite().then(|| value.clamp(0.0, COORD_LIMIT))
}

/// Fit `value` between `min` and `max` and keep it finite. Unlike
/// `f32::clamp` this never panics, whatever the constraints (NaN, infinite or
/// `min > max`); a lower bound that exceeds the upper one loses.
fn fit(value: f32, min: f32, max: f32) -> f32 {
    sane(value.max(min).min(max))
}

/// Keep whichever of the two samples has the larger magnitude (first wins ties).
fn larger_magnitude(best: f32, candidate: f32) -> f32 {
    if candidate.abs() > best.abs() {
        candidate
    } else {
        best
    }
}

/// Reduce `samples` to between 2 and `max(max_points, 2)` sanitized levels.
///
/// With more samples than the target, each bucket keeps its sample of largest
/// magnitude so that peaks survive. Fewer than two levels (silence, or a lone
/// sample) become a flat two-level line.
fn reduce_levels(samples: &[f32], max_points: usize) -> Vec<f32> {
    let target = max_points.max(2);
    let len = samples.len();
    let levels: Vec<f32> = if len <= target {
        samples.iter().copied().map(sanitize).collect()
    } else {
        (0..target)
            .map(|bucket| {
                let start = (bucket.saturating_mul(len) / target).min(len);
                let end = (bucket.saturating_add(1).saturating_mul(len) / target).min(len);
                samples
                    .get(start..end)
                    .unwrap_or_default()
                    .iter()
                    .copied()
                    .map(sanitize)
                    .fold(0.0, larger_magnitude)
            })
            .collect()
    };
    if levels.len() < 2 {
        let level = levels.first().copied().unwrap_or(0.0);
        return vec![level, level];
    }
    levels
}

/// Map audio samples to a polyline inside `bounds`.
///
/// * Samples are amplitudes nominally in [-1, 1]; larger magnitudes are
///   clamped, NaN and infinities count as 0.
/// * x runs from `bounds.x` (first point) to `bounds.x + bounds.width` (last
///   point); `y = centre - sample * (height / 2)`, so +1 is the top edge, -1
///   the bottom edge and 0 the centre line.
/// * More samples than `max_points` are reduced by buckets, keeping the
///   largest-magnitude sample per bucket. The result never has more than
///   `max(max_points, 2)` points.
/// * No samples (silence) gives the centre line: two points, left-centre and
///   right-centre. A lone sample gives a flat line at its level.
/// * Every point is finite and lies inside `bounds` (edges inclusive), for any
///   input. Negative width/height count as 0, non-finite components as 0, and
///   components beyond +-1e15 are clamped to it so that no sum overflows; the
///   "inside" promise is then to those sanitized bounds, which equal `bounds`
///   whenever it is finite, not negative and within +-1e15.
#[must_use]
pub fn waveform_points(samples: &[f32], bounds: Rect, max_points: usize) -> Vec<Point> {
    let levels = reduce_levels(samples, max_points);
    let safe = sane_bounds(bounds);
    let (x0, y0, width, height) = (safe.x, safe.y, safe.width, safe.height);
    let y1 = y0 + height;
    let half = height / 2.0;
    let centre = y0 + half;
    let span = levels.len().saturating_sub(1).max(1) as f32;

    levels
        .iter()
        .enumerate()
        .map(|(index, &level)| {
            let t = index as f32 / span;
            // t is in [0, 1] and width >= 0, so x stays in [x0, x0 + width].
            let x = t.mul_add(width, x0);
            let y = (-level).mul_add(half, centre).clamp(y0, y1);
            Point::new(x, y)
        })
        .collect()
}

/// Escape text for an HTML attribute value.
fn escape_attr(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// Realtime audio waveform widget.
#[derive(Debug, Clone)]
pub struct Waveform {
    /// Raw sample buffer (amplitudes, nominally in [-1, 1])
    samples: Vec<f32>,
    /// Line colour
    color: Color,
    /// Line width in pixels
    line_width: f32,
    /// Upper bound on drawn points
    max_points: usize,
    /// Preferred width
    width: Option<f32>,
    /// Preferred height
    height: Option<f32>,
    /// Accessible name
    accessible_name_value: Option<String>,
    /// Test ID
    test_id_value: Option<String>,
    /// Cached bounds
    bounds: Rect,
}

impl Default for Waveform {
    fn default() -> Self {
        Self {
            samples: Vec::new(),
            color: Color::new(0.1, 0.6, 0.9, 1.0),
            line_width: 1.5,
            max_points: DEFAULT_MAX_POINTS,
            width: None,
            height: None,
            accessible_name_value: None,
            test_id_value: None,
            bounds: Rect::default(),
        }
    }
}

impl Waveform {
    /// Create an empty waveform.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the sample buffer (builder form).
    #[must_use]
    pub fn samples(mut self, samples: Vec<f32>) -> Self {
        self.samples = samples;
        self
    }

    /// Replace the sample buffer in place; call this once per frame.
    pub fn set_samples(&mut self, samples: &[f32]) {
        self.samples.clear();
        self.samples.extend_from_slice(samples);
    }

    /// Set the line colour.
    #[must_use]
    pub fn color(mut self, color: Color) -> Self {
        self.color = color;
        self
    }

    /// Set the line width (negative values become 0).
    #[must_use]
    pub fn line_width(mut self, width: f32) -> Self {
        self.line_width = width.max(0.0);
        self
    }

    /// Set the upper bound on drawn points.
    #[must_use]
    pub fn max_points(mut self, max_points: usize) -> Self {
        self.max_points = max_points;
        self
    }

    /// Set the preferred width.
    ///
    /// Negative values become 0 and values above 1e15 are clamped to it; NaN
    /// and infinities count as "not set", so `measure` falls back to the
    /// default width instead of reporting a non-finite size.
    #[must_use]
    pub fn width(mut self, width: f32) -> Self {
        self.width = sane_extent(width);
        self
    }

    /// Set the preferred height (same rules as [`Waveform::width`]).
    #[must_use]
    pub fn height(mut self, height: f32) -> Self {
        self.height = sane_extent(height);
        self
    }

    /// Set both preferred width and height.
    #[must_use]
    pub fn size(self, width: f32, height: f32) -> Self {
        self.width(width).height(height)
    }

    /// Set the accessible name.
    #[must_use]
    pub fn accessible_name(mut self, name: impl Into<String>) -> Self {
        self.accessible_name_value = Some(name.into());
        self
    }

    /// Set the test ID.
    #[must_use]
    pub fn test_id(mut self, id: impl Into<String>) -> Self {
        self.test_id_value = Some(id.into());
        self
    }

    /// The raw sample buffer.
    #[must_use]
    pub fn get_samples(&self) -> &[f32] {
        &self.samples
    }

    /// The line colour.
    #[must_use]
    pub const fn get_color(&self) -> Color {
        self.color
    }

    /// The line width.
    #[must_use]
    pub const fn get_line_width(&self) -> f32 {
        self.line_width
    }

    /// The upper bound on drawn points.
    #[must_use]
    pub const fn get_max_points(&self) -> usize {
        self.max_points
    }

    /// Largest magnitude among the samples after clamping; 0.0 when empty.
    #[must_use]
    pub fn peak(&self) -> f32 {
        self.samples
            .iter()
            .copied()
            .map(sanitize)
            .fold(0.0, |best, v| best.max(v.abs()))
    }

    /// Root mean square of the samples after clamping; 0.0 when empty.
    #[must_use]
    pub fn rms(&self) -> f32 {
        let sum: f32 = self
            .samples
            .iter()
            .copied()
            .map(sanitize)
            .map(|v| v * v)
            .sum();
        (sum / self.samples.len().max(1) as f32).sqrt()
    }

    /// Whether a non-blank accessible name has been set.
    fn has_accessible_name(&self) -> bool {
        self.accessible_name_value
            .as_deref()
            .is_some_and(|name| !name.trim().is_empty())
    }
}

impl Widget for Waveform {
    fn type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }

    fn measure(&self, constraints: Constraints) -> Size {
        // Not `Constraints::constrain`: its `f32::clamp` panics on NaN or
        // inverted constraints and passes an infinite minimum through.
        Size::new(
            fit(
                self.width.unwrap_or(200.0),
                constraints.min_width,
                constraints.max_width,
            ),
            fit(
                self.height.unwrap_or(80.0),
                constraints.min_height,
                constraints.max_height,
            ),
        )
    }

    fn layout(&mut self, bounds: Rect) -> LayoutResult {
        self.bounds = bounds;
        LayoutResult {
            size: bounds.size(),
        }
    }

    fn paint(&self, canvas: &mut dyn Canvas) {
        let points = waveform_points(&self.samples, self.bounds, self.max_points);
        canvas.draw_path(&points, self.color, self.line_width);
    }

    fn event(&mut self, _event: &Event) -> Option<Box<dyn Any + Send>> {
        None
    }

    fn children(&self) -> &[Box<dyn Widget>] {
        &[]
    }

    fn children_mut(&mut self) -> &mut [Box<dyn Widget>] {
        &mut []
    }

    fn accessible_name(&self) -> Option<&str> {
        self.accessible_name_value.as_deref()
    }

    fn accessible_role(&self) -> AccessibleRole {
        AccessibleRole::Image
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }

    fn bounds(&self) -> Rect {
        self.bounds
    }
}

// PROBAR-SPEC-009: Brick Architecture - Tests define interface
impl Brick for Waveform {
    fn brick_name(&self) -> &'static str {
        "Waveform"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[BrickAssertion::MaxLatencyMs(16)]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = self.assertions().to_vec();
        let mut failed = Vec::new();

        // A waveform has no text of its own: screen readers need a name.
        let named = BrickAssertion::ElementPresent("accessible-name".to_string());
        if self.has_accessible_name() {
            passed.push(named);
        } else {
            failed.push((named, "Waveform has no accessible name".to_string()));
        }

        // The laid-out bounds must be finite and not negative.
        let bounds_ok = BrickAssertion::ElementPresent("valid-bounds".to_string());
        if valid_bounds(self.bounds) {
            passed.push(bounds_ok);
        } else {
            failed.push((
                bounds_ok,
                "Waveform bounds are not finite and >= 0".to_string(),
            ));
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(10),
        }
    }

    fn to_html(&self) -> String {
        let label = escape_attr(self.accessible_name_value.as_deref().unwrap_or(""));
        format!(r#"<canvas class="brick-waveform" role="img" aria-label="{label}"></canvas>"#)
    }

    fn to_css(&self) -> String {
        ".brick-waveform { display: block; }".to_string()
    }

    fn test_id(&self) -> Option<&str> {
        self.test_id_value.as_deref()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use presentar_core::draw::DrawCommand;
    use presentar_core::RecordingCanvas;
    use proptest::prelude::*;

    const B: Rect = Rect {
        x: 10.0,
        y: 20.0,
        width: 100.0,
        height: 40.0,
    };

    fn centre_y(b: Rect) -> f32 {
        b.y + b.height / 2.0
    }

    fn inside(p: &Point, b: Rect) -> bool {
        p.x >= b.x && p.x <= b.x + b.width && p.y >= b.y && p.y <= b.y + b.height
    }

    // ===== waveform_points =====

    #[test]
    fn points_empty_is_centre_line() {
        let pts = waveform_points(&[], B, 64);
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0], Point::new(10.0, 40.0));
        assert_eq!(pts[1], Point::new(110.0, 40.0));
    }

    #[test]
    fn points_all_zero_on_centre() {
        let pts = waveform_points(&[0.0; 50], B, 64);
        assert_eq!(pts.len(), 50);
        assert!(pts.iter().all(|p| p.y == centre_y(B)));
    }

    #[test]
    fn points_map_extremes_to_edges() {
        let pts = waveform_points(&[1.0, 0.0, -1.0], B, 64);
        assert_eq!(pts[0].y, 20.0); // +1 = top edge
        assert_eq!(pts[1].y, 40.0); // 0 = centre
        assert_eq!(pts[2].y, 60.0); // -1 = bottom edge
    }

    #[test]
    fn points_x_spans_bounds() {
        let pts = waveform_points(&[0.1, 0.2, 0.3, 0.4, 0.5], B, 64);
        assert_eq!(pts[0].x, 10.0);
        assert_eq!(pts[4].x, 110.0);
        assert_eq!(pts[2].x, 60.0);
    }

    #[test]
    fn points_clamp_out_of_range() {
        let pts = waveform_points(&[5.0, -5.0, f32::MAX, f32::MIN], B, 64);
        assert_eq!(pts[0].y, 20.0);
        assert_eq!(pts[1].y, 60.0);
        assert_eq!(pts[2].y, 20.0);
        assert_eq!(pts[3].y, 60.0);
    }

    #[test]
    fn points_nonfinite_is_silence() {
        let pts = waveform_points(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY], B, 64);
        assert!(pts.iter().all(|p| p.y == centre_y(B)));
    }

    #[test]
    fn points_downsample_keeps_peaks() {
        let mut samples = vec![0.1_f32; 100];
        samples[37] = -0.9;
        let bounds = Rect::new(0.0, 0.0, 100.0, 20.0);
        let pts = waveform_points(&samples, bounds, 10);
        assert_eq!(pts.len(), 10);
        // -0.9 maps to y = 10 + 0.9 * 10 = 19
        assert!(pts.iter().any(|p| (p.y - 19.0).abs() < 1e-4));
        // the bucket holding the spike is not the spike's neighbour bucket
        assert!((pts[3].y - 19.0).abs() < 1e-4);
        assert!((pts[2].y - 9.0).abs() < 1e-4);
    }

    #[test]
    fn points_downsample_largest_magnitude_ignores_sign() {
        // bucket of 2: -0.5 then 0.8 -> keeps 0.8 (larger magnitude), then
        // 0.8 then -0.5 -> keeps 0.8 (first wins)
        let pts = waveform_points(&[-0.5, 0.8, 0.8, -0.5, 0.0, 0.0], B, 3);
        assert_eq!(pts.len(), 3);
        let top = centre_y(B) - 0.8 * 20.0;
        assert!((pts[0].y - top).abs() < 1e-4);
        assert!((pts[1].y - top).abs() < 1e-4);
        assert_eq!(pts[2].y, centre_y(B));
    }

    #[test]
    fn points_max_points_zero_and_one_do_not_panic() {
        for max in [0_usize, 1] {
            let pts = waveform_points(&[0.5, -0.5, 0.25, 0.0], B, max);
            assert_eq!(pts.len(), 2);
            assert!(pts.iter().all(|p| inside(p, B)));
            let none = waveform_points(&[], B, max);
            assert_eq!(none.len(), 2);
        }
    }

    #[test]
    fn points_huge_max_points_is_all_samples() {
        let pts = waveform_points(&[0.1, 0.2, 0.3], B, usize::MAX);
        assert_eq!(pts.len(), 3);
    }

    #[test]
    fn points_single_sample_is_flat_line() {
        let pts = waveform_points(&[0.5], B, 64);
        assert_eq!(pts.len(), 2);
        assert_eq!(pts[0].y, pts[1].y);
        assert_eq!(pts[0].x, 10.0);
        assert_eq!(pts[1].x, 110.0);
    }

    #[test]
    fn points_degenerate_bounds_stay_inside() {
        for b in [
            Rect::new(5.0, 7.0, 0.0, 0.0),
            Rect::new(5.0, 7.0, 0.0, 30.0),
            Rect::new(5.0, 7.0, 30.0, 0.0),
            Rect::new(5.0, 7.0, -3.0, -3.0),
            Rect::new(f32::NAN, f32::INFINITY, f32::NAN, f32::INFINITY),
        ] {
            let pts = waveform_points(&[1.0, -1.0, 0.3], b, 8);
            assert!(finite_and_inside(&pts, b), "{b:?}");
        }
    }

    #[test]
    fn points_table_of_generated_inputs_stay_inside() {
        // deterministic pseudo-random sweep, including non-finite values
        let mut state = 0x2545_F491_4F6C_DD1D_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let specials = [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MAX,
            f32::MIN,
        ];
        for _ in 0..200 {
            let n = (next() % 300) as usize;
            let samples: Vec<f32> = (0..n)
                .map(|_| {
                    let r = next();
                    if r % 11 == 0 {
                        specials[(r >> 8) as usize % specials.len()]
                    } else {
                        ((r >> 16) % 4001) as f32 / 1000.0 - 2.0
                    }
                })
                .collect();
            let b = Rect::new(
                (next() % 2000) as f32 - 1000.0,
                (next() % 2000) as f32 - 1000.0,
                (next() % 500) as f32,
                (next() % 500) as f32,
            );
            let max = (next() % 70) as usize;
            let pts = waveform_points(&samples, b, max);
            assert!(pts.len() >= 2 && pts.len() <= max.max(2));
            assert!(pts.iter().all(|p| inside(p, b)));
            assert!(pts.windows(2).all(|w| w[0].x <= w[1].x));
        }
    }

    /// Ordinary coordinates mixed with every f32, NaN and infinities included.
    fn bound_coord() -> impl Strategy<Value = f32> {
        prop_oneof![-1.0e6_f32..1.0e6, proptest::num::f32::ANY]
    }

    proptest! {
        #[test]
        fn prop_points_inside_bounds(
            samples in proptest::collection::vec(proptest::num::f32::ANY, 0..200),
            x in bound_coord(),
            y in bound_coord(),
            w in bound_coord(),
            h in bound_coord(),
            max in 0_usize..300,
        ) {
            let b = Rect::new(x, y, w, h);
            let pts = waveform_points(&samples, b, max);
            prop_assert!(finite_and_inside(&pts, b), "{:?} -> {:?}", b, pts);
        }

        #[test]
        fn prop_points_len_bounded(
            samples in proptest::collection::vec(proptest::num::f32::ANY, 0..200),
            max in 0_usize..300,
        ) {
            let pts = waveform_points(&samples, B, max);
            prop_assert!(pts.len() >= 2);
            prop_assert!(pts.len() <= max.max(2));
        }

        #[test]
        fn prop_points_x_non_decreasing(
            samples in proptest::collection::vec(proptest::num::f32::ANY, 0..200),
            x in -1.0e6_f32..1.0e6,
            w in 0.0_f32..1.0e6,
            max in 0_usize..300,
        ) {
            let pts = waveform_points(&samples, Rect::new(x, 0.0, w, 10.0), max);
            prop_assert!(pts.windows(2).all(|p| p[0].x <= p[1].x));
        }

        #[test]
        fn prop_silence_is_centre_line(
            n in 0_usize..200,
            y in -1.0e6_f32..1.0e6,
            h in 0.0_f32..1.0e6,
            max in 0_usize..300,
        ) {
            let b = Rect::new(0.0, y, 50.0, h);
            let pts = waveform_points(&vec![0.0; n], b, max);
            let c = y + h / 2.0;
            prop_assert!(pts.iter().all(|p| p.y == c));
        }
    }

    // ===== Overflow-proof bounds (APR-6897 review, defect 2) =====

    /// Documented coordinate limit: independent of the implementation's constant.
    const LIMIT: f32 = 1.0e15;

    fn oracle_coord(value: f32) -> f32 {
        if value.is_finite() {
            value.clamp(-LIMIT, LIMIT)
        } else {
            0.0
        }
    }

    /// The rectangle the contract promises the points stay in.
    fn oracle_bounds(b: Rect) -> Rect {
        Rect::new(
            oracle_coord(b.x),
            oracle_coord(b.y),
            oracle_coord(b.width).max(0.0),
            oracle_coord(b.height).max(0.0),
        )
    }

    fn finite_and_inside(pts: &[Point], b: Rect) -> bool {
        let eff = oracle_bounds(b);
        pts.iter()
            .all(|p| p.x.is_finite() && p.y.is_finite() && inside(p, eff))
    }

    #[test]
    fn points_reviewer_repro_f32_max_bounds_are_finite_and_inside() {
        let b = Rect::new(f32::MAX, f32::MAX, f32::MAX, f32::MAX);
        let pts = waveform_points(&[0.0, 1.0, -1.0], b, 4);
        assert_eq!(pts.len(), 3);
        assert!(finite_and_inside(&pts, b), "{pts:?}");
        // The clamped rectangle is [1e15, 2e15] on both axes (f32 ulp there is
        // about 1e8, so compare with a tolerance).
        let near = |actual: f32, want: f32| (actual - want).abs() <= 4.0e8;
        assert!(near(pts[0].x, LIMIT) && near(pts[0].y, 1.5 * LIMIT));
        assert!(near(pts[1].x, 1.5 * LIMIT) && near(pts[1].y, LIMIT));
        assert!(near(pts[2].x, 2.0 * LIMIT) && near(pts[2].y, 2.0 * LIMIT));
    }

    #[test]
    fn points_extreme_bounds_table_stays_finite_and_inside() {
        let extremes = [
            f32::MAX,
            f32::MIN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::MIN_POSITIVE,
            -0.0,
            LIMIT,
            -LIMIT,
            3.0e15,
        ];
        for &x in &extremes {
            for &w in &extremes {
                let b = Rect::new(x, f32::MAX, w, w);
                let pts = waveform_points(&[1.0, -1.0, 0.25, f32::NAN], b, 8);
                assert!(finite_and_inside(&pts, b), "{b:?} -> {pts:?}");
                let b = Rect::new(f32::MIN, x, w, f32::MAX);
                let pts = waveform_points(&[1.0, -1.0, 0.25, f32::NAN], b, 8);
                assert!(finite_and_inside(&pts, b), "{b:?} -> {pts:?}");
            }
        }
    }

    #[test]
    fn points_negative_extents_collapse_to_the_origin_point() {
        let b = Rect::new(10.0, 20.0, -50.0, -40.0);
        let pts = waveform_points(&[1.0, -1.0, 0.5], b, 8);
        assert!(pts.iter().all(|p| *p == Point::new(10.0, 20.0)), "{pts:?}");
    }

    proptest! {
        #[test]
        fn prop_points_finite_and_inside_any_f32_bounds(
            samples in proptest::collection::vec(proptest::num::f32::ANY, 0..100),
            x in proptest::num::f32::ANY,
            y in proptest::num::f32::ANY,
            w in proptest::num::f32::ANY,
            h in proptest::num::f32::ANY,
            max in 0_usize..100,
        ) {
            let b = Rect::new(x, y, w, h);
            let pts = waveform_points(&samples, b, max);
            prop_assert!(finite_and_inside(&pts, b), "{:?} -> {:?}", b, pts);
        }
    }

    // ===== Non-finite preferred sizes and constraints =====

    #[test]
    fn non_finite_preferred_size_counts_as_absent() {
        let unbounded = Constraints::unbounded();
        for bad in [f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
            let w = Waveform::new().size(bad, bad);
            assert_eq!(w.measure(unbounded), Size::new(200.0, 80.0), "{bad}");
        }
    }

    #[test]
    fn huge_preferred_size_is_clamped_to_the_coordinate_limit() {
        let w = Waveform::new().size(f32::MAX, f32::MAX);
        assert_eq!(w.measure(Constraints::unbounded()), Size::new(LIMIT, LIMIT));
    }

    #[test]
    fn measure_is_finite_for_degenerate_constraints() {
        let w = Waveform::new().size(f32::MAX, 10.0);
        for c in [
            Constraints::unbounded(),
            Constraints::new(f32::INFINITY, f32::INFINITY, 0.0, 5.0),
            Constraints::new(0.0, f32::NAN, f32::NAN, f32::INFINITY),
            Constraints::new(9.0, 3.0, 9.0, 3.0),
        ] {
            let s = w.measure(c);
            assert!(
                s.width.is_finite() && s.height.is_finite(),
                "{c:?} -> {s:?}"
            );
        }
    }

    proptest! {
        #[test]
        fn prop_measure_finite_for_any_constraints(
            a in proptest::num::f32::ANY,
            b in proptest::num::f32::ANY,
            c in proptest::num::f32::ANY,
            d in proptest::num::f32::ANY,
        ) {
            let s = Waveform::new().measure(Constraints::new(a, b, c, d));
            prop_assert!(s.width.is_finite() && s.height.is_finite());
        }
    }

    // ===== verify: valid bounds =====

    #[test]
    fn brick_verify_flags_invalid_bounds() {
        for b in [
            Rect::new(f32::NAN, 0.0, 10.0, 10.0),
            Rect::new(0.0, f32::INFINITY, 10.0, 10.0),
            Rect::new(0.0, 0.0, -1.0, 10.0),
            Rect::new(0.0, 0.0, 10.0, -1.0),
            Rect::new(0.0, 0.0, f32::NAN, 10.0),
            Rect::new(0.0, 0.0, 10.0, f32::NEG_INFINITY),
        ] {
            let mut w = Waveform::new().accessible_name("Mic level");
            w.layout(b);
            let v = w.verify();
            assert!(!v.is_valid(), "{b:?}");
            assert_eq!(v.failed.len(), 1, "{b:?}");
            assert_eq!(
                v.failed[0].0,
                BrickAssertion::ElementPresent("valid-bounds".to_string())
            );
            assert!(v.failed[0].1.contains("bounds"));
        }
    }

    #[test]
    fn brick_verify_accepts_valid_laid_out_bounds() {
        let mut w = Waveform::new().accessible_name("Mic level");
        w.layout(B);
        let v = w.verify();
        assert!(v.is_valid());
        assert!(v
            .passed
            .contains(&BrickAssertion::ElementPresent("valid-bounds".to_string())));
    }

    // ===== peak / rms =====

    #[test]
    fn peak_is_max_clamped_magnitude() {
        assert_eq!(Waveform::new().peak(), 0.0);
        assert_eq!(Waveform::new().samples(vec![0.2, -0.7, 0.5]).peak(), 0.7);
        assert_eq!(Waveform::new().samples(vec![9.0, -3.0]).peak(), 1.0);
        assert_eq!(
            Waveform::new()
                .samples(vec![f32::NAN, f32::INFINITY])
                .peak(),
            0.0
        );
    }

    #[test]
    fn rms_of_known_signals() {
        assert_eq!(Waveform::new().rms(), 0.0);
        assert_eq!(
            Waveform::new().samples(vec![1.0, -1.0, 1.0, -1.0]).rms(),
            1.0
        );
        let half = Waveform::new().samples(vec![0.5; 4]).rms();
        assert!((half - 0.5).abs() < 1e-6);
        // 3.0 clamps to 1.0, NaN counts as 0 -> sqrt(1/2)
        let clamped = Waveform::new().samples(vec![3.0, f32::NAN]).rms();
        assert!((clamped - 0.5_f32.sqrt()).abs() < 1e-6);
    }

    // ===== Waveform builder =====

    #[test]
    fn widget_defaults_and_builders() {
        let w = Waveform::new();
        assert!(w.get_samples().is_empty());
        assert_eq!(w.get_max_points(), DEFAULT_MAX_POINTS);
        assert_eq!(w.get_line_width(), 1.5);

        let red = Color::new(1.0, 0.0, 0.0, 1.0);
        let w = Waveform::new()
            .samples(vec![0.1, 0.2])
            .color(red)
            .line_width(3.0)
            .max_points(64)
            .size(300.0, 90.0)
            .accessible_name("Microphone level")
            .test_id("wave");
        assert_eq!(w.get_samples(), &[0.1, 0.2]);
        assert_eq!(w.get_color(), red);
        assert_eq!(w.get_line_width(), 3.0);
        assert_eq!(w.get_max_points(), 64);
        assert_eq!(Widget::accessible_name(&w), Some("Microphone level"));
        assert_eq!(Widget::test_id(&w), Some("wave"));
        assert_eq!(Brick::test_id(&w), Some("wave"));
    }

    #[test]
    fn line_width_and_size_never_negative() {
        let w = Waveform::new().line_width(-2.0).width(-1.0).height(-1.0);
        assert_eq!(w.get_line_width(), 0.0);
        assert_eq!(
            w.measure(Constraints::loose(Size::new(500.0, 500.0))),
            Size::new(0.0, 0.0)
        );
    }

    #[test]
    fn set_samples_replaces_buffer() {
        let mut w = Waveform::new().samples(vec![0.9; 10]);
        w.set_samples(&[0.1, 0.2, 0.3]);
        assert_eq!(w.get_samples(), &[0.1, 0.2, 0.3]);
        w.set_samples(&[]);
        assert!(w.get_samples().is_empty());
    }

    // ===== Widget =====

    #[test]
    fn measure_defaults_and_preferred() {
        let loose = Constraints::loose(Size::new(500.0, 500.0));
        assert_eq!(Waveform::new().measure(loose), Size::new(200.0, 80.0));
        assert_eq!(
            Waveform::new().size(320.0, 60.0).measure(loose),
            Size::new(320.0, 60.0)
        );
    }

    #[test]
    fn layout_caches_bounds() {
        let mut w = Waveform::new();
        let result = w.layout(B);
        assert_eq!(result.size, Size::new(100.0, 40.0));
        assert_eq!(Widget::bounds(&w), B);
    }

    #[test]
    fn widget_trait_surface() {
        let mut w = Waveform::new();
        assert_eq!(Widget::type_id(&w), TypeId::of::<Waveform>());
        assert!(w
            .event(&Event::MouseMove {
                position: Point::ORIGIN
            })
            .is_none());
        assert!(w.children().is_empty());
        assert!(w.children_mut().is_empty());
        assert!(!w.is_interactive());
        assert!(!w.is_focusable());
        assert_eq!(w.accessible_role(), AccessibleRole::Image);
        assert_eq!(Widget::accessible_name(&w), None);
        assert_eq!(Widget::test_id(&w), None);
    }

    #[test]
    fn paint_draws_one_path_from_waveform_points() {
        let red = Color::new(1.0, 0.0, 0.0, 1.0);
        let mut w = Waveform::new()
            .samples(vec![0.0, 1.0, -1.0])
            .color(red)
            .line_width(2.0);
        w.layout(B);
        let mut canvas = RecordingCanvas::new();
        w.paint(&mut canvas);
        let cmds = canvas.take_commands();
        assert_eq!(cmds.len(), 1);
        let expected = waveform_points(&[0.0, 1.0, -1.0], B, DEFAULT_MAX_POINTS);
        assert!(matches!(
            &cmds[0],
            DrawCommand::Path { points, closed, style }
                if *points == expected && !closed && style.color == red && style.width == 2.0
        ));
    }

    #[test]
    fn paint_honours_max_points() {
        let samples: Vec<f32> = (0..100).map(|i| (i as f32 / 50.0) - 1.0).collect();
        let mut w = Waveform::new().samples(samples.clone()).max_points(4);
        w.layout(B);
        let mut canvas = RecordingCanvas::new();
        w.paint(&mut canvas);
        let expected = waveform_points(&samples, B, 4);
        assert_eq!(expected.len(), 4);
        assert!(matches!(
            &canvas.take_commands()[0],
            DrawCommand::Path { points, .. } if *points == expected
        ));
    }

    #[test]
    fn paint_without_samples_draws_centre_line() {
        let mut w = Waveform::new();
        w.layout(B);
        let mut canvas = RecordingCanvas::new();
        w.paint(&mut canvas);
        let expected = vec![Point::new(10.0, 40.0), Point::new(110.0, 40.0)];
        assert!(matches!(
            &canvas.take_commands()[0],
            DrawCommand::Path { points, .. } if *points == expected
        ));
    }

    // ===== Brick =====

    #[test]
    fn brick_verify_passes_with_accessible_name() {
        let w = Waveform::new().accessible_name("Mic level");
        let v = w.verify();
        assert!(v.is_valid());
        assert!(v.failed.is_empty());
        assert!(w.can_render());
        assert_eq!(w.brick_name(), "Waveform");
        assert_eq!(w.assertions(), &[BrickAssertion::MaxLatencyMs(16)]);
        assert_eq!(w.budget(), BrickBudget::uniform(16));
    }

    #[test]
    fn brick_verify_fails_without_accessible_name() {
        for w in [
            Waveform::new(),
            Waveform::new().accessible_name(""),
            Waveform::new().accessible_name("   "),
        ] {
            let v = w.verify();
            assert!(!v.is_valid());
            assert!(!w.can_render());
            assert_eq!(v.failed.len(), 1);
            assert!(matches!(v.failed[0].0, BrickAssertion::ElementPresent(_)));
            assert!(v.failed[0].1.contains("accessible name"));
        }
    }

    #[test]
    fn brick_html_and_css() {
        let w = Waveform::new().accessible_name(r#"Mic <"A"> & B"#);
        let html = w.to_html();
        assert!(html.contains("brick-waveform"));
        assert!(html.contains("Mic &lt;&quot;A&quot;&gt; &amp; B"));
        assert!(Waveform::new().to_html().contains(r#"aria-label="""#));
        assert!(w.to_css().contains(".brick-waveform"));
    }
}
