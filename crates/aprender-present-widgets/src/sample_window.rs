//! `SampleWindow`: a fixed-capacity window of the newest audio samples.
//!
//! This is the step before the [`Waveform`](crate::waveform::Waveform) widget
//! in a live meter: capture code pushes frames as they arrive, and each paint
//! reads the window, oldest sample first. The window is a ring buffer over a
//! `Vec<f32>` allocated once in [`SampleWindow::new`], so [`SampleWindow::push`]
//! never allocates. Pure Rust: nothing here touches time, threads or files, so
//! it builds for `wasm32`.
//!
//! Contract: `contracts/present-sample-window-v1.yaml`.

/// Treat NaN and infinities as silence; clamp everything else to [-1, 1].
///
/// The same rule as `sanitize` in `waveform.rs`, which is private there.
fn sanitize(sample: f32) -> f32 {
    if sample.is_finite() {
        sample.clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

/// The newest `capacity` audio samples, oldest first.
///
/// Stored samples are already sanitized: non-finite values are 0.0 and finite
/// values lie in [-1, 1]. Capacity 0 is allowed and holds nothing.
#[derive(Debug, Clone)]
pub struct SampleWindow {
    /// Ring storage; `buf.len()` is the capacity and never changes.
    buf: Vec<f32>,
    /// Index the next sample is written to, in `0..capacity` (0 when capacity is 0).
    next: usize,
    /// Number of valid samples, `<= capacity`. While `len < capacity`, `next == len`.
    len: usize,
}

/// Two windows are equal when they have the same capacity and hold the same
/// samples in the same order; the rotation of the ring storage is irrelevant.
/// Stored samples are never NaN, so this is a true equivalence.
impl PartialEq for SampleWindow {
    fn eq(&self, other: &Self) -> bool {
        self.capacity() == other.capacity() && self.iter().eq(other.iter())
    }
}

impl SampleWindow {
    /// Create an empty window holding at most `capacity` samples.
    ///
    /// Capacity 0 is allowed: such a window ignores every push and stays
    /// empty. The only allocation this type ever makes is here (and in
    /// [`Self::samples`] / [`Self::copy_into`] for the caller's copy).
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: vec![0.0; capacity],
            next: 0,
            len: 0,
        }
    }

    /// Append a frame, keeping the newest samples and dropping the oldest.
    ///
    /// A frame longer than the capacity keeps its tail. Non-finite samples are
    /// stored as 0.0 and finite ones are clamped to [-1, 1]. Never allocates.
    pub fn push(&mut self, frame: &[f32]) {
        let capacity = self.buf.len();
        if capacity == 0 {
            return;
        }
        // Only the last `capacity` samples of the frame can survive.
        let skip = frame.len().saturating_sub(capacity);
        let mut written = 0;
        for &sample in frame.iter().skip(skip) {
            if let Some(slot) = self.buf.get_mut(self.next) {
                *slot = sanitize(sample);
            }
            self.next += 1;
            if self.next == capacity {
                self.next = 0;
            }
            written += 1;
        }
        self.len = (self.len + written).min(capacity);
    }

    /// Number of samples currently held.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Whether the window holds no samples.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Maximum number of samples the window holds.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Drop every sample; the capacity (and the storage) is unchanged.
    pub fn clear(&mut self) {
        self.next = 0;
        self.len = 0;
    }

    /// The window, oldest sample first. Allocates; see [`Self::copy_into`].
    #[must_use]
    pub fn samples(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.len);
        self.copy_into(&mut out);
        out
    }

    /// Clear `out` and fill it with the window, oldest sample first.
    ///
    /// Reuses `out`'s allocation when it is large enough.
    pub fn copy_into(&self, out: &mut Vec<f32>) {
        out.clear();
        out.extend(self.iter());
    }

    /// Largest magnitude in the window; 0.0 when empty.
    #[must_use]
    pub fn peak(&self) -> f32 {
        self.iter().fold(0.0, |best, v| best.max(v.abs()))
    }

    /// Root mean square of the window; 0.0 when empty.
    #[must_use]
    pub fn rms(&self) -> f32 {
        let sum: f32 = self.iter().map(|v| v * v).sum();
        (sum / self.len.max(1) as f32).sqrt()
    }

    /// The window, oldest first, without allocating.
    fn iter(&self) -> impl Iterator<Item = f32> + '_ {
        // Not full: samples sit at 0..len and `next == len`. Full: the oldest
        // sample is at `next`. (Capacity 0 is "full" with `next == 0`.)
        let start = if self.len == self.buf.len() {
            self.next
        } else {
            0
        };
        self.buf.iter().copied().cycle().skip(start).take(self.len)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;
    use crate::waveform::Waveform;
    use presentar_core::{RecordingCanvas, Rect, Widget};
    use proptest::prelude::*;

    /// Address and capacity of the ring storage, to prove push does not reallocate.
    fn buffer_identity(w: &SampleWindow) -> (*const f32, usize) {
        (w.buf.as_ptr(), w.buf.capacity())
    }

    /// Independent model of the sanitizing rule.
    fn model_sanitize(v: f32) -> f32 {
        if v.is_nan() || v.is_infinite() {
            0.0
        } else {
            v.clamp(-1.0, 1.0)
        }
    }

    fn sample_strategy() -> impl Strategy<Value = f32> {
        prop_oneof![
            proptest::num::f32::ANY,
            -3.0_f32..3.0,
            Just(f32::NAN),
            Just(f32::INFINITY),
            Just(f32::NEG_INFINITY),
        ]
    }

    proptest! {
        #[test]
        fn prop_window_equals_model(
            capacity in 0_usize..64,
            frames in proptest::collection::vec(
                proptest::collection::vec(sample_strategy(), 0..200),
                0..8,
            ),
        ) {
            let mut window = SampleWindow::new(capacity);
            let mut model: Vec<f32> = Vec::new();
            for frame in &frames {
                window.push(frame);
                model.extend(frame.iter().copied().map(model_sanitize));
                let keep = model.len().min(capacity);
                let expected = model[model.len() - keep..].to_vec();
                prop_assert_eq!(window.len(), expected.len());
                prop_assert_eq!(window.is_empty(), expected.is_empty());
                prop_assert_eq!(window.samples(), expected);
            }
            prop_assert_eq!(window.capacity(), capacity);
        }
    }

    #[test]
    fn new_is_empty_with_capacity() {
        let w = SampleWindow::new(5);
        assert_eq!(w.capacity(), 5);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
        assert!(w.samples().is_empty());
    }

    #[test]
    fn frame_larger_than_capacity_keeps_tail() {
        let mut w = SampleWindow::new(3);
        w.push(&[0.1, 0.2, 0.3, 0.4, 0.5]);
        assert_eq!(w.samples(), vec![0.3, 0.4, 0.5]);
        assert_eq!(w.len(), 3);
        // and again over a non-empty window
        w.push(&[-0.1, -0.2, -0.3, -0.4]);
        assert_eq!(w.samples(), vec![-0.2, -0.3, -0.4]);
    }

    #[test]
    fn empty_frame_changes_nothing() {
        let mut w = SampleWindow::new(4);
        w.push(&[]);
        assert!(w.is_empty());
        w.push(&[0.1, 0.2]);
        let before = w.clone();
        w.push(&[]);
        assert_eq!(w, before);
        assert_eq!(w.samples(), vec![0.1, 0.2]);
    }

    #[test]
    fn capacity_zero_never_panics_and_stays_empty() {
        let mut w = SampleWindow::new(0);
        w.push(&[0.5, 0.6]);
        w.push(&[]);
        assert_eq!(w.capacity(), 0);
        assert_eq!(w.len(), 0);
        assert!(w.is_empty());
        assert!(w.samples().is_empty());
        assert_eq!(w.peak(), 0.0);
        assert_eq!(w.rms(), 0.0);
        let mut out = vec![9.0];
        w.copy_into(&mut out);
        assert!(out.is_empty());
        w.clear();
        assert!(w.is_empty());
    }

    #[test]
    fn wrap_around_keeps_arrival_order() {
        let mut w = SampleWindow::new(4);
        w.push(&[0.1, 0.2, 0.3]);
        assert_eq!(w.samples(), vec![0.1, 0.2, 0.3]);
        w.push(&[0.4, 0.5, 0.6]);
        assert_eq!(w.samples(), vec![0.3, 0.4, 0.5, 0.6]);
        w.push(&[0.7]);
        assert_eq!(w.samples(), vec![0.4, 0.5, 0.6, 0.7]);
        // exactly full, no wrap yet, then a push that lands on index 0
        let mut full = SampleWindow::new(3);
        full.push(&[0.1, 0.2, 0.3]);
        assert_eq!(full.samples(), vec![0.1, 0.2, 0.3]);
        full.push(&[0.4]);
        assert_eq!(full.samples(), vec![0.2, 0.3, 0.4]);
    }

    #[test]
    fn non_finite_is_zero_and_finite_is_clamped() {
        let mut w = SampleWindow::new(8);
        w.push(&[
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            5.0,
            -5.0,
            f32::MAX,
            0.25,
        ]);
        assert_eq!(w.samples(), vec![0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.25]);
    }

    #[test]
    fn copy_into_equals_samples_and_reuses_buffer() {
        let mut w = SampleWindow::new(4);
        w.push(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let mut out: Vec<f32> = Vec::with_capacity(64);
        out.extend_from_slice(&[9.0; 10]);
        let ptr = out.as_ptr();
        w.copy_into(&mut out);
        assert_eq!(out, w.samples());
        assert_eq!(out, vec![0.3, 0.4, 0.5, 0.6]);
        assert!(out.capacity() >= 64);
        assert_eq!(out.as_ptr(), ptr);
        // an empty window empties the buffer but keeps its capacity
        let empty = SampleWindow::new(4);
        empty.copy_into(&mut out);
        assert!(out.is_empty());
        assert!(out.capacity() >= 64);
    }

    #[test]
    fn clear_empties_and_later_push_works() {
        let mut w = SampleWindow::new(3);
        w.push(&[0.1, 0.2, 0.3, 0.4]);
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.len(), 0);
        assert_eq!(w.capacity(), 3);
        assert!(w.samples().is_empty());
        assert_eq!(w.peak(), 0.0);
        w.push(&[0.5, 0.6]);
        assert_eq!(w.samples(), vec![0.5, 0.6]);
        w.push(&[0.7, 0.8]);
        assert_eq!(w.samples(), vec![0.6, 0.7, 0.8]);
    }

    #[test]
    fn clear_of_a_part_filled_window_restarts_at_the_front() {
        // The write position is 3 here, not 0: a clear that forgot it would
        // put the next frame at the wrong end of the storage.
        let mut w = SampleWindow::new(4);
        w.push(&[0.1, 0.2, 0.3]);
        w.clear();
        w.push(&[0.5, 0.6]);
        assert_eq!(w.samples(), vec![0.5, 0.6]);
        w.push(&[0.7, 0.8, 0.9]);
        assert_eq!(w.samples(), vec![0.6, 0.7, 0.8, 0.9]);
    }

    #[test]
    fn peak_and_rms_hand_computed() {
        let empty = SampleWindow::new(4);
        assert_eq!(empty.peak(), 0.0);
        assert_eq!(empty.rms(), 0.0);

        let mut w = SampleWindow::new(4);
        w.push(&[0.2, -0.7, 0.5]);
        assert_eq!(w.peak(), 0.7);

        let mut sq = SampleWindow::new(4);
        sq.push(&[1.0, -1.0, 1.0, -1.0]);
        assert_eq!(sq.peak(), 1.0);
        assert_eq!(sq.rms(), 1.0);

        let mut half = SampleWindow::new(4);
        half.push(&[0.5; 4]);
        assert!((half.rms() - 0.5).abs() < 1e-6);

        // 3.0 clamps to 1.0, NaN counts as 0 -> sqrt(1/2)
        let mut clamped = SampleWindow::new(4);
        clamped.push(&[3.0, f32::NAN]);
        assert_eq!(clamped.peak(), 1.0);
        assert!((clamped.rms() - 0.5_f32.sqrt()).abs() < 1e-6);

        // only the window counts: 0.9 has been dropped
        let mut old = SampleWindow::new(2);
        old.push(&[0.9, 0.1, 0.2]);
        assert_eq!(old.peak(), 0.2);
        assert!((old.rms() - (0.05_f32 / 2.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn push_never_reallocates() {
        let mut w = SampleWindow::new(16);
        let identity = buffer_identity(&w);
        for i in 0..1000_u32 {
            let base = (i % 7) as f32 / 10.0;
            w.push(&[base, -base, f32::NAN]);
            w.push(&[0.5; 40]);
        }
        w.clear();
        w.push(&[0.1, 0.2]);
        assert_eq!(buffer_identity(&w), identity);
        assert_eq!(w.capacity(), 16);
    }

    #[test]
    fn partial_eq_compares_content_not_rotation() {
        let mut a = SampleWindow::new(3);
        a.push(&[0.1, 0.2, 0.3]);
        let mut b = SampleWindow::new(3);
        b.push(&[0.9, 0.9, 0.1, 0.2, 0.3]); // rotated storage, same window
        assert_eq!(a, b);
        b.push(&[0.4]);
        assert_ne!(a, b);
        let mut c = SampleWindow::new(4);
        c.push(&[0.1, 0.2, 0.3]);
        assert_ne!(a, c); // same content, different capacity
    }

    fn paint_commands(samples: Vec<f32>) -> Vec<presentar_core::draw::DrawCommand> {
        let mut widget = Waveform::new().samples(samples);
        widget.layout(Rect::new(10.0, 20.0, 100.0, 40.0));
        let mut canvas = RecordingCanvas::new();
        widget.paint(&mut canvas);
        canvas.take_commands()
    }

    #[test]
    fn waveform_paints_the_same_path_from_the_window() {
        let mut w = SampleWindow::new(4);
        w.push(&[0.1, 0.9, -0.4]);
        w.push(&[0.3, -0.8, 0.2]);
        let plain = [0.9_f32, -0.4, 0.3, -0.8, 0.2];
        let expected = paint_commands(plain[plain.len() - 4..].to_vec());
        let actual = paint_commands(w.samples());
        assert_eq!(expected.len(), 1);
        assert_eq!(actual, expected);

        // an empty window paints the same centre line as an empty slice
        let none = SampleWindow::new(4);
        assert_eq!(paint_commands(none.samples()), paint_commands(Vec::new()));
    }

    // ===== Class property: every public entry point, any f32 (PR #6897, round 3) =====
    //
    // Entry point                       covered by
    // new / capacity / len / is_empty   class_window_every_entry_point_is_total_for_any_f32
    // push                              class_window_every_entry_point_is_total_for_any_f32
    // samples / copy_into               class_window_every_entry_point_is_total_for_any_f32
    // peak / rms                        class_window_every_entry_point_is_total_for_any_f32
    // clear                             class_window_every_entry_point_is_total_for_any_f32
    // (feeds Waveform::paint)           class_window_samples_paint_finite_through_a_waveform

    /// Any f32, with the awkward ones over-represented.
    fn wild() -> impl Strategy<Value = f32> {
        prop_oneof![
            proptest::num::f32::ANY,
            Just(f32::NAN),
            Just(f32::INFINITY),
            Just(f32::NEG_INFINITY),
            Just(f32::MAX),
            Just(f32::MIN),
            Just(f32::MIN_POSITIVE),
            Just(1.0e-40_f32),
            Just(-0.0_f32),
            -1000.0_f32..1000.0,
        ]
    }

    fn unit_interval(v: f32) -> bool {
        v.is_finite() && (-1.0..=1.0).contains(&v)
    }

    proptest! {
        #[test]
        fn class_window_every_entry_point_is_total_for_any_f32(
            capacity in 0_usize..48,
            frames in proptest::collection::vec(
                proptest::collection::vec(wild(), 0..120),
                0..6,
            ),
        ) {
            let mut window = SampleWindow::new(capacity);
            for frame in &frames {
                window.push(frame);
                let held = window.samples();
                prop_assert_eq!(held.len(), window.len());
                prop_assert!(window.len() <= window.capacity());
                prop_assert!(held.iter().copied().all(unit_interval), "{held:?}");
                let peak = window.peak();
                let rms = window.rms();
                prop_assert!(peak.is_finite() && (0.0..=1.0).contains(&peak), "{peak}");
                prop_assert!(rms.is_finite() && (0.0..=1.0 + 1.0e-3).contains(&rms), "{rms}");
                prop_assert!(rms <= peak + 1.0e-3);
                let mut out = vec![f32::NAN; 3];
                window.copy_into(&mut out);
                prop_assert_eq!(out, held);
            }
            window.clear();
            prop_assert!(window.is_empty());
            prop_assert!(window.peak() == 0.0);
            prop_assert!(window.rms() == 0.0); // -0.0 is allowed: an empty f32 sum is -0.0
            prop_assert_eq!(window.capacity(), capacity);
        }

        #[test]
        fn class_window_samples_paint_finite_through_a_waveform(
            capacity in 0_usize..48,
            frame in proptest::collection::vec(wild(), 0..120),
            x in wild(),
            y in wild(),
            w in wild(),
            h in wild(),
        ) {
            let mut window = SampleWindow::new(capacity);
            window.push(&frame);
            let mut widget = Waveform::new().samples(window.samples());
            widget.layout(Rect::new(x, y, w, h));
            let mut canvas = RecordingCanvas::new();
            widget.paint(&mut canvas);
            for cmd in canvas.commands() {
                let presentar_core::draw::DrawCommand::Path { points, style, .. } = cmd else {
                    return Err(TestCaseError::fail("not a path"));
                };
                prop_assert!(style.width.is_finite() && style.width >= 0.0);
                prop_assert!(points.iter().all(|p| p.x.is_finite() && p.y.is_finite()));
            }
        }
    }
}
