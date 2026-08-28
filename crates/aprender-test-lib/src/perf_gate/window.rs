//! §4.4.2 termination and §4.4.7 boundary effects, as a pure state machine.
//!
//! The measurement window is a *shared admission decision*, not a per-worker
//! deadline check. Written as a deadline test inside each worker's loop, the
//! sample-count bound cannot be expressed at all (no worker knows the total),
//! and there is no single instant `T` to measure `drain_ms` from. Both of those
//! are why this is a controller rather than an `if` in the loop.
//!
//! It is deliberately clock-free: every method takes the current offset in
//! seconds. That is what lets the whole termination rule be unit-tested in
//! microseconds instead of the 60 s it governs.

use serde::{Deserialize, Serialize};

use super::protocol::{BandConfig, DRAIN_SUSPECT_FRACTION};

/// Shared admission gate for one band's closed-loop workers.
#[derive(Debug, Clone)]
pub struct WindowController {
    min_samples: usize,
    min_wall_s: f64,
    issued: usize,
    closed_at_s: Option<f64>,
    last_completion_s: f64,
    in_flight: usize,
    peak_in_flight: usize,
}

impl WindowController {
    /// Build the controller for `config`. The window opens at offset `0.0`;
    /// callers pass offsets measured from the first sampled request's origin.
    #[must_use]
    pub fn new(config: &BandConfig) -> Self {
        Self {
            min_samples: config.min_samples,
            min_wall_s: config.min_wall_clock.as_secs_f64(),
            issued: 0,
            closed_at_s: None,
            last_completion_s: 0.0,
            in_flight: 0,
            peak_in_flight: 0,
        }
    }

    /// A controller with explicit bounds, for the warmup phase (§4.4.2: exactly
    /// `2 × c` requests, no wall-clock floor) and for tests.
    #[must_use]
    pub fn with_bounds(min_samples: usize, min_wall_s: f64) -> Self {
        Self {
            min_samples,
            min_wall_s,
            issued: 0,
            closed_at_s: None,
            last_completion_s: 0.0,
            in_flight: 0,
            peak_in_flight: 0,
        }
    }

    /// §4.4.2 — "termination is whichever bound is satisfied **last**".
    ///
    /// Returns the index reserved for a new request, or `None` once the window
    /// has closed. §4.4.7: **no new request is issued at or after `T`**, and `T`
    /// is stamped here, once, at the first refusal.
    pub fn try_admit(&mut self, now_s: f64) -> Option<usize> {
        self.try_admit_with_in_flight(now_s).map(|(index, _)| index)
    }

    /// [`Self::try_admit`], also returning the in-flight count **including this
    /// request**, read under the same lock that made the admission decision.
    ///
    /// A caller that admits, unlocks, and then asks for `in_flight()` reads a
    /// number from a different instant. The per-request in-flight figure is the
    /// evidence that the client was concurrent, so it must not be a racy guess.
    pub fn try_admit_with_in_flight(&mut self, now_s: f64) -> Option<(usize, usize)> {
        if self.closed_at_s.is_some() {
            return None;
        }
        if self.issued >= self.min_samples && now_s >= self.min_wall_s {
            self.closed_at_s = Some(now_s);
            return None;
        }
        let index = self.issued;
        self.issued += 1;
        self.in_flight += 1;
        self.peak_in_flight = self.peak_in_flight.max(self.in_flight);
        Some((index, self.in_flight))
    }

    /// Record a completion at `now_s`. Returns `true` when this request
    /// completed during the drain, i.e. after the window closed.
    pub fn complete(&mut self, now_s: f64) -> bool {
        self.in_flight = self.in_flight.saturating_sub(1);
        self.last_completion_s = self.last_completion_s.max(now_s);
        self.closed_at_s.is_some_and(|t| now_s > t)
    }

    /// Peak concurrent requests the client had in flight. This is the *client's*
    /// number and is named as such: `max_in_flight` in §4.4.9 is what the
    /// **server** admitted, which only the server can report.
    #[must_use]
    pub fn peak_in_flight(&self) -> usize {
        self.peak_in_flight
    }

    /// Requests admitted so far.
    #[must_use]
    pub fn issued(&self) -> usize {
        self.issued
    }

    /// Requests currently outstanding.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }

    /// Whether the window has closed.
    #[must_use]
    pub fn is_closed(&self) -> bool {
        self.closed_at_s.is_some()
    }

    /// Finalise the band. `window_close_s` defaults to the last completion when
    /// the window never closed (a run cut short), which yields `drain_ms = 0`
    /// rather than a negative number.
    #[must_use]
    pub fn report(&self) -> WindowReport {
        let close = self.closed_at_s.unwrap_or(self.last_completion_s);
        let drain_ms = ((self.last_completion_s - close).max(0.0)) * 1000.0;
        let window_ms = close * 1000.0;
        let mut suspect = Vec::new();
        if window_ms > 0.0 && drain_ms > DRAIN_SUSPECT_FRACTION * window_ms {
            suspect.push(format!(
                "§4.4.7 drain_ms={drain_ms:.1} > 0.5 x window_ms={window_ms:.1}: one request \
                 dominated the window; re-run this band with a longer window"
            ));
        }
        if self.closed_at_s.is_none() {
            suspect.push(
                "§4.4.2 the window never closed: neither termination bound was reached, so this \
                 band did not run the protocol"
                    .to_string(),
            );
        }
        WindowReport {
            requested: self.issued,
            window_ms,
            drain_ms,
            client_peak_in_flight: self.peak_in_flight,
            suspect,
        }
    }
}

/// What the controller observed, for the receipt.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WindowReport {
    /// Requests admitted before `T`.
    pub requested: usize,
    /// `T` minus window open, in milliseconds.
    pub window_ms: f64,
    /// §4.4.7 — last drained completion minus `T`, in milliseconds.
    pub drain_ms: f64,
    /// Peak concurrent requests **the client** had outstanding.
    pub client_peak_in_flight: usize,
    /// §4.4.7 `SUSPECT` annotations, empty when the band is clean.
    pub suspect: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// The sample bound alone must not close the window: §4.4.2 says the last
    /// bound wins. A fast host that hit 30 samples in 2 s still owes 60 s.
    #[test]
    fn sample_bound_alone_does_not_close_the_window() {
        let mut w = WindowController::with_bounds(3, 60.0);
        for i in 0..3 {
            assert_eq!(w.try_admit(0.1 * f64::from(i)), Some(i as usize));
        }
        assert_eq!(w.try_admit(0.4), Some(3), "still open: 0.4s < 60s");
        assert!(!w.is_closed());
    }

    /// And the wall-clock bound alone must not close it either: a slow host that
    /// burned 60 s on 4 requests still owes `max(30, 8c)` samples.
    #[test]
    fn wall_clock_bound_alone_does_not_close_the_window() {
        let mut w = WindowController::with_bounds(30, 1.0);
        for _ in 0..4 {
            assert!(
                w.try_admit(100.0).is_some(),
                "still open: only 4 of 30 samples"
            );
        }
        assert!(!w.is_closed());
    }

    #[test]
    fn window_closes_only_when_both_bounds_are_satisfied() {
        let mut w = WindowController::with_bounds(3, 10.0);
        assert!(w.try_admit(0.0).is_some());
        assert!(w.try_admit(1.0).is_some());
        assert!(w.try_admit(2.0).is_some());
        assert!(!w.is_closed());
        assert_eq!(w.try_admit(10.0), None, "3 samples AND 10s -> closed");
        assert!(w.is_closed());
    }

    /// §4.4.7 — once closed, the gate stays closed. A late worker must not
    /// sneak a request in after `T`.
    #[test]
    fn no_new_request_is_admitted_at_or_after_t() {
        let mut w = WindowController::with_bounds(1, 1.0);
        assert!(w.try_admit(0.0).is_some());
        assert_eq!(w.try_admit(1.0), None);
        assert_eq!(w.try_admit(1.0001), None);
        assert_eq!(w.try_admit(500.0), None);
        assert_eq!(w.issued(), 1, "exactly one request was ever admitted");
    }

    /// The drain is measured from `T`, not from the last admission.
    #[test]
    fn drain_ms_is_measured_from_window_close() {
        let mut w = WindowController::with_bounds(2, 4.0);
        assert!(w.try_admit(0.0).is_some());
        assert!(w.try_admit(1.0).is_some());
        assert!(!w.complete(2.0), "completed inside the window");
        assert_eq!(w.try_admit(4.0), None, "T = 4.0");
        assert!(w.complete(5.5), "completed during the drain");
        let r = w.report();
        assert!((r.window_ms - 4000.0).abs() < 1e-9, "{}", r.window_ms);
        assert!((r.drain_ms - 1500.0).abs() < 1e-9, "{}", r.drain_ms);
        assert!(r.suspect.is_empty(), "{:?}", r.suspect);
    }

    /// §4.4.7 — `drain_ms > 0.5 x window` is annotated SUSPECT.
    #[test]
    fn a_dominating_request_is_annotated_suspect() {
        let mut w = WindowController::with_bounds(1, 2.0);
        assert!(w.try_admit(0.0).is_some());
        assert_eq!(w.try_admit(2.0), None);
        assert!(w.complete(20.0));
        let r = w.report();
        assert!((r.drain_ms - 18000.0).abs() < 1e-9);
        assert_eq!(r.suspect.len(), 1, "{:?}", r.suspect);
        assert!(r.suspect[0].contains("drain_ms"));
    }

    /// A band whose window never closed did not run the protocol, and must say
    /// so rather than reporting a clean `drain_ms = 0`.
    #[test]
    fn an_unclosed_window_is_suspect_not_clean() {
        let mut w = WindowController::with_bounds(100, 60.0);
        assert!(w.try_admit(0.0).is_some());
        assert!(!w.complete(1.0));
        let r = w.report();
        assert_eq!(r.drain_ms, 0.0);
        assert!(
            r.suspect.iter().any(|s| s.contains("never closed")),
            "{:?}",
            r.suspect
        );
    }

    /// The controller is the only place that knows how many requests were in
    /// flight at once, and it is the evidence that `c` workers really overlapped.
    #[test]
    fn peak_in_flight_tracks_concurrent_admissions() {
        let mut w = WindowController::with_bounds(100, 100.0);
        for _ in 0..8 {
            assert!(w.try_admit(0.0).is_some());
        }
        assert_eq!(w.in_flight(), 8);
        assert_eq!(w.peak_in_flight(), 8);
        for _ in 0..8 {
            w.complete(1.0);
        }
        assert_eq!(w.in_flight(), 0);
        assert_eq!(w.peak_in_flight(), 8, "peak is a high-water mark");
    }

    #[test]
    fn controller_reads_its_bounds_from_the_band_config() {
        let cfg = BandConfig::conformant(8);
        let w = WindowController::new(&cfg);
        assert_eq!(w.min_samples, 64, "max(30, 8*8)");
        assert!((w.min_wall_s - 60.0).abs() < 1e-9);
        assert_eq!(cfg.quiesce, Duration::from_secs(5));
    }
}

/// A closed-loop concurrency proof over real OS threads.
///
/// Not `#[cfg(test)]`: this is the falsifier for the defect class that produced
/// this ticket — a client that *says* concurrency `c` and issues requests one at
/// a time. It is exported so any harness can run it against its own admission
/// path, and it is exercised by [`concurrency_proof_tests`] under default
/// features, where CI can actually see it.
///
/// `c` threads each loop: admit, "work" for `work` , complete. Returns the
/// observed peak in flight and the wall time. A serialising implementation
/// yields `peak == 1` and `wall ≈ requests × work`; a concurrent one yields
/// `peak == c` and `wall ≈ requests / c × work`.
#[must_use]
pub fn closed_loop_probe(
    concurrency: usize,
    requests: usize,
    work: std::time::Duration,
) -> (usize, std::time::Duration) {
    use std::sync::Mutex;
    use std::time::Instant;

    let controller = Mutex::new(WindowController::with_bounds(requests, 0.0));
    let origin = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..concurrency {
            scope.spawn(|| loop {
                let admitted = {
                    let mut c = controller
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner);
                    c.try_admit(origin.elapsed().as_secs_f64())
                };
                if admitted.is_none() {
                    break;
                }
                std::thread::sleep(work);
                let mut c = controller
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                c.complete(origin.elapsed().as_secs_f64());
            });
        }
    });
    let peak = controller
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .peak_in_flight();
    (peak, origin.elapsed())
}

#[cfg(test)]
mod concurrency_proof_tests {
    use super::*;
    use std::time::Duration;

    /// The direct claim: `c` workers are in flight at the same time.
    #[test]
    fn eight_workers_are_actually_concurrent() {
        let (peak, _) = closed_loop_probe(8, 64, Duration::from_millis(20));
        assert_eq!(
            peak, 8,
            "peak in-flight must reach c; a serialising client gives 1"
        );
    }

    /// The indirect claim, which a fake concurrent client cannot also satisfy:
    /// the same request count must finish far faster at c=8 than at c=1.
    ///
    /// The bound is deliberately loose (4x, not 8x) so scheduler noise on a
    /// loaded CI runner cannot red it, while a secretly-sequential client — which
    /// would score 1.0x — still fails by a wide margin.
    #[test]
    fn wall_time_at_c8_is_not_eight_times_the_c1_time() {
        let requests = 32;
        let work = Duration::from_millis(20);
        let (peak1, wall1) = closed_loop_probe(1, requests, work);
        let (peak8, wall8) = closed_loop_probe(8, requests, work);

        assert_eq!(peak1, 1);
        assert_eq!(peak8, 8);
        let speedup = wall1.as_secs_f64() / wall8.as_secs_f64();
        eprintln!(
            "closed_loop_probe: {requests} requests x {work:?} -- \
             c=1 peak={peak1} wall={wall1:?}; c=8 peak={peak8} wall={wall8:?}; speedup={speedup:.2}x"
        );
        assert!(
            speedup > 4.0,
            "c=1 took {wall1:?}, c=8 took {wall8:?} (speedup {speedup:.2}x); \
             a concurrent client must be much faster, a sequential one scores ~1.0x"
        );
    }

    /// And the probe can tell the difference — a c=1 run is the negative control.
    #[test]
    fn the_probe_reports_one_for_a_sequential_client() {
        let (peak, wall) = closed_loop_probe(1, 8, Duration::from_millis(10));
        assert_eq!(peak, 1);
        assert!(
            wall >= Duration::from_millis(70),
            "8 x 10ms sequential must take ~80ms, got {wall:?}"
        );
    }
}
