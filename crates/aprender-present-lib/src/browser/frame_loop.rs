//! The frame loop's bookkeeping, with no browser API: a clock that turns
//! `requestAnimationFrame` timestamps into frame deltas, and a loop state machine that owns
//! "is a frame requested", "was it stopped" and "how many ticks ran" (aprender#6892).
//!
//! The wasm32 shell in `browser::entry` implements `FrameScheduler` over
//! `requestAnimationFrame` / `cancelAnimationFrame` and is a thin layer; everything that
//! decides lives here and is tested natively.
//!
//! Contract: `contracts/present-frame-loop-v1.yaml`.

/// Asks the host for one callback at the next frame, and withdraws it.
pub trait FrameScheduler {
    /// Request one tick. `None` means the host refused (no window); nothing is pending.
    fn request(&mut self) -> Option<i32>;
    /// Withdraw a pending request by its id.
    fn cancel(&mut self, id: i32);
}

/// Turns host timestamps (milliseconds) into frame deltas that are never negative.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FrameClock {
    last: Option<f64>,
}

impl FrameClock {
    /// A clock that has seen no frame.
    #[must_use]
    pub const fn new() -> Self {
        Self { last: None }
    }

    /// Record a frame at `now` and return the milliseconds since the previous one. The first
    /// frame has delta 0. A timestamp that is not finite or goes backwards gives 0 and does
    /// not move the clock, so the delta is non-negative and the recorded time is monotone.
    pub fn tick(&mut self, now: f64) -> f64 {
        if !now.is_finite() {
            return 0.0;
        }
        match self.last {
            Some(last) if now >= last => {
                self.last = Some(now);
                now - last
            }
            Some(_) => 0.0,
            None => {
                self.last = Some(now);
                0.0
            }
        }
    }
}

/// The loop: at most one request in flight, and nothing after `stop`.
#[derive(Debug)]
pub struct FrameLoop<S: FrameScheduler> {
    scheduler: S,
    clock: FrameClock,
    pending: Option<i32>,
    stopped: bool,
    ticks: u64,
}

impl<S: FrameScheduler> FrameLoop<S> {
    /// A loop that is idle: nothing requested yet.
    #[must_use]
    pub const fn new(scheduler: S) -> Self {
        Self {
            scheduler,
            clock: FrameClock::new(),
            pending: None,
            stopped: false,
            ticks: 0,
        }
    }

    /// The host side, for the shell to configure.
    pub fn scheduler_mut(&mut self) -> &mut S {
        &mut self.scheduler
    }

    /// Whether a frame is requested and not yet delivered.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.pending.is_some()
    }

    /// Whether `stop` was called.
    #[must_use]
    pub const fn is_stopped(&self) -> bool {
        self.stopped
    }

    /// Ticks that ran (were not dropped by `stop`).
    #[must_use]
    pub const fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Start the loop if it is idle, live and a frame is owed. Never a second request.
    pub fn kick(&mut self, frame_owed: bool) {
        if self.stopped || self.pending.is_some() || !frame_owed {
            return;
        }
        self.pending = self.scheduler.request();
    }

    /// A host callback arrived at `now`. Returns the frame delta in ms, or `None` when the
    /// loop was stopped (no tick runs after `stop`).
    pub fn begin_tick(&mut self, now: f64) -> Option<f64> {
        self.pending = None;
        if self.stopped {
            return None;
        }
        self.ticks += 1;
        Some(self.clock.tick(now))
    }

    /// The tick is done. Ask for another frame only if `wants_next` and the loop is live.
    /// When the loop ends up idle (nothing owed, or the request was refused) the clock is
    /// reset, so the first tick after the gap follows the first-frame rule (delta 0) rather
    /// than spanning the idle time.
    pub fn end_tick(&mut self, wants_next: bool) {
        if wants_next && !self.stopped && self.pending.is_none() {
            self.pending = self.scheduler.request();
        }
        if self.pending.is_none() {
            self.clock = FrameClock::new();
        }
    }

    /// Stop for good: withdraw the pending request. Idempotent. Stop is final; build a new
    /// loop to restart. Later `kick`s request nothing and no tick runs, so state changes
    /// after `stop` are never repainted.
    pub fn stop(&mut self) {
        self.stopped = true;
        if let Some(id) = self.pending.take() {
            self.scheduler.cancel(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct Fake {
        next: i32,
        requested: Vec<i32>,
        cancelled: Vec<i32>,
        refuse: bool,
    }

    impl FrameScheduler for Fake {
        fn request(&mut self) -> Option<i32> {
            if self.refuse {
                return None;
            }
            self.next += 1;
            self.requested.push(self.next);
            Some(self.next)
        }
        fn cancel(&mut self, id: i32) {
            self.cancelled.push(id);
        }
    }

    #[test]
    fn clock_first_frame_has_zero_delta_then_differences() {
        let mut c = FrameClock::new();
        assert_eq!(c.tick(100.0), 0.0);
        assert_eq!(c.tick(116.5), 16.5);
        assert_eq!(c.tick(116.5), 0.0);
    }

    #[test]
    fn clock_never_returns_a_negative_delta_and_ignores_backward_time() {
        let mut c = FrameClock::new();
        c.tick(100.0);
        assert_eq!(c.tick(50.0), 0.0);
        assert_eq!(c.tick(f64::NAN), 0.0);
        assert_eq!(c.tick(f64::INFINITY), 0.0);
        // the clock did not move backwards: 110 is 10 after 100
        assert_eq!(c.tick(110.0), 10.0);
    }

    #[test]
    fn kick_requests_once_and_only_when_a_frame_is_owed() {
        let mut l = FrameLoop::new(Fake::default());
        l.kick(false);
        assert!(!l.is_running());
        l.kick(true);
        l.kick(true);
        assert!(l.is_running());
        assert_eq!(l.scheduler_mut().requested, vec![1]);
    }

    #[test]
    fn a_refused_request_leaves_the_loop_idle() {
        let mut l = FrameLoop::new(Fake {
            refuse: true,
            ..Fake::default()
        });
        l.kick(true);
        assert!(!l.is_running());
        l.end_tick(true);
        assert!(!l.is_running());
    }

    #[test]
    fn ticks_continue_while_wanted_and_the_loop_stops_asking_otherwise() {
        let mut l = FrameLoop::new(Fake::default());
        l.kick(true);
        assert_eq!(l.begin_tick(0.0), Some(0.0));
        l.end_tick(true);
        assert_eq!(l.begin_tick(16.0), Some(16.0));
        l.end_tick(false);
        assert!(!l.is_running());
        assert_eq!(l.ticks(), 2);
        assert_eq!(l.scheduler_mut().requested.len(), 2);
        l.kick(true);
        assert!(l.is_running());
    }

    #[test]
    fn stop_cancels_the_pending_frame_and_no_tick_runs_after() {
        let mut l = FrameLoop::new(Fake::default());
        l.kick(true);
        l.stop();
        l.stop();
        assert!(l.is_stopped());
        assert!(!l.is_running());
        assert_eq!(l.scheduler_mut().cancelled, vec![1]);
        // a callback already in flight is dropped
        assert_eq!(l.begin_tick(5.0), None);
        l.end_tick(true);
        l.kick(true);
        assert_eq!(l.ticks(), 0);
        assert_eq!(l.scheduler_mut().requested.len(), 1);
    }

    #[test]
    fn stop_inside_a_tick_prevents_the_next_request() {
        let mut l = FrameLoop::new(Fake::default());
        l.kick(true);
        assert!(l.begin_tick(0.0).is_some());
        l.stop();
        l.end_tick(true);
        assert!(!l.is_running());
        assert_eq!(l.scheduler_mut().requested.len(), 1);
        assert!(l.scheduler_mut().cancelled.is_empty());
    }

    #[test]
    fn the_first_tick_after_an_idle_gap_has_zero_delta() {
        let mut l = FrameLoop::new(Fake::default());
        l.kick(true);
        assert_eq!(l.begin_tick(0.0), Some(0.0));
        l.end_tick(true);
        assert_eq!(l.begin_tick(1000.0), Some(1000.0));
        l.end_tick(false);
        l.kick(true);
        assert_eq!(l.begin_tick(60016.0), Some(0.0));
        l.end_tick(true);
        assert_eq!(l.begin_tick(60032.0), Some(16.0));
    }

    #[test]
    fn a_kick_during_a_tick_never_makes_two_requests_for_one_frame() {
        let mut l = FrameLoop::new(Fake::default());
        l.kick(true);
        assert!(l.begin_tick(0.0).is_some());
        l.kick(true);
        assert_eq!(l.scheduler_mut().requested.len(), 2);
        l.end_tick(true);
        assert_eq!(l.scheduler_mut().requested.len(), 2);
        assert!(l.is_running());
    }

    #[test]
    fn after_stop_a_later_kick_requests_nothing_and_no_tick_runs() {
        let mut l = FrameLoop::new(Fake::default());
        l.kick(true);
        assert!(l.begin_tick(0.0).is_some());
        l.end_tick(false);
        l.stop();
        let before = l.scheduler_mut().requested.len();
        l.kick(true);
        l.end_tick(true);
        assert_eq!(l.scheduler_mut().requested.len(), before);
        assert!(!l.is_running());
        assert_eq!(l.begin_tick(99.0), None);
        assert_eq!(l.ticks(), 1);
    }
}
