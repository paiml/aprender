//! Simulation clock management.
//!
//! Handles time progression with support for:
//! - Fixed timestep mode
//! - Adaptive timestep mode (future)
//! - Time bounds and limits

use crate::engine::SimTime;
use serde::{Deserialize, Serialize};

/// Simulation clock.
///
/// Manages time progression through the simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimClock {
    /// Current simulation time.
    current: SimTime,
    /// Timestep duration in nanoseconds.
    timestep_nanos: u64,
    /// Number of steps taken.
    step_count: u64,
    /// Maximum simulation time (optional limit).
    max_time: Option<SimTime>,
}

impl SimClock {
    /// Create a new clock with the given timestep in seconds.
    ///
    /// # Panics
    ///
    /// Panics if timestep is not positive or not finite.
    #[must_use]
    pub fn new(timestep_secs: f64) -> Self {
        assert!(timestep_secs > 0.0, "Timestep must be positive");
        assert!(timestep_secs.is_finite(), "Timestep must be finite");

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let timestep_nanos = (timestep_secs * 1_000_000_000.0) as u64;

        Self {
            current: SimTime::ZERO,
            timestep_nanos,
            step_count: 0,
            max_time: None,
        }
    }

    /// Create a new clock with timestep in nanoseconds.
    #[must_use]
    pub const fn from_nanos(timestep_nanos: u64) -> Self {
        Self {
            current: SimTime::ZERO,
            timestep_nanos,
            step_count: 0,
            max_time: None,
        }
    }

    /// Get current simulation time.
    #[must_use]
    pub const fn current_time(&self) -> SimTime {
        self.current
    }

    /// Get timestep duration as seconds.
    #[must_use]
    pub fn timestep_secs(&self) -> f64 {
        contract_pre_iterator!();
        let result = self.timestep_nanos as f64 / 1_000_000_000.0;
        contract_post_configuration!(&"ok");
        result
    }

    /// Alias for `timestep_secs`.
    #[must_use]
    pub fn dt(&self) -> f64 {
        self.timestep_secs()
    }

    /// Get timestep duration in nanoseconds.
    #[must_use]
    pub const fn timestep_nanos(&self) -> u64 {
        self.timestep_nanos
    }

    /// Get number of steps taken.
    #[must_use]
    pub const fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Set maximum simulation time.
    #[allow(clippy::missing_const_for_fn)] // Mutable const not stable
    pub fn set_max_time(&mut self, max: SimTime) {
        self.max_time = Some(max);
    }

    /// Check if simulation has reached max time.
    #[must_use]
    pub fn at_max_time(&self) -> bool {
        self.max_time.is_some_and(|max| self.current >= max)
    }

    /// Advance clock by one timestep.
    ///
    /// Returns the new time.
    #[allow(clippy::missing_const_for_fn)] // Mutable const not stable
    pub fn tick(&mut self) -> SimTime {
        self.current = self.current.add_nanos(self.timestep_nanos);
        self.step_count += 1;
        self.current
    }

    /// Advance clock by multiple timesteps.
    ///
    /// Returns the new time.
    pub fn tick_n(&mut self, n: u64) -> SimTime {
        for _ in 0..n {
            self.tick();
        }
        self.current
    }

    /// Set current time (for replay/restore).
    #[allow(clippy::missing_const_for_fn)] // Mutable const not stable
    pub fn set_time(&mut self, time: SimTime) {
        self.current = time;
    }

    /// Reset clock to initial state.
    #[allow(clippy::missing_const_for_fn)] // Mutable const not stable
    pub fn reset(&mut self) {
        self.current = SimTime::ZERO;
        self.step_count = 0;
    }

    /// Calculate time until a target time.
    #[must_use]
    pub fn time_until(&self, target: SimTime) -> SimTime {
        if target > self.current {
            target - self.current
        } else {
            SimTime::ZERO
        }
    }

    /// Calculate number of steps to reach target time.
    #[must_use]
    pub fn steps_until(&self, target: SimTime) -> u64 {
        contract_pre_iterator!();
        let time_diff = self.time_until(target);
        let nanos = time_diff.as_nanos();

        if self.timestep_nanos == 0 {
            contract_post_configuration!(&"ok");
            return 0;
        }

        let result = nanos.div_ceil(self.timestep_nanos);
        contract_post_configuration!(&"ok");
        result
    }
}

impl Default for SimClock {
    fn default() -> Self {
        // Default 1ms timestep
        Self::from_nanos(1_000_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_creation() {
        let clock = SimClock::new(0.001); // 1ms

        assert_eq!(clock.current_time(), SimTime::ZERO);
        assert!((clock.timestep_secs() - 0.001).abs() < 1e-9);
        assert_eq!(clock.step_count(), 0);
    }

    #[test]
    fn test_clock_tick() {
        let mut clock = SimClock::new(0.001);

        clock.tick();
        assert_eq!(clock.step_count(), 1);
        assert!((clock.current_time().as_secs_f64() - 0.001).abs() < 1e-9);

        clock.tick();
        assert_eq!(clock.step_count(), 2);
        assert!((clock.current_time().as_secs_f64() - 0.002).abs() < 1e-9);
    }

    #[test]
    fn test_clock_tick_n() {
        let mut clock = SimClock::new(0.001);

        clock.tick_n(100);
        assert_eq!(clock.step_count(), 100);
        assert!((clock.current_time().as_secs_f64() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_clock_max_time() {
        let mut clock = SimClock::new(0.1);
        clock.set_max_time(SimTime::from_secs(0.5));

        assert!(!clock.at_max_time());

        clock.tick_n(4);
        assert!(!clock.at_max_time());

        clock.tick();
        assert!(clock.at_max_time());
    }

    #[test]
    fn test_clock_reset() {
        let mut clock = SimClock::new(0.001);

        clock.tick_n(100);
        assert!(clock.step_count() > 0);

        clock.reset();
        assert_eq!(clock.step_count(), 0);
        assert_eq!(clock.current_time(), SimTime::ZERO);
    }

    #[test]
    fn test_clock_time_until() {
        let mut clock = SimClock::new(0.001);
        clock.tick_n(10); // Now at 0.01s

        let until = clock.time_until(SimTime::from_secs(0.1));
        assert!((until.as_secs_f64() - 0.09).abs() < 1e-9);

        // Past time returns zero
        let until_past = clock.time_until(SimTime::from_secs(0.005));
        assert_eq!(until_past, SimTime::ZERO);
    }

    #[test]
    fn test_clock_steps_until() {
        let clock = SimClock::new(0.01); // 10ms steps

        // 1 second = 100 steps
        let steps = clock.steps_until(SimTime::from_secs(1.0));
        assert_eq!(steps, 100);

        // Partial step rounds up
        let steps2 = clock.steps_until(SimTime::from_secs(0.015));
        assert_eq!(steps2, 2); // 15ms needs 2 steps of 10ms
    }

    #[test]
    fn test_clock_set_time() {
        let mut clock = SimClock::new(0.001);

        clock.set_time(SimTime::from_secs(5.0));
        assert!((clock.current_time().as_secs_f64() - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_clock_from_nanos() {
        let clock = SimClock::from_nanos(1_000_000); // 1ms

        assert_eq!(clock.current_time(), SimTime::ZERO);
        assert_eq!(clock.timestep_nanos(), 1_000_000);
        assert!((clock.timestep_secs() - 0.001).abs() < 1e-9);
        assert_eq!(clock.step_count(), 0);
    }

    #[test]
    fn test_clock_default() {
        let clock = SimClock::default();

        assert_eq!(clock.timestep_nanos(), 1_000_000); // Default 1ms
        assert_eq!(clock.current_time(), SimTime::ZERO);
        assert_eq!(clock.step_count(), 0);
    }

    #[test]
    fn test_clock_steps_until_zero_timestep() {
        // Create clock with zero timestep (edge case)
        let clock = SimClock::from_nanos(0);
        let steps = clock.steps_until(SimTime::from_secs(1.0));
        assert_eq!(steps, 0); // Should handle gracefully
    }

    #[test]
    fn test_clock_at_max_time_no_max() {
        let clock = SimClock::new(0.001);
        assert!(!clock.at_max_time()); // No max set, should be false
    }

    #[test]
    fn test_clock_timestep_nanos_accessor() {
        let clock = SimClock::new(0.5); // 500ms
        assert_eq!(clock.timestep_nanos(), 500_000_000);
    }

    #[test]
    fn test_clock_tick_returns_new_time() {
        let mut clock = SimClock::new(0.1);
        let new_time = clock.tick();
        assert!((new_time.as_secs_f64() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_clock_tick_n_returns_final_time() {
        let mut clock = SimClock::new(0.1);
        let final_time = clock.tick_n(5);
        assert!((final_time.as_secs_f64() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_clock_clone() {
        let clock = SimClock::new(0.001);
        let cloned = clock.clone();
        assert_eq!(cloned.timestep_nanos(), clock.timestep_nanos());
        assert_eq!(cloned.current_time(), clock.current_time());
    }

    #[test]
    fn test_clock_debug() {
        let clock = SimClock::new(0.001);
        let debug = format!("{:?}", clock);
        assert!(debug.contains("SimClock"));
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        /// Falsification: time always increases after tick.
        #[test]
        fn prop_time_increases(timestep in 0.0001f64..1.0, ticks in 1u64..1000) {
            let mut clock = SimClock::new(timestep);
            let initial = clock.current_time();

            clock.tick_n(ticks);

            prop_assert!(clock.current_time() > initial);
        }

        /// Falsification: step count equals number of ticks.
        #[test]
        fn prop_step_count_accurate(timestep in 0.0001f64..1.0, ticks in 0u64..1000) {
            let mut clock = SimClock::new(timestep);

            clock.tick_n(ticks);

            prop_assert_eq!(clock.step_count(), ticks);
        }

        /// Falsification: time advances by correct amount.
        #[test]
        fn prop_time_advance_correct(timestep in 0.0001f64..0.1, ticks in 1u64..100) {
            let mut clock = SimClock::new(timestep);

            clock.tick_n(ticks);

            let expected = timestep * ticks as f64;
            let actual = clock.current_time().as_secs_f64();

            // Allow floating point error due to nanosecond quantization
            // The error comes from converting f64 timestep to u64 nanos and back
            let tolerance = 1e-6 * expected.max(1.0);
            prop_assert!((actual - expected).abs() < tolerance,
                "Expected {}, got {}, diff {}", expected, actual, (actual - expected).abs());
        }
    }
}
