//! Circuit Breaker Pattern
//!
//! AWP-02: Protect against cascading failures with three-state circuit breaker.

use std::time::{Duration, Instant};

// ----------------------------------------------------------------------------
// AWP-02: Circuit Breaker
// ----------------------------------------------------------------------------

/// Circuit breaker states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Circuit is closed (normal operation)
    Closed,
    /// Circuit is open (failing fast)
    Open,
    /// Circuit is half-open (testing recovery)
    HalfOpen,
}

/// Circuit breaker for protecting against cascading failures.
///
/// # Example
/// ```rust
/// use trueno::brick::CircuitBreaker;
/// use std::time::Duration;
///
/// let mut breaker = CircuitBreaker::new(3, Duration::from_secs(30));
///
/// // Record failures
/// breaker.record_failure();
/// breaker.record_failure();
/// assert!(breaker.allow_request()); // Still closed
///
/// breaker.record_failure();
/// assert!(!breaker.allow_request()); // Now open
/// ```
pub struct CircuitBreaker {
    /// Current state
    state: CircuitState,
    /// Failure count in current window
    failure_count: usize,
    /// Failure threshold to trip the circuit
    failure_threshold: usize,
    /// Time when circuit opened
    opened_at: Option<Instant>,
    /// Duration to stay open before trying half-open
    open_duration: Duration,
    /// Success count in half-open state
    half_open_successes: usize,
    /// Successes needed to close from half-open
    half_open_threshold: usize,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(failure_threshold: usize, open_duration: Duration) -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            failure_threshold,
            opened_at: None,
            open_duration,
            half_open_successes: 0,
            half_open_threshold: 1,
        }
    }

    /// Get current state.
    #[must_use]
    pub fn state(&self) -> CircuitState {
        self.state
    }

    /// Check if a request should be allowed.
    #[must_use]
    pub fn allow_request(&mut self) -> bool {
        match self.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                // Check if we should transition to half-open
                if let Some(opened_at) = self.opened_at {
                    if opened_at.elapsed() >= self.open_duration {
                        self.state = CircuitState::HalfOpen;
                        self.half_open_successes = 0;
                        return true; // Allow one request to test
                    }
                }
                false
            }
            CircuitState::HalfOpen => true, // Allow requests in half-open
        }
    }

    /// Record a successful operation.
    pub fn record_success(&mut self) {
        match self.state {
            CircuitState::Closed => {
                // Reset failure count on success
                self.failure_count = 0;
            }
            CircuitState::HalfOpen => {
                self.half_open_successes += 1;
                if self.half_open_successes >= self.half_open_threshold {
                    // Recovered - close the circuit
                    self.state = CircuitState::Closed;
                    self.failure_count = 0;
                    self.opened_at = None;
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed operation.
    pub fn record_failure(&mut self) {
        match self.state {
            CircuitState::Closed => {
                self.failure_count += 1;
                if self.failure_count >= self.failure_threshold {
                    // Trip the circuit
                    self.state = CircuitState::Open;
                    self.opened_at = Some(Instant::now());
                }
            }
            CircuitState::HalfOpen => {
                // Failed during recovery - reopen
                self.state = CircuitState::Open;
                self.opened_at = Some(Instant::now());
            }
            CircuitState::Open => {}
        }
    }

    /// Reset the circuit breaker to closed state.
    pub fn reset(&mut self) {
        self.state = CircuitState::Closed;
        self.failure_count = 0;
        self.opened_at = None;
        self.half_open_successes = 0;
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(5, Duration::from_secs(30))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_state_eq() {
        assert_eq!(CircuitState::Closed, CircuitState::Closed);
        assert_eq!(CircuitState::Open, CircuitState::Open);
        assert_eq!(CircuitState::HalfOpen, CircuitState::HalfOpen);
        assert_ne!(CircuitState::Closed, CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_new() {
        let cb = CircuitBreaker::new(5, Duration::from_secs(30));
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count, 0);
        assert_eq!(cb.failure_threshold, 5);
    }

    #[test]
    fn test_circuit_breaker_default() {
        let cb = CircuitBreaker::default();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_threshold, 5);
        assert_eq!(cb.open_duration, Duration::from_secs(30));
    }

    #[test]
    fn test_circuit_breaker_closed_allows_requests() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(30));
        assert!(cb.allow_request());
        assert!(cb.allow_request());
        assert!(cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_trips_on_failures() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(30));

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);

        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_open_blocks_requests() {
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(30));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_success_resets_failures() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(30));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.failure_count, 2);

        cb.record_success();
        assert_eq!(cb.failure_count, 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_transition() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(10));

        // Trip the circuit
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Wait for open duration
        std::thread::sleep(Duration::from_millis(15));

        // Should transition to half-open on allow_request
        assert!(cb.allow_request());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_half_open_success_closes() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(10));

        // Trip and wait for half-open
        cb.record_failure();
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(15));
        let _ = cb.allow_request(); // Transition to half-open

        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Success should close the circuit
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_half_open_failure_reopens() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(10));

        // Trip and wait for half-open
        cb.record_failure();
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(15));
        let _ = cb.allow_request(); // Transition to half-open

        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Failure should reopen
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(30));

        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count, 0);
    }

    /// FALSIFICATION TEST: Circuit MUST trip at exact threshold
    ///
    /// The circuit breaker must trip at exactly failure_threshold failures,
    /// not before and not after. Off-by-one errors are common.
    #[test]
    fn test_falsify_exact_threshold_trip() {
        for threshold in 1..=10 {
            let mut cb = CircuitBreaker::new(threshold, Duration::from_secs(30));

            // Record threshold-1 failures - should NOT trip
            for i in 0..(threshold - 1) {
                cb.record_failure();
                assert_eq!(
                    cb.state(),
                    CircuitState::Closed,
                    "FALSIFICATION FAILED: Circuit tripped after {} failures (threshold={})",
                    i + 1,
                    threshold
                );
            }

            // The threshold-th failure MUST trip the circuit
            cb.record_failure();
            assert_eq!(
                cb.state(),
                CircuitState::Open,
                "FALSIFICATION FAILED: Circuit did NOT trip at exact threshold={}",
                threshold
            );
        }
    }

    /// FALSIFICATION TEST: Open circuit MUST block all requests
    ///
    /// While open (and before timeout), the circuit MUST block 100% of requests.
    /// Any leak would cause cascading failures.
    #[test]
    fn test_falsify_open_blocks_all() {
        let mut cb = CircuitBreaker::new(1, Duration::from_secs(30));

        // Trip immediately
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // Try many requests - ALL must be blocked
        let mut allowed = 0;
        for _ in 0..1000 {
            if cb.allow_request() {
                allowed += 1;
            }
        }

        assert_eq!(
            allowed, 0,
            "FALSIFICATION FAILED: {} requests leaked through open circuit",
            allowed
        );
    }

    /// FALSIFICATION TEST: Half-open failure MUST immediately reopen
    ///
    /// A single failure in half-open state must immediately reopen the circuit.
    /// If it doesn't, the system could repeatedly hammer a failing service.
    #[test]
    fn test_falsify_half_open_single_failure_reopens() {
        let mut cb = CircuitBreaker::new(1, Duration::from_millis(5));

        // Trip and wait for half-open
        cb.record_failure();
        std::thread::sleep(Duration::from_millis(10));
        let _ = cb.allow_request(); // Transition to half-open

        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // Single failure MUST reopen
        cb.record_failure();

        assert_eq!(
            cb.state(),
            CircuitState::Open,
            "FALSIFICATION FAILED: Half-open did not reopen after failure"
        );

        // Verify opened_at was updated (circuit should wait again)
        assert!(cb.opened_at.is_some(), "FALSIFICATION FAILED: opened_at not set after reopen");
    }

    /// FALSIFICATION TEST: State machine must be deterministic
    ///
    /// Same sequence of events must always produce same state.
    /// Non-determinism would make the system unpredictable.
    #[test]
    fn test_falsify_state_machine_determinism() {
        // Run the same sequence multiple times
        for _ in 0..10 {
            let mut cb = CircuitBreaker::new(3, Duration::from_millis(5));

            // Sequence: F, S, F, F, F, wait, S
            cb.record_failure();
            assert_eq!(cb.state(), CircuitState::Closed);

            cb.record_success();
            assert_eq!(cb.failure_count, 0); // Reset

            cb.record_failure();
            cb.record_failure();
            cb.record_failure();
            assert_eq!(cb.state(), CircuitState::Open);

            std::thread::sleep(Duration::from_millis(10));
            let _ = cb.allow_request();
            assert_eq!(cb.state(), CircuitState::HalfOpen);

            cb.record_success();
            assert_eq!(cb.state(), CircuitState::Closed);
        }
    }

    /// FALSIFICATION TEST: Failures in Open state are no-ops
    ///
    /// Recording failures while Open should not affect anything.
    /// This prevents "double-counting" failures.
    #[test]
    fn test_falsify_open_ignores_failures() {
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(30));

        // Trip the circuit
        cb.record_failure();
        cb.record_failure();
        let opened_at = cb.opened_at;

        // Record more failures while open
        for _ in 0..100 {
            cb.record_failure();
        }

        // State should still be Open, opened_at unchanged
        assert_eq!(cb.state(), CircuitState::Open);
        assert_eq!(
            cb.opened_at, opened_at,
            "FALSIFICATION FAILED: opened_at changed while in Open state"
        );
    }
}
