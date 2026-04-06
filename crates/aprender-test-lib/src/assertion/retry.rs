//! Retry Assertions with Polling (Feature 18)
//!
//! Auto-retrying assertions for eventually-consistent UI states.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application:
//! - **Heijunka**: Consistent polling intervals for predictable test timing
//! - **Jidoka**: Fail with comprehensive error messages after timeout

use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::time::{Duration, Instant};

/// Result of a retry assertion check
#[derive(Debug, Clone)]
pub enum AssertionCheckResult {
    /// Assertion passed
    Pass,
    /// Assertion failed with message
    Fail(String),
}

impl AssertionCheckResult {
    /// Check if the result is a pass
    #[must_use]
    pub const fn is_pass(&self) -> bool {
        matches!(self, Self::Pass)
    }

    /// Check if the result is a fail
    #[must_use]
    pub const fn is_fail(&self) -> bool {
        matches!(self, Self::Fail(_))
    }
}

/// Configuration for retry behavior
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Total timeout duration
    pub timeout: Duration,
    /// Interval between retry attempts
    pub poll_interval: Duration,
    /// Maximum number of retries (0 = unlimited within timeout)
    pub max_retries: usize,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(5),
            poll_interval: Duration::from_millis(100),
            max_retries: 0,
        }
    }
}

impl RetryConfig {
    /// Create a new retry config with timeout
    #[must_use]
    pub const fn new(timeout: Duration) -> Self {
        contract_pre_retry_assertion!();
        Self {
            timeout,
            poll_interval: Duration::from_millis(100),
            max_retries: 0,
        }
    }

    /// Set the poll interval
    #[must_use]
    pub const fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Set maximum retries
    #[must_use]
    pub const fn with_max_retries(mut self, max: usize) -> Self {
        self.max_retries = max;
        self
    }

    /// Create a fast config (short timeout, fast polling)
    #[must_use]
    pub const fn fast() -> Self {
        Self {
            timeout: Duration::from_millis(500),
            poll_interval: Duration::from_millis(50),
            max_retries: 0,
        }
    }

    /// Create a slow config (long timeout, slower polling)
    #[must_use]
    pub const fn slow() -> Self {
        Self {
            timeout: Duration::from_secs(30),
            poll_interval: Duration::from_millis(500),
            max_retries: 0,
        }
    }
}

/// A retry assertion that polls until success or timeout
///
/// ## Example
///
/// ```ignore
/// let assertion = RetryAssertion::new(|| {
///     if some_condition() {
///         AssertionCheckResult::Pass
///     } else {
///         AssertionCheckResult::Fail("condition not met".into())
///     }
/// });
///
/// assertion.with_timeout(Duration::from_secs(5))
///     .verify()?;
/// ```
pub struct RetryAssertion<F>
where
    F: Fn() -> AssertionCheckResult,
{
    check: F,
    config: RetryConfig,
    description: Option<String>,
}

impl<F> RetryAssertion<F>
where
    F: Fn() -> AssertionCheckResult,
{
    /// Create a new retry assertion
    #[must_use]
    pub fn new(check: F) -> Self {
        Self {
            check,
            config: RetryConfig::default(),
            description: None,
        }
    }

    /// Set the timeout duration
    #[must_use]
    pub const fn with_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Set the poll interval
    #[must_use]
    pub const fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.config.poll_interval = interval;
        self
    }

    /// Set the maximum number of retries
    #[must_use]
    pub const fn with_max_retries(mut self, max: usize) -> Self {
        self.config.max_retries = max;
        self
    }

    /// Set a description for the assertion
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the full config
    #[must_use]
    pub const fn with_config(mut self, config: RetryConfig) -> Self {
        self.config = config;
        self
    }

    /// Get the current config
    #[must_use]
    pub const fn config(&self) -> &RetryConfig {
        &self.config
    }

    /// Verify the assertion, retrying until success or timeout
    ///
    /// # Errors
    ///
    /// Returns error if the assertion fails after all retries
    #[allow(unused_assignments)]
    pub fn verify(&self) -> Result<RetryResult, RetryError> {
        contract_pre_retry_assertion!();
        let start = Instant::now();
        let mut attempts = 0;
        let mut last_error: Option<String> = None;

        loop {
            attempts += 1;

            match (self.check)() {
                AssertionCheckResult::Pass => {
                    return Ok(RetryResult {
                        attempts,
                        duration: start.elapsed(),
                    });
                }
                AssertionCheckResult::Fail(msg) => {
                    last_error = Some(msg);
                }
            }

            // Check timeout
            if start.elapsed() >= self.config.timeout {
                return Err(RetryError {
                    message: last_error.unwrap_or_default(),
                    attempts,
                    duration: start.elapsed(),
                    description: self.description.clone(),
                });
            }

            // Check max retries
            if self.config.max_retries > 0 && attempts >= self.config.max_retries {
                return Err(RetryError {
                    message: last_error.unwrap_or_default(),
                    attempts,
                    duration: start.elapsed(),
                    description: self.description.clone(),
                });
            }

            // Wait before next attempt
            std::thread::sleep(self.config.poll_interval);
        }
    }

    /// Verify the assertion once without retrying
    ///
    /// # Errors
    ///
    /// Returns error if the assertion fails
    pub fn verify_once(&self) -> Result<(), RetryError> {
        match (self.check)() {
            AssertionCheckResult::Pass => Ok(()),
            AssertionCheckResult::Fail(msg) => Err(RetryError {
                message: msg,
                attempts: 1,
                duration: Duration::ZERO,
                description: self.description.clone(),
            }),
        }
    }
}

impl<F> Debug for RetryAssertion<F>
where
    F: Fn() -> AssertionCheckResult,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetryAssertion")
            .field("config", &self.config)
            .field("description", &self.description)
            .finish()
    }
}

/// Result of a successful retry assertion
#[derive(Debug, Clone, Copy)]
pub struct RetryResult {
    /// Number of attempts before success
    pub attempts: usize,
    /// Total duration of all attempts
    pub duration: Duration,
}

/// Error when retry assertion fails
#[derive(Debug, Clone)]
pub struct RetryError {
    /// Last failure message
    pub message: String,
    /// Number of attempts made
    pub attempts: usize,
    /// Total duration of all attempts
    pub duration: Duration,
    /// Description of the assertion
    pub description: Option<String>,
}

impl std::fmt::Display for RetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref desc) = self.description {
            write!(f, "{desc}: ")?;
        }
        write!(
            f,
            "assertion failed after {} attempt(s) ({:.2}s): {}",
            self.attempts,
            self.duration.as_secs_f64(),
            self.message
        )
    }
}

impl std::error::Error for RetryError {}

// ============================================================================
// Builder helpers for common assertion patterns
// ============================================================================

/// Helper to create retry assertions for equality checks
pub fn retry_eq<T: PartialEq + Debug + Clone + 'static>(
    get_actual: impl Fn() -> T + 'static,
    expected: T,
) -> RetryAssertion<impl Fn() -> AssertionCheckResult> {
    let expected = expected;
    RetryAssertion::new(move || {
        let actual = get_actual();
        if actual == expected {
            AssertionCheckResult::Pass
        } else {
            AssertionCheckResult::Fail(format!("expected {expected:?}, got {actual:?}"))
        }
    })
}

/// Helper to create retry assertions for boolean conditions
pub fn retry_true(
    check: impl Fn() -> bool + 'static,
    message: impl Into<String>,
) -> RetryAssertion<impl Fn() -> AssertionCheckResult> {
    let message = message.into();
    RetryAssertion::new(move || {
        if check() {
            AssertionCheckResult::Pass
        } else {
            AssertionCheckResult::Fail(message.clone())
        }
    })
}

/// Helper to create retry assertions for `Option::is_some`
pub fn retry_some<T>(
    get_opt: impl Fn() -> Option<T> + 'static,
) -> RetryAssertion<impl Fn() -> AssertionCheckResult> {
    RetryAssertion::new(move || {
        if get_opt().is_some() {
            AssertionCheckResult::Pass
        } else {
            AssertionCheckResult::Fail("expected Some, got None".into())
        }
    })
}

/// Helper to create retry assertions for `Option::is_none`
pub fn retry_none<T>(
    get_opt: impl Fn() -> Option<T> + 'static,
) -> RetryAssertion<impl Fn() -> AssertionCheckResult> {
    RetryAssertion::new(move || {
        if get_opt().is_none() {
            AssertionCheckResult::Pass
        } else {
            AssertionCheckResult::Fail("expected None, got Some".into())
        }
    })
}

/// Helper to create retry assertions for string contains
pub fn retry_contains(
    get_haystack: impl Fn() -> String + 'static,
    needle: impl Into<String>,
) -> RetryAssertion<impl Fn() -> AssertionCheckResult> {
    let needle = needle.into();
    RetryAssertion::new(move || {
        let haystack = get_haystack();
        if haystack.contains(&needle) {
            AssertionCheckResult::Pass
        } else {
            AssertionCheckResult::Fail(format!("expected '{haystack}' to contain '{needle}'"))
        }
    })
}

// ============================================================================
// EXTREME TDD: Tests written FIRST per spec
// ============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    mod assertion_check_result {
        use super::*;

        #[test]
        fn test_pass() {
            let result = AssertionCheckResult::Pass;
            assert!(result.is_pass());
            assert!(!result.is_fail());
        }

        #[test]
        fn test_fail() {
            let result = AssertionCheckResult::Fail("error".into());
            assert!(result.is_fail());
            assert!(!result.is_pass());
        }
    }

    mod retry_config {
        use super::*;

        #[test]
        fn test_default() {
            let config = RetryConfig::default();
            assert_eq!(config.timeout, Duration::from_secs(5));
            assert_eq!(config.poll_interval, Duration::from_millis(100));
            assert_eq!(config.max_retries, 0);
        }

        #[test]
        fn test_new() {
            let config = RetryConfig::new(Duration::from_secs(10));
            assert_eq!(config.timeout, Duration::from_secs(10));
        }

        #[test]
        fn test_with_poll_interval() {
            let config = RetryConfig::default().with_poll_interval(Duration::from_millis(50));
            assert_eq!(config.poll_interval, Duration::from_millis(50));
        }

        #[test]
        fn test_with_max_retries() {
            let config = RetryConfig::default().with_max_retries(3);
            assert_eq!(config.max_retries, 3);
        }

        #[test]
        fn test_fast() {
            let config = RetryConfig::fast();
            assert_eq!(config.timeout, Duration::from_millis(500));
            assert_eq!(config.poll_interval, Duration::from_millis(50));
        }

        #[test]
        fn test_slow() {
            let config = RetryConfig::slow();
            assert_eq!(config.timeout, Duration::from_secs(30));
            assert_eq!(config.poll_interval, Duration::from_millis(500));
        }
    }

    mod retry_assertion {
        use super::*;

        #[test]
        fn test_immediate_pass() {
            let assertion = RetryAssertion::new(|| AssertionCheckResult::Pass);
            let result = assertion.verify().unwrap();
            assert_eq!(result.attempts, 1);
        }

        #[test]
        fn test_immediate_fail_with_timeout() {
            let assertion =
                RetryAssertion::new(|| AssertionCheckResult::Fail("always fails".into()))
                    .with_timeout(Duration::from_millis(100))
                    .with_poll_interval(Duration::from_millis(20));

            let err = assertion.verify().unwrap_err();
            assert!(err.attempts > 1);
            assert!(err.message.contains("always fails"));
        }

        #[test]
        fn test_eventual_pass() {
            let counter = Arc::new(AtomicUsize::new(0));
            let counter_clone = counter;

            let assertion = RetryAssertion::new(move || {
                let count = counter_clone.fetch_add(1, Ordering::SeqCst);
                if count >= 2 {
                    AssertionCheckResult::Pass
                } else {
                    AssertionCheckResult::Fail("not yet".into())
                }
            })
            .with_timeout(Duration::from_secs(1))
            .with_poll_interval(Duration::from_millis(10));

            let result = assertion.verify().unwrap();
            assert_eq!(result.attempts, 3);
        }

        #[test]
        fn test_max_retries() {
            let counter = Arc::new(AtomicUsize::new(0));
            let counter_clone = counter;

            let assertion = RetryAssertion::new(move || {
                let _ = counter_clone.fetch_add(1, Ordering::SeqCst);
                AssertionCheckResult::Fail("always fails".into())
            })
            .with_max_retries(3)
            .with_timeout(Duration::from_secs(10));

            let err = assertion.verify().unwrap_err();
            assert_eq!(err.attempts, 3);
        }

        #[test]
        fn test_with_description() {
            let assertion = RetryAssertion::new(|| AssertionCheckResult::Fail("error".into()))
                .with_description("checking visibility")
                .with_max_retries(1);

            let err = assertion.verify().unwrap_err();
            assert_eq!(err.description, Some("checking visibility".to_string()));
        }

        #[test]
        fn test_with_config() {
            let config = RetryConfig::fast();
            let assertion = RetryAssertion::new(|| AssertionCheckResult::Pass).with_config(config);
            assert_eq!(assertion.config().timeout, Duration::from_millis(500));
        }

        #[test]
        fn test_verify_once_pass() {
            let assertion = RetryAssertion::new(|| AssertionCheckResult::Pass);
            assert!(assertion.verify_once().is_ok());
        }

        #[test]
        fn test_verify_once_fail() {
            let assertion = RetryAssertion::new(|| AssertionCheckResult::Fail("error".into()));
            let err = assertion.verify_once().unwrap_err();
            assert_eq!(err.attempts, 1);
        }

        #[test]
        fn test_debug() {
            let assertion =
                RetryAssertion::new(|| AssertionCheckResult::Pass).with_description("test");
            let debug = format!("{assertion:?}");
            assert!(debug.contains("RetryAssertion"));
        }
    }

    mod retry_error {
        use super::*;

        #[test]
        fn test_display_without_description() {
            let err = RetryError {
                message: "failed".into(),
                attempts: 5,
                duration: Duration::from_millis(500),
                description: None,
            };
            let display = format!("{err}");
            assert!(display.contains("5 attempt(s)"));
            assert!(display.contains("failed"));
        }

        #[test]
        fn test_display_with_description() {
            let err = RetryError {
                message: "failed".into(),
                attempts: 3,
                duration: Duration::from_secs(1),
                description: Some("visibility check".into()),
            };
            let display = format!("{err}");
            assert!(display.contains("visibility check"));
            assert!(display.contains("failed"));
        }
    }

    mod helper_functions {
        use super::*;

        #[test]
        fn test_retry_eq_pass() {
            let assertion = retry_eq(|| 42, 42).with_max_retries(1);
            assert!(assertion.verify().is_ok());
        }

        #[test]
        fn test_retry_eq_fail() {
            let assertion = retry_eq(|| 1, 2).with_max_retries(1);
            let err = assertion.verify().unwrap_err();
            assert!(err.message.contains("expected"));
        }

        #[test]
        fn test_retry_true_pass() {
            let assertion = retry_true(|| true, "should be true").with_max_retries(1);
            assert!(assertion.verify().is_ok());
        }

        #[test]
        fn test_retry_true_fail() {
            let assertion = retry_true(|| false, "should be true").with_max_retries(1);
            let err = assertion.verify().unwrap_err();
            assert!(err.message.contains("should be true"));
        }

        #[test]
        fn test_retry_some_pass() {
            let assertion = retry_some(|| Some(42)).with_max_retries(1);
            assert!(assertion.verify().is_ok());
        }

        #[test]
        fn test_retry_some_fail() {
            let assertion = retry_some::<i32>(|| None).with_max_retries(1);
            assert!(assertion.verify().is_err());
        }

        #[test]
        fn test_retry_none_pass() {
            let assertion = retry_none::<i32>(|| None).with_max_retries(1);
            assert!(assertion.verify().is_ok());
        }

        #[test]
        fn test_retry_none_fail() {
            let assertion = retry_none(|| Some(42)).with_max_retries(1);
            assert!(assertion.verify().is_err());
        }

        #[test]
        fn test_retry_contains_pass() {
            let assertion =
                retry_contains(|| "hello world".to_string(), "world").with_max_retries(1);
            assert!(assertion.verify().is_ok());
        }

        #[test]
        fn test_retry_contains_fail() {
            let assertion = retry_contains(|| "hello".to_string(), "world").with_max_retries(1);
            let err = assertion.verify().unwrap_err();
            assert!(err.message.contains("contain"));
        }
    }

    mod retry_result {
        use super::*;

        #[test]
        fn test_result_fields() {
            let result = RetryResult {
                attempts: 3,
                duration: Duration::from_millis(100),
            };
            assert_eq!(result.attempts, 3);
            assert_eq!(result.duration, Duration::from_millis(100));
        }
    }
}
