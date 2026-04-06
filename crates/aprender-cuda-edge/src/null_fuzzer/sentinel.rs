//! Null sentinel fuzzer with configurable injection.
//!
//! [`NullSentinelFuzzer`] drives null-pointer injection across kernel
//! arguments using the configured [`InjectionStrategy`].

use serde::{Deserialize, Serialize};

use super::guard_types::InjectionStrategy;

/// Configuration for the null sentinel fuzzer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NullFuzzerConfig {
    /// Injection strategy to use.
    pub strategy: InjectionStrategy,
    /// Total number of kernel calls to fuzz.
    pub total_calls: u64,
    /// Whether to abort on first null injection failure.
    pub fail_fast: bool,
}

impl Default for NullFuzzerConfig {
    fn default() -> Self {
        Self {
            strategy: InjectionStrategy::Periodic { interval: 10 },
            total_calls: 1000,
            fail_fast: false,
        }
    }
}

/// Report from a null fuzzing session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NullFuzzerReport {
    /// Total calls executed.
    pub total_calls: u64,
    /// Calls where null was injected.
    pub injections: u64,
    /// Calls where the null injection was caught (handled gracefully).
    pub caught: u64,
    /// Calls where the null injection caused an unhandled crash.
    pub crashes: u64,
}

impl NullFuzzerReport {
    /// Returns the catch rate (caught / injections), or 0 if no injections.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn catch_rate(&self) -> f64 {
        if self.injections == 0 {
            return 0.0;
        }
        self.caught as f64 / self.injections as f64
    }
}

/// Null sentinel fuzzer that injects null pointers into kernel calls.
#[derive(Debug, Clone)]
pub struct NullSentinelFuzzer {
    config: NullFuzzerConfig,
    call_index: u64,
}

impl NullSentinelFuzzer {
    /// Create a new fuzzer with the given config.
    #[must_use]
    pub fn new(config: NullFuzzerConfig) -> Self {
        Self {
            config,
            call_index: 0,
        }
    }

    /// Returns the current configuration.
    #[must_use]
    pub fn config(&self) -> &NullFuzzerConfig {
        &self.config
    }

    /// Check whether the next call should be injected with null.
    #[must_use]
    pub fn should_inject(&self) -> bool {
        self.config.strategy.should_inject(self.call_index)
    }

    /// Advance to the next call index, returning whether injection
    /// should occur for the *current* call.
    pub fn next_call(&mut self) -> bool {
        let inject = self.should_inject();
        self.call_index += 1;
        inject
    }

    /// Returns the current call index.
    #[must_use]
    pub fn call_index(&self) -> u64 {
        self.call_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn periodic_fuzzer_injects_at_interval() {
        let config = NullFuzzerConfig {
            strategy: InjectionStrategy::Periodic { interval: 3 },
            total_calls: 10,
            fail_fast: false,
        };
        let mut fuzzer = NullSentinelFuzzer::new(config);

        // call 0: inject (0 % 3 == 0)
        assert!(fuzzer.next_call());
        // call 1: no
        assert!(!fuzzer.next_call());
        // call 2: no
        assert!(!fuzzer.next_call());
        // call 3: inject
        assert!(fuzzer.next_call());
    }

    #[test]
    fn fuzzer_tracks_call_index() {
        let config = NullFuzzerConfig::default();
        let mut fuzzer = NullSentinelFuzzer::new(config);
        assert_eq!(fuzzer.call_index(), 0);
        fuzzer.next_call();
        assert_eq!(fuzzer.call_index(), 1);
        fuzzer.next_call();
        assert_eq!(fuzzer.call_index(), 2);
    }

    #[test]
    fn report_catch_rate_no_injections() {
        let report = NullFuzzerReport::default();
        assert!((report.catch_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn report_catch_rate_all_caught() {
        let report = NullFuzzerReport {
            total_calls: 100,
            injections: 10,
            caught: 10,
            crashes: 0,
        };
        assert!((report.catch_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fuzzer_config_accessor() {
        let config = NullFuzzerConfig {
            strategy: InjectionStrategy::Periodic { interval: 7 },
            total_calls: 500,
            fail_fast: true,
        };
        let fuzzer = NullSentinelFuzzer::new(config);
        let retrieved = fuzzer.config();
        assert!(matches!(
            retrieved.strategy,
            InjectionStrategy::Periodic { interval: 7 }
        ));
        assert_eq!(retrieved.total_calls, 500);
        assert!(retrieved.fail_fast);
    }

    #[test]
    fn fuzzer_should_inject_uses_strategy() {
        let config = NullFuzzerConfig {
            strategy: InjectionStrategy::Probabilistic { probability: 0.0 },
            total_calls: 100,
            fail_fast: false,
        };
        let fuzzer = NullSentinelFuzzer::new(config);
        // 0% probability should never inject
        assert!(!fuzzer.should_inject());
    }
}
