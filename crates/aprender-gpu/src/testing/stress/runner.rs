//! Stress test runner and configuration.

use std::time::{Duration, Instant};

use super::rng::StressRng;
use super::types::{
    verify_performance, Anomaly, AnomalyKind, FrameProfile, PerformanceResult,
    PerformanceThresholds, StressReport,
};

/// Stress test configuration
#[derive(Debug, Clone)]
pub struct StressConfig {
    /// Number of cycles to run
    pub cycles: u32,
    /// Interval between cycles (ms)
    pub interval_ms: u64,
    /// Base seed for RNG
    pub seed: u64,
    /// Min input size
    pub min_input_size: usize,
    /// Max input size
    pub max_input_size: usize,
    /// Performance thresholds
    pub thresholds: PerformanceThresholds,
}

impl Default for StressConfig {
    fn default() -> Self {
        Self {
            cycles: 100,
            interval_ms: 100,
            seed: 42,
            min_input_size: 64,
            max_input_size: 512,
            thresholds: PerformanceThresholds::default(),
        }
    }
}

/// Stress test runner
pub struct StressTestRunner {
    rng: StressRng,
    config: StressConfig,
    report: StressReport,
}

impl StressTestRunner {
    /// Create new stress test runner
    #[must_use]
    pub fn new(config: StressConfig) -> Self {
        Self {
            rng: StressRng::new(config.seed),
            config,
            report: StressReport::default(),
        }
    }

    /// Generate randomized input for a cycle
    pub fn generate_input(&mut self) -> (u64, Vec<f32>) {
        let seed = self.rng.next_u64();
        let size = self.rng.gen_range_u32(
            self.config.min_input_size as u32,
            self.config.max_input_size as u32,
        ) as usize;

        let mut input_rng = StressRng::new(seed);
        let input: Vec<f32> = (0..size).map(|_| input_rng.gen_f32()).collect();

        (seed, input)
    }

    /// Run a single cycle with provided test function
    pub fn run_cycle<F>(&mut self, cycle: u32, test_fn: F) -> FrameProfile
    where
        F: FnOnce(&[f32]) -> (u32, u32), // Returns (passed, failed)
    {
        let (input_seed, input) = self.generate_input();
        let input_size = input.len();

        let start = Instant::now();
        let (tests_passed, tests_failed) = test_fn(&input);
        let duration = start.elapsed();

        let profile = FrameProfile {
            cycle,
            duration_ms: duration.as_millis() as u64,
            memory_bytes: input_size * std::mem::size_of::<f32>(),
            tests_passed,
            tests_failed,
            input_seed,
            input_size,
        };

        // Check for anomalies
        if profile.duration_ms > self.config.thresholds.max_frame_time_ms {
            self.report.anomalies.push(Anomaly {
                cycle,
                kind: AnomalyKind::SlowFrame,
                description: format!(
                    "Frame {}ms exceeds threshold {}ms",
                    profile.duration_ms, self.config.thresholds.max_frame_time_ms
                ),
            });
        }

        if tests_failed > 0 {
            self.report.anomalies.push(Anomaly {
                cycle,
                kind: AnomalyKind::TestFailure,
                description: format!("{} tests failed in cycle {}", tests_failed, cycle),
            });
        }

        self.report.add_frame(profile.clone());
        profile
    }

    /// Run all cycles
    pub fn run_all<F>(&mut self, mut test_fn: F) -> &StressReport
    where
        F: FnMut(&[f32]) -> (u32, u32),
    {
        let interval = Duration::from_millis(self.config.interval_ms);

        for cycle in 0..self.config.cycles {
            let start = Instant::now();
            self.run_cycle(cycle, &mut test_fn);

            let elapsed = start.elapsed();
            if let Some(remaining) = interval.checked_sub(elapsed) {
                std::thread::sleep(remaining);
            }
        }

        &self.report
    }

    /// Get the current report
    #[must_use]
    pub fn report(&self) -> &StressReport {
        &self.report
    }

    /// Verify performance and return result
    #[must_use]
    pub fn verify(&self) -> PerformanceResult {
        verify_performance(&self.report, &self.config.thresholds)
    }
}
