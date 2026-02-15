
impl ConvergenceMetrics {
    /// Create new convergence metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            total_iterations: 0,
            successful_fixes: 0,
            failed_fixes: 0,
            iteration_histogram: HashMap::new(),
        }
    }

    /// Record a convergence result.
    pub fn record(&mut self, iterations: usize, success: bool) {
        self.total_iterations += iterations as u64;
        *self.iteration_histogram.entry(iterations).or_insert(0) += 1;

        if success {
            self.successful_fixes += 1;
        } else {
            self.failed_fixes += 1;
        }
    }

    /// Get average iterations to fix.
    #[must_use]
    pub fn average_iterations(&self) -> f64 {
        let total = self.successful_fixes + self.failed_fixes;
        if total == 0 {
            0.0
        } else {
            self.total_iterations as f64 / total as f64
        }
    }

    /// Get convergence success rate.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.successful_fixes + self.failed_fixes;
        if total == 0 {
            0.0
        } else {
            self.successful_fixes as f64 / total as f64
        }
    }

    /// Get total fix attempts.
    #[must_use]
    pub fn total_attempts(&self) -> u64 {
        self.successful_fixes + self.failed_fixes
    }

    /// Get iteration histogram.
    #[must_use]
    pub fn histogram(&self) -> &HashMap<usize, u64> {
        &self.iteration_histogram
    }
}

impl Default for ConvergenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of all metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Total fix attempts
    pub total_fix_attempts: u64,
    /// Fix success rate (0.0 to 1.0)
    pub fix_success_rate: f64,
    /// Total compilations
    pub total_compilations: u64,
    /// Average compilation time in milliseconds
    pub avg_compilation_time_ms: f64,
    /// Most common error codes with counts
    pub most_common_errors: Vec<(String, u64)>,
    /// Average iterations to fix
    pub avg_iterations_to_fix: f64,
    /// Convergence rate (0.0 to 1.0)
    pub convergence_rate: f64,
    /// Session duration
    pub session_duration: Duration,
}

impl MetricsSummary {
    /// Format as a human-readable string.
    #[must_use]
    pub fn to_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== CITL Metrics Summary ===\n\n");

        let _ = writeln!(
            report,
            "Fix Attempts: {} (success rate: {:.1}%)",
            self.total_fix_attempts,
            self.fix_success_rate * 100.0
        );

        let _ = writeln!(
            report,
            "Compilations: {} (avg time: {:.1}ms)",
            self.total_compilations, self.avg_compilation_time_ms
        );

        let _ = writeln!(
            report,
            "Convergence: {:.1}% (avg {:.1} iterations)",
            self.convergence_rate * 100.0,
            self.avg_iterations_to_fix
        );

        if !self.most_common_errors.is_empty() {
            report.push_str("\nMost Common Errors:\n");
            for (code, count) in &self.most_common_errors {
                let _ = writeln!(report, "  {code}: {count}");
            }
        }

        let _ = writeln!(
            report,
            "\nSession Duration: {:.1}s",
            self.session_duration.as_secs_f64()
        );

        report
    }
}
