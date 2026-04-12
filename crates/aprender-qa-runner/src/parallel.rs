//! Parallel execution support using Rayon
//!
//! Implements Heijunka (load-balanced) parallel execution across workers.

use crate::evidence::{Evidence, Outcome, PerformanceMetrics};
use aprender_qa_gen::QaScenario;
use rayon::prelude::*;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Parallel executor configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads
    pub num_workers: usize,
    /// Timeout per scenario in milliseconds
    pub timeout_ms: u64,
    /// Path to model file
    pub model_path: String,
    /// Stop on first failure
    pub stop_on_failure: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get().min(4),
            timeout_ms: 60_000,
            model_path: "model.gguf".to_string(),
            stop_on_failure: false,
        }
    }
}

/// Result of parallel execution
#[derive(Debug)]
pub struct ParallelResult {
    /// All evidence collected
    pub evidence: Vec<Evidence>,
    /// Number of passed scenarios
    pub passed: usize,
    /// Number of failed scenarios
    pub failed: usize,
    /// Number of skipped scenarios
    pub skipped: usize,
    /// Total duration in milliseconds
    pub duration_ms: u64,
    /// Whether execution was stopped early
    pub stopped_early: bool,
}

/// Parallel scenario executor
pub struct ParallelExecutor {
    config: ParallelConfig,
}

impl ParallelExecutor {
    /// Create a new parallel executor
    #[must_use]
    pub fn new(config: ParallelConfig) -> Self {
        // Configure rayon thread pool
        rayon::ThreadPoolBuilder::new()
            .num_threads(config.num_workers)
            .build_global()
            .ok(); // Ignore if already configured
        Self { config }
    }

    /// Execute scenarios in parallel
    #[must_use]
    pub fn execute(&self, scenarios: &[QaScenario]) -> ParallelResult {
        let start = Instant::now();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let passed = Arc::new(AtomicUsize::new(0));
        let failed = Arc::new(AtomicUsize::new(0));
        let skipped = Arc::new(AtomicUsize::new(0));

        let evidence: Vec<Evidence> = scenarios
            .par_iter()
            .map(|scenario| {
                // Check if we should stop — emit Skipped evidence (Popperian audit trail)
                if self.config.stop_on_failure && stop_flag.load(Ordering::Relaxed) {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    return Evidence::skipped(
                        format!("F-{}-001", scenario.mqs_category()),
                        scenario.clone(),
                        "Skipped: execution stopped early per stop_on_failure policy",
                    );
                }

                let result = self.execute_single(scenario);

                if result.outcome.is_pass() {
                    passed.fetch_add(1, Ordering::Relaxed);
                } else if result.outcome == Outcome::Skipped {
                    // Skipped is neither pass nor fail — no counter increment
                } else {
                    failed.fetch_add(1, Ordering::Relaxed);
                    if self.config.stop_on_failure {
                        stop_flag.store(true, Ordering::Relaxed);
                    }
                }

                result
            })
            .collect();

        ParallelResult {
            evidence,
            passed: passed.load(Ordering::Relaxed),
            failed: failed.load(Ordering::Relaxed),
            skipped: skipped.load(Ordering::Relaxed),
            duration_ms: start.elapsed().as_millis() as u64,
            stopped_early: stop_flag.load(Ordering::Relaxed),
        }
    }

    /// Execute a single scenario
    fn execute_single(&self, scenario: &QaScenario) -> Evidence {
        let start = Instant::now();

        let (output, exit_code, stderr) = self.subprocess_execution(scenario);

        let duration = start.elapsed().as_millis() as u64;
        let gate_id = format!("F-{}-001", scenario.mqs_category());

        // Check for crash
        if exit_code != 0 {
            return Evidence::crashed(
                &gate_id,
                scenario.clone(),
                "Non-zero exit code",
                exit_code,
                duration,
            )
            .with_stderr(stderr);
        }

        // Check for timeout
        if duration > self.config.timeout_ms {
            return Evidence::timeout(&gate_id, scenario.clone(), duration);
        }

        // Evaluate output with oracle
        let oracle_result = scenario.evaluate(&output);

        match oracle_result {
            aprender_qa_gen::OracleResult::Corroborated { evidence: reason } => {
                Evidence::corroborated(&gate_id, scenario.clone(), &output, duration)
                    .with_metrics(PerformanceMetrics {
                        duration_ms: duration,
                        total_tokens: Some(estimate_tokens(&output)),
                        ..Default::default()
                    })
                    .with_reason(reason)
            }
            aprender_qa_gen::OracleResult::Falsified {
                reason,
                evidence: _,
            } => Evidence::falsified(&gate_id, scenario.clone(), reason, &output, duration),
        }
    }

    /// Execute via subprocess (real execution)
    fn subprocess_execution(&self, scenario: &QaScenario) -> (String, i32, Option<String>) {
        let cmd_str = scenario.to_command(&self.config.model_path);
        let parts: Vec<&str> = cmd_str.split_whitespace().collect();

        if parts.is_empty() {
            return (String::new(), -1, Some("Empty command".to_string()));
        }

        let result = Command::new(parts[0])
            .args(&parts[1..])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();

        match result {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                let exit_code = output.status.code().unwrap_or(-1);
                (
                    stdout,
                    exit_code,
                    if stderr.is_empty() {
                        None
                    } else {
                        Some(stderr)
                    },
                )
            }
            Err(e) => (String::new(), -1, Some(e.to_string())),
        }
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new(ParallelConfig::default())
    }
}

/// Estimate token count from output (rough heuristic)
fn estimate_tokens(text: &str) -> u32 {
    // Rough estimate: ~4 chars per token for English
    (text.len() / 4).max(1) as u32
}

/// Extension trait for Evidence to add optional fields
trait EvidenceExt {
    fn with_stderr(self, stderr: Option<String>) -> Self;
    fn with_reason(self, reason: String) -> Self;
}

impl EvidenceExt for Evidence {
    fn with_stderr(mut self, stderr: Option<String>) -> Self {
        self.stderr = stderr;
        self
    }

    fn with_reason(mut self, reason: String) -> Self {
        self.reason = reason;
        self
    }
}

#[cfg(test)]
#[path = "parallel_tests.rs"]
mod parallel_tests;
