//! Evidence collection for falsification results
//!
//! Every test produces evidence that is recorded regardless of outcome.

use apr_qa_gen::QaScenario;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Outcome of a test
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Outcome {
    /// Hypothesis not falsified
    Corroborated,
    /// Hypothesis falsified
    Falsified,
    /// Test skipped
    Skipped,
    /// Test timed out
    Timeout,
    /// Test crashed
    Crashed,
}

impl Outcome {
    /// Check if this is a passing (corroborated) outcome.
    ///
    /// Only `Corroborated` counts as a pass. `Skipped` tests have not
    /// survived falsification and do not earn credit (Popperian principle).
    #[must_use]
    pub const fn is_pass(&self) -> bool {
        matches!(self, Self::Corroborated)
    }

    /// Check if this is a failing outcome
    #[must_use]
    pub const fn is_fail(&self) -> bool {
        matches!(self, Self::Falsified | Self::Timeout | Self::Crashed)
    }
}

/// Performance metrics from a test run
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Tokens per second
    pub tokens_per_second: Option<f64>,
    /// Time to first token in milliseconds
    pub time_to_first_token_ms: Option<f64>,
    /// Total tokens generated
    pub total_tokens: Option<u32>,
    /// Peak memory usage in MB
    pub memory_peak_mb: Option<u64>,
    /// Total duration in milliseconds
    pub duration_ms: u64,
}

/// Host information for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    /// Hostname
    pub hostname: String,
    /// Operating system
    pub os: String,
    /// CPU model
    pub cpu: String,
    /// GPU model (if available)
    pub gpu: Option<String>,
    /// apr-cli version
    pub apr_version: String,
}

impl Default for HostInfo {
    fn default() -> Self {
        Self {
            hostname: hostname::get().map_or_else(
                |_| "unknown".to_string(),
                |h| h.to_string_lossy().to_string(),
            ),
            os: std::env::consts::OS.to_string(),
            cpu: "unknown".to_string(),
            gpu: None,
            apr_version: "unknown".to_string(),
        }
    }
}

/// Evidence from a single test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Evidence {
    /// Unique evidence ID
    pub id: String,
    /// Gate ID (e.g., "F-HTTP-001")
    pub gate_id: String,
    /// Scenario that was tested
    pub scenario: QaScenario,
    /// Test outcome
    pub outcome: Outcome,
    /// Human-readable reason
    pub reason: String,
    /// Raw output from the command
    pub output: String,
    /// Standard error output
    pub stderr: Option<String>,
    /// Exit code
    pub exit_code: Option<i32>,
    /// Performance metrics
    pub metrics: PerformanceMetrics,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Host information
    pub host: HostInfo,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl Evidence {
    /// Create new evidence for a corroborated test
    #[must_use]
    pub fn corroborated(
        gate_id: impl Into<String>,
        scenario: QaScenario,
        output: impl Into<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Corroborated,
            reason: "Test passed".to_string(),
            output: output.into(),
            stderr: None,
            exit_code: Some(0),
            metrics: PerformanceMetrics {
                duration_ms,
                ..Default::default()
            },
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create new evidence for a falsified test
    #[must_use]
    pub fn falsified(
        gate_id: impl Into<String>,
        scenario: QaScenario,
        reason: impl Into<String>,
        output: impl Into<String>,
        duration_ms: u64,
    ) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Falsified,
            reason: reason.into(),
            output: output.into(),
            stderr: None,
            exit_code: None,
            metrics: PerformanceMetrics {
                duration_ms,
                ..Default::default()
            },
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create new evidence for a timeout
    #[must_use]
    pub fn timeout(gate_id: impl Into<String>, scenario: QaScenario, timeout_ms: u64) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Timeout,
            reason: format!("Timed out after {timeout_ms}ms"),
            output: String::new(),
            stderr: None,
            exit_code: None,
            metrics: PerformanceMetrics {
                duration_ms: timeout_ms,
                ..Default::default()
            },
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create new evidence for a crash
    #[must_use]
    pub fn crashed(
        gate_id: impl Into<String>,
        scenario: QaScenario,
        stderr: impl Into<String>,
        exit_code: i32,
        duration_ms: u64,
    ) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Crashed,
            reason: format!("Process crashed with exit code {exit_code}"),
            output: String::new(),
            stderr: Some(stderr.into()),
            exit_code: Some(exit_code),
            metrics: PerformanceMetrics {
                duration_ms,
                ..Default::default()
            },
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Create new evidence for a skipped test
    #[must_use]
    pub fn skipped(
        gate_id: impl Into<String>,
        scenario: QaScenario,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            id: uuid_v4(),
            gate_id: gate_id.into(),
            scenario,
            outcome: Outcome::Skipped,
            reason: reason.into(),
            output: String::new(),
            stderr: None,
            exit_code: None,
            metrics: PerformanceMetrics::default(),
            timestamp: Utc::now(),
            host: HostInfo::default(),
            metadata: HashMap::new(),
        }
    }

    /// Add performance metrics
    #[must_use]
    pub const fn with_metrics(mut self, metrics: PerformanceMetrics) -> Self {
        self.metrics = metrics;
        self
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }
}

/// Collector for evidence from multiple tests
#[derive(Debug, Clone, Default)]
pub struct EvidenceCollector {
    evidence: Vec<Evidence>,
}

impl EvidenceCollector {
    /// Create a new collector
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add evidence
    pub fn add(&mut self, evidence: Evidence) {
        self.evidence.push(evidence);
    }

    /// Get all evidence
    #[must_use]
    pub fn all(&self) -> &[Evidence] {
        &self.evidence
    }

    /// Get count of each outcome type
    #[must_use]
    pub fn counts(&self) -> HashMap<Outcome, usize> {
        let mut counts = HashMap::new();
        for e in &self.evidence {
            *counts.entry(e.outcome).or_insert(0) += 1;
        }
        counts
    }

    /// Get pass count
    #[must_use]
    pub fn pass_count(&self) -> usize {
        self.evidence.iter().filter(|e| e.outcome.is_pass()).count()
    }

    /// Get fail count
    #[must_use]
    pub fn fail_count(&self) -> usize {
        self.evidence.iter().filter(|e| e.outcome.is_fail()).count()
    }

    /// Get total count
    #[must_use]
    pub fn total(&self) -> usize {
        self.evidence.len()
    }

    /// Get failed evidence
    #[must_use]
    pub fn failures(&self) -> Vec<&Evidence> {
        self.evidence
            .iter()
            .filter(|e| e.outcome.is_fail())
            .collect()
    }

    /// Export to JSON
    ///
    /// # Errors
    ///
    /// Returns an error if serialization fails.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.evidence)
    }
}

/// Generate a unique ID from timestamp + atomic counter.
/// Counter ensures uniqueness even when multiple Evidence objects are
/// created within the same nanosecond (parallel Rayon execution).
fn uuid_v4() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{timestamp:024x}{seq:08x}")
}

#[cfg(test)]
#[path = "evidence_tests.rs"]
mod evidence_tests;
