//! Golden trace capture, comparison, and regression detection.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use super::types::{GoldenTraceError, GoldenTraceResult, SyscallBreakdownDelta, TraceMetrics};

/// Golden trace containing baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenTrace {
    /// Trace version (for compatibility checking)
    pub version: String,
    /// Trace name/identifier
    pub name: String,
    /// Capture timestamp (Unix epoch seconds)
    pub timestamp: u64,
    /// Git commit hash (if available)
    #[serde(default)]
    pub git_commit: Option<String>,
    /// Metrics captured
    pub metrics: TraceMetrics,
    /// Hash of the trace for verification
    #[serde(default)]
    pub hash: String,
    /// Environment info
    #[serde(default)]
    pub environment: HashMap<String, String>,
}

impl GoldenTrace {
    /// Create a new golden trace
    pub fn new(name: &str, metrics: TraceMetrics) -> Self {
        let timestamp =
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO).as_secs();

        let mut trace = Self {
            version: "1.0".to_string(),
            name: name.to_string(),
            timestamp,
            git_commit: None,
            metrics,
            hash: String::new(),
            environment: HashMap::new(),
        };

        trace.hash = trace.compute_hash();
        trace
    }

    /// Create with specific version
    pub fn with_version(name: &str, version: &str, metrics: TraceMetrics) -> Self {
        let mut trace = Self::new(name, metrics);
        trace.version = version.to_string();
        trace.hash = trace.compute_hash();
        trace
    }

    /// Set git commit
    pub fn git_commit(mut self, commit: &str) -> Self {
        self.git_commit = Some(commit.to_string());
        self.hash = self.compute_hash();
        self
    }

    /// Add environment info
    pub fn with_env(mut self, key: &str, value: &str) -> Self {
        self.environment.insert(key.to_string(), value.to_string());
        self
    }

    /// Compute deterministic hash of trace
    pub fn compute_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        self.name.hash(&mut hasher);
        self.version.hash(&mut hasher);
        self.timestamp.hash(&mut hasher);
        (self.metrics.total_time_us as u64).hash(&mut hasher);
        (self.metrics.p50_latency_us as u64).hash(&mut hasher);
        (self.metrics.p99_latency_us as u64).hash(&mut hasher);
        (self.metrics.throughput as u64).hash(&mut hasher);
        self.metrics.peak_memory_bytes.hash(&mut hasher);

        format!("{:016x}", hasher.finish())
    }

    /// Verify hash matches
    pub fn verify_hash(&self) -> bool {
        self.hash == self.compute_hash()
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> GoldenTraceResult<String> {
        serde_json::to_string_pretty(self).map_err(|e| GoldenTraceError::ParseError(e.to_string()))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> GoldenTraceResult<Self> {
        serde_json::from_str(json).map_err(|e| GoldenTraceError::ParseError(e.to_string()))
    }

    /// Serialize to TOML
    pub fn to_toml(&self) -> GoldenTraceResult<String> {
        toml::to_string_pretty(self).map_err(|e| GoldenTraceError::ParseError(e.to_string()))
    }

    /// Deserialize from TOML
    pub fn from_toml(toml_str: &str) -> GoldenTraceResult<Self> {
        toml::from_str(toml_str).map_err(|e| GoldenTraceError::ParseError(e.to_string()))
    }

    /// Save to file (auto-detects format from extension)
    pub fn save(&self, path: &Path) -> GoldenTraceResult<()> {
        let content = if path.extension().map_or(false, |ext| ext == "json") {
            self.to_json()?
        } else {
            self.to_toml()?
        };

        std::fs::write(path, content).map_err(|e| GoldenTraceError::IoError(e.to_string()))
    }

    /// Load from file (auto-detects format from extension)
    pub fn load(path: &Path) -> GoldenTraceResult<Self> {
        if !path.exists() {
            return Err(GoldenTraceError::NoBaseline);
        }

        let content =
            std::fs::read_to_string(path).map_err(|e| GoldenTraceError::IoError(e.to_string()))?;

        if path.extension().map_or(false, |ext| ext == "json") {
            Self::from_json(&content)
        } else {
            Self::from_toml(&content)
        }
    }
}

/// Comparison result between current and golden trace
#[derive(Debug, Clone)]
pub struct TraceComparison {
    /// Golden trace name
    pub golden_name: String,
    /// Golden trace version
    pub golden_version: String,
    /// Total time delta (percentage)
    pub time_delta_percent: f64,
    /// P50 latency delta (percentage)
    pub p50_delta_percent: f64,
    /// P99 latency delta (percentage)
    pub p99_delta_percent: f64,
    /// Throughput delta (percentage)
    pub throughput_delta_percent: f64,
    /// Memory delta (percentage)
    pub memory_delta_percent: f64,
    /// Syscall breakdown delta
    pub syscall_delta: SyscallBreakdownDelta,
    /// Is this a regression?
    pub is_regression: bool,
    /// Regression threshold used (percentage)
    pub threshold_percent: f64,
    /// Summary message
    pub summary: String,
}

impl TraceComparison {
    /// Get maximum positive delta (regression)
    pub fn max_regression(&self) -> f64 {
        self.time_delta_percent
            .max(self.p50_delta_percent)
            .max(self.p99_delta_percent)
            .max(-self.throughput_delta_percent) // Negative throughput is regression
            .max(self.memory_delta_percent)
    }

    /// Get all regressions above threshold
    pub fn regressions(&self) -> Vec<(String, f64)> {
        let mut regressions = Vec::new();
        let threshold = self.threshold_percent;

        if self.time_delta_percent > threshold {
            regressions.push(("total_time".to_string(), self.time_delta_percent));
        }
        if self.p50_delta_percent > threshold {
            regressions.push(("p50_latency".to_string(), self.p50_delta_percent));
        }
        if self.p99_delta_percent > threshold {
            regressions.push(("p99_latency".to_string(), self.p99_delta_percent));
        }
        if -self.throughput_delta_percent > threshold {
            regressions.push(("throughput".to_string(), self.throughput_delta_percent));
        }
        if self.memory_delta_percent > threshold {
            regressions.push(("memory".to_string(), self.memory_delta_percent));
        }

        regressions
    }
}

/// Golden trace comparator
#[derive(Debug, Clone)]
pub struct GoldenComparator {
    /// Regression threshold (default 10%)
    pub threshold_percent: f64,
    /// Allow version mismatch
    pub allow_version_mismatch: bool,
}

impl Default for GoldenComparator {
    fn default() -> Self {
        Self { threshold_percent: 10.0, allow_version_mismatch: false }
    }
}

impl GoldenComparator {
    /// Create new comparator
    pub fn new() -> Self {
        Self::default()
    }

    /// Set threshold
    pub fn with_threshold(mut self, percent: f64) -> Self {
        self.threshold_percent = percent;
        self
    }

    /// Allow version mismatch
    pub fn allow_version_mismatch(mut self) -> Self {
        self.allow_version_mismatch = true;
        self
    }

    /// Compare current metrics against golden trace
    pub fn compare(
        &self,
        current: &TraceMetrics,
        golden: &GoldenTrace,
    ) -> GoldenTraceResult<TraceComparison> {
        if !current.is_valid() {
            return Err(GoldenTraceError::InvalidTrace("Current metrics are invalid".to_string()));
        }

        if !golden.metrics.is_valid() {
            return Err(GoldenTraceError::InvalidTrace("Golden metrics are invalid".to_string()));
        }

        let time_delta = Self::calc_delta(current.total_time_us, golden.metrics.total_time_us);
        let p50_delta = Self::calc_delta(current.p50_latency_us, golden.metrics.p50_latency_us);
        let p99_delta = Self::calc_delta(current.p99_latency_us, golden.metrics.p99_latency_us);
        let throughput_delta = Self::calc_delta(current.throughput, golden.metrics.throughput);
        let memory_delta = Self::calc_delta(
            current.peak_memory_bytes as f64,
            golden.metrics.peak_memory_bytes as f64,
        );

        let syscall_delta = current.syscalls.percentage_diff(&golden.metrics.syscalls);

        // Regression is when metrics get worse
        // For time/latency/memory: higher is worse
        // For throughput: lower is worse
        let is_regression = time_delta > self.threshold_percent
            || p50_delta > self.threshold_percent
            || p99_delta > self.threshold_percent
            || -throughput_delta > self.threshold_percent
            || memory_delta > self.threshold_percent;

        let summary = if is_regression {
            format!(
                "REGRESSION detected vs {}: max delta {:.1}%",
                golden.name,
                time_delta.max(p50_delta).max(p99_delta).max(-throughput_delta).max(memory_delta)
            )
        } else {
            format!(
                "No regression vs {}: within {:.1}% threshold",
                golden.name, self.threshold_percent
            )
        };

        Ok(TraceComparison {
            golden_name: golden.name.clone(),
            golden_version: golden.version.clone(),
            time_delta_percent: time_delta,
            p50_delta_percent: p50_delta,
            p99_delta_percent: p99_delta,
            throughput_delta_percent: throughput_delta,
            memory_delta_percent: memory_delta,
            syscall_delta,
            is_regression,
            threshold_percent: self.threshold_percent,
            summary,
        })
    }

    fn calc_delta(current: f64, baseline: f64) -> f64 {
        if baseline == 0.0 {
            if current == 0.0 {
                0.0
            } else {
                100.0
            }
        } else {
            ((current - baseline) / baseline) * 100.0
        }
    }
}
