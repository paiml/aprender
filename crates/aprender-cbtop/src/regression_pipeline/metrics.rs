//! Benchmark metrics and results for the regression pipeline.

/// Benchmark result for a single metric
#[derive(Debug, Clone)]
pub struct BenchmarkMetric {
    /// Metric name
    pub name: String,
    /// Sample values
    pub samples: Vec<f64>,
    /// Unit of measurement
    pub unit: String,
}

impl BenchmarkMetric {
    /// Create new benchmark metric
    pub fn new(name: impl Into<String>, samples: Vec<f64>, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            samples,
            unit: unit.into(),
        }
    }

    /// Get mean value
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        self.samples.iter().sum::<f64>() / self.samples.len() as f64
    }

    /// Get standard deviation
    pub fn std_dev(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance = self.samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (self.samples.len() - 1) as f64;
        variance.sqrt()
    }

    /// Get coefficient of variation
    pub fn cv(&self) -> f64 {
        let mean = self.mean();
        if mean.abs() < 1e-10 {
            return 0.0;
        }
        (self.std_dev() / mean) * 100.0
    }
}

/// Results from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Git commit SHA
    pub commit: String,
    /// Branch name
    pub branch: String,
    /// Timestamp
    pub timestamp_ns: u64,
    /// Benchmark metrics
    pub metrics: Vec<BenchmarkMetric>,
    /// Total execution time in milliseconds
    pub duration_ms: u64,
    /// Host information
    pub host: String,
}

impl BenchmarkResults {
    /// Create new benchmark results
    pub fn new(commit: impl Into<String>, branch: impl Into<String>) -> Self {
        Self {
            commit: commit.into(),
            branch: branch.into(),
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0),
            metrics: Vec::new(),
            duration_ms: 0,
            host: hostname(),
        }
    }

    /// Add a metric
    pub fn add_metric(&mut self, metric: BenchmarkMetric) {
        self.metrics.push(metric);
    }

    /// Get metric by name
    pub fn get_metric(&self, name: &str) -> Option<&BenchmarkMetric> {
        self.metrics.iter().find(|m| m.name == name)
    }
}

/// Get hostname
fn hostname() -> String {
    std::env::var("HOSTNAME")
        .or_else(|_| std::env::var("HOST"))
        .unwrap_or_else(|_| "unknown".to_string())
}
