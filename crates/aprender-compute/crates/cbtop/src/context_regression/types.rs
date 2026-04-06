//! Context regression types: system context, baselines, thresholds, and trends.

/// Default cold start margin (15%)
pub const DEFAULT_COLD_START_MARGIN: f64 = 15.0;

/// Default minimum samples for learning
pub const MIN_SAMPLES_FOR_CONTEXT: usize = 5;

/// Default context staleness (seconds)
pub const DEFAULT_STALENESS_SEC: u64 = 3600;

/// System context snapshot
#[derive(Debug, Clone)]
pub struct SystemContext {
    /// Timestamp (Unix seconds)
    pub timestamp: u64,
    /// CPU temperature (Celsius)
    pub cpu_temp_c: f64,
    /// GPU temperature (Celsius)
    pub gpu_temp_c: f64,
    /// Memory utilization (0-100)
    pub memory_percent: f64,
    /// CPU frequency (MHz)
    pub cpu_freq_mhz: f64,
    /// Maximum CPU frequency (MHz)
    pub cpu_freq_max_mhz: f64,
    /// Is cache warm
    pub cache_warm: bool,
    /// Load average (1 min)
    pub load_average: f64,
}

impl Default for SystemContext {
    fn default() -> Self {
        Self {
            timestamp: 0,
            cpu_temp_c: 50.0,
            gpu_temp_c: 50.0,
            memory_percent: 50.0,
            cpu_freq_mhz: 3000.0,
            cpu_freq_max_mhz: 4000.0,
            cache_warm: false,
            load_average: 1.0,
        }
    }
}

impl SystemContext {
    /// Create new context
    pub fn new() -> Self {
        Self::default()
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    /// Set CPU temperature
    pub fn with_cpu_temp(mut self, temp_c: f64) -> Self {
        self.cpu_temp_c = temp_c;
        self
    }

    /// Set GPU temperature
    pub fn with_gpu_temp(mut self, temp_c: f64) -> Self {
        self.gpu_temp_c = temp_c;
        self
    }

    /// Set memory utilization
    pub fn with_memory(mut self, percent: f64) -> Self {
        self.memory_percent = percent.clamp(0.0, 100.0);
        self
    }

    /// Set CPU frequency
    pub fn with_cpu_freq(mut self, freq_mhz: f64, max_mhz: f64) -> Self {
        self.cpu_freq_mhz = freq_mhz;
        self.cpu_freq_max_mhz = max_mhz;
        self
    }

    /// Set cache state
    pub fn with_cache_warm(mut self, warm: bool) -> Self {
        self.cache_warm = warm;
        self
    }

    /// Set load average
    pub fn with_load(mut self, load: f64) -> Self {
        self.load_average = load;
        self
    }

    /// Capture current system context
    pub fn capture() -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        // In a real implementation, these would read from sysfs/procfs
        // For now, return reasonable defaults
        Self {
            timestamp,
            cpu_temp_c: 60.0,
            gpu_temp_c: 55.0,
            memory_percent: 50.0,
            cpu_freq_mhz: 3000.0,
            cpu_freq_max_mhz: 4000.0,
            cache_warm: false,
            load_average: 1.0,
        }
    }

    /// Get frequency utilization (0-1)
    pub fn freq_utilization(&self) -> f64 {
        if self.cpu_freq_max_mhz > 0.0 {
            self.cpu_freq_mhz / self.cpu_freq_max_mhz
        } else {
            1.0
        }
    }

    /// Get thermal headroom (degrees below throttle threshold)
    pub fn thermal_headroom(&self, throttle_temp: f64) -> f64 {
        (throttle_temp - self.cpu_temp_c).max(0.0)
    }

    /// Check if context is stale
    pub fn is_stale(&self, max_age_sec: u64) -> bool {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        now.saturating_sub(self.timestamp) > max_age_sec
    }

    /// Export to JSON
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"timestamp":{},"cpu_temp_c":{},"memory_percent":{},"cpu_freq_mhz":{},"cache_warm":{}}}"#,
            self.timestamp,
            self.cpu_temp_c,
            self.memory_percent,
            self.cpu_freq_mhz,
            self.cache_warm
        )
    }
}

/// Historical baseline entry
#[derive(Debug, Clone)]
pub struct BaselineEntry {
    /// Context at measurement time
    pub context: SystemContext,
    /// Measured value
    pub value: f64,
    /// Metric name
    pub metric: String,
}

impl BaselineEntry {
    /// Create new entry
    pub fn new(metric: &str, value: f64, context: SystemContext) -> Self {
        Self { metric: metric.to_string(), value, context }
    }
}

/// Computed regression threshold
#[derive(Debug, Clone)]
pub struct RegressionThreshold {
    /// Base threshold (from historical mean)
    pub base_percent: f64,
    /// Temperature adjustment
    pub temp_adjustment: f64,
    /// Memory adjustment
    pub memory_adjustment: f64,
    /// Frequency adjustment
    pub freq_adjustment: f64,
    /// Cache adjustment
    pub cache_adjustment: f64,
    /// Final threshold
    pub final_percent: f64,
    /// Confidence in threshold
    pub confidence: f64,
    /// Number of samples used
    pub sample_count: usize,
}

impl RegressionThreshold {
    /// Check if regression detected
    pub fn is_regression(&self, percent_change: f64) -> bool {
        percent_change.abs() > self.final_percent
    }
}

/// Detected trend
#[derive(Debug, Clone)]
pub struct Trend {
    /// Slope (change per day)
    pub slope_per_day: f64,
    /// R² of linear fit
    pub r_squared: f64,
    /// Direction description
    pub direction: &'static str,
}

impl Trend {
    /// Check if trend is significant
    pub fn is_significant(&self) -> bool {
        self.r_squared > 0.5 && self.slope_per_day.abs() > 0.1
    }
}

/// Result of regression check
#[derive(Debug, Clone)]
pub struct RegressionCheck {
    /// Metric name
    pub metric: String,
    /// Current value
    pub current_value: f64,
    /// Baseline mean
    pub baseline_mean: f64,
    /// Percent change from baseline
    pub percent_change: f64,
    /// Computed threshold
    pub threshold: RegressionThreshold,
    /// Is regression detected
    pub is_regression: bool,
    /// Detected trend
    pub trend: Option<Trend>,
}

impl RegressionCheck {
    /// Check if passed (no regression)
    pub fn passed(&self) -> bool {
        !self.is_regression
    }
}
