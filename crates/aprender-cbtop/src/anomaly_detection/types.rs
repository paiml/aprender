//! Anomaly detection types: severity, classification, anomalies, and reports.

/// Default Z-score threshold for outlier detection
pub const DEFAULT_ZSCORE_THRESHOLD: f64 = 3.0;

/// Default IQR multiplier for outlier detection
pub const DEFAULT_IQR_MULTIPLIER: f64 = 1.5;

/// Minimum samples for statistical analysis
pub const MIN_SAMPLES_FOR_DETECTION: usize = 10;

/// Anomaly severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AnomalySeverity {
    /// Informational - minor deviation
    Info,
    /// Warning - notable deviation
    Warning,
    /// Critical - severe deviation requiring attention
    Critical,
}

impl AnomalySeverity {
    /// Get severity name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }

    /// Get severity from deviation magnitude
    pub fn from_deviation(deviation: f64) -> Self {
        if deviation >= 5.0 {
            Self::Critical
        } else if deviation >= 3.0 {
            Self::Warning
        } else {
            Self::Info
        }
    }
}

/// Anomaly type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnomalyType {
    /// Statistical outlier (Z-score based)
    Outlier,
    /// Performance spike (sudden increase)
    Spike,
    /// Performance drop (sudden decrease)
    Drop,
    /// Change point (sustained shift)
    ChangePoint,
    /// Periodic anomaly (recurring pattern)
    Periodic,
    /// Correlated anomaly (multi-metric)
    Correlated,
}

impl AnomalyType {
    /// Get anomaly type name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Outlier => "outlier",
            Self::Spike => "spike",
            Self::Drop => "drop",
            Self::ChangePoint => "change_point",
            Self::Periodic => "periodic",
            Self::Correlated => "correlated",
        }
    }
}

/// Detected anomaly
#[derive(Debug, Clone)]
pub struct Anomaly {
    /// Index in the data series
    pub index: usize,
    /// The anomalous value
    pub value: f64,
    /// Expected value (mean or predicted)
    pub expected: f64,
    /// Deviation from expected (in standard deviations)
    pub deviation: f64,
    /// Anomaly type
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Optional description
    pub description: Option<String>,
}

impl Anomaly {
    /// Create new anomaly
    pub fn new(
        index: usize,
        value: f64,
        expected: f64,
        deviation: f64,
        anomaly_type: AnomalyType,
    ) -> Self {
        Self {
            index,
            value,
            expected,
            deviation,
            anomaly_type,
            severity: AnomalySeverity::from_deviation(deviation.abs()),
            description: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Check if anomaly is critical
    pub fn is_critical(&self) -> bool {
        self.severity == AnomalySeverity::Critical
    }

    /// Export to JSON format
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"index":{},"value":{},"expected":{},"deviation":{},"type":"{}","severity":"{}"}}"#,
            self.index,
            self.value,
            self.expected,
            self.deviation,
            self.anomaly_type.name(),
            self.severity.name()
        )
    }
}

/// Change point in data series
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Index where change occurs
    pub index: usize,
    /// Mean before change
    pub mean_before: f64,
    /// Mean after change
    pub mean_after: f64,
    /// Magnitude of change
    pub magnitude: f64,
    /// Direction of change (positive = increase)
    pub direction: f64,
}

impl ChangePoint {
    /// Create new change point
    pub fn new(index: usize, mean_before: f64, mean_after: f64) -> Self {
        let magnitude = (mean_after - mean_before).abs();
        let direction = mean_after - mean_before;
        Self {
            index,
            mean_before,
            mean_after,
            magnitude,
            direction,
        }
    }

    /// Check if change is significant (>10% shift)
    pub fn is_significant(&self) -> bool {
        if self.mean_before.abs() < 1e-10 {
            return self.magnitude > 1e-10;
        }
        (self.magnitude / self.mean_before.abs()) > 0.1
    }
}

/// Anomaly detection result summary
#[derive(Debug, Clone)]
pub struct AnomalyReport {
    /// Total data points analyzed
    pub total_points: usize,
    /// Detected anomalies
    pub anomalies: Vec<Anomaly>,
    /// Detected change points
    pub change_points: Vec<ChangePoint>,
    /// Data mean
    pub mean: f64,
    /// Data standard deviation
    pub std_dev: f64,
    /// Detection method used
    pub method: &'static str,
}

impl AnomalyReport {
    /// Count anomalies by severity
    pub fn count_by_severity(&self, severity: AnomalySeverity) -> usize {
        self.anomalies
            .iter()
            .filter(|a| a.severity == severity)
            .count()
    }

    /// Get critical anomalies
    pub fn critical_anomalies(&self) -> Vec<&Anomaly> {
        self.anomalies.iter().filter(|a| a.is_critical()).collect()
    }

    /// Check if any critical anomalies exist
    pub fn has_critical(&self) -> bool {
        self.anomalies.iter().any(|a| a.is_critical())
    }

    /// Export report to JSON
    pub fn to_json(&self) -> String {
        let anomalies_json: Vec<String> = self.anomalies.iter().map(|a| a.to_json()).collect();
        format!(
            r#"{{"total_points":{},"anomaly_count":{},"critical_count":{},"mean":{},"std_dev":{},"method":"{}","anomalies":[{}]}}"#,
            self.total_points,
            self.anomalies.len(),
            self.count_by_severity(AnomalySeverity::Critical),
            self.mean,
            self.std_dev,
            self.method,
            anomalies_json.join(",")
        )
    }
}
