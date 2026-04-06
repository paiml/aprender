//! Prometheus metric types and values.

/// Default max labels per metric
pub const DEFAULT_MAX_LABELS: usize = 10;

/// Default histogram buckets
pub const DEFAULT_BUCKETS: [f64; 11] =
    [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0];

/// Metric type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    /// Gauge (instantaneous value)
    Gauge,
    /// Counter (cumulative)
    Counter,
    /// Histogram (distribution)
    Histogram,
}

impl MetricType {
    /// Get type name for Prometheus format
    pub fn name(&self) -> &'static str {
        match self {
            Self::Gauge => "gauge",
            Self::Counter => "counter",
            Self::Histogram => "histogram",
        }
    }
}

/// Metric labels
#[derive(Debug, Clone, Default)]
pub struct Labels {
    pairs: Vec<(String, String)>,
}

impl Labels {
    /// Create empty labels
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Add label
    pub fn add(mut self, key: &str, value: &str) -> Self {
        self.pairs.push((key.to_string(), value.to_string()));
        self
    }

    /// Format as Prometheus label string
    pub fn format(&self) -> String {
        if self.pairs.is_empty() {
            return String::new();
        }

        let parts: Vec<String> = self
            .pairs
            .iter()
            .map(|(k, v)| format!("{}=\"{}\"", k, escape_label_value(v)))
            .collect();

        format!("{{{}}}", parts.join(","))
    }

    /// Get label count
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Escape label value for Prometheus format
pub fn escape_label_value(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
}

/// Histogram buckets
#[derive(Debug, Clone)]
pub struct HistogramBuckets {
    /// Bucket boundaries
    pub boundaries: Vec<f64>,
    /// Counts per bucket
    pub counts: Vec<u64>,
    /// Total sum
    pub sum: f64,
    /// Total count
    pub count: u64,
}

impl Default for HistogramBuckets {
    fn default() -> Self {
        Self::with_buckets(&DEFAULT_BUCKETS)
    }
}

impl HistogramBuckets {
    /// Create with custom buckets
    pub fn with_buckets(boundaries: &[f64]) -> Self {
        Self {
            boundaries: boundaries.to_vec(),
            counts: vec![0; boundaries.len()],
            sum: 0.0,
            count: 0,
        }
    }

    /// Observe a value
    pub fn observe(&mut self, value: f64) {
        self.sum += value;
        self.count += 1;

        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if value <= boundary {
                self.counts[i] += 1;
            }
        }
    }

    /// Format as Prometheus histogram
    pub fn format(&self, name: &str, labels: &Labels) -> String {
        let mut lines = Vec::new();
        let label_str = labels.format();

        // Bucket lines
        let mut cumulative = 0u64;
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            cumulative += self.counts[i];
            let bucket_label = if label_str.is_empty() {
                format!("{{le=\"{}\"}}", boundary)
            } else {
                format!("{{le=\"{}\",{}}}", boundary, &label_str[1..label_str.len() - 1])
            };
            lines.push(format!("{}_bucket{} {}", name, bucket_label, cumulative));
        }

        // +Inf bucket
        let inf_label = if label_str.is_empty() {
            "{le=\"+Inf\"}".to_string()
        } else {
            format!("{{le=\"+Inf\",{}}}", &label_str[1..label_str.len() - 1])
        };
        lines.push(format!("{}_bucket{} {}", name, inf_label, self.count));

        // Sum and count
        lines.push(format!("{}_sum{} {}", name, label_str, self.sum));
        lines.push(format!("{}_count{} {}", name, label_str, self.count));

        lines.join("\n")
    }
}

/// Single metric definition
#[derive(Debug, Clone)]
pub struct MetricDef {
    /// Metric name
    pub name: String,
    /// Help text
    pub help: String,
    /// Metric type
    pub metric_type: MetricType,
}

impl MetricDef {
    /// Create new metric definition
    pub fn new(name: &str, help: &str, metric_type: MetricType) -> Self {
        Self { name: name.to_string(), help: help.to_string(), metric_type }
    }

    /// Format HELP line
    pub fn format_help(&self) -> String {
        format!("# HELP {} {}", self.name, self.help)
    }

    /// Format TYPE line
    pub fn format_type(&self) -> String {
        format!("# TYPE {} {}", self.name, self.metric_type.name())
    }
}

/// Gauge metric value
#[derive(Debug, Clone)]
pub struct GaugeValue {
    /// Value
    pub value: f64,
    /// Labels
    pub labels: Labels,
    /// Timestamp (optional, milliseconds)
    pub timestamp: Option<u64>,
}

impl GaugeValue {
    /// Create new gauge value
    pub fn new(value: f64) -> Self {
        Self { value, labels: Labels::new(), timestamp: None }
    }

    /// With labels
    pub fn with_labels(mut self, labels: Labels) -> Self {
        self.labels = labels;
        self
    }

    /// With timestamp
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Format as Prometheus line
    pub fn format(&self, name: &str) -> String {
        let label_str = self.labels.format();
        let ts_str = self.timestamp.map(|t| format!(" {}", t)).unwrap_or_default();
        format!("{}{} {}{}", name, label_str, self.value, ts_str)
    }
}

/// Counter metric value
#[derive(Debug, Clone)]
pub struct CounterValue {
    /// Value (monotonically increasing)
    pub value: u64,
    /// Labels
    pub labels: Labels,
}

impl CounterValue {
    /// Create new counter
    pub fn new(value: u64) -> Self {
        Self { value, labels: Labels::new() }
    }

    /// With labels
    pub fn with_labels(mut self, labels: Labels) -> Self {
        self.labels = labels;
        self
    }

    /// Format as Prometheus line
    pub fn format(&self, name: &str) -> String {
        let label_str = self.labels.format();
        format!("{}{} {}", name, label_str, self.value)
    }
}

/// Histogram metric value
#[derive(Debug, Clone)]
pub struct HistogramValue {
    /// Buckets
    pub buckets: HistogramBuckets,
    /// Labels
    pub labels: Labels,
}

impl HistogramValue {
    /// Create new histogram
    pub fn new() -> Self {
        Self { buckets: HistogramBuckets::default(), labels: Labels::new() }
    }

    /// With custom buckets
    pub fn with_buckets(boundaries: &[f64]) -> Self {
        Self { buckets: HistogramBuckets::with_buckets(boundaries), labels: Labels::new() }
    }

    /// With labels
    pub fn with_labels(mut self, labels: Labels) -> Self {
        self.labels = labels;
        self
    }

    /// Observe value
    pub fn observe(&mut self, value: f64) {
        self.buckets.observe(value);
    }

    /// Format as Prometheus lines
    pub fn format(&self, name: &str) -> String {
        self.buckets.format(name, &self.labels)
    }
}

impl Default for HistogramValue {
    fn default() -> Self {
        Self::new()
    }
}

/// Validate metric name (snake_case, alphanumeric + underscores)
pub fn validate_metric_name(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    let first = name.chars().next().expect("non-empty string");
    if !first.is_ascii_lowercase() && first != '_' {
        return false;
    }

    name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}
