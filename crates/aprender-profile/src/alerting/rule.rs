//! Alert rule definitions
//!
//! Defines alert rules that can be evaluated against metrics.
//! Supports threshold, rate, and absence alert types.

use std::time::Duration;

pub use super::state::Severity;

/// Alert rule definition
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Unique rule name
    pub name: String,
    /// Alert expression
    pub expr: AlertExpr,
    /// Duration condition must be true before firing
    pub for_duration: Duration,
    /// Severity level
    pub severity: Severity,
    /// Annotations (summary, description)
    pub annotations: super::state::AlertAnnotations,
    /// Labels to add to alert
    pub labels: std::collections::HashMap<String, String>,
}

impl AlertRule {
    /// Create a new threshold alert rule
    pub fn threshold(
        name: impl Into<String>,
        metric: impl Into<String>,
        op: CompareOp,
        value: f64,
        for_duration: Duration,
        severity: Severity,
    ) -> Self {
        Self {
            name: name.into(),
            expr: AlertExpr::Threshold { metric: metric.into(), op, value },
            for_duration,
            severity,
            annotations: super::state::AlertAnnotations::default(),
            labels: std::collections::HashMap::new(),
        }
    }

    /// Create a rate alert rule
    pub fn rate(
        name: impl Into<String>,
        metric: impl Into<String>,
        window: Duration,
        op: CompareOp,
        threshold: f64,
        for_duration: Duration,
        severity: Severity,
    ) -> Self {
        Self {
            name: name.into(),
            expr: AlertExpr::Rate { metric: metric.into(), window, op, threshold },
            for_duration,
            severity,
            annotations: super::state::AlertAnnotations::default(),
            labels: std::collections::HashMap::new(),
        }
    }

    /// Create an absence alert rule
    pub fn absent(
        name: impl Into<String>,
        metric: impl Into<String>,
        for_duration: Duration,
        severity: Severity,
    ) -> Self {
        Self {
            name: name.into(),
            expr: AlertExpr::Absent { metric: metric.into() },
            for_duration,
            severity,
            annotations: super::state::AlertAnnotations::default(),
            labels: std::collections::HashMap::new(),
        }
    }

    /// Set annotations
    pub fn with_annotations(mut self, annotations: super::state::AlertAnnotations) -> Self {
        self.annotations = annotations;
        self
    }

    /// Set summary annotation
    pub fn with_summary(mut self, summary: impl Into<String>) -> Self {
        self.annotations.summary = summary.into();
        self
    }

    /// Add a label
    pub fn with_label(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.labels.insert(key.into(), value.into());
        self
    }
}

/// Alert expression types
#[derive(Debug, Clone)]
pub enum AlertExpr {
    /// Simple threshold comparison
    Threshold { metric: String, op: CompareOp, value: f64 },
    /// Rate of change over window
    Rate { metric: String, window: Duration, op: CompareOp, threshold: f64 },
    /// Metric absence detection
    Absent { metric: String },
    /// Anomaly detection score
    Anomaly { metric: String, threshold: f64 },
    /// Histogram quantile
    Quantile { metric: String, quantile: f64, op: CompareOp, value: f64 },
}

impl AlertExpr {
    /// Get the metric name referenced by this expression
    pub fn metric_name(&self) -> &str {
        match self {
            AlertExpr::Threshold { metric, .. } => metric,
            AlertExpr::Rate { metric, .. } => metric,
            AlertExpr::Absent { metric } => metric,
            AlertExpr::Anomaly { metric, .. } => metric,
            AlertExpr::Quantile { metric, .. } => metric,
        }
    }
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CompareOp {
    Greater,
    GreaterEqual,
    Less,
    LessEqual,
    Equal,
    NotEqual,
}

impl CompareOp {
    /// Evaluate comparison
    pub fn compare(&self, left: f64, right: f64) -> bool {
        match self {
            CompareOp::Greater => left > right,
            CompareOp::GreaterEqual => left >= right,
            CompareOp::Less => left < right,
            CompareOp::LessEqual => left <= right,
            CompareOp::Equal => (left - right).abs() < f64::EPSILON,
            CompareOp::NotEqual => (left - right).abs() >= f64::EPSILON,
        }
    }
}

impl std::fmt::Display for CompareOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompareOp::Greater => write!(f, ">"),
            CompareOp::GreaterEqual => write!(f, ">="),
            CompareOp::Less => write!(f, "<"),
            CompareOp::LessEqual => write!(f, "<="),
            CompareOp::Equal => write!(f, "=="),
            CompareOp::NotEqual => write!(f, "!="),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_threshold_rule() {
        let rule = AlertRule::threshold(
            "high_latency",
            "request_latency_seconds",
            CompareOp::Greater,
            0.1,
            Duration::from_secs(60),
            Severity::Warning,
        );

        assert_eq!(rule.name, "high_latency");
        assert_eq!(rule.for_duration, Duration::from_secs(60));
        assert_eq!(rule.severity, Severity::Warning);

        if let AlertExpr::Threshold { metric, op, value } = &rule.expr {
            assert_eq!(metric, "request_latency_seconds");
            assert_eq!(*op, CompareOp::Greater);
            assert!((value - 0.1).abs() < f64::EPSILON);
        } else {
            panic!("expected threshold expression");
        }
    }

    #[test]
    fn test_rate_rule() {
        let rule = AlertRule::rate(
            "high_error_rate",
            "errors_total",
            Duration::from_secs(300),
            CompareOp::Greater,
            100.0,
            Duration::from_secs(60),
            Severity::Critical,
        );

        if let AlertExpr::Rate { metric, window, .. } = &rule.expr {
            assert_eq!(metric, "errors_total");
            assert_eq!(*window, Duration::from_secs(300));
        } else {
            panic!("expected rate expression");
        }
    }

    #[test]
    fn test_absent_rule() {
        let rule = AlertRule::absent(
            "no_heartbeat",
            "heartbeat_total",
            Duration::from_secs(300),
            Severity::Critical,
        );

        if let AlertExpr::Absent { metric } = &rule.expr {
            assert_eq!(metric, "heartbeat_total");
        } else {
            panic!("expected absent expression");
        }
    }

    #[test]
    fn test_compare_ops() {
        assert!(CompareOp::Greater.compare(5.0, 3.0));
        assert!(!CompareOp::Greater.compare(3.0, 5.0));

        assert!(CompareOp::GreaterEqual.compare(5.0, 5.0));
        assert!(CompareOp::GreaterEqual.compare(5.0, 3.0));

        assert!(CompareOp::Less.compare(3.0, 5.0));
        assert!(CompareOp::LessEqual.compare(5.0, 5.0));

        assert!(CompareOp::Equal.compare(5.0, 5.0));
        assert!(CompareOp::NotEqual.compare(5.0, 3.0));
    }

    #[test]
    fn test_rule_with_annotations() {
        let rule = AlertRule::threshold(
            "test",
            "metric",
            CompareOp::Greater,
            0.0,
            Duration::ZERO,
            Severity::Info,
        )
        .with_summary("Test alert: {{$value}}")
        .with_label("team", "platform");

        assert_eq!(rule.annotations.summary, "Test alert: {{$value}}");
        assert_eq!(rule.labels.get("team"), Some(&"platform".to_string()));
    }

    #[test]
    fn test_alert_expr_metric_name_all_variants() {
        // Threshold
        let threshold = AlertExpr::Threshold {
            metric: "cpu_usage".to_string(),
            op: CompareOp::Greater,
            value: 80.0,
        };
        assert_eq!(threshold.metric_name(), "cpu_usage");

        // Rate
        let rate = AlertExpr::Rate {
            metric: "errors_total".to_string(),
            window: Duration::from_secs(60),
            op: CompareOp::Greater,
            threshold: 10.0,
        };
        assert_eq!(rate.metric_name(), "errors_total");

        // Absent
        let absent = AlertExpr::Absent { metric: "heartbeat".to_string() };
        assert_eq!(absent.metric_name(), "heartbeat");

        // Anomaly
        let anomaly = AlertExpr::Anomaly { metric: "response_time".to_string(), threshold: 2.5 };
        assert_eq!(anomaly.metric_name(), "response_time");

        // Quantile
        let quantile = AlertExpr::Quantile {
            metric: "latency".to_string(),
            quantile: 0.99,
            op: CompareOp::Greater,
            value: 0.5,
        };
        assert_eq!(quantile.metric_name(), "latency");
    }

    #[test]
    fn test_compare_op_display() {
        assert_eq!(format!("{}", CompareOp::Greater), ">");
        assert_eq!(format!("{}", CompareOp::GreaterEqual), ">=");
        assert_eq!(format!("{}", CompareOp::Less), "<");
        assert_eq!(format!("{}", CompareOp::LessEqual), "<=");
        assert_eq!(format!("{}", CompareOp::Equal), "==");
        assert_eq!(format!("{}", CompareOp::NotEqual), "!=");
    }

    #[test]
    fn test_compare_op_edge_cases() {
        // Test equal with tiny difference
        assert!(CompareOp::Equal.compare(1.0, 1.0));
        assert!(!CompareOp::Equal.compare(1.0, 1.1));

        // Test not equal
        assert!(!CompareOp::NotEqual.compare(1.0, 1.0));
        assert!(CompareOp::NotEqual.compare(1.0, 1.1));

        // Test boundary conditions
        assert!(!CompareOp::Greater.compare(5.0, 5.0));
        assert!(CompareOp::GreaterEqual.compare(5.0, 5.0));
        assert!(!CompareOp::Less.compare(5.0, 5.0));
        assert!(CompareOp::LessEqual.compare(5.0, 5.0));
    }

    #[test]
    fn test_rule_with_annotations_struct() {
        let annotations = super::super::state::AlertAnnotations {
            summary: "High CPU".to_string(),
            description: "CPU usage exceeds threshold".to_string(),
            runbook_url: None,
        };

        let rule = AlertRule::threshold(
            "cpu_alert",
            "cpu_percent",
            CompareOp::Greater,
            90.0,
            Duration::from_secs(30),
            Severity::Critical,
        )
        .with_annotations(annotations);

        assert_eq!(rule.annotations.summary, "High CPU");
        assert_eq!(rule.annotations.description, "CPU usage exceeds threshold");
    }

    #[test]
    fn test_rule_clone() {
        let rule = AlertRule::threshold(
            "test",
            "metric",
            CompareOp::Greater,
            100.0,
            Duration::from_secs(60),
            Severity::Warning,
        )
        .with_label("env", "prod");

        let cloned = rule.clone();
        assert_eq!(cloned.name, rule.name);
        assert_eq!(cloned.labels.get("env"), Some(&"prod".to_string()));
    }

    #[test]
    fn test_alert_expr_clone() {
        let expr = AlertExpr::Quantile {
            metric: "latency".to_string(),
            quantile: 0.95,
            op: CompareOp::Greater,
            value: 0.1,
        };

        let cloned = expr.clone();
        assert_eq!(cloned.metric_name(), "latency");
    }

    #[test]
    fn test_multiple_labels() {
        let rule = AlertRule::threshold(
            "multi_label",
            "metric",
            CompareOp::Greater,
            0.0,
            Duration::ZERO,
            Severity::Info,
        )
        .with_label("team", "platform")
        .with_label("service", "api")
        .with_label("env", "production");

        assert_eq!(rule.labels.len(), 3);
        assert_eq!(rule.labels.get("team"), Some(&"platform".to_string()));
        assert_eq!(rule.labels.get("service"), Some(&"api".to_string()));
        assert_eq!(rule.labels.get("env"), Some(&"production".to_string()));
    }
}
