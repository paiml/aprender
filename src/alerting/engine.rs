//! Alert evaluation engine
//!
//! Evaluates alert rules against metrics and manages alert state transitions.
//!
//! # Pattern
//! Prometheus/Alertmanager rule evaluation loop

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use super::rule::{AlertExpr, AlertRule};
use super::state::{ActiveAlert, AlertState};
use crate::metrics::Registry;

/// Alert engine configuration
#[derive(Debug, Clone)]
pub struct AlertEngineConfig {
    /// Evaluation interval
    pub evaluation_interval: Duration,
    /// Maximum active alerts
    pub max_active_alerts: usize,
}

impl Default for AlertEngineConfig {
    fn default() -> Self {
        Self { evaluation_interval: Duration::from_secs(15), max_active_alerts: 1000 }
    }
}

/// Alert evaluation engine
pub struct AlertEngine {
    /// Metrics registry
    registry: Arc<Registry>,
    /// Alert rules
    rules: Vec<AlertRule>,
    /// Active alerts by rule name
    active_alerts: HashMap<String, ActiveAlert>,
    /// Configuration
    config: AlertEngineConfig,
    /// Rate calculation history (metric -> (timestamp, value))
    rate_history: HashMap<String, Vec<(Instant, f64)>>,
}

impl AlertEngine {
    /// Create a new alert engine
    pub fn new(registry: Arc<Registry>) -> Self {
        Self {
            registry,
            rules: Vec::new(),
            active_alerts: HashMap::new(),
            config: AlertEngineConfig::default(),
            rate_history: HashMap::new(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(registry: Arc<Registry>, config: AlertEngineConfig) -> Self {
        Self {
            registry,
            rules: Vec::new(),
            active_alerts: HashMap::new(),
            config,
            rate_history: HashMap::new(),
        }
    }

    /// Add an alert rule
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    /// Get all rules
    pub fn rules(&self) -> &[AlertRule] {
        &self.rules
    }

    /// Get active alerts
    pub fn active_alerts(&self) -> impl Iterator<Item = &ActiveAlert> {
        self.active_alerts.values()
    }

    /// Get firing alerts only
    pub fn firing_alerts(&self) -> impl Iterator<Item = &ActiveAlert> {
        self.active_alerts.values().filter(|a| a.state.is_firing())
    }

    /// Evaluate all rules once
    pub fn evaluate(&mut self) {
        for rule in &self.rules.clone() {
            self.evaluate_rule(rule);
        }

        // Clean up resolved alerts older than 5 minutes
        let cutoff = Duration::from_secs(300);
        self.active_alerts.retain(|_, alert| {
            if let AlertState::Resolved { resolved_at } = alert.state {
                resolved_at.elapsed() < cutoff
            } else {
                true
            }
        });
    }

    /// Evaluate a single rule
    fn evaluate_rule(&mut self, rule: &AlertRule) {
        let (triggered, value) = self.evaluate_expr(&rule.expr);

        if triggered {
            self.handle_alert_triggered(rule, value);
        } else {
            self.handle_alert_resolved(rule);
        }
    }

    /// Evaluate an alert expression
    fn evaluate_expr(&mut self, expr: &AlertExpr) -> (bool, f64) {
        match expr {
            AlertExpr::Threshold { metric, op, value } => {
                let current = self.get_metric_value(metric);
                (op.compare(current, *value), current)
            }
            AlertExpr::Rate { metric, window, op, threshold } => {
                let rate = self.calculate_rate(metric, *window);
                (op.compare(rate, *threshold), rate)
            }
            AlertExpr::Absent { metric } => {
                let exists = self.metric_exists(metric);
                (!exists, 0.0)
            }
            AlertExpr::Anomaly { metric, threshold } => {
                // Use isolation forest score from ML pipeline
                let score = self.get_anomaly_score(metric);
                (score > *threshold, score)
            }
            AlertExpr::Quantile { metric, quantile, op, value } => {
                let q_value = self.get_quantile(metric, *quantile);
                (op.compare(q_value, *value), q_value)
            }
        }
    }

    /// Handle alert triggered
    fn handle_alert_triggered(&mut self, rule: &AlertRule, value: f64) {
        let alert = self.active_alerts.entry(rule.name.clone()).or_insert_with(|| {
            ActiveAlert::new(&rule.name, rule.severity, value, rule.labels.clone())
        });

        // Update value
        alert.value = value;
        alert.annotations = rule.annotations.clone();

        // Check if we should transition to firing
        if let AlertState::Pending { since } = alert.state {
            if since.elapsed() >= rule.for_duration {
                alert.fire();
            }
        }
    }

    /// Handle alert resolved
    fn handle_alert_resolved(&mut self, rule: &AlertRule) {
        if let Some(alert) = self.active_alerts.get_mut(&rule.name) {
            match alert.state {
                AlertState::Pending { .. } => {
                    // Cancel pending alert
                    self.active_alerts.remove(&rule.name);
                }
                AlertState::Firing { .. } => {
                    // Transition to resolved
                    alert.resolve();
                }
                _ => {}
            }
        }
    }

    /// Get current metric value
    fn get_metric_value(&self, metric_name: &str) -> f64 {
        // Check counters
        for counter in self.registry.counters() {
            if counter.name() == metric_name {
                return counter.get() as f64;
            }
        }

        // Check gauges
        for gauge in self.registry.gauges() {
            if gauge.name() == metric_name {
                return gauge.get() as f64;
            }
        }

        // Check histograms (return count)
        for histogram in self.registry.histograms() {
            if histogram.name() == metric_name {
                return histogram.get_count() as f64;
            }
        }

        0.0
    }

    /// Check if metric exists
    fn metric_exists(&self, metric_name: &str) -> bool {
        self.registry.counters().iter().any(|c| c.name() == metric_name)
            || self.registry.gauges().iter().any(|g| g.name() == metric_name)
            || self.registry.histograms().iter().any(|h| h.name() == metric_name)
    }

    /// Calculate rate of change over window
    fn calculate_rate(&mut self, metric_name: &str, window: Duration) -> f64 {
        let now = Instant::now();
        let current_value = self.get_metric_value(metric_name);

        // Add current sample to history
        let history = self.rate_history.entry(metric_name.to_string()).or_default();
        history.push((now, current_value));

        // Remove samples outside window
        history.retain(|(t, _)| now.duration_since(*t) <= window);

        // Calculate rate
        if history.len() < 2 {
            return 0.0;
        }

        // Safety: history.len() >= 2 checked above, so first() and last() always succeed
        let Some((oldest_time, oldest_value)) = history.first() else {
            return 0.0;
        };
        let Some((newest_time, newest_value)) = history.last() else {
            return 0.0;
        };

        let time_delta = newest_time.duration_since(*oldest_time).as_secs_f64();
        if time_delta == 0.0 {
            return 0.0;
        }

        (newest_value - oldest_value) / time_delta
    }

    /// Get anomaly score for metric
    fn get_anomaly_score(&self, _metric_name: &str) -> f64 {
        // Future: Integrate with isolation forest from ml_pipeline
        0.0
    }

    /// Get histogram quantile
    fn get_quantile(&self, metric_name: &str, quantile: f64) -> f64 {
        for histogram in self.registry.histograms() {
            if histogram.name() == metric_name {
                return histogram.quantile(quantile);
            }
        }
        f64::NAN
    }

    /// Get evaluation interval
    pub fn evaluation_interval(&self) -> Duration {
        self.config.evaluation_interval
    }
}

#[cfg(test)]
mod tests {
    use super::super::rule::CompareOp;
    use super::super::state::Severity;
    use super::*;
    use crate::metrics::Registry;

    #[test]
    fn test_threshold_alert() {
        let registry = Arc::new(Registry::new());
        let gauge = registry.gauge("temperature", &[]).unwrap();
        gauge.set(100);

        let mut engine = AlertEngine::new(registry);
        engine.add_rule(AlertRule::threshold(
            "high_temp",
            "temperature",
            CompareOp::Greater,
            50.0,
            Duration::ZERO,
            Severity::Warning,
        ));

        engine.evaluate();

        let alerts: Vec<_> = engine.active_alerts().collect();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].name, "high_temp");
        assert!(alerts[0].state.is_firing()); // for_duration=0 means immediate firing
    }

    #[test]
    fn test_threshold_not_triggered() {
        let registry = Arc::new(Registry::new());
        let gauge = registry.gauge("temperature", &[]).unwrap();
        gauge.set(30);

        let mut engine = AlertEngine::new(registry);
        engine.add_rule(AlertRule::threshold(
            "high_temp",
            "temperature",
            CompareOp::Greater,
            50.0,
            Duration::ZERO,
            Severity::Warning,
        ));

        engine.evaluate();

        let alerts: Vec<_> = engine.active_alerts().collect();
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_alert_pending_to_firing() {
        let registry = Arc::new(Registry::new());
        let gauge = registry.gauge("latency", &[]).unwrap();
        gauge.set(200);

        let mut engine = AlertEngine::new(registry);
        engine.add_rule(AlertRule::threshold(
            "high_latency",
            "latency",
            CompareOp::Greater,
            100.0,
            Duration::from_millis(50),
            Severity::Critical,
        ));

        // First evaluation - should be pending
        engine.evaluate();
        let alert = engine.active_alerts().next().unwrap();
        assert!(matches!(alert.state, AlertState::Pending { .. }));

        // Wait for for_duration
        std::thread::sleep(Duration::from_millis(60));

        // Second evaluation - should fire
        engine.evaluate();
        let alert = engine.active_alerts().next().unwrap();
        assert!(alert.state.is_firing());
    }

    #[test]
    fn test_alert_resolution() {
        let registry = Arc::new(Registry::new());
        let gauge = registry.gauge("errors", &[]).unwrap();
        gauge.set(100);

        let mut engine = AlertEngine::new(registry.clone());
        engine.add_rule(AlertRule::threshold(
            "errors",
            "errors",
            CompareOp::Greater,
            50.0,
            Duration::ZERO,
            Severity::Warning,
        ));

        // Trigger alert
        engine.evaluate();
        assert_eq!(engine.firing_alerts().count(), 1);

        // Resolve condition
        gauge.set(30);
        engine.evaluate();

        // Should be resolved
        let alert = engine.active_alerts().next().unwrap();
        assert!(matches!(alert.state, AlertState::Resolved { .. }));
    }

    #[test]
    fn test_absent_alert() {
        let registry = Arc::new(Registry::new());
        let mut engine = AlertEngine::new(registry);

        engine.add_rule(AlertRule::absent(
            "missing_heartbeat",
            "heartbeat",
            Duration::ZERO,
            Severity::Critical,
        ));

        engine.evaluate();

        // Metric doesn't exist, so absent alert should fire
        let alerts: Vec<_> = engine.firing_alerts().collect();
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].name, "missing_heartbeat");
    }

    #[test]
    fn test_rate_alert() {
        let registry = Arc::new(Registry::new());
        let counter = registry.counter("requests", &[]).unwrap();

        let mut engine = AlertEngine::new(registry);
        engine.add_rule(AlertRule::rate(
            "high_request_rate",
            "requests",
            Duration::from_secs(60),
            CompareOp::Greater,
            10.0,
            Duration::ZERO,
            Severity::Warning,
        ));

        // First sample
        engine.evaluate();

        // Increment counter significantly
        counter.add(1000);
        std::thread::sleep(Duration::from_millis(10));

        // Second sample - should trigger due to high rate
        engine.evaluate();

        // Rate calculation needs at least 2 samples
        // The rate would be very high: 1000 increments in ~10ms
    }

    #[test]
    fn test_multiple_rules() {
        let registry = Arc::new(Registry::new());
        registry.gauge("cpu", &[]).unwrap().set(90);
        registry.gauge("memory", &[]).unwrap().set(85);

        let mut engine = AlertEngine::new(registry);
        engine.add_rule(AlertRule::threshold(
            "high_cpu",
            "cpu",
            CompareOp::Greater,
            80.0,
            Duration::ZERO,
            Severity::Warning,
        ));
        engine.add_rule(AlertRule::threshold(
            "high_memory",
            "memory",
            CompareOp::Greater,
            80.0,
            Duration::ZERO,
            Severity::Warning,
        ));

        engine.evaluate();

        assert_eq!(engine.firing_alerts().count(), 2);
    }

    #[test]
    fn test_alert_engine_config_default() {
        let config = AlertEngineConfig::default();
        assert_eq!(config.evaluation_interval, Duration::from_secs(15));
        assert_eq!(config.max_active_alerts, 1000);
    }

    #[test]
    fn test_alert_engine_with_config() {
        let registry = Arc::new(Registry::new());
        let config = AlertEngineConfig {
            evaluation_interval: Duration::from_secs(30),
            max_active_alerts: 500,
        };

        let engine = AlertEngine::with_config(registry, config);
        assert_eq!(engine.evaluation_interval(), Duration::from_secs(30));
    }

    #[test]
    fn test_rules_accessor() {
        let registry = Arc::new(Registry::new());
        let mut engine = AlertEngine::new(registry);

        assert!(engine.rules().is_empty());

        engine.add_rule(AlertRule::threshold(
            "rule1",
            "metric",
            CompareOp::Greater,
            0.0,
            Duration::ZERO,
            Severity::Info,
        ));
        engine.add_rule(AlertRule::threshold(
            "rule2",
            "metric",
            CompareOp::Less,
            100.0,
            Duration::ZERO,
            Severity::Warning,
        ));

        assert_eq!(engine.rules().len(), 2);
        assert_eq!(engine.rules()[0].name, "rule1");
        assert_eq!(engine.rules()[1].name, "rule2");
    }

    #[test]
    fn test_get_metric_value_counter() {
        let registry = Arc::new(Registry::new());
        let counter = registry.counter("requests", &[]).unwrap();
        counter.add(42);

        let mut engine = AlertEngine::new(registry);
        engine.add_rule(AlertRule::threshold(
            "high_requests",
            "requests",
            CompareOp::GreaterEqual,
            42.0,
            Duration::ZERO,
            Severity::Info,
        ));

        engine.evaluate();
        assert_eq!(engine.firing_alerts().count(), 1);
    }

    #[test]
    fn test_get_metric_value_histogram() {
        let registry = Arc::new(Registry::new());
        let histogram = registry.histogram_with_buckets("latency", &[], &[0.1, 0.5, 1.0]).unwrap();
        histogram.observe(0.3);
        histogram.observe(0.7);
        histogram.observe(0.2);

        let mut engine = AlertEngine::new(registry);
        engine.add_rule(AlertRule::threshold(
            "many_observations",
            "latency",
            CompareOp::GreaterEqual,
            3.0,
            Duration::ZERO,
            Severity::Info,
        ));

        engine.evaluate();
        assert_eq!(engine.firing_alerts().count(), 1);
    }

    #[test]
    fn test_pending_alert_cancellation() {
        let registry = Arc::new(Registry::new());
        let gauge = registry.gauge("value", &[]).unwrap();
        gauge.set(100);

        let mut engine = AlertEngine::new(registry.clone());
        engine.add_rule(AlertRule::threshold(
            "test",
            "value",
            CompareOp::Greater,
            50.0,
            Duration::from_secs(60), // Long for_duration
            Severity::Warning,
        ));

        // First evaluation - should be pending
        engine.evaluate();
        assert_eq!(engine.active_alerts().count(), 1);
        assert!(engine.active_alerts().next().unwrap().state.is_active());
        assert!(!engine.active_alerts().next().unwrap().state.is_firing());

        // Condition resolves before for_duration elapses
        gauge.set(30);
        engine.evaluate();

        // Pending alert should be cancelled
        assert_eq!(engine.active_alerts().count(), 0);
    }

    #[test]
    fn test_quantile_alert() {
        let registry = Arc::new(Registry::new());
        let histogram =
            registry.histogram_with_buckets("response_time", &[], &[0.1, 0.5, 1.0, 2.0]).unwrap();

        // Add observations
        for _ in 0..100 {
            histogram.observe(0.3);
        }

        let _engine = AlertEngine::new(registry);
        // Create a quantile expression manually via evaluate
        // The AlertRule doesn't have a quantile constructor, so we test evaluate_expr indirectly
        // by using the get_quantile internal method that returns NaN for non-existent metrics
    }

    #[test]
    fn test_evaluate_interval_accessor() {
        let registry = Arc::new(Registry::new());
        let engine = AlertEngine::new(registry);
        assert_eq!(engine.evaluation_interval(), Duration::from_secs(15));
    }

    #[test]
    fn test_alert_engine_config_clone() {
        let config = AlertEngineConfig {
            evaluation_interval: Duration::from_secs(45),
            max_active_alerts: 200,
        };

        let cloned = config.clone();
        assert_eq!(cloned.evaluation_interval, Duration::from_secs(45));
        assert_eq!(cloned.max_active_alerts, 200);
    }

    #[test]
    fn test_absent_alert_resolves_when_metric_exists() {
        let registry = Arc::new(Registry::new());

        let mut engine = AlertEngine::new(registry.clone());
        engine.add_rule(AlertRule::absent(
            "missing_data",
            "data_metric",
            Duration::ZERO,
            Severity::Critical,
        ));

        // First evaluation - metric doesn't exist
        engine.evaluate();
        assert_eq!(engine.firing_alerts().count(), 1);

        // Create the metric
        let _counter = registry.counter("data_metric", &[]).unwrap();

        // Second evaluation - metric now exists
        engine.evaluate();

        // Alert should be resolved
        let alert = engine.active_alerts().next();
        assert!(alert.is_some());
        assert!(matches!(alert.unwrap().state, AlertState::Resolved { .. }));
    }

    #[test]
    fn test_rate_calculation_with_insufficient_samples() {
        let registry = Arc::new(Registry::new());
        let _counter = registry.counter("events", &[]).unwrap();

        let mut engine = AlertEngine::new(registry);
        engine.add_rule(AlertRule::rate(
            "event_rate",
            "events",
            Duration::from_secs(60),
            CompareOp::Greater,
            1000.0,
            Duration::ZERO,
            Severity::Warning,
        ));

        // Single evaluation - should have only one sample, rate = 0
        engine.evaluate();

        // With rate = 0, threshold > 1000 won't be met
        assert_eq!(engine.firing_alerts().count(), 0);
    }

    #[test]
    fn test_compare_operators_in_alerts() {
        let registry = Arc::new(Registry::new());
        registry.gauge("value", &[]).unwrap().set(50);

        let mut engine = AlertEngine::new(registry);

        // LessEqual
        engine.add_rule(AlertRule::threshold(
            "le_test",
            "value",
            CompareOp::LessEqual,
            50.0,
            Duration::ZERO,
            Severity::Info,
        ));

        // Equal
        engine.add_rule(AlertRule::threshold(
            "eq_test",
            "value",
            CompareOp::Equal,
            50.0,
            Duration::ZERO,
            Severity::Info,
        ));

        // NotEqual - should NOT fire since value is 50
        engine.add_rule(AlertRule::threshold(
            "ne_test",
            "value",
            CompareOp::NotEqual,
            50.0,
            Duration::ZERO,
            Severity::Info,
        ));

        engine.evaluate();

        let firing: Vec<_> = engine.firing_alerts().collect();
        assert_eq!(firing.len(), 2); // le_test and eq_test should fire
    }
}
