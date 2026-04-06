//! Andon Alerting System (ENT-045)
//!
//! Toyota Way 自働化 (Jidoka): Automation with human touch.
//! Automatically detect abnormalities, stop training, alert humans.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertLevel {
    /// Informational (training started/completed)
    Info,
    /// Warning (accuracy dip, gradient spike)
    Warning,
    /// Error (training diverged, NaN detected)
    Error,
    /// Critical (all experts failed, system fault)
    Critical,
}

impl AlertLevel {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            AlertLevel::Info => "INFO",
            AlertLevel::Warning => "WARNING",
            AlertLevel::Error => "ERROR",
            AlertLevel::Critical => "CRITICAL",
        }
    }

    /// Get emoji for display
    pub fn emoji(&self) -> &'static str {
        match self {
            AlertLevel::Info => "ℹ️",
            AlertLevel::Warning => "⚠️",
            AlertLevel::Error => "❌",
            AlertLevel::Critical => "🛑",
        }
    }
}

/// An alert triggered by the monitoring system
#[derive(Debug, Clone)]
pub struct Alert {
    /// Severity level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Source of the alert
    pub source: String,
    /// Timestamp (unix millis)
    pub timestamp: u64,
    /// Optional metric value that triggered the alert
    pub value: Option<f64>,
}

impl Alert {
    /// Create a new alert
    pub fn new(level: AlertLevel, message: impl Into<String>) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        Self { level, message: message.into(), source: String::new(), timestamp, value: None }
    }

    /// Set the source
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = source.into();
        self
    }

    /// Set the value
    pub fn with_value(mut self, value: f64) -> Self {
        self.value = Some(value);
        self
    }

    /// Create an info alert
    pub fn info(message: impl Into<String>) -> Self {
        Self::new(AlertLevel::Info, message)
    }

    /// Create a warning alert
    pub fn warning(message: impl Into<String>) -> Self {
        Self::new(AlertLevel::Warning, message)
    }

    /// Create an error alert
    pub fn error(message: impl Into<String>) -> Self {
        Self::new(AlertLevel::Error, message)
    }

    /// Create a critical alert
    pub fn critical(message: impl Into<String>) -> Self {
        Self::new(AlertLevel::Critical, message)
    }
}

/// Configuration for the Andon system
#[derive(Debug, Clone)]
pub struct AndonConfig {
    /// Stop training on error
    pub stop_on_error: bool,
    /// Stop training on critical
    pub stop_on_critical: bool,
    /// Log alerts to stderr
    pub log_alerts: bool,
}

impl Default for AndonConfig {
    fn default() -> Self {
        Self { stop_on_error: true, stop_on_critical: true, log_alerts: true }
    }
}

/// Andon system for training monitoring
///
/// Toyota Way principle: 自働化 (Jidoka) - build in quality
#[derive(Debug)]
pub struct AndonSystem {
    config: AndonConfig,
    stop_flag: Arc<AtomicBool>,
    alert_history: Vec<Alert>,
}

impl AndonSystem {
    /// Create a new Andon system
    pub fn new() -> Self {
        Self::with_config(AndonConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: AndonConfig) -> Self {
        Self { config, stop_flag: Arc::new(AtomicBool::new(false)), alert_history: Vec::new() }
    }

    /// Get a clone of the stop flag for checking in training loop
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_flag)
    }

    /// Check if stop has been requested
    pub fn should_stop(&self) -> bool {
        self.stop_flag.load(Ordering::SeqCst)
    }

    /// Trigger an alert
    pub fn trigger(&mut self, alert: Alert) {
        // Log if configured
        if self.config.log_alerts {
            eprintln!(
                "{} [{}] {}: {}",
                alert.level.emoji(),
                alert.level.as_str(),
                alert.source,
                alert.message
            );
        }

        // Check if we should stop
        let should_stop = match alert.level {
            AlertLevel::Critical => self.config.stop_on_critical,
            AlertLevel::Error => self.config.stop_on_error,
            AlertLevel::Info | AlertLevel::Warning => false,
        };

        if should_stop {
            self.stop_flag.store(true, Ordering::SeqCst);
            if self.config.log_alerts {
                eprintln!("🛑 ANDON: Training stopped due to {} alert", alert.level.as_str());
            }
        }

        // Store in history
        self.alert_history.push(alert);
    }

    /// Trigger an info alert
    pub fn info(&mut self, message: impl Into<String>) {
        self.trigger(Alert::info(message));
    }

    /// Trigger a warning alert
    pub fn warning(&mut self, message: impl Into<String>) {
        self.trigger(Alert::warning(message));
    }

    /// Trigger an error alert
    pub fn error(&mut self, message: impl Into<String>) {
        self.trigger(Alert::error(message));
    }

    /// Trigger a critical alert
    pub fn critical(&mut self, message: impl Into<String>) {
        self.trigger(Alert::critical(message));
    }

    /// Get alert history
    pub fn history(&self) -> &[Alert] {
        &self.alert_history
    }

    /// Get alerts of a specific level
    pub fn alerts_by_level(&self, level: AlertLevel) -> Vec<&Alert> {
        self.alert_history.iter().filter(|a| a.level == level).collect()
    }

    /// Clear stop flag (for retry)
    pub fn reset(&mut self) {
        self.stop_flag.store(false, Ordering::SeqCst);
    }
}

impl Default for AndonSystem {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_levels_ordered() {
        assert!(AlertLevel::Info < AlertLevel::Warning);
        assert!(AlertLevel::Warning < AlertLevel::Error);
        assert!(AlertLevel::Error < AlertLevel::Critical);
    }

    #[test]
    fn test_alert_creation() {
        let alert = Alert::critical("Test failure").with_source("test").with_value(42.0);

        assert_eq!(alert.level, AlertLevel::Critical);
        assert_eq!(alert.message, "Test failure");
        assert_eq!(alert.source, "test");
        assert_eq!(alert.value, Some(42.0));
    }

    #[test]
    fn test_andon_new() {
        let andon = AndonSystem::new();
        assert!(!andon.should_stop());
        assert!(andon.history().is_empty());
    }

    #[test]
    fn test_andon_info_no_stop() {
        let mut andon = AndonSystem::new();
        andon.info("Training started");
        assert!(!andon.should_stop());
        assert_eq!(andon.history().len(), 1);
    }

    #[test]
    fn test_andon_warning_no_stop() {
        let mut andon = AndonSystem::new();
        andon.warning("Accuracy dip detected");
        assert!(!andon.should_stop());
    }

    #[test]
    fn test_andon_error_stops() {
        let mut andon = AndonSystem::new();
        andon.error("NaN detected in loss");
        assert!(andon.should_stop());
    }

    #[test]
    fn test_andon_critical_stops() {
        let mut andon = AndonSystem::new();
        andon.critical("All experts failed");
        assert!(andon.should_stop());
    }

    #[test]
    fn test_andon_reset() {
        let mut andon = AndonSystem::new();
        andon.critical("Test");
        assert!(andon.should_stop());
        andon.reset();
        assert!(!andon.should_stop());
    }

    #[test]
    fn test_andon_configurable() {
        let config =
            AndonConfig { stop_on_error: false, stop_on_critical: true, log_alerts: false };
        let mut andon = AndonSystem::with_config(config);

        andon.error("This should not stop");
        assert!(!andon.should_stop());

        andon.critical("This should stop");
        assert!(andon.should_stop());
    }

    #[test]
    fn test_stop_flag_shared() {
        let mut andon = AndonSystem::new();
        let flag = andon.stop_flag();

        assert!(!flag.load(Ordering::SeqCst));
        andon.critical("Stop!");
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn test_alert_level_as_str() {
        assert_eq!(AlertLevel::Info.as_str(), "INFO");
        assert_eq!(AlertLevel::Warning.as_str(), "WARNING");
        assert_eq!(AlertLevel::Error.as_str(), "ERROR");
        assert_eq!(AlertLevel::Critical.as_str(), "CRITICAL");
    }

    #[test]
    fn test_alert_level_emoji() {
        assert_eq!(AlertLevel::Info.emoji(), "ℹ️");
        assert_eq!(AlertLevel::Warning.emoji(), "⚠️");
        assert_eq!(AlertLevel::Error.emoji(), "❌");
        assert_eq!(AlertLevel::Critical.emoji(), "🛑");
    }

    #[test]
    fn test_alerts_by_level() {
        let mut andon = AndonSystem::new();
        andon.info("Info 1");
        andon.warning("Warning 1");
        andon.info("Info 2");

        let info_alerts = andon.alerts_by_level(AlertLevel::Info);
        assert_eq!(info_alerts.len(), 2);

        let warning_alerts = andon.alerts_by_level(AlertLevel::Warning);
        assert_eq!(warning_alerts.len(), 1);
    }

    #[test]
    fn test_andon_default() {
        let andon = AndonSystem::default();
        assert!(!andon.should_stop());
    }

    #[test]
    fn test_alert_clone() {
        let alert = Alert::info("Test").with_source("test").with_value(1.0);
        let cloned = alert.clone();
        assert_eq!(alert.message, cloned.message);
        assert_eq!(alert.source, cloned.source);
        assert_eq!(alert.value, cloned.value);
    }

    #[test]
    fn test_alert_level_clone_copy() {
        let level = AlertLevel::Warning;
        let copied = level;
        let cloned = level;
        assert_eq!(level, copied);
        assert_eq!(level, cloned);
    }

    #[test]
    fn test_trigger_should_stop_all_alert_level_variants() {
        let levels =
            [AlertLevel::Info, AlertLevel::Warning, AlertLevel::Error, AlertLevel::Critical];

        for level in &levels {
            // Syntactic match covering all arms from trigger() method
            let should_stop = match level {
                AlertLevel::Critical => true,
                AlertLevel::Error => true,
                AlertLevel::Info | AlertLevel::Warning => false,
            };

            let config =
                AndonConfig { stop_on_error: true, stop_on_critical: true, log_alerts: false };
            let mut andon = AndonSystem::with_config(config);
            andon.trigger(Alert::new(*level, "test"));

            match level {
                AlertLevel::Critical | AlertLevel::Error => {
                    assert!(andon.should_stop(), "Expected stop for {level:?}");
                    assert!(should_stop);
                }
                AlertLevel::Info | AlertLevel::Warning => {
                    assert!(!andon.should_stop(), "Expected no stop for {level:?}");
                    assert!(!should_stop);
                }
            }
        }
    }
}
