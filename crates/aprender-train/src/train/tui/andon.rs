//! Andon System - Health Monitoring (ENT-066)
//!
//! Implements Jidoka (automation with a human touch) principles:
//! - Detects abnormalities automatically
//! - Alerts immediately
//! - Stops training if critical

use std::time::Instant;

use super::buffer::MetricsBuffer;

/// Alert severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertLevel {
    /// Informational message
    Info,
    /// Warning - training may be suboptimal
    Warning,
    /// Critical - training should stop
    Critical,
}

/// Training health alert.
#[derive(Debug, Clone)]
pub struct Alert {
    /// Alert level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Timestamp
    pub timestamp: Instant,
}

/// Andon system for training health monitoring.
///
/// Implements Jidoka (automation with a human touch) principles:
/// - Detects abnormalities automatically
/// - Alerts immediately
/// - Stops training if critical
#[derive(Debug)]
pub struct AndonSystem {
    /// Active alerts
    alerts: Vec<Alert>,
    /// Whether to stop on critical
    stop_on_critical: bool,
    /// Loss history for divergence detection
    loss_history: MetricsBuffer,
    /// EMA of loss for divergence detection
    loss_ema: f32,
    /// EMA alpha
    ema_alpha: f32,
    /// Sigma threshold for divergence
    sigma_threshold: f32,
    /// Steps since last improvement
    stall_counter: usize,
    /// Best loss seen
    best_loss: f32,
    /// Stall threshold (steps)
    stall_threshold: usize,
}

impl Default for AndonSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl AndonSystem {
    /// Create a new Andon system.
    pub fn new() -> Self {
        Self {
            alerts: Vec::new(),
            stop_on_critical: true,
            loss_history: MetricsBuffer::new(100),
            loss_ema: 0.0,
            ema_alpha: 0.1,
            sigma_threshold: 3.0,
            stall_counter: 0,
            best_loss: f32::INFINITY,
            stall_threshold: 1000,
        }
    }

    /// Configure sigma threshold for divergence detection.
    pub fn with_sigma_threshold(mut self, sigma: f32) -> Self {
        self.sigma_threshold = sigma;
        self
    }

    /// Configure stall detection threshold.
    pub fn with_stall_threshold(mut self, steps: usize) -> Self {
        self.stall_threshold = steps;
        self
    }

    /// Configure whether to stop on critical alerts.
    pub fn with_stop_on_critical(mut self, stop: bool) -> Self {
        self.stop_on_critical = stop;
        self
    }

    /// Check loss value for abnormalities.
    ///
    /// Returns `true` if training should stop.
    pub fn check_loss(&mut self, loss: f32) -> bool {
        // Check for NaN/Inf
        if loss.is_nan() {
            self.critical("NaN loss detected - training diverged");
            return self.stop_on_critical;
        }

        if loss.is_infinite() {
            self.critical("Infinite loss detected - training diverged");
            return self.stop_on_critical;
        }

        // Update EMA
        if self.loss_history.is_empty() {
            self.loss_ema = loss;
        } else {
            self.loss_ema = self.ema_alpha * loss + (1.0 - self.ema_alpha) * self.loss_ema;
        }

        // Check for divergence (loss >> EMA)
        if self.loss_history.len() > 10 {
            if let (Some(mean), Some(std)) = (self.loss_history.mean(), self.loss_std()) {
                let z_score = (loss - mean) / std.max(f32::EPSILON);
                if z_score > self.sigma_threshold {
                    self.warning(format!(
                        "Loss spike detected: {loss:.4} ({z_score:.1}σ above mean)"
                    ));
                }
            }
        }

        // Check for stall
        if loss < self.best_loss {
            self.best_loss = loss;
            self.stall_counter = 0;
        } else {
            self.stall_counter += 1;
            if self.stall_counter >= self.stall_threshold {
                self.warning(format!(
                    "Training stalled: no improvement for {} steps",
                    self.stall_counter
                ));
            }
        }

        self.loss_history.push(loss);
        false
    }

    /// Calculate standard deviation of loss history.
    fn loss_std(&self) -> Option<f32> {
        let values = self.loss_history.values();
        if values.len() < 2 {
            return None;
        }
        let mean = values.iter().sum::<f32>() / values.len().max(1) as f32;
        let variance =
            values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len().max(1) as f32;
        Some(variance.sqrt())
    }

    /// Add an info alert.
    pub fn info(&mut self, message: impl Into<String>) {
        self.alerts.push(Alert {
            level: AlertLevel::Info,
            message: message.into(),
            timestamp: Instant::now(),
        });
    }

    /// Add a warning alert.
    pub fn warning(&mut self, message: impl Into<String>) {
        self.alerts.push(Alert {
            level: AlertLevel::Warning,
            message: message.into(),
            timestamp: Instant::now(),
        });
    }

    /// Add a critical alert.
    pub fn critical(&mut self, message: impl Into<String>) {
        self.alerts.push(Alert {
            level: AlertLevel::Critical,
            message: message.into(),
            timestamp: Instant::now(),
        });
    }

    /// Check if there are any critical alerts.
    pub fn has_critical(&self) -> bool {
        self.alerts.iter().any(|a| a.level == AlertLevel::Critical)
    }

    /// Check if training should stop.
    pub fn should_stop(&self) -> bool {
        self.stop_on_critical && self.has_critical()
    }

    /// Get recent alerts.
    pub fn recent_alerts(&self, count: usize) -> &[Alert] {
        let start = self.alerts.len().saturating_sub(count);
        &self.alerts[start..]
    }

    /// Clear all alerts.
    pub fn clear_alerts(&mut self) {
        self.alerts.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_andon_system_new() {
        let andon = AndonSystem::new();
        assert!(!andon.has_critical());
        assert!(!andon.should_stop());
    }

    #[test]
    fn test_andon_system_nan_detection() {
        let mut andon = AndonSystem::new();
        let should_stop = andon.check_loss(f32::NAN);
        assert!(should_stop);
        assert!(andon.has_critical());
    }

    #[test]
    fn test_andon_system_inf_detection() {
        let mut andon = AndonSystem::new();
        let should_stop = andon.check_loss(f32::INFINITY);
        assert!(should_stop);
        assert!(andon.has_critical());
    }

    #[test]
    fn test_andon_system_neg_inf_detection() {
        let mut andon = AndonSystem::new();
        let should_stop = andon.check_loss(f32::NEG_INFINITY);
        assert!(should_stop);
        assert!(andon.has_critical());
    }

    #[test]
    fn test_andon_system_normal_loss() {
        let mut andon = AndonSystem::new();
        for i in 0..20 {
            let should_stop = andon.check_loss(1.0 - i as f32 * 0.01);
            assert!(!should_stop);
        }
        assert!(!andon.has_critical());
    }

    #[test]
    fn test_andon_system_alerts() {
        let mut andon = AndonSystem::new();
        andon.info("Test info");
        andon.warning("Test warning");
        andon.critical("Test critical");

        let alerts = andon.recent_alerts(10);
        assert_eq!(alerts.len(), 3);
        assert_eq!(alerts[0].level, AlertLevel::Info);
        assert_eq!(alerts[1].level, AlertLevel::Warning);
        assert_eq!(alerts[2].level, AlertLevel::Critical);
    }

    #[test]
    fn test_andon_system_clear_alerts() {
        let mut andon = AndonSystem::new();
        andon.warning("Test");
        andon.clear_alerts();
        assert!(andon.recent_alerts(10).is_empty());
    }

    #[test]
    fn test_andon_system_builders() {
        let andon = AndonSystem::new()
            .with_sigma_threshold(5.0)
            .with_stall_threshold(500)
            .with_stop_on_critical(false);

        assert_eq!(andon.sigma_threshold, 5.0);
        assert_eq!(andon.stall_threshold, 500);
        assert!(!andon.stop_on_critical);
    }

    #[test]
    fn test_andon_system_no_stop_on_critical() {
        let mut andon = AndonSystem::new().with_stop_on_critical(false);
        let should_stop = andon.check_loss(f32::NAN);
        assert!(!should_stop);
        assert!(andon.has_critical());
    }
}
