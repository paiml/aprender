//! Real-Time Alert Integration System (PMAT-040)
//!
//! Vendor-agnostic alert routing for anomaly detection with webhook support.
//!
//! # Features
//!
//! - Multi-channel alert routing (Slack, PagerDuty, Email, Webhook)
//! - Alert severity levels (INFO, WARNING, CRITICAL)
//! - Rate limiting and deduplication
//! - Message templating
//! - Dry-run mode for testing
//!
//! # Falsification Criteria (F1321-F1330)
//!
//! See `tests/alerting_f1321.rs` for falsification tests.

mod router;
mod types;

pub use router::{AlertRouter, AlertRouterConfig};
pub use types::{Alert, AlertChannel, AlertSeverity, DeliveryResult};

/// Message template for alerts
#[derive(Debug, Clone)]
pub struct MessageTemplate {
    /// Template string with placeholders
    pub template: String,
}

impl MessageTemplate {
    /// Create new template
    pub fn new(template: &str) -> Self {
        Self { template: template.to_string() }
    }

    /// Render template with alert data
    pub fn render(&self, alert: &Alert) -> String {
        self.template
            .replace("{title}", &alert.title)
            .replace("{message}", &alert.message)
            .replace("{severity}", alert.severity.name())
            .replace("{source}", &alert.source)
            .replace("{value}", &alert.value.map(|v| format!("{:.2}", v)).unwrap_or_default())
            .replace(
                "{threshold}",
                &alert.threshold.map(|t| format!("{:.2}", t)).unwrap_or_default(),
            )
    }
}

/// Create alert from anomaly detection result
pub fn alert_from_anomaly(
    metric: &str,
    value: f64,
    expected: f64,
    severity: AlertSeverity,
) -> Alert {
    let deviation = ((value - expected) / expected * 100.0).abs();
    Alert::new(
        &format!("Anomaly detected: {}", metric),
        &format!("{} deviated {:.1}% from expected value {:.2}", metric, deviation, expected),
        severity,
    )
    .with_source(metric)
    .with_value(value)
    .with_threshold(expected)
}

#[cfg(test)]
mod tests;
