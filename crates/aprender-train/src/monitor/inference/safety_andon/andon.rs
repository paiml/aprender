//! Safety Andon for inference monitoring

use super::emergency::EmergencyCondition;
use super::sil::SafetyIntegrityLevel;
use crate::monitor::andon::{Alert, AndonSystem};
use crate::monitor::inference::path::DecisionPath;
use crate::monitor::inference::trace::DecisionTrace;

/// Safety Andon for inference monitoring
///
/// # Features
/// - Confidence monitoring
/// - Latency monitoring
/// - Emergency condition detection
/// - Integration with base Andon system
pub struct SafetyAndon {
    /// Base Andon system
    andon: AndonSystem,
    /// Safety integrity level
    sil: SafetyIntegrityLevel,
    /// Minimum acceptable confidence
    pub(crate) min_confidence: f32,
    /// Maximum acceptable latency in nanoseconds
    pub(crate) max_latency_ns: u64,
    /// Consecutive low-confidence counter
    low_confidence_count: usize,
    /// Threshold for low confidence alert
    pub(crate) low_confidence_threshold: usize,
    /// Alert on unknown classification
    pub(crate) alert_on_unknown: bool,
}

impl SafetyAndon {
    /// Create a new safety Andon system
    pub fn new(sil: SafetyIntegrityLevel) -> Self {
        Self {
            andon: AndonSystem::new(),
            min_confidence: sil.min_confidence(),
            max_latency_ns: sil.max_latency_ns(),
            sil,
            low_confidence_count: 0,
            low_confidence_threshold: 5,
            alert_on_unknown: true,
        }
    }

    /// Set custom confidence threshold
    pub fn with_min_confidence(mut self, threshold: f32) -> Self {
        self.min_confidence = threshold;
        self
    }

    /// Set custom latency threshold
    pub fn with_max_latency_ns(mut self, max_ns: u64) -> Self {
        self.max_latency_ns = max_ns;
        self
    }

    /// Set consecutive low-confidence threshold
    pub fn with_low_confidence_threshold(mut self, threshold: usize) -> Self {
        self.low_confidence_threshold = threshold;
        self
    }

    /// Disable unknown classification alerts
    pub fn without_unknown_alerts(mut self) -> Self {
        self.alert_on_unknown = false;
        self
    }

    /// Check a trace against safety rules
    pub fn check_trace<P: DecisionPath>(
        &mut self,
        trace: &DecisionTrace<P>,
        latency_budget_ns: u64,
    ) {
        let confidence = trace.confidence();
        let latency_ns = trace.latency_ns;

        // Check for invalid output
        if trace.output.is_nan() || trace.output.is_infinite() {
            self.trigger_emergency(EmergencyCondition::InvalidOutput);
            return;
        }

        // Check confidence
        if confidence < self.min_confidence {
            self.low_confidence_count += 1;

            if self.low_confidence_count >= self.low_confidence_threshold {
                self.trigger_emergency(EmergencyCondition::ConsecutiveLowConfidence {
                    count: self.low_confidence_count,
                    threshold: self.min_confidence,
                });
            } else {
                self.andon.trigger(
                    Alert::warning(format!(
                        "Low confidence: {:.1}% (threshold: {:.1}%)",
                        confidence * 100.0,
                        self.min_confidence * 100.0
                    ))
                    .with_source("SafetyAndon")
                    .with_value(f64::from(confidence)),
                );
            }
        } else {
            self.low_confidence_count = 0;
        }

        // Check latency
        let effective_budget = latency_budget_ns.min(self.max_latency_ns);
        if latency_ns > effective_budget {
            let latency_ms = latency_ns as f64 / 1_000_000.0;
            let budget_ms = effective_budget as f64 / 1_000_000.0;

            if latency_ns > self.max_latency_ns * 2 {
                // Critical: more than 2x over budget
                self.trigger_emergency(EmergencyCondition::DecisionTimeout {
                    max_ms: budget_ms as f32,
                });
            } else {
                self.andon.trigger(
                    Alert::warning(format!(
                        "Latency exceeded: {latency_ms:.2}ms > {budget_ms:.2}ms budget"
                    ))
                    .with_source("SafetyAndon")
                    .with_value(latency_ms),
                );
            }
        }
    }

    /// Trigger an emergency condition
    pub fn trigger_emergency(&mut self, condition: EmergencyCondition) {
        let alert =
            Alert::new(condition.alert_level(), condition.message()).with_source("SafetyAndon");
        self.andon.trigger(alert);
    }

    /// Check if stop has been requested
    pub fn should_stop(&self) -> bool {
        self.andon.should_stop()
    }

    /// Reset the Andon system
    pub fn reset(&mut self) {
        self.andon.reset();
        self.low_confidence_count = 0;
    }

    /// Get alert history
    pub fn history(&self) -> &[Alert] {
        self.andon.history()
    }

    /// Get the safety integrity level
    pub fn sil(&self) -> SafetyIntegrityLevel {
        self.sil
    }

    /// Get reference to the underlying Andon system
    pub fn andon(&self) -> &AndonSystem {
        &self.andon
    }
}

impl Default for SafetyAndon {
    fn default() -> Self {
        Self::new(SafetyIntegrityLevel::QM)
    }
}
