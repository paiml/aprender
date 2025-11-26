//! Andon mechanism for synthetic data generation (Toyota Jidoka).
//!
//! The Andon system provides automatic halt and alert capabilities when
//! synthetic data generation quality degrades. Named after the Toyota
//! Production System's Andon cord that workers pull to stop the line.
//!
//! # Design Principles
//!
//! - **Jidoka**: Automation with human touch - halt on quality issues
//! - **Fast Feedback**: Alert immediately when metrics drift
//! - **Zero Tolerance**: Default to halt rather than produce garbage
//!
//! # References
//!
//! - Toyota Production System (Ohno, 1988)
//! - Code Review: automl-with-synthetic-data-review.md (2025-11-26)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Alert severity levels for Andon events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AndonSeverity {
    /// Informational - no action required.
    Info,
    /// Warning - potential issue, continue with caution.
    Warning,
    /// Critical - halt pipeline immediately.
    Critical,
}

impl std::fmt::Display for AndonSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warning => write!(f, "WARNING"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Event types that can trigger Andon alerts.
#[derive(Debug, Clone, PartialEq)]
pub enum AndonEvent {
    /// Rejection rate exceeded threshold.
    HighRejectionRate {
        /// Current rejection rate (0.0-1.0).
        rate: f32,
        /// Configured threshold.
        threshold: f32,
    },
    /// Quality score drifted below baseline.
    QualityDrift {
        /// Current quality score.
        current: f32,
        /// Historical baseline.
        baseline: f32,
    },
    /// Diversity collapsed (mode collapse detected).
    DiversityCollapse {
        /// Current diversity score.
        score: f32,
        /// Minimum acceptable diversity.
        minimum: f32,
    },
    /// Generation completely failed.
    GenerationFailure {
        /// Error message.
        message: String,
    },
}

impl AndonEvent {
    /// Get the severity of this event.
    #[must_use]
    pub fn severity(&self) -> AndonSeverity {
        match self {
            Self::HighRejectionRate { rate, threshold } => {
                // Critical if rate exceeds threshold by more than 2%
                if *rate > threshold + 0.02 {
                    AndonSeverity::Critical
                } else {
                    AndonSeverity::Warning
                }
            }
            Self::QualityDrift { current, baseline } => {
                if *current < baseline * 0.8 {
                    AndonSeverity::Critical
                } else {
                    AndonSeverity::Warning
                }
            }
            Self::DiversityCollapse { .. } => AndonSeverity::Warning,
            Self::GenerationFailure { .. } => AndonSeverity::Critical,
        }
    }

    /// Check if this event should halt the pipeline.
    #[must_use]
    pub fn should_halt(&self) -> bool {
        self.severity() == AndonSeverity::Critical
    }
}

impl std::fmt::Display for AndonEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HighRejectionRate { rate, threshold } => {
                write!(
                    f,
                    "ANDON: High rejection rate {:.1}% > {:.1}%",
                    rate * 100.0,
                    threshold * 100.0
                )
            }
            Self::QualityDrift { current, baseline } => {
                write!(
                    f,
                    "ANDON: Quality drift {current:.3} < baseline {baseline:.3}"
                )
            }
            Self::DiversityCollapse { score, minimum } => {
                write!(
                    f,
                    "ANDON: Diversity collapse {score:.3} < minimum {minimum:.3}"
                )
            }
            Self::GenerationFailure { message } => {
                write!(f, "ANDON: Generation failed - {message}")
            }
        }
    }
}

/// Handler trait for Andon alerts (Toyota Jidoka pattern).
///
/// Implement this trait to customize how Andon events are handled.
/// The default implementation logs warnings and panics on critical events.
///
/// # Example
///
/// ```
/// use aprender::synthetic::andon::{AndonHandler, AndonEvent, AndonSeverity};
///
/// struct LoggingAndon;
///
/// impl AndonHandler for LoggingAndon {
///     fn on_event(&self, event: &AndonEvent) {
///         eprintln!("[{}] {}", event.severity(), event);
///     }
///
///     fn should_halt(&self, event: &AndonEvent) -> bool {
///         event.severity() == AndonSeverity::Critical
///     }
/// }
/// ```
pub trait AndonHandler: Send + Sync {
    /// Called when an Andon event occurs.
    fn on_event(&self, event: &AndonEvent);

    /// Determine if pipeline should halt for this event.
    fn should_halt(&self, event: &AndonEvent) -> bool;

    /// Called when high rejection rate is detected.
    fn on_high_rejection(&self, rate: f32, threshold: f32) {
        let event = AndonEvent::HighRejectionRate { rate, threshold };
        self.on_event(&event);
    }

    /// Called when quality drifts below baseline.
    fn on_quality_drift(&self, current: f32, baseline: f32) {
        let event = AndonEvent::QualityDrift { current, baseline };
        self.on_event(&event);
    }

    /// Called when diversity collapses.
    fn on_diversity_collapse(&self, score: f32, minimum: f32) {
        let event = AndonEvent::DiversityCollapse { score, minimum };
        self.on_event(&event);
    }
}

/// Default Andon handler that logs and halts on critical events.
///
/// This is the production-safe default that will:
/// - Log all events to stderr
/// - Panic on critical events (high rejection, severe quality drift)
///
/// # Panics
///
/// Panics when `should_halt` returns true for an event.
#[derive(Debug, Clone, Default)]
pub struct DefaultAndon {
    /// Track if pipeline has been halted.
    halted: Arc<AtomicBool>,
}

impl DefaultAndon {
    /// Create a new default Andon handler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if the Andon has triggered a halt.
    #[must_use]
    pub fn is_halted(&self) -> bool {
        self.halted.load(Ordering::SeqCst)
    }

    /// Reset the halt state.
    pub fn reset(&self) {
        self.halted.store(false, Ordering::SeqCst);
    }
}

impl AndonHandler for DefaultAndon {
    fn on_event(&self, event: &AndonEvent) {
        eprintln!("[ANDON {}] {}", event.severity(), event);
        if self.should_halt(event) {
            self.halted.store(true, Ordering::SeqCst);
        }
    }

    fn should_halt(&self, event: &AndonEvent) -> bool {
        event.should_halt()
    }
}

/// Silent Andon handler for testing that collects events.
///
/// Does not log or panic, just records events for inspection.
#[derive(Debug, Default)]
pub struct TestAndon {
    /// Collected events.
    events: std::sync::Mutex<Vec<AndonEvent>>,
    /// Whether any halt was triggered.
    halted: AtomicBool,
}

impl TestAndon {
    /// Create a new test Andon handler.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get collected events.
    #[must_use]
    pub fn events(&self) -> Vec<AndonEvent> {
        self.events
            .lock()
            .expect("TestAndon mutex poisoned")
            .clone()
    }

    /// Check if halt was triggered.
    #[must_use]
    pub fn was_halted(&self) -> bool {
        self.halted.load(Ordering::SeqCst)
    }

    /// Clear collected events.
    pub fn clear(&self) {
        self.events
            .lock()
            .expect("TestAndon mutex poisoned")
            .clear();
        self.halted.store(false, Ordering::SeqCst);
    }

    /// Count events of a specific type.
    #[must_use]
    pub fn count_high_rejection(&self) -> usize {
        self.events()
            .iter()
            .filter(|e| matches!(e, AndonEvent::HighRejectionRate { .. }))
            .count()
    }

    /// Count quality drift events.
    #[must_use]
    pub fn count_quality_drift(&self) -> usize {
        self.events()
            .iter()
            .filter(|e| matches!(e, AndonEvent::QualityDrift { .. }))
            .count()
    }
}

impl AndonHandler for TestAndon {
    fn on_event(&self, event: &AndonEvent) {
        self.events
            .lock()
            .expect("TestAndon mutex poisoned")
            .push(event.clone());
        if self.should_halt(event) {
            self.halted.store(true, Ordering::SeqCst);
        }
    }

    fn should_halt(&self, event: &AndonEvent) -> bool {
        event.should_halt()
    }
}

/// Andon configuration for SyntheticConfig.
#[derive(Debug, Clone, PartialEq)]
pub struct AndonConfig {
    /// Whether Andon monitoring is enabled.
    pub enabled: bool,
    /// Rejection rate threshold to trigger alert (0.0-1.0).
    pub rejection_threshold: f32,
    /// Quality baseline for drift detection.
    pub quality_baseline: Option<f32>,
    /// Minimum diversity before collapse alert.
    pub diversity_minimum: f32,
}

impl Default for AndonConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rejection_threshold: 0.90, // Halt if >90% rejected
            quality_baseline: None,    // Set from first batch
            diversity_minimum: 0.1,    // Minimum acceptable diversity
        }
    }
}

impl AndonConfig {
    /// Create a new Andon configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable Andon monitoring.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Set rejection rate threshold.
    #[must_use]
    pub fn with_rejection_threshold(mut self, threshold: f32) -> Self {
        self.rejection_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set quality baseline for drift detection.
    #[must_use]
    pub fn with_quality_baseline(mut self, baseline: f32) -> Self {
        self.quality_baseline = Some(baseline.clamp(0.0, 1.0));
        self
    }

    /// Set minimum diversity threshold.
    #[must_use]
    pub fn with_diversity_minimum(mut self, minimum: f32) -> Self {
        self.diversity_minimum = minimum.clamp(0.0, 1.0);
        self
    }

    /// Check if rejection rate exceeds threshold.
    #[must_use]
    pub fn exceeds_rejection_threshold(&self, rate: f32) -> bool {
        self.enabled && rate > self.rejection_threshold
    }

    /// Check if quality has drifted from baseline.
    #[must_use]
    pub fn has_quality_drift(&self, current: f32) -> bool {
        if !self.enabled {
            return false;
        }
        match self.quality_baseline {
            Some(baseline) => current < baseline * 0.9, // 10% drift tolerance
            None => false,
        }
    }

    /// Check if diversity has collapsed.
    #[must_use]
    pub fn has_diversity_collapse(&self, score: f32) -> bool {
        self.enabled && score < self.diversity_minimum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================================
    // EXTREME TDD: AndonSeverity Tests
    // ============================================================================

    #[test]
    fn test_andon_severity_display() {
        assert_eq!(format!("{}", AndonSeverity::Info), "INFO");
        assert_eq!(format!("{}", AndonSeverity::Warning), "WARNING");
        assert_eq!(format!("{}", AndonSeverity::Critical), "CRITICAL");
    }

    #[test]
    fn test_andon_severity_equality() {
        assert_eq!(AndonSeverity::Info, AndonSeverity::Info);
        assert_ne!(AndonSeverity::Info, AndonSeverity::Warning);
    }

    #[test]
    fn test_andon_severity_clone_copy() {
        let s = AndonSeverity::Critical;
        let s2 = s;
        assert_eq!(s, s2);
    }

    // ============================================================================
    // EXTREME TDD: AndonEvent Tests
    // ============================================================================

    #[test]
    fn test_high_rejection_rate_event() {
        let event = AndonEvent::HighRejectionRate {
            rate: 0.95,
            threshold: 0.90,
        };
        assert_eq!(event.severity(), AndonSeverity::Critical);
        assert!(event.should_halt());
        assert!(format!("{event}").contains("95.0%"));
    }

    #[test]
    fn test_high_rejection_rate_warning() {
        // Just slightly over threshold - warning, not critical
        let event = AndonEvent::HighRejectionRate {
            rate: 0.91,
            threshold: 0.90,
        };
        assert_eq!(event.severity(), AndonSeverity::Warning);
        assert!(!event.should_halt());
    }

    #[test]
    fn test_quality_drift_event_critical() {
        let event = AndonEvent::QualityDrift {
            current: 0.5,
            baseline: 0.8,
        };
        // 0.5 < 0.8 * 0.8 = 0.64, so critical
        assert_eq!(event.severity(), AndonSeverity::Critical);
        assert!(event.should_halt());
    }

    #[test]
    fn test_quality_drift_event_warning() {
        let event = AndonEvent::QualityDrift {
            current: 0.7,
            baseline: 0.8,
        };
        // 0.7 >= 0.8 * 0.8 = 0.64, so warning
        assert_eq!(event.severity(), AndonSeverity::Warning);
        assert!(!event.should_halt());
    }

    #[test]
    fn test_diversity_collapse_event() {
        let event = AndonEvent::DiversityCollapse {
            score: 0.05,
            minimum: 0.1,
        };
        assert_eq!(event.severity(), AndonSeverity::Warning);
        assert!(!event.should_halt());
        assert!(format!("{event}").contains("collapse"));
    }

    #[test]
    fn test_generation_failure_event() {
        let event = AndonEvent::GenerationFailure {
            message: "out of memory".to_string(),
        };
        assert_eq!(event.severity(), AndonSeverity::Critical);
        assert!(event.should_halt());
        assert!(format!("{event}").contains("out of memory"));
    }

    #[test]
    fn test_andon_event_clone() {
        let event = AndonEvent::HighRejectionRate {
            rate: 0.95,
            threshold: 0.90,
        };
        let event2 = event.clone();
        assert_eq!(event, event2);
    }

    // ============================================================================
    // EXTREME TDD: DefaultAndon Tests
    // ============================================================================

    #[test]
    fn test_default_andon_new() {
        let andon = DefaultAndon::new();
        assert!(!andon.is_halted());
    }

    #[test]
    fn test_default_andon_halt_on_critical() {
        let andon = DefaultAndon::new();
        let event = AndonEvent::HighRejectionRate {
            rate: 0.99,
            threshold: 0.90,
        };
        andon.on_event(&event);
        assert!(andon.is_halted());
    }

    #[test]
    fn test_default_andon_no_halt_on_warning() {
        let andon = DefaultAndon::new();
        let event = AndonEvent::DiversityCollapse {
            score: 0.05,
            minimum: 0.1,
        };
        andon.on_event(&event);
        assert!(!andon.is_halted());
    }

    #[test]
    fn test_default_andon_reset() {
        let andon = DefaultAndon::new();
        let event = AndonEvent::GenerationFailure {
            message: "test".to_string(),
        };
        andon.on_event(&event);
        assert!(andon.is_halted());
        andon.reset();
        assert!(!andon.is_halted());
    }

    #[test]
    fn test_default_andon_clone() {
        let andon1 = DefaultAndon::new();
        let andon2 = andon1.clone();
        // Both share same halted state via Arc
        let event = AndonEvent::GenerationFailure {
            message: "x".to_string(),
        };
        andon1.on_event(&event);
        assert!(andon2.is_halted());
    }

    // ============================================================================
    // EXTREME TDD: TestAndon Tests
    // ============================================================================

    #[test]
    fn test_test_andon_collects_events() {
        let andon = TestAndon::new();
        andon.on_high_rejection(0.95, 0.90);
        andon.on_quality_drift(0.5, 0.8);

        let events = andon.events();
        assert_eq!(events.len(), 2);
        assert_eq!(andon.count_high_rejection(), 1);
        assert_eq!(andon.count_quality_drift(), 1);
    }

    #[test]
    fn test_test_andon_was_halted() {
        let andon = TestAndon::new();
        assert!(!andon.was_halted());

        andon.on_event(&AndonEvent::GenerationFailure {
            message: "x".to_string(),
        });
        assert!(andon.was_halted());
    }

    #[test]
    fn test_test_andon_clear() {
        let andon = TestAndon::new();
        andon.on_high_rejection(0.95, 0.90);
        assert_eq!(andon.events().len(), 1);

        andon.clear();
        assert!(andon.events().is_empty());
        assert!(!andon.was_halted());
    }

    // ============================================================================
    // EXTREME TDD: AndonConfig Tests
    // ============================================================================

    #[test]
    fn test_andon_config_default() {
        let config = AndonConfig::default();
        assert!(config.enabled);
        assert!((config.rejection_threshold - 0.90).abs() < f32::EPSILON);
        assert!(config.quality_baseline.is_none());
        assert!((config.diversity_minimum - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_andon_config_builder() {
        let config = AndonConfig::new()
            .with_enabled(false)
            .with_rejection_threshold(0.85)
            .with_quality_baseline(0.7)
            .with_diversity_minimum(0.2);

        assert!(!config.enabled);
        assert!((config.rejection_threshold - 0.85).abs() < f32::EPSILON);
        assert!(
            (config.quality_baseline.expect("baseline should be set") - 0.7).abs() < f32::EPSILON
        );
        assert!((config.diversity_minimum - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn test_andon_config_clamping() {
        let config = AndonConfig::new()
            .with_rejection_threshold(1.5)
            .with_quality_baseline(-0.5)
            .with_diversity_minimum(2.0);

        assert!((config.rejection_threshold - 1.0).abs() < f32::EPSILON);
        assert!(
            (config.quality_baseline.expect("baseline should be set") - 0.0).abs() < f32::EPSILON
        );
        assert!((config.diversity_minimum - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_exceeds_rejection_threshold() {
        let config = AndonConfig::new().with_rejection_threshold(0.90);

        assert!(!config.exceeds_rejection_threshold(0.85));
        assert!(!config.exceeds_rejection_threshold(0.90));
        assert!(config.exceeds_rejection_threshold(0.91));
    }

    #[test]
    fn test_exceeds_rejection_threshold_disabled() {
        let config = AndonConfig::new()
            .with_enabled(false)
            .with_rejection_threshold(0.90);

        assert!(!config.exceeds_rejection_threshold(0.99));
    }

    #[test]
    fn test_has_quality_drift() {
        let config = AndonConfig::new().with_quality_baseline(0.8);

        // 10% tolerance: 0.8 * 0.9 = 0.72
        assert!(!config.has_quality_drift(0.75)); // Above threshold
        assert!(config.has_quality_drift(0.70)); // Below threshold
    }

    #[test]
    fn test_has_quality_drift_no_baseline() {
        let config = AndonConfig::new();
        assert!(config.quality_baseline.is_none());
        assert!(!config.has_quality_drift(0.1)); // No baseline, no drift
    }

    #[test]
    fn test_has_quality_drift_disabled() {
        let config = AndonConfig::new()
            .with_enabled(false)
            .with_quality_baseline(0.8);

        assert!(!config.has_quality_drift(0.1));
    }

    #[test]
    fn test_has_diversity_collapse() {
        let config = AndonConfig::new().with_diversity_minimum(0.1);

        assert!(!config.has_diversity_collapse(0.15));
        assert!(!config.has_diversity_collapse(0.1));
        assert!(config.has_diversity_collapse(0.05));
    }

    #[test]
    fn test_has_diversity_collapse_disabled() {
        let config = AndonConfig::new()
            .with_enabled(false)
            .with_diversity_minimum(0.1);

        assert!(!config.has_diversity_collapse(0.01));
    }

    #[test]
    fn test_andon_config_clone() {
        let c1 = AndonConfig::new().with_rejection_threshold(0.85);
        let c2 = c1.clone();
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_andon_config_debug() {
        let config = AndonConfig::default();
        let debug = format!("{config:?}");
        assert!(debug.contains("AndonConfig"));
        assert!(debug.contains("rejection_threshold"));
    }
}
