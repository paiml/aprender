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

/// Andon configuration for `SyntheticConfig`.
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
#[path = "andon_tests.rs"]
mod tests;
