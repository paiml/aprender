//! Brick Architecture: Tests ARE the Interface (PROBAR-SPEC-009)
//!
//! This module implements the core Brick abstraction where UI components
//! are defined by their test assertions, not by their implementation.
//!
//! # Design Philosophy
//!
//! The Brick Architecture inverts the traditional UI/Test relationship:
//!
//! ```text
//! Traditional: Widget → Tests
//! Brick:       Brick(Assertions) → Widget + Tests
//! ```
//!
//! A `Brick` defines:
//! 1. **Assertions**: What must be true (contrast ratio, visibility, latency)
//! 2. **Budget**: Performance envelope (max render time in ms)
//! 3. **Events**: State transitions that trigger assertions
//!
//! # Zero-Artifact Architecture (PROBAR-SPEC-009-P7)
//!
//! The following brick types generate all web artifacts from Rust:
//!
//! - [`WorkerBrick`] - Web Worker JavaScript and Rust web_sys bindings
//! - [`EventBrick`] - DOM event handlers
//! - [`AudioBrick`] - AudioWorklet processor code
//!
//! # Popperian Falsification
//!
//! Each assertion is a falsifiable hypothesis. If ANY assertion fails,
//! the brick is falsified and cannot render.
//!
//! # Example
//!
//! ```rust,ignore
//! use probar::brick::{Brick, BrickAssertion, BrickBudget};
//!
//! #[derive(Brick)]
//! #[brick(
//!     html = "div.transcription",
//!     budget_ms = 100,
//!     assertions = [text_visible, contrast_ratio(4.5)]
//! )]
//! struct TranscriptionBrick {
//!     text: String,
//!     is_final: bool,
//! }
//! ```
//!
//! # References
//!
//! - Popper, K. (1959). The Logic of Scientific Discovery
//! - Beizer, B. (1990). Software Testing Techniques
//! - PROBAR-SPEC-009: Bug Hunting Probador

// Zero-Artifact submodules (PROBAR-SPEC-009-P7)
pub mod audio;
pub mod compute;
pub mod deterministic;
pub mod distributed;
pub mod event;
pub mod pipeline;
pub mod tui;
pub mod web_sys_gen;
pub mod widget;
pub mod worker;

// Re-export submodule types
pub use audio::{AudioBrick, AudioParam, RingBufferConfig};
pub use compute::{
    ComputeBrick, ElementwiseOp, ReduceKind, TensorBinding, TensorType, TileOp, TileStrategy,
};
pub use deterministic::{
    BrickHistory, BrickState, DeterministicBrick, DeterministicClock, DeterministicRng,
    ExecutionTrace, GuardSeverity, GuardViolation, GuardedBrick, InvariantGuard, StateValue,
};
pub use distributed::{
    Backend, BackendSelector, BrickCoordinator, BrickDataTracker, BrickInput, BrickMessage,
    BrickOutput, DataLocation, DistributedBrick, ExecutionMetrics, MultiBrickExecutor,
    SchedulerStats, Subscription, TaskSpec, WorkStealingScheduler, WorkStealingTask, WorkerId,
    WorkerQueue, WorkerStats,
};
pub use event::{EventBinding, EventBrick, EventHandler, EventType};
pub use pipeline::{
    AuditEntry, BrickPipeline, BrickStage, Checkpoint, PipelineAuditCollector, PipelineContext,
    PipelineData, PipelineError, PipelineMetadata, PipelineResult, PrivacyTier, StageTrace,
    ValidationLevel, ValidationMessage, ValidationResult,
};
pub use tui::{
    AnalyzerBrick, CielabColor, CollectorBrick, CollectorError, PanelBrick, PanelId, PanelState,
    RingBuffer,
};
pub use web_sys_gen::{
    get_base_url, BlobUrl, CustomEventDispatcher, EventDetail, FetchClient, GeneratedWebSys,
    GenerationMetadata, PerformanceTiming, WebSysError, GENERATION_METADATA,
};
pub use widget::{
    commands_to_gpu_instances, Canvas, Constraints, CornerRadius, DrawCommand, Event, GpuInstance,
    LayoutResult, LineCap, LineJoin, Modifiers, RecordingCanvas, Rect, RenderMetrics, Size,
    StrokeStyle, TextStyle, Transform2D, Widget, WidgetColor, WidgetExt, WidgetMouseButton,
    WidgetPoint,
};
pub use worker::{
    BrickWorkerMessage, BrickWorkerMessageDirection, FieldType, MessageField, WorkerBrick,
    WorkerTransition,
};

use std::time::Duration;

/// Brick assertion that must be verified at runtime.
///
/// Assertions are falsifiable hypotheses about the UI state.
/// If any assertion fails, the brick is falsified.
#[derive(Debug, Clone, PartialEq)]
pub enum BrickAssertion {
    /// Text content must be visible (not hidden, not zero-opacity)
    TextVisible,

    /// WCAG 2.1 AA contrast ratio requirement (4.5:1 for normal text)
    ContrastRatio(f32),

    /// Maximum render latency in milliseconds
    MaxLatencyMs(u32),

    /// Element must be present in DOM
    ElementPresent(String),

    /// Element must be focusable for accessibility
    Focusable,

    /// Custom assertion with name and validation function ID
    Custom {
        /// Assertion name for error reporting
        name: String,
        /// Validation function identifier
        validator_id: u64,
    },
}

impl BrickAssertion {
    /// Create a text visibility assertion
    #[must_use]
    pub const fn text_visible() -> Self {
        Self::TextVisible
    }

    /// Create a contrast ratio assertion (WCAG 2.1 AA)
    #[must_use]
    pub const fn contrast_ratio(ratio: f32) -> Self {
        Self::ContrastRatio(ratio)
    }

    /// Create a max latency assertion
    #[must_use]
    pub const fn max_latency_ms(ms: u32) -> Self {
        Self::MaxLatencyMs(ms)
    }

    /// Create an element presence assertion
    #[must_use]
    pub fn element_present(selector: impl Into<String>) -> Self {
        Self::ElementPresent(selector.into())
    }
}

/// Performance budget for a brick.
///
/// Budgets are enforced at runtime. Exceeding the budget triggers
/// a Jidoka (stop-the-line) alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrickBudget {
    /// Maximum time for measure phase
    pub measure_ms: u32,
    /// Maximum time for layout phase
    pub layout_ms: u32,
    /// Maximum time for paint phase
    pub paint_ms: u32,
    /// Total budget (may be less than sum of phases)
    pub total_ms: u32,
}

impl BrickBudget {
    /// Create a budget with equal distribution across phases
    #[must_use]
    pub const fn uniform(total_ms: u32) -> Self {
        let phase_ms = total_ms / 3;
        Self {
            measure_ms: phase_ms,
            layout_ms: phase_ms,
            paint_ms: phase_ms,
            total_ms,
        }
    }

    /// Create a custom budget with specified phase limits
    #[must_use]
    pub const fn new(measure_ms: u32, layout_ms: u32, paint_ms: u32) -> Self {
        Self {
            measure_ms,
            layout_ms,
            paint_ms,
            total_ms: measure_ms + layout_ms + paint_ms,
        }
    }

    /// Convert to Duration
    #[must_use]
    pub const fn as_duration(&self) -> Duration {
        Duration::from_millis(self.total_ms as u64)
    }
}

impl Default for BrickBudget {
    fn default() -> Self {
        // Default: 16ms total for 60fps
        Self::uniform(16)
    }
}

/// Result of verifying brick assertions
#[derive(Debug, Clone)]
pub struct BrickVerification {
    /// All assertions that passed
    pub passed: Vec<BrickAssertion>,
    /// All assertions that failed with reasons
    pub failed: Vec<(BrickAssertion, String)>,
    /// Time taken to verify
    pub verification_time: Duration,
}

impl BrickVerification {
    /// Check if all assertions passed
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.failed.is_empty()
    }

    /// Get the falsification score (passed / total)
    #[must_use]
    pub fn score(&self) -> f32 {
        let total = self.passed.len() + self.failed.len();
        if total == 0 {
            1.0
        } else {
            self.passed.len() as f32 / total as f32
        }
    }
}

/// Budget violation report
#[derive(Debug, Clone)]
pub struct BudgetViolation {
    /// Name of the brick that violated
    pub brick_name: String,
    /// Budget that was exceeded
    pub budget: BrickBudget,
    /// Actual time taken
    pub actual: Duration,
    /// Phase that exceeded (if known)
    pub phase: Option<BrickPhase>,
}

/// Rendering phase for budget tracking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BrickPhase {
    /// Measure phase (compute intrinsic size)
    Measure,
    /// Layout phase (position children)
    Layout,
    /// Paint phase (generate draw commands)
    Paint,
}

/// Core Brick trait - the foundation of the Brick Architecture.
///
/// All UI components implement this trait. The trait defines:
/// 1. Assertions that must pass for the brick to be valid
/// 2. Performance budget that must not be exceeded
/// 3. HTML/CSS generation for rendering targets
///
/// # Trait Bound
///
/// Presentar's `Widget` trait requires `Brick`:
/// ```rust,ignore
/// pub trait Widget: Brick + Send + Sync { ... }
/// ```
///
/// This ensures every widget has verifiable assertions and budgets.
pub trait Brick: Send + Sync {
    /// Get the brick's unique type name
    fn brick_name(&self) -> &'static str;

    /// Get all assertions for this brick
    fn assertions(&self) -> &[BrickAssertion];

    /// Get the performance budget
    fn budget(&self) -> BrickBudget;

    /// Verify all assertions against current state
    ///
    /// Returns a verification result with passed/failed assertions.
    fn verify(&self) -> BrickVerification;

    /// Generate HTML for this brick (WASM target)
    ///
    /// Returns the HTML string that represents this brick.
    /// Must be deterministic (same state → same output).
    fn to_html(&self) -> String;

    /// Generate CSS for this brick (WASM target)
    ///
    /// Returns the CSS rules for styling this brick.
    /// Must be deterministic and scoped to avoid conflicts.
    fn to_css(&self) -> String;

    /// Get the test ID for DOM queries
    fn test_id(&self) -> Option<&str> {
        None
    }

    /// Check if this brick can be rendered (all assertions pass)
    fn can_render(&self) -> bool {
        self.verify().is_valid()
    }
}

/// Yuan Gate: Zero-swallow error handling for bricks
///
/// Named after the Yuan dynasty's strict quality standards.
/// Every error must be explicitly handled - no silent drops.
#[derive(Debug, Clone)]
pub enum BrickError {
    /// Assertion failed during verification
    AssertionFailed {
        /// The assertion that failed
        assertion: BrickAssertion,
        /// Reason for failure
        reason: String,
    },

    /// Budget exceeded during rendering
    BudgetExceeded(BudgetViolation),

    /// Invalid state transition
    InvalidTransition {
        /// Current state
        from: String,
        /// Attempted target state
        to: String,
        /// Reason transition is invalid
        reason: String,
    },

    /// Missing required child brick
    MissingChild {
        /// Expected child brick name
        expected: String,
    },

    /// HTML generation failed
    HtmlGenerationFailed {
        /// Reason for failure
        reason: String,
    },
}

impl std::fmt::Display for BrickError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AssertionFailed { assertion, reason } => {
                write!(f, "Assertion {assertion:?} failed: {reason}")
            }
            Self::BudgetExceeded(violation) => {
                write!(
                    f,
                    "Budget exceeded for {}: {:?} > {:?}",
                    violation.brick_name, violation.actual, violation.budget.total_ms
                )
            }
            Self::InvalidTransition { from, to, reason } => {
                write!(f, "Invalid transition {from} -> {to}: {reason}")
            }
            Self::MissingChild { expected } => {
                write!(f, "Missing required child brick: {expected}")
            }
            Self::HtmlGenerationFailed { reason } => {
                write!(f, "HTML generation failed: {reason}")
            }
        }
    }
}

impl std::error::Error for BrickError {}

/// Result type for brick operations
pub type BrickResult<T> = Result<T, BrickError>;

#[cfg(test)]
mod tests {
    use super::*;

    struct TestBrick {
        text: String,
        visible: bool,
    }

    impl Brick for TestBrick {
        fn brick_name(&self) -> &'static str {
            "TestBrick"
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[
                BrickAssertion::TextVisible,
                BrickAssertion::ContrastRatio(4.5),
            ]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::uniform(16)
        }

        fn verify(&self) -> BrickVerification {
            let mut passed = Vec::new();
            let mut failed = Vec::new();

            for assertion in self.assertions() {
                match assertion {
                    BrickAssertion::TextVisible => {
                        if self.visible && !self.text.is_empty() {
                            passed.push(assertion.clone());
                        } else {
                            failed.push((assertion.clone(), "Text not visible".into()));
                        }
                    }
                    BrickAssertion::ContrastRatio(_) => {
                        // Assume pass for test
                        passed.push(assertion.clone());
                    }
                    _ => passed.push(assertion.clone()),
                }
            }

            BrickVerification {
                passed,
                failed,
                verification_time: Duration::from_micros(100),
            }
        }

        fn to_html(&self) -> String {
            format!(r#"<div class="test-brick">{}</div>"#, self.text)
        }

        fn to_css(&self) -> String {
            ".test-brick { color: #fff; background: #000; }".into()
        }
    }

    #[test]
    fn test_brick_verification_passes() {
        let brick = TestBrick {
            text: "Hello".into(),
            visible: true,
        };

        let result = brick.verify();
        assert!(result.is_valid());
        assert_eq!(result.score(), 1.0);
    }

    #[test]
    fn test_brick_verification_fails() {
        let brick = TestBrick {
            text: String::new(),
            visible: false,
        };

        let result = brick.verify();
        assert!(!result.is_valid());
        assert!(result.score() < 1.0);
    }

    #[test]
    fn test_budget_uniform() {
        let budget = BrickBudget::uniform(30);
        assert_eq!(budget.total_ms, 30);
        assert_eq!(budget.measure_ms, 10);
    }

    #[test]
    fn test_can_render_valid() {
        let brick = TestBrick {
            text: "Hello".into(),
            visible: true,
        };
        assert!(brick.can_render());
    }

    #[test]
    fn test_can_render_invalid() {
        let brick = TestBrick {
            text: String::new(),
            visible: false,
        };
        assert!(!brick.can_render());
    }

    #[test]
    fn test_brick_assertion_constructors() {
        let text_vis = BrickAssertion::text_visible();
        assert!(matches!(text_vis, BrickAssertion::TextVisible));

        let contrast = BrickAssertion::contrast_ratio(4.5);
        assert!(
            matches!(contrast, BrickAssertion::ContrastRatio(r) if (r - 4.5).abs() < f32::EPSILON)
        );

        let latency = BrickAssertion::max_latency_ms(100);
        assert!(matches!(latency, BrickAssertion::MaxLatencyMs(100)));

        let elem = BrickAssertion::element_present("div.test");
        assert!(matches!(elem, BrickAssertion::ElementPresent(s) if s == "div.test"));
    }

    #[test]
    fn test_brick_assertion_focusable() {
        let focusable = BrickAssertion::Focusable;
        assert!(matches!(focusable, BrickAssertion::Focusable));
    }

    #[test]
    fn test_brick_assertion_custom() {
        let custom = BrickAssertion::Custom {
            name: "test_assertion".into(),
            validator_id: 42,
        };
        match custom {
            BrickAssertion::Custom { name, validator_id } => {
                assert_eq!(name, "test_assertion");
                assert_eq!(validator_id, 42);
            }
            _ => panic!("Expected Custom variant"),
        }
    }

    #[test]
    fn test_budget_new() {
        let budget = BrickBudget::new(5, 10, 15);
        assert_eq!(budget.measure_ms, 5);
        assert_eq!(budget.layout_ms, 10);
        assert_eq!(budget.paint_ms, 15);
        assert_eq!(budget.total_ms, 30);
    }

    #[test]
    fn test_budget_default() {
        let budget = BrickBudget::default();
        assert_eq!(budget.total_ms, 16); // 60fps
    }

    #[test]
    fn test_budget_as_duration() {
        let budget = BrickBudget::uniform(100);
        let duration = budget.as_duration();
        assert_eq!(duration, Duration::from_millis(100));
    }

    #[test]
    fn test_verification_score_empty() {
        let verification = BrickVerification {
            passed: vec![],
            failed: vec![],
            verification_time: Duration::from_micros(10),
        };
        assert_eq!(verification.score(), 1.0); // Empty = perfect score
        assert!(verification.is_valid());
    }

    #[test]
    fn test_verification_score_partial() {
        let verification = BrickVerification {
            passed: vec![BrickAssertion::TextVisible],
            failed: vec![(BrickAssertion::Focusable, "Not focusable".into())],
            verification_time: Duration::from_micros(10),
        };
        assert_eq!(verification.score(), 0.5);
        assert!(!verification.is_valid());
    }

    #[test]
    fn test_brick_phase_variants() {
        let measure = BrickPhase::Measure;
        let layout = BrickPhase::Layout;
        let paint = BrickPhase::Paint;

        assert!(matches!(measure, BrickPhase::Measure));
        assert!(matches!(layout, BrickPhase::Layout));
        assert!(matches!(paint, BrickPhase::Paint));
        assert_ne!(measure, layout);
    }

    #[test]
    fn test_budget_violation() {
        let violation = BudgetViolation {
            brick_name: "TestBrick".into(),
            budget: BrickBudget::uniform(16),
            actual: Duration::from_millis(50),
            phase: Some(BrickPhase::Paint),
        };
        assert_eq!(violation.brick_name, "TestBrick");
        assert_eq!(violation.phase, Some(BrickPhase::Paint));
    }

    #[test]
    fn test_brick_to_html_css() {
        let brick = TestBrick {
            text: "Test".into(),
            visible: true,
        };
        let html = brick.to_html();
        let css = brick.to_css();

        assert!(html.contains("test-brick"));
        assert!(html.contains("Test"));
        assert!(css.contains(".test-brick"));
    }

    #[test]
    fn test_brick_name() {
        let brick = TestBrick {
            text: "Test".into(),
            visible: true,
        };
        assert_eq!(brick.brick_name(), "TestBrick");
    }

    #[test]
    fn test_brick_assertions_list() {
        let brick = TestBrick {
            text: "Test".into(),
            visible: true,
        };
        let assertions = brick.assertions();
        assert_eq!(assertions.len(), 2);
        assert!(matches!(assertions[0], BrickAssertion::TextVisible));
    }

    #[test]
    fn test_brick_budget_method() {
        let brick = TestBrick {
            text: "Test".into(),
            visible: true,
        };
        let budget = brick.budget();
        assert_eq!(budget.total_ms, 16);
    }

    // ===== BrickError Display tests =====

    #[test]
    fn test_brick_error_display_assertion_failed() {
        let error = BrickError::AssertionFailed {
            assertion: BrickAssertion::TextVisible,
            reason: "Element is hidden".into(),
        };
        let display = format!("{error}");
        assert!(display.contains("Assertion"));
        assert!(display.contains("TextVisible"));
        assert!(display.contains("Element is hidden"));
    }

    #[test]
    fn test_brick_error_display_budget_exceeded() {
        let violation = BudgetViolation {
            brick_name: "MyBrick".into(),
            budget: BrickBudget::uniform(16),
            actual: Duration::from_millis(100),
            phase: None,
        };
        let error = BrickError::BudgetExceeded(violation);
        let display = format!("{error}");
        assert!(display.contains("Budget exceeded"));
        assert!(display.contains("MyBrick"));
    }

    #[test]
    fn test_brick_error_display_invalid_transition() {
        let error = BrickError::InvalidTransition {
            from: "idle".into(),
            to: "running".into(),
            reason: "Missing prerequisite".into(),
        };
        let display = format!("{error}");
        assert!(display.contains("Invalid transition"));
        assert!(display.contains("idle"));
        assert!(display.contains("running"));
        assert!(display.contains("Missing prerequisite"));
    }

    #[test]
    fn test_brick_error_display_missing_child() {
        let error = BrickError::MissingChild {
            expected: "ChildWidget".into(),
        };
        let display = format!("{error}");
        assert!(display.contains("Missing required child brick"));
        assert!(display.contains("ChildWidget"));
    }

    #[test]
    fn test_brick_error_display_html_generation_failed() {
        let error = BrickError::HtmlGenerationFailed {
            reason: "Template parse error".into(),
        };
        let display = format!("{error}");
        assert!(display.contains("HTML generation failed"));
        assert!(display.contains("Template parse error"));
    }

    #[test]
    fn test_brick_error_is_std_error() {
        let error: &dyn std::error::Error = &BrickError::MissingChild {
            expected: "Test".into(),
        };
        // Verify Error trait is implemented
        assert!(error.source().is_none());
    }

    // ===== test_id default implementation =====

    #[test]
    fn test_brick_test_id_default() {
        let brick = TestBrick {
            text: "Test".into(),
            visible: true,
        };
        // Default implementation returns None
        assert!(brick.test_id().is_none());
    }

    // ===== BrickAssertion Clone and PartialEq =====

    #[test]
    fn test_brick_assertion_clone() {
        let original = BrickAssertion::Custom {
            name: "custom_check".into(),
            validator_id: 123,
        };
        let cloned = original.clone();
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_brick_assertion_partial_eq() {
        let a1 = BrickAssertion::ContrastRatio(4.5);
        let a2 = BrickAssertion::ContrastRatio(4.5);
        let a3 = BrickAssertion::ContrastRatio(7.0);

        assert_eq!(a1, a2);
        assert_ne!(a1, a3);
    }

    #[test]
    fn test_brick_assertion_element_present_eq() {
        let a1 = BrickAssertion::element_present("div.test");
        let a2 = BrickAssertion::element_present("div.test");
        let a3 = BrickAssertion::element_present("span.other");

        assert_eq!(a1, a2);
        assert_ne!(a1, a3);
    }

    // ===== BrickBudget Clone and PartialEq =====

    #[test]
    fn test_brick_budget_clone() {
        let original = BrickBudget::new(5, 10, 15);
        let cloned = original;
        assert_eq!(original, cloned);
    }

    #[test]
    fn test_brick_budget_partial_eq() {
        let b1 = BrickBudget::new(5, 10, 15);
        let b2 = BrickBudget::new(5, 10, 15);
        let b3 = BrickBudget::new(1, 2, 3);

        assert_eq!(b1, b2);
        assert_ne!(b1, b3);
    }

    // ===== BrickVerification Clone and Debug =====

    #[test]
    fn test_brick_verification_clone() {
        let original = BrickVerification {
            passed: vec![BrickAssertion::TextVisible],
            failed: vec![(BrickAssertion::Focusable, "Not focusable".into())],
            verification_time: Duration::from_micros(50),
        };
        let cloned = original;
        assert_eq!(cloned.passed.len(), 1);
        assert_eq!(cloned.failed.len(), 1);
        assert_eq!(cloned.verification_time, Duration::from_micros(50));
    }

    #[test]
    fn test_brick_verification_debug() {
        let verification = BrickVerification {
            passed: vec![BrickAssertion::TextVisible],
            failed: vec![],
            verification_time: Duration::from_micros(100),
        };
        let debug_str = format!("{verification:?}");
        assert!(debug_str.contains("BrickVerification"));
        assert!(debug_str.contains("passed"));
        assert!(debug_str.contains("TextVisible"));
    }

    // ===== BudgetViolation Clone and Debug =====

    #[test]
    fn test_budget_violation_clone() {
        let original = BudgetViolation {
            brick_name: "ClonedBrick".into(),
            budget: BrickBudget::uniform(16),
            actual: Duration::from_millis(32),
            phase: Some(BrickPhase::Measure),
        };
        let cloned = original;
        assert_eq!(cloned.brick_name, "ClonedBrick");
        assert_eq!(cloned.phase, Some(BrickPhase::Measure));
    }

    #[test]
    fn test_budget_violation_debug() {
        let violation = BudgetViolation {
            brick_name: "DebugBrick".into(),
            budget: BrickBudget::uniform(16),
            actual: Duration::from_millis(50),
            phase: None,
        };
        let debug_str = format!("{violation:?}");
        assert!(debug_str.contains("BudgetViolation"));
        assert!(debug_str.contains("DebugBrick"));
    }

    #[test]
    fn test_budget_violation_no_phase() {
        let violation = BudgetViolation {
            brick_name: "NoPhaseBrick".into(),
            budget: BrickBudget::uniform(16),
            actual: Duration::from_millis(50),
            phase: None,
        };
        assert!(violation.phase.is_none());
    }

    // ===== BrickPhase Copy and Clone =====

    #[test]
    fn test_brick_phase_copy() {
        let phase = BrickPhase::Layout;
        let copied = phase;
        assert_eq!(phase, copied);
    }

    #[test]
    fn test_brick_phase_clone() {
        let phase = BrickPhase::Paint;
        #[allow(clippy::clone_on_copy)]
        let cloned = phase.clone();
        assert_eq!(phase, cloned);
    }

    #[test]
    fn test_brick_phase_debug() {
        let phase = BrickPhase::Measure;
        let debug_str = format!("{phase:?}");
        assert_eq!(debug_str, "Measure");
    }

    // ===== BrickAssertion Debug =====

    #[test]
    fn test_brick_assertion_debug() {
        let assertion = BrickAssertion::MaxLatencyMs(100);
        let debug_str = format!("{assertion:?}");
        assert!(debug_str.contains("MaxLatencyMs"));
        assert!(debug_str.contains("100"));
    }

    #[test]
    fn test_brick_assertion_custom_debug() {
        let assertion = BrickAssertion::Custom {
            name: "my_validator".into(),
            validator_id: 999,
        };
        let debug_str = format!("{assertion:?}");
        assert!(debug_str.contains("Custom"));
        assert!(debug_str.contains("my_validator"));
        assert!(debug_str.contains("999"));
    }

    // ===== BrickBudget Debug =====

    #[test]
    fn test_brick_budget_debug() {
        let budget = BrickBudget::new(5, 10, 15);
        let debug_str = format!("{budget:?}");
        assert!(debug_str.contains("BrickBudget"));
        assert!(debug_str.contains("measure_ms"));
        assert!(debug_str.contains('5'));
    }

    // ===== BrickError Clone and Debug =====

    #[test]
    fn test_brick_error_clone() {
        let original = BrickError::MissingChild {
            expected: "SomeChild".into(),
        };
        let cloned = original;
        match cloned {
            BrickError::MissingChild { expected } => {
                assert_eq!(expected, "SomeChild");
            }
            _ => panic!("Expected MissingChild variant"),
        }
    }

    #[test]
    fn test_brick_error_debug() {
        let error = BrickError::HtmlGenerationFailed {
            reason: "Syntax error".into(),
        };
        let debug_str = format!("{error:?}");
        assert!(debug_str.contains("HtmlGenerationFailed"));
        assert!(debug_str.contains("Syntax error"));
    }

    // ===== Edge cases =====

    #[test]
    fn test_verification_all_failed() {
        let verification = BrickVerification {
            passed: vec![],
            failed: vec![
                (BrickAssertion::TextVisible, "Hidden".into()),
                (BrickAssertion::Focusable, "Not focusable".into()),
            ],
            verification_time: Duration::from_micros(10),
        };
        assert_eq!(verification.score(), 0.0);
        assert!(!verification.is_valid());
    }

    #[test]
    fn test_verification_all_passed() {
        let verification = BrickVerification {
            passed: vec![
                BrickAssertion::TextVisible,
                BrickAssertion::Focusable,
                BrickAssertion::ContrastRatio(4.5),
            ],
            failed: vec![],
            verification_time: Duration::from_micros(10),
        };
        assert_eq!(verification.score(), 1.0);
        assert!(verification.is_valid());
    }

    #[test]
    fn test_budget_uniform_edge_case() {
        // Test with value that doesn't divide evenly by 3
        let budget = BrickBudget::uniform(10);
        assert_eq!(budget.measure_ms, 3);
        assert_eq!(budget.layout_ms, 3);
        assert_eq!(budget.paint_ms, 3);
        // total_ms is preserved exactly
        assert_eq!(budget.total_ms, 10);
    }

    #[test]
    fn test_budget_zero() {
        let budget = BrickBudget::uniform(0);
        assert_eq!(budget.measure_ms, 0);
        assert_eq!(budget.layout_ms, 0);
        assert_eq!(budget.paint_ms, 0);
        assert_eq!(budget.total_ms, 0);
        assert_eq!(budget.as_duration(), Duration::from_millis(0));
    }

    // ===== Test brick with custom test_id =====

    struct TestBrickWithId {
        id: &'static str,
    }

    impl Brick for TestBrickWithId {
        fn brick_name(&self) -> &'static str {
            "TestBrickWithId"
        }

        fn assertions(&self) -> &[BrickAssertion] {
            &[]
        }

        fn budget(&self) -> BrickBudget {
            BrickBudget::default()
        }

        fn verify(&self) -> BrickVerification {
            BrickVerification {
                passed: vec![],
                failed: vec![],
                verification_time: Duration::ZERO,
            }
        }

        fn to_html(&self) -> String {
            String::new()
        }

        fn to_css(&self) -> String {
            String::new()
        }

        fn test_id(&self) -> Option<&str> {
            Some(self.id)
        }
    }

    #[test]
    fn test_brick_with_custom_test_id() {
        let brick = TestBrickWithId { id: "my-test-id" };
        assert_eq!(brick.test_id(), Some("my-test-id"));
    }

    #[test]
    fn test_brick_with_id_can_render() {
        let brick = TestBrickWithId { id: "test" };
        // Empty assertions means valid
        assert!(brick.can_render());
    }

    // ===== Additional assertion variant coverage =====

    #[test]
    fn test_all_assertion_variants_are_covered() {
        // Ensure we exercise all variants for coverage
        let variants: Vec<BrickAssertion> = vec![
            BrickAssertion::TextVisible,
            BrickAssertion::ContrastRatio(4.5),
            BrickAssertion::MaxLatencyMs(100),
            BrickAssertion::ElementPresent("div".into()),
            BrickAssertion::Focusable,
            BrickAssertion::Custom {
                name: "test".into(),
                validator_id: 1,
            },
        ];

        for variant in &variants {
            // Exercise Debug
            let _ = format!("{variant:?}");
            // Exercise Clone
            let cloned = variant.clone();
            // Exercise PartialEq
            assert_eq!(variant, &cloned);
        }
    }

    // ===== Brick trait method coverage for TestBrick =====

    #[test]
    fn test_brick_verify_with_empty_text_visible() {
        // Edge case: visible = true but text is empty
        let brick = TestBrick {
            text: String::new(),
            visible: true,
        };
        let result = brick.verify();
        // Should fail because text is empty even though visible is true
        assert!(!result.is_valid());
    }

    #[test]
    fn test_brick_verify_with_text_not_visible() {
        // Edge case: text is present but not visible
        let brick = TestBrick {
            text: "Some text".into(),
            visible: false,
        };
        let result = brick.verify();
        // Should fail because visible is false
        assert!(!result.is_valid());
    }
}
