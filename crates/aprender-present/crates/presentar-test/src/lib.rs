#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::should_implement_trait)]
#![allow(clippy::similar_names)]
#![allow(clippy::derive_partial_eq_without_eq)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::if_not_else)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::type_complexity)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::manual_assert)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::float_cmp)]
#![allow(clippy::clone_on_copy)]
#![allow(clippy::cloned_instead_of_copied)]
#![allow(clippy::single_match)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::struct_field_names)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::let_and_return)]
#![allow(clippy::explicit_iter_loop)]
#![allow(clippy::ptr_arg)]
#![allow(clippy::use_self)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::default_trait_access)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::iter_without_into_iter)]
#![allow(clippy::if_then_some_else_none)]
#![allow(clippy::semicolon_if_nothing_returned)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::case_sensitive_file_extension_comparisons)]
#![allow(clippy::format_push_string)]
#![allow(clippy::same_item_push)]
#![allow(clippy::naive_bytecount)]
#![allow(clippy::unnecessary_to_owned)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::test_attr_in_doctest)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::new_without_default)]
#![allow(clippy::comparison_to_empty)]
#![allow(clippy::get_first)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::duplicated_attributes)]
//! Testing harness for Presentar applications.
//!
//! Zero external dependencies. Pure Rust + WASM only.
//!
//! # Proc Macros
//!
//! Use `#[presentar_test]` for widget and integration tests:
//!
//! ```ignore
//! use presentar_test::presentar_test;
//!
//! #[presentar_test]
//! fn test_button() {
//!     let harness = Harness::new(Button::new("Click"));
//!     harness.assert_exists("Button");
//! }
//!
//! #[presentar_test(timeout = 5000, fixture = "app.tar")]
//! fn test_with_fixture() {
//!     // Fixture is loaded automatically
//! }
//! ```

mod a11y;
pub mod bdd;
pub mod build;
pub mod fixture;
pub mod grade;
mod harness;
mod selector;
mod snapshot;
pub mod tui;

pub use a11y::{
    aria_from_widget, A11yChecker, A11yConfig, A11yReport, A11yViolation, AriaAttributes,
    AriaChecked, AriaLive, AutocompleteValue, FormA11yChecker, FormA11yReport, FormA11yRule,
    FormAccessibility, FormFieldA11y, FormFieldGroup, FormViolation, Impact, InputType,
    MIN_FOCUS_INDICATOR_AREA, MIN_TOUCH_TARGET_SIZE,
};
pub use bdd::{describe, describe_and_assert, expect, Expectation, TestContext};
pub use build::{
    BuildInfo, BuildMode, BundleAnalysis, BundleAnalyzer, BundleError, SizeRecord, SizeTracker,
    WasmSection,
};
pub use fixture::{
    Fixture, FixtureBuilder, FixtureContext, FixtureError, FixtureManifest, TestData,
};
pub use grade::{
    AccessibilityGates, AppQualityScore, Criterion, DataGates, DocumentationGates,
    EvaluationBuilder, GateCheckResult, GateViolation, Grade, PerformanceGates, QualityGates,
    QualityScoreBuilder, ReportCard, ScoreBreakdown, ViolationSeverity,
};
pub use harness::Harness;
pub use selector::{Selector, SelectorParser};
pub use snapshot::{ComparisonResult, Image, Snapshot};

// TUI Testing Framework (SPEC-024 Section 12 & 13)
// Tests DEFINE interface - implementation follows
pub use tui::{
    expect_frame, AsyncUpdateAssertion, BenchmarkHarness, BenchmarkResult, DiffEntry,
    FrameAssertion, PerformanceTargets, RenderMetrics, SnapshotDiff, SnapshotError, TuiCell,
    TuiSnapshot, TuiTestBackend,
};

// Re-export proc macros for convenient access
pub use presentar_test_macros::{assert_snapshot, describe_suite, fixture, presentar_test};

// =============================================================================
// COMPUTEBLOCK ARCHITECTURAL ENFORCEMENT (SPEC-024)
// =============================================================================
//
// TESTS DEFINE INTERFACE. IMPLEMENTATION FOLLOWS.
//
// These macros make it IMPOSSIBLE to build presentar applications without tests.
// The test creates a "proof" type that the implementation requires.
// Without the test -> no proof type -> compile error.
//
// Usage:
//   1. In tests: #[interface_test(MyFeature)] fn test_xxx() { ... }
//   2. In impl:  #[requires_interface(MyFeature)] impl MyFeature { ... }
//
// If you try to use #[requires_interface] without the corresponding
// #[interface_test], the build FAILS. This is architectural, not advisory.

pub use presentar_test_macros::{computeblock, interface_test, requires_interface};
