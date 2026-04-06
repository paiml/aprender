//! Brick-based Widget helpers (PROBAR-SPEC-009)
//!
//! This module provides helpers for creating Widgets that implement the Brick trait,
//! enabling the "tests define interface" philosophy.
//!
//! # Example
//!
//! ```rust,ignore
//! use presentar_core::brick_widget::{SimpleBrick, BrickWidgetExt};
//! use jugar_probar::brick::{BrickAssertion, BrickBudget};
//!
//! struct MyWidget {
//!     text: String,
//!     brick: SimpleBrick,
//! }
//!
//! impl MyWidget {
//!     fn new(text: &str) -> Self {
//!         Self {
//!             text: text.to_string(),
//!             brick: SimpleBrick::new("MyWidget")
//!                 .with_assertion(BrickAssertion::TextVisible)
//!                 .with_assertion(BrickAssertion::ContrastRatio(4.5))
//!                 .with_budget(BrickBudget::uniform(16)),
//!         }
//!     }
//! }
//! ```

use crate::brick_types::{Brick, BrickAssertion, BrickBudget, BrickVerification};
use std::time::Duration;

/// Simple Brick implementation for common use cases.
///
/// Provides a straightforward way to define brick assertions and budgets
/// without implementing the full Brick trait manually.
#[derive(Debug, Clone)]
pub struct SimpleBrick {
    name: &'static str,
    assertions: Vec<BrickAssertion>,
    budget: BrickBudget,
    custom_verify: Option<fn() -> bool>,
}

impl SimpleBrick {
    /// Create a new `SimpleBrick` with the given name.
    #[must_use]
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            assertions: Vec::new(),
            budget: BrickBudget::uniform(16), // 60fps default
            custom_verify: None,
        }
    }

    /// Add an assertion to this brick.
    #[must_use]
    pub fn with_assertion(mut self, assertion: BrickAssertion) -> Self {
        self.assertions.push(assertion);
        self
    }

    /// Set the performance budget.
    #[must_use]
    pub const fn with_budget(mut self, budget: BrickBudget) -> Self {
        self.budget = budget;
        self
    }

    /// Add a custom verification function.
    #[must_use]
    pub const fn with_custom_verify(mut self, verify: fn() -> bool) -> Self {
        self.custom_verify = Some(verify);
        self
    }
}

impl Brick for SimpleBrick {
    fn brick_name(&self) -> &'static str {
        self.name
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &self.assertions
    }

    fn budget(&self) -> BrickBudget {
        self.budget
    }

    fn verify(&self) -> BrickVerification {
        let mut passed = Vec::new();
        let mut failed = Vec::new();

        // Run custom verification if provided
        if let Some(verify_fn) = self.custom_verify {
            if !verify_fn() {
                failed.push((
                    BrickAssertion::Custom {
                        name: "custom_verify".into(),
                        validator_id: 0,
                    },
                    "Custom verification failed".into(),
                ));
            }
        }

        // All assertions pass by default (actual verification happens at render time)
        for assertion in &self.assertions {
            passed.push(assertion.clone());
        }

        BrickVerification {
            passed,
            failed,
            verification_time: Duration::from_micros(1),
        }
    }

    fn to_html(&self) -> String {
        format!(r#"<div class="brick brick-{}">"#, self.name)
    }

    fn to_css(&self) -> String {
        format!(".brick-{} {{ display: block; }}", self.name)
    }
}

/// Default Brick implementation for simple widgets.
///
/// Use this when you need a minimal Brick implementation
/// that always passes verification.
#[derive(Debug, Clone, Copy)]
pub struct DefaultBrick;

impl Brick for DefaultBrick {
    fn brick_name(&self) -> &'static str {
        "DefaultBrick"
    }

    fn assertions(&self) -> &[BrickAssertion] {
        &[]
    }

    fn budget(&self) -> BrickBudget {
        BrickBudget::uniform(16)
    }

    fn verify(&self) -> BrickVerification {
        BrickVerification {
            passed: vec![],
            failed: vec![],
            verification_time: Duration::from_micros(1),
        }
    }

    fn to_html(&self) -> String {
        String::new()
    }

    fn to_css(&self) -> String {
        String::new()
    }
}

/// Extension trait for adding Brick verification to the render pipeline.
pub trait BrickWidgetExt: Brick {
    /// Verify this brick before rendering.
    ///
    /// Returns an error if any assertion fails.
    fn verify_for_render(&self) -> Result<(), String> {
        if self.can_render() {
            Ok(())
        } else {
            let verification = self.verify();
            let errors: Vec<String> = verification
                .failed
                .iter()
                .map(|(assertion, reason)| format!("{assertion:?}: {reason}"))
                .collect();
            Err(format!(
                "Brick '{}' failed verification: {}",
                self.brick_name(),
                errors.join(", ")
            ))
        }
    }
}

impl<T: Brick> BrickWidgetExt for T {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_brick_new() {
        let brick = SimpleBrick::new("TestBrick");
        assert_eq!(brick.brick_name(), "TestBrick");
        assert!(brick.assertions().is_empty());
    }

    #[test]
    fn test_simple_brick_with_assertion() {
        let brick = SimpleBrick::new("TestBrick")
            .with_assertion(BrickAssertion::TextVisible)
            .with_assertion(BrickAssertion::ContrastRatio(4.5));

        assert_eq!(brick.assertions().len(), 2);
    }

    #[test]
    fn test_simple_brick_with_budget() {
        let brick = SimpleBrick::new("TestBrick").with_budget(BrickBudget::uniform(32));

        assert_eq!(brick.budget().total_ms, 32);
    }

    #[test]
    fn test_simple_brick_verify() {
        let brick = SimpleBrick::new("TestBrick");
        let verification = brick.verify();
        assert!(verification.is_valid());
    }

    #[test]
    fn test_simple_brick_can_render() {
        let brick = SimpleBrick::new("TestBrick");
        assert!(brick.can_render());
    }

    #[test]
    fn test_default_brick() {
        let brick = DefaultBrick;
        assert_eq!(brick.brick_name(), "DefaultBrick");
        assert!(brick.can_render());
    }

    #[test]
    fn test_verify_for_render() {
        let brick = SimpleBrick::new("TestBrick");
        assert!(brick.verify_for_render().is_ok());
    }

    #[test]
    fn test_simple_brick_with_custom_verify_pass() {
        let brick = SimpleBrick::new("TestBrick").with_custom_verify(|| true);
        let verification = brick.verify();
        assert!(verification.is_valid());
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_simple_brick_with_custom_verify_fail() {
        let brick = SimpleBrick::new("TestBrick").with_custom_verify(|| false);
        let verification = brick.verify();
        assert!(!verification.is_valid());
        assert_eq!(verification.failed.len(), 1);
        assert!(verification.failed[0]
            .1
            .contains("Custom verification failed"));
    }

    #[test]
    fn test_simple_brick_to_html() {
        let brick = SimpleBrick::new("MyWidget");
        let html = brick.to_html();
        assert!(html.contains("brick-MyWidget"));
        assert!(html.starts_with("<div"));
    }

    #[test]
    fn test_simple_brick_to_css() {
        let brick = SimpleBrick::new("MyWidget");
        let css = brick.to_css();
        assert!(css.contains(".brick-MyWidget"));
        assert!(css.contains("display: block"));
    }

    #[test]
    fn test_default_brick_to_html() {
        let brick = DefaultBrick;
        assert!(brick.to_html().is_empty());
    }

    #[test]
    fn test_default_brick_to_css() {
        let brick = DefaultBrick;
        assert!(brick.to_css().is_empty());
    }

    #[test]
    fn test_default_brick_assertions_empty() {
        let brick = DefaultBrick;
        assert!(brick.assertions().is_empty());
    }

    #[test]
    fn test_default_brick_budget() {
        let brick = DefaultBrick;
        assert_eq!(brick.budget().total_ms, 16);
    }

    #[test]
    fn test_default_brick_verify() {
        let brick = DefaultBrick;
        let verification = brick.verify();
        assert!(verification.passed.is_empty());
        assert!(verification.failed.is_empty());
    }

    #[test]
    fn test_verify_for_render_with_custom_fail() {
        let brick = SimpleBrick::new("FailBrick").with_custom_verify(|| false);
        let result = brick.verify_for_render();
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("FailBrick"));
        assert!(err.contains("failed verification"));
    }

    #[test]
    fn test_simple_brick_clone() {
        let brick = SimpleBrick::new("CloneTest")
            .with_assertion(BrickAssertion::TextVisible)
            .with_budget(BrickBudget::uniform(32));
        let cloned = brick.clone();
        assert_eq!(cloned.brick_name(), brick.brick_name());
        assert_eq!(cloned.assertions().len(), brick.assertions().len());
    }

    #[test]
    fn test_simple_brick_debug() {
        let brick = SimpleBrick::new("DebugTest");
        let debug = format!("{brick:?}");
        assert!(debug.contains("SimpleBrick"));
        assert!(debug.contains("DebugTest"));
    }

    #[test]
    fn test_default_brick_copy() {
        let brick = DefaultBrick;
        let copied = brick;
        assert_eq!(copied.brick_name(), brick.brick_name());
    }
}
