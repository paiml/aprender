//! Core Brick trait and types (PROBAR-SPEC-009 alignment)
//!
//! # Brick Invariants (MANDATORY)
//!
//! 1. `assertions().len() > 0` - At least one falsifiable claim
//! 2. `verify()` checks ALL assertions - No skipping
//! 3. `can_render() == verify().is_valid()` - Jidoka gate
//! 4. `budget().total_ms() > 0` - Performance accountability
//!
//! # Reference
//!
//! Popper, K. (1959). "The Logic of Scientific Discovery"
//! - A theory that makes no falsifiable predictions is not scientific.

mod profiler;
mod score;
#[cfg(test)]
mod tests;
mod types;
mod widget;

use std::any::Any;

// Re-export all public items so `use crate::brick::*` still works
pub use profiler::BrickProfiler;
pub use score::{BrickGrade, BrickScore, Scorable};
pub use types::{
    fnv1a_f32, BrickAssertion, BrickBudget, BrickVerification, DivergenceReport, KernelTrace,
};
pub use widget::{Canvas, Color, Constraints, Point, Rect, Size, TextStyle, Widget};

/// Core Brick trait - all cbtop components implement this.
///
/// Provides quality infrastructure: assertions, budgets, verification.
pub trait Brick: Send + Sync {
    /// Unique brick name for identification
    fn brick_name(&self) -> &'static str;

    /// Falsifiable assertions (MUST be non-empty per Popper)
    fn assertions(&self) -> Vec<BrickAssertion>;

    /// Performance budget (Muda elimination)
    fn budget(&self) -> BrickBudget;

    /// Verification (Jidoka gate)
    fn verify(&self) -> BrickVerification;

    /// Test identifier for automation
    fn test_id(&self) -> Option<&str> {
        None
    }

    /// Can this brick render? (Jidoka gate)
    fn can_render(&self) -> bool {
        self.verify().is_valid()
    }

    /// Downcast to concrete type for assertion validation
    fn as_any(&self) -> &dyn Any;
}
