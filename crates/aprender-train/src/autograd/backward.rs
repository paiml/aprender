//! Backward operation trait

/// Trait for backward pass operations
pub trait BackwardOp {
    /// Perform backward pass
    fn backward(&self);
}
