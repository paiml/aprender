//! Averaging strategies for multi-class metrics

/// Averaging strategy for multi-class metrics
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Average {
    /// Calculate metrics for each label, return unweighted mean
    Macro,
    /// Calculate metrics globally by counting total TP, FP, FN
    Micro,
    /// Weighted mean by support (number of true instances per label)
    Weighted,
    /// Return metrics per class (no averaging) - used internally
    None,
}
