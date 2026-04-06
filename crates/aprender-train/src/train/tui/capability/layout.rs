//! Dashboard layout types.

/// Dashboard layout style.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DashboardLayout {
    /// Single-line progress only
    Minimal,
    /// Compact 5-line summary
    #[default]
    Compact,
    /// Full dashboard with charts
    Full,
}
