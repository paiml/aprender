//! Platform Efficiency (ENT-012)
//!
//! Provides platform-specific efficiency metrics for server and edge deployments.

mod budget;
mod edge;
mod platform_type;
mod server;

#[cfg(test)]
mod tests;

// Re-export all public types
pub use budget::{BudgetViolation, WasmBudget};
pub use edge::EdgeEfficiency;
pub use platform_type::PlatformEfficiency;
pub use server::ServerEfficiency;
