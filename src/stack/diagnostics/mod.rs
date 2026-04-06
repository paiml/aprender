//! Stack Visualization, Diagnostics, and Reporting
//!
//! ML-driven system for visualizing, diagnosing, and reporting on the health
//! of the Sovereign AI Stack. Implements Toyota Way principles for observability.
//!
//! ## Toyota Way Principles
//!
//! - **Mieruka (Visual Control)**: Rich ASCII dashboards make health visible
//! - **Jidoka**: ML anomaly detection surfaces issues automatically
//! - **Genchi Genbutsu**: Evidence-based diagnosis from actual dependency data
//! - **Andon**: Red/Yellow/Green status with stop-the-line alerts
//! - **Yokoten**: Cross-component insight sharing via knowledge graph

mod dashboard;
mod engine;
mod types;

#[cfg(test)]
#[path = "tests/mod.rs"]
mod tests;

// Re-export all public types from submodules
pub use dashboard::render_dashboard;
pub use engine::StackDiagnostics;
pub use types::{
    AndonStatus, Anomaly, AnomalyCategory, ComponentMetrics, ComponentNode, GraphMetrics,
    HealthStatus, HealthSummary,
};

// Re-export ML components from diagnostics_ml module
pub use super::diagnostics_ml::{ErrorForecaster, ForecastMetrics, IsolationForest};
