//! Chart components for terminal visualization
//!
//! - Feature Importance Display (ENT-064)
//! - Gradient Flow Heatmap (ENT-065)
//! - LossCurve Integration (ENT-056)

mod feature_importance;
mod gradient_flow;
mod loss_curve;

#[cfg(test)]
mod tests;

pub use feature_importance::FeatureImportanceChart;
pub use gradient_flow::GradientFlowHeatmap;
pub use loss_curve::{LossCurveDisplay, SeriesSummaryTuple};
