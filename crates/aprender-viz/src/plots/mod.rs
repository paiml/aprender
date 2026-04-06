//! High-level plot types.
//!
//! Provides ready-to-use visualization types with builder APIs.

mod boxplot;
mod confusion_matrix;
mod force_graph;
mod heatmap;
mod histogram;
mod line;
mod loss_curve;
mod roc_pr;
mod scatter;

pub use boxplot::{BoxPlot, BoxStats, BuiltBoxPlot, BuiltViolinPlot, ViolinPlot};
pub use confusion_matrix::{ConfusionMatrix, ConfusionMatrixMetrics, Normalization};
pub use force_graph::{BuiltForceGraph, ForceGraph, GraphEdge, GraphNode};
pub use heatmap::{Heatmap, HeatmapPalette};
pub use histogram::{BinStrategy, Histogram};
pub use line::{douglas_peucker, LineChart, LineSeries};
pub use loss_curve::{LossCurve, MetricSeries, SeriesSummary};
pub use roc_pr::{compute_pr, compute_roc, PrCurve, PrData, RocCurve, RocData};
pub use scatter::ScatterPlot;
