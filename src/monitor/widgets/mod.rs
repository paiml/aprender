//! TUI widgets for the monitoring system.
//!
//! This module provides reusable widgets for building monitoring panels:
//!
//! - [`Graph`]: Time-series visualization with braille/block/TTY modes
//! - [`Meter`]: Percentage bar with gradient coloring
//! - [`Gauge`]: Arc/circular gauge for compact display
//! - [`Table`]: Sortable, scrollable data table
//! - [`Tree`]: Collapsible hierarchy view
//! - [`Sparkline`]: Inline mini-graph
//! - [`Heatmap`]: Grid heatmap for temperature/load visualization
//! - [`HorizonGraph`]: Ultra-dense time-series with layered bands
//! - [`BoxPlot`]: Box-and-whisker plot with quartiles and outliers
//! - [`ViolinPlot`]: Distribution plot with KDE
//! - [`DataFrame`]: Tabular data with inline visualizations
//!
//! All widgets implement the ratatui `Widget` trait for rendering.

pub mod boxplot;
pub mod confusion;
pub mod dataframe;
pub mod gauge;
pub mod graph;
pub mod heatmap;
pub mod histogram;
pub mod horizon;
pub mod meter;
pub mod sparkline;
pub mod table;
pub mod tree;
pub mod violin;

pub use boxplot::{BoxOrientation, BoxPlot, BoxStats};
pub use confusion::{ConfusionMatrix, MatrixPalette, Normalization};
pub use dataframe::{CellValue, Column, ColumnAlign, DataFrame, StatusLevel};
pub use gauge::{Gauge, GaugeMode};
pub use graph::{Graph, GraphMode};
pub use heatmap::{Heatmap, HeatmapCell, HeatmapPalette};
pub use histogram::{BarStyle, Bin, BinStrategy, Histogram, HistogramOrientation};
pub use horizon::{HorizonGraph, HorizonScheme};
pub use meter::Meter;
pub use sparkline::MonitorSparkline;
pub use table::{MonitorTable, SortDirection};
pub use tree::Tree;
pub use violin::{ViolinData, ViolinOrientation, ViolinPlot, ViolinStats};
