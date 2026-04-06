//! Dashboard widgets for experiment tracking and visualization.
//!
//! This module provides lightweight, composable widgets for ML experiment dashboards:
//!
//! - **Sparkline**: Mini line charts for loss/accuracy trends
//! - **ResourceBar**: Horizontal bars showing planned vs actual resource usage
//! - **RunTable**: Sortable tables for experiment run status
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno_viz::widgets::experiment::{Sparkline, ResourceBar, RunTable};
//!
//! // Create a sparkline for loss values
//! let sparkline = Sparkline::new(&[0.9, 0.7, 0.5, 0.3, 0.2])
//!     .dimensions(100, 20)
//!     .with_trend_indicator();
//! ```

/// Experiment dashboard widgets (sparklines, resource bars, run tables).
pub mod experiment;

pub use experiment::{ResourceBar, RunRow, RunStatus, RunTable, Sparkline, TrendDirection};
