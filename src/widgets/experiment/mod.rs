//! Experiment dashboard widgets for ML training visualization.
//!
//! Provides compact, embeddable widgets for experiment tracking dashboards.

mod resource_bar;
mod run_table;
mod sparkline;

pub use resource_bar::ResourceBar;
pub use run_table::{RunRow, RunStatus, RunTable};
pub use sparkline::{Sparkline, TrendDirection};
