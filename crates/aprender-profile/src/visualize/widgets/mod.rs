//! Custom widgets for renacer visualize
//!
//! Provides specialized visualization widgets:
//! - braille_scatter: 2D scatter plot using braille characters
//! - gantt: Horizontal bar chart for span timing

pub mod braille_scatter;
pub mod gantt;

pub use braille_scatter::render as render_scatter;
pub use gantt::{GanttChart, GanttConfig};
