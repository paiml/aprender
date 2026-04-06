//! TUI Dataset Viewer Module
//!
//! Provides terminal-based visualization of Arrow datasets.
//! Designed for pure WASM compatibility with zero JavaScript dependencies.
//!
//! # Architecture
//!
//! The TUI module follows the presentar-terminal architecture:
//! - `DatasetAdapter` - Uniform access to Arrow datasets
//! - `DatasetViewer` - Scrollable table widget
//! - `SchemaInspector` - Schema display widget
//! - `RowDetailView` - Expanded row view widget
//!
//! # WASM Compatibility
//!
//! All components are designed for `wasm32-unknown-unknown`:
//! - No panic paths (no unwrap/expect)
//! - No filesystem access in WASM mode
//! - No threading (single-threaded model)
//! - Zero JavaScript dependencies
//!
//! # Example
//!
//! ```ignore
//! use alimentar::tui::{DatasetAdapter, DatasetViewer};
//! use alimentar::ArrowDataset;
//!
//! // Load dataset
//! let dataset = ArrowDataset::from_parquet("data.parquet")?;
//! let adapter = DatasetAdapter::from_dataset(&dataset)?;
//!
//! // Create viewer
//! let viewer = DatasetViewer::new(adapter);
//!
//! // Render to canvas
//! viewer.paint(&mut canvas);
//! ```

mod adapter;
mod error;
mod format;
mod row_detail;
mod schema_inspector;
mod scroll;
mod viewer;

// Public exports
pub use adapter::DatasetAdapter;
pub use error::{TuiError, TuiResult};
pub use format::{display_width, format_array_value, truncate_string};
pub use row_detail::RowDetailView;
pub use schema_inspector::SchemaInspector;
pub use scroll::ScrollState;
pub use viewer::DatasetViewer;
