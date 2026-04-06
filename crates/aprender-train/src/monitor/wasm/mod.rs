//! WebAssembly bindings for training monitor.
//!
//! Provides JavaScript-accessible functions for real-time training
//! visualization in the browser.
//!
//! # Usage (JavaScript)
//!
//! ```javascript
//! import init, { WasmMetricsCollector, WasmDashboard } from 'entrenar-monitor';
//!
//! await init();
//!
//! const collector = new WasmMetricsCollector();
//! collector.record_loss(0.5);
//! collector.record_accuracy(0.85);
//!
//! const stats = collector.summary_json();
//! console.log(JSON.parse(stats));
//!
//! const dashboard = new WasmDashboard(800, 400);
//! dashboard.update(collector);
//! const pngData = dashboard.render_png();
//! ```

mod collector;
mod dashboard;
mod options;
mod utils;

// Re-export all public types for API compatibility
pub use collector::WasmMetricsCollector;
pub use dashboard::WasmDashboard;
pub use options::WasmDashboardOptions;
pub use utils::{generate_sparkline, normalize_values};
