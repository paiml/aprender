//! Renacer visualization integration (VIZ-001/002/003/004).
//!
//! This module provides trueno-ublk metrics to the renacer visualization
//! framework, enabling real-time TUI dashboards, HTML reports, and OTLP export.
//!
//! # Architecture
//!
//! ```text
//! TieredPageStore::stats() ─┐
//!                           ├→ TruenoCollector → Renacer
//! BatchedPageStore::stats() ┘
//! ```

mod collector;

// Re-export for future integration (VIZ-001)
#[allow(unused_imports)]
pub use collector::TruenoCollector;
