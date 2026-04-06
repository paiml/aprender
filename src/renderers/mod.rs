//! Generic Renderers for `DemoEngine` Implementations
//!
//! Per specification SIMULAR-DEMO-002: These renderers are renderer-agnostic.
//! Both TUI and WASM use the SAME `DemoEngine` and produce SAME state sequences.
//!
//! # Architecture
//!
//! ```text
//! DemoEngine (from YAML)
//!       ↓
//! ┌─────────────────────┐
//! │   GenericRenderer   │
//! │  (trait-based API)  │
//! └─────────────────────┘
//!       ↓           ↓           ↓
//!    TUI Impl    WASM Impl   SVG Impl
//! ```

pub mod keyframes;
pub mod svg;
pub mod tui;
pub mod wasm;

pub use keyframes::{KeyframeRecorder, KeyframesExport};
pub use svg::{SvgConfig, SvgRenderer};
pub use tui::{DemoRenderer, RenderFrame, RenderableDemo};
pub use wasm::{WasmRunner, WasmState};
