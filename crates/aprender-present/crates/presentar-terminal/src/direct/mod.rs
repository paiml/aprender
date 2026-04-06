//! Direct terminal backend (PROBAR-SPEC-009 compliant).
//!
//! This module provides a high-performance terminal rendering backend
//! that communicates directly with crossterm for optimal performance.
//!
//! # Architecture
//!
//! ```text
//! Canvas trait → DirectTerminalCanvas → crossterm
//!      ↑              ↑                    ↑
//!   presentar      unified              I/O
//! ```
//!
//! # Key Features
//!
//! - **Zero-allocation steady state**: Uses `CompactString` for inline strings
//! - **Smart diffing**: Only renders changed cells
//! - **Batched I/O**: Single `write()` syscall per frame
//! - **PROBAR-SPEC-009 compliant**: Implements Brick Architecture

mod cell_buffer;
mod diff_renderer;
mod direct_canvas;

pub use cell_buffer::{Cell, CellBuffer, Modifiers};
pub use diff_renderer::DiffRenderer;
pub use direct_canvas::DirectTerminalCanvas;
