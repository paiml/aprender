//! UI layout and rendering for ptop.
//!
//! Pixel-perfect ttop clone using presentar-terminal widgets.
//!
//! # Module Structure
//!
//! - `colors` - Panel color constants and gradient functions
//! - `helpers` - Formatting utilities and symbols
//! - `panels/` - Individual panel rendering (cpu, memory, disk, etc.)
//! - `overlays` - Help overlay, signal dialog, filter overlay

// Allow style-only clippy warnings that don't affect correctness
#![allow(clippy::too_many_lines)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::option_map_or_none)]

pub mod colors;
pub mod helpers;
pub mod overlays;
pub mod panels;

// Core rendering logic
mod core;

// Re-export commonly used items
pub use colors::*;
pub use helpers::*;

// Re-export core rendering functions
pub use core::{draw, panel_border_color, read_gpu_info, GpuInfo};
