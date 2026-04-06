#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::disallowed_methods)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::similar_names)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::manual_let_else)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::doc_markdown)]
//! Layout engine for Presentar UI framework.
//!
//! Implements Flexbox-inspired and CSS Grid layout with SIMD acceleration.
//!
//! # ComputeBlock Grid Compositor
//!
//! The `compute_block` module provides a compositor for managing TUI panel layouts:
//!
//! - **Intrinsic Sizing**: Widgets report min/preferred/max sizes via `SizeHint`
//! - **Cell Ownership**: Prevents rendering conflicts via `GridCompositor`
//! - **Clipping**: Enforces bounds at render time to prevent artifacts
//!
//! See `ComputeBlock` and `GridCompositor` for details.

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

mod cache;
mod compute_block;
mod engine;
mod flex;
mod grid;

pub use cache::LayoutCache;
pub use compute_block::{
    compute_intrinsic_layout, ClipMode, CompositorError, ComputeBlock, FlexConstraint,
    GridCompositor, IntrinsicSize, Rect, Size, SizeHint,
};
pub use engine::LayoutEngine;
pub use flex::{FlexAlign, FlexDirection, FlexItem, FlexJustify};
pub use grid::{
    auto_place_items, compute_grid_layout, GridAlign, GridArea, GridAutoFlow, GridItem, GridLayout,
    GridTemplate, TrackSize,
};
