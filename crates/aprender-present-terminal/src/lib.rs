//! Terminal backend for Presentar UI framework.
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::cast_lossless)]
// Style-only lints that don't affect correctness (SPEC-024 v5.8.0 quality compliance)
#![allow(clippy::use_self)] // "unnecessary structure name repetition"
#![allow(clippy::uninlined_format_args)] // "variables can be used directly in format!"
#![allow(clippy::needless_range_loop)] // "loop variable only used to index"
#![allow(clippy::bool_to_int_with_if)] // "unnecessary boolean not operation"
#![allow(clippy::manual_map)] // "map_or can be simplified"
#![allow(clippy::match_same_arms)] // "match arms have identical bodies" (intentional for readability)
#![allow(clippy::explicit_iter_loop)] // "more concise to loop over references"
#![allow(clippy::semicolon_if_nothing_returned)] // "consider adding semicolon"
#![allow(clippy::format_collect)] // "use of format! to build string"
#![allow(clippy::needless_pass_by_value)] // "argument passed by value but not consumed"
#![allow(clippy::redundant_closure)] // "redundant closure"
#![allow(clippy::struct_excessive_bools)] // "more than 3 bools in struct"
#![allow(clippy::manual_clamp)] // "clamp-like pattern"
#![allow(clippy::cast_possible_wrap)] // "casting may wrap around"
#![allow(clippy::nonminimal_bool)] // "unnecessary boolean not operation"
#![allow(clippy::option_map_or_none)] // "map_or can be simplified"
#![allow(clippy::redundant_closure_for_method_calls)] // "redundant closure"
#![allow(clippy::to_string_trait_impl)] // "calling to_string on &&str"
#![allow(clippy::map_clone)] // "explicit closure for copying elements"
#![allow(clippy::derivable_impls)] // "impl can be derived"
#![allow(clippy::if_same_then_else)] // "if has identical blocks"
#![allow(clippy::too_many_lines)] // "function has too many lines"
#![allow(clippy::needless_borrow)] // "borrowed expression implements traits"
#![allow(clippy::manual_str_repeat)] // "manual str::repeat"
#![allow(clippy::unreadable_literal)] // "long literal lacking separators"
#![allow(clippy::iter_cloned_collect)] // "implicitly cloning Vec"
#![allow(clippy::or_fun_call)] // "function call inside map_or"
#![allow(clippy::struct_field_names)] // "fields have same postfix"
#![allow(clippy::items_after_statements)] // "adding items after statements"
#![allow(clippy::collapsible_if)] // "all if blocks contain same code"
#![allow(clippy::map_unwrap_or)] // "map().unwrap_or()"
#![allow(clippy::implicit_clone)] // "implicitly cloning Vec"
#![allow(clippy::doc_markdown)] // "item in documentation missing backticks"
//!
//! This crate bridges `presentar_core` abstractions (Canvas, Widget, Brick) to
//! the terminal using `crossterm` directly.
//!
//! # Architecture (PROBAR-SPEC-009)
//!
//! The crate follows the Brick Architecture from PROBAR-SPEC-009:
//!
//! - All widgets implement `Brick` trait (tests define interface)
//! - Jidoka gate prevents rendering if assertions fail
//! - Performance budgets are enforced
//! - Zero-allocation steady-state rendering via direct crossterm backend
//!
//! # Example
//!
//! ```ignore
//! use presentar_terminal::{TuiApp, TuiConfig};
//! use presentar_core::{Brick, Widget};
//!
//! // Create your root widget (must implement Widget + Brick)
//! let root = MyRootWidget::new();
//!
//! // Run the application
//! TuiApp::new(root)?.run()?;
//! ```
//!
//! # Design Principles Enforcement
//!
//! This library enforces strict adherence to design principles (Tufte, Popper, Nielsen).
//! The following `include_str!` ensures that the design principles test suite exists at compile time.
#[cfg(test)]
const _DESIGN_PRINCIPLES_TESTS: &str = include_str!("../tests/design_principles_interface.rs");

#[macro_use]
#[allow(unused_macros)]
mod generated_contracts;

mod app;
pub mod cli;
mod color;
pub mod compute_block;
pub mod direct;
mod error;
mod input;
pub mod perf_trace;
pub mod random_seed;
pub mod seed;
pub mod theme;
pub mod tools;
pub mod widgets;

#[cfg(feature = "ptop")]
pub mod ptop;

// Re-export main types
pub use app::{AsyncCollector, QaTimings, Snapshot, SnapshotReceiver, TuiApp, TuiConfig};
pub use color::ColorMode;
pub use direct::{Cell, CellBuffer, DiffRenderer, DirectTerminalCanvas, Modifiers};
pub use error::TuiError;
pub use input::{InputHandler, KeyBinding};
pub use theme::{Gradient, Theme};

// Re-export widget types
pub use widgets::{
    truncate,
    // Data science widgets (sklearn/ggplot style)
    Axis,
    BarStyle,
    BinStrategy,
    // Legacy widgets (being phased out)
    Border,
    BorderStyle,
    BoxPlot,
    BoxStats,
    BrailleGraph,
    ColumnHighlight,
    CompactBreakdown,
    ConfusionMatrix,
    CpuGrid,
    Cursor,
    CurveData,
    CurveMode,
    EmaConfig,
    // UX utilities
    EmptyState,
    FocusRing,
    ForceGraph,
    ForceParams,
    Gauge,
    GaugeMode,
    GraphEdge,
    GraphMode,
    GraphNode,
    HealthStatus,
    HeatBarStyle,
    HeatScheme,
    Heatmap,
    HeatmapCell,
    HeatmapPalette,
    Histogram,
    HistogramOrientation,
    HorizonGraph,
    HorizonScheme,
    HugePages,
    LegendPosition,
    LineChart,
    LineStyle,
    LossCurve,
    LossSeries,
    MarkerStyle,
    MatrixPalette,
    MemoryBar,
    MemorySegment,
    Meter,
    // Tufte-inspired data visualization widgets
    MicroHeatBar,
    MultiBarGraph,
    MultiBarMode,
    NetworkInterface,
    NetworkPanel,
    NodeId,
    Normalization,
    Orientation,
    ProcessEntry,
    ProcessSort,
    ProcessState,
    ProcessTable,
    RocPrCurve,
    // Tufte-inspired selection highlighting
    RowHighlight,
    ScatterAxis,
    ScatterPlot,
    Segment,
    SegmentedMeter,
    Series,
    Simplification,
    Sparkline,
    Table,
    TitleBar,
    TitleBarPosition,
    TitleBarStyle,
    Tree,
    TreeNode,
    Treemap,
    TreemapLayout,
    TreemapNode,
    TrendDirection,
    ViolinData,
    ViolinOrientation,
    ViolinPlot,
    ViolinStats,
    DIMMED_BG,
    SELECTION_ACCENT,
    SELECTION_BG,
    SELECTION_GUTTER,
};

// Re-export core types for convenience
pub use presentar_core::{
    Brick, BrickAssertion, BrickBudget, BrickVerification, Canvas, Color, Constraints, Point, Rect,
    Size, TextStyle, Widget,
};

// Re-export random seed types (F1: Popper falsifiability)
pub use random_seed::{
    get_seed, init_from_env, set_global_seed, with_seed, SeededRng, DEFAULT_SEED,
};

// Re-export ComputeBlock types (SPEC-024 Section 15, 20)
pub use compute_block::{
    ComputeBlock, ComputeBlockId, CpuFrequencyBlock, CpuGovernor, CpuGovernorBlock,
    FrequencyScalingState, GpuThermalBlock, GpuThermalState, GpuVramBlock, HugePagesBlock,
    LoadTrendBlock, MemPressureBlock, MemoryPressureLevel, SimdInstructionSet, SparklineBlock,
    TrendDirection as ComputeTrendDirection,
};
