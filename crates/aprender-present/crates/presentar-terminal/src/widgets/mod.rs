//! Standard terminal widgets implementing Brick + Widget traits.
//!
//! All widgets in this module follow PROBAR-SPEC-009:
//! - Non-empty assertions (tests define interface)
//! - Performance budgets
//! - Jidoka verification gates
//!
//! ## Display Rules (SPEC-024 Section 28)
//!
//! All text and numbers MUST use the `display_rules` module for:
//! - Byte formatting: `format_bytes_si(1500)` → "1.5K"
//! - Percentages: `format_percent(45.3)` → "45.3%"
//! - Truncation: `truncate(s, 10, TruncateStrategy::Path)` → "/home/…/file"
//! - Columns: `format_column(s, 10, ColumnAlign::Left, ...)` → NEVER bleeds

mod border;
mod box_plot;
mod cluster_plot;
mod collapsible_panel;
mod confusion_matrix;
mod connections_panel;
mod containers_panel;
mod core_stats_dataframe;
mod cpu_exploded;
mod cpu_grid;
mod dataframe;
pub mod display_rules;
mod feature_importance;
mod files_panel;
mod force_graph;
mod gauge;
mod gpu_panel;
mod graph;
mod heatmap;
mod histogram;
mod horizon;
mod info_dense;
mod layout;
mod line_chart;
mod loss_curve;
mod memory_bar;
mod meter;
mod micro_heat_bar;
mod multi_bar;
mod network_panel;
mod parallel_coords;
mod pca_plot;
mod process_dataframe;
mod process_table;
mod radar_plot;
mod roc_pr_curve;
mod scatter_plot;
mod scrollbar;
mod segmented_meter;
pub mod selection;
mod sensors_panel;
mod sparkline;
mod symbols;
mod table;
mod text;
mod text_input;
mod title_bar;
mod tree;
mod treemap;
mod ux;
mod violin_plot;

pub use border::{Border, BorderStyle};
pub use box_plot::{BoxPlot, BoxStats, Orientation};
pub use cluster_plot::{ClusterAlgorithm, ClusterPlot};
pub use collapsible_panel::{CollapseDirection, CollapseIndicators, CollapsiblePanel};
pub use confusion_matrix::{ConfusionMatrix, MatrixPalette, Normalization};
pub use connections_panel::{ConnectionEntry, ConnectionsPanel, TcpState};
pub use containers_panel::{ContainerEntry, ContainerState, ContainersPanel};
pub use core_stats_dataframe::{CoreStatsDataFrame, CoreStatsRow, CoreStatsSortColumn};
pub use cpu_exploded::{
    CpuCoreState, CpuStateBreakdown, FreqTempHeatmap, LoadAverageTimeline, PerCoreSparklineGrid,
    TopProcess, TopProcessesMini,
};
pub use cpu_grid::CpuGrid;
pub use dataframe::{CellValue, Column, ColumnAlign, DataFrame, StatusLevel};
pub use feature_importance::FeatureImportance;
pub use files_panel::{FileEntry, FilesPanel};
pub use force_graph::{ForceGraph, ForceParams, GraphEdge, GraphNode};
pub use gauge::{Gauge, GaugeMode};
pub use gpu_panel::{GpuDevice, GpuPanel, GpuProcess, GpuVendor};
pub use graph::{BrailleGraph, GraphMode};
pub use heatmap::{Heatmap, HeatmapCell, HeatmapPalette};
pub use histogram::{BarStyle, BinStrategy, Histogram, HistogramOrientation};
pub use horizon::{HorizonGraph, HorizonScheme};
pub use info_dense::{
    CoreUtilizationHistogram, CpuConsumer, HealthLevel, SystemStatus, TopProcessesTable,
    TrendSparkline,
};
pub use layout::{Direction, Layout, LayoutItem, SizeSpec};
pub use line_chart::{Axis, LegendPosition, LineChart, LineStyle, Series, Simplification};
pub use loss_curve::{EmaConfig, LossCurve, LossSeries};
pub use memory_bar::{HugePages, MemoryBar, MemorySegment};
pub use meter::Meter;
pub use micro_heat_bar::{BarStyle as HeatBarStyle, CompactBreakdown, HeatScheme, MicroHeatBar};
pub use multi_bar::{MultiBarGraph, MultiBarMode};
pub use network_panel::{NetworkInterface, NetworkPanel};
pub use parallel_coords::ParallelCoordinates;
pub use pca_plot::{EigenPlotType, PCAPlot};
pub use process_dataframe::{
    ProcessColumnWidths, ProcessDataFrame, ProcessDisplayState, ProcessRow, ProcessSortColumn,
};
pub use process_table::{ProcessEntry, ProcessSort, ProcessState, ProcessTable};
pub use radar_plot::{RadarPlot, RadarSeries};
pub use roc_pr_curve::{CurveData, CurveMode, RocPrCurve};
pub use scatter_plot::{MarkerStyle, ScatterAxis, ScatterPlot};
pub use scrollbar::{ScrollOrientation, Scrollbar, ScrollbarChars};
pub use segmented_meter::{Segment, SegmentedMeter};
pub use selection::{
    ColumnHighlight, Cursor, FocusRing, RowHighlight, DIMMED_BG, SELECTION_ACCENT, SELECTION_BG,
    SELECTION_GUTTER,
};
pub use sensors_panel::{SensorReading, SensorStatus, SensorsPanel};
pub use sparkline::{Sparkline, TrendDirection};
pub use symbols::{
    BrailleSymbols, CustomSymbols, SymbolSet, BLOCK_DOWN, BLOCK_UP, BRAILLE_DOWN, BRAILLE_UP,
    SPARKLINE, SUBSCRIPT, SUPERSCRIPT, TTY_DOWN, TTY_UP,
};
pub use table::Table;
pub use text::{Text, TextAlign};
pub use text_input::TextInput;
pub use title_bar::{TitleBar, TitleBarPosition, TitleBarStyle};
pub use tree::{NodeId, Tree, TreeNode};
pub use treemap::{Treemap, TreemapLayout, TreemapNode};
pub use ux::{truncate, truncate_middle, truncate_with, EmptyState, HealthStatus};
pub use violin_plot::{ViolinData, ViolinOrientation, ViolinPlot, ViolinStats};

// Display Rules (SPEC-024 Section 28) - Grammar of Graphics formatting
pub use display_rules::{
    format_bytes_column,
    format_bytes_iec,
    // Byte formatting
    format_bytes_si,
    format_column,
    // Duration/time formatting
    format_duration,
    format_duration_compact,
    // Other formatters
    format_freq_mhz,
    format_number_column,
    // Percentage formatting
    format_percent,
    format_percent_clamped,
    format_percent_column,
    format_percent_fixed,
    format_rate,
    format_temp_c,
    truncate as truncate_display,
    // Declarative Display Rules (SPEC-024 Appendix F)
    BatteryDisplayRules,
    // Column formatting (NEVER bleeds) - use DisplayColumnAlign to avoid conflict with dataframe::ColumnAlign
    ColumnAlign as DisplayColumnAlign,
    DataAvailability,
    DefaultDisplayRules,
    DisplayAction,
    DisplayContext,
    DisplayRules,
    FilesDisplayRules,
    // O(1) fuzzy search
    FuzzyIndex,
    GpuDisplayRules,
    PsiDisplayRules,
    SearchResult,
    SensorsDisplayRules,
    SystemCapabilities,
    TerminalSize as DisplayTerminalSize,
    // Truncation
    TruncateStrategy,
};
mod proportional_bar;
pub use proportional_bar::{BarSegment, ProportionalBar};

// Atomic Widget Layer (SPEC-024 Appendix I)
mod flex_cell;
mod gutter_cursor;
mod labeled_bar;
mod semantic_label;

pub use flex_cell::{Alignment as FlexAlignment, FlexCell, Overflow};
pub use gutter_cursor::{CursorStyle, GutterCursor, SelectionState};
pub use labeled_bar::{LabeledBar, LabeledBarLayout};
pub use semantic_label::{SemanticLabel, SemanticStatus};
