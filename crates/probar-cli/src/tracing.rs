//! Deep Tracing Module (PROBAR-SPEC-006 Section J)
//!
//! Implements syscall tracing, WASM event capture, flamegraph generation,
//! and source correlation for performance analysis.
//!
//! Based on research:
//! - [C9] Treadmill methodology for trace attribution

#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::format_push_string)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::use_self)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::map_unwrap_or)]
#![allow(clippy::redundant_closure_for_method_calls)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

// =============================================================================
// J.2 Trace Categories
// =============================================================================

/// Trace category for filtering
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TraceCategory {
    /// System call tracing
    Syscall,
    /// WASM-specific events
    Wasm,
    /// Network events
    Network,
    /// Memory operations
    Memory,
    /// GPU/rendering events
    Gpu,
}

impl TraceCategory {
    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Syscall => "Syscall",
            Self::Wasm => "WASM",
            Self::Network => "Network",
            Self::Memory => "Memory",
            Self::Gpu => "GPU",
        }
    }

    /// Get all categories
    pub fn all() -> Vec<Self> {
        vec![
            Self::Syscall,
            Self::Wasm,
            Self::Network,
            Self::Memory,
            Self::Gpu,
        ]
    }
}

// =============================================================================
// J.4 Trace Configuration
// =============================================================================

/// Deep trace configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    /// Categories to trace
    pub categories: Vec<TraceCategory>,
    /// Sample rate (0.0 - 1.0)
    pub sample_rate: f64,
    /// Maximum events to capture
    pub max_events: usize,
    /// Source map path for Rust code
    pub source_map: Option<PathBuf>,
    /// WASM source map path
    pub wasm_source_map: Option<PathBuf>,
    /// Output path for trace file
    pub output_path: Option<PathBuf>,
}

impl TraceConfig {
    /// Create default config
    pub fn new() -> Self {
        Self {
            categories: TraceCategory::all(),
            sample_rate: 1.0,
            max_events: 100_000,
            source_map: None,
            wasm_source_map: None,
            output_path: None,
        }
    }

    /// Enable only specific categories
    pub fn with_categories(mut self, cats: Vec<TraceCategory>) -> Self {
        self.categories = cats;
        self
    }

    /// Set sample rate
    pub fn with_sample_rate(mut self, rate: f64) -> Self {
        self.sample_rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set source map
    pub fn with_source_map(mut self, path: PathBuf) -> Self {
        self.source_map = Some(path);
        self
    }
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// J.3 Trace Events
// =============================================================================

/// A span in the trace timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    /// Span name
    pub name: String,
    /// Category
    pub category: TraceCategory,
    /// Start time in microseconds
    pub start_us: u64,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Thread ID
    pub thread_id: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TraceSpan {
    /// Create a new span
    pub fn new(name: &str, category: TraceCategory, start_us: u64, duration_us: u64) -> Self {
        Self {
            name: name.to_string(),
            category,
            start_us,
            duration_us,
            thread_id: 0,
            metadata: HashMap::new(),
        }
    }

    /// End time
    pub fn end_us(&self) -> u64 {
        self.start_us + self.duration_us
    }

    /// Duration in milliseconds
    pub fn duration_ms(&self) -> f64 {
        self.duration_us as f64 / 1000.0
    }

    /// Add metadata
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Syscall statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyscallStats {
    /// Syscall name
    pub name: String,
    /// Call count
    pub count: u64,
    /// Total time in microseconds
    pub total_us: u64,
    /// Average time in microseconds
    pub avg_us: u64,
    /// Maximum time in microseconds
    pub max_us: u64,
    /// Percentage of total time
    pub percent: f64,
}

impl SyscallStats {
    /// Create new stats
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            ..Default::default()
        }
    }

    /// Record a syscall
    pub fn record(&mut self, duration_us: u64) {
        self.count += 1;
        self.total_us += duration_us;
        self.max_us = self.max_us.max(duration_us);
        self.avg_us = self.total_us / self.count;
    }
}

/// WASM-specific event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WasmEvent {
    /// Event type
    pub event_type: WasmEventType,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Memory impact in bytes
    pub memory_impact: i64,
    /// Source location (if available)
    pub source_location: Option<SourceLocation>,
}

/// WASM event types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WasmEventType {
    /// WASM module compilation
    Compile,
    /// Module instantiation
    Instantiate,
    /// Function call
    Call,
    /// Memory grow operation
    MemoryGrow,
    /// Table operation
    TableOp,
}

impl WasmEventType {
    /// Get display name
    pub fn name(&self) -> &'static str {
        match self {
            Self::Compile => "wasm_compile",
            Self::Instantiate => "wasm_instantiate",
            Self::Call => "wasm_call",
            Self::MemoryGrow => "wasm_memory_grow",
            Self::TableOp => "wasm_table_op",
        }
    }
}

/// Source code location
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    /// File path
    pub file: PathBuf,
    /// Line number
    pub line: u32,
    /// Column number
    pub column: Option<u32>,
    /// Function name
    pub function: Option<String>,
}

impl SourceLocation {
    /// Create new location
    pub fn new(file: PathBuf, line: u32) -> Self {
        Self {
            file,
            line,
            column: None,
            function: None,
        }
    }

    /// Format as string
    pub fn display(&self) -> String {
        if let Some(ref func) = self.function {
            format!("{}:{} ({})", self.file.display(), self.line, func)
        } else {
            format!("{}:{}", self.file.display(), self.line)
        }
    }
}

// =============================================================================
// J.3 Source Hotspots
// =============================================================================

/// Source code hotspot (performance bottleneck)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceHotspot {
    /// File path
    pub file: PathBuf,
    /// Line number
    pub line: u32,
    /// Function name
    pub function: String,
    /// Total time in microseconds
    pub total_us: u64,
    /// Call count
    pub call_count: u64,
    /// Optimization suggestion
    pub suggestion: Option<OptimizationSuggestion>,
}

impl SourceHotspot {
    /// Create new hotspot
    pub fn new(file: PathBuf, line: u32, function: &str) -> Self {
        Self {
            file,
            line,
            function: function.to_string(),
            total_us: 0,
            call_count: 0,
            suggestion: None,
        }
    }

    /// Total time in milliseconds
    pub fn total_ms(&self) -> f64 {
        self.total_us as f64 / 1000.0
    }
}

/// Optimization suggestion for hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationSuggestion {
    /// Use SIMD instructions
    UseSIMD {
        /// Expected speedup factor
        expected_speedup: f64,
    },
    /// Use object pool
    UsePool {
        /// Current allocation count
        current_allocs: u64,
    },
    /// Batch operations
    BatchOperations {
        /// Current call count
        current_calls: u64,
    },
    /// Use async I/O
    AsyncIO {
        /// Blocking time in microseconds
        blocking_us: u64,
    },
}

impl OptimizationSuggestion {
    /// Get display hint
    pub fn hint(&self) -> &'static str {
        match self {
            Self::UseSIMD { .. } => "⚠ SIMD",
            Self::UsePool { .. } => "⚠ Pool",
            Self::BatchOperations { .. } => "⚠ Batch",
            Self::AsyncIO { .. } => "⚠ Async",
        }
    }
}

// =============================================================================
// J.3 Trace Analysis
// =============================================================================

/// Complete trace analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceAnalysis {
    /// Request identifier
    pub request_id: String,
    /// Total duration in microseconds
    pub total_us: u64,
    /// Timeline of spans
    pub timeline: Vec<TraceSpan>,
    /// Syscall breakdown
    pub syscall_breakdown: HashMap<String, SyscallStats>,
    /// WASM events
    pub wasm_events: Vec<WasmEvent>,
    /// Source hotspots
    pub source_hotspots: Vec<SourceHotspot>,
    /// Critical path components
    pub critical_path: Vec<String>,
}

impl TraceAnalysis {
    /// Create new analysis
    pub fn new(request_id: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            total_us: 0,
            timeline: Vec::new(),
            syscall_breakdown: HashMap::new(),
            wasm_events: Vec::new(),
            source_hotspots: Vec::new(),
            critical_path: Vec::new(),
        }
    }

    /// Add a span
    pub fn add_span(&mut self, span: TraceSpan) {
        self.total_us = self.total_us.max(span.end_us());
        self.timeline.push(span);
    }

    /// Record a syscall
    pub fn record_syscall(&mut self, name: &str, duration_us: u64) {
        self.syscall_breakdown
            .entry(name.to_string())
            .or_insert_with(|| SyscallStats::new(name))
            .record(duration_us);
    }

    /// Add a WASM event
    pub fn add_wasm_event(&mut self, event: WasmEvent) {
        self.wasm_events.push(event);
    }

    /// Add a hotspot
    pub fn add_hotspot(&mut self, hotspot: SourceHotspot) {
        self.source_hotspots.push(hotspot);
    }

    /// Calculate critical path
    pub fn calculate_critical_path(&mut self) {
        // Sort spans by duration descending
        let mut spans: Vec<_> = self.timeline.iter().collect();
        spans.sort_by(|a, b| b.duration_us.cmp(&a.duration_us));

        // Take top contributors
        self.critical_path = spans
            .iter()
            .take(5)
            .map(|s| {
                format!(
                    "{} ({:.1}%)",
                    s.name,
                    s.duration_us as f64 / self.total_us as f64 * 100.0
                )
            })
            .collect();
    }

    /// Calculate syscall percentages
    pub fn calculate_syscall_percentages(&mut self) {
        let total: u64 = self.syscall_breakdown.values().map(|s| s.total_us).sum();
        if total > 0 {
            for stats in self.syscall_breakdown.values_mut() {
                stats.percent = stats.total_us as f64 / total as f64 * 100.0;
            }
        }
    }

    /// Total duration in milliseconds
    pub fn total_ms(&self) -> f64 {
        self.total_us as f64 / 1000.0
    }
}

// =============================================================================
// J.2 Flamegraph Data
// =============================================================================

/// Flamegraph node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamegraphNode {
    /// Function/span name
    pub name: String,
    /// Self time (not including children)
    pub self_time_us: u64,
    /// Total time (including children)
    pub total_time_us: u64,
    /// Children nodes
    pub children: Vec<FlamegraphNode>,
}

impl FlamegraphNode {
    /// Create new node
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            self_time_us: 0,
            total_time_us: 0,
            children: Vec::new(),
        }
    }

    /// Add time
    pub fn add_time(&mut self, us: u64) {
        self.self_time_us += us;
        self.total_time_us += us;
    }

    /// Add child
    pub fn add_child(&mut self, child: FlamegraphNode) {
        self.total_time_us += child.total_time_us;
        self.children.push(child);
    }
}

/// Flamegraph data structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flamegraph {
    /// Root nodes
    pub roots: Vec<FlamegraphNode>,
    /// Total time in microseconds
    pub total_us: u64,
}

impl Flamegraph {
    /// Create new flamegraph
    pub fn new() -> Self {
        Self {
            roots: Vec::new(),
            total_us: 0,
        }
    }

    /// Add a root node
    pub fn add_root(&mut self, node: FlamegraphNode) {
        self.total_us += node.total_time_us;
        self.roots.push(node);
    }

    /// Render as folded stack format (for external tools)
    pub fn to_folded(&self) -> String {
        let mut out = String::new();
        for root in &self.roots {
            self.fold_node(&mut out, root, "");
        }
        out
    }

    #[allow(clippy::self_only_used_in_recursion)]
    fn fold_node(&self, out: &mut String, node: &FlamegraphNode, prefix: &str) {
        let path = if prefix.is_empty() {
            node.name.clone()
        } else {
            format!("{};{}", prefix, node.name)
        };

        if node.self_time_us > 0 {
            out.push_str(&format!("{} {}\n", path, node.self_time_us));
        }

        for child in &node.children {
            self.fold_node(out, child, &path);
        }
    }
}

impl Default for Flamegraph {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Rendering
// =============================================================================

/// Render trace analysis as TUI
#[allow(clippy::too_many_lines)]
pub fn render_trace_report(analysis: &TraceAnalysis) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "DEEP TRACE ANALYSIS: {} (total: {:.1}ms)\n",
        analysis.request_id,
        analysis.total_ms()
    ));
    out.push_str("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");

    // Timeline
    out.push_str("TIMELINE\n");
    out.push_str("┌──────────────────────────────────────────────────────────────────────────┐\n");
    for span in &analysis.timeline {
        let bar_len = (span.duration_us as f64 / analysis.total_us as f64 * 30.0) as usize;
        let bar: String = "█".repeat(bar_len.max(1));
        out.push_str(&format!(
            "│ [{:<8}] {:30} {:.1}ms\n",
            span.category.name(),
            bar,
            span.duration_ms()
        ));
    }
    if !analysis.critical_path.is_empty() {
        out.push_str(&format!(
            "│ Critical Path: {}\n",
            analysis.critical_path.join(" → ")
        ));
    }
    out.push_str(
        "└──────────────────────────────────────────────────────────────────────────┘\n\n",
    );

    // Syscall breakdown
    if !analysis.syscall_breakdown.is_empty() {
        out.push_str("SYSCALL BREAKDOWN\n");
        out.push_str(
            "┌─────────────────┬───────┬────────────┬──────────┬──────────┬────────────┐\n",
        );
        out.push_str(
            "│ Syscall         │ Count │ Total Time │ Avg Time │ Max Time │ % of Total │\n",
        );
        out.push_str(
            "├─────────────────┼───────┼────────────┼──────────┼──────────┼────────────┤\n",
        );

        let mut stats: Vec<_> = analysis.syscall_breakdown.values().collect();
        stats.sort_by(|a, b| b.total_us.cmp(&a.total_us));

        for stat in stats.iter().take(10) {
            out.push_str(&format!(
                "│ {:<15} │ {:>5} │ {:>8.1}ms │ {:>6}μs │ {:>6}μs │ {:>9.1}% │\n",
                truncate(&stat.name, 15),
                stat.count,
                stat.total_us as f64 / 1000.0,
                stat.avg_us,
                stat.max_us,
                stat.percent
            ));
        }
        out.push_str(
            "└─────────────────┴───────┴────────────┴──────────┴──────────┴────────────┘\n\n",
        );
    }

    // WASM events
    if !analysis.wasm_events.is_empty() {
        out.push_str("WASM-SPECIFIC EVENTS\n");
        out.push_str(
            "┌────────────────────┬──────────┬───────────────┬─────────────────────────┐\n",
        );
        out.push_str(
            "│ Event              │ Duration │ Memory Impact │ Source Location         │\n",
        );
        out.push_str(
            "├────────────────────┼──────────┼───────────────┼─────────────────────────┤\n",
        );

        for event in &analysis.wasm_events {
            let memory_str = if event.memory_impact >= 0 {
                format!("+{}KB", event.memory_impact / 1024)
            } else {
                format!("{}KB", event.memory_impact / 1024)
            };
            let source = event
                .source_location
                .as_ref()
                .map(|s| s.display())
                .unwrap_or_else(|| "(internal)".to_string());
            out.push_str(&format!(
                "│ {:<18} │ {:>6}ms │ {:>13} │ {:<23} │\n",
                event.event_type.name(),
                event.duration_us / 1000,
                memory_str,
                truncate(&source, 23)
            ));
        }
        out.push_str(
            "└────────────────────┴──────────┴───────────────┴─────────────────────────┘\n\n",
        );
    }

    // Source hotspots
    if !analysis.source_hotspots.is_empty() {
        out.push_str("SOURCE CORRELATION (top hotspots)\n");
        out.push_str(
            "┌─────────────────────────┬─────────────────────┬───────┬───────┬───────────┐\n",
        );
        out.push_str(
            "│ File:Line               │ Function            │ Time  │ Calls │ Suggestion│\n",
        );
        out.push_str(
            "├─────────────────────────┼─────────────────────┼───────┼───────┼───────────┤\n",
        );

        for hotspot in analysis.source_hotspots.iter().take(5) {
            let file_line = format!("{}:{}", hotspot.file.display(), hotspot.line);
            let suggestion = hotspot
                .suggestion
                .as_ref()
                .map(|s| s.hint())
                .unwrap_or("✓ OK");
            out.push_str(&format!(
                "│ {:<23} │ {:<19} │ {:>4}ms│ {:>5} │ {:<9} │\n",
                truncate(&file_line, 23),
                truncate(&hotspot.function, 19),
                hotspot.total_ms() as u64,
                hotspot.call_count,
                suggestion
            ));
        }
        out.push_str(
            "└─────────────────────────┴─────────────────────┴───────┴───────┴───────────┘\n",
        );
    }

    out
}

/// Render as JSON
pub fn render_trace_json(analysis: &TraceAnalysis) -> String {
    serde_json::to_string_pretty(analysis).unwrap_or_else(|_| "{}".to_string())
}

/// Truncate string
fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}…", &s[..max - 1])
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_trace_category() {
        assert_eq!(TraceCategory::Syscall.name(), "Syscall");
        assert_eq!(TraceCategory::all().len(), 5);
    }

    #[test]
    fn test_trace_config() {
        let config = TraceConfig::new()
            .with_sample_rate(0.5)
            .with_categories(vec![TraceCategory::Wasm]);

        assert_eq!(config.sample_rate, 0.5);
        assert_eq!(config.categories.len(), 1);
    }

    #[test]
    fn test_trace_span() {
        let span =
            TraceSpan::new("test", TraceCategory::Network, 1000, 500).with_metadata("key", "value");

        assert_eq!(span.end_us(), 1500);
        assert_eq!(span.duration_ms(), 0.5);
        assert_eq!(span.metadata.get("key"), Some(&"value".to_string()));
    }

    #[test]
    fn test_syscall_stats() {
        let mut stats = SyscallStats::new("read");
        stats.record(100);
        stats.record(200);

        assert_eq!(stats.count, 2);
        assert_eq!(stats.total_us, 300);
        assert_eq!(stats.avg_us, 150);
        assert_eq!(stats.max_us, 200);
    }

    #[test]
    fn test_wasm_event_type() {
        assert_eq!(WasmEventType::Compile.name(), "wasm_compile");
        assert_eq!(WasmEventType::MemoryGrow.name(), "wasm_memory_grow");
    }

    #[test]
    fn test_source_location() {
        let mut loc = SourceLocation::new(PathBuf::from("src/main.rs"), 42);
        loc.function = Some("main".to_string());

        assert!(loc.display().contains("src/main.rs:42"));
        assert!(loc.display().contains("main"));
    }

    #[test]
    fn test_source_hotspot() {
        let hotspot = SourceHotspot::new(PathBuf::from("src/lib.rs"), 100, "process");
        assert_eq!(hotspot.function, "process");
    }

    #[test]
    fn test_optimization_suggestion() {
        let suggestion = OptimizationSuggestion::UseSIMD {
            expected_speedup: 4.0,
        };
        assert_eq!(suggestion.hint(), "⚠ SIMD");
    }

    #[test]
    fn test_trace_analysis() {
        let mut analysis = TraceAnalysis::new("req-123");
        analysis.add_span(TraceSpan::new("network", TraceCategory::Network, 0, 1000));
        analysis.record_syscall("read", 500);
        analysis.calculate_critical_path();

        assert_eq!(analysis.total_us, 1000);
        assert_eq!(analysis.timeline.len(), 1);
        assert!(analysis.syscall_breakdown.contains_key("read"));
    }

    #[test]
    fn test_flamegraph() {
        let mut fg = Flamegraph::new();
        let mut root = FlamegraphNode::new("main");
        root.add_time(1000);

        let mut child = FlamegraphNode::new("process");
        child.add_time(500);
        root.add_child(child);

        fg.add_root(root);

        assert_eq!(fg.total_us, 1500);
        let folded = fg.to_folded();
        assert!(folded.contains("main"));
    }

    #[test]
    fn test_render_trace_report() {
        let mut analysis = TraceAnalysis::new("test-request");
        analysis.add_span(TraceSpan::new("dns", TraceCategory::Network, 0, 5000));
        analysis.record_syscall("read", 1000);

        let report = render_trace_report(&analysis);
        assert!(report.contains("test-request"));
        assert!(report.contains("TIMELINE"));
    }

    #[test]
    fn test_render_trace_json() {
        let analysis = TraceAnalysis::new("json-test");
        let json = render_trace_json(&analysis);
        assert!(json.contains("json-test"));
    }

    #[test]
    fn test_trace_category_all_names() {
        // Cover all TraceCategory::name() branches
        assert_eq!(TraceCategory::Syscall.name(), "Syscall");
        assert_eq!(TraceCategory::Wasm.name(), "WASM");
        assert_eq!(TraceCategory::Network.name(), "Network");
        assert_eq!(TraceCategory::Memory.name(), "Memory");
        assert_eq!(TraceCategory::Gpu.name(), "GPU");
    }

    #[test]
    fn test_trace_config_default() {
        let config = TraceConfig::default();
        assert_eq!(config.sample_rate, 1.0);
        assert_eq!(config.max_events, 100_000);
        assert_eq!(config.categories.len(), 5);
    }

    #[test]
    fn test_trace_config_with_source_map() {
        let config = TraceConfig::new().with_source_map(PathBuf::from("/src/source.map"));
        assert_eq!(config.source_map, Some(PathBuf::from("/src/source.map")));
    }

    #[test]
    fn test_trace_config_sample_rate_clamp() {
        let config_high = TraceConfig::new().with_sample_rate(1.5);
        assert_eq!(config_high.sample_rate, 1.0);

        let config_low = TraceConfig::new().with_sample_rate(-0.5);
        assert_eq!(config_low.sample_rate, 0.0);
    }

    #[test]
    fn test_wasm_event_type_all_names() {
        // Cover all WasmEventType::name() branches
        assert_eq!(WasmEventType::Compile.name(), "wasm_compile");
        assert_eq!(WasmEventType::Instantiate.name(), "wasm_instantiate");
        assert_eq!(WasmEventType::Call.name(), "wasm_call");
        assert_eq!(WasmEventType::MemoryGrow.name(), "wasm_memory_grow");
        assert_eq!(WasmEventType::TableOp.name(), "wasm_table_op");
    }

    #[test]
    fn test_source_location_without_function() {
        let loc = SourceLocation::new(PathBuf::from("src/lib.rs"), 10);
        let display = loc.display();
        assert!(display.contains("src/lib.rs:10"));
        assert!(!display.contains('('));
    }

    #[test]
    fn test_source_hotspot_total_ms() {
        let mut hotspot = SourceHotspot::new(PathBuf::from("src/hot.rs"), 50, "hot_function");
        hotspot.total_us = 5000;
        assert_eq!(hotspot.total_ms(), 5.0);
    }

    #[test]
    fn test_optimization_suggestion_all_hints() {
        // Cover all OptimizationSuggestion::hint() branches
        let simd = OptimizationSuggestion::UseSIMD {
            expected_speedup: 4.0,
        };
        assert_eq!(simd.hint(), "⚠ SIMD");

        let pool = OptimizationSuggestion::UsePool {
            current_allocs: 100,
        };
        assert_eq!(pool.hint(), "⚠ Pool");

        let batch = OptimizationSuggestion::BatchOperations { current_calls: 50 };
        assert_eq!(batch.hint(), "⚠ Batch");

        let async_io = OptimizationSuggestion::AsyncIO { blocking_us: 1000 };
        assert_eq!(async_io.hint(), "⚠ Async");
    }

    #[test]
    fn test_trace_analysis_wasm_events() {
        let mut analysis = TraceAnalysis::new("wasm-test");

        let event = WasmEvent {
            event_type: WasmEventType::Compile,
            duration_us: 1000,
            memory_impact: 1024,
            source_location: Some(SourceLocation::new(PathBuf::from("src/wasm.rs"), 1)),
        };
        analysis.add_wasm_event(event);

        assert_eq!(analysis.wasm_events.len(), 1);
    }

    #[test]
    fn test_trace_analysis_hotspots() {
        let mut analysis = TraceAnalysis::new("hotspot-test");

        let hotspot = SourceHotspot::new(PathBuf::from("src/slow.rs"), 100, "slow_function");
        analysis.add_hotspot(hotspot);

        assert_eq!(analysis.source_hotspots.len(), 1);
    }

    #[test]
    fn test_trace_analysis_syscall_percentages() {
        let mut analysis = TraceAnalysis::new("syscall-test");
        analysis.record_syscall("read", 700);
        analysis.record_syscall("write", 300);
        analysis.calculate_syscall_percentages();

        let read_stats = analysis.syscall_breakdown.get("read").unwrap();
        let write_stats = analysis.syscall_breakdown.get("write").unwrap();

        assert_eq!(read_stats.percent, 70.0);
        assert_eq!(write_stats.percent, 30.0);
    }

    #[test]
    fn test_trace_analysis_empty_syscall_percentages() {
        let mut analysis = TraceAnalysis::new("empty-syscall");
        analysis.calculate_syscall_percentages();
        // Should not panic with empty syscalls
        assert!(analysis.syscall_breakdown.is_empty());
    }

    #[test]
    fn test_flamegraph_default() {
        let fg = Flamegraph::default();
        assert!(fg.roots.is_empty());
        assert_eq!(fg.total_us, 0);
    }

    #[test]
    fn test_flamegraph_to_folded_with_children() {
        let mut fg = Flamegraph::new();

        let mut root = FlamegraphNode::new("root");
        root.add_time(100);

        let mut child1 = FlamegraphNode::new("child1");
        child1.add_time(50);

        let mut grandchild = FlamegraphNode::new("grandchild");
        grandchild.add_time(25);
        child1.add_child(grandchild);

        root.add_child(child1);
        fg.add_root(root);

        let folded = fg.to_folded();
        assert!(folded.contains("root 100"));
        assert!(folded.contains("root;child1 50"));
        assert!(folded.contains("root;child1;grandchild 25"));
    }

    #[test]
    fn test_render_trace_report_full() {
        let mut analysis = TraceAnalysis::new("full-test");

        // Add spans
        analysis.add_span(TraceSpan::new(
            "dns_lookup",
            TraceCategory::Network,
            0,
            2000,
        ));
        analysis.add_span(TraceSpan::new("compile", TraceCategory::Wasm, 2000, 5000));

        // Add syscalls
        analysis.record_syscall("read", 500);
        analysis.record_syscall("write", 300);
        analysis.calculate_syscall_percentages();

        // Add WASM events
        let wasm_event = WasmEvent {
            event_type: WasmEventType::Instantiate,
            duration_us: 3000,
            memory_impact: 2048,
            source_location: None,
        };
        analysis.add_wasm_event(wasm_event);

        // Add negative memory impact WASM event
        let wasm_event_negative = WasmEvent {
            event_type: WasmEventType::MemoryGrow,
            duration_us: 1000,
            memory_impact: -1024,
            source_location: Some(SourceLocation::new(PathBuf::from("src/mem.rs"), 42)),
        };
        analysis.add_wasm_event(wasm_event_negative);

        // Add hotspots
        let mut hotspot = SourceHotspot::new(PathBuf::from("src/hot.rs"), 100, "hot_func");
        hotspot.total_us = 4000;
        hotspot.call_count = 1000;
        hotspot.suggestion = Some(OptimizationSuggestion::UseSIMD {
            expected_speedup: 2.0,
        });
        analysis.add_hotspot(hotspot);

        // Calculate critical path
        analysis.calculate_critical_path();

        let report = render_trace_report(&analysis);

        // Verify all sections present
        assert!(report.contains("DEEP TRACE ANALYSIS"));
        assert!(report.contains("TIMELINE"));
        assert!(report.contains("SYSCALL BREAKDOWN"));
        assert!(report.contains("WASM-SPECIFIC EVENTS"));
        assert!(report.contains("SOURCE CORRELATION"));
        assert!(report.contains("Critical Path"));
    }

    #[test]
    fn test_truncate_helper() {
        assert_eq!(truncate("short", 10), "short");
        assert_eq!(truncate("verylongstring", 5), "very…");
    }
}
