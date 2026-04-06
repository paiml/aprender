//! Probar CLI Library (Feature 5)
//!
//! Command-line interface for the Probar testing framework.
//!
//! ## EXTREME TDD: Tests written FIRST per spec

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![warn(clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::format_push_string)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::float_cmp)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::if_not_else)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::option_if_let_else)]
#![allow(missing_docs)]
#![allow(clippy::fn_params_excessive_bools)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::large_enum_variant)]
#![allow(clippy::only_used_in_recursion)]
#![allow(irrefutable_let_patterns)]
#![allow(clippy::assigning_clones)]
#![allow(clippy::incompatible_msrv)]
#![allow(clippy::single_match_else)]

mod commands;
mod config;
pub mod debug;
pub mod dev_server;
mod error;
pub mod generate;
pub mod handlers;
pub mod lint;
pub mod load_testing;
mod output;
mod runner;
pub mod score;
pub mod simulation;
pub mod statistics;
pub mod stress;
pub mod tracing;
pub mod tree;
pub mod visualization;
pub mod wasm_testing;

pub use commands::{
    AnimationArgs, AnimationCheckArgs, AnimationSubcommand, AudioArgs, AudioCheckArgs,
    AudioSubcommand, AvSyncArgs, AvSyncCheckArgs, AvSyncOutputFormat, AvSyncReportArgs,
    AvSyncSubcommand, BuildArgs, Cli, Commands, ComplyArgs, ComplyCheckArgs, ComplyDiffArgs,
    ComplyEnforceArgs, ComplyMigrateArgs, ComplyOutputFormat, ComplyReportArgs, ComplyReportFormat,
    ComplySubcommand, ConfigArgs, CoverageArgs, DataAuditArgs, DiagramFormat, ExperimentArgs,
    ExperimentCompareArgs, ExperimentInitArgs, ExperimentStatusArgs, ExperimentSubcommand,
    InitArgs, LlmArgs, LlmBenchArgs, LlmGenDatasetArgs, LlmLoadArgs, LlmReportArgs, LlmScoreArgs,
    LlmSubcommand, LlmSweepArgs, LlmTestArgs, OutputFormat, PaletteArg, PlaybookArgs,
    PlaybookOutputFormat, RecordArgs, RecordFormat, ReportArgs, ReportFormat, ScoreArgs,
    ScoreOutputFormat, ServeArgs, ServeSubcommand, StressArgs, TestArgs, TreeArgs, VideoArgs,
    VideoCheckArgs, VideoSubcommand, VizArgs, WasmTarget, WatchArgs,
};
pub use config::{CliConfig, ColorChoice, Verbosity};
pub use debug::{create_tracer, DebugCategory, DebugTracer, DebugVerbosity, ResolutionRule};
pub use dev_server::{
    get_mime_type, DevServer, DevServerConfig, DevServerConfigBuilder, FileChangeEvent,
    FileWatcher, FileWatcherBuilder, HotReloadMessage, ImportRef, ImportType,
    ImportValidationError, ModuleValidationResult, ModuleValidator,
};
pub use error::{CliError, CliResult};
pub use lint::{
    render_lint_json, render_lint_report, ContentLinter, LintReport, LintResult, LintSeverity,
};
pub use load_testing::{
    render_load_test_json, render_load_test_report, AssertionResult as LoadAssertionResult,
    EndpointStats, HttpMethod, LatencyHistogram, LoadTestAssertion, LoadTestConfig, LoadTestError,
    LoadTestErrorKind, LoadTestOutputFormat, LoadTestRequest, LoadTestResult, LoadTestScenario,
    LoadTestStage, ResourceUsage, UserConfig,
};
pub use output::{OutputFormat as CliOutputFormat, ProgressReporter};
pub use runner::TestRunner;
pub use score::{
    CategoryScore, CategoryStatus, CriterionResult, Effort, Grade, ProjectScore, Recommendation,
    ScoreCalculator,
};
pub use tree::{build_tree, display_tree, render_tree, FileNode, TreeConfig};
pub use wasm_testing::{
    compare_performance, render_performance_report, Browser, BrowserMatrix, BrowserTestResult,
    ComparisonStatus, KeyModifiers, MemoryGrowthEvent, MemoryProfile, MemorySnapshot,
    PerformanceBaseline, PerformanceComparison, PerformanceMetric, RecordedEvent, Recording,
    RecordingMetadata, Viewport,
};
// PROBAR-SPEC-006 Section H: Enhanced Visualization
pub use visualization::{
    render_comparison, render_dashboard, ComparisonVerdict, DashboardState, DataPoint,
    EndpointMetrics, ExportFormat, MetricsStream, ReportComparison, ReportViewerConfig, StageInfo,
    StreamingHistogram, TimeSeries,
};
// PROBAR-SPEC-006 Section I: Statistical Analysis
pub use statistics::{
    render_statistical_json, render_statistical_report, ApdexCalculator, ApdexRating, KneeDetector,
    LatencySample, QuantileRegression, StatisticalAnalysis, TailAttribution, VarianceComponent,
    VarianceTree,
};
// PROBAR-SPEC-006 Section J: Deep Tracing
pub use tracing::{
    render_trace_json, render_trace_report, Flamegraph, FlamegraphNode, OptimizationSuggestion,
    SourceHotspot, SourceLocation, SyscallStats, TraceAnalysis, TraceCategory, TraceConfig,
    TraceSpan, WasmEvent, WasmEventType,
};
// PROBAR-SPEC-006 Section K: Simulation Playback
pub use simulation::{
    render_chaos_report, render_monte_carlo_json, render_monte_carlo_report, ChaosObservation,
    ChaosResult, Distribution, FailureInjection, ImpactLevel, InjectionType, LatencyDistribution,
    MonteCarloResult, ObservationSeverity, ParameterVariation, RiskLevel, SensitivityFactor,
    SimulationConfig, SimulationMode, SimulationOutput, SlaProbability,
};
// PROBAR-SPEC-WASM-001 Section H: Browser/WASM Stress Testing
pub use stress::{
    render_stress_json, render_stress_report, LatencyStats, MemoryStats, StressConfig, StressError,
    StressErrorKind, StressMode, StressResult, StressRunner,
};
