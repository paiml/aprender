//! cbtop - Compute Block Top
//!
//! Real-time load testing and hardware monitoring TUI built on the Brick Architecture.
//!
// Allow new_without_default - explicit new() is clearer for these types
#![cfg_attr(test, allow(clippy::disallowed_methods, clippy::float_cmp))]
#![allow(clippy::new_without_default)]
// Allow derivable_impls - explicit Default is clearer
#![allow(clippy::derivable_impls)]
// Allow missing_panics_doc - not critical for internal methods
#![allow(clippy::missing_panics_doc)]
// Allow missing_errors_doc - errors are self-explanatory
#![allow(clippy::missing_errors_doc)]
// Allow unnecessary_map_or - map_or is clearer
#![allow(clippy::unnecessary_map_or)]
// Allow collapsible_if - clarity over conciseness
#![allow(clippy::collapsible_if)]
// Allow needless_range_loop - clearer in some cases
#![allow(clippy::needless_range_loop)]
// Allow cast_precision_loss - acceptable for display values
#![allow(clippy::cast_precision_loss)]
// Allow cast_possible_truncation - handled appropriately
#![allow(clippy::cast_possible_truncation)]
// Allow dead_code - development in progress
#![allow(dead_code)]
// Allow field_reassign_with_default - clearer initialization
#![allow(clippy::field_reassign_with_default)]
// Allow manual_flatten - clearer error handling
#![allow(clippy::manual_flatten)]
//!
//! # Design Philosophy
//!
//! - **Test-as-Interface**: Every component is a falsifiable Brick (PROBAR-SPEC-009)
//! - **presentar-terminal**: All widgets and canvas from presentar-terminal (no custom reimplementation)
//! - **Toyota Way**: Jidoka, Poka-Yoke, Genchi Genbutsu principles throughout
//!
//! # Architecture
//!
//! ```text
//! Layer 4: Load Generators  → SimdLoadBrick, CudaLoadBrick, WgpuLoadBrick
//! Layer 3: Panels           → Overview, CPU, GPU, PCIe, Memory, Thermal
//! Layer 2: Analyzers        → Throughput, Bottleneck, Thermal
//! Layer 1: Collectors       → CPU, GPU, PCIe, Memory, Thermal
//! ```
//!
//! # Widget Source Policy
//!
//! All widgets and canvas implementations come from `presentar-terminal`.
//! cbtop does NOT implement its own widgets. If a widget is missing, it MUST
//! be added to presentar-terminal FIRST, then used here.

pub mod adaptive_ml;
pub mod adaptive_threshold;
pub mod adversarial;
pub mod alerting;
pub mod anomaly_detection;
pub mod app;
pub mod backend_regression;
pub mod baseline;
pub mod brick;
pub mod bricks;
pub mod cache_analysis;

/// Command surface for `apr compute` (was the standalone `cbtop` binary).
pub mod cli;

pub mod config;
pub mod context_regression;
pub mod continuous_batcher;
pub mod correlation_analysis;
pub mod cost_tracker;
pub mod double_blind;
pub mod error;
pub mod event_streaming;
pub mod export_reporting;
pub mod federated_metrics;
pub mod frequency_control;
pub mod fuzz;
pub mod golden_trace;
pub mod grammar;
pub mod headless;
pub mod incremental_snapshot;
pub mod ironman;
pub mod latency_distribution;
pub mod observability_backend;
pub mod optimize;
pub mod paged_kv;
pub mod performance_prediction;
pub mod predictive_scheduler;
pub mod profile_compare;
pub mod profile_persistence;
pub mod prometheus;
pub mod quantize;
pub mod regression_pipeline;
pub mod remote_agent;
pub mod ring_buffer;
pub mod roofline;
pub mod statistics;
pub mod thermal_prediction;
pub mod tracing_escalation;
pub mod variance_analysis;
pub mod workload_characterization;

// Core brick traits (cbtop-specific)
pub use brick::{Brick, BrickAssertion, BrickBudget, BrickVerification};
// ComputeBrick Scoring Framework (§29)
pub use brick::{BrickGrade, BrickScore, Scorable};
// CORRECTNESS-011: Per-kernel divergence detection
pub use brick::{fnv1a_f32, BrickProfiler, DivergenceReport, KernelTrace};

// Application
pub use app::CbtopApp;
pub use config::Config;
pub use error::CbtopError;

// Headless benchmarking API (HL-007)
pub use headless::{
    Benchmark, BenchmarkBuilder, BenchmarkConfig, BenchmarkResult, BenchmarkResults,
    ComparisonResult, LatencyStats, OutputFormat, RegressionResult, ScoreInfo, SystemInfo,
};
// Re-export config types for benchmark builder
pub use config::{ComputeBackend, WorkloadType};

// Optimization tooling (OPT-001 to OPT-004)
pub use optimize::{
    AnalysisSummary, BaselineEntry, BaselineReport, BottleneckAnalysis, BottleneckEntry,
    BottleneckSeverity, CpuCapabilities, OptimizationSuite, OptimizationValidator,
    RegressionDetector, RegressionEntry, RegressionReport, ValidationResult, WorkloadConfig,
};

// Industry Baseline Validation (PMAT-016)
pub use baseline::{
    BaselineComparison, BaselineValidator, GpuClass, ServerBaseline, SingleComparison, SmHealth,
    ThroughputGrade, ValidationSummary, INDUSTRY_BASELINES, TGI_BASELINE, TRITON_BASELINE,
    VLLM_BASELINE,
};

// Quantized Weight Support (PMAT-013)
pub use quantize::{
    ggml_type_to_format, DequantStrategy, GgufError, GgufHeader, GgufLoader, GgufResult,
    GgufTensorInfo, GgufValue, LayerQuantStats, QuantFormat, QuantStats, QuantizedBrick,
    QuantizedWeights,
};

// Paged KV Cache (PMAT-014)
pub use paged_kv::{
    BlockId, CacheStats, EvictionStrategy, KvBlock, PagedKvCache, PagedKvError, PagedKvResult,
    SeqId, SequenceInfo,
};

// Continuous Batcher (PMAT-015)
pub use continuous_batcher::{
    BatchSchedule, BatcherStats, ContinuousBatcher, ExponentialMovingAverage, InferenceRequest,
    Priority, SchedulingPolicy, SequenceGroup, SpeculativeDecoder, SpeculativeOutput, Token,
    TokenOutput,
};

// Ironman Falsification Suite (PMAT-017)
pub use ironman::{
    full_validate, quick_validate, GateCategory, GateResult, IronmanScorecard, IronmanValidator,
    QualityGate, IRONMAN_GATES,
};

// Grammar of ComputeBlock (PMAT-018)
pub use grammar::{
    BuiltComputeBlock, ByteSize, CompositionMode, ComputeBlock, ComputeBlockBuilder, CpuAffinity,
    DataTransform, DataType, Dimensions, ExecutionContext, ExecutionPolicy, ExecutionResult,
    ExecutionStrategy, GpuDevice, GrammarError, GrammarResult, KernelSpec, LinearResourceScale,
    LogResourceScale, ObservabilityConfig, Operation, QosLevel, QuantScheme, ResourceLimits,
    ResourceMapping, ResourceScale, RetryPolicy, ScaleBinding, SimdWidth, StrategyLayer,
    TensorSpec, WorkloadSpec,
};

// Adversarial Falsification Testing (PMAT-019)
pub use adversarial::{
    AdversarialError, AdversarialResult, AdversarialTactic, AdversarialTestSummary,
    BitFlipInjector, CancellationToken, CheckedArithmetic, ConfigValidator, InputValidator,
    MonotonicClock, RecoveryHandler, ResourceLimiter, ResourceUsage,
};

// Double-Blind Verification Framework (PMAT-020)
pub use double_blind::{
    AuditEntry, BlackBoxArtifact, FalsificationClaim, FalsificationCriterion, ReleaseDecision,
    Role, ScorecardComponent, ScorecardV2, SessionState, VerificationAttempt, VerificationReport,
    VerificationResult, VerificationSession,
};

// Tracing Escalation Framework (PMAT-021)
pub use tracing_escalation::{
    EscalationReason, EscalationThresholds, OtlpSpanAttributes, SyscallBreakdown, TraceResult,
    TracingEscalation,
};

// Roofline Model Analyzer (PMAT-022)
pub use roofline::{
    BatchRooflineAnalysis, BatchSummary, BottleneckType, HardwareProfile, RooflineAnalysis,
    RooflinePlot, RooflinePlotPoint, WorkloadMetrics,
};

// Fuzz Testing Integration (PMAT-023)
pub use fuzz::{
    bound_value, checked_add_u64, checked_mul_u64, safe_div, sanitize_float, test_float_edge_cases,
    test_u64_edge_cases, FuzzFailure, FuzzInputValidator, FuzzResult, FuzzSuite, FuzzSummary,
    FuzzTargetConfig, FuzzValidationError,
};

// Statistical Analysis (PMAT-024)
pub use statistics::{
    bootstrap_ci, percentile, trimmed_mean, ComparisonResult as StatisticalComparison,
    EffectCategory, EffectSize, MannWhitneyResult, OutlierFilter, StatisticalAnalysis,
};

// Cache Efficiency Analysis (PMAT-025)
pub use cache_analysis::{
    elementwise_working_set, matrix_working_set, optimal_matmul_tile, AccessPattern,
    BandwidthPrediction, CacheConfig, CacheLevel, WorkingSetAnalysis,
};

// Latency Distribution Analysis (PMAT-026)
pub use latency_distribution::{
    DistributionShape, HistogramBucket, LatencyDistribution, LatencyHistogram, TailSeverity,
};

// Variance Source Analysis (PMAT-027)
pub use variance_analysis::{VarianceAnalysis, VarianceInput, VarianceSource};

// Profile Persistence and Rotation (PMAT-028)
pub use profile_persistence::{
    BackendConfig as ProfileBackend, ProfileConfig, ProfileError, ProfileManager, ProfileOverlay,
    ProfileResult, WorkloadConfig as ProfileWorkload,
};

// Golden Trace Comparison (PMAT-029)
pub use golden_trace::{
    GoldenComparator, GoldenTrace, GoldenTraceError, GoldenTraceManager, GoldenTraceResult,
    SyscallBreakdown as TraceSyscallBreakdown, SyscallBreakdownDelta, TraceComparison,
    TraceMetrics,
};

// Thermal Trend Prediction (PMAT-030)
pub use thermal_prediction::{
    analyze_thermal, assess_throttle_risk, CooldownRecommendation, RiskCategory, ThermalAnalyzer,
    ThermalCorrelation, ThermalPrediction, ThermalSample, ThermalVariance, ThrottleRisk,
    DEFAULT_THROTTLE_THRESHOLD_C, MIN_SAMPLES_FOR_ANALYSIS,
};

// Cross-Backend Regression Detector (PMAT-031)
pub use backend_regression::{
    Backend, BackendComparison, BackendMeasurement, BackendRecommendation,
    BackendRegressionDetector, BackendSummary, SizeCliff, TransferAnalysis,
    WorkloadType as BackendWorkload,
};

// Multi-Metric Correlation Analysis (PMAT-032)
pub use correlation_analysis::{
    CorrelationAnalyzer, CorrelationResult, EventSample, EventType, InterferenceCategory,
    InterferenceResult, IsolationAction, IsolationRecommendation, PerformanceSample,
    SystemSnapshot,
};

// Performance Prediction Model (PMAT-033)
pub use performance_prediction::{
    DataPoint, FittedModel, ModelType, PerformancePredictor, Prediction, MIN_SAMPLES_FOR_FIT,
};

// Anomaly Detection Engine (PMAT-034)
pub use anomaly_detection::{
    Anomaly, AnomalyDetector, AnomalyReport, AnomalySeverity, AnomalyType, ChangePoint,
    DEFAULT_IQR_MULTIPLIER, DEFAULT_ZSCORE_THRESHOLD, MIN_SAMPLES_FOR_DETECTION,
};

// Workload Characterization System (PMAT-035)
pub use workload_characterization::{
    ClassificationResult, RecommendedBackend, WorkloadCategory, WorkloadCharacterizer,
    WorkloadFeatures,
};

// Multi-Format Export System (PMAT-036)
pub use export_reporting::{
    BenchmarkMetric, BenchmarkReport, ComparisonEntry, ComparisonReport, ExportFormat,
    ReportBuilder, ReportExporter, ReportType,
};

// Adaptive Threshold Learning System (PMAT-037)
pub use adaptive_threshold::{
    LearnedThreshold, ThresholdCheck, ThresholdDirection, ThresholdLearner,
    DEFAULT_CONFIDENCE_LEVEL, DEFAULT_OUTLIER_THRESHOLD, MIN_SAMPLES_FOR_LEARNING,
};

// CPU Frequency Control Backend (PMAT-038)
pub use frequency_control::{
    CpuFrequencyInfo, CpuGovernor, FrequencyController, FrequencyLock, FrequencyReading,
    FrequencyVariance,
};

// Context-Aware Regression Predictor (PMAT-039)
pub use context_regression::{
    BaselineEntry as ContextBaselineEntry, ContextRegressionPredictor, RegressionCheck,
    RegressionThreshold, SystemContext, Trend, DEFAULT_COLD_START_MARGIN, DEFAULT_STALENESS_SEC,
    MIN_SAMPLES_FOR_CONTEXT,
};

// Real-Time Alert Integration System (PMAT-040)
pub use alerting::{
    alert_from_anomaly, Alert, AlertChannel, AlertRouter, AlertRouterConfig, AlertSeverity,
    DeliveryResult, MessageTemplate,
};

// Prometheus Metrics Exporter (PMAT-041)
pub use prometheus::{
    validate_metric_name, CounterValue, GaugeValue, HistogramBuckets, HistogramValue, Labels,
    MetricDef, MetricType, MetricsRegistry, DEFAULT_BUCKETS, DEFAULT_MAX_LABELS,
};

// Cost and Energy Efficiency Tracker (PMAT-042)
pub use cost_tracker::{
    default_gpu_pricing, BudgetAlert, CloudProvider, CostComparison, CostResult, CostTracker,
    EnergyMeasurement, GpuPricing, DEFAULT_CARBON_INTENSITY, JOULES_PER_KWH,
};

// Structured Event Streaming (PMAT-043)
pub use event_streaming::{
    compress_data, event_from_sample, EventBatch, EventStreamer, MetricEvent, RetryConfig,
    SinkHealth, SinkType, DEFAULT_BATCH_SIZE, SCHEMA_VERSION,
};

// Remote SSH/Headless Agent Integration (PMAT-044)
pub use remote_agent::{
    AggregatedResult, AggregationStrategy, AuthMethod, CommandResult, HostBenchmark, HostConfig,
    HostHealth, HostState, RemoteAgent, RemoteAgentConfig, RemoteError, RemoteResult,
    DEFAULT_HEALTH_CHECK_INTERVAL_SEC, DEFAULT_MAX_CONCURRENT, DEFAULT_RETRY_DELAY_MS,
};

// Profile Diffing and A/B Comparison (PMAT-045)
pub use profile_compare::{
    BenchmarkProfile, ChangeDirection, CompareConfig, CompareError, CompareResult,
    ComparisonVerdict, EffectMagnitude, EffectSizeResult, MetricComparison, MetricSamples,
    ProfileComparator, ProfileComparison, WelchTestResult,
    DEFAULT_CONFIDENCE_LEVEL as COMPARE_DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_REGRESSION_THRESHOLD as COMPARE_DEFAULT_REGRESSION_THRESHOLD, MIN_COMPARISON_SAMPLES,
};

// Observability Backend Integrations (PMAT-046)
pub use observability_backend::{
    format_dogstatsd, BackendHealth, DatadogConfig, ExportMetric, ExportResult, HoneycombConfig,
    MetricExportType, NewRelicConfig, ObservabilityBackend,
    ObservabilityConfig as ObsBackendConfig, ObservabilityError, ObservabilityExporter,
    ObservabilityResult, OtlpConfig, WebhookConfig, DEFAULT_BATCH_SIZE as OBS_DEFAULT_BATCH_SIZE,
    DEFAULT_FLUSH_INTERVAL_MS,
};

// CI/CD Regression Detection Pipeline (PMAT-047)
pub use regression_pipeline::{
    BenchmarkMetric as PipelineBenchmarkMetric, BenchmarkResults as PipelineBenchmarkResults,
    GitRef, MetricRegression, PipelineConfig, PipelineError, PipelineResult, PipelineStatus,
    RegressionAnalysis, RegressionPipeline, StatusCheck,
    DEFAULT_REGRESSION_THRESHOLD as PIPELINE_REGRESSION_THRESHOLD, DEFAULT_TIMEOUT_SEC,
    DEFAULT_WARNING_THRESHOLD,
};

// Federated Metrics Aggregation (PMAT-048)
pub use federated_metrics::{
    AggregatedMetrics, FederatedHost, FederationConfig, GCounter, LwwRegister, MetricsFederation,
    OrSet,
};

// Dynamic Adaptive Thresholds with ML (PMAT-049)
pub use adaptive_ml::{
    AdaptiveThresholdMl, AnomalyResult as MlAnomalyResult,
    ClassificationMetrics as MlClassificationMetrics, LearnedWorkloadThreshold, MlThresholdConfig,
    MlThresholdError, TimeSeriesFeatures, WorkloadClass,
};

// Incremental Profile Snapshots (PMAT-050)
pub use incremental_snapshot::{
    DeltaMetric, DeltaSnapshot, IncrementalSnapshotStore, MetricData, ProfileSnapshot,
    RetentionTier, SnapshotConfig, SnapshotError, SnapshotIndex, SnapshotQuery,
};

// Predictive Scheduling Optimizer (PMAT-051)
pub use predictive_scheduler::{
    HostProfile, InstanceType, PredictiveScheduler, PredictiveSchedulerConfig, SchedulerMetrics,
    SchedulingDecision, WorkloadSpec as SchedulerWorkloadSpec,
};

// Re-export presentar-terminal widgets and canvas for convenience
// All widgets MUST come from presentar-terminal - DO NOT reimplement
pub use presentar_terminal::direct::{CellBuffer, DiffRenderer, DirectTerminalCanvas};
pub use presentar_terminal::{BrailleGraph, ColorMode, GraphMode, Meter, Table};

// Re-export presentar-core traits
pub use presentar_core::{Canvas, Color, Constraints, Point, Rect, Size, TextStyle};
