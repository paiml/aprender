//! LLM Testing: Correctness assertions and load testing for OpenAI-compatible APIs.
//!
//! This module provides tools for testing LLM inference endpoints:
//! - **Client types**: Typed request/response structs for OpenAI-compatible APIs (feature: `llm-types`)
//! - **Assertions**: Structural and semantic correctness checks on LLM outputs (feature: `llm-types`)
//! - **Client**: HTTP client for OpenAI-compatible chat completion APIs (feature: `llm`)
//! - **Load testing**: Concurrent request generation with latency/throughput metrics (feature: `llm`)
//! - **Reporting**: JSON and Markdown report generation with historical tracking (feature: `llm`)

pub mod assertion;
#[cfg(feature = "llm")]
pub mod benchmark;
pub mod client;
pub mod experiment;
#[cfg(feature = "llm")]
pub mod gpu_telemetry;
#[cfg(feature = "llm")]
pub mod loadtest;
pub mod prompts;
#[cfg(feature = "llm")]
pub mod report;
#[cfg(feature = "llm")]
#[allow(missing_docs)]
pub mod score;
pub mod training_scorecard;

pub use assertion::{LlmAssertion, LlmAssertionError, LlmAssertionResult};
pub use client::{
    BrickTrace, BrickTraceOp, ChatMessage, ChatRequest, ChatResponse, ChatResponseChoice, Role,
    StreamChunk, StreamedChatResponse, TimedChatResponse, Usage,
};
#[cfg(feature = "llm")]
pub use client::{LlmClient, LlmClientError};
pub use experiment::{
    BudgetConfig, DataAuditResult, EarlyStoppingConfig, Experiment, ExperimentRun,
    ExperimentStatus, KillCriterion, MetricSnapshot,
};
#[cfg(feature = "llm")]
pub use gpu_telemetry::{extract_host_from_url, GpuTelemetryCollector};
#[cfg(feature = "llm")]
pub use loadtest::{
    BrickTraceOpSummary, DatasetStats, DriftAnalysis, GpuTelemetry, JitterAnalysis, LatencySpike,
    LoadTest, LoadTestConfig, LoadTestResult, QualityFailure, QualityResult, RequestDetail,
    RequestRate, SweepLevel, SweepResult, TailAnalysis, TelemetryStat, ValidationMode,
};
pub use prompts::{load_from_file as load_prompts_from_file, load_profile, PromptProfile};
#[cfg(feature = "llm")]
pub use report::{to_json, to_markdown_row, to_markdown_table, update_performance_md};
#[cfg(feature = "llm")]
pub use score::{
    assign_grade, compute_cold_start_scorecard, compute_concurrency_scaling_scorecard,
    compute_correctness_scorecard, compute_layer_scorecard, compute_memory_scorecard,
    compute_metric_score, compute_output_length_scorecard, compute_power_efficiency_scorecard,
    compute_profile_scorecard, compute_scorecard, format_cold_start_markdown,
    format_cold_start_table, format_correctness_markdown, format_correctness_table,
    format_layer_markdown, format_layer_table, format_markdown, format_memory_markdown,
    format_memory_table, format_output_length_markdown, format_output_length_table,
    format_power_markdown, format_power_table, format_profile_markdown, format_profile_table,
    format_scaling_markdown, format_scaling_table, format_table, ColdStartScore,
    ColdStartScorecard, ConcurrencyScalingScore, ConcurrencyScalingScorecard, ConsistencyScore,
    CorrectnessScore, CorrectnessScorecard, LayerScore, LayerScorecard, MemoryScore,
    MemoryScorecard, MetricScore, MetricThreshold, OutputLengthCategory, OutputLengthEntry,
    OutputLengthScorecard, PowerEfficiencyScore, PowerEfficiencyScorecard, ProfileEntry,
    ProfileScorecard, PromptCategory, RuntimeScore, Scorecard, ScoringContract,
};
pub use training_scorecard::{
    compute_parity_comparison, compute_training_scorecard, format_parity_markdown,
    format_training_scorecard_markdown, Bottleneck, Grade, HardwareSpec, ParityComparison,
    ParityGap, ParityProfile, StepProfilerInput, TrainingScorecard,
};
