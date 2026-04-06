//! Inference Tracing for debugging LLM pipelines (AWS Step Functions Parity)
//!
//! Per spec: APR-TRACE-001 v3.0.0
//! Toyota Way: Genchi Genbutsu (Go and See) + Jidoka (Built-in Quality)
//!
//! This module models inference as a deterministic **State Machine**:
//! 1. TOKENIZE: Text -> Token IDs
//! 2. EMBED: Token IDs -> Vectors
//! 3. TRANSFORMER_BLOCK: Vectors -> Vectors (×N layers)
//! 4. LM_HEAD: Vectors -> Logits
//! 5. SAMPLE: Logits -> Token ID
//! 6. DECODE: Token ID -> Text
//!
//! Each state transition emits `TaskStateEntered` and `TaskStateExited` events
//! with verified Input/Output payloads (AWS Step Functions Execution History format).
//!
//! Example:
//! ```bash
//! apr run model.gguf --prompt "Hello" --trace
//! apr run model.gguf --prompt "Hi" --trace=tokenize,sample,decode
//! apr run model.gguf --prompt "Hi" --trace --trace-output trace.json
//! ```

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::Instant;

/// Trace configuration
#[derive(Debug, Clone, Default)]
pub struct TraceConfig {
    /// Whether tracing is enabled
    pub enabled: bool,
    /// Which steps to trace (empty = all)
    pub steps: HashSet<TraceStep>,
    /// Verbose output (show tensor values)
    pub verbose: bool,
    /// Output file path for JSON trace (None = stderr)
    pub output: Option<PathBuf>,
}

impl TraceConfig {
    /// Create a new trace config with tracing enabled
    #[must_use]
    pub fn enabled() -> Self {
        Self {
            enabled: true,
            ..Default::default()
        }
    }

    /// Check if a specific step should be traced
    #[must_use]
    pub fn should_trace(&self, step: TraceStep) -> bool {
        self.enabled && (self.steps.is_empty() || self.steps.contains(&step))
    }

    /// Parse trace steps from comma-separated string
    #[must_use]
    pub fn parse_steps(s: &str) -> HashSet<TraceStep> {
        s.split(',')
            .filter_map(|part| TraceStep::parse(part.trim()))
            .collect()
    }
}

/// Inference pipeline steps (State Machine states per AWS Step Functions model)
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum TraceStep {
    /// Tokenization (text -> token IDs)
    Tokenize,
    /// Token embedding lookup
    Embed,
    /// Layer normalization
    LayerNorm,
    /// Attention computation
    Attention,
    /// Feed-forward network
    FFN,
    /// Transformer block (combines attention + FFN)
    TransformerBlock,
    /// LM head projection (hidden -> logits)
    LmHead,
    /// Token sampling
    Sample,
    /// Token decoding (token ID -> text)
    Decode,
    /// GPU kernel launch (PTX-level tracing, GH-219)
    KernelLaunch,
    /// Compute brick profiling breakdown (trueno BrickProfiler)
    BrickProfile,
}

impl TraceStep {
    /// Parse step from string
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "tokenize" | "encode" => Some(Self::Tokenize),
            "embed" | "embedding" => Some(Self::Embed),
            "layernorm" | "ln" | "norm" => Some(Self::LayerNorm),
            "attention" | "attn" => Some(Self::Attention),
            "ffn" | "mlp" => Some(Self::FFN),
            "transformer" | "transformer_block" | "layer" => Some(Self::TransformerBlock),
            "lmhead" | "lm_head" | "head" => Some(Self::LmHead),
            "sample" | "sampling" => Some(Self::Sample),
            "decode" | "detokenize" => Some(Self::Decode),
            "kernel" | "kernel_launch" | "ptx" | "cuda" => Some(Self::KernelLaunch),
            "brick" | "brick_profile" | "profiler" | "bricks" => Some(Self::BrickProfile),
            _ => None,
        }
    }

    /// Get display name for step (AWS Step Functions state name)
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tokenize => "TOKENIZE",
            Self::Embed => "EMBED",
            Self::LayerNorm => "LAYER_NORM",
            Self::Attention => "ATTENTION",
            Self::FFN => "FFN",
            Self::TransformerBlock => "TRANSFORMER_BLOCK",
            Self::LmHead => "LM_HEAD",
            Self::Sample => "SAMPLE",
            Self::Decode => "DECODE",
            Self::KernelLaunch => "KERNEL_LAUNCH",
            Self::BrickProfile => "BRICK_PROFILE",
        }
    }

    /// Get legacy name for backwards compatibility (deprecated)
    #[deprecated(since = "3.0.0", note = "Use name() instead")]
    #[must_use]
    pub fn legacy_name(&self) -> &'static str {
        match self {
            Self::Tokenize => "ENCODE",
            Self::Embed => "EMBED",
            Self::LayerNorm => "LAYER_NORM",
            Self::Attention => "ATTENTION",
            Self::FFN => "FFN",
            Self::TransformerBlock => "TRANSFORMER",
            Self::LmHead => "LM_HEAD",
            Self::Sample => "SAMPLE",
            Self::Decode => "DECODE",
            Self::KernelLaunch => "KERNEL_LAUNCH",
            Self::BrickProfile => "BRICK_PROFILE",
        }
    }

    /// Get step number for 8-step pipeline (7 = kernel-level)
    #[must_use]
    pub fn step_number(&self) -> usize {
        match self {
            Self::Tokenize => 1,
            Self::Embed => 2,
            Self::LayerNorm | Self::Attention | Self::FFN | Self::TransformerBlock => 3,
            Self::LmHead => 4,
            Self::Sample => 5,
            Self::Decode => 6,
            Self::KernelLaunch => 7,
            Self::BrickProfile => 8,
        }
    }
}

/// Tensor statistics for tracing
#[derive(Debug, Clone, Default)]
pub struct TensorStats {
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Whether NaN values were detected
    pub has_nan: bool,
    /// Whether Inf values were detected
    pub has_inf: bool,
}

impl TensorStats {
    /// Compute stats from tensor data
    #[must_use]
    pub fn from_slice(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self::default();
        }

        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        let mut sum = 0.0f64;
        let mut has_nan = false;
        let mut has_inf = false;

        for &v in data {
            if v.is_nan() {
                has_nan = true;
            } else if v.is_infinite() {
                has_inf = true;
            } else {
                min = min.min(v);
                max = max.max(v);
                sum += f64::from(v);
            }
        }

        let mean = (sum / data.len() as f64) as f32;

        // Compute std dev
        let mut var_sum = 0.0f64;
        for &v in data {
            if !v.is_nan() && !v.is_infinite() {
                let diff = f64::from(v) - f64::from(mean);
                var_sum += diff * diff;
            }
        }
        let std = ((var_sum / data.len() as f64).sqrt()) as f32;

        Self {
            min,
            max,
            mean,
            std,
            has_nan,
            has_inf,
        }
    }

    /// Check if stats indicate an error (Jidoka)
    #[must_use]
    pub fn has_error(&self) -> bool {
        self.has_nan || self.has_inf
    }
}

/// Trace error types (Jidoka: stop-the-line errors)
#[derive(Debug, Clone)]
pub enum TraceError {
    /// Token ID exceeds vocabulary size
    VocabOverflow {
        /// The offending token ID
        token_id: u32,
        /// Size of the vocabulary
        vocab_size: usize,
    },
    /// NaN values detected in tensor
    NaNDetected {
        /// Layer index where NaN was detected (None if embedding)
        layer: Option<usize>,
    },
    /// Inf values detected in tensor
    InfDetected {
        /// Layer index where Inf was detected (None if embedding)
        layer: Option<usize>,
    },
    /// Garbage characters in decoded output (APR-TOK-001)
    GarbageOutput {
        /// Sample of garbage output
        sample: String,
    },
    /// Unknown token (OOV)
    UnknownToken {
        /// The unknown token ID
        token_id: u32,
    },
    /// Shape mismatch
    ShapeMismatch {
        /// Expected shape
        expected: Vec<usize>,
        /// Actual shape
        actual: Vec<usize>,
    },
    /// Execution failed (F-JID-01: Jidoka)
    ExecutionFailed {
        /// Cause of failure
        cause: String,
    },
}

impl std::fmt::Display for TraceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VocabOverflow {
                token_id,
                vocab_size,
            } => {
                write!(f, "Token ID {} exceeds vocab size {}", token_id, vocab_size)
            },
            Self::NaNDetected { layer } => {
                if let Some(l) = layer {
                    write!(f, "NaN values detected in layer {}", l)
                } else {
                    write!(f, "NaN values detected")
                }
            },
            Self::InfDetected { layer } => {
                if let Some(l) = layer {
                    write!(f, "Inf values detected in layer {}", l)
                } else {
                    write!(f, "Inf values detected")
                }
            },
            Self::GarbageOutput { sample } => {
                write!(f, "Garbage output detected: {:?}", sample)
            },
            Self::UnknownToken { token_id } => {
                write!(f, "Unknown token ID: {}", token_id)
            },
            Self::ShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Shape mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            },
            Self::ExecutionFailed { cause } => {
                write!(f, "Execution failed: {}", cause)
            },
        }
    }
}

/// AWS Step Functions event type (per spec v3.1.0)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AwsEventType {
    /// State machine entered a state
    TaskStateEntered,
    /// State machine exited a state
    TaskStateExited,
    /// Execution failed with error
    ExecutionFailed,
}

impl AwsEventType {
    /// Get the event type name (AWS Step Functions format)
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::TaskStateEntered => "TaskStateEntered",
            Self::TaskStateExited => "TaskStateExited",
            Self::ExecutionFailed => "ExecutionFailed",
        }
    }
}

/// Trace event emitted during inference (AWS Step Functions Parity)
#[derive(Debug, Clone)]
pub struct TraceEvent {
    /// Unique event ID (AWS Step Functions: monotonically increasing)
    pub id: u64,
    /// ISO 8601 timestamp
    pub timestamp: String,
    /// AWS Step Functions event type
    pub event_type: AwsEventType,
    /// Link to the entry event (for TaskStateExited)
    pub previous_event_id: Option<u64>,
    /// Pipeline step (state name)
    pub step: TraceStep,
    /// Generation iteration (0 for prefill)
    pub iteration: usize,
    /// Layer index (for transformer layers)
    pub layer: Option<usize>,
    /// Input shape
    pub input_shape: Vec<usize>,
    /// Output shape
    pub output_shape: Vec<usize>,
    /// Tensor statistics
    pub stats: TensorStats,
    /// Duration in microseconds
    pub duration_us: u64,
    /// Error if any (Jidoka)
    pub error: Option<TraceError>,
    /// Cause of failure (F-AWS-05: required for ExecutionFailed events)
    pub cause: Option<String>,
    /// Additional details (step-specific)
    pub details: TraceDetails,
}

/// Step-specific trace details
#[derive(Debug, Clone, Default)]
pub struct TraceDetails {
    /// Input text (for encode step)
    pub input_text: Option<String>,
    /// Output tokens (for encode step)
    pub output_tokens: Option<Vec<u32>>,
    /// Vocabulary entries (for encode step, OOV detection)
    pub vocab_entries: Option<Vec<String>>,
    /// Top-k logits with token IDs (for lm_head/sample step)
    pub top_k_logits: Option<Vec<(u32, f32)>>,
    /// Top-k probabilities with token IDs (for sample step)
    pub top_k_probs: Option<Vec<(u32, f32)>>,
    /// Sampled token ID (for sample/decode step)
    pub sampled_token: Option<u32>,
    /// Decoded text output (for decode step)
    pub decoded_text: Option<String>,
    /// Token string representation (for decode step)
    pub token_string: Option<String>,
    /// Temperature parameter used (for sample step)
    pub temperature: Option<f32>,
    /// Top-k parameter used (for sample step)
    pub top_k: Option<usize>,
    /// Kernel name (for kernel_launch step, GH-219)
    pub kernel_name: Option<String>,
    /// Grid dimensions [x, y, z] (for kernel_launch step)
    pub grid_dims: Option<[u32; 3]>,
    /// Block dimensions [x, y, z] (for kernel_launch step)
    pub block_dims: Option<[u32; 3]>,
    /// Shared memory bytes (for kernel_launch step)
    pub shared_mem_bytes: Option<u32>,
    /// Layer index within the transformer (for kernel_launch context)
    pub kernel_layer: Option<usize>,
    /// Dispatch strategy: "grid_y" or "register_unroll" (for kernel_launch step)
    pub dispatch_strategy: Option<String>,
    /// Brick category breakdown (for brick_profile step)
    /// Maps category name ("Norm", "Attention", "FFN", "Other") to nanoseconds
    pub brick_categories: Option<Vec<(String, u64)>>,
    /// Per-brick timing (for brick_profile step)
    /// Maps brick name ("RmsNorm", "QkvProjection", etc.) to (total_ns, count)
    pub brick_timings: Option<Vec<(String, u64, u64)>>,
}

/// Inference tracer
#[derive(Debug)]
pub struct InferenceTracer {
    /// Configuration
    config: TraceConfig,
    /// Collected events
    events: Vec<TraceEvent>,
    /// Model info
    model_info: ModelInfo,
    /// Current step timer
    step_start: Option<Instant>,
    /// Total errors count
    error_count: usize,
    /// Total warnings count
    warning_count: usize,
    /// Next event ID (monotonically increasing per AWS Step Functions)
    next_event_id: u64,
    /// ID of the last TaskStateEntered event (for linking TaskStateExited)
    last_entered_id: Option<u64>,
}

include!("model_info.rs");
include!("execution_failure.rs");
include!("mod_get_top_compute.rs");
include!("tracer_contracts.rs");
