//! QA Scenario generation
//!
//! Defines test scenarios for model qualification using property-based testing.

use crate::models::ModelId;
use crate::oracle::{select_oracle, OracleResult};
use serde::{Deserialize, Serialize};

/// Modality: inference or transformation operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Modality {
    /// Direct inference via `apr run`
    Run,
    /// Interactive chat via `apr chat`
    Chat,
    /// HTTP server via `apr serve`
    Serve,
    /// Model quantization via `apr quantize`
    Quantize,
    /// Format import via `apr import`
    Import,
    /// Weight pruning via `apr prune`
    Prune,
    /// Knowledge distillation via `apr distill`
    Distill,
}

/// Core methods for the Modality enum
impl Modality {
    /// Get all inference modalities (Run, Chat, Serve).
    ///
    /// These are the three modalities that execute inference. For all seven
    /// modalities including transformations, use `all_with_transformations()`.
    #[must_use]
    pub const fn inference_modalities() -> [Self; 3] {
        [Self::Run, Self::Chat, Self::Serve]
    }

    /// Get all modalities including transformations
    #[must_use]
    pub const fn all_with_transformations() -> [Self; 7] {
        [
            Self::Run,
            Self::Chat,
            Self::Serve,
            Self::Quantize,
            Self::Import,
            Self::Prune,
            Self::Distill,
        ]
    }

    /// Get the apr command for this modality
    #[must_use]
    pub const fn command(&self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Chat => "chat",
            Self::Serve => "serve",
            Self::Quantize => "quantize",
            Self::Import => "import",
            Self::Prune => "prune",
            Self::Distill => "distill",
        }
    }

    /// Whether this modality is a transformation (not inference)
    #[must_use]
    pub const fn is_transformation(&self) -> bool {
        matches!(
            self,
            Self::Quantize | Self::Import | Self::Prune | Self::Distill
        )
    }
}

/// Display implementation for Modality
impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.command())
    }
}

/// Compute backend
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Backend {
    /// CPU with SIMD acceleration
    Cpu,
    /// GPU with CUDA acceleration
    Gpu,
}

/// Core methods for the Backend enum
impl Backend {
    /// Get all backends
    #[must_use]
    pub const fn all() -> [Self; 2] {
        [Self::Cpu, Self::Gpu]
    }

    /// Get the CLI flag for this backend
    #[must_use]
    pub const fn flag(&self) -> &'static str {
        match self {
            Self::Cpu => "",
            Self::Gpu => "--gpu",
        }
    }
}

/// Display implementation for Backend
impl std::fmt::Display for Backend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "cpu"),
            Self::Gpu => write!(f, "gpu"),
        }
    }
}

/// Model format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Format {
    /// GGUF quantized format
    Gguf,
    /// `HuggingFace` `SafeTensors` format
    SafeTensors,
    /// Native APR format
    Apr,
}

/// Core methods for the Format enum
impl Format {
    /// Get all formats
    #[must_use]
    pub const fn all() -> [Self; 3] {
        [Self::Gguf, Self::SafeTensors, Self::Apr]
    }

    /// Get the file extension for this format
    #[must_use]
    pub const fn extension(&self) -> &'static str {
        match self {
            Self::Gguf => ".gguf",
            Self::SafeTensors => ".safetensors",
            Self::Apr => ".apr",
        }
    }

    /// Get the class (A=quantized, B=full precision)
    #[must_use]
    pub const fn class(&self) -> char {
        match self {
            Self::Gguf | Self::Apr => 'A', // Quantized
            Self::SafeTensors => 'B',      // Full precision
        }
    }
}

/// Display implementation for Format
impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf => write!(f, "gguf"),
            Self::SafeTensors => write!(f, "safetensors"),
            Self::Apr => write!(f, "apr"),
        }
    }
}

/// APR tool/subcommand to test
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AprTool {
    /// `apr run` - Direct inference
    Run,
    /// `apr chat` - Interactive chat
    Chat,
    /// `apr serve` - HTTP server
    Serve,
    /// `apr inspect` - Model inspection
    Inspect,
    /// `apr validate` - Model validation
    Validate,
    /// `apr bench` - Benchmarking
    Bench,
    /// `apr profile` - Profiling
    Profile,
    /// `apr trace` - Tracing
    Trace,
    /// `apr check` - Self-test
    Check,
    /// `apr canary` - Canary tests
    Canary,
}

/// Core methods for the AprTool enum
impl AprTool {
    /// Get all tools
    #[must_use]
    pub const fn all() -> [Self; 10] {
        [
            Self::Run,
            Self::Chat,
            Self::Serve,
            Self::Inspect,
            Self::Validate,
            Self::Bench,
            Self::Profile,
            Self::Trace,
            Self::Check,
            Self::Canary,
        ]
    }

    /// Get the CLI subcommand
    #[must_use]
    pub const fn command(&self) -> &'static str {
        match self {
            Self::Run => "run",
            Self::Chat => "chat",
            Self::Serve => "serve",
            Self::Inspect => "inspect",
            Self::Validate => "validate",
            Self::Bench => "bench",
            Self::Profile => "profile",
            Self::Trace => "trace",
            Self::Check => "check",
            Self::Canary => "canary",
        }
    }

    /// Check if this tool requires a prompt
    #[must_use]
    pub const fn requires_prompt(&self) -> bool {
        matches!(self, Self::Run | Self::Chat)
    }

    /// Check if this tool supports trace levels
    #[must_use]
    pub const fn supports_trace(&self) -> bool {
        matches!(self, Self::Run | Self::Trace)
    }
}

/// Display implementation for AprTool
impl std::fmt::Display for AprTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.command())
    }
}

/// Trace level for debugging
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TraceLevel {
    /// No tracing
    None,
    /// Basic timing and token counts
    Basic,
    /// Per-layer statistics
    Layer,
    /// Full tensor values
    Payload,
}

/// Core methods for the TraceLevel enum
impl TraceLevel {
    /// Get all trace levels
    #[must_use]
    pub const fn all() -> [Self; 4] {
        [Self::None, Self::Basic, Self::Layer, Self::Payload]
    }

    /// Get the CLI value for this trace level
    #[must_use]
    pub const fn value(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Basic => "basic",
            Self::Layer => "layer",
            Self::Payload => "payload",
        }
    }
}

/// A single QA test scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QaScenario {
    /// Unique scenario ID
    pub id: String,
    /// Model to test
    pub model: ModelId,
    /// Inference modality
    pub modality: Modality,
    /// Compute backend
    pub backend: Backend,
    /// Model format
    pub format: Format,
    /// Test prompt
    pub prompt: String,
    /// Sampling temperature
    pub temperature: f32,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Random seed for reproducibility
    pub seed: u64,
    /// Trace level
    pub trace_level: TraceLevel,
    /// Expected oracle type
    pub oracle_type: String,
}

/// Construction and evaluation methods for QaScenario
impl QaScenario {
    /// Create a new scenario
    #[must_use]
    pub fn new(
        model: ModelId,
        modality: Modality,
        backend: Backend,
        format: Format,
        prompt: String,
        seed: u64,
    ) -> Self {
        let oracle = select_oracle(&prompt);
        Self {
            id: format!(
                "{}_{}_{}_{}_{:016x}",
                model.name, modality, backend, format, seed
            ),
            model,
            modality,
            backend,
            format,
            prompt,
            temperature: 0.0, // Deterministic by default
            max_tokens: 32,
            seed,
            trace_level: TraceLevel::None,
            oracle_type: oracle.name().to_string(),
        }
    }

    /// Set temperature
    #[must_use]
    pub const fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// Set max tokens
    #[must_use]
    pub const fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }

    /// Set trace level
    #[must_use]
    pub const fn with_trace_level(mut self, level: TraceLevel) -> Self {
        self.trace_level = level;
        self
    }

    /// Generate the apr CLI command for this scenario
    #[must_use]
    pub fn to_command(&self, model_path: &str) -> String {
        let backend_flag = self.backend.flag();
        let trace_flag = if self.trace_level == TraceLevel::None {
            String::new()
        } else {
            format!("--trace --trace-level {}", self.trace_level.value())
        };

        match self.modality {
            Modality::Run => {
                format!(
                    "apr run {model_path} '{}' -n {} --seed {} --temperature {} {backend_flag} {trace_flag}",
                    escape_prompt(&self.prompt),
                    self.max_tokens,
                    self.seed,
                    self.temperature
                )
                .trim()
                .to_string()
            }
            Modality::Chat => {
                format!(
                    "echo '{}' | apr chat {model_path} --temperature {} {backend_flag} {trace_flag}",
                    escape_prompt(&self.prompt),
                    self.temperature
                )
                .trim()
                .to_string()
            }
            Modality::Serve => {
                format!(
                    r#"apr serve {model_path} --port ${{PORT}} {backend_flag} &
sleep 2
curl -s http://localhost:${{PORT}}/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{{"prompt": "{}", "max_tokens": {}, "temperature": {}}}'
kill %1"#,
                    escape_json(&self.prompt),
                    self.max_tokens,
                    self.temperature
                )
            }
            Modality::Quantize => {
                format!("apr quantize {model_path} --scheme q4_k_m --json")
            }
            Modality::Import => {
                format!("apr import {model_path} --json")
            }
            Modality::Prune => {
                format!("apr prune {model_path} --method magnitude --target-ratio 0.5 --json")
            }
            Modality::Distill => {
                format!("apr distill {model_path} --json")
            }
        }
    }

    /// Evaluate the output using the appropriate oracle
    #[must_use]
    pub fn evaluate(&self, output: &str) -> OracleResult {
        let oracle = select_oracle(&self.prompt);
        oracle.evaluate(&self.prompt, output)
    }

    /// Get the MQS category this scenario contributes to
    #[must_use]
    pub const fn mqs_category(&self) -> &'static str {
        match self.modality {
            Modality::Run => match self.backend {
                Backend::Cpu => "A1",
                Backend::Gpu => "A2",
            },
            Modality::Chat => match self.backend {
                Backend::Cpu => "A3",
                Backend::Gpu => "A4",
            },
            Modality::Serve => match self.backend {
                Backend::Cpu => "A5",
                Backend::Gpu => "A6",
            },
            Modality::Quantize => "T1",
            Modality::Import => "T2",
            Modality::Prune => "T3",
            Modality::Distill => "T4",
        }
    }
}

/// Escape a prompt for shell usage
fn escape_prompt(prompt: &str) -> String {
    prompt.replace('\'', "'\\''")
}

/// Escape a prompt for JSON usage (RFC 8259 §7: control chars must be escaped)
fn escape_json(prompt: &str) -> String {
    let mut result = String::with_capacity(prompt.len());
    for c in prompt.chars() {
        match c {
            '\\' => result.push_str("\\\\"),
            '"' => result.push_str("\\\""),
            '\n' => result.push_str("\\n"),
            '\r' => result.push_str("\\r"),
            '\t' => result.push_str("\\t"),
            c if c < '\u{20}' => {
                // Escape other control characters as \u00XX
                use std::fmt::Write;
                let _ = write!(result, "\\u{:04x}", c as u32);
            }
            _ => result.push(c),
        }
    }
    result
}

/// Scenario generator for property-based testing
#[derive(Debug, Clone)]
pub struct ScenarioGenerator {
    /// Model to generate scenarios for
    pub model: ModelId,
    /// Number of scenarios per modality/backend/format combination
    pub scenarios_per_combination: usize,
    /// Prompts to use
    pub prompts: Vec<String>,
}

include!("scenarios.rs");

#[cfg(test)]
#[allow(clippy::expect_used, clippy::redundant_closure_for_method_calls)]
#[path = "scenario_tests.rs"]
mod scenario_tests;
