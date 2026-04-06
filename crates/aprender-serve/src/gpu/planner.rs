//! GPU Batch Planner - Pure Decision Logic (Phase 47)
//!
//! This module implements the Plan/Execute separation pattern.
//! All logic here is pure Rust with no GPU dependencies - 100% testable under llvm-cov.
//!
//! ## The Pattern
//!
//! ```text
//! ┌─────────────────────┐     ┌─────────────────────┐
//! │  BatchPlanner       │ --> │  GenerationStep     │ --> Executor
//! │  (Pure Rust)        │     │  (Data struct)      │     (GPU calls)
//! │  100% testable      │     │  100% testable      │     untestable
//! └─────────────────────┘     └─────────────────────┘
//! ```
//!
//! ## Why This Matters
//!
//! Before: `generate_gpu()` mixed decisions with GPU calls - llvm-cov couldn't instrument it.
//! After: `planner.plan()` returns a pure data struct - 100% covered by unit tests.

use crate::gpu::GpuModelConfig;

/// A single step in the generation process.
///
/// This is pure data - no methods that call the GPU.
/// The Executor consumes these steps and performs the actual computation.
#[derive(Debug, Clone, PartialEq)]
pub enum GenerationStep {
    /// Process initial prompt through full forward pass
    ProcessPrompt {
        /// Token IDs to process
        tokens: Vec<usize>,
    },

    /// Generate next token incrementally (single-token forward)
    GenerateToken {
        /// All tokens so far (for KV cache position)
        tokens: Vec<usize>,
        /// Use optimized greedy path (fused LM head + argmax)
        use_greedy_optimization: bool,
    },

    /// Generation is complete
    Done {
        /// Final generated tokens
        tokens: Vec<usize>,
    },
}

/// Configuration for generation planning
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Vocabulary size (affects optimization path)
    pub vocab_size: usize,
    /// Threshold for switching to greedy optimization
    pub greedy_vocab_threshold: usize,
    /// Optional stop token ID
    pub stop_token: Option<usize>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 100,
            vocab_size: 32000,
            greedy_vocab_threshold: 8192,
            stop_token: None,
        }
    }
}

impl GenerationConfig {
    /// Create config from model config
    #[must_use]
    pub fn from_model(model_config: &GpuModelConfig, max_tokens: usize) -> Self {
        Self {
            max_tokens,
            vocab_size: model_config.vocab_size,
            greedy_vocab_threshold: 8192,
            stop_token: None,
        }
    }

    /// Should we use the greedy optimization path?
    #[must_use]
    pub fn use_greedy_path(&self) -> bool {
        self.vocab_size > self.greedy_vocab_threshold
    }
}

/// Planner state machine for token generation.
///
/// Pure Rust - no GPU dependencies. All decisions are made here,
/// then the Executor performs the actual computation.
#[derive(Debug, Clone)]
pub struct BatchPlanner {
    /// Generation configuration
    config: GenerationConfig,
    /// Current state
    state: PlannerState,
    /// Tokens generated so far
    tokens: Vec<usize>,
    /// Number of tokens generated (excluding prompt)
    generated_count: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum PlannerState {
    /// Waiting for prompt
    Initial,
    /// Prompt processed, generating tokens
    Generating,
    /// Generation complete
    Done,
}

impl BatchPlanner {
    /// Create a new planner with the given configuration
    #[must_use]
    pub fn new(config: GenerationConfig) -> Self {
        Self {
            config,
            state: PlannerState::Initial,
            tokens: Vec::new(),
            generated_count: 0,
        }
    }

    /// Plan the next step in the generation process.
    ///
    /// This is the core decision function - pure logic, no GPU calls.
    ///
    /// # Arguments
    ///
    /// * `last_token` - The token produced by the previous step (if any)
    ///
    /// # Returns
    ///
    /// The next step to execute
    #[must_use]
    pub fn plan_next(&mut self, last_token: Option<usize>) -> GenerationStep {
        match self.state {
            PlannerState::Initial => {
                // Should have been given a prompt via set_prompt
                GenerationStep::Done {
                    tokens: self.tokens.clone(),
                }
            },

            PlannerState::Generating => {
                // Add the token from the previous step
                if let Some(token) = last_token {
                    self.tokens.push(token);
                    self.generated_count += 1;

                    // Check stop conditions
                    if self.should_stop(token) {
                        self.state = PlannerState::Done;
                        return GenerationStep::Done {
                            tokens: self.tokens.clone(),
                        };
                    }
                }

                // Plan next generation step
                GenerationStep::GenerateToken {
                    tokens: self.tokens.clone(),
                    use_greedy_optimization: self.config.use_greedy_path(),
                }
            },

            PlannerState::Done => GenerationStep::Done {
                tokens: self.tokens.clone(),
            },
        }
    }

    /// Set the initial prompt and get the first step
    #[must_use]
    pub fn start_with_prompt(&mut self, prompt: &[usize]) -> GenerationStep {
        self.tokens = prompt.to_vec();
        self.state = PlannerState::Generating;
        self.generated_count = 0;

        GenerationStep::ProcessPrompt {
            tokens: self.tokens.clone(),
        }
    }

    /// Check if generation should stop
    fn should_stop(&self, last_token: usize) -> bool {
        // Stop if we've generated enough tokens
        if self.generated_count >= self.config.max_tokens {
            return true;
        }

        // Stop if we hit the stop token
        if let Some(stop) = self.config.stop_token {
            if last_token == stop {
                return true;
            }
        }

        false
    }

    /// Get the current tokens
    #[must_use]
    pub fn tokens(&self) -> &[usize] {
        &self.tokens
    }

    /// Get the number of tokens generated (excluding prompt)
    #[must_use]
    pub fn generated_count(&self) -> usize {
        self.generated_count
    }

    /// Is generation complete?
    #[must_use]
    pub fn is_done(&self) -> bool {
        self.state == PlannerState::Done
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &GenerationConfig {
        &self.config
    }
}

/// Plans for a single transformer block forward pass.
///
/// Extracted decision logic for which operations to perform.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockForwardPlan {
    /// Block index
    pub block_idx: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// KV dimension (for GQA)
    pub kv_dim: usize,
    /// QKV dimension
    pub qkv_dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of KV heads (for GQA)
    pub num_kv_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Intermediate FFN dimension
    pub intermediate_dim: usize,
    /// Use SwiGLU activation (vs GELU)
    pub use_swiglu: bool,
    /// Number of Q heads per KV head (for GQA repetition)
    pub heads_per_kv: usize,
}

impl BlockForwardPlan {
    /// Create a plan from model config
    #[must_use]
    pub fn from_config(config: &GpuModelConfig, block_idx: usize, has_gate_weight: bool) -> Self {
        let head_dim = config.head_dim();
        let num_kv_heads = config.num_kv_heads;
        let heads_per_kv = config.num_heads / num_kv_heads;

        Self {
            block_idx,
            hidden_dim: config.hidden_dim,
            kv_dim: config.kv_dim(),
            qkv_dim: config.qkv_dim(),
            num_heads: config.num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim: config.intermediate_dim,
            use_swiglu: has_gate_weight,
            heads_per_kv,
        }
    }

    /// Does this config use GQA (Grouped Query Attention)?
    #[must_use]
    pub fn is_gqa(&self) -> bool {
        self.heads_per_kv > 1
    }

    /// Calculate attention output size
    #[must_use]
    pub fn attention_output_size(&self) -> usize {
        self.hidden_dim
    }
}

/// Sampling strategy decision
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SamplingStrategy {
    /// Greedy: always pick the highest probability token
    #[default]
    Greedy,
    /// Top-K: sample from top K tokens
    TopK {
        /// Number of top tokens to sample from
        k: usize,
    },
    /// Top-P (nucleus): sample from tokens with cumulative probability >= p
    TopP {
        /// Cumulative probability threshold
        p: f32,
    },
    /// Temperature-scaled sampling
    Temperature {
        /// Temperature value (higher = more random)
        temp: f32,
    },
}

/// Check if temperature value is valid for sampling
#[inline]
fn is_valid_temperature(temp: f32) -> bool {
    temp > 0.0 && temp != 1.0
}

/// Check if top_p value is valid for sampling
#[inline]
fn is_valid_top_p(p: f32) -> bool {
    p < 1.0 && p > 0.0
}

/// Check if top_k value is valid for sampling
#[inline]
fn is_valid_top_k(k: usize) -> bool {
    k > 0 && k < usize::MAX
}

/// Plans which sampling strategy to use based on parameters
#[must_use]
pub fn plan_sampling(
    temperature: Option<f32>,
    top_k: Option<usize>,
    top_p: Option<f32>,
) -> SamplingStrategy {
    // Priority: temperature > top_p > top_k > greedy
    if let Some(temp) = temperature.filter(|&t| is_valid_temperature(t)) {
        return SamplingStrategy::Temperature { temp };
    }
    if let Some(p) = top_p.filter(|&p| is_valid_top_p(p)) {
        return SamplingStrategy::TopP { p };
    }
    if let Some(k) = top_k.filter(|&k| is_valid_top_k(k)) {
        return SamplingStrategy::TopK { k };
    }
    SamplingStrategy::Greedy
}

/// Decides whether to use CPU or GPU for LM head computation
#[derive(Debug, Clone, PartialEq)]
pub enum LmHeadPath {
    /// Use CPU with transposed weights (cache-friendly)
    CpuTransposed,
    /// Use GPU matmul
    Gpu,
}

/// Plan which compute path to use for LM head
#[must_use]
pub fn plan_lm_head_path(
    vocab_size: usize,
    hidden_dim: usize,
    gpu_buffer_limit: usize,
) -> LmHeadPath {
    let elements = vocab_size * hidden_dim;

    // Use CPU for large vocab (better cache behavior)
    // or when GPU buffer would be exceeded
    if vocab_size > 8192 || elements > gpu_buffer_limit {
        LmHeadPath::CpuTransposed
    } else {
        LmHeadPath::Gpu
    }
}

include!("planner_generation_config.rs");
