//! High-level inference API for CLI tools
//!
//! This module provides a simple, high-level API for running inference
//! that can be used by CLI tools like `apr run` and `apr chat`.
//!
//! # Architecture (APR-CLI-DELEGATE-001)
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │  apr-cli    │ --> │  realizar   │ --> │   trueno    │
//! │  (100 LOC)  │     │   infer.rs  │     │   SIMD/GPU  │
//! └─────────────┘     └─────────────┘     └─────────────┘
//! ```
//!
//! The `apr run` command delegates ALL inference to this module.
//! This eliminates ~1800 lines of duplicated code in apr-cli.
//!
//! # Example
//!
//! ```rust,ignore
//! use realizar::infer::{InferenceConfig, run_inference};
//!
//! let config = InferenceConfig::new("model.gguf")
//!     .with_prompt("Hello, world!")
//!     .with_max_tokens(32);
//!
//! let result = run_inference(config)?;
//! println!("{}", result.text);
//! ```

use crate::error::{RealizarError, Result};
use crate::format::{detect_format, ModelFormat};
use std::path::PathBuf;
use std::time::Instant;

/// PMAT-173 / GH-321: Convert GGML quantization type to human-readable string.
/// Uses unified `GgmlQuantType` enum — single source of truth.
pub(crate) fn qtype_to_dtype_str(qtype: u32) -> &'static str {
    crate::gguf::GgmlQuantType::from_id(qtype).map_or("Unknown", crate::gguf::GgmlQuantType::as_str)
}

/// Configuration for inference
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Path to model file (GGUF, APR, or SafeTensors)
    pub model_path: PathBuf,
    /// Text prompt for generation
    pub prompt: Option<String>,
    /// Token IDs for generation (alternative to prompt)
    pub input_tokens: Option<Vec<u32>>,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Disable GPU acceleration
    pub no_gpu: bool,
    /// Enable inference tracing (APR-TRACE-001)
    pub trace: bool,
    /// Verbose tracing output
    pub trace_verbose: bool,
    /// Trace output file path
    pub trace_output: Option<PathBuf>,
    /// Specific trace steps to capture
    pub trace_steps: Option<Vec<String>>,
    /// Show verbose loading/progress output
    pub verbose: bool,
    /// Stop token IDs for early termination (GH-373)
    pub stop_tokens: Vec<u32>,
    /// INTERNAL: Use mock backend for testing (PMAT-COV-95)
    #[doc(hidden)]
    pub use_mock_backend: bool,
}

impl InferenceConfig {
    /// Create a new inference config for a model file
    #[must_use]
    pub fn new(model_path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: model_path.into(),
            prompt: None,
            input_tokens: None,
            max_tokens: 32,
            temperature: 0.0, // Greedy by default
            top_k: 1,
            no_gpu: false,
            trace: false,
            trace_verbose: false,
            trace_output: None,
            trace_steps: None,
            verbose: false,
            stop_tokens: Vec::new(),
            use_mock_backend: false,
        }
    }

    /// Set the text prompt
    #[must_use]
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set input tokens directly
    #[must_use]
    pub fn with_input_tokens(mut self, tokens: Vec<u32>) -> Self {
        self.input_tokens = Some(tokens);
        self
    }

    /// Set maximum tokens to generate
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Set temperature (0.0 = greedy)
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set top-k sampling
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Disable GPU acceleration
    #[must_use]
    pub fn without_gpu(mut self) -> Self {
        self.no_gpu = true;
        self
    }

    /// Enable verbose output
    #[must_use]
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enable inference tracing
    #[must_use]
    pub fn with_trace(mut self, trace: bool) -> Self {
        self.trace = trace;
        self
    }

    /// Set trace output file path
    #[must_use]
    pub fn with_trace_output(mut self, path: impl Into<PathBuf>) -> Self {
        self.trace_output = Some(path.into());
        self
    }

    /// Set stop token IDs for early termination (GH-373)
    #[must_use]
    pub fn with_stop_tokens(mut self, stop_tokens: Vec<u32>) -> Self {
        self.stop_tokens = stop_tokens;
        self
    }
}

// ============================================================================
// PreparedTokens - Compile-time chat template enforcement (PMAT-236)
// ============================================================================

/// Tokenized input that has been processed through chat template formatting.
///
/// # Compile-time enforcement (Poka-Yoke)
///
/// The inner `Vec<u32>` is **private** - the only way to construct `PreparedTokens`
/// is via `prepare_tokens()`, which ALWAYS applies chat template formatting for
/// instruct models. This makes it a **compile error** to pass raw tokens to
/// inference functions, preventing the bug where SafeTensors inference skipped
/// chat template application (producing "4" then garbage).
///
/// # Theoretical basis
///
/// Shingo, S. (1986). *Zero Quality Control: Source Inspection and the Poka-Yoke System*.
/// Brady, E. (2017). *Type-Driven Development with Idris*.
///
/// # References
///
/// - PMAT-236: Chat template enforcement for multi-format inference
/// - GH-205: SafeTensors inference garbage root cause
#[derive(Debug, Clone)]
pub struct PreparedTokens {
    /// Tokenized input (PRIVATE - enforces construction via prepare_tokens only)
    tokens: Vec<u32>,
    /// Number of input tokens (for separating prefill from generated tokens)
    input_count: usize,
}

impl PreparedTokens {
    /// Access the prepared token IDs (read-only).
    #[must_use]
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Number of input tokens.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.input_count
    }
}

/// Prepare tokens for inference, applying chat template for instruct models.
///
/// This is the ONLY way to create `PreparedTokens`. It handles:
/// 1. Format detection (GGUF vs SafeTensors vs APR)
/// 2. Architecture detection (Qwen2, LLaMA, Phi, etc.)
/// 3. Chat template application for instruct models
/// 4. Tokenization using the appropriate tokenizer
///
/// # Chat Template Rules
///
/// - If model name/architecture contains "instruct", chat template is applied
/// - GGUF: uses embedded tokenizer + architecture from metadata
/// - SafeTensors: uses sibling tokenizer.json + config.json architecture
/// - APR: uses sibling tokenizer.json + model metadata
///
/// # Errors
///
/// Returns error if the model cannot be read or tokenization fails.
pub fn prepare_tokens(config: &InferenceConfig, format: &ModelFormat) -> Result<PreparedTokens> {
    // If raw token IDs are provided, use them directly (user knows what they're doing)
    if let Some(ref tokens) = config.input_tokens {
        return Ok(PreparedTokens {
            input_count: tokens.len(),
            tokens: tokens.clone(),
        });
    }

    let prompt = match config.prompt {
        Some(ref p) => p.clone(),
        None => {
            return Ok(PreparedTokens {
                tokens: vec![1u32],
                input_count: 1,
            })
        },
    };

    match format {
        ModelFormat::Gguf => prepare_tokens_gguf(config, &prompt),
        ModelFormat::SafeTensors => prepare_tokens_safetensors(config, &prompt),
        ModelFormat::Apr => prepare_tokens_apr(config, &prompt),
    }
}

/// Prepare tokens for GGUF format (chat template from GGUF metadata)
///
/// GH-278: Only apply chat template when the GGUF actually contains one in its
/// metadata (`tokenizer.chat_template`). Previously, ALL models with known
/// architectures (llama, qwen2, etc.) got chat-template wrapping even if they
/// were base completion models, causing complete output divergence vs llama.cpp.
///
/// BOS token: Prepend BOS when the model metadata says `add_bos_token = true`
/// or when a BOS token ID exists and `add_bos_token` is not explicitly false.
/// This matches llama.cpp behavior for LLaMA-family models.
fn prepare_tokens_gguf(config: &InferenceConfig, prompt: &str) -> Result<PreparedTokens> {
    use crate::chat_template::{format_messages, ChatMessage};
    use crate::gguf::{GGUFValue, MappedGGUFModel};

    let mapped = MappedGGUFModel::from_path(&config.model_path)?;
    let gguf_arch = mapped.model.architecture().unwrap_or("transformer");

    // GH-278: Check if model actually has a chat template in its GGUF metadata.
    // Base models (SmolLM-135M, GPT-2) don't have one — only instruct/chat models do.
    let has_chat_template = mapped
        .model
        .metadata
        .get("tokenizer.chat_template")
        .is_some_and(|v| matches!(v, GGUFValue::String(s) if !s.is_empty()));

    let model_name = config
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");
    let filename_instruct = model_name.to_lowercase().contains("instruct")
        || model_name.to_lowercase().contains("-chat");

    // Only apply chat template if the model actually has one, or filename says instruct
    let formatted_prompt = if has_chat_template || filename_instruct {
        let template_hint = apr_arch_to_template_hint(gguf_arch, model_name);
        let messages = vec![ChatMessage::user(prompt)];
        format_messages(&messages, Some(template_hint)).unwrap_or_else(|_| prompt.to_string())
    } else {
        prompt.to_string()
    };

    if config.verbose {
        eprintln!(
            "[DEBUG] has_chat_template={}, filename_instruct={}",
            has_chat_template, filename_instruct
        );
        eprintln!(
            "[DEBUG] formatted_prompt={:?}",
            &formatted_prompt[..formatted_prompt.len().min(200)]
        );
    }

    let mut tokens = mapped.model.encode(&formatted_prompt).ok_or_else(|| {
        RealizarError::InferenceError(format!(
            "Tokenizer encode failed for GGUF model (no tokenizer data in GGUF file?). \
                 Prompt length: {} chars",
            formatted_prompt.len()
        ))
    })?;

    // GH-278: Prepend BOS token to match llama.cpp behavior.
    // llama.cpp adds BOS when add_bos_token is true (default for LLaMA-family).
    // Only add if not already present AND model has a BOS token defined.
    let add_bos = match mapped
        .model
        .metadata
        .get(crate::gguf::keys::TOKENIZER_ADD_BOS)
    {
        Some(GGUFValue::Bool(b)) => *b,
        // GH-326: Derive BOS default from architecture constraints, not hardcoded string.
        // Models with absolute position embeddings (GPT-2, BERT) use BPE → no BOS.
        // Models with RoPE (LLaMA, Qwen, Mistral) use SentencePiece → add BOS.
        _ => {
            let arch = mapped
                .model
                .metadata
                .get(crate::gguf::keys::GENERAL_ARCHITECTURE)
                .and_then(|v| {
                    if let GGUFValue::String(s) = v {
                        Some(s.as_str())
                    } else {
                        None
                    }
                })
                // R-01 (Meyer DbC): "unknown" — don't pretend unidentified model is LLaMA.
                .unwrap_or("unknown");
            let constraints = crate::gguf::ArchConstraints::from_architecture(arch);
            constraints.positional_encoding != crate::gguf::PositionalEncoding::Absolute
        },
    };

    if add_bos {
        if let Some(bos_id) = mapped.model.bos_token_id() {
            if tokens.first() != Some(&bos_id) {
                tokens.insert(0, bos_id);
            }
        }
    }

    if config.verbose {
        eprintln!(
            "[DEBUG] add_bos={}, encoded {} tokens: {:?}",
            add_bos,
            tokens.len(),
            &tokens[..tokens.len().min(30)]
        );
    }

    Ok(PreparedTokens {
        input_count: tokens.len(),
        tokens,
    })
}

/// Prepare tokens for SafeTensors format (chat template from config.json)
fn prepare_tokens_safetensors(config: &InferenceConfig, prompt: &str) -> Result<PreparedTokens> {
    use crate::apr::AprV2Model;
    use crate::chat_template::{format_messages, ChatMessage};
    use crate::safetensors::SafetensorsConfig;

    // Load config.json for architecture detection
    let st_config = SafetensorsConfig::load_from_sibling(&config.model_path);
    let architecture = st_config
        .as_ref()
        .map(SafetensorsConfig::architecture)
        .unwrap_or_default();

    let model_name = config
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // Detect instruct model from architecture or filename
    let arch_lower = architecture.to_lowercase();
    let is_instruct = arch_lower.contains("instruct")
        || model_name.to_lowercase().contains("instruct")
        || matches!(
            arch_lower.as_str(),
            "qwen2forcausallm" | "llamaforcausallm" | "mistralforcausallm" | "phiforcausallm"
        );

    let formatted_prompt = if is_instruct {
        let template_hint = safetensors_arch_to_template_hint(&architecture, model_name);
        let messages = vec![ChatMessage::user(prompt)];
        format_messages(&messages, Some(template_hint)).unwrap_or_else(|_| prompt.to_string())
    } else {
        prompt.to_string()
    };

    let tokens =
        AprV2Model::encode_text(&config.model_path, &formatted_prompt).ok_or_else(|| {
            RealizarError::InferenceError(format!(
                "Tokenizer encode failed for SafeTensors model (no tokenizer.json sibling?). \
                 Prompt length: {} chars",
                formatted_prompt.len()
            ))
        })?;

    Ok(PreparedTokens {
        input_count: tokens.len(),
        tokens,
    })
}

/// Prepare tokens for APR format (chat template from model metadata)
fn prepare_tokens_apr(config: &InferenceConfig, prompt: &str) -> Result<PreparedTokens> {
    use crate::apr::AprV2Model;
    use crate::chat_template::{format_messages, ChatMessage};

    let model_name = config
        .model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("");

    // PMAT-237: Detect instruct from MODEL DATA, not filename.
    // Filename heuristic silently skips chat template for hash-named APR files.
    // Three-tier detection: architecture metadata > vocab special tokens > filename fallback.
    let (apr_arch, has_chatml_tokens) =
        if config.model_path.extension().is_some_and(|e| e == "apr") {
            match AprV2Model::load(&config.model_path) {
                Ok(model) => {
                    let arch = model.metadata().architecture.clone().unwrap_or_default();
                    let has_chatml = model.metadata().get_embedded_vocabulary().is_some_and(
                        |vocab: Vec<String>| vocab.iter().any(|t| t == "<|im_start|>"),
                    );
                    (arch, has_chatml)
                },
                Err(_) => (String::new(), false),
            }
        } else {
            (String::new(), false)
        };

    let is_instruct_arch = matches!(
        apr_arch.to_lowercase().as_str(),
        "qwen2" | "qwen" | "llama" | "mistral" | "phi" | "phi3"
    );
    let filename_instruct = model_name.to_lowercase().contains("instruct");

    let is_instruct = is_instruct_arch || has_chatml_tokens || filename_instruct;

    let formatted_prompt = if is_instruct {
        let template_hint = apr_arch_to_template_hint(&apr_arch, model_name);
        let messages = vec![ChatMessage::user(prompt)];
        format_messages(&messages, Some(template_hint)).unwrap_or_else(|_| prompt.to_string())
    } else {
        prompt.to_string()
    };

    let tokens =
        AprV2Model::encode_text(&config.model_path, &formatted_prompt).ok_or_else(|| {
            RealizarError::InferenceError(format!(
                "Tokenizer encode failed for APR model (no tokenizer in APR metadata?). \
                 Prompt length: {} chars",
                formatted_prompt.len()
            ))
        })?;

    Ok(PreparedTokens {
        input_count: tokens.len(),
        tokens,
    })
}

/// Map SafeTensors architecture string to chat template hint.
///
/// GH-317/318: Contract-driven — uses normalize_architecture() from
/// tensor-names-v1.yaml. No contains() heuristics, no model_name fallback.
fn safetensors_arch_to_template_hint(architecture: &str, _model_name: &str) -> &'static str {
    crate::tensor_names::normalize_architecture(architecture)
}

include!("inference_result.rs");
include!("gguf_gpu_generate.rs");
include!("mod_log_transformer_eos.rs");
include!("mod_05.rs");
include!("batch.rs");
