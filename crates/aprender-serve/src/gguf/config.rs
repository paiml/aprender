//! GGUF configuration extraction
//!
//! Extracts model configuration from GGUF metadata.
//!
//! This module defines `GGUFConfig` which holds the transformer
//! architecture parameters needed for inference, and `ArchConstraints`
//! which encodes compile-time model family contract data from
//! `aprender/contracts/model-families/*.yaml`.

use super::types::GGUFModel;
use crate::error::{RealizarError, Result};

// ---------------------------------------------------------------------------
// ArchConstraints — contract-driven architecture behavior (GH-278)
// ---------------------------------------------------------------------------

/// Normalization type per model family contract.
///
/// Source: `constraints.norm_type` in `aprender/contracts/model-families/*.yaml`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// Standard Layer Normalization with optional bias (GPT-2, phi, BERT, whisper)
    LayerNorm,
    /// Root Mean Square Normalization without bias (LLaMA, Qwen2, Mistral, etc.)
    RmsNorm,
}

/// Activation function per model family contract.
///
/// Source: `constraints.activation` in `aprender/contracts/model-families/*.yaml`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// Gaussian Error Linear Unit (GPT-2, BERT, gemma, whisper)
    Gelu,
    /// Sigmoid Linear Unit (LLaMA, Qwen2, Mistral, phi, etc.)
    Silu,
}

/// Positional encoding type per model family contract.
///
/// Source: `constraints.positional_encoding` in `aprender/contracts/model-families/*.yaml`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PositionalEncoding {
    /// Learned absolute position embeddings (GPT-2, BERT, whisper)
    Absolute,
    /// Rotary Position Embedding (LLaMA, Qwen2, Mistral, phi, etc.)
    Rope,
    /// Attention with Linear Biases (BLOOM, MPT)
    Alibi,
    /// Relative position bias in attention (T5, PMAT-395)
    Relative,
    /// No positional encoding (mamba, rwkv7)
    None,
}

/// FFN/MLP structure per model family contract.
///
/// Source: `constraints.mlp_type` in `aprender/contracts/model-families/*.yaml`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlpType {
    /// Standard GELU MLP: up → GELU → down (GPT-2, BERT, whisper)
    GeluMlp,
    /// SwiGLU: gate ⊙ SiLU(up) → down (LLaMA, Qwen2, Mistral, phi, etc.)
    SwiGlu,
    /// Gated MLP: gate ⊙ GELU(up) → down (gemma, moonshine)
    GatedMlp,
}

/// Weight storage layout per model family contract.
///
/// Source: `constraints.mlp_type` + `shape_template` in contracts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightLayout {
    /// Standard Linear layout: `[out_features, in_features]` — no transpose needed
    Linear,
    /// Conv1D layout: `[in_features, out_features]` — requires transpose for `y = x @ W^T`
    Conv1D,
}

/// Architecture constraints derived from model family contracts.
///
/// These are compile-time constants per architecture, NOT runtime heuristics.
/// Source of truth: `aprender/contracts/model-families/*.yaml`
///
/// # Usage
///
/// ```ignore
/// let c = ArchConstraints::from_architecture("gpt2");
/// // c.norm_type == NormType::LayerNorm
/// // c.activation == Activation::Gelu
/// // c.positional_encoding == PositionalEncoding::Absolute
/// // c.mlp_type == MlpType::GeluMlp
/// // c.weight_layout == WeightLayout::Conv1D
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArchConstraints {
    /// Normalization type (LayerNorm or RMSNorm)
    pub norm_type: NormType,
    /// Activation function (GELU or SiLU)
    pub activation: Activation,
    /// Positional encoding (Absolute, RoPE, or None)
    pub positional_encoding: PositionalEncoding,
    /// FFN structure (GeluMlp, SwiGlu, or GatedMlp)
    pub mlp_type: MlpType,
    /// Weight storage layout (Linear or Conv1D)
    pub weight_layout: WeightLayout,
    /// Whether the architecture has bias terms in attention/FFN layers
    pub has_bias: bool,
    /// Whether embedding and LM head weights are tied
    pub tied_embeddings: bool,
    /// Whether Q and K projections have per-head RMSNorm (GH-279: Qwen3)
    pub has_qk_norm: bool,
    /// Default norm epsilon when GGUF metadata is missing
    pub default_eps: f32,
}

// GH-323: Generated from arch-constraints-v1.yaml by build.rs.
// The include! pulls in from_architecture_generated() which does the actual match.
// Fallback: if build.rs can't find the YAML, it uses arch_constraints_fallback.rs.
include!(concat!(env!("OUT_DIR"), "/arch_constraints_generated.rs"));

impl ArchConstraints {
    /// Look up architecture constraints from the GGUF `general.architecture` value.
    ///
    /// Maps architecture names to their contract-defined behavior per
    /// `provable-contracts/contracts/arch-constraints-v1.yaml`.
    /// Unknown architectures fall back to LLaMA-like defaults.
    ///
    /// AUTO-GENERATED via build.rs from arch-constraints-v1.yaml.
    #[must_use]
    pub fn from_architecture(arch: &str) -> Self {
        contract_pre_o_projection_transpose!();
        from_architecture_generated(arch)
    }

    /// Whether this architecture uses RoPE positional encoding.
    #[must_use]
    pub fn uses_rope(&self) -> bool {
        self.positional_encoding == PositionalEncoding::Rope
    }

    /// Whether this architecture uses RMSNorm (vs LayerNorm).
    #[must_use]
    pub fn uses_rmsnorm(&self) -> bool {
        self.norm_type == NormType::RmsNorm
    }

    /// Whether weight matrices need Conv1D transpose.
    #[must_use]
    pub fn needs_transpose(&self) -> bool {
        self.weight_layout == WeightLayout::Conv1D
    }

    /// Whether this architecture uses a gated FFN (SwiGLU or GatedMLP).
    ///
    /// Gated FFN architectures require `ffn_gate_weight` to be present in model layers.
    /// Non-gated architectures (GeluMlp) use a simple up → activation → down path.
    #[must_use]
    pub fn has_gate_ffn(&self) -> bool {
        !matches!(self.mlp_type, MlpType::GeluMlp)
    }

    /// Whether this architecture uses learned absolute position embeddings.
    ///
    /// Architectures with absolute encoding (GPT-2, BERT, whisper) add learned
    /// position vectors to token embeddings. RoPE-based models skip this.
    #[must_use]
    pub fn uses_absolute_positions(&self) -> bool {
        self.positional_encoding == PositionalEncoding::Absolute
    }

    /// PMAT-395: Whether this is a T5-style relative position bias arch.
    #[must_use]
    pub fn uses_relative_positions(&self) -> bool {
        self.positional_encoding == PositionalEncoding::Relative
    }
}

/// Infer RoPE type from architecture string.
///
/// Returns 0 for NORM style (adjacent pairs), 2 for NEOX style (split halves).
/// Matches llama.cpp's rope type inference (llama-model.cpp:7763-7811).
///
/// GH-329: Single source of truth — all rope_type inference MUST go through here.
#[must_use]
pub fn infer_rope_type(arch: &str) -> u32 {
    let arch_lower = arch.to_lowercase();
    // NEOX style (type 2): pairs offset by n_rot/2
    // This list matches llama.cpp's llama-model.cpp:7763-7811
    const NEOX_ARCHITECTURES: &[&str] = &[
        "qwen",
        "qwen2",
        "qwen3",
        "qwen3_5",
        "qwen3.5",
        "stablelm",
        "phi2",
        "phi3",
        "phi",
        "gemma",
        "gemma2",
        "gemma3",
        "starcoder2",
        "gptneox",
        "gpt_neox",
        "falcon",
        "falcon_h1",
        "codeshell",
        "orion",
        "bert",
        "nomic-bert",
        "dbrx",
        "olmo2",
        "olmoe",
        "plamo",
        "plamo2",
        "openelm",
        "exaone",
        "minicpm3",
        "nemotron",
        "internlm2",
        "deepseek",
        "deepseek2",
    ];
    for &neox_arch in NEOX_ARCHITECTURES {
        if arch_lower.contains(neox_arch) {
            return 2; // NEOX style
        }
    }
    // NORM style (type 0): adjacent pairs - default for LLaMA, TinyLlama
    0
}

/// Configuration for GGUF transformer inference
#[derive(Debug, Clone)]
pub struct GGUFConfig {
    /// Model architecture (e.g., "phi2", "llama", "qwen2")
    pub architecture: String,
    /// Contract-derived architecture constraints (norm type, activation, etc.)
    pub constraints: ArchConstraints,
    /// Embedding dimension (hidden size)
    pub hidden_dim: usize,
    /// Number of transformer layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Number of key-value heads (for GQA, often num_heads or num_heads/8)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// FFN intermediate dimension
    pub intermediate_dim: usize,
    /// Context length
    pub context_length: usize,
    /// RoPE theta (position encoding base)
    pub rope_theta: f32,
    /// Layer norm epsilon
    pub eps: f32,
    /// RoPE type: 0 = NORM (adjacent pairs), 2 = NEOX (split halves)
    pub rope_type: u32,
    /// Explicit per-head dimension for Q/K projections (from GGUF metadata).
    ///
    /// `None` means derive as `hidden_dim / num_heads` (correct for most models).
    /// `Some(128)` for Qwen3-0.6B where `hidden_dim=1024, num_heads=16` but `head_dim=128`
    /// meaning `q_dim = num_heads * head_dim = 2048 ≠ hidden_dim`.
    ///
    /// Source: GGUF metadata `{arch}.attention.key_length`.
    pub explicit_head_dim: Option<usize>,
    /// BOS token ID from GGUF metadata (used for GPU validation probe)
    /// None means BOS is unknown — GPU validation will be skipped.
    pub bos_token_id: Option<u32>,
    /// GH-330: EOS token ID from GGUF metadata.
    ///
    /// **Design by Contract (Meyer 1992)**: This is a class invariant —
    /// after construction, the model must know its own EOS token.
    /// Callers must NOT use hardcoded fallbacks like `unwrap_or(151645)`.
    pub eos_token_id: Option<u32>,
}

/// Architecture-default BOS token IDs for weights-only GGUFs.
///
/// Weights-only GGUF files (e.g., from pacha) contain only 4 metadata keys
/// and lack `tokenizer.ggml.bos_token_id`. This function provides a known-good
/// BOS token ID based on the architecture, enabling GPU validation (F2-VALIDATION)
/// that would otherwise be skipped.
///
/// Source of truth: `special-tokens-registry-v1.yaml`
fn default_bos_for_architecture(arch: &str) -> Option<u32> {
    match arch {
        "qwen2" | "qwen3" | "qwen3moe" => Some(151_643),
        // qwen3_5: no BOS token in config.json
        "llama" => Some(128_000),
        "mistral" => Some(1),
        "gemma" | "gemma2" => Some(2),
        "deepseek" | "deepseek2" => Some(0),
        "phi3" => Some(1),
        // phi2/phi/gpt2: no BOS token
        _ => None,
    }
}

/// Architecture-default EOS token IDs for weights-only GGUFs.
///
/// Same rationale as `default_bos_for_architecture` — weights-only GGUFs
/// lack `tokenizer.ggml.eos_token_id`. This provides a known-good EOS
/// based on the architecture contract.
///
/// Source of truth: `special-tokens-registry-v1.yaml`
pub(crate) fn default_eos_for_architecture(arch: &str) -> Option<u32> {
    match arch {
        "qwen2" | "qwen3" | "qwen3moe" => Some(151_645),
        "qwen3_5" => Some(248_044),
        "llama" => Some(128_001),
        "mistral" => Some(2),
        "gemma" | "gemma2" => Some(1),
        "deepseek" | "deepseek2" => Some(1),
        "phi3" => Some(32_000),
        "phi2" | "phi" | "gpt2" => Some(50_256),
        _ => None,
    }
}

/// C-02 (Meyer DbC): Architecture-default rope_theta values.
///
/// Different architectures use fundamentally different RoPE frequency bases:
/// - LLaMA/Mistral/Gemma: 10,000.0 (original RoPE paper)
/// - Qwen2/Qwen3: 1,000,000.0 (extended context)
/// - DeepSeek: 10,000.0
/// - Phi: 10,000.0
///
/// Using the wrong rope_theta produces completely wrong positional encodings.
/// Sources: HuggingFace config.json files, architecture papers.
pub(crate) fn default_rope_theta_for_architecture(arch: &str) -> f32 {
    match arch {
        "qwen2" | "qwen3" => 1_000_000.0,
        "llama" | "mistral" | "gemma" | "gemma2" | "deepseek" | "deepseek2" => 10_000.0,
        "phi2" | "phi3" | "phi" => 10_000.0,
        // Conservative default: LLaMA's 10K is the original RoPE value
        _ => 10_000.0,
    }
}

impl GGUFConfig {
    /// Extract configuration from APR model metadata.
    ///
    /// Shared by both F32 loading (`loading.rs`) and quantized loading
    /// (`loader_apr_quantized.rs`). Single source of truth for APR → `GGUFConfig`.
    ///
    /// # Arguments
    ///
    /// * `apr` - Memory-mapped APR model
    /// * `vocab_size` - Pre-computed vocab size (caller infers from metadata or embedding tensor)
    ///
    /// # Errors
    ///
    /// Returns an error if required metadata fields (`architecture`, `hidden_size`,
    /// `num_layers`, `num_heads`, `intermediate_size`) are missing.
    pub fn from_apr(apr: &crate::apr::MappedAprModel, vocab_size: usize) -> Result<Self> {
        // C-01 (Meyer DbC): Architecture is required (determines rope_theta/eps defaults).
        let architecture = apr.metadata.architecture.clone().ok_or_else(|| {
            RealizarError::InvalidConfiguration(
                "C-01: APR model missing 'architecture' metadata — cannot infer model type".into(),
            )
        })?;
        // C-03 (Meyer DbC): Required model dimensions — no silent defaults.
        let hidden_dim = apr.metadata.hidden_size.ok_or_else(|| {
            RealizarError::InvalidConfiguration(
                "C-03: APR model missing 'hidden_size' metadata".into(),
            )
        })?;
        let num_layers = apr.metadata.num_layers.ok_or_else(|| {
            RealizarError::InvalidConfiguration(
                "C-03: APR model missing 'num_layers' metadata".into(),
            )
        })?;
        let num_heads = apr.metadata.num_heads.ok_or_else(|| {
            RealizarError::InvalidConfiguration(
                "C-03: APR model missing 'num_heads' metadata".into(),
            )
        })?;
        let num_kv_heads = apr.metadata.num_kv_heads.unwrap_or(num_heads);
        let intermediate_dim = apr.metadata.intermediate_size.ok_or_else(|| {
            RealizarError::InvalidConfiguration(
                "C-03: APR model missing 'intermediate_size' metadata".into(),
            )
        })?;
        let constraints = ArchConstraints::from_architecture(&architecture);
        let eps = apr.metadata.rms_norm_eps.unwrap_or(constraints.default_eps);
        // C-02: rope_theta from metadata, or architecture-specific default
        let rope_theta = apr
            .metadata
            .rope_theta
            .unwrap_or_else(|| default_rope_theta_for_architecture(&architecture));
        // GH-329: Read from APR metadata, infer from architecture when absent
        let rope_type = apr
            .metadata
            .rope_type
            .unwrap_or_else(|| infer_rope_type(&architecture));
        let context_length = apr.metadata.max_position_embeddings.unwrap_or(0);
        // GH-330: EOS from APR metadata, with architecture contract fallback
        let eos_token_id = apr
            .metadata
            .get_embedded_eos_token_id()
            .or_else(|| default_eos_for_architecture(&architecture));
        let bos_token_id = apr.metadata.get_embedded_bos_token_id();

        Ok(Self {
            architecture,
            constraints,
            vocab_size,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            intermediate_dim,
            eps,
            rope_theta,
            rope_type,
            context_length,
            explicit_head_dim: None,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Per-head dimension for Q/K projections.
    ///
    /// Uses explicit value from GGUF metadata if available, otherwise `hidden_dim / num_heads`.
    #[inline]
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.explicit_head_dim.unwrap_or(if self.num_heads > 0 {
            self.hidden_dim / self.num_heads
        } else {
            self.hidden_dim
        })
    }

    /// Total Q projection dimension: `num_heads * head_dim`.
    ///
    /// For most models this equals `hidden_dim`, but Qwen3-0.6B has
    /// `q_dim = 16 * 128 = 2048` while `hidden_dim = 1024`.
    #[inline]
    #[must_use]
    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim()
    }

    /// Total KV projection dimension: `num_kv_heads * head_dim`.
    #[inline]
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim()
    }

    /// PMAT-395: Whether this is an encoder-decoder architecture.
    ///
    /// Derived from constraints — T5 and Whisper use relative/absolute
    /// position bias with LayerNorm + GELU, but the key signal is the
    /// architecture string matching "t5" or "encoder-decoder".
    #[must_use]
    pub fn is_encoder_decoder(&self) -> bool {
        let arch = self.architecture.to_lowercase();
        arch == "t5" || arch == "encoder-decoder" || arch == "whisper"
    }

    /// GH-305: Infer explicit head_dim from GGUF metadata or tensor shapes.
    ///
    /// Returns `Some(head_dim)` only when it differs from `hidden_dim / num_heads`.
    fn infer_explicit_head_dim(
        model: &GGUFModel,
        hidden_dim: usize,
        num_heads: usize,
    ) -> Option<usize> {
        let default_head_dim = if num_heads > 0 {
            hidden_dim / num_heads
        } else {
            hidden_dim
        };
        model
            .key_length()
            .or_else(|| {
                // Fallback: infer from blk.0.attn_q.weight shape: [q_dim, hidden_dim] → q_dim / num_heads
                model
                    .tensors
                    .iter()
                    .find(|t| t.name == "blk.0.attn_q.weight")
                    .and_then(|t| {
                        let d0 = t.dims.first().copied()? as usize;
                        if d0 > 0 && num_heads > 0 && d0.is_multiple_of(num_heads) {
                            Some(d0 / num_heads)
                        } else {
                            None
                        }
                    })
            })
            .filter(|&hd| hd != default_head_dim)
    }

    /// GH-306: Infer intermediate_dim from ffn_down or ffn_up tensor shapes.
    ///
    /// ffn_down is preferred because some architectures (Phi-3.5) fuse gate+up,
    /// doubling ffn_up's first dimension.
    fn infer_intermediate_dim(model: &GGUFModel, hidden_dim: usize) -> usize {
        let extract_dim = |dims: &[u64]| -> usize {
            let d0 = dims.first().copied().unwrap_or(hidden_dim as u64 * 4) as usize;
            let d1 = dims.get(1).copied().unwrap_or(hidden_dim as u64) as usize;
            if d1 == hidden_dim {
                d0
            } else if d0 == hidden_dim {
                d1
            } else {
                d0
            }
        };

        model
            .tensors
            .iter()
            .find(|t| t.name == "blk.0.ffn_down.weight")
            .map(|t| extract_dim(&t.dims))
            .or_else(|| {
                model
                    .tensors
                    .iter()
                    .find(|t| t.name == "blk.0.ffn_up.weight")
                    .map(|t| extract_dim(&t.dims))
            })
            .unwrap_or(hidden_dim * 4)
    }

    /// Extract configuration from GGUF model metadata
    ///
    /// # Errors
    ///
    /// Returns an error if required metadata fields are missing from the GGUF model.
    pub fn from_gguf(model: &GGUFModel) -> Result<Self> {
        let architecture = model
            .architecture()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Missing general.architecture in GGUF metadata".to_string(),
            })?
            .to_string();

        let hidden_dim = model
            .embedding_dim()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Missing embedding_length in GGUF metadata".to_string(),
            })?;

        let num_layers = model
            .num_layers()
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "Missing block_count in GGUF metadata".to_string(),
            })?;

        // Try to get num_heads, default based on hidden_dim if not found
        let num_heads = model.num_heads().unwrap_or(hidden_dim / 64);

        // C-13 (Meyer DbC): vocab_size from token_embd tensor dims, not hardcoded.
        // After dims.reverse(), shape is [vocab_size, hidden_dim] - vocab is at index 0.
        // Falls back to 0 (unknown) if tensor not found — downstream code must handle.
        let vocab_size = model
            .tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.dims.first().copied())
            .unwrap_or(0) as usize;

        let intermediate_dim = Self::infer_intermediate_dim(model, hidden_dim);

        let context_length = model.context_length().unwrap_or(0);

        // C-02 (Meyer DbC): rope_theta from GGUF metadata, or architecture-specific default.
        let rope_theta = model
            .rope_freq_base()
            .unwrap_or_else(|| default_rope_theta_for_architecture(&architecture));

        // GH-278: Look up contract constraints for this architecture.
        // This replaces ALL runtime heuristics (tensor presence checks, string matching)
        // with compile-time contract data from aprender/contracts/model-families/*.yaml.
        let constraints = ArchConstraints::from_architecture(&architecture);

        // Read norm epsilon from GGUF metadata, falling back to contract default.
        // The contract default is architecture-specific (e.g., 1e-5 for LLaMA, 1e-6 for Qwen2).
        let eps = model.rms_epsilon().unwrap_or(constraints.default_eps);

        // num_kv_heads (for GQA - e.g., Qwen uses fewer KV heads than Q heads)
        let num_kv_heads = model.num_kv_heads().unwrap_or(num_heads);

        // GH-305: Infer head_dim from GGUF metadata or tensor shapes.
        let explicit_head_dim = Self::infer_explicit_head_dim(model, hidden_dim, num_heads);

        // Read rope_type: 0 = NORM (adjacent pairs, default for LLaMA), 2 = NEOX (split halves)
        // LLaMA models use type 0 (adjacent pairs) per llama.cpp's LLAMA_ROPE_TYPE_NORM
        let rope_type = model.rope_type().unwrap_or(0);

        // BOS token ID from GGUF metadata, with architecture-based fallback.
        // Weights-only GGUFs (e.g., from pacha) lack tokenizer.ggml.bos_token_id.
        // Without a BOS token, GPU validation (F2-VALIDATION) is skipped entirely.
        let bos_token_id = model.bos_token_id().or_else(|| {
            let fallback = default_bos_for_architecture(&architecture);
            if fallback.is_some() {
                eprintln!(
                    "[BOS-FALLBACK] No tokenizer.ggml.bos_token_id in GGUF — using architecture default for '{architecture}'"
                );
            }
            fallback
        });

        // GH-330: EOS token ID from GGUF metadata.
        // Design by Contract: this is the class invariant — the config carries
        // the model's own EOS token. No hardcoded fallback (Meyer 1992).
        let eos_token_id = model
            .eos_token_id()
            .or_else(|| default_eos_for_architecture(&architecture));

        Ok(Self {
            architecture,
            constraints,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
            rope_type,
            explicit_head_dim,
            bos_token_id,
            eos_token_id,
        })
    }
}

// ---------------------------------------------------------------------------
// ValidatedModelConfig — newtype Poka-Yoke wrapper (PMAT-235)
// ---------------------------------------------------------------------------

/// A validated model configuration that guarantees structural invariants.
///
/// Wraps `GGUFConfig` and enforces:
/// - `hidden_dim > 0`, `num_layers > 0`, `vocab_size > 0`
/// - `num_heads > 0`, `num_kv_heads > 0`
/// - `hidden_dim % num_heads == 0` (head_dim must divide evenly)
/// - `num_heads % num_kv_heads == 0` (GQA ratio must be an integer)
///
/// The inner `GGUFConfig` is private — access fields via getters or `Deref`.
#[derive(Debug, Clone)]
pub struct ValidatedModelConfig {
    inner: GGUFConfig,
}

impl ValidatedModelConfig {
    /// Validate a `GGUFConfig` and return a `ValidatedModelConfig`.
    ///
    /// # Errors
    ///
    /// Returns `RealizarError::InvalidShape` if any structural invariant is violated.
    pub fn validate(config: GGUFConfig) -> Result<Self> {
        if config.hidden_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "hidden_dim must be > 0".to_string(),
            });
        }
        if config.num_layers == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_layers must be > 0".to_string(),
            });
        }
        if config.vocab_size == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "vocab_size must be > 0".to_string(),
            });
        }
        if config.num_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_heads must be > 0".to_string(),
            });
        }
        if config.num_kv_heads == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "num_kv_heads must be > 0".to_string(),
            });
        }
        if config.intermediate_dim == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "intermediate_dim must be > 0".to_string(),
            });
        }
        // GH-305: When head_dim is explicitly set (from GGUF metadata), hidden_dim may not
        // equal num_heads * head_dim (e.g., Qwen3-0.6B: hidden=1024, heads=16, head_dim=128).
        // Only enforce divisibility when head_dim is NOT explicitly overridden.
        if config.explicit_head_dim.is_none() && !config.hidden_dim.is_multiple_of(config.num_heads)
        {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "hidden_dim ({}) must be divisible by num_heads ({}) when head_dim is derived",
                    config.hidden_dim, config.num_heads
                ),
            });
        }
        if config.head_dim() == 0 {
            return Err(RealizarError::InvalidShape {
                reason: "head_dim must be > 0".to_string(),
            });
        }
        if !config.num_heads.is_multiple_of(config.num_kv_heads) {
            return Err(RealizarError::InvalidShape {
                reason: format!(
                    "num_heads ({}) must be divisible by num_kv_heads ({}) — GQA ratio must be an integer",
                    config.num_heads, config.num_kv_heads
                ),
            });
        }

        // Upper-bound + range checks from model-metadata-bounds-v1.yaml
        validate_metadata_bounds(&config)?;

        Ok(Self { inner: config })
    }

    /// Load and validate directly from a GGUF model.
    ///
    /// Calls `GGUFConfig::from_gguf()` then validates the result.
    ///
    /// # Errors
    ///
    /// Returns an error if metadata extraction or validation fails.
    pub fn from_gguf(model: &GGUFModel) -> Result<Self> {
        let config = GGUFConfig::from_gguf(model)?;
        Self::validate(config)
    }

    /// Load and validate directly from an APR model.
    ///
    /// Calls `GGUFConfig::from_apr()` then validates the result.
    ///
    /// # Arguments
    ///
    /// * `apr` - Memory-mapped APR model
    /// * `vocab_size` - Pre-computed vocab size (caller infers from metadata or embedding tensor)
    ///
    /// # Errors
    ///
    /// Returns an error if metadata extraction or validation fails.
    pub fn from_apr(apr: &crate::apr::MappedAprModel, vocab_size: usize) -> Result<Self> {
        let config = GGUFConfig::from_apr(apr, vocab_size)?;
        Self::validate(config)
    }

    /// Load and validate from a `SafetensorsConfig` (config.json).
    ///
    /// Constructs a `GGUFConfig` from SafeTensors fields and validates dimension invariants.
    /// This is a validation gate — the SafeTensors path continues building its own
    /// `AprTransformerConfig`, but validates dimensions first.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing or invariants are violated.
    pub fn from_safetensors_config(config: &crate::SafetensorsConfig) -> Result<Self> {
        let hidden_dim = config
            .hidden_size
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "config.json missing hidden_size".to_string(),
            })?;
        let num_layers = config
            .num_hidden_layers
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "config.json missing num_hidden_layers".to_string(),
            })?;
        let num_heads = config
            .num_attention_heads
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "config.json missing num_attention_heads".to_string(),
            })?;
        let num_kv_heads = config.num_kv_heads();
        let vocab_size = config
            .vocab_size
            .ok_or_else(|| RealizarError::InvalidShape {
                reason: "config.json missing vocab_size".to_string(),
            })?;
        let intermediate_dim = config.intermediate_size.unwrap_or(hidden_dim * 4);
        let context_length = config.max_position_embeddings.unwrap_or(0);
        let architecture = config.architecture();
        let rope_theta = config
            .rope_theta
            .unwrap_or_else(|| default_rope_theta_for_architecture(&architecture));
        let eps = config.rms_norm_eps.unwrap_or(1e-6);
        let constraints = ArchConstraints::from_architecture(&architecture);

        let rope_type = infer_rope_type(&architecture);
        let raw = GGUFConfig {
            architecture,
            constraints,
            hidden_dim,
            num_layers,
            num_heads,
            num_kv_heads,
            vocab_size,
            intermediate_dim,
            context_length,
            rope_theta,
            eps,
            rope_type,
            explicit_head_dim: None,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        };
        Self::validate(raw)
    }

    // -- Getters for all fields --

    /// Model architecture (e.g., "llama", "qwen2")
    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.inner.architecture
    }

    /// Contract-derived architecture constraints
    #[must_use]
    pub fn constraints(&self) -> &ArchConstraints {
        &self.inner.constraints
    }

    /// Embedding dimension (hidden size)
    #[must_use]
    pub fn hidden_dim(&self) -> usize {
        self.inner.hidden_dim
    }

    /// Number of transformer layers
    #[must_use]
    pub fn num_layers(&self) -> usize {
        self.inner.num_layers
    }

    /// Number of attention heads
    #[must_use]
    pub fn num_heads(&self) -> usize {
        self.inner.num_heads
    }

    /// Number of key-value heads (for GQA)
    #[must_use]
    pub fn num_kv_heads(&self) -> usize {
        self.inner.num_kv_heads
    }

    /// Vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size
    }

    /// FFN intermediate dimension
    #[must_use]
    pub fn intermediate_dim(&self) -> usize {
        self.inner.intermediate_dim
    }

    /// Context length
    #[must_use]
    pub fn context_length(&self) -> usize {
        self.inner.context_length
    }

    /// RoPE theta (position encoding base)
    #[must_use]
    pub fn rope_theta(&self) -> f32 {
        self.inner.rope_theta
    }

    /// Layer norm epsilon
    #[must_use]
    pub fn eps(&self) -> f32 {
        self.inner.eps
    }

    /// RoPE type: 0 = NORM (adjacent pairs), 2 = NEOX (split halves)
    #[must_use]
    pub fn rope_type(&self) -> u32 {
        self.inner.rope_type
    }

    /// BOS token ID (None if unknown)
    #[must_use]
    pub fn bos_token_id(&self) -> Option<u32> {
        self.inner.bos_token_id
    }

    /// EOS token ID (None if unknown)
    #[must_use]
    pub fn eos_token_id(&self) -> Option<u32> {
        self.inner.eos_token_id
    }

    // -- Derived getters --

    /// Per-head dimension for Q/K projections.
    ///
    /// From GGUF metadata `{arch}.attention.key_length`, or `hidden_dim / num_heads`.
    #[must_use]
    pub fn head_dim(&self) -> usize {
        self.inner.head_dim()
    }

    /// Total Q projection dimension (`num_heads * head_dim`).
    ///
    /// May differ from `hidden_dim` (e.g., Qwen3-0.6B: q_dim=2048, hidden_dim=1024).
    #[must_use]
    pub fn q_dim(&self) -> usize {
        self.inner.q_dim()
    }

    /// Total KV dimension (`num_kv_heads * head_dim`).
    #[must_use]
    pub fn kv_dim(&self) -> usize {
        self.inner.kv_dim()
    }

    /// Borrow the inner `GGUFConfig` (backward compatibility escape hatch).
    #[must_use]
    pub fn config(&self) -> &GGUFConfig {
        &self.inner
    }

    /// Consume and return the inner `GGUFConfig`.
    ///
    /// Use this when the caller stores `GGUFConfig` (e.g., `OwnedQuantizedModel.config`)
    /// but wants the validation guarantee at the construction boundary.
    #[must_use]
    pub fn into_inner(self) -> GGUFConfig {
        self.inner
    }
}

impl std::ops::Deref for ValidatedModelConfig {
    type Target = GGUFConfig;

    fn deref(&self) -> &GGUFConfig {
        &self.inner
    }
}

/// Upper-bound and range checks from `model-metadata-bounds-v1.yaml`.
///
/// Extracted from `ValidatedModelConfig::validate()` for complexity compliance.
/// Catches corrupted/impossible configs before they cause OOM or panics.
fn validate_metadata_bounds(config: &GGUFConfig) -> Result<()> {
    check_usize_max(config.hidden_dim, 65_536, "hidden_dim")?;
    check_usize_max(config.num_layers, 256, "num_layers")?;
    check_usize_max(config.num_heads, 256, "num_heads")?;
    check_usize_max(config.num_kv_heads, 256, "num_kv_heads")?;
    check_usize_max(config.vocab_size, 1_000_000, "vocab_size")?;
    check_usize_max(config.intermediate_dim, 262_144, "intermediate_dim")?;
    check_usize_max(config.context_length, 2_097_152, "context_length")?;

    // rope_theta: must be >= 1.0 when set (0.0 means "not configured")
    if config.rope_theta > 0.0 && config.rope_theta < 1.0 {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "rope_theta {} below minimum 1.0 (model-metadata-bounds-v1)",
                config.rope_theta
            ),
        });
    }
    if config.rope_theta > 100_000_000.0 {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "rope_theta {} exceeds max 100000000.0 (model-metadata-bounds-v1)",
                config.rope_theta
            ),
        });
    }
    // eps: must be in [1e-10, 0.01] when non-zero
    if config.eps > 0.0 && config.eps < 1e-10 {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "eps {} below minimum 1e-10 (model-metadata-bounds-v1)",
                config.eps
            ),
        });
    }
    if config.eps > 0.01 {
        return Err(RealizarError::InvalidShape {
            reason: format!(
                "eps {} exceeds max 0.01 (model-metadata-bounds-v1)",
                config.eps
            ),
        });
    }
    Ok(())
}

/// Check a `usize` field against its contract maximum.
fn check_usize_max(value: usize, max: usize, field: &str) -> Result<()> {
    if value > max {
        return Err(RealizarError::InvalidShape {
            reason: format!("{field} {value} exceeds max {max} (model-metadata-bounds-v1)"),
        });
    }
    Ok(())
}

include!("config_validated.rs");
