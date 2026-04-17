//! GGUF metadata key schema — GH-322
//!
//! Central definition of all GGUF metadata keys used by realizar.
//! All GGUF key construction MUST go through this module.
//!
//! Architecture-parameterized keys follow the GGUF spec pattern:
//!   `{arch}.{suffix}` where `arch` = `general.architecture` value
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

// ─── Fixed keys (no architecture prefix) ─────────────────────────────────────

/// `general.architecture` — the model architecture string
pub(crate) const GENERAL_ARCHITECTURE: &str = "general.architecture";

/// `tokenizer.ggml.model` — tokenizer type (e.g. "gpt2", "llama", "bpe")
pub(crate) const TOKENIZER_MODEL: &str = "tokenizer.ggml.model";

/// `tokenizer.ggml.tokens` — vocabulary token array
pub(crate) const TOKENIZER_TOKENS: &str = "tokenizer.ggml.tokens";

/// `tokenizer.ggml.bos_token_id` — beginning-of-sentence token ID
pub(crate) const TOKENIZER_BOS_ID: &str = "tokenizer.ggml.bos_token_id";

/// `tokenizer.ggml.eos_token_id` — end-of-sentence token ID
pub(crate) const TOKENIZER_EOS_ID: &str = "tokenizer.ggml.eos_token_id";

/// `tokenizer.ggml.add_bos_token` — whether to prepend BOS token
pub(crate) const TOKENIZER_ADD_BOS: &str = "tokenizer.ggml.add_bos_token";

/// `tokenizer.ggml.vocab_size` — vocabulary size (alternative location)
pub(crate) const TOKENIZER_VOCAB_SIZE: &str = "tokenizer.ggml.vocab_size";

/// `tokenizer.ggml.tokens.size` — vocabulary size (test factory location)
#[allow(dead_code)]
pub(crate) const TOKENIZER_TOKENS_SIZE: &str = "tokenizer.ggml.tokens.size";

// ─── Architecture-parameterized key suffixes ─────────────────────────────────
//
// Use with `arch_key(arch, SUFFIX)` to get the full `{arch}.{suffix}` key.

/// `{arch}.embedding_length` — hidden dimension / embedding size
pub(crate) const EMBEDDING_LENGTH: &str = "embedding_length";

/// `{arch}.block_count` — number of transformer layers
pub(crate) const BLOCK_COUNT: &str = "block_count";

/// `{arch}.attention.head_count` — number of attention heads
pub(crate) const ATTENTION_HEAD_COUNT: &str = "attention.head_count";

/// `{arch}.attention.head_count_kv` — number of key-value heads (GQA)
pub(crate) const ATTENTION_HEAD_COUNT_KV: &str = "attention.head_count_kv";

/// `{arch}.attention.key_length` — per-head key dimension
pub(crate) const ATTENTION_KEY_LENGTH: &str = "attention.key_length";

/// `{arch}.attention.value_length` — per-head value dimension
pub(crate) const ATTENTION_VALUE_LENGTH: &str = "attention.value_length";

/// `{arch}.attention.layer_norm_rms_epsilon` — RMSNorm epsilon
pub(crate) const ATTENTION_LAYER_NORM_RMS_EPSILON: &str = "attention.layer_norm_rms_epsilon";

/// `{arch}.attention.layer_norm_epsilon` — LayerNorm epsilon (GPT-2/Phi-2)
pub(crate) const ATTENTION_LAYER_NORM_EPSILON: &str = "attention.layer_norm_epsilon";

/// `{arch}.context_length` — maximum sequence length
pub(crate) const CONTEXT_LENGTH: &str = "context_length";

/// `{arch}.rope.freq_base` — RoPE frequency base (LLaMA: 10000, Qwen2: 1000000)
pub(crate) const ROPE_FREQ_BASE: &str = "rope.freq_base";

/// `{arch}.rope.scaling.type` — RoPE scaling type (none, linear, yarn, neox)
pub(crate) const ROPE_SCALING_TYPE: &str = "rope.scaling.type";

/// `{arch}.feed_forward_length` — FFN intermediate dimension
pub(crate) const FEED_FORWARD_LENGTH: &str = "feed_forward_length";

/// `{arch}.vocab_size` — vocabulary size (architecture-specific location)
pub(crate) const VOCAB_SIZE: &str = "vocab_size";

// ─── Key construction ────────────────────────────────────────────────────────

/// Construct an architecture-parameterized GGUF metadata key.
///
/// Returns `"{arch}.{suffix}"` where `suffix` is one of the constants above.
///
/// # Examples
///
/// ```rust,ignore
/// use realizar::gguf::keys;
/// let key = keys::arch_key("llama", keys::EMBEDDING_LENGTH);
/// assert_eq!(key, "llama.embedding_length");
/// ```
#[must_use]
pub(crate) fn arch_key(arch: &str, suffix: &str) -> String {
    format!("{arch}.{suffix}")
}
