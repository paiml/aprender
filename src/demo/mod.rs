//! End-to-End Demo Module
//!
//! Provides verification infrastructure for the Qwen2-0.5B WASM demo.
//!
//! # QA Verification (Section J: 15 points)
//!
//! - J1: Qwen2-0.5B imports from HF
//! - J2: INT4 quantization completes
//! - J3: Quantized perplexity <15% degradation
//! - J4: WASM compilation succeeds
//! - J5: Browser loads model <5s
//! - J6-J15: See tests below
//!
//! # Reference Model
//!
//! Qwen2-0.5B-Instruct (Apache 2.0):
//! - Parameters: 0.5B
//! - INT4 Size: ~300MB
//! - Context: 32K tokens
//! - HF: Qwen/Qwen2-0.5B-Instruct
//!
//! # References
//!
//! - Bai et al. (2023). "Qwen Technical Report"
//! - HuggingFace Transformers Documentation

pub mod reliable;

/// Model configuration for Qwen2-0.5B-Instruct
#[derive(Debug, Clone)]
pub struct Qwen2Config {
    /// Hidden size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_kv_heads: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Intermediate size (FFN)
    pub intermediate_size: usize,
    /// `RoPE` theta
    pub rope_theta: f64,
}

impl Default for Qwen2Config {
    fn default() -> Self {
        Self::qwen2_0_5b_instruct()
    }
}

impl Qwen2Config {
    /// Configuration for Qwen2-0.5B-Instruct
    #[must_use]
    pub fn qwen2_0_5b_instruct() -> Self {
        Self {
            hidden_size: 896,
            num_attention_heads: 14,
            num_kv_heads: 2,
            num_layers: 24,
            vocab_size: 151936,
            max_seq_len: 32768,
            intermediate_size: 4864,
            rope_theta: 1_000_000.0,
        }
    }

    /// Configuration for Qwen2.5-Coder-0.5B-Instruct
    ///
    /// Same architecture as Qwen2-0.5B-Instruct (shared base model).
    /// Both use: 896 hidden, 14 heads, 2 KV heads, 24 layers, 151936 vocab.
    #[must_use]
    pub fn qwen25_coder_0_5b_instruct() -> Self {
        // Qwen2.5-Coder shares architecture with Qwen2-0.5B
        Self::qwen2_0_5b_instruct()
    }

    /// Calculate model size in bytes (FP16)
    #[must_use]
    pub fn model_size_fp16(&self) -> usize {
        // Rough estimate: embeddings + layers + lm_head
        let embedding_size = self.vocab_size * self.hidden_size * 2; // FP16
        let layer_size = self.hidden_size * self.hidden_size * 4 * 2; // QKV + O
        let ffn_size = self.hidden_size * self.intermediate_size * 3 * 2; // up, gate, down
        let total_layers = (layer_size + ffn_size) * self.num_layers;
        let lm_head = self.vocab_size * self.hidden_size * 2;

        embedding_size + total_layers + lm_head
    }

    /// Calculate model size in bytes (INT4)
    #[must_use]
    pub fn model_size_int4(&self) -> usize {
        // INT4 is ~4x smaller than FP16 for weights
        self.model_size_fp16() / 4
    }

    /// Estimate KV cache size for a given sequence length
    #[must_use]
    pub fn kv_cache_size(&self, seq_len: usize) -> usize {
        // KV cache: 2 * num_layers * num_kv_heads * seq_len * head_dim * 2 (FP16)
        let head_dim = self.hidden_size / self.num_attention_heads;
        2 * self.num_layers * self.num_kv_heads * seq_len * head_dim * 2
    }
}

/// Tokenizer configuration for Qwen2
#[derive(Debug, Clone)]
pub struct Qwen2Tokenizer {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Special tokens
    pub special_tokens: SpecialTokens,
}

/// Special tokens for instruction format
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning of sequence
    pub bos_id: u32,
    /// End of sequence
    pub eos_id: u32,
    /// Padding token
    pub pad_id: u32,
    /// Start of turn
    pub im_start_id: u32,
    /// End of turn
    pub im_end_id: u32,
}

impl SpecialTokens {
    /// C-09 (Meyer DbC): Qwen2-specific token IDs — NOT a generic default.
    /// These are the correct IDs for Qwen2/Qwen2.5 models only.
    /// Other architectures (LLaMA, Mistral, Phi) have different special token IDs.
    #[must_use]
    pub fn qwen2() -> Self {
        Self {
            bos_id: 151643,
            eos_id: 151645,
            pad_id: 151643,
            im_start_id: 151644,
            im_end_id: 151645,
        }
    }
}

// N-08 (Meyer DbC): Default impl REMOVED — callers must explicitly use
// SpecialTokens::qwen2() or construct from model metadata. The Default trait
// violated the class invariant for non-Qwen2 models.

impl Qwen2Tokenizer {
    /// Create tokenizer with Qwen2 configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            vocab_size: 151936,
            special_tokens: SpecialTokens::qwen2(),
        }
    }

    /// Check if token is EOS
    #[must_use]
    pub fn is_eos(&self, token_id: u32) -> bool {
        token_id == self.special_tokens.eos_id || token_id == self.special_tokens.im_end_id
    }

    /// Check if token is special
    #[must_use]
    pub fn is_special(&self, token_id: u32) -> bool {
        token_id == self.special_tokens.bos_id
            || token_id == self.special_tokens.eos_id
            || token_id == self.special_tokens.pad_id
            || token_id == self.special_tokens.im_start_id
            || token_id == self.special_tokens.im_end_id
    }

    /// Format instruction prompt
    #[must_use]
    pub fn format_instruction(&self, instruction: &str) -> String {
        format!("<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n")
    }
}

impl Default for Qwen2Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Demo metrics for verification
#[derive(Debug, Clone, Default)]
pub struct DemoMetrics {
    /// Model load time in milliseconds
    pub load_time_ms: u64,
    /// First token latency in milliseconds
    pub first_token_ms: u64,
    /// Tokens per second (sustained)
    pub tokens_per_sec: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Total tokens generated
    pub tokens_generated: usize,
}

impl DemoMetrics {
    /// Check if metrics meet performance targets
    #[must_use]
    pub fn meets_targets(&self) -> bool {
        self.load_time_ms < 5000
            && self.first_token_ms < 2000
            && self.tokens_per_sec >= 15.0
            && self.peak_memory_bytes < 512 * 1024 * 1024
    }
}

/// Quantization configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationType {
    /// 4-bit integer quantization
    Int4,
    /// 8-bit integer quantization
    Int8,
    /// 16-bit floating point
    Fp16,
    /// 32-bit floating point
    Fp32,
}

impl QuantizationType {
    /// Bits per weight
    #[must_use]
    pub fn bits(&self) -> usize {
        match self {
            Self::Int4 => 4,
            Self::Int8 => 8,
            Self::Fp16 => 16,
            Self::Fp32 => 32,
        }
    }

    /// Compression ratio vs FP32
    #[must_use]
    pub fn compression_ratio(&self) -> f64 {
        32.0 / self.bits() as f64
    }
}

/// Perplexity degradation checker
#[derive(Debug)]
pub struct PerplexityChecker {
    /// Baseline perplexity (FP16)
    pub baseline_ppl: f64,
    /// Maximum allowed degradation (percentage)
    pub max_degradation_pct: f64,
}

impl PerplexityChecker {
    /// Create checker with 15% max degradation
    #[must_use]
    pub fn new(baseline_ppl: f64) -> Self {
        Self {
            baseline_ppl,
            max_degradation_pct: 15.0,
        }
    }

    /// Check if quantized perplexity is acceptable
    #[must_use]
    pub fn is_acceptable(&self, quantized_ppl: f64) -> bool {
        let degradation_pct = ((quantized_ppl - self.baseline_ppl) / self.baseline_ppl) * 100.0;
        degradation_pct <= self.max_degradation_pct
    }

    /// Calculate degradation percentage
    #[must_use]
    pub fn degradation_pct(&self, quantized_ppl: f64) -> f64 {
        ((quantized_ppl - self.baseline_ppl) / self.baseline_ppl) * 100.0
    }
}

/// Browser compatibility checker
#[derive(Debug, Clone)]
pub struct BrowserCompatibility {
    /// Chrome version requirement
    pub chrome_min: u32,
    /// Firefox version requirement
    pub firefox_min: u32,
    /// Safari version requirement
    pub safari_min: u32,
}

impl Default for BrowserCompatibility {
    fn default() -> Self {
        Self {
            chrome_min: 120,
            firefox_min: 120,
            safari_min: 17,
        }
    }
}

impl BrowserCompatibility {
    /// Check Chrome compatibility
    #[must_use]
    pub fn supports_chrome(&self, version: u32) -> bool {
        version >= self.chrome_min
    }

    /// Check Firefox compatibility
    #[must_use]
    pub fn supports_firefox(&self, version: u32) -> bool {
        version >= self.firefox_min
    }

    /// Check Safari compatibility
    #[must_use]
    pub fn supports_safari(&self, version: u32) -> bool {
        version >= self.safari_min
    }
}

#[cfg(test)]
#[path = "demo_tests.rs"]
mod tests;
