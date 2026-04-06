
/// Model metadata from .apr file
///
/// # Schema Resilience (PMAT-111)
///
/// This struct uses serde aliases to accept multiple field name variations
/// commonly found in different model formats (HuggingFace, GGUF, etc.):
///
/// - `hidden_size` also accepts: `hidden_dim`, `d_model`, `n_embd`
/// - `num_layers` also accepts: `n_layers`, `num_hidden_layers`, `n_layer`
/// - `num_heads` also accepts: `n_heads`, `num_attention_heads`, `n_head`
/// - `vocab_size` also accepts: `n_vocab`
/// - `intermediate_size` also accepts: `ffn_dim`, `intermediate_dim`, `n_inner`
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AprMetadata {
    /// Model type (e.g., "transformer_lm", "whisper", "llama")
    #[serde(default)]
    pub model_type: Option<String>,
    /// Human-readable model name
    #[serde(default)]
    pub name: Option<String>,
    /// Model architecture family
    #[serde(default)]
    pub architecture: Option<String>,
    /// Hidden dimension size
    /// Aliases: hidden_dim, d_model, n_embd (PMAT-111 schema resilience)
    #[serde(default, alias = "hidden_dim", alias = "d_model", alias = "n_embd")]
    pub hidden_size: Option<usize>,
    /// Number of transformer layers
    /// Aliases: n_layers, num_hidden_layers, n_layer (PMAT-111 schema resilience)
    #[serde(
        default,
        alias = "n_layers",
        alias = "num_hidden_layers",
        alias = "n_layer"
    )]
    pub num_layers: Option<usize>,
    /// Number of attention heads
    /// Aliases: n_heads, num_attention_heads, n_head (PMAT-111 schema resilience)
    #[serde(
        default,
        alias = "n_heads",
        alias = "num_attention_heads",
        alias = "n_head"
    )]
    pub num_heads: Option<usize>,
    /// Number of key-value heads (for GQA, defaults to num_heads)
    /// Aliases: n_kv_heads, num_key_value_heads (PMAT-111 schema resilience)
    #[serde(
        default,
        alias = "n_kv_heads",
        alias = "num_key_value_heads"
    )]
    pub num_kv_heads: Option<usize>,
    /// Vocabulary size
    /// Aliases: n_vocab (PMAT-111 schema resilience)
    #[serde(default, alias = "n_vocab")]
    pub vocab_size: Option<usize>,
    /// FFN intermediate dimension
    /// Aliases: ffn_dim, intermediate_dim, n_inner (PMAT-111 schema resilience)
    #[serde(
        default,
        alias = "ffn_dim",
        alias = "intermediate_dim",
        alias = "n_inner"
    )]
    pub intermediate_size: Option<usize>,
    /// Maximum context/sequence length
    /// Aliases: max_seq_len, context_length, n_ctx (PMAT-111 schema resilience)
    #[serde(
        default,
        alias = "max_seq_len",
        alias = "context_length",
        alias = "n_ctx"
    )]
    pub max_position_embeddings: Option<usize>,
    /// RoPE theta for position encoding
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// RoPE type: 0=NORM (adjacent pairs), 2=NEOX (split halves)
    /// CORRECTNESS-011: Qwen2.5 models require rope_type=2 (NEOX style)
    #[serde(default)]
    pub rope_type: Option<u32>,
    /// Layer norm epsilon
    /// Aliases: layer_norm_eps, norm_eps (PMAT-111 schema resilience)
    #[serde(default, alias = "layer_norm_eps", alias = "norm_eps")]
    pub rms_norm_eps: Option<f32>,
    /// Additional metadata fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl AprMetadata {
    /// Check if this model has transformer configuration
    #[must_use]
    pub fn is_transformer(&self) -> bool {
        self.hidden_size.is_some()
            && self.num_layers.is_some()
            && self.num_heads.is_some()
            && self.vocab_size.is_some()
    }

    /// Extract embedded tokenizer vocabulary from APR metadata (GH-156)
    ///
    /// APR files can contain embedded tokenizer data in the metadata section.
    /// This is the preferred way to decode tokens - no sibling files needed.
    ///
    /// # Returns
    /// - `Some(vocab)` if tokenizer.vocabulary is present in metadata
    /// - `None` if no embedded tokenizer
    #[must_use]
    pub fn get_embedded_vocabulary(&self) -> Option<Vec<String>> {
        let vocab_value = self.extra.get("tokenizer.vocabulary")?;
        let vocab_array = vocab_value.as_array()?;

        let vocab: Vec<String> = vocab_array
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        if vocab.is_empty() {
            None
        } else {
            Some(vocab)
        }
    }

    /// Get embedded BOS token ID from APR metadata
    #[must_use]
    pub fn get_embedded_bos_token_id(&self) -> Option<u32> {
        self.extra
            .get("tokenizer.bos_token_id")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32)
    }

    /// Get embedded EOS token ID from APR metadata
    #[must_use]
    pub fn get_embedded_eos_token_id(&self) -> Option<u32> {
        self.extra
            .get("tokenizer.eos_token_id")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32)
    }

    /// Extract embedded BPE merge rules from APR metadata (PMAT-171)
    ///
    /// APR files converted from GGUF can contain embedded BPE merge rules
    /// for standalone encoding - no sibling tokenizer.json needed.
    ///
    /// # Returns
    /// - `Some(merges)` if tokenizer.merges is present in metadata
    /// - `None` if no embedded merges
    #[must_use]
    pub fn get_embedded_merges(&self) -> Option<Vec<(String, String)>> {
        let merges_value = self.extra.get("tokenizer.merges")?;
        let merges_array = merges_value.as_array()?;

        let merges: Vec<(String, String)> = merges_array
            .iter()
            .filter_map(|v| {
                let s = v.as_str()?;
                let parts: Vec<&str> = s.splitn(2, ' ').collect();
                if parts.len() == 2 {
                    Some((parts[0].to_string(), parts[1].to_string()))
                } else {
                    None
                }
            })
            .collect();

        if merges.is_empty() {
            None
        } else {
            Some(merges)
        }
    }

    /// GH-366: Extract embedded SentencePiece scores from APR metadata
    ///
    /// Returns token log-probability scores for Unigram/Viterbi encoding.
    /// Parallel to vocabulary — scores[i] is the score for vocabulary[i].
    #[must_use]
    pub fn get_embedded_scores(&self) -> Option<Vec<f32>> {
        let scores_value = self.extra.get("tokenizer.scores")?;
        let scores_array = scores_value.as_array()?;

        let scores: Vec<f32> = scores_array
            .iter()
            .filter_map(|v| v.as_f64().map(|f| f as f32))
            .collect();

        if scores.is_empty() {
            None
        } else {
            Some(scores)
        }
    }

    /// GH-366: Get embedded tokenizer model type
    #[must_use]
    pub fn get_embedded_model_type(&self) -> Option<String> {
        self.extra
            .get("tokenizer.model_type")
            .and_then(serde_json::Value::as_str)
            .map(String::from)
    }
}

/// APR v2 model for realizar inference
///
/// # Memory Management
///
/// Uses memory-mapped I/O for uncompressed files to avoid zram pressure.
/// After loading tensors to GPU, call `release_cpu_pages()` to advise
/// the kernel that pages can be dropped (re-faulted from disk if needed).
///
/// # References
///
/// - Didona et al. (2022): mmap vs read() performance
/// - See docs/model-loading.md for full design rationale
#[derive(Debug)]
pub struct AprV2Model {
    /// Header information
    header: AprHeader,
    /// Model metadata
    metadata: AprMetadata,
    /// Tensor index
    tensors: Vec<TensorEntry>,
    /// Raw file data (mmap for uncompressed, heap for compressed)
    data: ModelData,
}
