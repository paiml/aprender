//! Model slot — tracks which model is loaded in Banco.
//!
//! Phase 2a: metadata-only (path, format, size, loaded_at).
//! Phase 2b: GGUF metadata extraction via realizar (behind `inference` feature).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Detected model format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelFormat {
    Gguf,
    Apr,
    SafeTensors,
    Unknown,
}

impl ModelFormat {
    /// Detect format from file extension.
    #[must_use]
    pub fn from_path(path: &Path) -> Self {
        match path.extension().and_then(|e| e.to_str()) {
            Some("gguf") => Self::Gguf,
            Some("apr") => Self::Apr,
            Some("safetensors") => Self::SafeTensors,
            _ => Self::Unknown,
        }
    }
}

/// Metadata about a loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSlotInfo {
    pub model_id: String,
    pub path: String,
    pub format: ModelFormat,
    pub size_bytes: u64,
    pub loaded_at_secs: u64,
    /// Architecture name (e.g., "llama", "phi2", "qwen2"). Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    /// Vocabulary size. Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocab_size: Option<usize>,
    /// Hidden dimension. Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hidden_dim: Option<usize>,
    /// Number of transformer layers. Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_layers: Option<usize>,
    /// Context length. Available when inference feature enabled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_length: Option<usize>,
    /// Number of tensors in the model file.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tensor_count: Option<usize>,
}

/// Model slot — holds the currently loaded model (or None).
pub struct ModelSlot {
    info: RwLock<Option<ModelSlotInfo>>,
    loaded_at: RwLock<Option<Instant>>,
    /// The actual quantized model for inference (behind inference feature).
    #[cfg(feature = "realizar")]
    quantized_model: RwLock<Option<Arc<realizar::gguf::OwnedQuantizedModel>>>,
    /// Vocabulary tokens for encoding/decoding.
    #[cfg(feature = "realizar")]
    vocab: RwLock<Vec<String>>,
    /// Proper BPE tokenizer from aprender (behind ml feature).
    /// Uses merge rules for correct tokenization instead of greedy longest-match.
    #[cfg(feature = "aprender")]
    bpe_tokenizer: RwLock<Option<aprender::text::bpe::BpeTokenizer>>,
}

impl ModelSlot {
    /// Create an empty slot.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            info: RwLock::new(None),
            loaded_at: RwLock::new(None),
            #[cfg(feature = "realizar")]
            quantized_model: RwLock::new(None),
            #[cfg(feature = "realizar")]
            vocab: RwLock::new(Vec::new()),
            #[cfg(feature = "aprender")]
            bpe_tokenizer: RwLock::new(None),
        }
    }

    /// Load a model from a path.
    ///
    /// With `inference` feature: parses GGUF metadata (architecture, vocab, layers).
    /// Without: records file metadata only.
    pub fn load(&self, path: &str) -> Result<ModelSlotInfo, ModelSlotError> {
        let pb = PathBuf::from(path);

        let model_id = pb.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown").to_string();
        let format = ModelFormat::from_path(&pb);
        let size_bytes = std::fs::metadata(&pb).map(|m| m.len()).unwrap_or(0);

        // Extract model metadata + quantized model when inference feature is available
        let gguf_meta = extract_model_metadata(&pb, format);

        let info = ModelSlotInfo {
            model_id,
            path: path.to_string(),
            format,
            size_bytes,
            loaded_at_secs: epoch_secs(),
            architecture: gguf_meta.as_ref().map(|m| m.architecture.clone()),
            vocab_size: gguf_meta.as_ref().map(|m| m.vocab_size),
            hidden_dim: gguf_meta.as_ref().map(|m| m.hidden_dim),
            num_layers: gguf_meta.as_ref().map(|m| m.num_layers),
            context_length: gguf_meta.as_ref().map(|m| m.context_length),
            tensor_count: gguf_meta.as_ref().map(|m| m.tensor_count),
        };

        // Store the quantized model when inference feature is enabled
        #[cfg(feature = "realizar")]
        if let Some(ref meta) = gguf_meta {
            if let Ok(mut m) = self.quantized_model.write() {
                *m = meta.model.clone();
            }
            if let Ok(mut v) = self.vocab.write() {
                *v = meta.vocab.clone();
            }
        }

        // Try to load a proper BPE tokenizer (correct merge rules vs greedy)
        #[cfg(feature = "aprender")]
        {
            let bpe = load_bpe_tokenizer(&pb);
            if let Ok(mut t) = self.bpe_tokenizer.write() {
                *t = bpe;
            }
        }

        if let Ok(mut slot) = self.info.write() {
            *slot = Some(info.clone());
        }
        if let Ok(mut t) = self.loaded_at.write() {
            *t = Some(Instant::now());
        }

        Ok(info)
    }

    /// Unload the current model, freeing the quantized model and vocabulary.
    pub fn unload(&self) -> Result<(), ModelSlotError> {
        let had_model = self.info.write().map(|mut s| s.take().is_some()).unwrap_or(false);
        if let Ok(mut t) = self.loaded_at.write() {
            *t = None;
        }
        #[cfg(feature = "realizar")]
        {
            if let Ok(mut m) = self.quantized_model.write() {
                *m = None;
            }
            if let Ok(mut v) = self.vocab.write() {
                v.clear();
            }
        }
        #[cfg(feature = "aprender")]
        {
            if let Ok(mut t) = self.bpe_tokenizer.write() {
                *t = None;
            }
        }
        if had_model {
            Ok(())
        } else {
            Err(ModelSlotError::NoModelLoaded)
        }
    }

    /// Get current model info (None if empty).
    #[must_use]
    pub fn info(&self) -> Option<ModelSlotInfo> {
        self.info.read().ok()?.clone()
    }

    /// Check if a model is loaded.
    #[must_use]
    pub fn is_loaded(&self) -> bool {
        self.info.read().map(|s| s.is_some()).unwrap_or(false)
    }

    /// Get the quantized model for inference (None if not loaded or inference feature disabled).
    #[cfg(feature = "realizar")]
    #[must_use]
    pub fn quantized_model(&self) -> Option<Arc<realizar::gguf::OwnedQuantizedModel>> {
        self.quantized_model.read().ok()?.clone()
    }

    /// Get the vocabulary tokens.
    #[cfg(feature = "realizar")]
    #[must_use]
    pub fn vocabulary(&self) -> Vec<String> {
        self.vocab.read().map(|v| v.clone()).unwrap_or_default()
    }

    /// Check if inference-capable model is loaded (not just metadata).
    #[cfg(feature = "realizar")]
    #[must_use]
    pub fn has_inference_model(&self) -> bool {
        self.quantized_model.read().map(|m| m.is_some()).unwrap_or(false)
    }

    /// Encode text to token IDs using proper BPE when available, else greedy fallback.
    ///
    /// Priority: BPE tokenizer (correct merge rules) → greedy longest-match (approximate).
    #[cfg(feature = "realizar")]
    #[must_use]
    pub fn encode_text(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // Try BPE tokenizer first (correct tokenization)
        #[cfg(feature = "aprender")]
        if let Ok(guard) = self.bpe_tokenizer.read() {
            if let Some(ref bpe) = *guard {
                return bpe.encode(text);
            }
        }

        // Fall back to greedy longest-match (approximate)
        let vocab = self.vocabulary();
        super::inference::encode_prompt(&vocab, text)
    }

    /// Check if a proper BPE tokenizer is loaded (not just greedy fallback).
    #[cfg(feature = "aprender")]
    #[must_use]
    pub fn has_bpe_tokenizer(&self) -> bool {
        self.bpe_tokenizer.read().map(|t| t.is_some()).unwrap_or(false)
    }

    /// How long the model has been loaded.
    #[must_use]
    pub fn uptime_secs(&self) -> u64 {
        self.loaded_at.read().ok().and_then(|t| t.map(|i| i.elapsed().as_secs())).unwrap_or(0)
    }
}

/// Model slot errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelSlotError {
    NoModelLoaded,
}

impl std::fmt::Display for ModelSlotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoModelLoaded => write!(f, "No model loaded"),
        }
    }
}

impl std::error::Error for ModelSlotError {}

fn epoch_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// Extracted GGUF metadata (architecture, vocab, etc.).
struct GgufMeta {
    architecture: String,
    vocab_size: usize,
    hidden_dim: usize,
    num_layers: usize,
    context_length: usize,
    tensor_count: usize,
    /// The quantized model for inference (only with inference feature).
    #[cfg(feature = "realizar")]
    model: Option<Arc<realizar::gguf::OwnedQuantizedModel>>,
    /// Vocabulary tokens.
    #[cfg(feature = "realizar")]
    vocab: Vec<String>,
}

/// Extract model metadata + quantized model from GGUF or APR file.
#[cfg(feature = "realizar")]
fn extract_model_metadata(path: &Path, format: ModelFormat) -> Option<GgufMeta> {
    match format {
        ModelFormat::Gguf => extract_gguf_metadata(path),
        ModelFormat::Apr => extract_apr_metadata(path),
        _ => None,
    }
}

/// Extract GGUF metadata + model from a .gguf file.
#[cfg(feature = "realizar")]
fn extract_gguf_metadata(path: &Path) -> Option<GgufMeta> {
    // Memory-map for efficient loading
    let mapped = realizar::gguf::MappedGGUFModel::from_path(path.to_str()?).ok()?;
    let config = realizar::gguf::GGUFConfig::from_gguf(&mapped.model).ok()?;

    // Extract vocabulary
    let vocab = mapped
        .model
        .vocabulary()
        .unwrap_or_else(|| (0..config.vocab_size).map(|i| format!("token{i}")).collect());

    // Build quantized model for inference
    let quantized = match realizar::gguf::OwnedQuantizedModel::from_mapped(&mapped) {
        Ok(m) => {
            eprintln!("[banco] Quantized model loaded successfully");
            Some(m)
        }
        Err(e) => {
            eprintln!("[banco] WARNING: Failed to build quantized model: {e}");
            eprintln!("[banco] Metadata available but inference disabled for this model");
            None
        }
    };

    Some(GgufMeta {
        architecture: config.architecture.clone(),
        vocab_size: config.vocab_size,
        hidden_dim: config.hidden_dim,
        num_layers: config.num_layers,
        context_length: config.context_length,
        tensor_count: mapped.model.tensors.len(),
        model: quantized.map(Arc::new),
        vocab,
    })
}

/// Extract APR metadata + model from a .apr file.
#[cfg(feature = "realizar")]
fn extract_apr_metadata(path: &Path) -> Option<GgufMeta> {
    let apr = realizar::apr::MappedAprModel::from_path(path).ok()?;

    let meta = &apr.metadata;
    let architecture = meta.architecture.clone().unwrap_or_else(|| "unknown".to_string());
    let hidden_dim = meta.hidden_size.unwrap_or(0);
    let num_layers = meta.num_layers.unwrap_or(0);
    let vocab_size = meta.vocab_size.unwrap_or(0);
    let context_length = meta.max_position_embeddings.unwrap_or(2048);
    let tensor_count = apr.tensor_count();

    eprintln!(
        "[banco] APR model: {architecture} | {num_layers} layers | {hidden_dim}d | {vocab_size} vocab | {tensor_count} tensors"
    );

    // Build vocabulary from APR metadata or generate placeholders
    let vocab: Vec<String> = if vocab_size > 0 {
        (0..vocab_size).map(|i| format!("token{i}")).collect()
    } else {
        Vec::new()
    };

    // Build quantized model for inference via realizar
    let quantized = match realizar::gguf::OwnedQuantizedModel::from_apr(&apr) {
        Ok(m) => {
            eprintln!("[banco] APR quantized model loaded successfully");
            Some(m)
        }
        Err(e) => {
            eprintln!("[banco] WARNING: Failed to build quantized model from APR: {e}");
            None
        }
    };

    Some(GgufMeta {
        architecture,
        vocab_size,
        hidden_dim,
        num_layers,
        context_length,
        tensor_count,
        model: quantized.map(Arc::new),
        vocab,
    })
}

/// Stub when inference feature is not enabled.
#[cfg(not(feature = "realizar"))]
fn extract_model_metadata(_path: &Path, _format: ModelFormat) -> Option<GgufMeta> {
    None
}

/// Load a proper BPE tokenizer for a model file.
///
/// Search order (same as apr-cli):
/// 1. Sibling `{stem}.tokenizer.json` (e.g., `model.tokenizer.json`)
/// 2. `tokenizer.json` in the same directory
///
/// Returns `None` if no tokenizer.json found — caller falls back to greedy.
#[cfg(feature = "aprender")]
fn load_bpe_tokenizer(model_path: &Path) -> Option<aprender::text::bpe::BpeTokenizer> {
    use aprender::text::bpe::BpeTokenizer;

    // 1. Sibling {stem}.tokenizer.json
    let stem = model_path.file_stem()?.to_string_lossy();
    let sibling = model_path.with_file_name(format!("{stem}.tokenizer.json"));
    if sibling.exists() {
        match BpeTokenizer::from_huggingface(&sibling) {
            Ok(tok) => {
                eprintln!("[banco] BPE tokenizer loaded from {}", sibling.display());
                return Some(tok);
            }
            Err(e) => {
                eprintln!(
                    "[banco] WARNING: Failed to load tokenizer from {}: {e}",
                    sibling.display()
                );
            }
        }
    }

    // 2. tokenizer.json in same directory
    if let Some(parent) = model_path.parent() {
        let tokenizer_json = parent.join("tokenizer.json");
        if tokenizer_json.exists() {
            match BpeTokenizer::from_huggingface(&tokenizer_json) {
                Ok(tok) => {
                    eprintln!("[banco] BPE tokenizer loaded from {}", tokenizer_json.display());
                    return Some(tok);
                }
                Err(e) => {
                    eprintln!(
                        "[banco] WARNING: Failed to load tokenizer from {}: {e}",
                        tokenizer_json.display()
                    );
                }
            }
        }
    }

    eprintln!(
        "[banco] No tokenizer.json found for '{}' — using greedy tokenization",
        model_path.display()
    );
    None
}
