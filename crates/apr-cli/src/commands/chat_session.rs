
// =============================================================================
// ChatSession with realizar (Y13/Y14: architecture and format agnostic)
// =============================================================================

#[cfg(feature = "inference")]
mod realizar_chat {
    use super::*;
    use aprender::text::bpe::Qwen2BpeTokenizer;
    use std::fs::File;
    use std::io::Read;

    /// Chat session using realizar for high-performance inference
    /// Y13: Architecture-agnostic (detected from model metadata)
    /// Y14: Format-agnostic (APR, GGUF, SafeTensors)
    ///
    /// PMAT-108: ALL inference delegated to realizar engine.
    /// aprender::models is NOT used for inference (only training).
    pub struct ChatSession {
        /// Model bytes (kept for regeneration if needed)
        model_bytes: Vec<u8>,
        /// Model path (for mmap-based loading)
        model_path: std::path::PathBuf,
        /// Detected format
        format: ModelFormat,
        /// Conversation history as ChatMessage objects
        history: Vec<ChatMessage>,
        /// Chat template engine (Toyota Way: Standardized Work)
        chat_template: Box<dyn ChatTemplateEngine + Send + Sync>,
        /// Detected template format name (for display)
        template_format: TemplateFormat,
        /// LLaMA tokenizer (for GGUF format)
        llama_tokenizer: Option<LlamaTokenizer>,
        /// Qwen2 BPE tokenizer (for SafeTensors/APR format)
        qwen_tokenizer: Option<Qwen2BpeTokenizer>,
        /// GH-224: Cached GGUF mmap model (for tokenizer encode/decode across messages)
        cached_gguf_mapped: Option<realizar::gguf::MappedGGUFModel>,
        /// GH-224: Cached GGUF CUDA model (avoids re-uploading weights per message)
        #[cfg(feature = "cuda")]
        cached_gguf_cuda: Option<realizar::gguf::OwnedQuantizedModelCuda>,
        /// GH-224: Cached APR CUDA model (avoids re-uploading weights per message)
        #[cfg(feature = "cuda")]
        cached_apr_cuda: Option<realizar::apr::AprV2ModelCuda>,
        /// GH-224: Cached SafeTensors CUDA model (avoids re-loading per message)
        #[cfg(feature = "cuda")]
        cached_safetensors_cuda: Option<realizar::safetensors_cuda::SafeTensorsCudaModel>,
        /// GH-224: Whether CUDA init was attempted and failed (skip retries)
        #[cfg(feature = "cuda")]
        cuda_init_failed: bool,
    }

include!("chat_load_tokenizers.rs");
include!("chat_session_02.rs");
include!("chat_generate_session_02.rs");
include!("chat_generate_safetensors.rs");
}


#[cfg(feature = "inference")]
use realizar_chat::ChatSession;
