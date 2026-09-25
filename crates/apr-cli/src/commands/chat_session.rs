
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
        /// #3595: the Qwen3.5 hybrid, built once with its decode state kept across
        /// turns. GH-224's dense cache below cannot hold it, so every turn used to
        /// rebuild, re-upload and re-validate the model.
        qwen35_session: Option<realizar::gguf::qwen35_session::Qwen35Session>,
        /// #4268: a dense GGUF model on the one engine, its decode state kept
        /// across turns so a turn prefills only what the history added. The CUDA
        /// model is built at load (GH-224: no re-upload per message); the CPU one
        /// on the first turn that needs it.
        gguf_session: Option<realizar::gguf::dense_session::DenseSession>,
        /// #4268: the same for an `.apr`. #3922: its CUDA model is the fused-kernel
        /// class `run` and `bench` use, not the generic transformer that produced
        /// garbage on Q4K and refused everything else.
        apr_session: Option<realizar::gguf::dense_session::DenseSession>,
        /// GH-224: Cached SafeTensors CUDA model (avoids re-loading per message)
        #[cfg(feature = "cuda")]
        cached_safetensors_cuda: Option<realizar::safetensors_cuda::SafeTensorsCudaModel>,
        /// GH-224: Whether CUDA init was attempted and failed (skip retries)
        #[cfg(feature = "cuda")]
        cuda_init_failed: bool,
        /// #3367: set by `render_assistant_turn` when a turn failed to generate. Read
        /// once, at session end, to decide the command's exit code. Not reset by
        /// `/clear` — a failed generation happened whether or not the history is kept.
        had_generate_error: bool,
        /// #3794: set when a turn was generated on an accelerator. Read at session
        /// end so `--json` can report the backend that ACTUALLY answered, rather
        /// than the one that was asked for — `apr run --format json` already
        /// reports `{requested, ran, fell_back}` and `apr chat` reported nothing,
        /// so a harness could not hold chat to its lane the way it holds run.
        generated_on_gpu: bool,
        /// #3937: set when any turn produced an answer. A forced `--gpu` session is
        /// only refused for running on CPU if something actually ran; a session that
        /// generated nothing has no backend to reconcile.
        answered_turn: bool,
    }

include!("chat_load_tokenizers.rs");
include!("chat_session_02.rs");
include!("chat_generate_session_02.rs");
include!("chat_generate_safetensors.rs");

    #[cfg(test)]
    mod engine_identity_4263 {
        use super::*;
        include!("chat_engine_identity_4263.rs");
    }
}


#[cfg(feature = "inference")]
use realizar_chat::ChatSession;
