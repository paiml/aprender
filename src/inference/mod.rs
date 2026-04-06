//! End-to-end LLM inference engine.
//!
//! Composes trueno's compute primitives (Q4K matmul, RMS norm, fused attention,
//! SIMD softmax) into a complete transformer that loads GGUF models and generates text.
//!
//! # Example
//!
//! ```rust,ignore
//! use trueno::inference::{GgufFile, LlamaModel, generate, SampleParams};
//!
//! let gguf = GgufFile::load(Path::new("model.gguf"))?;
//! let model = LlamaModel::from_gguf(&gguf)?;
//! let tokens = generate(&model, &[1], 100, &SampleParams::default(), 2)?;
//! ```

pub mod generate;
pub mod gguf;
pub mod model;

pub use generate::{generate, SampleParams};
pub use gguf::GgufFile;
pub use model::{ForwardArena, KvCache, LlamaModel, ModelConfig};
