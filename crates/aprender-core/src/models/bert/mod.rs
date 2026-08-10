//! BERT encoder + cross-encoder reranking (GH-326).
//!
//! Sovereign-stack BERT inference for cross-encoder reranking models like
//! `BAAI/bge-reranker-base` and `cross-encoder/ms-marco-MiniLM-L-6-v2`.
//! No ONNX Runtime dependency — pure Rust + trueno SIMD.
//!
//! # Architecture
//!
//! ```text
//! Input:  [CLS] query_tokens [SEP] passage_tokens [SEP]
//!          ↓
//!       BertEmbeddings (word + position + token_type → LayerNorm + Dropout)
//!          ↓
//!       BertEncoder (N × BertLayer, post-norm: attention → FFN)
//!          ↓
//!       CLS pooling (extract [CLS] hidden state)
//!          ↓
//!       Linear classifier head (hidden_dim → 1) → sigmoid
//!          ↓
//!       Relevance score ∈ [0, 1]
//! ```
//!
//! # Scope
//!
//! - **In scope**: encoder forward pass, cross-encoder scoring API,
//!   dimension-correctness tests.
//! - **Out of scope** (separate tickets): HuggingFace-parity numerical
//!   validation (requires reference activations), training, decoder-only
//!   variants, batched cross-encoder scoring optimization.

pub mod config;
pub mod cross_encoder;
pub mod embeddings;
pub mod encoder;
pub mod layer;
pub mod load;

pub use config::{BertConfig, BertConfigError};
pub use cross_encoder::CrossEncoder;
pub use embeddings::BertEmbeddings;
pub use encoder::BertEncoder;
pub use layer::BertLayer;
pub use load::{expected_bert_tensor_names, BertLoadError};
