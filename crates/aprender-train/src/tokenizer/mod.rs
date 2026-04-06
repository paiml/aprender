//! Subword Tokenization Module (#26)
//!
//! Just-in-Time tokenization for training pipelines with BPE and WordPiece support.
//! Includes integration with aprender for HuggingFace-compatible tokenizer loading.
//!
//! # Toyota Principle: Just-in-Time (ジャスト・イン・タイム)
//!
//! Tokenize on demand during training, not upfront - reducing memory footprint
//! and enabling dynamic vocabulary adaptation.
//!
//! # Example
//!
//! ```
//! use entrenar::tokenizer::{BPETokenizer, Tokenizer, TokenizerConfig};
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create a BPE tokenizer
//!     let config = TokenizerConfig::bpe().with_vocab_size(1000);
//!     let mut tokenizer = BPETokenizer::new(config);
//!
//!     // Train on corpus
//!     let corpus = vec!["hello world", "hello there"];
//!     tokenizer.train(&corpus)?;
//!
//!     // Tokenize text
//!     let tokens = tokenizer.encode("hello world")?;
//!     let decoded = tokenizer.decode(&tokens)?;
//!     Ok(())
//! }
//! ```
//!
//! # HuggingFace Integration
//!
//! Load pre-trained tokenizers from HuggingFace tokenizer.json files:
//!
//! ```rust,ignore
//! use entrenar::tokenizer::HfTokenizer;
//!
//! fn example() -> Result<(), Box<dyn std::error::Error>> {
//!     // Load from HuggingFace tokenizer.json
//!     let tokenizer = HfTokenizer::from_file("path/to/tokenizer.json")?;
//!     let tokens = tokenizer.encode("Hello, world!");
//!     Ok(())
//! }
//! ```

mod bpe;
mod char;
mod config;
mod error;
mod hf;
mod traits;

// Re-export all public types for API compatibility
pub use bpe::BPETokenizer;
pub use char::CharTokenizer;
pub use config::{SpecialTokens, TokenizerConfig, TokenizerType};
pub use error::{Result, TokenizerError};
pub use hf::{
    bytes_to_unicode, load_hf_from_files, load_hf_from_json, HfBpeConfig, HfBpeTokenizer,
    HfTokenizer, MergeRule, Qwen2BpeTokenizer,
};
pub use traits::{TokenId, Tokenizer};
