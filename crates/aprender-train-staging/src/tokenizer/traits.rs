//! Tokenizer trait definition.

use super::error::Result;

/// Token ID type
pub type TokenId = u32;

/// Tokenizer trait
pub trait Tokenizer: Send + Sync {
    /// Train the tokenizer on a corpus
    fn train(&mut self, corpus: &[&str]) -> Result<()>;

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> Result<Vec<TokenId>>;

    /// Decode token IDs to text
    fn decode(&self, ids: &[TokenId]) -> Result<String>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Check if tokenizer is trained
    fn is_trained(&self) -> bool;

    /// Get token for ID
    fn id_to_token(&self, id: TokenId) -> Option<&str>;

    /// Get ID for token
    fn token_to_id(&self, token: &str) -> Option<TokenId>;
}
