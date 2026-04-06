//! Data endpoint types (tokenize, detokenize, embeddings).

use serde::{Deserialize, Serialize};

/// Tokenize request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeRequest {
    pub text: String,
}

/// Tokenize response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizeResponse {
    pub tokens: Vec<u32>,
    pub count: u32,
}

/// Detokenize request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetokenizeRequest {
    pub tokens: Vec<u32>,
}

/// Detokenize response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetokenizeResponse {
    pub text: String,
}

/// OpenAI-compatible embeddings request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub input: EmbeddingsInput,
}

/// Embeddings input — single string or array of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingsInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingsInput {
    pub fn texts(&self) -> Vec<&str> {
        match self {
            Self::Single(s) => vec![s.as_str()],
            Self::Batch(v) => v.iter().map(String::as_str).collect(),
        }
    }
}

/// OpenAI-compatible embeddings response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingsUsage,
}

/// A single embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: u32,
    pub embedding: Vec<f32>,
}

/// Token usage for embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingsUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}
