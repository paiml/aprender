//! HuggingFace tokenizer integration via aprender.

use super::error::{Result, TokenizerError};

/// GPT-2 vocabulary size (50257 BPE tokens, with ID 50256 used as both pad and EOS).
const GPT2_VOCAB_SIZE: u32 = 50256;

// Re-export aprender BPE types for HuggingFace compatibility
pub use aprender::text::bpe::{
    bytes_to_unicode, load_from_files as load_hf_from_files, load_from_json as load_hf_from_json,
    BpeConfig as HfBpeConfig, BpeTokenizer as HfBpeTokenizer, MergeRule, Qwen2BpeTokenizer,
};

/// HuggingFace-compatible tokenizer wrapper
///
/// Wraps aprender's BPE tokenizer to provide training batch utilities.
#[derive(Debug, Clone)]
pub struct HfTokenizer {
    inner: HfBpeTokenizer,
    pad_id: u32,
    eos_id: Option<u32>,
    bos_id: Option<u32>,
}

impl HfTokenizer {
    /// Create a GPT-2 tokenizer with base vocabulary
    #[must_use]
    pub fn gpt2() -> Self {
        Self {
            inner: HfBpeTokenizer::gpt2_base(),
            pad_id: GPT2_VOCAB_SIZE,
            eos_id: Some(GPT2_VOCAB_SIZE),
            bos_id: None,
        }
    }

    /// Create a Qwen2 tokenizer
    #[must_use]
    pub fn qwen2() -> Self {
        Self {
            inner: HfBpeTokenizer::new(HfBpeConfig::qwen2()),
            pad_id: Qwen2BpeTokenizer::ENDOFTEXT_ID,
            eos_id: Some(Qwen2BpeTokenizer::IM_END_ID),
            bos_id: Some(Qwen2BpeTokenizer::IM_START_ID),
        }
    }

    /// Load tokenizer from HuggingFace tokenizer.json file
    ///
    /// # Errors
    /// Returns error if file cannot be read or parsed.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let json = std::fs::read_to_string(path.as_ref())?;
        Self::from_json(&json)
    }

    /// Load tokenizer from JSON string
    ///
    /// # Errors
    /// Returns error if JSON parsing fails.
    pub fn from_json(json: &str) -> Result<Self> {
        let inner = load_hf_from_json(json).map_err(|e| {
            TokenizerError::Serialization(format!("Failed to parse tokenizer JSON: {e}"))
        })?;

        // Detect special tokens from vocab
        let pad_id =
            inner.token_to_id("<pad>").or_else(|| inner.token_to_id("<|endoftext|>")).unwrap_or(0);
        let eos_id = inner
            .token_to_id("</s>")
            .or_else(|| inner.token_to_id("<|im_end|>"))
            .or_else(|| inner.token_to_id("<|endoftext|>"));
        let bos_id = inner.token_to_id("<s>").or_else(|| inner.token_to_id("<|im_start|>"));

        Ok(Self { inner, pad_id, eos_id, bos_id })
    }

    /// Get vocabulary size
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Encode text to token IDs
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.inner.encode(text)
    }

    /// Encode text with special tokens (BOS/EOS)
    #[must_use]
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut tokens = Vec::new();
        if let Some(bos) = self.bos_id {
            tokens.push(bos);
        }
        tokens.extend(self.inner.encode(text));
        if let Some(eos) = self.eos_id {
            tokens.push(eos);
        }
        tokens
    }

    /// Decode token IDs to text
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> String {
        self.inner.decode(ids)
    }

    /// Get padding token ID
    #[must_use]
    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    /// Get EOS token ID
    #[must_use]
    pub fn eos_id(&self) -> Option<u32> {
        self.eos_id
    }

    /// Get BOS token ID
    #[must_use]
    pub fn bos_id(&self) -> Option<u32> {
        self.bos_id
    }

    /// Batch encode texts with padding
    #[must_use]
    pub fn batch_encode(&self, texts: &[&str], max_len: usize) -> Vec<Vec<u32>> {
        let mut encoded: Vec<Vec<u32>> = texts
            .iter()
            .map(|text| {
                let mut tokens = self.encode_with_special(text);
                tokens.truncate(max_len);
                tokens
            })
            .collect();

        let batch_max = encoded.iter().map(Vec::len).max().unwrap_or(0);
        let pad_to = batch_max.min(max_len);

        for tokens in &mut encoded {
            while tokens.len() < pad_to {
                tokens.push(self.pad_id);
            }
        }

        encoded
    }

    /// Create training batches from text pairs
    pub fn create_batches(
        &self,
        pairs: &[(&str, &str)],
        max_len: usize,
        batch_size: usize,
    ) -> Vec<crate::train::Batch> {
        use crate::Tensor;

        pairs
            .chunks(batch_size)
            .map(|chunk| {
                let inputs: Vec<&str> = chunk.iter().map(|(i, _)| *i).collect();
                let targets: Vec<&str> = chunk.iter().map(|(_, t)| *t).collect();

                let input_tokens = self.batch_encode(&inputs, max_len);
                let target_tokens = self.batch_encode(&targets, max_len);

                let input_data: Vec<f32> =
                    input_tokens.into_iter().flatten().map(|t| t as f32).collect();
                let target_data: Vec<f32> =
                    target_tokens.into_iter().flatten().map(|t| t as f32).collect();

                crate::train::Batch::new(
                    Tensor::from_vec(input_data, false),
                    Tensor::from_vec(target_data, false),
                )
            })
            .collect()
    }

    /// Create causal LM batches (target = shifted input)
    pub fn create_causal_batches(
        &self,
        texts: &[&str],
        max_len: usize,
        batch_size: usize,
    ) -> Vec<crate::train::Batch> {
        use crate::Tensor;

        texts
            .chunks(batch_size)
            .map(|chunk| {
                let encoded = self.batch_encode(chunk, max_len);

                let mut input_data: Vec<f32> = Vec::new();
                let mut target_data: Vec<f32> = Vec::new();

                for tokens in &encoded {
                    if tokens.len() > 1 {
                        input_data.extend(tokens[..tokens.len() - 1].iter().map(|&t| t as f32));
                        target_data.extend(tokens[1..].iter().map(|&t| t as f32));
                    }
                }

                crate::train::Batch::new(
                    Tensor::from_vec(input_data, false),
                    Tensor::from_vec(target_data, false),
                )
            })
            .collect()
    }
}

impl Default for HfTokenizer {
    fn default() -> Self {
        Self::gpt2()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_tokenizer_gpt2() {
        let tokenizer = HfTokenizer::gpt2();
        assert!(tokenizer.vocab_size() > 0);
        assert_eq!(tokenizer.pad_id(), GPT2_VOCAB_SIZE);
    }

    #[test]
    fn test_hf_tokenizer_qwen2() {
        let tokenizer = HfTokenizer::qwen2();
        assert_eq!(tokenizer.eos_id(), Some(Qwen2BpeTokenizer::IM_END_ID));
    }

    #[test]
    fn test_hf_tokenizer_encode() {
        let tokenizer = HfTokenizer::gpt2();
        let tokens = tokenizer.encode("Hello");
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_hf_tokenizer_encode_with_special() {
        let tokenizer = HfTokenizer::gpt2();
        let tokens = tokenizer.encode_with_special("Hi");
        assert!(tokens.last() == tokenizer.eos_id().as_ref());
    }

    #[test]
    fn test_hf_tokenizer_batch_encode() {
        let tokenizer = HfTokenizer::gpt2();
        let texts = vec!["Hello", "Hi there"];
        let encoded = tokenizer.batch_encode(&texts, 32);

        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded[0].len(), encoded[1].len());
    }

    #[test]
    fn test_hf_tokenizer_create_batches() {
        let tokenizer = HfTokenizer::gpt2();
        let pairs = vec![("Hello", "World"), ("How are", "you")];
        let batches = tokenizer.create_batches(&pairs, 16, 2);

        assert_eq!(batches.len(), 1);
        assert!(!batches[0].inputs.is_empty());
    }

    #[test]
    fn test_hf_tokenizer_create_causal_batches() {
        let tokenizer = HfTokenizer::gpt2();
        let texts = vec!["Hello world", "Test text"];
        let batches = tokenizer.create_causal_batches(&texts, 16, 2);

        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].inputs.len(), batches[0].targets.len());
    }

    #[test]
    fn test_hf_tokenizer_from_json() {
        let json = r#"{
            "model": {
                "vocab": {"hello": 0, "world": 1, "<|endoftext|>": 2},
                "merges": []
            },
            "added_tokens": []
        }"#;

        let result = HfTokenizer::from_json(json);
        assert!(result.is_ok());
        assert_eq!(result.expect("operation should succeed").vocab_size(), 3);
    }

    #[test]
    fn test_hf_tokenizer_from_json_invalid() {
        let result = HfTokenizer::from_json("invalid json");
        assert!(result.is_err());
    }
}
