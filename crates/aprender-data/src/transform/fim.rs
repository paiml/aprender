//! Fill-in-the-Middle (FIM) data transform for code model training.
//!
//! Implements PSM (Prefix-Suffix-Middle) and SPM (Suffix-Prefix-Middle)
//! FIM formats from Bavarian et al. (2022):
//! "Efficient Training of Language Models to Fill in the Middle"
//!
//! Given a code sequence, FIM randomly splits it into (prefix, middle, suffix)
//! and rearranges with sentinel tokens so the model learns to infill code.

use std::sync::Arc;

use arrow::array::{Array, RecordBatch, StringArray};
use rand::{Rng, SeedableRng};

use super::Transform;
use crate::error::{Error, Result};

/// FIM format variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FimFormat {
    /// Prefix-Suffix-Middle: `<PRE>prefix<SUF>suffix<MID>middle`
    PSM,
    /// Suffix-Prefix-Middle: `<SUF>suffix<PRE>prefix<MID>middle`
    SPM,
}

/// Configuration for FIM sentinel tokens.
#[derive(Debug, Clone)]
pub struct FimTokens {
    /// Prefix sentinel token
    pub prefix: String,
    /// Suffix sentinel token
    pub suffix: String,
    /// Middle sentinel token
    pub middle: String,
}

impl Default for FimTokens {
    fn default() -> Self {
        Self {
            prefix: "<|fim_prefix|>".to_string(),
            suffix: "<|fim_suffix|>".to_string(),
            middle: "<|fim_middle|>".to_string(),
        }
    }
}

/// Fill-in-the-Middle transform for code training data.
///
/// Applies FIM transformation to a text column in a RecordBatch.
/// Each row is randomly split into (prefix, middle, suffix) and
/// reassembled in PSM or SPM format with sentinel tokens.
///
/// Rows shorter than `min_chars` are left unchanged.
#[derive(Debug, Clone)]
pub struct Fim {
    /// Column name containing code text
    column: String,
    /// Probability of applying FIM to each row (0.0-1.0)
    rate: f64,
    /// FIM format variant
    format: FimFormat,
    /// Sentinel tokens
    tokens: FimTokens,
    /// Minimum characters for FIM to apply
    min_chars: usize,
    /// Random seed for reproducibility
    seed: u64,
}

impl Fim {
    /// Create a new FIM transform for the given column.
    pub fn new(column: impl Into<String>) -> Self {
        Self {
            column: column.into(),
            rate: 0.5,
            format: FimFormat::PSM,
            tokens: FimTokens::default(),
            min_chars: 10,
            seed: 42,
        }
    }

    /// Set the FIM application rate (0.0-1.0).
    #[must_use]
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = rate.clamp(0.0, 1.0);
        self
    }

    /// Set the FIM format variant.
    #[must_use]
    pub fn with_format(mut self, format: FimFormat) -> Self {
        self.format = format;
        self
    }

    /// Set custom sentinel tokens.
    #[must_use]
    pub fn with_tokens(mut self, tokens: FimTokens) -> Self {
        self.tokens = tokens;
        self
    }

    /// Set minimum character count for FIM to apply.
    #[must_use]
    pub fn with_min_chars(mut self, min_chars: usize) -> Self {
        self.min_chars = min_chars;
        self
    }

    /// Set random seed for reproducibility.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Apply FIM transformation to a single text string.
fn apply_fim_to_text(
    text: &str,
    format: FimFormat,
    tokens: &FimTokens,
    rng: &mut impl Rng,
) -> String {
    let len = text.len();
    if len < 2 {
        return text.to_string();
    }

    // Pick two random split points to create (prefix, middle, suffix)
    let mut a = rng.gen_range(0..len);
    let mut b = rng.gen_range(0..len);
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }

    // Align to char boundaries
    let a = find_char_boundary(text, a);
    let b = find_char_boundary(text, b);

    let prefix = &text[..a];
    let middle = &text[a..b];
    let suffix = &text[b..];

    match format {
        FimFormat::PSM => {
            format!(
                "{}{}{}{}{}{}",
                tokens.prefix, prefix, tokens.suffix, suffix, tokens.middle, middle
            )
        }
        FimFormat::SPM => {
            format!(
                "{}{}{}{}{}{}",
                tokens.suffix, suffix, tokens.prefix, prefix, tokens.middle, middle
            )
        }
    }
}

/// Find the nearest char boundary at or after the given byte offset.
fn find_char_boundary(s: &str, byte_offset: usize) -> usize {
    let mut offset = byte_offset.min(s.len());
    while offset < s.len() && !s.is_char_boundary(offset) {
        offset += 1;
    }
    offset.min(s.len())
}

impl Transform for Fim {
    fn apply(&self, batch: RecordBatch) -> Result<RecordBatch> {
        let schema = batch.schema();
        let col_idx = schema
            .index_of(&self.column)
            .map_err(|_| Error::column_not_found(&self.column))?;

        let col = batch
            .column(col_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| {
                Error::transform(format!(
                    "Column '{}' must be Utf8 type for FIM transform",
                    self.column
                ))
            })?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.seed);
        let transformed: Vec<Option<String>> = (0..col.len())
            .map(|i| {
                if col.is_null(i) {
                    return None;
                }
                let text = col.value(i);
                if text.len() < self.min_chars {
                    return Some(text.to_string());
                }
                let apply_fim: bool = rng.gen_bool(self.rate);
                if apply_fim {
                    Some(apply_fim_to_text(text, self.format, &self.tokens, &mut rng))
                } else {
                    Some(text.to_string())
                }
            })
            .collect();

        let new_col = StringArray::from(transformed);
        let mut columns: Vec<Arc<dyn arrow::array::Array>> = batch.columns().to_vec();
        columns[col_idx] = Arc::new(new_col);
        RecordBatch::try_new(schema, columns).map_err(Error::Arrow)
    }
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn create_code_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("code", DataType::Utf8, false)]));
        let code = StringArray::from(vec![
            "def hello():\n    print('hello world')\n",
            "class Foo:\n    def bar(self):\n        return 42\n",
            "x = 1",
        ]);
        RecordBatch::try_new(schema, vec![Arc::new(code)]).expect("batch creation should succeed")
    }

    #[test]
    fn test_fim_psm_format() {
        let text = "def hello():\n    print('hello')";
        let tokens = FimTokens::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = apply_fim_to_text(text, FimFormat::PSM, &tokens, &mut rng);
        assert!(result.contains("<|fim_prefix|>"));
        assert!(result.contains("<|fim_suffix|>"));
        assert!(result.contains("<|fim_middle|>"));
    }

    #[test]
    fn test_fim_spm_format() {
        let text = "def hello():\n    print('hello')";
        let tokens = FimTokens::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = apply_fim_to_text(text, FimFormat::SPM, &tokens, &mut rng);
        // SPM starts with suffix token
        assert!(result.starts_with("<|fim_suffix|>"));
    }

    #[test]
    fn test_fim_transform_applies_to_batch() {
        let batch = create_code_batch();
        let fim = Fim::new("code").with_rate(1.0).with_seed(42);
        let result = fim.apply(batch);
        assert!(result.is_ok());
        let result = result.expect("should succeed");
        assert_eq!(result.num_rows(), 3);

        let col = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("should be string");
        // First two rows should have FIM tokens (long enough)
        assert!(col.value(0).contains("<|fim_prefix|>"));
        assert!(col.value(1).contains("<|fim_prefix|>"));
        // Third row too short (5 chars < 10 min_chars default)
        assert_eq!(col.value(2), "x = 1");
    }

    #[test]
    fn test_fim_rate_zero_leaves_unchanged() {
        let batch = create_code_batch();
        let fim = Fim::new("code").with_rate(0.0).with_seed(42);
        let result = fim.apply(batch.clone()).expect("should succeed");
        let original = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string");
        let transformed = result
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string");
        for i in 0..original.len() {
            assert_eq!(original.value(i), transformed.value(i));
        }
    }

    #[test]
    fn test_fim_deterministic_with_seed() {
        let batch = create_code_batch();
        let fim1 = Fim::new("code").with_rate(1.0).with_seed(123);
        let fim2 = Fim::new("code").with_rate(1.0).with_seed(123);
        let r1 = fim1.apply(batch.clone()).expect("should succeed");
        let r2 = fim2.apply(batch).expect("should succeed");
        let c1 = r1
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("s");
        let c2 = r2
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("s");
        for i in 0..c1.len() {
            assert_eq!(c1.value(i), c2.value(i));
        }
    }

    #[test]
    fn test_fim_wrong_column_errors() {
        let batch = create_code_batch();
        let fim = Fim::new("nonexistent");
        let result = fim.apply(batch);
        assert!(result.is_err());
    }

    #[test]
    fn test_fim_custom_tokens() {
        let text = "def foo(): pass";
        let tokens = FimTokens {
            prefix: "<PRE>".to_string(),
            suffix: "<SUF>".to_string(),
            middle: "<MID>".to_string(),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = apply_fim_to_text(text, FimFormat::PSM, &tokens, &mut rng);
        assert!(result.contains("<PRE>"));
        assert!(result.contains("<SUF>"));
        assert!(result.contains("<MID>"));
    }

    #[test]
    fn test_fim_preserves_content() {
        let text = "def hello():\n    print('hello')";
        let tokens = FimTokens::default();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let result = apply_fim_to_text(text, FimFormat::PSM, &tokens, &mut rng);
        // Remove sentinel tokens and verify all original content is present
        let stripped = result
            .replace("<|fim_prefix|>", "")
            .replace("<|fim_suffix|>", "")
            .replace("<|fim_middle|>", "");
        // All chars from original should be in stripped (just reordered)
        for ch in text.chars() {
            assert!(stripped.contains(ch), "Missing char: {ch}");
        }
    }

    #[test]
    fn test_find_char_boundary() {
        let s = "hello";
        assert_eq!(find_char_boundary(s, 0), 0);
        assert_eq!(find_char_boundary(s, 3), 3);
        assert_eq!(find_char_boundary(s, 10), 5);
    }

    #[test]
    fn test_find_char_boundary_multibyte() {
        let s = "héllo"; // é is 2 bytes
        let boundary = find_char_boundary(s, 2);
        assert!(s.is_char_boundary(boundary));
    }
}
