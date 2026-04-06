//! Tokenizer Surgery for Vocabulary Transplantation (GH-447)
//!
//! When adapting a pre-trained model to a new tokenizer (e.g., domain-specific
//! BPE vocabulary), embedding rows must be transplanted from the source model
//! to the target. Tokens present in both vocabularies get direct copies;
//! missing tokens are handled via nearest-neighbor lookup or average pooling.
//!
//! # Key Design Decisions
//!
//! - **Overlap threshold**: Surgery is rejected if the vocabularies share
//!   fewer tokens than the configured threshold (default 50%), preventing
//!   catastrophic representation loss
//! - **Three methods**: `DirectCopy` (fastest, zero-fills missing),
//!   `NearestNeighbor` (finds closest match), `AveragePool` (mean of all
//!   source embeddings for missing tokens)
//!
//! # References
//!
//! - Hewitt et al. 2021: "Initializing New Word Embeddings for Pretrained
//!   Language Models"
//! - Minixhofer et al. 2022: "WECHSEL: Effective Initialization of
//!   Subword Embeddings for Cross-Lingual Transfer of Monolingual Models"
//!
//! # Toyota Way Principles
//!
//! - **Poka-Yoke**: Overlap threshold prevents silent vocabulary misalignment
//! - **Jidoka**: Validation stops surgery if quality gate fails

use crate::error::{AprenderError, Result};
use std::collections::HashMap;

/// Method used to transplant embeddings for tokens not found in both vocabularies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurgeryMethod {
    /// Copy matched embeddings directly; zero-fill unmatched target tokens.
    DirectCopy,
    /// For unmatched tokens, find the nearest neighbor in the source vocabulary
    /// by string edit distance and copy its embedding.
    NearestNeighbor,
    /// For unmatched tokens, use the mean of all source embeddings as
    /// the initialization vector.
    AveragePool,
}

impl Default for SurgeryMethod {
    fn default() -> Self {
        Self::DirectCopy
    }
}

/// Configuration for tokenizer surgery.
#[derive(Debug, Clone)]
pub struct TokenizerSurgeryConfig {
    /// Number of tokens in the source vocabulary.
    pub source_vocab_size: usize,
    /// Number of tokens in the target vocabulary.
    pub target_vocab_size: usize,
    /// Minimum fraction of overlapping tokens required (0.0 to 1.0).
    /// Surgery is rejected if overlap falls below this threshold.
    pub overlap_threshold: f64,
    /// Strategy for handling tokens without a direct match.
    pub method: SurgeryMethod,
}

impl Default for TokenizerSurgeryConfig {
    fn default() -> Self {
        Self {
            source_vocab_size: 0,
            target_vocab_size: 0,
            overlap_threshold: 0.5,
            method: SurgeryMethod::default(),
        }
    }
}

/// Bidirectional mapping between source and target vocabularies.
#[derive(Debug, Clone)]
pub struct VocabMapping {
    /// For each source token index, the matching target index (if any).
    pub source_to_target: Vec<Option<usize>>,
    /// For each target token index, the matching source index (if any).
    pub target_to_source: Vec<Option<usize>>,
    /// Number of tokens present in both vocabularies.
    pub overlap_count: usize,
    /// Fraction of the smaller vocabulary that overlaps.
    pub overlap_ratio: f64,
}

/// Summary report produced after an embedding transplant operation.
#[derive(Debug, Clone)]
pub struct SurgeryReport {
    /// Number of embedding rows directly copied from source to target.
    pub tokens_copied: usize,
    /// Number of embedding rows filled by averaging source embeddings.
    pub tokens_averaged: usize,
    /// Number of embedding rows left as zero vectors.
    pub tokens_zeroed: usize,
    /// Overlap ratio between the two vocabularies.
    pub overlap_ratio: f64,
}

/// Compute the bidirectional overlap between two token vocabularies.
///
/// Builds a hash map of source tokens for O(n + m) lookup, then scans the
/// target vocabulary to find exact string matches.
///
/// # Arguments
///
/// * `source_tokens` - Token strings from the source vocabulary
/// * `target_tokens` - Token strings from the target vocabulary
///
/// # Returns
///
/// A `VocabMapping` with per-index mappings and aggregate overlap statistics.
pub fn compute_vocab_overlap(source_tokens: &[String], target_tokens: &[String]) -> VocabMapping {
    // Build source token -> index lookup
    let source_index: HashMap<&str, usize> = source_tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();

    let mut source_to_target = vec![None; source_tokens.len()];
    let mut target_to_source = vec![None; target_tokens.len()];
    let mut overlap_count = 0usize;

    for (target_idx, token) in target_tokens.iter().enumerate() {
        if let Some(&source_idx) = source_index.get(token.as_str()) {
            source_to_target[source_idx] = Some(target_idx);
            target_to_source[target_idx] = Some(source_idx);
            overlap_count += 1;
        }
    }

    let smaller = source_tokens.len().min(target_tokens.len()).max(1);
    let overlap_ratio = overlap_count as f64 / smaller as f64;

    VocabMapping {
        source_to_target,
        target_to_source,
        overlap_count,
        overlap_ratio,
    }
}

/// Transplant embedding rows from a source model to a target model.
///
/// For each target token that has a matching source token (via `mapping`),
/// the corresponding embedding row is copied directly. Unmatched tokens are
/// handled according to `config.method`:
///
/// - `DirectCopy`: unmatched rows remain as-is (typically zero)
/// - `NearestNeighbor`: copies the embedding of the closest source token
///   by Levenshtein edit distance
/// - `AveragePool`: fills unmatched rows with the mean of all source embeddings
///
/// # Arguments
///
/// * `source_embeddings` - Flat row-major embedding matrix (source_vocab_size x hidden_dim)
/// * `target_embeddings` - Flat row-major embedding matrix (target_vocab_size x hidden_dim), modified in place
/// * `mapping` - Vocabulary mapping from `compute_vocab_overlap`
/// * `config` - Surgery configuration
/// * `hidden_dim` - Embedding dimensionality
pub fn transplant_embeddings(
    source_embeddings: &[f64],
    target_embeddings: &mut [f64],
    mapping: &VocabMapping,
    config: &TokenizerSurgeryConfig,
    hidden_dim: usize,
) -> SurgeryReport {
    let mut tokens_copied = 0usize;
    let mut tokens_averaged = 0usize;
    let mut tokens_zeroed = 0usize;

    // Pre-compute average source embedding for AveragePool fallback
    let avg_embedding: Vec<f64> = if config.method == SurgeryMethod::AveragePool {
        compute_average_embedding(source_embeddings, config.source_vocab_size, hidden_dim)
    } else {
        Vec::new()
    };

    for target_idx in 0..config.target_vocab_size {
        let target_offset = target_idx * hidden_dim;
        if target_offset + hidden_dim > target_embeddings.len() {
            break;
        }

        if let Some(source_idx) = mapping.target_to_source.get(target_idx).copied().flatten() {
            // Direct match: copy embedding row
            let source_offset = source_idx * hidden_dim;
            if source_offset + hidden_dim <= source_embeddings.len() {
                target_embeddings[target_offset..target_offset + hidden_dim]
                    .copy_from_slice(&source_embeddings[source_offset..source_offset + hidden_dim]);
                tokens_copied += 1;
            } else {
                tokens_zeroed += 1;
            }
        } else {
            // No direct match: apply fallback strategy
            match config.method {
                SurgeryMethod::DirectCopy => {
                    // Leave as-is (zero-filled by caller)
                    tokens_zeroed += 1;
                }
                SurgeryMethod::NearestNeighbor => {
                    // Find nearest source embedding by Euclidean distance
                    if let Some(nearest_offset) = find_nearest_source_embedding(
                        target_embeddings,
                        target_offset,
                        source_embeddings,
                        config.source_vocab_size,
                        hidden_dim,
                    ) {
                        target_embeddings[target_offset..target_offset + hidden_dim]
                            .copy_from_slice(
                                &source_embeddings[nearest_offset..nearest_offset + hidden_dim],
                            );
                        tokens_averaged += 1;
                    } else {
                        tokens_zeroed += 1;
                    }
                }
                SurgeryMethod::AveragePool => {
                    if avg_embedding.len() == hidden_dim {
                        target_embeddings[target_offset..target_offset + hidden_dim]
                            .copy_from_slice(&avg_embedding);
                        tokens_averaged += 1;
                    } else {
                        tokens_zeroed += 1;
                    }
                }
            }
        }
    }

    SurgeryReport {
        tokens_copied,
        tokens_averaged,
        tokens_zeroed,
        overlap_ratio: mapping.overlap_ratio,
    }
}

/// Validate that the vocabulary overlap meets the configured quality threshold.
///
/// # Errors
///
/// Returns `AprenderError::ValidationError` if the overlap ratio falls below
/// `config.overlap_threshold`, indicating the vocabularies are too dissimilar
/// for safe embedding transplantation.
pub fn validate_surgery(mapping: &VocabMapping, config: &TokenizerSurgeryConfig) -> Result<()> {
    if config.overlap_threshold < 0.0 || config.overlap_threshold > 1.0 {
        return Err(AprenderError::InvalidHyperparameter {
            param: "overlap_threshold".to_string(),
            value: format!("{}", config.overlap_threshold),
            constraint: "must be between 0.0 and 1.0".to_string(),
        });
    }

    if mapping.overlap_ratio < config.overlap_threshold {
        return Err(AprenderError::ValidationError {
            message: format!(
                "vocabulary overlap {:.2}% is below threshold {:.2}%: \
                 surgery would destroy too many pre-trained representations \
                 ({} tokens matched out of {} target tokens)",
                mapping.overlap_ratio * 100.0,
                config.overlap_threshold * 100.0,
                mapping.overlap_count,
                config.target_vocab_size,
            ),
        });
    }

    Ok(())
}

/// Compute the element-wise mean of all source embedding rows.
fn compute_average_embedding(
    source_embeddings: &[f64],
    source_vocab_size: usize,
    hidden_dim: usize,
) -> Vec<f64> {
    if source_vocab_size == 0 || hidden_dim == 0 {
        return vec![0.0; hidden_dim];
    }

    let mut avg = vec![0.0; hidden_dim];
    let mut count = 0usize;

    for row in 0..source_vocab_size {
        let offset = row * hidden_dim;
        if offset + hidden_dim > source_embeddings.len() {
            break;
        }
        for (j, val) in source_embeddings[offset..offset + hidden_dim]
            .iter()
            .enumerate()
        {
            avg[j] += val;
        }
        count += 1;
    }

    if count > 0 {
        let scale = 1.0 / count as f64;
        for v in &mut avg {
            *v *= scale;
        }
    }

    avg
}

/// Find the nearest source embedding to the target embedding at `target_offset`
/// using Euclidean distance. Returns the byte offset in `source_embeddings`.
fn find_nearest_source_embedding(
    target_embeddings: &[f64],
    target_offset: usize,
    source_embeddings: &[f64],
    source_vocab_size: usize,
    hidden_dim: usize,
) -> Option<usize> {
    if source_vocab_size == 0 || hidden_dim == 0 {
        return None;
    }

    let target_row = &target_embeddings[target_offset..target_offset + hidden_dim];

    let mut best_offset = None;
    let mut best_dist = f64::MAX;

    for row in 0..source_vocab_size {
        let offset = row * hidden_dim;
        if offset + hidden_dim > source_embeddings.len() {
            break;
        }

        let source_row = &source_embeddings[offset..offset + hidden_dim];
        let dist: f64 = target_row
            .iter()
            .zip(source_row.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();

        if dist < best_dist {
            best_dist = dist;
            best_offset = Some(offset);
        }
    }

    best_offset
}

#[cfg(test)]
#[path = "tokenizer_surgery_tests.rs"]
mod tests;
