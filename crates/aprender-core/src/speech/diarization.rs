//! Speaker diarization module.
//!
//! Provides speaker identification and segmentation:
//! - Speaker embedding extraction
//! - Clustering for speaker identification
//! - Timeline segmentation with speaker labels
//!
//! # Architecture
//!
//! ```text
//! Audio → VAD → Segments → Embeddings → Clustering → Speaker Labels
//!                            ↓
//!                     ECAPA-TDNN / X-Vector
//! ```
//!
//! # Example
//!
//! ```rust
//! use aprender::speech::diarization::{DiarizationConfig, SpeakerSegment};
//!
//! let config = DiarizationConfig::default();
//! assert!(config.min_speakers >= 1);
//! ```
//!
//! # PMAT Compliance
//!
//! - Zero `unwrap()` calls
//! - All public APIs return `Result<T, E>`

use super::{SpeechError, SpeechResult};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for speaker diarization
#[derive(Debug, Clone)]
pub struct DiarizationConfig {
    /// Minimum number of speakers to detect
    pub min_speakers: usize,
    /// Maximum number of speakers (None = unlimited)
    pub max_speakers: Option<usize>,
    /// Minimum segment length in milliseconds
    pub min_segment_ms: u32,
    /// Clustering threshold (cosine similarity)
    pub clustering_threshold: f32,
    /// Embedding dimension (typically 192 or 512)
    pub embedding_dim: usize,
}

impl Default for DiarizationConfig {
    fn default() -> Self {
        Self {
            min_speakers: 1,
            max_speakers: None,
            min_segment_ms: 500,
            clustering_threshold: 0.7,
            embedding_dim: 192,
        }
    }
}

impl DiarizationConfig {
    /// Set expected number of speakers
    #[must_use]
    pub fn with_speakers(mut self, min: usize, max: Option<usize>) -> Self {
        self.min_speakers = min.max(1);
        self.max_speakers = max;
        self
    }

    /// Set clustering threshold
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.clustering_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> SpeechResult<()> {
        if self.min_speakers == 0 {
            return Err(SpeechError::InvalidConfig(
                "min_speakers must be >= 1".to_string(),
            ));
        }
        if let Some(max) = self.max_speakers {
            if max < self.min_speakers {
                return Err(SpeechError::InvalidConfig(
                    "max_speakers must be >= min_speakers".to_string(),
                ));
            }
        }
        if self.clustering_threshold < 0.0 || self.clustering_threshold > 1.0 {
            return Err(SpeechError::InvalidConfig(
                "clustering_threshold must be in [0.0, 1.0]".to_string(),
            ));
        }
        Ok(())
    }
}

// ============================================================================
// Speaker Types
// ============================================================================

/// Identified speaker with embedding
#[derive(Debug, Clone)]
pub struct Speaker {
    /// Speaker ID (0-indexed)
    pub id: usize,
    /// Speaker label (e.g., "`SPEAKER_00`")
    pub label: String,
    /// Speaker embedding vector (for similarity comparison)
    pub embedding: Vec<f32>,
    /// Total speaking time in milliseconds
    pub total_speaking_time_ms: u64,
}

impl Speaker {
    /// Create a new speaker with ID
    #[must_use]
    pub fn new(id: usize, embedding: Vec<f32>) -> Self {
        Self {
            id,
            label: format!("SPEAKER_{id:02}"),
            embedding,
            total_speaking_time_ms: 0,
        }
    }

    /// Create with custom label
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Compute cosine similarity with another speaker
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f32 {
        cosine_similarity(&self.embedding, &other.embedding)
    }
}

/// A segment attributed to a specific speaker
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerSegment {
    /// Speaker ID
    pub speaker_id: usize,
    /// Speaker label
    pub speaker_label: String,
    /// Start time in milliseconds
    pub start_ms: u64,
    /// End time in milliseconds
    pub end_ms: u64,
    /// Confidence score
    pub confidence: f32,
}

impl SpeakerSegment {
    /// Create a new speaker segment
    #[must_use]
    pub fn new(speaker_id: usize, start_ms: u64, end_ms: u64) -> Self {
        Self {
            speaker_id,
            speaker_label: format!("SPEAKER_{speaker_id:02}"),
            start_ms,
            end_ms,
            confidence: 1.0,
        }
    }

    /// Duration in milliseconds
    #[must_use]
    pub fn duration_ms(&self) -> u64 {
        self.end_ms.saturating_sub(self.start_ms)
    }
}

/// Complete diarization result
#[derive(Debug, Clone, Default)]
pub struct DiarizationResult {
    /// Identified speakers
    pub speakers: Vec<Speaker>,
    /// Timeline of speaker segments
    pub segments: Vec<SpeakerSegment>,
    /// Total audio duration in milliseconds
    pub duration_ms: u64,
}

impl DiarizationResult {
    /// Create empty result
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of speakers detected
    #[must_use]
    pub fn speaker_count(&self) -> usize {
        self.speakers.len()
    }

    /// Get segments for a specific speaker
    #[must_use]
    pub fn segments_for_speaker(&self, speaker_id: usize) -> Vec<&SpeakerSegment> {
        self.segments
            .iter()
            .filter(|s| s.speaker_id == speaker_id)
            .collect()
    }

    /// Get speaking time for a speaker
    #[must_use]
    pub fn speaking_time_ms(&self, speaker_id: usize) -> u64 {
        self.segments_for_speaker(speaker_id)
            .iter()
            .map(|s| s.duration_ms())
            .sum()
    }

    /// Check if result is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

// ============================================================================
// Diarization Functions
// ============================================================================

/// Perform speaker diarization on audio segments
///
/// # Arguments
/// * `embeddings` - Speaker embeddings for each segment (shape: `n_segments` × `embedding_dim`)
/// * `segment_times` - Start/end times for each segment [(`start_ms`, `end_ms`), ...]
/// * `config` - Diarization configuration
///
/// # Returns
/// Diarization result with speakers and timeline
pub fn diarize(
    embeddings: &[Vec<f32>],
    segment_times: &[(u64, u64)],
    config: &DiarizationConfig,
) -> SpeechResult<DiarizationResult> {
    config.validate()?;

    if embeddings.len() != segment_times.len() {
        return Err(SpeechError::InvalidAudio(format!(
            "embeddings count ({}) must match segment_times count ({})",
            embeddings.len(),
            segment_times.len()
        )));
    }

    if embeddings.is_empty() {
        return Ok(DiarizationResult::new());
    }

    // Validate embedding dimensions
    for (i, emb) in embeddings.iter().enumerate() {
        if emb.len() != config.embedding_dim {
            return Err(SpeechError::InvalidAudio(format!(
                "embedding {} has wrong dimension: {} (expected {})",
                i,
                emb.len(),
                config.embedding_dim
            )));
        }
    }

    // Simple agglomerative clustering
    let cluster_labels = cluster_embeddings(embeddings, config);

    // Build speaker list
    let max_label = cluster_labels.iter().copied().max().unwrap_or(0);
    let mut speakers: Vec<Speaker> = (0..=max_label)
        .map(|id| {
            // Compute average embedding for this speaker
            let speaker_embeddings: Vec<_> = embeddings
                .iter()
                .zip(cluster_labels.iter())
                .filter(|(_, &label)| label == id)
                .map(|(emb, _)| emb)
                .collect();

            let avg_embedding = if speaker_embeddings.is_empty() {
                vec![0.0; config.embedding_dim]
            } else {
                average_embeddings(&speaker_embeddings, config.embedding_dim)
            };

            Speaker::new(id, avg_embedding)
        })
        .collect();

    // Build segments
    let segments: Vec<SpeakerSegment> = segment_times
        .iter()
        .zip(cluster_labels.iter())
        .map(|(&(start, end), &label)| SpeakerSegment::new(label, start, end))
        .collect();

    // Update speaker speaking times
    for speaker in &mut speakers {
        speaker.total_speaking_time_ms = segments
            .iter()
            .filter(|s| s.speaker_id == speaker.id)
            .map(SpeakerSegment::duration_ms)
            .sum();
    }

    let duration_ms = segment_times.iter().map(|(_, end)| *end).max().unwrap_or(0);

    Ok(DiarizationResult {
        speakers,
        segments,
        duration_ms,
    })
}

/// Simple agglomerative clustering for speaker embeddings
/// Compute pairwise cosine similarity matrix.
fn pairwise_similarity_matrix(embeddings: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = embeddings.len();
    let mut similarities = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            similarities[i][j] = sim;
            similarities[j][i] = sim;
        }
    }
    similarities
}

/// Find the most similar pair of segments belonging to different clusters.
fn find_best_cluster_pair(
    labels: &[usize],
    similarities: &[Vec<f32>],
) -> (f32, Option<(usize, usize)>) {
    let mut best_sim = 0.0f32;
    let mut best_pair = None;
    for i in 0..labels.len() {
        for j in i + 1..labels.len() {
            if labels[i] != labels[j] && similarities[i][j] > best_sim {
                best_sim = similarities[i][j];
                best_pair = Some((labels[i], labels[j]));
            }
        }
    }
    (best_sim, best_pair)
}

/// Renumber cluster labels to be contiguous (0, 1, 2, ...).
fn renumber_clusters(labels: &[usize]) -> Vec<usize> {
    let mut unique: Vec<_> = labels
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    unique.sort_unstable();
    let map: std::collections::HashMap<usize, usize> = unique
        .into_iter()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect();
    labels.iter().map(|l| *map.get(l).unwrap_or(l)).collect()
}

fn cluster_embeddings(embeddings: &[Vec<f32>], config: &DiarizationConfig) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return vec![];
    }

    let mut labels: Vec<usize> = (0..n).collect();
    let similarities = pairwise_similarity_matrix(embeddings);

    loop {
        let (best_sim, best_pair) = find_best_cluster_pair(&labels, &similarities);

        if best_sim < config.clustering_threshold {
            break;
        }

        let current_clusters: std::collections::HashSet<_> = labels.iter().collect();
        if config
            .max_speakers
            .is_some_and(|max| current_clusters.len() <= max)
        {
            break;
        }

        let Some((cluster_a, cluster_b)) = best_pair else {
            break;
        };
        let (target, source) = (cluster_a.min(cluster_b), cluster_a.max(cluster_b));
        for label in &mut labels {
            if *label == source {
                *label = target;
            }
        }
    }

    renumber_clusters(&labels)
}

/// ONE PATH: Delegates to `nn::functional::cosine_similarity_slice` (UCBD §4).
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    crate::nn::functional::cosine_similarity_slice(a, b)
}

/// Average multiple embeddings
fn average_embeddings(embeddings: &[&Vec<f32>], dim: usize) -> Vec<f32> {
    if embeddings.is_empty() {
        return vec![0.0; dim];
    }

    let mut result = vec![0.0; dim];
    for emb in embeddings {
        for (i, &val) in emb.iter().enumerate() {
            if i < dim {
                result[i] += val;
            }
        }
    }

    let n = embeddings.len() as f32;
    for val in &mut result {
        *val /= n;
    }

    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[path = "diarization_tests.rs"]
mod tests;
