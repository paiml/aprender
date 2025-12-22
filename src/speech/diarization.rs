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
    /// Speaker label (e.g., "SPEAKER_00")
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
/// * `embeddings` - Speaker embeddings for each segment (shape: n_segments × embedding_dim)
/// * `segment_times` - Start/end times for each segment [(start_ms, end_ms), ...]
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
fn cluster_embeddings(embeddings: &[Vec<f32>], config: &DiarizationConfig) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return vec![];
    }

    // Each segment starts as its own cluster
    let mut labels: Vec<usize> = (0..n).collect();

    // Compute pairwise similarities
    let mut similarities = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let sim = cosine_similarity(&embeddings[i], &embeddings[j]);
            similarities[i][j] = sim;
            similarities[j][i] = sim;
        }
    }

    // Merge clusters until threshold or max_speakers reached
    loop {
        // Find most similar pair of different clusters
        let mut best_sim = 0.0f32;
        let mut best_pair = None;

        for i in 0..n {
            for j in i + 1..n {
                if labels[i] != labels[j] && similarities[i][j] > best_sim {
                    best_sim = similarities[i][j];
                    best_pair = Some((labels[i], labels[j]));
                }
            }
        }

        // Stop if no pair above threshold
        if best_sim < config.clustering_threshold {
            break;
        }

        // Check max_speakers constraint
        let current_clusters: std::collections::HashSet<_> = labels.iter().collect();
        if let Some(max) = config.max_speakers {
            if current_clusters.len() <= max {
                break;
            }
        }

        // Merge clusters
        if let Some((cluster_a, cluster_b)) = best_pair {
            let target = cluster_a.min(cluster_b);
            let source = cluster_a.max(cluster_b);
            for label in &mut labels {
                if *label == source {
                    *label = target;
                }
            }
        } else {
            break;
        }
    }

    // Renumber clusters to be contiguous (0, 1, 2, ...)
    let unique_labels: Vec<usize> = {
        let mut v: Vec<_> = labels
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        v.sort_unstable();
        v
    };

    let label_map: std::collections::HashMap<usize, usize> = unique_labels
        .into_iter()
        .enumerate()
        .map(|(new, old)| (old, new))
        .collect();

    labels
        .iter()
        .map(|l| *label_map.get(l).unwrap_or(l))
        .collect()
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
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
mod tests {
    use super::*;

    #[test]
    fn test_diarization_config_default() {
        let config = DiarizationConfig::default();
        assert!(config.validate().is_ok());
        assert_eq!(config.min_speakers, 1);
        assert!(config.max_speakers.is_none());
    }

    #[test]
    fn test_diarization_config_with_speakers() {
        let config = DiarizationConfig::default().with_speakers(2, Some(4));
        assert_eq!(config.min_speakers, 2);
        assert_eq!(config.max_speakers, Some(4));
    }

    #[test]
    fn test_diarization_config_validation() {
        let mut config = DiarizationConfig::default();
        config.min_speakers = 0;
        assert!(config.validate().is_err());

        config.min_speakers = 2;
        config.max_speakers = Some(1);
        assert!(config.validate().is_err());

        config.max_speakers = Some(3);
        config.clustering_threshold = 1.5;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_speaker_new() {
        let speaker = Speaker::new(0, vec![1.0, 0.0, 0.0]);
        assert_eq!(speaker.id, 0);
        assert_eq!(speaker.label, "SPEAKER_00");
        assert_eq!(speaker.embedding.len(), 3);
    }

    #[test]
    fn test_speaker_similarity() {
        let s1 = Speaker::new(0, vec![1.0, 0.0, 0.0]);
        let s2 = Speaker::new(1, vec![1.0, 0.0, 0.0]);
        let s3 = Speaker::new(2, vec![0.0, 1.0, 0.0]);

        assert!((s1.similarity(&s2) - 1.0).abs() < 0.001); // Identical
        assert!(s1.similarity(&s3).abs() < 0.001); // Orthogonal
    }

    #[test]
    fn test_speaker_segment_new() {
        let seg = SpeakerSegment::new(0, 1000, 2000);
        assert_eq!(seg.speaker_id, 0);
        assert_eq!(seg.speaker_label, "SPEAKER_00");
        assert_eq!(seg.duration_ms(), 1000);
    }

    #[test]
    fn test_diarization_result_empty() {
        let result = DiarizationResult::new();
        assert!(result.is_empty());
        assert_eq!(result.speaker_count(), 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![-1.0, 0.0, 0.0];

        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);
        assert!(cosine_similarity(&a, &c).abs() < 0.001);
        assert!((cosine_similarity(&a, &d) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_edge_cases() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0); // Different lengths
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0); // Zero vector
    }

    #[test]
    fn test_average_embeddings() {
        let e1 = vec![1.0, 0.0];
        let e2 = vec![0.0, 1.0];
        let embeddings: Vec<&Vec<f32>> = vec![&e1, &e2];

        let avg = average_embeddings(&embeddings, 2);
        assert!((avg[0] - 0.5).abs() < 0.001);
        assert!((avg[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_diarize_empty() {
        let config = DiarizationConfig::default();
        let result = diarize(&[], &[], &config);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_diarize_single_speaker() {
        let config = DiarizationConfig {
            embedding_dim: 3,
            clustering_threshold: 0.9,
            ..Default::default()
        };

        // Two very similar embeddings = same speaker
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.99, 0.1, 0.0]];
        let times = vec![(0, 1000), (1000, 2000)];

        let result = diarize(&embeddings, &times, &config).unwrap();
        assert_eq!(result.speaker_count(), 1);
        assert_eq!(result.segments.len(), 2);
    }

    #[test]
    fn test_diarize_two_speakers() {
        let config = DiarizationConfig {
            embedding_dim: 3,
            clustering_threshold: 0.5,
            ..Default::default()
        };

        // Two very different embeddings = different speakers
        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let times = vec![(0, 1000), (1000, 2000)];

        let result = diarize(&embeddings, &times, &config).unwrap();
        assert_eq!(result.speaker_count(), 2);
    }

    #[test]
    fn test_diarize_speaking_time() {
        let config = DiarizationConfig {
            embedding_dim: 3,
            clustering_threshold: 0.9,
            ..Default::default()
        };

        let embeddings = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
        let times = vec![(0, 1000), (1000, 3000)];

        let result = diarize(&embeddings, &times, &config).unwrap();
        assert_eq!(result.speaking_time_ms(0), 3000);
    }

    #[test]
    fn test_diarize_mismatched_lengths() {
        let config = DiarizationConfig::default();
        let embeddings = vec![vec![0.0; 192]];
        let times = vec![(0, 1000), (1000, 2000)]; // One extra

        let result = diarize(&embeddings, &times, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_diarize_wrong_embedding_dim() {
        let config = DiarizationConfig {
            embedding_dim: 192,
            ..Default::default()
        };

        let embeddings = vec![vec![0.0; 100]]; // Wrong dim
        let times = vec![(0, 1000)];

        let result = diarize(&embeddings, &times, &config);
        assert!(result.is_err());
    }
}
