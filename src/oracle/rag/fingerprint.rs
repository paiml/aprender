//! Document Fingerprinting for Poka-Yoke Stale Detection
//!
//! Uses BLAKE3 content hashing for content-addressable index invalidation.
//! Supports the Toyota Way principle of mistake-proofing (Poka-Yoke).

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Document fingerprint for change detection (Poka-Yoke)
///
/// Content-addressable storage pattern from Quinlan & Dorward (2002).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DocumentFingerprint {
    /// BLAKE3 hash of document content
    pub content_hash: [u8; 32],
    /// Hash of chunking parameters (for reproducibility)
    pub chunker_config_hash: [u8; 32],
    /// Hash of embedding model weights
    pub embedding_model_hash: [u8; 32],
    /// Timestamp of last successful index (Unix epoch ms)
    pub indexed_at: u64,
}

impl DocumentFingerprint {
    /// Create a new fingerprint from content
    pub fn new(content: &[u8], chunker_config: &ChunkerConfig, model_hash: [u8; 32]) -> Self {
        Self {
            content_hash: blake3_hash(content),
            chunker_config_hash: chunker_config.hash(),
            embedding_model_hash: model_hash,
            indexed_at: current_timestamp_ms(),
        }
    }

    /// Check if document needs reindexing (Poka-Yoke validation)
    ///
    /// Returns true if ANY component changed:
    /// - Content changed
    /// - Chunking config changed
    /// - Embedding model changed
    pub fn needs_reindex(&self, current: &Self) -> bool {
        self.content_hash != current.content_hash
            || self.chunker_config_hash != current.chunker_config_hash
            || self.embedding_model_hash != current.embedding_model_hash
    }

    /// Get age in seconds since indexing
    pub fn age_seconds(&self) -> u64 {
        let now = current_timestamp_ms();
        (now.saturating_sub(self.indexed_at)) / 1000
    }

    /// Check if fingerprint is stale (older than max_age_seconds)
    pub fn is_stale(&self, max_age_seconds: u64) -> bool {
        self.age_seconds() > max_age_seconds
    }
}

/// Chunker configuration for reproducible chunking
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChunkerConfig {
    /// Chunk size in tokens
    pub chunk_size: usize,
    /// Overlap between chunks
    pub chunk_overlap: usize,
    /// Separator list hash
    pub separators_hash: [u8; 32],
}

impl ChunkerConfig {
    /// Create a new chunker config
    pub fn new(chunk_size: usize, chunk_overlap: usize, separators: &[&str]) -> Self {
        let sep_bytes: Vec<u8> = separators.join("\n").into_bytes();
        Self { chunk_size, chunk_overlap, separators_hash: blake3_hash(&sep_bytes) }
    }

    /// Compute hash of this config
    pub fn hash(&self) -> [u8; 32] {
        let mut data = Vec::new();
        data.extend_from_slice(&self.chunk_size.to_le_bytes());
        data.extend_from_slice(&self.chunk_overlap.to_le_bytes());
        data.extend_from_slice(&self.separators_hash);
        blake3_hash(&data)
    }
}

impl Default for ChunkerConfig {
    fn default() -> Self {
        Self::new(512, 64, &["\n## ", "\n### ", "\nfn ", "\nimpl ", "\nstruct ", "\n\n", "\n", " "])
    }
}

/// Compute BLAKE3 hash of data
///
/// BLAKE3 chosen for:
/// - Speed: 4x faster than SHA-256
/// - Security: 256-bit security level
/// - Parallelism: Built-in SIMD acceleration
pub fn blake3_hash(data: &[u8]) -> [u8; 32] {
    // Use a simple hash for now - will integrate blake3 crate
    // This is a placeholder that still provides deterministic hashing
    let mut hash = [0u8; 32];

    // Simple deterministic hash based on content
    // In production, use blake3::hash(data).into()
    let mut state: u64 = 0xcbf2_9ce4_8422_2325; // FNV offset basis
    for &byte in data {
        state ^= byte as u64;
        state = state.wrapping_mul(0x0100_0000_01b3); // FNV prime
    }

    // Expand to 32 bytes
    for i in 0..4 {
        let chunk = state.wrapping_add(i as u64).to_le_bytes();
        hash[i * 8..(i + 1) * 8].copy_from_slice(&chunk);
    }

    hash
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_creation() {
        let content = b"Hello, World!";
        let config = ChunkerConfig::default();
        let model_hash = [1u8; 32];

        let fp = DocumentFingerprint::new(content, &config, model_hash);

        assert_ne!(fp.content_hash, [0u8; 32]);
        assert_ne!(fp.chunker_config_hash, [0u8; 32]);
        assert_eq!(fp.embedding_model_hash, model_hash);
        assert!(fp.indexed_at > 0);
    }

    #[test]
    fn test_fingerprint_content_change_detection() {
        let config = ChunkerConfig::default();
        let model_hash = [1u8; 32];

        let fp1 = DocumentFingerprint::new(b"content v1", &config, model_hash);
        let fp2 = DocumentFingerprint::new(b"content v2", &config, model_hash);

        assert!(fp1.needs_reindex(&fp2));
    }

    #[test]
    fn test_fingerprint_no_change() {
        let config = ChunkerConfig::default();
        let model_hash = [1u8; 32];

        let fp1 = DocumentFingerprint::new(b"same content", &config, model_hash);
        let fp2 = DocumentFingerprint::new(b"same content", &config, model_hash);

        // Content hash should match
        assert_eq!(fp1.content_hash, fp2.content_hash);
        // But timestamps differ, so needs_reindex compares hashes only
        assert!(!fp1.needs_reindex(&fp2));
    }

    #[test]
    fn test_fingerprint_config_change_detection() {
        let config1 = ChunkerConfig::new(512, 64, &["\n\n"]);
        let config2 = ChunkerConfig::new(256, 32, &["\n\n"]); // Different sizes
        let model_hash = [1u8; 32];

        let fp1 = DocumentFingerprint::new(b"same content", &config1, model_hash);
        let fp2 = DocumentFingerprint::new(b"same content", &config2, model_hash);

        assert!(fp1.needs_reindex(&fp2));
    }

    #[test]
    fn test_fingerprint_model_change_detection() {
        let config = ChunkerConfig::default();
        let model_hash1 = [1u8; 32];
        let model_hash2 = [2u8; 32]; // Different model

        let fp1 = DocumentFingerprint::new(b"same content", &config, model_hash1);
        let fp2 = DocumentFingerprint::new(b"same content", &config, model_hash2);

        assert!(fp1.needs_reindex(&fp2));
    }

    #[test]
    fn test_blake3_hash_deterministic() {
        let data = b"test data";
        let hash1 = blake3_hash(data);
        let hash2 = blake3_hash(data);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_blake3_hash_different_inputs() {
        let hash1 = blake3_hash(b"input 1");
        let hash2 = blake3_hash(b"input 2");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_chunker_config_hash_deterministic() {
        let config1 = ChunkerConfig::new(512, 64, &["\n\n", "\n"]);
        let config2 = ChunkerConfig::new(512, 64, &["\n\n", "\n"]);
        assert_eq!(config1.hash(), config2.hash());
    }

    #[test]
    fn test_chunker_config_different_params() {
        let config1 = ChunkerConfig::new(512, 64, &["\n\n"]);
        let config2 = ChunkerConfig::new(256, 64, &["\n\n"]);
        assert_ne!(config1.hash(), config2.hash());
    }

    #[test]
    fn test_fingerprint_age() {
        let config = ChunkerConfig::default();
        let model_hash = [1u8; 32];
        let fp = DocumentFingerprint::new(b"content", &config, model_hash);

        // Just created, age should be very small
        assert!(fp.age_seconds() < 2);
    }

    #[test]
    fn test_fingerprint_staleness() {
        let config = ChunkerConfig::default();
        let model_hash = [1u8; 32];
        let fp = DocumentFingerprint::new(b"content", &config, model_hash);

        // Just created, should not be stale
        assert!(!fp.is_stale(60)); // 1 minute threshold
    }
}
