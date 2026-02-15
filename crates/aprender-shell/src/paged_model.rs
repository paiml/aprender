//! Memory-Paged Markov Model for Large Shell Histories
//!
//! Uses aprender's bundle module for efficient memory management when
//! dealing with large shell histories that exceed available RAM.

use aprender::bundle::{BundleBuilder, BundleConfig, PagedBundle, PagingConfig, PagingStats};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

use crate::trie::Trie;

/// Minimum memory limit (1MB)
const MIN_MEMORY_LIMIT: usize = 1024 * 1024;

/// N-gram segment for paged storage.
///
/// Each segment contains n-grams for a specific context prefix,
/// allowing on-demand loading.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NgramSegment {
    /// Context prefix this segment covers (e.g., "git", "cargo")
    pub prefix: String,
    /// N-gram data: context -> (next_token -> count)
    pub ngrams: HashMap<String, HashMap<String, u32>>,
    /// Size estimate in bytes
    pub size_bytes: usize,
}

impl NgramSegment {
    /// Create a new empty segment.
    #[must_use]
    pub fn new(prefix: String) -> Self {
        Self {
            prefix,
            ngrams: HashMap::new(),
            size_bytes: 0,
        }
    }

    /// Add an n-gram to this segment.
    pub fn add(&mut self, context: String, next_token: String, count: u32) {
        let entry = self.ngrams.entry(context).or_default();
        *entry.entry(next_token).or_insert(0) += count;
        self.update_size();
    }

    /// Update size estimate.
    fn update_size(&mut self) {
        self.size_bytes = self
            .ngrams
            .iter()
            .map(|(k, v)| k.len() + v.keys().map(|k2| k2.len() + 4).sum::<usize>())
            .sum();
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Simple binary format: prefix_len(4) + prefix + ngram_count(4) + ngrams
        let mut bytes = Vec::new();

        // Prefix
        let prefix_bytes = self.prefix.as_bytes();
        bytes.extend(&(prefix_bytes.len() as u32).to_le_bytes());
        bytes.extend(prefix_bytes);

        // N-gram count
        bytes.extend(&(self.ngrams.len() as u32).to_le_bytes());

        for (context, next_tokens) in &self.ngrams {
            // Context
            let ctx_bytes = context.as_bytes();
            bytes.extend(&(ctx_bytes.len() as u32).to_le_bytes());
            bytes.extend(ctx_bytes);

            // Next tokens count
            bytes.extend(&(next_tokens.len() as u32).to_le_bytes());

            for (token, count) in next_tokens {
                // Token
                let tok_bytes = token.as_bytes();
                bytes.extend(&(tok_bytes.len() as u32).to_le_bytes());
                bytes.extend(tok_bytes);
                // Count
                bytes.extend(&count.to_le_bytes());
            }
        }

        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> std::io::Result<Self> {
        let mut pos = 0;

        // Helper to read 4 bytes as u32
        let read_u32 = |data: &[u8], offset: usize| -> std::io::Result<u32> {
            let slice = data
                .get(offset..offset + 4)
                .ok_or_else(|| std::io::Error::other("Truncated segment data"))?;
            let arr: [u8; 4] = slice
                .try_into()
                .map_err(|_| std::io::Error::other("Invalid byte slice"))?;
            Ok(u32::from_le_bytes(arr))
        };

        // Read prefix
        let prefix_len = read_u32(bytes, pos)? as usize;
        pos += 4;

        if bytes.len() < pos + prefix_len {
            return Err(std::io::Error::other("Truncated prefix"));
        }
        let prefix = String::from_utf8_lossy(&bytes[pos..pos + prefix_len]).to_string();
        pos += prefix_len;

        // Read n-gram count
        let ngram_count = read_u32(bytes, pos)? as usize;
        pos += 4;

        let mut ngrams = HashMap::with_capacity(ngram_count);

        for _ in 0..ngram_count {
            // Read context
            let ctx_len = read_u32(bytes, pos)? as usize;
            pos += 4;

            if bytes.len() < pos + ctx_len {
                return Err(std::io::Error::other("Truncated context"));
            }
            let context = String::from_utf8_lossy(&bytes[pos..pos + ctx_len]).to_string();
            pos += ctx_len;

            // Read next tokens count
            let token_count = read_u32(bytes, pos)? as usize;
            pos += 4;

            let mut next_tokens = HashMap::with_capacity(token_count);

            for _ in 0..token_count {
                // Read token
                let tok_len = read_u32(bytes, pos)? as usize;
                pos += 4;

                if bytes.len() < pos + tok_len {
                    return Err(std::io::Error::other("Truncated token"));
                }
                let token = String::from_utf8_lossy(&bytes[pos..pos + tok_len]).to_string();
                pos += tok_len;

                // Read count
                let count = read_u32(bytes, pos)?;
                pos += 4;

                next_tokens.insert(token, count);
            }

            ngrams.insert(context, next_tokens);
        }

        let mut segment = Self {
            prefix,
            ngrams,
            size_bytes: 0,
        };
        segment.update_size();
        Ok(segment)
    }
}

/// Model metadata stored in the bundle manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PagedModelMetadata {
    /// N-gram size
    pub n: usize,
    /// Total commands trained on
    pub total_commands: usize,
    /// Number of segments
    pub segment_count: usize,
    /// Command frequency map (kept in memory - relatively small)
    pub command_freq: HashMap<String, u32>,
    /// Segment prefixes for index lookup
    pub segment_prefixes: Vec<String>,
}

/// Memory-paged Markov model for shell command prediction.
///
/// Uses aprender's bundle module to store n-gram data on disk and
/// load segments on-demand, enabling handling of large shell histories
/// without exhausting RAM.
pub struct PagedMarkovModel {
    /// N-gram size
    n: usize,
    /// Memory limit in bytes
    memory_limit: usize,
    /// Metadata
    metadata: PagedModelMetadata,
    /// Paged bundle (when loaded from file)
    bundle: Option<PagedBundle>,
    /// In-memory segments (for training/small models)
    segments: HashMap<String, NgramSegment>,
    /// Prefix trie for fast lookup
    trie: Option<Trie>,
    /// Path to bundle file (if loaded)
    bundle_path: Option<std::path::PathBuf>,
}

include!("paged_model_part_02.rs");
include!("paged_model_part_03.rs");
