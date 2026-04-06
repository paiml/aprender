//! RAG Index Persistence - Section 9.7 of oracle-mode-spec.md
//!
//! Persistent storage for the RAG index at `~/.cache/batuta/rag/`.
//! Uses JSON format with BLAKE3 checksums for integrity validation.
//!
//! # Toyota Production System Principles
//!
//! - **Jidoka**: Graceful degradation on corruption (rebuild instead of crash)
//! - **Poka-Yoke**: Version compatibility prevents format mismatches
//! - **Heijunka**: Incremental updates via fingerprint-based invalidation
//! - **Muda**: JSON for debugging, future P2 uses bincode

use super::fingerprint::{blake3_hash, DocumentFingerprint};
use super::types::{Bm25Config, RrfConfig};
use super::IndexedDocument;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Index format version (semver major.minor.patch)
///
/// 1.1.0: Added chunk_contents, stemming, stop words, TF-IDF dense search
pub const INDEX_VERSION: &str = "1.1.0";

/// Cache directory relative to user cache
const CACHE_SUBDIR: &str = "batuta/rag";

/// Manifest filename
const MANIFEST_FILE: &str = "manifest.json";

/// Index filename
const INDEX_FILE: &str = "index.json";

/// Documents filename
const DOCUMENTS_FILE: &str = "documents.json";

/// Fingerprints-only filename (lightweight, for `is_index_current` checks)
const FINGERPRINTS_FILE: &str = "fingerprints.json";

/// Persisted RAG index manifest
///
/// Contains metadata and checksums for integrity validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagManifest {
    /// Index format version (semver)
    pub version: String,
    /// BLAKE3 checksum of index.json
    pub index_checksum: [u8; 32],
    /// BLAKE3 checksum of documents.json
    pub docs_checksum: [u8; 32],
    /// Indexed corpus sources
    pub sources: Vec<CorpusSource>,
    /// Unix timestamp when indexed (milliseconds)
    pub indexed_at: u64,
    /// Batuta version that created this index
    pub batuta_version: String,
}

/// Source corpus information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusSource {
    /// Corpus identifier (e.g., "trueno", "hf-ground-truth-corpus")
    pub id: String,
    /// Git commit hash at index time (if available)
    pub commit: Option<String>,
    /// Number of documents indexed from this source
    pub doc_count: usize,
    /// Number of chunks indexed from this source
    pub chunk_count: usize,
}

/// Serializable inverted index state
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedIndex {
    /// Inverted index: term -> (doc_id -> term_frequency)
    pub inverted_index: HashMap<String, HashMap<String, usize>>,
    /// Document lengths for BM25
    pub doc_lengths: HashMap<String, usize>,
    /// BM25 configuration
    pub bm25_config: Bm25Config,
    /// RRF configuration
    pub rrf_config: RrfConfig,
    /// Average document length
    pub avg_doc_length: f64,
}

/// Serializable document metadata
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PersistedDocuments {
    /// Documents by ID
    pub documents: HashMap<String, IndexedDocument>,
    /// Fingerprints for change detection
    pub fingerprints: HashMap<String, DocumentFingerprint>,
    /// Total chunks indexed
    pub total_chunks: usize,
    /// Chunk content snippets (first 200 chars) for result display
    #[serde(default)]
    pub chunk_contents: HashMap<String, String>,
}

/// Persistence errors
#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Checksum mismatch (Jidoka halt)
    #[error("Checksum mismatch for {file}: expected {expected:x?}, got {actual:x?}")]
    ChecksumMismatch { file: String, expected: [u8; 32], actual: [u8; 32] },

    /// Version mismatch
    #[error("Version mismatch: index version {index_version}, expected {expected_version}")]
    VersionMismatch { index_version: String, expected_version: String },

    /// Cache directory not found
    #[error("Cache directory not found")]
    CacheDirNotFound,

    /// Manifest not found (no cached index)
    #[error("No cached index found")]
    NoCachedIndex,
}

/// RAG index persistence manager
///
/// Handles saving and loading the RAG index to/from disk.
#[derive(Debug)]
pub struct RagPersistence {
    /// Cache path
    cache_path: PathBuf,
}

impl RagPersistence {
    /// Create persistence manager with default cache path
    ///
    /// Default path: `~/.cache/batuta/rag/`
    pub fn new() -> Self {
        Self { cache_path: Self::default_cache_path() }
    }

    /// Create persistence manager with custom cache path
    pub fn with_path(path: PathBuf) -> Self {
        Self { cache_path: path }
    }

    /// Get default cache path
    ///
    /// Uses `dirs::cache_dir()` for platform-specific cache location.
    fn default_cache_path() -> PathBuf {
        #[cfg(feature = "native")]
        {
            dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".cache")).join(CACHE_SUBDIR)
        }
        #[cfg(not(feature = "native"))]
        {
            PathBuf::from(".cache").join(CACHE_SUBDIR)
        }
    }

    /// Get the cache path
    pub fn cache_path(&self) -> &Path {
        &self.cache_path
    }

    /// Save index to disk using two-phase commit
    ///
    /// Writes three files with crash safety:
    /// - **Prepare phase**: Write all `.tmp` files (crash here = old cache intact)
    /// - **Commit phase**: Rename all 3, manifest LAST (crash before manifest
    ///   rename = old manifest still valid or checksum mismatch triggers rebuild)
    ///
    /// Files written:
    /// - `manifest.json`: Version and checksums
    /// - `index.json`: Inverted index data
    /// - `documents.json`: Document metadata
    pub fn save(
        &self,
        index: &PersistedIndex,
        docs: &PersistedDocuments,
        sources: Vec<CorpusSource>,
    ) -> Result<(), PersistenceError> {
        // Ensure cache directory exists
        fs::create_dir_all(&self.cache_path)?;

        // Clean up any orphaned .tmp files from a previous crashed save
        self.cleanup_tmp_files();

        // Serialize index and documents
        let index_json = serde_json::to_string_pretty(index)?;
        let docs_json = serde_json::to_string_pretty(docs)?;

        // Serialize fingerprints separately for fast is_index_current checks
        let fingerprints_json = serde_json::to_string_pretty(&docs.fingerprints)?;

        // Compute checksums
        let index_checksum = blake3_hash(index_json.as_bytes());
        let docs_checksum = blake3_hash(docs_json.as_bytes());

        // Create manifest
        let manifest = RagManifest {
            version: INDEX_VERSION.to_string(),
            index_checksum,
            docs_checksum,
            sources,
            indexed_at: current_timestamp_ms(),
            batuta_version: env!("CARGO_PKG_VERSION").to_string(),
        };
        let manifest_json = serde_json::to_string_pretty(&manifest)?;

        // Phase 1: Prepare — write all .tmp files (crash here = old cache intact)
        self.prepare_write(INDEX_FILE, index_json.as_bytes())?;
        self.prepare_write(DOCUMENTS_FILE, docs_json.as_bytes())?;
        self.prepare_write(FINGERPRINTS_FILE, fingerprints_json.as_bytes())?;
        self.prepare_write(MANIFEST_FILE, manifest_json.as_bytes())?;

        // Phase 2: Commit — rename all, manifest LAST
        // Crash before manifest rename = old manifest checksums won't match new
        // data files, which triggers graceful rebuild on next load().
        self.commit_rename(INDEX_FILE)?;
        self.commit_rename(DOCUMENTS_FILE)?;
        self.commit_rename(FINGERPRINTS_FILE)?;
        self.commit_rename(MANIFEST_FILE)?;

        Ok(())
    }

    /// Load index from disk
    ///
    /// Returns `None` if no cached index exists or if the cache is corrupted
    /// (IO error, checksum mismatch, invalid JSON). Corruption triggers a
    /// warning to stderr so the caller can rebuild gracefully.
    ///
    /// Returns `Err` only for `VersionMismatch` (incompatible format requires
    /// a code update, not just a re-index).
    pub fn load(
        &self,
    ) -> Result<Option<(PersistedIndex, PersistedDocuments, RagManifest)>, PersistenceError> {
        let manifest_path = self.cache_path.join(MANIFEST_FILE);

        // Check if manifest exists
        if !manifest_path.exists() {
            return Ok(None);
        }

        // Load manifest — graceful on IO/JSON errors
        let manifest_json = match fs::read_to_string(&manifest_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Warning: failed to read RAG manifest, will rebuild: {e}");
                return Ok(None);
            }
        };
        let manifest: RagManifest = match serde_json::from_str(&manifest_json) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Warning: corrupt RAG manifest JSON, will rebuild: {e}");
                return Ok(None);
            }
        };

        // Validate version (Poka-Yoke) — hard error, needs code update
        self.validate_version(&manifest)?;

        // Load and validate index — graceful on IO/JSON/checksum errors
        let index_json = match fs::read_to_string(self.cache_path.join(INDEX_FILE)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Warning: failed to read RAG index file, will rebuild: {e}");
                return Ok(None);
            }
        };
        if let Err(e) = self.validate_checksum(&index_json, manifest.index_checksum, "index.json") {
            eprintln!("Warning: {e}, will rebuild");
            return Ok(None);
        }
        let index: PersistedIndex = match serde_json::from_str(&index_json) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("Warning: corrupt RAG index JSON, will rebuild: {e}");
                return Ok(None);
            }
        };

        // Load and validate documents — graceful on IO/JSON/checksum errors
        let docs_json = match fs::read_to_string(self.cache_path.join(DOCUMENTS_FILE)) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Warning: failed to read RAG documents file, will rebuild: {e}");
                return Ok(None);
            }
        };
        if let Err(e) = self.validate_checksum(&docs_json, manifest.docs_checksum, "documents.json")
        {
            eprintln!("Warning: {e}, will rebuild");
            return Ok(None);
        }
        let docs: PersistedDocuments = match serde_json::from_str(&docs_json) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Warning: corrupt RAG documents JSON, will rebuild: {e}");
                return Ok(None);
            }
        };

        Ok(Some((index, docs, manifest)))
    }

    /// Load only fingerprints for fast `is_index_current` checks.
    ///
    /// Reads ~KB fingerprints.json instead of ~600MB (index.json + documents.json).
    /// Falls back to full `load()` if fingerprints.json doesn't exist (pre-upgrade cache).
    pub fn load_fingerprints_only(
        &self,
    ) -> Result<Option<HashMap<String, DocumentFingerprint>>, PersistenceError> {
        let fp_path = self.cache_path.join(FINGERPRINTS_FILE);

        if fp_path.exists() {
            let fp_json = match fs::read_to_string(&fp_path) {
                Ok(s) => s,
                Err(_) => return self.load_fingerprints_fallback(),
            };
            match serde_json::from_str(&fp_json) {
                Ok(fps) => return Ok(Some(fps)),
                Err(_) => return self.load_fingerprints_fallback(),
            }
        }

        // Fallback: fingerprints.json doesn't exist (pre-upgrade cache)
        self.load_fingerprints_fallback()
    }

    /// Fallback: extract fingerprints from full documents.json load
    fn load_fingerprints_fallback(
        &self,
    ) -> Result<Option<HashMap<String, DocumentFingerprint>>, PersistenceError> {
        self.load().map(|opt| opt.map(|(_, docs, _)| docs.fingerprints))
    }

    /// Save only fingerprints.json for fast `is_index_current` checks.
    ///
    /// Used by the SQLite indexing path to persist fingerprints without
    /// writing the full 600MB JSON index/documents files.
    pub fn save_fingerprints_only(
        &self,
        fingerprints: &HashMap<String, DocumentFingerprint>,
    ) -> Result<(), PersistenceError> {
        fs::create_dir_all(&self.cache_path)?;
        let fingerprints_json = serde_json::to_string_pretty(fingerprints)?;
        self.prepare_write(FINGERPRINTS_FILE, fingerprints_json.as_bytes())?;
        self.commit_rename(FINGERPRINTS_FILE)?;
        Ok(())
    }

    /// Clear cached index
    pub fn clear(&self) -> Result<(), PersistenceError> {
        if self.cache_path.exists() {
            // Remove individual files
            let _ = fs::remove_file(self.cache_path.join(MANIFEST_FILE));
            let _ = fs::remove_file(self.cache_path.join(INDEX_FILE));
            let _ = fs::remove_file(self.cache_path.join(DOCUMENTS_FILE));
            let _ = fs::remove_file(self.cache_path.join(FINGERPRINTS_FILE));

            // Try to remove directory if empty
            let _ = fs::remove_dir(&self.cache_path);
        }
        Ok(())
    }

    /// Get index statistics without full load
    pub fn stats(&self) -> Result<Option<RagManifest>, PersistenceError> {
        let manifest_path = self.cache_path.join(MANIFEST_FILE);

        if !manifest_path.exists() {
            return Ok(None);
        }

        let manifest_json = fs::read_to_string(&manifest_path)?;
        let manifest: RagManifest = serde_json::from_str(&manifest_json)?;

        Ok(Some(manifest))
    }

    /// Phase 1: Write data to a `.tmp` file (prepare)
    fn prepare_write(&self, filename: &str, data: &[u8]) -> Result<(), io::Error> {
        let tmp_path = self.cache_path.join(format!("{}.tmp", filename));

        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(data)?;
        file.sync_all()?;

        Ok(())
    }

    /// Phase 2: Rename `.tmp` file to final path (commit)
    fn commit_rename(&self, filename: &str) -> Result<(), io::Error> {
        let tmp_path = self.cache_path.join(format!("{}.tmp", filename));
        let final_path = self.cache_path.join(filename);

        fs::rename(&tmp_path, &final_path)?;

        Ok(())
    }

    /// Remove orphaned `.tmp` files from a previous crashed save
    fn cleanup_tmp_files(&self) {
        for filename in &[MANIFEST_FILE, INDEX_FILE, DOCUMENTS_FILE] {
            let tmp_path = self.cache_path.join(format!("{}.tmp", filename));
            let _ = fs::remove_file(tmp_path);
        }
    }

    /// Validate version compatibility (Poka-Yoke)
    fn validate_version(&self, manifest: &RagManifest) -> Result<(), PersistenceError> {
        // Parse versions
        let index_parts: Vec<&str> = manifest.version.split('.').collect();
        let expected_parts: Vec<&str> = INDEX_VERSION.split('.').collect();

        // Major version must match for compatibility
        if index_parts.first() != expected_parts.first() {
            return Err(PersistenceError::VersionMismatch {
                index_version: manifest.version.clone(),
                expected_version: INDEX_VERSION.to_string(),
            });
        }

        Ok(())
    }

    /// Validate checksum (Jidoka)
    fn validate_checksum(
        &self,
        data: &str,
        expected: [u8; 32],
        filename: &str,
    ) -> Result<(), PersistenceError> {
        let actual = blake3_hash(data.as_bytes());

        if actual != expected {
            return Err(PersistenceError::ChecksumMismatch {
                file: filename.to_string(),
                expected,
                actual,
            });
        }

        Ok(())
    }
}

impl Default for RagPersistence {
    fn default() -> Self {
        Self::new()
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
#[path = "persistence_tests.rs"]
mod tests;
