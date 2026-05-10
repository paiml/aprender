//! HELIX-IDEA-001 Phase 1 — `PersistentHnsw` save/load wrapper around
//! [`HNSWIndex`].
//!
//! Contract: `contracts/apr-hnsw-persistence-v1.yaml` (ACTIVE).
//! Pattern source: helix-db `helix_engine` LMDB-backed HNSW
//! (re-implemented; no code lift). Phase 1 ships round-trip identity
//! semantics; Phases 2-4 amend the contract with crash-safety, recall
//! threshold, and cold-open latency gates per
//! `docs/specifications/helix-db-feature-ideas.md` §2.1.
//!
//! # Example
//!
//! ```no_run
//! use aprender::index::persistent_hnsw::PersistentHnsw;
//! use aprender::primitives::Vector;
//!
//! // Open or create.
//! let mut idx = PersistentHnsw::open("vectors.bin", 16, 200).unwrap();
//! idx.add("doc1", Vector::from_slice(&[1.0, 0.0, 0.0]));
//! idx.add("doc2", Vector::from_slice(&[0.0, 1.0, 0.0]));
//! idx.flush().unwrap();
//!
//! // Reopen later.
//! let idx = PersistentHnsw::open("vectors.bin", 16, 200).unwrap();
//! let hits = idx.search(&Vector::from_slice(&[0.9, 0.1, 0.0]), 1);
//! assert_eq!(hits[0].0, "doc1");
//! ```

use crate::index::HNSWIndex;
use crate::primitives::Vector;
use std::fs;
use std::path::{Path, PathBuf};

/// Errors surfaced by [`PersistentHnsw`]. Hand-rolled (no `thiserror`)
/// because that crate is gated behind the `audio` feature in
/// `aprender-core/Cargo.toml`; promoting it for one error type is more
/// dep-tree churn than the two manual `impl`s below.
#[derive(Debug)]
pub enum PersistentHnswError {
    /// I/O failure while reading or writing the snapshot file.
    Io(std::io::Error),
    /// `bincode` encode/decode failure. Usually means the snapshot
    /// was produced by an incompatible version of the type or got
    /// truncated mid-write. Phase 2 (FALSIFY-HNSW-PERSIST-002) will
    /// guarantee that a clean error here, not silent corruption, is
    /// what callers see for any partial-write scenario.
    Decode(bincode::Error),
}

impl std::fmt::Display for PersistentHnswError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Decode(e) => write!(f, "decode: {e}"),
        }
    }
}

impl std::error::Error for PersistentHnswError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Decode(e) => Some(e),
        }
    }
}

impl From<std::io::Error> for PersistentHnswError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<bincode::Error> for PersistentHnswError {
    fn from(e: bincode::Error) -> Self {
        Self::Decode(e)
    }
}

/// On-disk wrapper around [`HNSWIndex`] with bincode-serialized
/// snapshots. Phase 1: full snapshot via [`Self::flush`]; reopen
/// rebuilds from snapshot bytes (the embedded graph is restored
/// directly — no per-vector re-insertion, so the result is byte-stable
/// against the original).
pub struct PersistentHnsw {
    inner: HNSWIndex,
    path: PathBuf,
    /// Track whether this handle has accepted writes since last flush.
    /// Surfaced via [`Self::is_dirty`] for callers that want to short-
    /// circuit redundant flushes.
    dirty: bool,
}

impl PersistentHnsw {
    /// Open an existing snapshot at `path`, or create a new index if
    /// the path does not exist.
    ///
    /// # Errors
    ///
    /// Returns `PersistentHnswError::Io` if the path exists but cannot
    /// be read, or `PersistentHnswError::Decode` if the file is not a
    /// valid `HNSWIndex` bincode snapshot.
    pub fn open<P: AsRef<Path>>(
        path: P,
        m: usize,
        ef_construction: usize,
    ) -> Result<Self, PersistentHnswError> {
        let path = path.as_ref().to_path_buf();
        let inner = if path.is_file() {
            let bytes = fs::read(&path)?;
            bincode::deserialize::<HNSWIndex>(&bytes)?
        } else {
            HNSWIndex::new(m, ef_construction, 0.0)
        };
        Ok(Self {
            inner,
            path,
            dirty: false,
        })
    }

    /// Add an item to the index. State change is in-memory only until
    /// [`Self::flush`] is called.
    pub fn add(&mut self, item_id: impl Into<String>, vector: Vector<f64>) {
        self.inner.add(item_id, vector);
        self.dirty = true;
    }

    /// Search for the `k` nearest neighbours to `query`. Delegates to
    /// the in-memory index; persistence is irrelevant to query
    /// behaviour.
    #[must_use]
    pub fn search(&self, query: &Vector<f64>, k: usize) -> Vec<(String, f64)> {
        self.inner.search(query, k)
    }

    /// Number of items in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True iff no items have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// True iff the in-memory state has accepted writes since the
    /// last successful [`Self::flush`].
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Flush the current in-memory index to disk as a single bincode
    /// snapshot. Phase 1: a single overwrite — atomicity and crash
    /// safety land with FALSIFY-HNSW-PERSIST-002 in Phase 2.
    ///
    /// # Errors
    ///
    /// Returns `PersistentHnswError::Io` if the file system rejects
    /// the write, or `PersistentHnswError::Decode` if the index
    /// itself fails to serialize (should not happen with the shipped
    /// types).
    pub fn flush(&mut self) -> Result<(), PersistentHnswError> {
        let bytes = bincode::serialize(&self.inner)?;
        fs::write(&self.path, bytes)?;
        self.dirty = false;
        Ok(())
    }

    /// Borrow the underlying [`HNSWIndex`] read-only. Useful when
    /// callers want to inspect index params (`m()`, `ef_construction()`)
    /// without taking ownership.
    #[must_use]
    pub fn inner(&self) -> &HNSWIndex {
        &self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn fixture(idx: &mut PersistentHnsw) {
        idx.add("a", Vector::from_slice(&[1.0, 0.0, 0.0]));
        idx.add("b", Vector::from_slice(&[0.0, 1.0, 0.0]));
        idx.add("c", Vector::from_slice(&[0.0, 0.0, 1.0]));
        idx.add("d", Vector::from_slice(&[0.5, 0.5, 0.0]));
    }

    #[test]
    fn open_creates_empty_when_path_does_not_exist() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("new.bin");
        let idx = PersistentHnsw::open(&path, 8, 64).unwrap();
        assert!(idx.is_empty());
        assert!(!idx.is_dirty());
    }

    #[test]
    fn add_marks_dirty_flush_clears() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("dirty.bin");
        let mut idx = PersistentHnsw::open(&path, 8, 64).unwrap();
        idx.add("x", Vector::from_slice(&[1.0, 0.0]));
        assert!(idx.is_dirty());
        idx.flush().unwrap();
        assert!(!idx.is_dirty());
        assert!(path.is_file());
    }

    #[test]
    fn flush_then_reopen_preserves_search_byte_stable() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("rt.bin");
        let mut idx = PersistentHnsw::open(&path, 8, 64).unwrap();
        fixture(&mut idx);
        let pre = idx.search(&Vector::from_slice(&[0.9, 0.1, 0.0]), 3);
        idx.flush().unwrap();
        drop(idx);

        let reopened = PersistentHnsw::open(&path, 8, 64).unwrap();
        let post = reopened.search(&Vector::from_slice(&[0.9, 0.1, 0.0]), 3);
        assert_eq!(pre, post);
        assert_eq!(reopened.len(), 4);
    }

    #[test]
    fn open_after_decode_failure_returns_error_not_panic() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("garbage.bin");
        fs::write(&path, b"not a bincode payload").unwrap();
        let result = PersistentHnsw::open(&path, 8, 64);
        assert!(matches!(result, Err(PersistentHnswError::Decode(_))));
    }
}
