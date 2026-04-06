//! SQLite+FTS5 persistent storage backend for RAG indices.
//!
//! Provides `SqliteIndex` (implements `SparseIndex`) and `SqliteStore`
//! (convenience wrapper for document + chunk persistence).
//!
//! This module replaces in-memory HashMap-based indices with SQLite-backed
//! storage using FTS5 for BM25 ranking (Robertson & Zaragoza, 2009).
//!
//! # Performance Contract
//!
//! Median search latency: 10–50 ms on a 5000+ document corpus with warm
//! page cache (see sqlite-rag-integration spec, Section 3.1).

pub mod fts;
pub mod schema;

use crate::index::SparseIndex;
use crate::{Chunk, ChunkId, Document, Result};
use rusqlite::Connection;
use std::path::Path;
use std::sync::Mutex;

/// SQLite-backed sparse index using FTS5 for BM25 search.
///
/// Unlike `BM25Index` (in-memory HashMap), this persists to disk and
/// delegates BM25 scoring to SQLite's FTS5 extension.
///
/// The `Connection` is wrapped in a `Mutex` to satisfy the `Send + Sync`
/// bounds required by `SparseIndex`. `Mutex<T>` is `Sync` when `T: Send`,
/// and `rusqlite::Connection` is `Send`. SQLite in WAL mode supports
/// concurrent readers via separate connections; this single-connection
/// design serializes access within one process.
pub struct SqliteIndex {
    conn: Mutex<Connection>,
}

// Mutex<Connection> is automatically Send+Sync because Connection: Send.
// No unsafe impl needed.

impl std::fmt::Debug for SqliteIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteIndex").finish_non_exhaustive()
    }
}

/// Helper to map mutex poison errors.
fn lock_err<T>(e: &std::sync::PoisonError<T>) -> crate::Error {
    crate::Error::Query(format!("Mutex poisoned: {e}"))
}

impl SqliteIndex {
    /// Open or create an index at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path.as_ref())
            .map_err(|e| crate::Error::Query(format!("Failed to open SQLite database: {e}")))?;
        schema::initialize(&conn)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Open an in-memory index (for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| crate::Error::Query(format!("Failed to open in-memory database: {e}")))?;
        schema::initialize(&conn)?;
        Ok(Self { conn: Mutex::new(conn) })
    }

    /// Get document count.
    pub fn document_count(&self) -> Result<usize> {
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM documents", [], |r| r.get(0))
            .map_err(|e| crate::Error::Query(format!("Failed to count documents: {e}")))?;
        Ok(count as usize)
    }

    /// Get chunk count.
    pub fn chunk_count(&self) -> Result<usize> {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!();
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))
            .map_err(|e| crate::Error::Query(format!("Failed to count chunks: {e}")))?;
        Ok(count as usize)
    }

    /// Check if a document needs reindexing by fingerprint.
    pub fn needs_reindex(&self, path: &str, hash: &[u8; 32]) -> Result<bool> {
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        let stored: Option<Vec<u8>> = conn
            .query_row("SELECT blake3_hash FROM fingerprints WHERE doc_path = ?1", [path], |row| {
                row.get(0)
            })
            .ok();

        match stored {
            Some(stored_hash) => Ok(stored_hash.as_slice() != hash),
            None => Ok(true),
        }
    }

    /// Batch-insert a document and its chunks within a transaction.
    pub fn insert_document(
        &self,
        doc_id: &str,
        title: Option<&str>,
        source: Option<&str>,
        content: &str,
        chunks: &[(String, String)],
        fingerprint: Option<(&str, &[u8; 32])>,
    ) -> Result<()> {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(doc_id.as_bytes());

        let mut conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        let tx = conn
            .transaction()
            .map_err(|e| crate::Error::Query(format!("Failed to begin transaction: {e}")))?;

        // Delete old document's chunks first (fires FTS5 sync triggers),
        // then delete the document itself.
        tx.execute("DELETE FROM chunks WHERE doc_id = ?1", [doc_id])
            .map_err(|e| crate::Error::Query(format!("Failed to delete old chunks: {e}")))?;
        tx.execute("DELETE FROM documents WHERE id = ?1", [doc_id])
            .map_err(|e| crate::Error::Query(format!("Failed to delete old document: {e}")))?;

        tx.execute(
            "INSERT INTO documents (id, title, source, content, chunk_count) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![doc_id, title, source, content, chunks.len() as i64],
        )
        .map_err(|e| crate::Error::Query(format!("Failed to insert document: {e}")))?;

        {
            let mut stmt = tx
                .prepare_cached(
                    "INSERT OR REPLACE INTO chunks (id, doc_id, content, position) VALUES (?1, ?2, ?3, ?4)",
                )
                .map_err(|e| crate::Error::Query(format!("Failed to prepare chunk insert: {e}")))?;

            for (i, (chunk_id, chunk_content)) in chunks.iter().enumerate() {
                stmt.execute(rusqlite::params![chunk_id, doc_id, chunk_content, i as i64])
                    .map_err(|e| crate::Error::Query(format!("Failed to insert chunk: {e}")))?;
            }
        }

        if let Some((path, hash)) = fingerprint {
            tx.execute(
                "INSERT OR REPLACE INTO fingerprints (doc_path, blake3_hash, chunk_count) VALUES (?1, ?2, ?3)",
                rusqlite::params![path, hash.as_slice(), chunks.len() as i64],
            )
            .map_err(|e| crate::Error::Query(format!("Failed to update fingerprint: {e}")))?;
        }

        tx.commit()
            .map_err(|e| crate::Error::Query(format!("Failed to commit transaction: {e}")))?;

        Ok(())
    }

    /// Remove a document and its chunks.
    ///
    /// Explicitly deletes chunks first (which fires FTS5 sync triggers),
    /// then deletes the document row.
    pub fn remove_document(&self, doc_id: &str) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        conn.execute("DELETE FROM chunks WHERE doc_id = ?1", [doc_id])
            .map_err(|e| crate::Error::Query(format!("Failed to delete chunks: {e}")))?;
        conn.execute("DELETE FROM documents WHERE id = ?1", [doc_id])
            .map_err(|e| crate::Error::Query(format!("Failed to remove document: {e}")))?;
        Ok(())
    }

    /// List all tracked fingerprints (path → blake3 hash).
    ///
    /// Used by incremental indexing to detect deleted or changed files.
    pub fn list_fingerprints(&self) -> Result<Vec<(String, Vec<u8>)>> {
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        let mut stmt = conn
            .prepare("SELECT doc_path, blake3_hash FROM fingerprints")
            .map_err(|e| crate::Error::Query(format!("Failed to list fingerprints: {e}")))?;
        let rows = stmt
            .query_map([], |row| {
                let path: String = row.get(0)?;
                let hash: Vec<u8> = row.get(1)?;
                Ok((path, hash))
            })
            .map_err(|e| crate::Error::Query(format!("Failed to query fingerprints: {e}")))?;
        let mut results = Vec::new();
        for row in rows {
            results.push(
                row.map_err(|e| crate::Error::Query(format!("Failed to read fingerprint: {e}")))?,
            );
        }
        Ok(results)
    }

    /// Remove all documents (and their chunks) with a given source path.
    ///
    /// Used by incremental indexing to remove stale documents before re-inserting.
    pub fn remove_by_source(&self, source: &str) -> Result<usize> {
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        // Find doc IDs with this source
        let mut stmt = conn
            .prepare("SELECT id FROM documents WHERE source = ?1")
            .map_err(|e| crate::Error::Query(format!("Failed to find docs by source: {e}")))?;
        let ids: Vec<String> = stmt
            .query_map([source], |row| row.get(0))
            .map_err(|e| crate::Error::Query(format!("Failed to query docs: {e}")))?
            .filter_map(|r| r.ok())
            .collect();

        for doc_id in &ids {
            conn.execute("DELETE FROM chunks WHERE doc_id = ?1", [doc_id])
                .map_err(|e| crate::Error::Query(format!("Failed to delete chunks: {e}")))?;
            conn.execute("DELETE FROM documents WHERE id = ?1", [doc_id])
                .map_err(|e| crate::Error::Query(format!("Failed to delete document: {e}")))?;
        }

        // Remove fingerprint
        conn.execute("DELETE FROM fingerprints WHERE doc_path = ?1", [source])
            .map_err(|e| crate::Error::Query(format!("Failed to delete fingerprint: {e}")))?;

        Ok(ids.len())
    }

    /// FTS5 BM25 search. Returns results ordered by descending relevance.
    pub fn search_fts(&self, query: &str, k: usize) -> Result<Vec<fts::FtsResult>> {
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        fts::search(&conn, query, k)
    }

    /// Get chunk content by ID.
    pub fn get_chunk(&self, chunk_id: &str) -> Result<Option<String>> {
        // Contract: configuration-v1.yaml precondition (pv codegen)
        contract_pre_configuration!(chunk_id.as_bytes());
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        let content: Option<String> = conn
            .query_row("SELECT content FROM chunks WHERE id = ?1", [chunk_id], |row| row.get(0))
            .ok();
        Ok(content)
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Result<Option<String>> {
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        let value: Option<String> = conn
            .query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| row.get(0))
            .ok();
        Ok(value)
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&self, key: &str, value: &str) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        conn.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?1, ?2)", [key, value])
            .map_err(|e| crate::Error::Query(format!("Failed to set metadata: {e}")))?;
        Ok(())
    }

    /// Vacuum and optimize the database.
    pub fn optimize(&self) -> Result<()> {
        let conn = self.conn.lock().map_err(|e| lock_err(&e))?;
        fts::optimize(&conn)?;
        conn.execute_batch("VACUUM;")
            .map_err(|e| crate::Error::Query(format!("VACUUM failed: {e}")))?;
        Ok(())
    }
}

impl SparseIndex for SqliteIndex {
    fn add(&mut self, chunk: &Chunk) {
        let doc_id = chunk.document_id.to_string();
        let chunk_id = chunk.id.to_string();
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute(
                "INSERT OR IGNORE INTO documents (id, content) VALUES (?1, '')",
                [&doc_id],
            );
            let _ = conn.execute(
                "INSERT OR REPLACE INTO chunks (id, doc_id, content, position) VALUES (?1, ?2, ?3, 0)",
                rusqlite::params![chunk_id, doc_id, chunk.content],
            );
        }
    }

    fn add_batch(&mut self, chunks: &[Chunk]) {
        let Ok(mut conn) = self.conn.lock() else {
            return;
        };
        let Ok(tx) = conn.transaction() else {
            return;
        };

        // Track position per document for UNIQUE(doc_id, position)
        let mut doc_positions: std::collections::HashMap<String, i64> =
            std::collections::HashMap::new();

        for chunk in chunks {
            let doc_id = chunk.document_id.to_string();
            let chunk_id = chunk.id.to_string();
            let pos = doc_positions.entry(doc_id.clone()).or_insert(0);
            let _ = tx.execute(
                "INSERT OR IGNORE INTO documents (id, content) VALUES (?1, '')",
                [&doc_id],
            );
            let _ = tx.execute(
                "INSERT OR REPLACE INTO chunks (id, doc_id, content, position) VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![chunk_id, doc_id, chunk.content, *pos],
            );
            *pos += 1;
        }

        let _ = tx.commit();
    }

    fn search(&self, query: &str, k: usize) -> Vec<(ChunkId, f32)> {
        let Ok(conn) = self.conn.lock() else {
            return Vec::new();
        };
        let Ok(results) = fts::search(&conn, query, k) else {
            return Vec::new();
        };

        results
            .into_iter()
            .filter_map(|r| {
                uuid::Uuid::parse_str(&r.chunk_id).ok().map(|uuid| (ChunkId(uuid), r.score as f32))
            })
            .collect()
    }

    fn remove(&mut self, chunk_id: ChunkId) {
        let id_str = chunk_id.to_string();
        if let Ok(conn) = self.conn.lock() {
            let _ = conn.execute("DELETE FROM chunks WHERE id = ?1", [&id_str]);
        }
    }

    fn len(&self) -> usize {
        self.chunk_count().unwrap_or(0)
    }
}

// --- SqliteStore: convenience wrapper ---

/// Statistics about the SQLite store.
#[derive(Debug, Clone)]
pub struct StoreStats {
    /// Number of documents indexed.
    pub document_count: usize,
    /// Number of chunks indexed.
    pub chunk_count: usize,
    /// Number of fingerprints tracked.
    pub fingerprint_count: usize,
    /// Database file size in bytes (0 for in-memory).
    pub db_size_bytes: u64,
}

/// Combined document store + BM25 index backed by SQLite.
///
/// Replaces the pattern of `BM25Index` + `VectorStore` + JSON persistence
/// for users who want disk-backed RAG without managing separate components.
pub struct SqliteStore {
    index: SqliteIndex,
    path: Option<std::path::PathBuf>,
}

impl std::fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStore").field("path", &self.path).finish_non_exhaustive()
    }
}

impl SqliteStore {
    /// Open or create a store at the given path.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let index = SqliteIndex::open(&path)?;
        Ok(Self { index, path: Some(path) })
    }

    /// Open an in-memory store (for testing).
    pub fn open_in_memory() -> Result<Self> {
        let index = SqliteIndex::open_in_memory()?;
        Ok(Self { index, path: None })
    }

    /// Index a document with its pre-chunked content.
    pub fn index_document(
        &self,
        doc: &Document,
        chunks: &[Chunk],
        fingerprint: Option<(&str, &[u8; 32])>,
    ) -> Result<()> {
        let doc_id = doc.id.to_string();
        let chunk_pairs: Vec<(String, String)> =
            chunks.iter().map(|c| (c.id.to_string(), c.content.clone())).collect();

        self.index.insert_document(
            &doc_id,
            doc.title.as_deref(),
            doc.source.as_deref(),
            &doc.content,
            &chunk_pairs,
            fingerprint,
        )
    }

    /// Search with BM25 and return results.
    ///
    /// **Performance contract:** Median latency 10–50 ms (spec Section 3.1).
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<fts::FtsResult>> {
        self.index.search_fts(query, k)
    }

    /// Check if a document needs reindexing by fingerprint.
    pub fn needs_reindex(&self, path: &str, hash: &[u8; 32]) -> Result<bool> {
        self.index.needs_reindex(path, hash)
    }

    /// List all tracked fingerprints.
    pub fn list_fingerprints(&self) -> Result<Vec<(String, Vec<u8>)>> {
        self.index.list_fingerprints()
    }

    /// Remove all documents with a given source path.
    pub fn remove_by_source(&self, source: &str) -> Result<usize> {
        self.index.remove_by_source(source)
    }

    /// Get store statistics.
    pub fn stats(&self) -> Result<StoreStats> {
        let db_size_bytes = self
            .path
            .as_ref()
            .and_then(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .unwrap_or(0);

        Ok(StoreStats {
            document_count: self.index.document_count()?,
            chunk_count: self.index.chunk_count()?,
            fingerprint_count: self.fingerprint_count()?,
            db_size_bytes,
        })
    }

    /// Get the number of tracked fingerprints.
    fn fingerprint_count(&self) -> Result<usize> {
        let conn = self.index.conn.lock().map_err(|e| lock_err(&e))?;
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM fingerprints", [], |r| r.get(0))
            .map_err(|e| crate::Error::Query(format!("Failed to count fingerprints: {e}")))?;
        Ok(count as usize)
    }

    /// Get/set metadata.
    pub fn get_metadata(&self, key: &str) -> Result<Option<String>> {
        self.index.get_metadata(key)
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&self, key: &str, value: &str) -> Result<()> {
        self.index.set_metadata(key, value)
    }

    /// Optimize the database (VACUUM + FTS5 merge).
    pub fn optimize(&self) -> Result<()> {
        self.index.optimize()
    }

    /// Get a reference to the underlying SqliteIndex.
    pub fn as_index(&self) -> &SqliteIndex {
        &self.index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Document, DocumentId};

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    fn make_chunk(doc_id: DocumentId, content: &str) -> Chunk {
        Chunk {
            id: ChunkId::new(),
            document_id: doc_id,
            content: content.to_string(),
            start_offset: 0,
            end_offset: content.len(),
            metadata: crate::ChunkMetadata::default(),
            embedding: None,
        }
    }

    // --- SqliteIndex tests ---

    #[test]
    fn test_index_roundtrip() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        idx.insert_document(
            "doc1",
            Some("Test Doc"),
            Some("/test.md"),
            "full content here",
            &[
                ("c1".into(), "SIMD vector operations".into()),
                ("c2".into(), "GPU kernel dispatch".into()),
            ],
            None,
        )
        .unwrap();

        assert_eq!(idx.document_count().unwrap(), 1);
        assert_eq!(idx.chunk_count().unwrap(), 2);

        let content = idx.get_chunk("c1").unwrap();
        assert_eq!(content.unwrap(), "SIMD vector operations");
    }

    #[test]
    fn test_index_search() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        idx.insert_document(
            "doc1",
            None,
            None,
            "",
            &[
                ("c1".into(), "machine learning algorithms for classification".into()),
                ("c2".into(), "database indexing and query optimization".into()),
            ],
            None,
        )
        .unwrap();

        let results = idx.search_fts("machine learning", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].chunk_id, "c1");
    }

    #[test]
    fn test_index_fingerprint_reindex() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        let hash1 = [1u8; 32];
        let hash2 = [2u8; 32];

        // First insert with fingerprint
        idx.insert_document(
            "doc1",
            None,
            None,
            "",
            &[("c1".into(), "content".into())],
            Some(("/test.md", &hash1)),
        )
        .unwrap();

        // Same hash should not need reindex
        assert!(!idx.needs_reindex("/test.md", &hash1).unwrap());

        // Different hash should need reindex
        assert!(idx.needs_reindex("/test.md", &hash2).unwrap());

        // Unknown path should need reindex
        assert!(idx.needs_reindex("/unknown.md", &hash1).unwrap());
    }

    #[test]
    fn test_index_remove_document() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        idx.insert_document("doc1", None, None, "", &[("c1".into(), "some content".into())], None)
            .unwrap();

        assert_eq!(idx.document_count().unwrap(), 1);
        idx.remove_document("doc1").unwrap();
        assert_eq!(idx.document_count().unwrap(), 0);
        assert_eq!(idx.chunk_count().unwrap(), 0);
    }

    #[test]
    fn test_index_metadata() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        idx.set_metadata("version", "1.0.0").unwrap();
        assert_eq!(idx.get_metadata("version").unwrap(), Some("1.0.0".to_string()));
        assert_eq!(idx.get_metadata("nonexistent").unwrap(), None);
    }

    #[test]
    fn test_index_update_document() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        idx.insert_document("doc1", None, None, "", &[("c1".into(), "old content".into())], None)
            .unwrap();
        idx.insert_document("doc1", None, None, "", &[("c2".into(), "new content".into())], None)
            .unwrap();

        // Old chunk should be gone, new chunk present
        assert_eq!(idx.chunk_count().unwrap(), 1);
        assert!(idx.get_chunk("c1").unwrap().is_none());
        assert_eq!(idx.get_chunk("c2").unwrap().unwrap(), "new content");
    }

    // --- SparseIndex trait tests ---

    #[test]
    fn test_sparse_index_add_and_len() {
        let mut idx = SqliteIndex::open_in_memory().unwrap();
        let doc_id = DocumentId::new();
        let chunk = make_chunk(doc_id, "sparse index test content");
        idx.add(&chunk);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_sparse_index_add_batch() {
        let mut idx = SqliteIndex::open_in_memory().unwrap();
        let doc_id = DocumentId::new();
        let chunks = vec![
            make_chunk(doc_id, "first chunk content"),
            make_chunk(doc_id, "second chunk content"),
        ];
        idx.add_batch(&chunks);
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn test_sparse_index_remove() {
        let mut idx = SqliteIndex::open_in_memory().unwrap();
        let doc_id = DocumentId::new();
        let chunk = make_chunk(doc_id, "content to remove");
        let chunk_id = chunk.id;
        idx.add(&chunk);
        assert_eq!(idx.len(), 1);
        idx.remove(chunk_id);
        assert_eq!(idx.len(), 0);
    }

    // --- SqliteStore tests ---

    #[test]
    fn test_store_index_and_search() {
        let store = SqliteStore::open_in_memory().unwrap();
        let doc = make_doc("SIMD vector operations for tensor computation");
        let chunks = vec![make_chunk(doc.id, "SIMD vector operations for tensor computation")];
        store.index_document(&doc, &chunks, None).unwrap();

        let results = store.search("SIMD tensor", 10).unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_store_stats() {
        let store = SqliteStore::open_in_memory().unwrap();
        let doc = make_doc("content");
        let chunks = vec![make_chunk(doc.id, "chunk one"), make_chunk(doc.id, "chunk two")];
        store.index_document(&doc, &chunks, Some(("/test.md", &[0u8; 32]))).unwrap();

        let stats = store.stats().unwrap();
        assert_eq!(stats.document_count, 1);
        assert_eq!(stats.chunk_count, 2);
        assert_eq!(stats.fingerprint_count, 1);
    }

    #[test]
    fn test_store_needs_reindex() {
        let store = SqliteStore::open_in_memory().unwrap();
        let doc = make_doc("content");
        let chunks = vec![make_chunk(doc.id, "chunk")];
        let hash = [42u8; 32];
        store.index_document(&doc, &chunks, Some(("/doc.md", &hash))).unwrap();

        assert!(!store.needs_reindex("/doc.md", &hash).unwrap());
        assert!(store.needs_reindex("/doc.md", &[0u8; 32]).unwrap());
        assert!(store.needs_reindex("/other.md", &hash).unwrap());
    }

    #[test]
    fn test_store_metadata() {
        let store = SqliteStore::open_in_memory().unwrap();
        store.set_metadata("batuta_version", "0.6.0").unwrap();
        assert_eq!(store.get_metadata("batuta_version").unwrap(), Some("0.6.0".to_string()));
    }

    #[test]
    fn test_store_optimize() {
        let store = SqliteStore::open_in_memory().unwrap();
        let doc = make_doc("content");
        let chunks = vec![make_chunk(doc.id, "some chunk content")];
        store.index_document(&doc, &chunks, None).unwrap();
        store.optimize().unwrap(); // Should not panic
    }

    #[test]
    fn test_store_large_batch() {
        let store = SqliteStore::open_in_memory().unwrap();

        // Insert 100 documents with 5 chunks each
        for i in 0..100 {
            let doc = make_doc(&format!("Document {i} about machine learning"));
            let chunks: Vec<Chunk> = (0..5)
                .map(|j| {
                    make_chunk(
                        doc.id,
                        &format!("Chunk {j} of doc {i}: machine learning algorithms topic {j}"),
                    )
                })
                .collect();
            store.index_document(&doc, &chunks, None).unwrap();
        }

        let stats = store.stats().unwrap();
        assert_eq!(stats.document_count, 100);
        assert_eq!(stats.chunk_count, 500);

        let results = store.search("machine learning", 10).unwrap();
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_search_deterministic() {
        let store = SqliteStore::open_in_memory().unwrap();
        let doc = make_doc("determinism test");
        let chunks = vec![
            make_chunk(doc.id, "alpha beta gamma delta"),
            make_chunk(doc.id, "epsilon zeta alpha alpha"),
        ];
        store.index_document(&doc, &chunks, None).unwrap();

        // Run the same query 10 times, results should be identical
        let baseline = store.search("alpha", 10).unwrap();
        for _ in 0..10 {
            let results = store.search("alpha", 10).unwrap();
            assert_eq!(results.len(), baseline.len());
            for (a, b) in baseline.iter().zip(results.iter()) {
                assert_eq!(a.chunk_id, b.chunk_id);
                assert!((a.score - b.score).abs() < f64::EPSILON);
            }
        }
    }

    // --- Incremental indexing tests ---

    #[test]
    fn test_list_fingerprints_empty() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        let fps = idx.list_fingerprints().unwrap();
        assert!(fps.is_empty());
    }

    #[test]
    fn test_list_fingerprints_populated() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        let hash1 = [1u8; 32];
        let hash2 = [2u8; 32];

        idx.insert_document(
            "doc1",
            None,
            Some("/a.md"),
            "",
            &[("c1".into(), "content a".into())],
            Some(("/a.md", &hash1)),
        )
        .unwrap();
        idx.insert_document(
            "doc2",
            None,
            Some("/b.md"),
            "",
            &[("c2".into(), "content b".into())],
            Some(("/b.md", &hash2)),
        )
        .unwrap();

        let fps = idx.list_fingerprints().unwrap();
        assert_eq!(fps.len(), 2);
        let paths: Vec<&str> = fps.iter().map(|(p, _)| p.as_str()).collect();
        assert!(paths.contains(&"/a.md"));
        assert!(paths.contains(&"/b.md"));
    }

    #[test]
    fn test_remove_by_source() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        let hash = [1u8; 32];

        idx.insert_document(
            "doc1",
            None,
            Some("/a.md"),
            "full content",
            &[("c1".into(), "chunk 1".into()), ("c2".into(), "chunk 2".into())],
            Some(("/a.md", &hash)),
        )
        .unwrap();
        idx.insert_document(
            "doc2",
            None,
            Some("/b.md"),
            "other content",
            &[("c3".into(), "chunk 3".into())],
            Some(("/b.md", &hash)),
        )
        .unwrap();

        assert_eq!(idx.document_count().unwrap(), 2);
        assert_eq!(idx.chunk_count().unwrap(), 3);

        let removed = idx.remove_by_source("/a.md").unwrap();
        assert_eq!(removed, 1);
        assert_eq!(idx.document_count().unwrap(), 1);
        assert_eq!(idx.chunk_count().unwrap(), 1);

        // Fingerprint should also be removed
        assert!(idx.needs_reindex("/a.md", &hash).unwrap());
        // Other doc unaffected
        assert!(!idx.needs_reindex("/b.md", &hash).unwrap());
    }

    #[test]
    fn test_remove_by_source_nonexistent() {
        let idx = SqliteIndex::open_in_memory().unwrap();
        let removed = idx.remove_by_source("/nonexistent.md").unwrap();
        assert_eq!(removed, 0);
    }
}
