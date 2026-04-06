//! Compressed Index Serialization (GH-2)
//!
//! Provides LZ4/ZSTD compression for BM25 and vector index storage.
//! Reduces storage footprint by 5-10x for typical RAG indices.
//!
//! Compression algorithm is shared via `batuta_common::compression`.

use crate::{BM25Index, Result};
use serde::{de::DeserializeOwned, Serialize};

// Note: VectorStore compression can be added in the future
// by implementing Serialize/Deserialize for VectorStore

pub use batuta_common::compression::Compression;

/// Serialize an index to compressed bytes
///
/// # Errors
/// Returns error if serialization or compression fails
pub fn serialize_compressed<T: Serialize>(index: &T, compression: Compression) -> Result<Vec<u8>> {
    let bytes = bincode::serialize(index).map_err(|e| {
        crate::Error::SerializationError(format!("Bincode serialization failed: {e}"))
    })?;
    Ok(compression.compress(&bytes)?)
}

/// Deserialize an index from compressed bytes
///
/// # Errors
/// Returns error if decompression or deserialization fails
pub fn deserialize_compressed<T: DeserializeOwned>(
    data: &[u8],
    compression: Compression,
) -> Result<T> {
    let decompressed = compression.decompress(data)?;
    bincode::deserialize(&decompressed).map_err(|e| {
        crate::Error::SerializationError(format!("Bincode deserialization failed: {e}"))
    })
}

impl BM25Index {
    /// Serialize to compressed bytes using specified compression
    ///
    /// # Errors
    /// Returns error if serialization or compression fails
    pub fn to_compressed_bytes(&self, compression: Compression) -> Result<Vec<u8>> {
        serialize_compressed(self, compression)
    }

    /// Deserialize from compressed bytes
    ///
    /// # Errors
    /// Returns error if decompression or deserialization fails
    pub fn from_compressed_bytes(data: &[u8], compression: Compression) -> Result<Self> {
        deserialize_compressed(data, compression)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{index::SparseIndex, Chunk, DocumentId};

    fn create_test_chunk(content: &str) -> Chunk {
        Chunk::new(DocumentId::new(), content.to_string(), 0, content.len())
    }

    // ============================================================
    // RED PHASE: These tests define the expected behavior
    // ============================================================

    #[test]
    fn test_compression_as_str() {
        assert_eq!(Compression::Lz4.as_str(), "lz4");
        assert_eq!(Compression::Zstd.as_str(), "zstd");
    }

    #[test]
    fn test_compression_default() {
        assert_eq!(Compression::default(), Compression::Lz4);
    }

    #[test]
    fn test_lz4_compress_decompress() {
        let data = b"hello world hello world hello world".to_vec();
        let compressed = Compression::Lz4.compress(&data).unwrap();
        let decompressed = Compression::Lz4.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_zstd_compress_decompress() {
        let data = b"hello world hello world hello world".to_vec();
        let compressed = Compression::Zstd.compress(&data).unwrap();
        let decompressed = Compression::Zstd.decompress(&compressed).unwrap();
        assert_eq!(decompressed, data);
    }

    #[test]
    fn test_empty_data_compression() {
        let empty: Vec<u8> = vec![];

        let lz4_compressed = Compression::Lz4.compress(&empty).unwrap();
        assert!(lz4_compressed.is_empty());
        let lz4_decompressed = Compression::Lz4.decompress(&lz4_compressed).unwrap();
        assert!(lz4_decompressed.is_empty());

        let zstd_compressed = Compression::Zstd.compress(&empty).unwrap();
        assert!(zstd_compressed.is_empty());
        let zstd_decompressed = Compression::Zstd.decompress(&zstd_compressed).unwrap();
        assert!(zstd_decompressed.is_empty());
    }

    #[test]
    fn test_lz4_compresses_repeated_data() {
        let data = vec![0u8; 10000];
        let compressed = Compression::Lz4.compress(&data).unwrap();
        // LZ4 should achieve >10x compression on zeros
        assert!(compressed.len() < data.len() / 10);
    }

    #[test]
    fn test_zstd_compresses_repeated_data() {
        let data = vec![0u8; 10000];
        let compressed = Compression::Zstd.compress(&data).unwrap();
        // ZSTD should achieve >10x compression on zeros
        assert!(compressed.len() < data.len() / 10);
    }

    // ============================================================
    // BM25Index Compression Tests
    // ============================================================

    #[test]
    fn test_bm25_lz4_roundtrip() {
        let mut index = BM25Index::new();
        index.add(&create_test_chunk("machine learning is great"));
        index.add(&create_test_chunk("deep learning neural networks"));
        index.add(&create_test_chunk("natural language processing"));

        let compressed = index.to_compressed_bytes(Compression::Lz4).unwrap();
        let restored = BM25Index::from_compressed_bytes(&compressed, Compression::Lz4).unwrap();

        // Verify restored index works correctly
        assert_eq!(index.len(), restored.len());
        let original_results = index.search("machine learning", 10);
        let restored_results = restored.search("machine learning", 10);
        assert_eq!(original_results.len(), restored_results.len());
    }

    #[test]
    fn test_bm25_zstd_roundtrip() {
        let mut index = BM25Index::new();
        index.add(&create_test_chunk("rust programming language"));
        index.add(&create_test_chunk("systems programming with rust"));

        let compressed = index.to_compressed_bytes(Compression::Zstd).unwrap();
        let restored = BM25Index::from_compressed_bytes(&compressed, Compression::Zstd).unwrap();

        assert_eq!(index.len(), restored.len());
    }

    #[test]
    fn test_bm25_compression_reduces_size() {
        let mut index = BM25Index::new();
        // Add many documents to make index larger
        for i in 0..100 {
            index.add(&create_test_chunk(&format!(
                "document number {i} about machine learning and artificial intelligence"
            )));
        }

        let uncompressed = bincode::serialize(&index).unwrap();
        let lz4_compressed = index.to_compressed_bytes(Compression::Lz4).unwrap();
        let zstd_compressed = index.to_compressed_bytes(Compression::Zstd).unwrap();

        // Both should achieve some compression
        assert!(lz4_compressed.len() < uncompressed.len());
        assert!(zstd_compressed.len() < uncompressed.len());

        // ZSTD typically achieves better compression than LZ4
        assert!(zstd_compressed.len() <= lz4_compressed.len());
    }

    #[test]
    fn test_bm25_empty_index_compression() {
        let index = BM25Index::new();

        let compressed = index.to_compressed_bytes(Compression::Lz4).unwrap();
        let restored = BM25Index::from_compressed_bytes(&compressed, Compression::Lz4).unwrap();

        assert!(restored.is_empty());
    }

    #[test]
    fn test_bm25_preserved_search_behavior() {
        let mut index = BM25Index::new();
        index.add(&create_test_chunk("python programming language scripting"));
        index.add(&create_test_chunk("javascript web development frontend"));
        index.add(&create_test_chunk("rust systems programming performance"));

        // Serialize and restore
        let compressed = index.to_compressed_bytes(Compression::Lz4).unwrap();
        let restored = BM25Index::from_compressed_bytes(&compressed, Compression::Lz4).unwrap();

        // Search should return same results
        let query = "programming language";
        let original_results = index.search(query, 3);
        let restored_results = restored.search(query, 3);

        assert_eq!(original_results.len(), restored_results.len());
        // Scores should match
        for ((orig_id, orig_score), (rest_id, rest_score)) in
            original_results.iter().zip(restored_results.iter())
        {
            assert_eq!(orig_id, rest_id);
            assert!((orig_score - rest_score).abs() < 1e-5);
        }
    }
}
