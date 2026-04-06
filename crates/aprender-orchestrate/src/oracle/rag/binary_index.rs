//! Binary Index Format for RAG
//!
//! Memory-mapped binary format for faster cold start and reduced memory.
//!
//! # Format
//!
//! ```text
//! Header (64 bytes):
//!   magic:      [u8; 4]  "BRAG"
//!   version:    u32      2
//!   doc_count:  u64
//!   term_count: u64
//!   checksum:   [u8; 32] BLAKE3
//!   reserved:   [u8; 8]
//!
//! Term Index (sorted by term):
//!   term_len:     u16
//!   term:         [u8; term_len]
//!   posting_count: u32
//!   postings:     [(doc_id: u32, tf: u16); posting_count]
//!
//! Document Metadata:
//!   path_offset:    u64
//!   content_offset: u64
//!   length:         u32
//!   fingerprint:    [u8; 32]
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// Magic number for binary index
pub const MAGIC: [u8; 4] = *b"BRAG";

/// Current format version
pub const VERSION: u32 = 2;

/// Binary index header
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct IndexHeader {
    /// Magic number ("BRAG")
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// Number of documents
    pub doc_count: u64,
    /// Number of unique terms
    pub term_count: u64,
    /// BLAKE3 checksum of content
    pub checksum: [u8; 32],
    /// Reserved for future use
    pub reserved: [u8; 8],
}

impl IndexHeader {
    /// Create a new header
    pub fn new(doc_count: u64, term_count: u64, checksum: [u8; 32]) -> Self {
        Self { magic: MAGIC, version: VERSION, doc_count, term_count, checksum, reserved: [0; 8] }
    }

    /// Validate header
    pub fn validate(&self) -> Result<(), BinaryIndexError> {
        if self.magic != MAGIC {
            return Err(BinaryIndexError::InvalidMagic);
        }
        if self.version != VERSION {
            return Err(BinaryIndexError::VersionMismatch {
                expected: VERSION,
                found: self.version,
            });
        }
        Ok(())
    }

    /// Serialize to bytes
    #[allow(clippy::wrong_self_convention)]
    pub fn to_bytes(&self) -> [u8; 64] {
        let mut bytes = [0u8; 64];
        let (a, rest) = bytes.split_at_mut(4);
        a.copy_from_slice(&self.magic);
        let (b, rest) = rest.split_at_mut(4);
        b.copy_from_slice(&self.version.to_le_bytes());
        let (c, rest) = rest.split_at_mut(8);
        c.copy_from_slice(&self.doc_count.to_le_bytes());
        let (d, rest) = rest.split_at_mut(8);
        d.copy_from_slice(&self.term_count.to_le_bytes());
        let (e, f) = rest.split_at_mut(32);
        e.copy_from_slice(&self.checksum);
        f.copy_from_slice(&self.reserved);
        bytes
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8; 64]) -> Self {
        let (magic_s, rest) = bytes.split_at(4);
        let mut magic = [0u8; 4];
        magic.copy_from_slice(magic_s);

        let version = u32::from_le_bytes([rest[0], rest[1], rest[2], rest[3]]);
        let (_, rest) = rest.split_at(4);
        let doc_count = u64::from_le_bytes([
            rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7],
        ]);
        let (_, rest) = rest.split_at(8);
        let term_count = u64::from_le_bytes([
            rest[0], rest[1], rest[2], rest[3], rest[4], rest[5], rest[6], rest[7],
        ]);
        let (_, rest) = rest.split_at(8);

        let (checksum_s, reserved_s) = rest.split_at(32);
        let mut checksum = [0u8; 32];
        checksum.copy_from_slice(checksum_s);

        let mut reserved = [0u8; 8];
        reserved.copy_from_slice(reserved_s);

        Self { magic, version, doc_count, term_count, checksum, reserved }
    }
}

/// Document metadata entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentEntry {
    /// Path to document
    pub path: String,
    /// Content hash
    pub fingerprint: [u8; 32],
    /// Document length in chars
    pub length: u32,
}

/// Term posting
#[derive(Debug, Clone, Copy)]
pub struct Posting {
    /// Document ID
    pub doc_id: u32,
    /// Term frequency
    pub tf: u16,
}

/// Binary index writer
pub struct BinaryIndexWriter {
    /// Document entries
    documents: Vec<DocumentEntry>,
    /// Term to postings mapping
    terms: HashMap<String, Vec<Posting>>,
}

impl BinaryIndexWriter {
    /// Create a new writer
    pub fn new() -> Self {
        Self { documents: Vec::new(), terms: HashMap::new() }
    }

    /// Add a document
    pub fn add_document(&mut self, path: String, fingerprint: [u8; 32], length: u32) -> u32 {
        let doc_id = self.documents.len() as u32;
        self.documents.push(DocumentEntry { path, fingerprint, length });
        doc_id
    }

    /// Add a term posting
    pub fn add_posting(&mut self, term: &str, doc_id: u32, tf: u16) {
        self.terms.entry(term.to_string()).or_default().push(Posting { doc_id, tf });
    }

    /// Write to file
    pub fn write_to_file(&self, path: &Path) -> Result<(), BinaryIndexError> {
        let mut file = std::fs::File::create(path)?;

        // Compute checksum
        let checksum = self.compute_checksum();

        // Write header
        let header =
            IndexHeader::new(self.documents.len() as u64, self.terms.len() as u64, checksum);
        file.write_all(&header.to_bytes())?;

        // Write documents (JSON for simplicity, could be more compact)
        let docs_json = serde_json::to_vec(&self.documents)?;
        file.write_all(&(docs_json.len() as u64).to_le_bytes())?;
        file.write_all(&docs_json)?;

        // Write terms (sorted for binary search)
        let mut sorted_terms: Vec<_> = self.terms.iter().collect();
        sorted_terms.sort_by_key(|(k, _)| *k);

        file.write_all(&(sorted_terms.len() as u64).to_le_bytes())?;
        for (term, postings) in sorted_terms {
            let term_bytes = term.as_bytes();
            file.write_all(&(term_bytes.len() as u16).to_le_bytes())?;
            file.write_all(term_bytes)?;
            file.write_all(&(postings.len() as u32).to_le_bytes())?;
            for posting in postings {
                file.write_all(&posting.doc_id.to_le_bytes())?;
                file.write_all(&posting.tf.to_le_bytes())?;
            }
        }

        Ok(())
    }

    /// Compute checksum of content
    fn compute_checksum(&self) -> [u8; 32] {
        use super::fingerprint::blake3_hash;

        let mut data = Vec::new();
        for doc in &self.documents {
            data.extend_from_slice(doc.path.as_bytes());
            data.extend_from_slice(&doc.fingerprint);
        }
        for (term, postings) in &self.terms {
            data.extend_from_slice(term.as_bytes());
            for posting in postings {
                data.extend_from_slice(&posting.doc_id.to_le_bytes());
                data.extend_from_slice(&posting.tf.to_le_bytes());
            }
        }
        blake3_hash(&data)
    }
}

impl Default for BinaryIndexWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary index reader
pub struct BinaryIndexReader {
    /// Index header
    header: IndexHeader,
    /// Document entries
    documents: Vec<DocumentEntry>,
    /// Term index (sorted)
    terms: Vec<(String, Vec<Posting>)>,
}

impl BinaryIndexReader {
    /// Load from file
    pub fn load(path: &Path) -> Result<Self, BinaryIndexError> {
        let mut file = std::fs::File::open(path)?;

        // Read header
        let mut header_bytes = [0u8; 64];
        file.read_exact(&mut header_bytes)?;
        let header = IndexHeader::from_bytes(&header_bytes);
        header.validate()?;

        // Read documents
        let mut doc_len_bytes = [0u8; 8];
        file.read_exact(&mut doc_len_bytes)?;
        let doc_len = u64::from_le_bytes(doc_len_bytes) as usize;

        let mut docs_json = vec![0u8; doc_len];
        file.read_exact(&mut docs_json)?;
        let documents: Vec<DocumentEntry> = serde_json::from_slice(&docs_json)?;

        // Read terms
        let mut term_count_bytes = [0u8; 8];
        file.read_exact(&mut term_count_bytes)?;
        let term_count = u64::from_le_bytes(term_count_bytes) as usize;

        let mut terms = Vec::with_capacity(term_count);
        for _ in 0..term_count {
            let mut term_len_bytes = [0u8; 2];
            file.read_exact(&mut term_len_bytes)?;
            let term_len = u16::from_le_bytes(term_len_bytes) as usize;

            let mut term_bytes = vec![0u8; term_len];
            file.read_exact(&mut term_bytes)?;
            let term = String::from_utf8(term_bytes).map_err(|_| BinaryIndexError::InvalidUtf8)?;

            let mut posting_count_bytes = [0u8; 4];
            file.read_exact(&mut posting_count_bytes)?;
            let posting_count = u32::from_le_bytes(posting_count_bytes) as usize;

            let mut postings = Vec::with_capacity(posting_count);
            for _ in 0..posting_count {
                let mut doc_id_bytes = [0u8; 4];
                let mut tf_bytes = [0u8; 2];
                file.read_exact(&mut doc_id_bytes)?;
                file.read_exact(&mut tf_bytes)?;
                postings.push(Posting {
                    doc_id: u32::from_le_bytes(doc_id_bytes),
                    tf: u16::from_le_bytes(tf_bytes),
                });
            }

            terms.push((term, postings));
        }

        Ok(Self { header, documents, terms })
    }

    /// Get document by ID
    pub fn get_document(&self, doc_id: u32) -> Option<&DocumentEntry> {
        self.documents.get(doc_id as usize)
    }

    /// Binary search for term postings
    pub fn get_postings(&self, term: &str) -> Option<&[Posting]> {
        match self.terms.binary_search_by_key(&term, |(t, _)| t.as_str()) {
            Ok(idx) => Some(&self.terms[idx].1),
            Err(_) => None,
        }
    }

    /// Document count
    pub fn doc_count(&self) -> usize {
        self.documents.len()
    }

    /// Term count
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }
}

/// Binary index error
#[derive(Debug, thiserror::Error)]
pub enum BinaryIndexError {
    #[error("Invalid magic number")]
    InvalidMagic,

    #[error("Version mismatch: expected {expected}, found {found}")]
    VersionMismatch { expected: u32, found: u32 },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid UTF-8")]
    InvalidUtf8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_header_roundtrip() {
        let header = IndexHeader::new(100, 5000, [42u8; 32]);
        let bytes = header.to_bytes();
        let parsed = IndexHeader::from_bytes(&bytes);

        assert_eq!(parsed.magic, MAGIC);
        assert_eq!(parsed.version, VERSION);
        assert_eq!(parsed.doc_count, 100);
        assert_eq!(parsed.term_count, 5000);
    }

    #[test]
    fn test_write_and_read() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let index_path = temp.path().join("test.brag");

        // Write index
        let mut writer = BinaryIndexWriter::new();
        let doc_id = writer.add_document("test.txt".to_string(), [1u8; 32], 100);
        writer.add_posting("hello", doc_id, 5);
        writer.add_posting("world", doc_id, 3);
        writer.write_to_file(&index_path).expect("unexpected failure");

        // Read index
        let reader = BinaryIndexReader::load(&index_path).expect("unexpected failure");
        assert_eq!(reader.doc_count(), 1);
        assert_eq!(reader.term_count(), 2);

        let postings = reader.get_postings("hello").expect("unexpected failure");
        assert_eq!(postings.len(), 1);
        assert_eq!(postings[0].doc_id, 0);
        assert_eq!(postings[0].tf, 5);
    }

    #[test]
    fn test_header_validation() {
        let mut header = IndexHeader::new(100, 5000, [42u8; 32]);
        assert!(header.validate().is_ok());

        // Invalid magic
        header.magic = *b"XXXX";
        assert!(matches!(header.validate(), Err(BinaryIndexError::InvalidMagic)));
    }

    #[test]
    fn test_header_version_mismatch() {
        let mut header = IndexHeader::new(100, 5000, [42u8; 32]);
        header.version = 999;
        assert!(matches!(header.validate(), Err(BinaryIndexError::VersionMismatch { .. })));
    }

    #[test]
    fn test_document_lookup() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let index_path = temp.path().join("test.brag");

        let mut writer = BinaryIndexWriter::new();
        writer.add_document("doc1.txt".to_string(), [1u8; 32], 100);
        writer.add_document("doc2.txt".to_string(), [2u8; 32], 200);
        writer.write_to_file(&index_path).expect("unexpected failure");

        let reader = BinaryIndexReader::load(&index_path).expect("unexpected failure");

        let doc = reader.get_document(0).expect("unexpected failure");
        assert_eq!(doc.path, "doc1.txt");
        assert_eq!(doc.length, 100);

        let doc = reader.get_document(1).expect("unexpected failure");
        assert_eq!(doc.path, "doc2.txt");
        assert_eq!(doc.length, 200);

        assert!(reader.get_document(999).is_none());
    }

    #[test]
    fn test_missing_term_returns_none() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let index_path = temp.path().join("test.brag");

        let mut writer = BinaryIndexWriter::new();
        let doc_id = writer.add_document("test.txt".to_string(), [1u8; 32], 100);
        writer.add_posting("exists", doc_id, 1);
        writer.write_to_file(&index_path).expect("unexpected failure");

        let reader = BinaryIndexReader::load(&index_path).expect("unexpected failure");
        assert!(reader.get_postings("exists").is_some());
        assert!(reader.get_postings("nonexistent").is_none());
    }

    #[test]
    fn test_multiple_documents_same_term() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let index_path = temp.path().join("test.brag");

        let mut writer = BinaryIndexWriter::new();
        let doc1 = writer.add_document("doc1.txt".to_string(), [1u8; 32], 100);
        let doc2 = writer.add_document("doc2.txt".to_string(), [2u8; 32], 200);
        writer.add_posting("common", doc1, 3);
        writer.add_posting("common", doc2, 7);
        writer.write_to_file(&index_path).expect("unexpected failure");

        let reader = BinaryIndexReader::load(&index_path).expect("unexpected failure");
        let postings = reader.get_postings("common").expect("unexpected failure");

        assert_eq!(postings.len(), 2);
        assert_eq!(postings[0].tf, 3);
        assert_eq!(postings[1].tf, 7);
    }

    #[test]
    fn test_binary_index_writer_default() {
        let writer = BinaryIndexWriter::default();
        // Default writer should have no documents or terms
        let temp = TempDir::new().expect("tempdir creation failed");
        let index_path = temp.path().join("empty.brag");
        writer.write_to_file(&index_path).expect("unexpected failure");

        let reader = BinaryIndexReader::load(&index_path).expect("unexpected failure");
        assert_eq!(reader.doc_count(), 0);
        assert_eq!(reader.term_count(), 0);
    }

    #[test]
    fn test_document_entry_fields() {
        let entry = DocumentEntry {
            path: "/test/path.rs".to_string(),
            fingerprint: [42u8; 32],
            length: 1234,
        };
        assert_eq!(entry.path, "/test/path.rs");
        assert_eq!(entry.fingerprint, [42u8; 32]);
        assert_eq!(entry.length, 1234);
    }

    #[test]
    fn test_posting_fields() {
        let posting = Posting { doc_id: 5, tf: 10 };
        assert_eq!(posting.doc_id, 5);
        assert_eq!(posting.tf, 10);
    }

    #[test]
    fn test_magic_and_version_constants() {
        assert_eq!(MAGIC, *b"BRAG");
        assert_eq!(VERSION, 2);
    }

    #[test]
    fn test_header_reserved_is_zeroed() {
        let header = IndexHeader::new(10, 20, [1u8; 32]);
        assert_eq!(header.reserved, [0u8; 8]);
    }

    #[test]
    fn test_error_display() {
        let err = BinaryIndexError::InvalidMagic;
        assert!(format!("{}", err).contains("Invalid magic"));

        let err = BinaryIndexError::VersionMismatch { expected: 2, found: 1 };
        assert!(format!("{}", err).contains('2'));
        assert!(format!("{}", err).contains('1'));

        let err = BinaryIndexError::InvalidUtf8;
        assert!(format!("{}", err).contains("UTF-8"));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = BinaryIndexReader::load(Path::new("/nonexistent/path.brag"));
        assert!(result.is_err());
    }

    #[test]
    fn test_add_document_returns_sequential_ids() {
        let mut writer = BinaryIndexWriter::new();
        let id0 = writer.add_document("doc0.txt".to_string(), [0u8; 32], 100);
        let id1 = writer.add_document("doc1.txt".to_string(), [1u8; 32], 200);
        let id2 = writer.add_document("doc2.txt".to_string(), [2u8; 32], 300);

        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn test_multiple_postings_same_term() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let index_path = temp.path().join("multi.brag");

        let mut writer = BinaryIndexWriter::new();
        let doc = writer.add_document("doc.txt".to_string(), [1u8; 32], 100);
        writer.add_posting("term", doc, 1);
        writer.add_posting("term", doc, 2);
        writer.add_posting("term", doc, 3);
        writer.write_to_file(&index_path).expect("unexpected failure");

        let reader = BinaryIndexReader::load(&index_path).expect("unexpected failure");
        let postings = reader.get_postings("term").expect("unexpected failure");
        assert_eq!(postings.len(), 3);
    }

    #[test]
    fn test_terms_sorted_alphabetically() {
        let temp = TempDir::new().expect("tempdir creation failed");
        let index_path = temp.path().join("sorted.brag");

        let mut writer = BinaryIndexWriter::new();
        let doc = writer.add_document("doc.txt".to_string(), [1u8; 32], 100);
        writer.add_posting("zebra", doc, 1);
        writer.add_posting("alpha", doc, 1);
        writer.add_posting("middle", doc, 1);
        writer.write_to_file(&index_path).expect("unexpected failure");

        let reader = BinaryIndexReader::load(&index_path).expect("unexpected failure");
        // Binary search should work because terms are sorted
        assert!(reader.get_postings("alpha").is_some());
        assert!(reader.get_postings("middle").is_some());
        assert!(reader.get_postings("zebra").is_some());
    }
}
