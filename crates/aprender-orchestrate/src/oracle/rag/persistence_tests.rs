use super::*;
use tempfile::TempDir;

fn test_persistence() -> (RagPersistence, TempDir) {
    let tmp = TempDir::new().unwrap();
    let persistence = RagPersistence::with_path(tmp.path().to_path_buf());
    (persistence, tmp)
}

fn sample_index() -> PersistedIndex {
    let mut inverted_index = HashMap::new();
    let mut postings = HashMap::new();
    postings.insert("doc1".to_string(), 3);
    postings.insert("doc2".to_string(), 1);
    inverted_index.insert("test".to_string(), postings);

    let mut doc_lengths = HashMap::new();
    doc_lengths.insert("doc1".to_string(), 10);
    doc_lengths.insert("doc2".to_string(), 5);

    PersistedIndex {
        inverted_index,
        doc_lengths,
        bm25_config: Bm25Config::default(),
        rrf_config: RrfConfig::default(),
        avg_doc_length: 7.5,
    }
}

fn sample_docs() -> PersistedDocuments {
    let mut chunk_contents = HashMap::new();
    chunk_contents.insert("doc1#1".to_string(), "SIMD GPU tensor operations".to_string());
    chunk_contents.insert("doc2#1".to_string(), "machine learning algorithms".to_string());

    PersistedDocuments {
        documents: HashMap::new(),
        fingerprints: HashMap::new(),
        total_chunks: 5,
        chunk_contents,
    }
}

fn sample_sources() -> Vec<CorpusSource> {
    vec![CorpusSource {
        id: "test-corpus".to_string(),
        commit: Some("abc123".to_string()),
        doc_count: 2,
        chunk_count: 5,
    }]
}

// RAG-PERSIST-001: Save/load roundtrip
#[test]
fn test_save_load_roundtrip() {
    let (persistence, _tmp) = test_persistence();

    let index = sample_index();
    let docs = sample_docs();
    let sources = sample_sources();

    // Save
    persistence.save(&index, &docs, sources.clone()).unwrap();

    // Load
    let result = persistence.load().unwrap();
    assert!(result.is_some());

    let (loaded_index, loaded_docs, manifest) = result.unwrap();

    // Verify index data
    assert_eq!(loaded_index.avg_doc_length, index.avg_doc_length);
    assert_eq!(loaded_index.doc_lengths.len(), index.doc_lengths.len());
    assert_eq!(loaded_index.inverted_index.len(), index.inverted_index.len());

    // Verify docs data
    assert_eq!(loaded_docs.total_chunks, docs.total_chunks);

    // Verify manifest
    assert_eq!(manifest.version, INDEX_VERSION);
    assert_eq!(manifest.sources.len(), 1);
    assert_eq!(manifest.sources[0].id, "test-corpus");
}

// RAG-PERSIST-001b: chunk_contents roundtrip
#[test]
fn test_chunk_contents_roundtrip() {
    let (persistence, _tmp) = test_persistence();

    let index = sample_index();
    let docs = sample_docs();
    let sources = sample_sources();

    persistence.save(&index, &docs, sources).unwrap();

    let (_, loaded_docs, _) = persistence.load().unwrap().unwrap();

    assert_eq!(loaded_docs.chunk_contents.len(), 2);
    assert_eq!(loaded_docs.chunk_contents.get("doc1#1").unwrap(), "SIMD GPU tensor operations");
    assert_eq!(loaded_docs.chunk_contents.get("doc2#1").unwrap(), "machine learning algorithms");
}

// RAG-PERSIST-002: Checksum corruption returns Ok(None) for graceful rebuild
#[test]
fn test_checksum_corruption_returns_none() {
    let (persistence, tmp) = test_persistence();

    let index = sample_index();
    let docs = sample_docs();
    let sources = sample_sources();

    // Save valid index
    persistence.save(&index, &docs, sources).unwrap();

    // Corrupt the index file
    let index_path = tmp.path().join(INDEX_FILE);
    fs::write(&index_path, r#"{"corrupted": true}"#).unwrap();

    // Load should return None (graceful degradation)
    let result = persistence.load().unwrap();
    assert!(result.is_none());
}

// RAG-PERSIST-003: Version compatibility check
#[test]
fn test_version_compatibility() {
    let (persistence, tmp) = test_persistence();

    let index = sample_index();
    let docs = sample_docs();
    let sources = sample_sources();

    // Save valid index
    persistence.save(&index, &docs, sources).unwrap();

    // Modify manifest to have incompatible version
    let manifest_path = tmp.path().join(MANIFEST_FILE);
    let manifest_json = fs::read_to_string(&manifest_path).unwrap();
    let modified = manifest_json.replace("\"1.1.0\"", "\"2.0.0\"");
    fs::write(&manifest_path, modified).unwrap();

    // Load should fail with version mismatch
    let result = persistence.load();
    assert!(result.is_err());

    let err = result.unwrap_err();
    assert!(matches!(err, PersistenceError::VersionMismatch { .. }));
}

// RAG-PERSIST-004: Empty cache returns None
#[test]
fn test_empty_cache_returns_none() {
    let (persistence, _tmp) = test_persistence();

    let result = persistence.load().unwrap();
    assert!(result.is_none());
}

// RAG-PERSIST-005: Invalid manifest JSON returns Ok(None) for graceful rebuild
#[test]
fn test_invalid_json_returns_none() {
    let (persistence, tmp) = test_persistence();

    // Create invalid manifest
    fs::create_dir_all(tmp.path()).unwrap();
    fs::write(tmp.path().join(MANIFEST_FILE), "not valid json").unwrap();

    // Load should return None (graceful degradation)
    let result = persistence.load().unwrap();
    assert!(result.is_none());
}

// RAG-PERSIST-006: Atomic write prevents partial files
#[test]
fn test_atomic_write() {
    let (persistence, tmp) = test_persistence();

    let index = sample_index();
    let docs = sample_docs();
    let sources = sample_sources();

    // Save should not leave .tmp files
    persistence.save(&index, &docs, sources).unwrap();

    // Check no .tmp files exist
    let entries: Vec<_> = fs::read_dir(tmp.path()).unwrap().collect();
    for entry in entries {
        let path = entry.unwrap().path();
        let filename = path.file_name().unwrap().to_str().unwrap();
        assert!(!filename.ends_with(".tmp"), "Found temp file: {}", filename);
    }
}

// RAG-PERSIST-007: Clear removes all cache files
#[test]
fn test_clear_removes_files() {
    let (persistence, tmp) = test_persistence();

    let index = sample_index();
    let docs = sample_docs();
    let sources = sample_sources();

    // Save index
    persistence.save(&index, &docs, sources).unwrap();

    // Verify files exist
    assert!(tmp.path().join(MANIFEST_FILE).exists());
    assert!(tmp.path().join(INDEX_FILE).exists());
    assert!(tmp.path().join(DOCUMENTS_FILE).exists());

    // Clear
    persistence.clear().unwrap();

    // Verify files removed
    assert!(!tmp.path().join(MANIFEST_FILE).exists());
    assert!(!tmp.path().join(INDEX_FILE).exists());
    assert!(!tmp.path().join(DOCUMENTS_FILE).exists());
}

// RAG-PERSIST-008: Stats returns manifest without full load
#[test]
fn test_stats_returns_manifest() {
    let (persistence, _tmp) = test_persistence();

    let index = sample_index();
    let docs = sample_docs();
    let sources = sample_sources();

    persistence.save(&index, &docs, sources).unwrap();

    let stats = persistence.stats().unwrap();
    assert!(stats.is_some());

    let manifest = stats.unwrap();
    assert_eq!(manifest.version, INDEX_VERSION);
    assert_eq!(manifest.sources.len(), 1);
    assert!(manifest.indexed_at > 0);
}

// RAG-PERSIST-009: Default path uses dirs crate
#[test]
fn test_default_path() {
    let persistence = RagPersistence::new();
    let path = persistence.cache_path();

    // Should end with batuta/rag
    let path_str = path.to_string_lossy();
    assert!(
        path_str.contains("batuta") && path_str.contains("rag"),
        "Path should contain batuta/rag: {}",
        path_str
    );
}

// RAG-PERSIST-010: Minor version differences are compatible
#[test]
fn test_minor_version_compatible() {
    let (persistence, tmp) = test_persistence();

    let index = sample_index();
    let docs = sample_docs();
    let sources = sample_sources();

    // Save valid index
    persistence.save(&index, &docs, sources).unwrap();

    // Modify manifest to have different minor version
    let manifest_path = tmp.path().join(MANIFEST_FILE);
    let manifest_json = fs::read_to_string(&manifest_path).unwrap();
    // Change 1.0.0 to 1.99.0 (same major, different minor)
    let modified = manifest_json.replace("\"1.1.0\"", "\"1.99.0\"");
    fs::write(&manifest_path, modified).unwrap();

    // Also need to update checksum in manifest - or we skip version check
    // For this test, we directly modify the validate_version behavior
    // by checking that only major version matters

    // Actually, let's verify the version parsing logic directly
    let manifest = RagManifest {
        version: "1.99.0".to_string(),
        index_checksum: [0; 32],
        docs_checksum: [0; 32],
        sources: vec![],
        indexed_at: 0,
        batuta_version: "test".to_string(),
    };

    // Should not error - minor version difference is OK
    let result = persistence.validate_version(&manifest);
    assert!(result.is_ok());
}

// RAG-PERSIST-011: Two-phase save leaves no .tmp orphans
#[test]
fn test_two_phase_save_no_tmp_orphans() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // No .tmp files should remain after successful save
    let entries: Vec<_> = fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().file_name().unwrap().to_str().unwrap().ends_with(".tmp"))
        .collect();
    assert!(entries.is_empty(), "Found orphaned .tmp files: {entries:?}");
}

// RAG-PERSIST-012: Save overwrites previous cache (proves clear() unnecessary)
#[test]
fn test_save_overwrites_previous_cache() {
    let (persistence, _tmp) = test_persistence();

    // Save initial data
    let mut index = sample_index();
    index.avg_doc_length = 1.0;
    persistence.save(&index, &sample_docs(), sample_sources()).unwrap();

    // Save different data (no clear() call)
    let mut index2 = sample_index();
    index2.avg_doc_length = 99.0;
    persistence.save(&index2, &sample_docs(), sample_sources()).unwrap();

    // Load should return the second save's data
    let (loaded, _, _) = persistence.load().unwrap().unwrap();
    assert!(
        (loaded.avg_doc_length - 99.0).abs() < f64::EPSILON,
        "Expected 99.0, got {}",
        loaded.avg_doc_length
    );
}

// RAG-PERSIST-013: Checksum mismatch returns Ok(None) (Jidoka graceful degradation)
#[test]
fn test_checksum_mismatch_graceful() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Corrupt documents file (manifest checksums won't match)
    fs::write(
        tmp.path().join(DOCUMENTS_FILE),
        r#"{"documents":{},"fingerprints":{},"total_chunks":0}"#,
    )
    .unwrap();

    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Expected None on checksum mismatch");
}

// RAG-PERSIST-014: Missing data file returns Ok(None)
#[test]
fn test_missing_data_file_returns_none() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Delete the index file (simulates partial crash)
    fs::remove_file(tmp.path().join(INDEX_FILE)).unwrap();

    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Expected None on missing data file");
}

// RAG-PERSIST-015: Orphan .tmp files cleaned on save
#[test]
fn test_orphan_tmp_cleaned_on_save() {
    let (persistence, tmp) = test_persistence();

    // Simulate crashed save by creating orphaned .tmp files
    fs::create_dir_all(tmp.path()).unwrap();
    fs::write(tmp.path().join("index.json.tmp"), "orphan").unwrap();
    fs::write(tmp.path().join("documents.json.tmp"), "orphan").unwrap();
    fs::write(tmp.path().join("manifest.json.tmp"), "orphan").unwrap();

    // Save should clean up orphans first
    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // No .tmp files should remain
    let tmp_files: Vec<_> = fs::read_dir(tmp.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().file_name().unwrap().to_str().unwrap().ends_with(".tmp"))
        .collect();
    assert!(tmp_files.is_empty(), "Orphan .tmp files not cleaned up");

    // And the save should be valid
    let result = persistence.load().unwrap();
    assert!(result.is_some(), "Save should produce valid cache");
}

// RAG-PERSIST-016: Manifest checksums consistent after save
#[test]
fn test_manifest_checksums_consistent() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Read manifest and data files, verify checksums match
    let manifest_json = fs::read_to_string(tmp.path().join(MANIFEST_FILE)).unwrap();
    let manifest: RagManifest = serde_json::from_str(&manifest_json).unwrap();

    let index_json = fs::read_to_string(tmp.path().join(INDEX_FILE)).unwrap();
    let docs_json = fs::read_to_string(tmp.path().join(DOCUMENTS_FILE)).unwrap();

    assert_eq!(
        blake3_hash(index_json.as_bytes()),
        manifest.index_checksum,
        "Index checksum mismatch"
    );
    assert_eq!(
        blake3_hash(docs_json.as_bytes()),
        manifest.docs_checksum,
        "Documents checksum mismatch"
    );
}

// ============================================================================
// Coverage gap tests: corrupt cache handling, checksum validation,
// fingerprints, version mismatch, missing data files
// ============================================================================

// RAG-PERSIST-017: Corrupt index JSON returns Ok(None)
#[test]
fn test_corrupt_index_json_returns_none() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Write syntactically valid JSON but wrong schema to index file
    // This will have correct JSON parse but wrong checksum
    let index_path = tmp.path().join(INDEX_FILE);
    fs::write(&index_path, r#"{"not_a_valid_index": true}"#).unwrap();

    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Corrupt index JSON should return None");
}

// RAG-PERSIST-018: Corrupt documents JSON returns Ok(None)
#[test]
fn test_corrupt_documents_json_returns_none() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Corrupt the documents file with invalid JSON
    let docs_path = tmp.path().join(DOCUMENTS_FILE);
    fs::write(&docs_path, "this is not json at all {{{").unwrap();

    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Corrupt documents JSON should return None");
}

// RAG-PERSIST-019: Missing documents file returns Ok(None)
#[test]
fn test_missing_documents_file_returns_none() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Delete the documents file
    fs::remove_file(tmp.path().join(DOCUMENTS_FILE)).unwrap();

    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Missing documents file should return None");
}

// RAG-PERSIST-020: Documents checksum mismatch returns Ok(None)
#[test]
fn test_documents_checksum_mismatch_returns_none() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Modify documents.json to have valid JSON but wrong checksum
    let docs_path = tmp.path().join(DOCUMENTS_FILE);
    let original = fs::read_to_string(&docs_path).unwrap();
    // Append a space to change the checksum
    fs::write(&docs_path, format!("{} ", original)).unwrap();

    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Documents checksum mismatch should return None");
}

// RAG-PERSIST-021: Index checksum mismatch returns Ok(None)
#[test]
fn test_index_checksum_mismatch_returns_none() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Modify index.json content to change checksum
    let index_path = tmp.path().join(INDEX_FILE);
    let original = fs::read_to_string(&index_path).unwrap();
    fs::write(&index_path, format!("{}  ", original)).unwrap();

    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Index checksum mismatch should return None");
}

// RAG-PERSIST-022: Version mismatch error display
#[test]
fn test_version_mismatch_error_display() {
    let err = PersistenceError::VersionMismatch {
        index_version: "2.0.0".to_string(),
        expected_version: INDEX_VERSION.to_string(),
    };
    let display = format!("{}", err);
    assert!(display.contains("2.0.0"));
    assert!(display.contains(INDEX_VERSION));
}

// RAG-PERSIST-023: Checksum mismatch error display
#[test]
fn test_checksum_mismatch_error_display() {
    let err = PersistenceError::ChecksumMismatch {
        file: "index.json".to_string(),
        expected: [0u8; 32],
        actual: [1u8; 32],
    };
    let display = format!("{}", err);
    assert!(display.contains("index.json"));
    assert!(display.contains("Checksum mismatch"));
}

// RAG-PERSIST-024: NoCachedIndex error display
#[test]
fn test_no_cached_index_error_display() {
    let err = PersistenceError::NoCachedIndex;
    let display = format!("{}", err);
    assert!(display.contains("No cached index"));
}

// RAG-PERSIST-025: CacheDirNotFound error display
#[test]
fn test_cache_dir_not_found_error_display() {
    let err = PersistenceError::CacheDirNotFound;
    let display = format!("{}", err);
    assert!(display.contains("Cache directory not found"));
}

// RAG-PERSIST-026: IO error wrapping
#[test]
fn test_io_error_wrapping() {
    let io_err = io::Error::new(io::ErrorKind::PermissionDenied, "test error");
    let err = PersistenceError::from(io_err);
    let display = format!("{}", err);
    assert!(display.contains("I/O error"));
}

// RAG-PERSIST-027: JSON error wrapping
#[test]
fn test_json_error_wrapping() {
    let json_result: Result<PersistedIndex, _> = serde_json::from_str("not json");
    let json_err = json_result.unwrap_err();
    let err = PersistenceError::from(json_err);
    let display = format!("{}", err);
    assert!(display.contains("JSON error"));
}

// RAG-PERSIST-028: Stats returns None for empty cache
#[test]
fn test_stats_empty_cache_returns_none() {
    let (persistence, _tmp) = test_persistence();
    let stats = persistence.stats().unwrap();
    assert!(stats.is_none());
}

// RAG-PERSIST-029: Clear on empty cache is no-op
#[test]
fn test_clear_empty_cache() {
    let (persistence, _tmp) = test_persistence();
    let result = persistence.clear();
    assert!(result.is_ok());
}

// RAG-PERSIST-030: Validate checksum directly
#[test]
fn test_validate_checksum_success() {
    let (persistence, _tmp) = test_persistence();
    let data = "test data";
    let hash = blake3_hash(data.as_bytes());
    let result = persistence.validate_checksum(data, hash, "test.json");
    assert!(result.is_ok());
}

// RAG-PERSIST-031: Validate checksum failure
#[test]
fn test_validate_checksum_failure() {
    let (persistence, _tmp) = test_persistence();
    let data = "test data";
    let wrong_hash = [0u8; 32];
    let result = persistence.validate_checksum(data, wrong_hash, "test.json");
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), PersistenceError::ChecksumMismatch { .. }));
}

// RAG-PERSIST-032: Validate version - major mismatch
#[test]
fn test_validate_version_major_mismatch() {
    let (persistence, _tmp) = test_persistence();
    let manifest = RagManifest {
        version: "99.0.0".to_string(),
        index_checksum: [0; 32],
        docs_checksum: [0; 32],
        sources: vec![],
        indexed_at: 0,
        batuta_version: "test".to_string(),
    };
    let result = persistence.validate_version(&manifest);
    assert!(matches!(result.unwrap_err(), PersistenceError::VersionMismatch { .. }));
}

// RAG-PERSIST-033: Validate version - same major, different patch
#[test]
fn test_validate_version_same_major_different_patch() {
    let (persistence, _tmp) = test_persistence();
    let manifest = RagManifest {
        version: "1.1.99".to_string(),
        index_checksum: [0; 32],
        docs_checksum: [0; 32],
        sources: vec![],
        indexed_at: 0,
        batuta_version: "test".to_string(),
    };
    let result = persistence.validate_version(&manifest);
    assert!(result.is_ok(), "Patch version difference should be OK");
}

// RAG-PERSIST-034: Load fingerprints only
#[test]
fn test_load_fingerprints_only() {
    let (persistence, _tmp) = test_persistence();

    let mut docs = sample_docs();
    let fp = DocumentFingerprint {
        content_hash: blake3_hash(b"test content"),
        chunker_config_hash: [0; 32],
        embedding_model_hash: [0; 32],
        indexed_at: 12345,
    };
    docs.fingerprints.insert("doc1".to_string(), fp.clone());

    persistence.save(&sample_index(), &docs, sample_sources()).unwrap();

    let fingerprints = persistence.load_fingerprints_only().unwrap();
    assert!(fingerprints.is_some());
    let fps = fingerprints.unwrap();
    assert!(fps.contains_key("doc1"));
    assert_eq!(fps.get("doc1").unwrap().content_hash, fp.content_hash);
}

// RAG-PERSIST-035: Load fingerprints only - no cache returns None
#[test]
fn test_load_fingerprints_only_no_cache() {
    let (persistence, _tmp) = test_persistence();
    let result = persistence.load_fingerprints_only().unwrap();
    assert!(result.is_none());
}

// RAG-PERSIST-036: Load fingerprints fallback when fingerprints.json missing
#[test]
fn test_load_fingerprints_fallback_no_fingerprints_file() {
    let (persistence, tmp) = test_persistence();

    let mut docs = sample_docs();
    let fp = DocumentFingerprint {
        content_hash: blake3_hash(b"test"),
        chunker_config_hash: [0; 32],
        embedding_model_hash: [0; 32],
        indexed_at: 12345,
    };
    docs.fingerprints.insert("doc1".to_string(), fp);

    persistence.save(&sample_index(), &docs, sample_sources()).unwrap();

    // Delete fingerprints.json to force fallback path
    let _ = fs::remove_file(tmp.path().join(FINGERPRINTS_FILE));

    let result = persistence.load_fingerprints_only().unwrap();
    assert!(result.is_some());
    let fps = result.unwrap();
    assert!(fps.contains_key("doc1"));
}

// RAG-PERSIST-037: Load fingerprints fallback when fingerprints.json is corrupt
#[test]
fn test_load_fingerprints_fallback_corrupt_file() {
    let (persistence, tmp) = test_persistence();

    let mut docs = sample_docs();
    let fp = DocumentFingerprint {
        content_hash: blake3_hash(b"test"),
        chunker_config_hash: [0; 32],
        embedding_model_hash: [0; 32],
        indexed_at: 12345,
    };
    docs.fingerprints.insert("doc1".to_string(), fp);

    persistence.save(&sample_index(), &docs, sample_sources()).unwrap();

    // Corrupt fingerprints.json
    fs::write(tmp.path().join(FINGERPRINTS_FILE), "not json").unwrap();

    // Should fall back to loading from documents.json
    let result = persistence.load_fingerprints_only().unwrap();
    assert!(result.is_some());
}

// RAG-PERSIST-038: Save fingerprints only
#[test]
fn test_save_fingerprints_only() {
    let (persistence, tmp) = test_persistence();

    let mut fingerprints = HashMap::new();
    let fp = DocumentFingerprint {
        content_hash: blake3_hash(b"content"),
        chunker_config_hash: [0; 32],
        embedding_model_hash: [0; 32],
        indexed_at: 99999,
    };
    fingerprints.insert("myfile.rs".to_string(), fp);

    persistence.save_fingerprints_only(&fingerprints).unwrap();

    // Verify fingerprints.json was written
    assert!(tmp.path().join(FINGERPRINTS_FILE).exists());

    // Load and verify
    let loaded = persistence.load_fingerprints_only().unwrap();
    // Since there's no manifest, load_fingerprints_only will try fingerprints.json first
    // which exists and was written correctly
    assert!(loaded.is_some());
    let fps = loaded.unwrap();
    assert_eq!(fps.len(), 1);
    assert!(fps.contains_key("myfile.rs"));
}

// RAG-PERSIST-039: Default impl
#[test]
fn test_default_impl() {
    let persistence = RagPersistence::default();
    let path = persistence.cache_path();
    assert!(path.to_string_lossy().contains("batuta"), "Default path should contain 'batuta'");
}

// RAG-PERSIST-040: Manifest batuta_version is set
#[test]
fn test_manifest_batuta_version() {
    let (persistence, _tmp) = test_persistence();
    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    let stats = persistence.stats().unwrap().unwrap();
    assert_eq!(stats.batuta_version, env!("CARGO_PKG_VERSION"));
}

// RAG-PERSIST-042: Manifest file unreadable returns Ok(None)
#[test]
fn test_manifest_unreadable_returns_none() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Make manifest unreadable by replacing with a directory of the same name
    let manifest_path = tmp.path().join(MANIFEST_FILE);
    fs::remove_file(&manifest_path).unwrap();
    fs::create_dir(&manifest_path).unwrap();

    // Reading a directory as a file should fail with IO error
    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Unreadable manifest should return None");
}

// RAG-PERSIST-043: Index JSON valid checksum but invalid schema returns Ok(None)
#[test]
fn test_index_valid_checksum_invalid_schema_returns_none() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Write valid JSON but wrong schema to index file
    let bad_index = r#"{"not_an_index": true}"#;
    let bad_checksum = blake3_hash(bad_index.as_bytes());

    // Update the index file
    fs::write(tmp.path().join(INDEX_FILE), bad_index).unwrap();

    // Update manifest to have correct checksum for the bad data
    let manifest_path = tmp.path().join(MANIFEST_FILE);
    let manifest_json = fs::read_to_string(&manifest_path).unwrap();
    let mut manifest: RagManifest = serde_json::from_str(&manifest_json).unwrap();
    manifest.index_checksum = bad_checksum;
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

    // Load should pass checksum but fail deserialization => Ok(None)
    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Valid checksum but bad schema should return None");
}

// RAG-PERSIST-044: Documents JSON valid checksum but invalid schema returns Ok(None)
#[test]
fn test_docs_valid_checksum_invalid_schema_returns_none() {
    let (persistence, tmp) = test_persistence();

    persistence.save(&sample_index(), &sample_docs(), sample_sources()).unwrap();

    // Write valid JSON but wrong schema to documents file
    let bad_docs = r#"{"not_documents": 42}"#;
    let bad_checksum = blake3_hash(bad_docs.as_bytes());

    // Update the documents file
    fs::write(tmp.path().join(DOCUMENTS_FILE), bad_docs).unwrap();

    // Update manifest to have correct checksum for the bad data
    let manifest_path = tmp.path().join(MANIFEST_FILE);
    let manifest_json = fs::read_to_string(&manifest_path).unwrap();
    let mut manifest: RagManifest = serde_json::from_str(&manifest_json).unwrap();
    manifest.docs_checksum = bad_checksum;
    fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();

    // Load should pass checksum but fail deserialization => Ok(None)
    let result = persistence.load().unwrap();
    assert!(result.is_none(), "Valid checksum but bad docs schema should return None");
}

// RAG-PERSIST-045: Fingerprints file unreadable triggers fallback
#[test]
fn test_fingerprints_unreadable_triggers_fallback() {
    let (persistence, tmp) = test_persistence();

    let mut docs = sample_docs();
    let fp = DocumentFingerprint {
        content_hash: blake3_hash(b"test"),
        chunker_config_hash: [0; 32],
        embedding_model_hash: [0; 32],
        indexed_at: 12345,
    };
    docs.fingerprints.insert("doc1".to_string(), fp);

    persistence.save(&sample_index(), &docs, sample_sources()).unwrap();

    // Make fingerprints file unreadable by replacing with directory
    let fp_path = tmp.path().join(FINGERPRINTS_FILE);
    fs::remove_file(&fp_path).unwrap();
    fs::create_dir(&fp_path).unwrap();

    // Should fall back to loading from documents.json
    let result = persistence.load_fingerprints_only().unwrap();
    assert!(result.is_some());
    let fps = result.unwrap();
    assert!(fps.contains_key("doc1"));
}

// RAG-PERSIST-046: Stats function returns manifest data
#[test]
fn test_stats_returns_sources_and_timestamp() {
    let (persistence, _tmp) = test_persistence();
    let sources = vec![
        CorpusSource {
            id: "corpus-a".to_string(),
            commit: Some("deadbeef".to_string()),
            doc_count: 5,
            chunk_count: 25,
        },
        CorpusSource { id: "corpus-b".to_string(), commit: None, doc_count: 10, chunk_count: 50 },
    ];
    persistence.save(&sample_index(), &sample_docs(), sources).unwrap();

    let manifest = persistence.stats().unwrap().unwrap();
    assert_eq!(manifest.sources.len(), 2);
    assert_eq!(manifest.sources[0].id, "corpus-a");
    assert_eq!(manifest.sources[1].chunk_count, 50);
    assert!(manifest.indexed_at > 0);
}

// RAG-PERSIST-041: Corpus source fields roundtrip
#[test]
fn test_corpus_source_roundtrip() {
    let (persistence, _tmp) = test_persistence();
    let sources = vec![
        CorpusSource {
            id: "trueno".to_string(),
            commit: Some("abc123".to_string()),
            doc_count: 10,
            chunk_count: 50,
        },
        CorpusSource { id: "aprender".to_string(), commit: None, doc_count: 20, chunk_count: 100 },
    ];

    persistence.save(&sample_index(), &sample_docs(), sources).unwrap();

    let (_, _, manifest) = persistence.load().unwrap().unwrap();
    assert_eq!(manifest.sources.len(), 2);
    assert_eq!(manifest.sources[0].id, "trueno");
    assert_eq!(manifest.sources[0].commit, Some("abc123".to_string()));
    assert_eq!(manifest.sources[0].doc_count, 10);
    assert_eq!(manifest.sources[0].chunk_count, 50);
    assert_eq!(manifest.sources[1].id, "aprender");
    assert!(manifest.sources[1].commit.is_none());
}

// Property-based tests
mod proptests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(20))]

        // Property: Roundtrip preserves avg_doc_length
        #[test]
        fn prop_roundtrip_preserves_avg_doc_length(avg in 0.0f64..1000.0) {
            let tmp = TempDir::new().unwrap();
            let persistence = RagPersistence::with_path(tmp.path().to_path_buf());

            let index = PersistedIndex {
                avg_doc_length: avg,
                ..Default::default()
            };
            let docs = PersistedDocuments::default();

            persistence.save(&index, &docs, vec![]).unwrap();
            let (loaded, _, _) = persistence.load().unwrap().unwrap();

            prop_assert!((loaded.avg_doc_length - avg).abs() < 1e-10);
        }

        // Property: Checksum is deterministic
        #[test]
        fn prop_checksum_deterministic(data in "[a-z]{10,100}") {
            let hash1 = blake3_hash(data.as_bytes());
            let hash2 = blake3_hash(data.as_bytes());
            prop_assert_eq!(hash1, hash2);
        }

        // Property: Different data produces different checksums
        #[test]
        fn prop_different_data_different_checksum(
            data1 in "[a-z]{10,50}",
            data2 in "[A-Z]{10,50}"
        ) {
            // Only check if data is actually different
            if data1 != data2 {
                let hash1 = blake3_hash(data1.as_bytes());
                let hash2 = blake3_hash(data2.as_bytes());
                prop_assert_ne!(hash1, hash2);
            }
        }

        // Property: Total chunks roundtrips correctly
        #[test]
        fn prop_total_chunks_roundtrip(chunks in 0usize..10000) {
            let tmp = TempDir::new().unwrap();
            let persistence = RagPersistence::with_path(tmp.path().to_path_buf());

            let docs = PersistedDocuments {
                total_chunks: chunks,
                ..Default::default()
            };

            persistence.save(&PersistedIndex::default(), &docs, vec![]).unwrap();
            let (_, loaded_docs, _) = persistence.load().unwrap().unwrap();

            prop_assert_eq!(loaded_docs.total_chunks, chunks);
        }
    }
}
