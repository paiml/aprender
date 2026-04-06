pub(crate) use super::*;
pub(crate) use crate::bundle::manifest::ModelMetadata;
pub(crate) use tempfile::NamedTempFile;

#[test]
fn test_bundle_format_validate_magic() {
    assert!(BundleFormat::validate_magic(BUNDLE_MAGIC));
    assert!(BundleFormat::validate_magic(b"APBUNDLEextra"));
    assert!(!BundleFormat::validate_magic(b"INVALID!"));
    assert!(!BundleFormat::validate_magic(b"SHORT"));
}

#[test]
fn test_bundle_format_read_version() {
    let mut header = vec![0u8; 20];
    header[0..8].copy_from_slice(BUNDLE_MAGIC);
    header[8..12].copy_from_slice(&1u32.to_le_bytes());

    assert_eq!(BundleFormat::read_version(&header), Some(1));
}

#[test]
fn test_bundle_format_read_manifest_length() {
    let mut header = vec![0u8; 20];
    header[0..8].copy_from_slice(BUNDLE_MAGIC);
    header[8..12].copy_from_slice(&1u32.to_le_bytes());
    header[12..20].copy_from_slice(&256u64.to_le_bytes());

    assert_eq!(BundleFormat::read_manifest_length(&header), Some(256));
}

#[test]
fn test_bundle_writer_reader_roundtrip() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    // Create manifest and models
    let mut manifest = BundleManifest::new().with_description("Test");
    manifest.add_model(ModelEntry::new("model1", 4));
    manifest.add_model(ModelEntry::new("model2", 3));

    let mut models = HashMap::new();
    models.insert("model1".to_string(), vec![1, 2, 3, 4]);
    models.insert("model2".to_string(), vec![5, 6, 7]);

    // Write
    let writer = BundleWriter::create(path).expect("Failed to create writer");
    writer
        .write_bundle(&manifest, &models)
        .expect("Failed to write bundle");

    // Read
    let mut reader = BundleReader::open(path).expect("Failed to open reader");
    assert_eq!(reader.version(), manifest.version);

    let read_manifest = reader.read_manifest().expect("Failed to read manifest");
    assert_eq!(read_manifest.len(), 2);
    assert_eq!(read_manifest.description, "Test");

    let read_models = reader
        .read_all_models(&read_manifest)
        .expect("Failed to read models");
    assert_eq!(read_models.get("model1"), Some(&vec![1, 2, 3, 4]));
    assert_eq!(read_models.get("model2"), Some(&vec![5, 6, 7]));
}

#[test]
fn test_bundle_reader_invalid_magic() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    // Write invalid data
    std::fs::write(path, b"INVALID!").expect("Failed to write");

    // Should fail to open
    let result = BundleReader::open(path);
    assert!(result.is_err());
}

#[test]
fn test_bundle_read_single_model() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let mut manifest = BundleManifest::new();
    manifest.add_model(ModelEntry::new("weights", 5));

    let mut models = HashMap::new();
    models.insert("weights".to_string(), vec![10, 20, 30, 40, 50]);

    let writer = BundleWriter::create(path).expect("Failed to create writer");
    writer
        .write_bundle(&manifest, &models)
        .expect("Failed to write");

    let mut reader = BundleReader::open(path).expect("Failed to open");
    let manifest = reader.read_manifest().expect("Failed to read manifest");
    let entry = manifest.get_model("weights").expect("Model not found");

    let data = reader.read_model(entry).expect("Failed to read model");
    assert_eq!(data, vec![10, 20, 30, 40, 50]);
}

// ============================================================================
// Additional Coverage Tests
// ============================================================================

#[test]
fn test_bundle_format_read_version_short_header() {
    // Header too short for version
    let header = vec![0u8; 8];
    assert_eq!(BundleFormat::read_version(&header), None);
}

#[test]
fn test_bundle_format_read_manifest_length_short_header() {
    // Header too short for manifest length
    let header = vec![0u8; 15];
    assert_eq!(BundleFormat::read_manifest_length(&header), None);
}

#[test]
fn test_bundle_reader_debug() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    // Create a valid bundle
    let manifest = BundleManifest::new();
    let models = HashMap::new();
    let writer = BundleWriter::create(path).expect("Failed to create writer");
    writer
        .write_bundle(&manifest, &models)
        .expect("Failed to write");

    let reader = BundleReader::open(path).expect("Failed to open");
    let debug_str = format!("{:?}", reader);
    assert!(debug_str.contains("BundleReader"));
    assert!(debug_str.contains("header_version"));
}

#[test]
fn test_bundle_writer_debug() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let writer = BundleWriter::create(path).expect("Failed to create writer");
    let debug_str = format!("{:?}", writer);
    assert!(debug_str.contains("BundleWriter"));
}

#[test]
fn test_bundle_reader_data_offset() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let manifest = BundleManifest::new().with_description("Test offset");
    let models = HashMap::new();
    let writer = BundleWriter::create(path).expect("Failed to create writer");
    writer
        .write_bundle(&manifest, &models)
        .expect("Failed to write");

    let reader = BundleReader::open(path).expect("Failed to open");
    let offset = reader.data_offset();
    // Header is 20 bytes, manifest follows
    assert!(offset >= BundleFormat::HEADER_SIZE as u64);
}

#[test]
fn test_bundle_format_header_size_constant() {
    assert_eq!(BundleFormat::HEADER_SIZE, 20);
}

#[test]
fn test_bundle_format_copy_clone() {
    let format = BundleFormat;
    let _cloned = format;
    let _copied = format;
    // Just testing that Copy + Clone derive works
}

#[test]
fn test_bundle_reader_open_nonexistent() {
    let result = BundleReader::open("/nonexistent/path/bundle.apbundle");
    assert!(result.is_err());
}

#[test]
fn test_bundle_reader_truncated_header() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    // Write only magic, no version or manifest length
    std::fs::write(path, BUNDLE_MAGIC).expect("Failed to write");

    let result = BundleReader::open(path);
    assert!(result.is_err());
}

#[test]
fn test_bundle_writer_create_invalid_path() {
    let result = BundleWriter::create("/nonexistent/directory/bundle.apbundle");
    assert!(result.is_err());
}

#[test]
fn test_bundle_empty_manifest() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let manifest = BundleManifest::new();
    let models = HashMap::new();

    let writer = BundleWriter::create(path).expect("Failed to create writer");
    writer
        .write_bundle(&manifest, &models)
        .expect("Failed to write empty bundle");

    let mut reader = BundleReader::open(path).expect("Failed to open");
    let read_manifest = reader.read_manifest().expect("Failed to read manifest");
    assert_eq!(read_manifest.len(), 0);
}

#[test]
fn test_bundle_multiple_models_order() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    let mut manifest = BundleManifest::new();
    manifest.add_model(ModelEntry::new("first", 3));
    manifest.add_model(ModelEntry::new("second", 4));
    manifest.add_model(ModelEntry::new("third", 5));

    let mut models = HashMap::new();
    models.insert("first".to_string(), vec![1, 2, 3]);
    models.insert("second".to_string(), vec![4, 5, 6, 7]);
    models.insert("third".to_string(), vec![8, 9, 10, 11, 12]);

    let writer = BundleWriter::create(path).expect("Failed to create writer");
    writer
        .write_bundle(&manifest, &models)
        .expect("Failed to write");

    let mut reader = BundleReader::open(path).expect("Failed to open");
    let manifest = reader.read_manifest().expect("Failed to read manifest");
    let all = reader
        .read_all_models(&manifest)
        .expect("Failed to read all");

    assert_eq!(all.len(), 3);
    assert_eq!(all.get("first"), Some(&vec![1, 2, 3]));
    assert_eq!(all.get("second"), Some(&vec![4, 5, 6, 7]));
    assert_eq!(all.get("third"), Some(&vec![8, 9, 10, 11, 12]));
}

// ========================================================================
// Additional Coverage Tests for bundle/format.rs (v2)
// ========================================================================

#[test]
fn test_model_entry_field_access() {
    let entry = ModelEntry::new("test_model", 1024);
    assert_eq!(entry.name, "test_model");
    assert_eq!(entry.size, 1024);
    assert_eq!(entry.offset, 0);
}

#[test]
fn test_model_entry_with_metadata_builder() {
    let meta = ModelMetadata::new(2048)
        .with_version("1.0")
        .with_architecture("transformer");
    let entry = ModelEntry::new("model", 2048).with_metadata(meta);

    assert_eq!(entry.metadata.version, "1.0");
    assert_eq!(entry.metadata.architecture, "transformer");
}

#[test]
fn test_model_entry_clone_v2() {
    let entry = ModelEntry::new("model", 512)
        .with_offset(100)
        .with_component("layer1");

    let cloned = entry.clone();
    assert_eq!(cloned.name, "model");
    assert_eq!(cloned.size, 512);
    assert_eq!(cloned.offset, 100);
    assert_eq!(cloned.components.len(), 1);
}

#[test]
fn test_model_entry_debug_v2() {
    let entry = ModelEntry::new("debug_model", 100);
    let debug_str = format!("{:?}", entry);
    assert!(debug_str.contains("debug_model"));
    assert!(debug_str.contains("100"));
}

#[test]
fn test_bundle_manifest_total_size() {
    let mut manifest = BundleManifest::new();
    manifest.add_model(ModelEntry::new("m1", 100));
    manifest.add_model(ModelEntry::new("m2", 200));
    manifest.add_model(ModelEntry::new("m3", 300));

    assert_eq!(manifest.total_size(), 600);
}

#[test]
fn test_bundle_manifest_get_model_v2() {
    let mut manifest = BundleManifest::new();
    manifest.add_model(ModelEntry::new("target", 1024));
    manifest.add_model(ModelEntry::new("other", 512));

    let model = manifest.get_model("target");
    assert!(model.is_some());
    assert_eq!(model.expect("model exists").size, 1024);

    assert!(manifest.get_model("nonexistent").is_none());
}

#[test]
fn test_bundle_manifest_model_names_order() {
    let mut manifest = BundleManifest::new();
    manifest.add_model(ModelEntry::new("z_last", 100));
    manifest.add_model(ModelEntry::new("a_first", 200));
    manifest.add_model(ModelEntry::new("m_middle", 300));

    // Order should be insertion order, not alphabetical
    let names = manifest.model_names();
    assert_eq!(names[0], "z_last");
    assert_eq!(names[1], "a_first");
    assert_eq!(names[2], "m_middle");
}

#[test]
fn test_bundle_manifest_clone_v2() {
    let mut manifest = BundleManifest::new().with_description("Test");
    manifest.add_model(ModelEntry::new("model", 256));

    let cloned = manifest.clone();
    assert_eq!(cloned.len(), 1);
    assert_eq!(cloned.description, "Test");
    assert!(cloned.get_model("model").is_some());
}

#[test]
fn test_bundle_manifest_debug_v2() {
    let manifest = BundleManifest::new();
    let debug_str = format!("{:?}", manifest);
    assert!(debug_str.contains("BundleManifest"));
}

#[test]
fn test_bundle_large_model_v2() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    // Create a larger model (1KB)
    let large_data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

    let mut manifest = BundleManifest::new();
    manifest.add_model(ModelEntry::new("large", large_data.len()));

    let mut models = HashMap::new();
    models.insert("large".to_string(), large_data.clone());

    let writer = BundleWriter::create(path).expect("Failed to create writer");
    writer
        .write_bundle(&manifest, &models)
        .expect("Failed to write");

    let mut reader = BundleReader::open(path).expect("Failed to open");
    let manifest = reader.read_manifest().expect("Failed to read manifest");
    let entry = manifest.get_model("large").expect("Model should exist");
    let read_data = reader.read_model(entry).expect("Failed to read");

    assert_eq!(read_data.len(), 1024);
    assert_eq!(read_data, large_data);
}

#[test]
fn test_bundle_format_constants() {
    assert_eq!(BundleFormat::HEADER_SIZE, 20);
}

#[test]
fn test_bundle_format_validate_magic_short() {
    // Too short
    let short = vec![0u8; 4];
    assert!(!BundleFormat::validate_magic(&short));
}

#[test]
fn test_bundle_manifest_get_model_mut() {
    let mut manifest = BundleManifest::new();
    manifest.add_model(ModelEntry::new("mutable", 100));

    let model = manifest.get_model_mut("mutable");
    assert!(model.is_some());
    let m = model.expect("model exists");
    m.size = 200;

    // Verify mutation
    let model2 = manifest.get_model("mutable");
    assert_eq!(model2.expect("model exists").size, 200);
}

#[test]
fn test_bundle_manifest_add_same_model_twice() {
    let mut manifest = BundleManifest::new();
    manifest.add_model(ModelEntry::new("dup", 100));
    manifest.add_model(ModelEntry::new("dup", 200)); // Should replace

    assert_eq!(manifest.len(), 1);
    assert_eq!(manifest.get_model("dup").expect("exists").size, 200);
}
