//! Tests for the offline model registry.

use super::*;
use std::fs;
use tempfile::TempDir;

#[test]
fn test_model_source_huggingface() {
    let source = ModelSource::huggingface("bert-base-uncased");
    assert_eq!(source.display_string(), "hf://bert-base-uncased");
}

#[test]
fn test_model_source_local() {
    let source = ModelSource::local("/path/to/model.gguf");
    assert!(source.display_string().contains("/path/to/model.gguf"));
}

#[test]
fn test_model_source_custom() {
    let source = ModelSource::custom("https://example.com/model.bin");
    assert_eq!(source.display_string(), "https://example.com/model.bin");
}

#[test]
fn test_model_entry_new() {
    let entry = ModelEntry::new(
        "test-model",
        "1.0.0",
        "abc123",
        1024 * 1024 * 100, // 100 MB
        ModelSource::huggingface("test/model"),
    );

    assert_eq!(entry.name, "test-model");
    assert_eq!(entry.version, "1.0.0");
    assert_eq!(entry.sha256, "abc123");
    assert_eq!(entry.size_bytes, 100 * 1024 * 1024);
    assert!(!entry.is_local());
}

#[test]
fn test_model_entry_size_conversions() {
    let entry = ModelEntry::new(
        "test",
        "1.0",
        "",
        1024 * 1024 * 1024, // 1 GB
        ModelSource::huggingface("test"),
    );

    assert!((entry.size_mb() - 1024.0).abs() < 0.01);
    assert!((entry.size_gb() - 1.0).abs() < 0.01);
}

#[test]
fn test_model_entry_with_format() {
    let entry =
        ModelEntry::new("test", "1.0", "", 0, ModelSource::huggingface("test")).with_format("gguf");

    assert_eq!(entry.format, Some("gguf".to_string()));
}

#[test]
fn test_model_entry_with_metadata() {
    let entry = ModelEntry::new("test", "1.0", "", 0, ModelSource::huggingface("test"))
        .with_metadata("architecture", "llama")
        .with_metadata("quantization", "q4_0");

    assert_eq!(entry.metadata.get("architecture"), Some(&"llama".to_string()));
    assert_eq!(entry.metadata.get("quantization"), Some(&"q4_0".to_string()));
}

#[test]
fn test_registry_manifest_new() {
    let manifest = RegistryManifest::new();

    assert!(manifest.models.is_empty());
    assert!(manifest.last_sync.is_none());
    assert_eq!(manifest.version, "1.0");
}

#[test]
fn test_registry_manifest_add_and_find() {
    let mut manifest = RegistryManifest::new();

    let entry = ModelEntry::new("test", "1.0", "", 100, ModelSource::huggingface("test"));
    manifest.add(entry);

    assert_eq!(manifest.len(), 1);
    assert!(manifest.find("test").is_some());
    assert!(manifest.find("nonexistent").is_none());
}

#[test]
fn test_registry_manifest_update_existing() {
    let mut manifest = RegistryManifest::new();

    let entry1 = ModelEntry::new("test", "1.0", "", 100, ModelSource::huggingface("test"));
    manifest.add(entry1);

    let entry2 = ModelEntry::new("test", "2.0", "", 200, ModelSource::huggingface("test"));
    manifest.add(entry2);

    assert_eq!(manifest.len(), 1);
    assert_eq!(manifest.find("test").expect("operation should succeed").version, "2.0");
}

#[test]
fn test_registry_manifest_total_size() {
    let mut manifest = RegistryManifest::new();

    manifest.add(ModelEntry::new("a", "1", "", 100, ModelSource::huggingface("a")));
    manifest.add(ModelEntry::new("b", "1", "", 200, ModelSource::huggingface("b")));

    assert_eq!(manifest.total_size_bytes(), 300);
}

#[test]
fn test_offline_registry_new() {
    let temp = TempDir::new().expect("temp file creation should succeed");
    let registry = OfflineModelRegistry::new(temp.path().to_path_buf());

    assert_eq!(registry.root(), temp.path());
    assert!(registry.manifest.is_empty());
}

#[test]
fn test_offline_registry_add_model() {
    let temp = TempDir::new().expect("temp file creation should succeed");
    let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());

    let entry = ModelEntry::new("test", "1.0", "", 100, ModelSource::huggingface("test"));
    registry.add_model(entry);

    assert_eq!(registry.manifest.len(), 1);
    assert!(registry.get("test").is_some());
}

#[test]
fn test_offline_registry_mirror_from_hub() {
    let temp = TempDir::new().expect("temp file creation should succeed");
    let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());

    let entry = registry.mirror_from_hub("bert-base-uncased").expect("operation should succeed");

    assert_eq!(entry.name, "bert-base-uncased");
    assert!(matches!(entry.source, ModelSource::HuggingFace { .. }));
}

#[test]
fn test_offline_registry_register_local() {
    let temp = TempDir::new().expect("temp file creation should succeed");
    let model_file = temp.path().join("test.gguf");
    fs::write(&model_file, "test model content").expect("file write should succeed");

    let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());
    let entry =
        registry.register_local("test-model", &model_file).expect("operation should succeed");

    assert_eq!(entry.name, "test-model");
    assert!(entry.is_local());
    assert!(!entry.sha256.is_empty());
    assert_eq!(entry.format, Some("gguf".to_string()));
}

#[test]
fn test_offline_registry_load() {
    let temp = TempDir::new().expect("temp file creation should succeed");
    let model_file = temp.path().join("test.gguf");
    fs::write(&model_file, "test content").expect("file write should succeed");

    let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());
    registry.register_local("test", &model_file).expect("operation should succeed");

    let loaded = registry.load("test").expect("load should succeed");
    assert_eq!(loaded, model_file);
}

#[test]
fn test_offline_registry_load_not_found() {
    let temp = TempDir::new().expect("temp file creation should succeed");
    let registry = OfflineModelRegistry::new(temp.path().to_path_buf());

    let result = registry.load("nonexistent");
    assert!(result.is_err());
}

#[test]
fn test_offline_registry_verify() {
    let temp = TempDir::new().expect("temp file creation should succeed");
    let model_file = temp.path().join("test.bin");
    fs::write(&model_file, "test content").expect("file write should succeed");

    let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());
    let entry = registry.register_local("test", &model_file).expect("operation should succeed");

    assert!(registry.verify(&entry).expect("operation should succeed"));

    // Modify file
    fs::write(&model_file, "modified content").expect("file write should succeed");
    assert!(!registry.verify(&entry).expect("operation should succeed"));
}

#[test]
fn test_offline_registry_list_available() {
    let temp = TempDir::new().expect("temp file creation should succeed");
    let model_file = temp.path().join("local.bin");
    fs::write(&model_file, "content").expect("file write should succeed");

    let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());

    // Add local model
    registry.register_local("local", &model_file).expect("operation should succeed");

    // Add remote model (not available locally)
    registry.add_model(ModelEntry::new(
        "remote",
        "1.0",
        "",
        100,
        ModelSource::huggingface("remote"),
    ));

    let available = registry.list_available();
    assert_eq!(available.len(), 1);
    assert_eq!(available[0].name, "local");
}

#[test]
fn test_offline_registry_remove() {
    let temp = TempDir::new().expect("temp file creation should succeed");
    let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());

    registry.add_model(ModelEntry::new("test", "1.0", "", 100, ModelSource::huggingface("test")));
    assert_eq!(registry.manifest.len(), 1);

    let removed = registry.remove("test");
    assert!(removed.is_some());
    assert_eq!(registry.manifest.len(), 0);
}

#[test]
fn test_offline_registry_save_and_load() {
    let temp = TempDir::new().expect("temp file creation should succeed");

    {
        let mut registry = OfflineModelRegistry::new(temp.path().to_path_buf());
        registry.add_model(ModelEntry::new(
            "test",
            "1.0",
            "abc",
            100,
            ModelSource::huggingface("test"),
        ));
        registry.save_manifest().expect("save should succeed");
    }

    // Load in new instance
    let registry = OfflineModelRegistry::new(temp.path().to_path_buf());
    assert_eq!(registry.manifest.len(), 1);
    assert!(registry.get("test").is_some());
}

#[test]
fn test_model_entry_serialization() {
    let entry =
        ModelEntry::new("test", "1.0", "abc123", 1000, ModelSource::huggingface("test/model"))
            .with_format("gguf")
            .with_metadata("arch", "llama");

    let json = serde_json::to_string(&entry).expect("JSON serialization should succeed");
    let parsed: ModelEntry =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(entry.name, parsed.name);
    assert_eq!(entry.format, parsed.format);
    assert_eq!(entry.metadata, parsed.metadata);
}
