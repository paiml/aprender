use super::*;
use tempfile::NamedTempFile;

#[test]
fn test_bundle_config_default() {
    let config = BundleConfig::default();
    assert!(!config.compress);
    assert_eq!(config.max_memory, DEFAULT_MAX_MEMORY);
    assert_eq!(config.page_size, DEFAULT_PAGE_SIZE);
    assert!(config.prefetch);
}

#[test]
fn test_bundle_config_builder() {
    let config = BundleConfig::new()
        .with_compression(true)
        .with_max_memory(50_000_000)
        .with_page_size(8192)
        .with_prefetch(false);

    assert!(config.compress);
    assert_eq!(config.max_memory, 50_000_000);
    assert_eq!(config.page_size, 8192);
    assert!(!config.prefetch);
}

#[test]
fn test_model_bundle_new() {
    let bundle = ModelBundle::new();
    assert!(bundle.is_empty());
    assert_eq!(bundle.len(), 0);
    assert_eq!(bundle.total_size(), 0);
}

#[test]
fn test_model_bundle_add_model() {
    let mut bundle = ModelBundle::new();
    let data = vec![1u8, 2, 3, 4, 5];

    bundle.add_model("test_model", data.clone());

    assert_eq!(bundle.len(), 1);
    assert!(!bundle.is_empty());
    assert_eq!(bundle.get_model("test_model"), Some(data.as_slice()));
    assert!(bundle.model_names().contains(&"test_model"));
}

#[test]
fn test_model_bundle_multiple_models() {
    let mut bundle = ModelBundle::new();

    bundle.add_model("model1", vec![1, 2, 3]);
    bundle.add_model("model2", vec![4, 5, 6, 7]);
    bundle.add_model("model3", vec![8]);

    assert_eq!(bundle.len(), 3);
    assert_eq!(bundle.total_size(), 8);
    assert_eq!(bundle.model_names().len(), 3);
}

#[test]
fn test_model_bundle_get_nonexistent() {
    let bundle = ModelBundle::new();
    assert!(bundle.get_model("nonexistent").is_none());
}

#[test]
fn test_bundle_builder() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path().to_string_lossy().to_string();

    let bundle = BundleBuilder::new(&path)
        .add_model("model1", vec![1, 2, 3])
        .add_model("model2", vec![4, 5])
        .with_config(BundleConfig::new().with_compression(false))
        .build()
        .expect("Failed to build bundle");

    assert_eq!(bundle.len(), 2);
    assert_eq!(bundle.get_model("model1"), Some(&[1u8, 2, 3][..]));
}

#[test]
fn test_bundle_save_load() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    // Create and save
    let mut original = ModelBundle::new();
    original.add_model("weights", vec![1, 2, 3, 4, 5]);
    original.add_model("biases", vec![6, 7, 8]);
    original.save(path).expect("Failed to save bundle");

    // Load and verify
    let loaded = ModelBundle::load(path).expect("Failed to load bundle");
    assert_eq!(loaded.len(), 2);
    assert_eq!(loaded.get_model("weights"), Some(&[1u8, 2, 3, 4, 5][..]));
    assert_eq!(loaded.get_model("biases"), Some(&[6u8, 7, 8][..]));
}

// =========================================================================
// Additional coverage tests
// =========================================================================

#[test]
fn test_bundle_config_clone() {
    let config = BundleConfig::new()
        .with_compression(true)
        .with_max_memory(10_000_000)
        .with_page_size(8192)
        .with_prefetch(false);

    let cloned = config.clone();
    assert_eq!(cloned.compress, config.compress);
    assert_eq!(cloned.max_memory, config.max_memory);
    assert_eq!(cloned.page_size, config.page_size);
    assert_eq!(cloned.prefetch, config.prefetch);
}

#[test]
fn test_bundle_config_debug() {
    let config = BundleConfig::default();
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("BundleConfig"));
    assert!(debug_str.contains("compress"));
}

#[test]
fn test_bundle_config_min_memory_enforced() {
    let config = BundleConfig::new().with_max_memory(100);
    // Should enforce minimum to page size
    assert_eq!(config.max_memory, DEFAULT_PAGE_SIZE);
}

#[test]
fn test_bundle_config_min_page_size_enforced() {
    let config = BundleConfig::new().with_page_size(100);
    // Should enforce minimum to 512
    assert_eq!(config.page_size, 512);
}

#[test]
fn test_model_bundle_builder_static() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path().to_string_lossy().to_string();

    // Test the static builder method
    let bundle = ModelBundle::builder(&path)
        .add_model("model1", vec![1, 2, 3])
        .build()
        .expect("Failed to build bundle");

    assert_eq!(bundle.len(), 1);
}

#[test]
fn test_model_bundle_manifest() {
    let mut bundle = ModelBundle::new();
    bundle.add_model("test", vec![1, 2, 3]);

    let manifest = bundle.manifest();
    assert_eq!(manifest.model_names().len(), 1);
}

#[test]
fn test_model_bundle_config_getter() {
    let bundle = ModelBundle::new();
    let config = bundle.config();
    assert!(!config.compress);
    assert!(config.prefetch);
}

#[test]
fn test_model_bundle_get_metadata() {
    let mut bundle = ModelBundle::new();
    bundle.add_model("test_model", vec![1, 2, 3, 4, 5]);

    let metadata = bundle.get_metadata("test_model");
    assert!(metadata.is_some());
    let entry = metadata.expect("Metadata should exist");
    assert_eq!(entry.name, "test_model");
}

#[test]
fn test_model_bundle_get_metadata_nonexistent() {
    let bundle = ModelBundle::new();
    assert!(bundle.get_metadata("nonexistent").is_none());
}

#[test]
fn test_model_bundle_default() {
    let bundle = ModelBundle::default();
    assert!(bundle.is_empty());
    assert_eq!(bundle.len(), 0);
}

#[test]
fn test_model_bundle_debug() {
    let bundle = ModelBundle::new();
    let debug_str = format!("{:?}", bundle);
    assert!(debug_str.contains("ModelBundle"));
}

#[test]
fn test_bundle_builder_debug() {
    let builder = BundleBuilder::new("test.apbundle");
    let debug_str = format!("{:?}", builder);
    assert!(debug_str.contains("BundleBuilder"));
}

#[test]
fn test_bundle_builder_with_metadata() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path().to_string_lossy().to_string();

    let metadata = ModelMetadata::new(100).with_version("1.0.0");

    let bundle = BundleBuilder::new(&path)
        .add_model_with_metadata("model1", vec![1, 2, 3], metadata)
        .build()
        .expect("Failed to build bundle");

    assert_eq!(bundle.len(), 1);
}

#[test]
fn test_model_bundle_load_paged() {
    let temp = NamedTempFile::new().expect("Failed to create temp file");
    let path = temp.path();

    // Create and save a bundle first
    let mut bundle = ModelBundle::new();
    bundle.add_model("weights", vec![1, 2, 3, 4, 5]);
    bundle.save(path).expect("Failed to save bundle");

    // Load with paging
    let paged = ModelBundle::load_paged(path, 1_000_000).expect("Failed to load paged");
    let names = paged.model_names();
    assert!(names.iter().any(|n| *n == "weights"));
}

#[test]
fn test_bundle_constants() {
    assert_eq!(BUNDLE_MAGIC.len(), 8);
    assert_eq!(BUNDLE_VERSION, 1);
    assert_eq!(DEFAULT_PAGE_SIZE, 4096);
    assert!(DEFAULT_MAX_MEMORY > 0);
}
