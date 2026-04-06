//! File I/O tests - loading and saving manifests

use crate::yaml_mode::*;
use tempfile::TempDir;

#[test]
fn test_load_manifest_success() {
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let manifest_path = temp_dir.path().join("train.yaml");

    let yaml_content = r#"
entrenar: "1.0"
name: "test-experiment"
version: "1.0.0"
"#;
    std::fs::write(&manifest_path, yaml_content).expect("file write should succeed");

    let manifest = load_manifest(&manifest_path).expect("load should succeed");
    assert_eq!(manifest.entrenar, "1.0");
    assert_eq!(manifest.name, "test-experiment");
    assert_eq!(manifest.version, "1.0.0");
}

#[test]
fn test_load_manifest_file_not_found() {
    let result = load_manifest(std::path::Path::new("/nonexistent/path/train.yaml"));
    assert!(result.is_err());
}

#[test]
fn test_load_manifest_invalid_yaml() {
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let manifest_path = temp_dir.path().join("invalid.yaml");

    std::fs::write(&manifest_path, "this is not valid yaml: [[[")
        .expect("file write should succeed");

    let result = load_manifest(&manifest_path);
    assert!(result.is_err());
}

#[test]
fn test_load_manifest_validation_fails() {
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let manifest_path = temp_dir.path().join("invalid_manifest.yaml");

    // Invalid version
    let yaml_content = r#"
entrenar: "99.0"
name: "test"
version: "1.0.0"
"#;
    std::fs::write(&manifest_path, yaml_content).expect("file write should succeed");

    let result = load_manifest(&manifest_path);
    assert!(result.is_err());
}

#[test]
fn test_save_manifest_success() {
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let manifest_path = temp_dir.path().join("output.yaml");

    let yaml = r#"
entrenar: "1.0"
name: "save-test"
version: "1.0.0"
description: "Test saving manifest"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");

    save_manifest(&manifest, &manifest_path).expect("save should succeed");

    // Verify file was written
    assert!(manifest_path.exists());

    // Verify content can be read back
    let loaded = load_manifest(&manifest_path).expect("load should succeed");
    assert_eq!(loaded.name, "save-test");
    assert_eq!(loaded.description, Some("Test saving manifest".to_string()));
}

#[test]
fn test_save_manifest_creates_parent_dir() {
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let nested_path = temp_dir.path().join("nested").join("dir").join("train.yaml");

    let yaml = r#"
entrenar: "1.0"
name: "nested-test"
version: "1.0.0"
"#;
    let manifest: TrainingManifest = serde_yaml::from_str(yaml).expect("operation should succeed");

    // This should fail since we don't create parent dirs in save_manifest
    let result = save_manifest(&manifest, &nested_path);
    assert!(result.is_err());
}
