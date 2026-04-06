//! Tests for RO-Crate module.

use super::*;
use crate::research::artifact::{Affiliation, ArtifactType, Author, License, ResearchArtifact};
use tempfile::TempDir;

fn create_test_artifact() -> ResearchArtifact {
    let author = Author::new("Alice Smith")
        .with_orcid("0000-0002-1825-0097")
        .expect("operation should succeed")
        .with_affiliation(Affiliation::new("MIT"));

    ResearchArtifact::new("dataset-001", "Test Dataset", ArtifactType::Dataset, License::CcBy4)
        .with_author(author)
        .with_doi("10.1234/test")
        .with_description("A test dataset")
        .with_keywords(["test", "dataset"])
}

#[test]
fn test_ro_crate_directory_creation() {
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let crate_path = temp_dir.path().join("test-crate");

    let crate_pkg = RoCrate::new(&crate_path);
    let result = crate_pkg.to_directory();

    assert!(result.is_ok());
    assert!(crate_path.exists());
    assert!(crate_path.join("ro-crate-metadata.json").exists());
}

#[test]
fn test_ro_crate_metadata_json() {
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let crate_path = temp_dir.path().join("test-crate");

    let crate_pkg = RoCrate::new(&crate_path);
    crate_pkg.to_directory().expect("operation should succeed");

    let metadata_content = std::fs::read_to_string(crate_path.join("ro-crate-metadata.json"))
        .expect("file read should succeed");
    let parsed: serde_json::Value =
        serde_json::from_str(&metadata_content).expect("JSON deserialization should succeed");

    assert_eq!(parsed["@context"], RO_CRATE_CONTEXT);
    assert!(parsed["@graph"].is_array());
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn test_ro_crate_zip_creation() {
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let crate_path = temp_dir.path().join("test-crate");

    let mut crate_pkg = RoCrate::new(&crate_path);
    crate_pkg.add_text_file("data.csv", "a,b,c\n1,2,3");

    let zip_data = crate_pkg.to_zip().expect("operation should succeed");

    assert!(!zip_data.is_empty());

    // Verify ZIP structure
    let reader = std::io::Cursor::new(zip_data);
    let mut archive = zip::ZipArchive::new(reader).expect("operation should succeed");

    assert!(archive.by_name("ro-crate-metadata.json").is_ok());
    assert!(archive.by_name("data.csv").is_ok());
}

#[test]
fn test_ro_crate_entities_linked() {
    let artifact = create_test_artifact();
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let crate_path = temp_dir.path().join("test-crate");

    let crate_pkg = RoCrate::from_artifact(&artifact, &crate_path);

    // Should have: metadata descriptor, root dataset, author, organization
    assert!(crate_pkg.entity_count() >= 4);

    // Check root dataset has author reference
    let root = crate_pkg.descriptor.root_dataset().expect("operation should succeed");
    assert!(root.properties.contains_key("author"));
    assert!(root.properties.contains_key("name"));
}

#[test]
fn test_ro_crate_includes_data_files() {
    let temp_dir = TempDir::new().expect("temp file creation should succeed");
    let crate_path = temp_dir.path().join("test-crate");

    let mut crate_pkg = RoCrate::new(&crate_path);
    crate_pkg.add_text_file("data/train.csv", "x,y\n1,2");
    crate_pkg.add_text_file("data/test.csv", "x,y\n3,4");
    crate_pkg.add_file("model.safetensors", vec![0u8; 100]);

    crate_pkg.to_directory().expect("operation should succeed");

    assert!(crate_path.join("data/train.csv").exists());
    assert!(crate_path.join("data/test.csv").exists());
    assert!(crate_path.join("model.safetensors").exists());

    // Check file entities in descriptor
    assert_eq!(crate_pkg.file_count(), 3);
}

#[test]
fn test_entity_creation() {
    let entity = RoCrateEntity::new("#test", EntityType::Dataset)
        .with_name("Test Dataset")
        .with_description("A test")
        .with_reference("author", "#person-1");

    assert_eq!(entity.id, "#test");
    assert_eq!(entity.type_field, "Dataset");
    assert!(entity.properties.contains_key("name"));
    assert!(entity.properties.contains_key("description"));
    assert!(entity.properties.contains_key("author"));
}

#[test]
fn test_person_entity() {
    use serde_json::json;

    let person = RoCrateEntity::person("#alice", "Alice Smith");

    assert_eq!(person.id, "#alice");
    assert_eq!(person.type_field, "Person");
    assert_eq!(person.properties.get("name"), Some(&json!("Alice Smith")));
}

#[test]
fn test_mime_type_guessing() {
    assert_eq!(guess_mime_type("data.json"), "application/json");
    assert_eq!(guess_mime_type("data.csv"), "text/csv");
    assert_eq!(guess_mime_type("script.py"), "text/x-python");
    assert_eq!(guess_mime_type("lib.rs"), "text/x-rust");
    assert_eq!(guess_mime_type("unknown.xyz"), "application/octet-stream");
}

#[test]
fn test_descriptor_serialization() {
    let descriptor = RoCrateDescriptor::new();
    let json = descriptor.to_json();

    assert!(json.contains("@context"));
    assert!(json.contains("@graph"));
    assert!(json.contains("ro-crate-metadata.json"));
}

#[test]
fn test_entity_type_display() {
    assert_eq!(format!("{}", EntityType::Dataset), "Dataset");
    assert_eq!(format!("{}", EntityType::Person), "Person");
    assert_eq!(format!("{}", EntityType::Custom("MyType".to_string())), "MyType");
}

#[test]
fn test_artifact_metadata_in_crate() {
    use serde_json::json;

    let artifact = create_test_artifact();
    let temp_dir = TempDir::new().expect("temp file creation should succeed");

    let crate_pkg = RoCrate::from_artifact(&artifact, temp_dir.path().join("crate"));

    let root = crate_pkg.descriptor.root_dataset().expect("operation should succeed");

    assert_eq!(root.properties.get("name"), Some(&json!("Test Dataset")));
    assert_eq!(root.properties.get("version"), Some(&json!("1.0.0")));
    assert_eq!(root.properties.get("license"), Some(&json!("CC-BY-4.0")));
    assert_eq!(root.properties.get("identifier"), Some(&json!("10.1234/test")));
}
