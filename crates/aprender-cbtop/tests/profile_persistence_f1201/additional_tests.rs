//! profile_persistence_f1201 - Part 2

use cbtop::{
    profile_persistence::templates, ProfileBackend as BackendConfig, ProfileConfig, ProfileError,
    ProfileManager, ProfileWorkload as WorkloadConfig,
};
use tempfile::TempDir;

// =============================================================================
// F1208: Profile Export Tests
// =============================================================================

/// F1208.1: Profile export creates file
#[test]
fn f1208_export_creates_file() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().join("profiles"));

    // Save a profile
    let profile = ProfileConfig::new("exportable").unwrap().backend(BackendConfig::Cuda);
    manager.save_profile(&profile).unwrap();

    // Export to different location
    let export_path = temp_dir.path().join("exported.toml");
    manager.export_profile("exportable", &export_path).unwrap();

    // Verify exported file exists and is valid
    assert!(export_path.exists());
    let content = std::fs::read_to_string(&export_path).unwrap();
    assert!(content.contains("name = \"exportable\""));
    assert!(content.contains("backend = \"cuda\""));
}

/// F1208.2: Export non-existent profile fails
#[test]
fn f1208_export_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ProfileManager::new(temp_dir.path().to_path_buf());
    manager.ensure_directory().unwrap();

    let export_path = temp_dir.path().join("fail.toml");
    let result = manager.export_profile("nonexistent", &export_path);
    assert!(matches!(result, Err(ProfileError::NotFound(_))));
}

// =============================================================================
// F1209: Default Profile Tests
// =============================================================================

/// F1209.1: Default profile used when none specified
#[test]
fn f1209_default_profile_fallback() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    // No default set - returns default config
    let profile = manager.load_default();
    assert_eq!(profile.name, "default");
}

/// F1209.2: Named default profile loaded
#[test]
fn f1209_named_default_profile() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    // Save and set as default
    let profile = ProfileConfig::new("my_default").unwrap().backend(BackendConfig::Cuda);
    manager.save_profile(&profile).unwrap();
    manager.set_default("my_default");

    let loaded = manager.load_default();
    assert_eq!(loaded.name, "my_default");
    assert_eq!(loaded.backend, BackendConfig::Cuda);
}

// =============================================================================
// F1210: Profile Description/Metadata Tests
// =============================================================================

/// F1210.1: Profile description stored
#[test]
fn f1210_description_stored() {
    let profile =
        ProfileConfig::with_description("described", "This is a test profile for stress testing")
            .unwrap();

    let toml = profile.to_toml().unwrap();
    assert!(toml.contains("description = \"This is a test profile"));

    let parsed = ProfileConfig::from_toml(&toml).unwrap();
    assert_eq!(parsed.description, "This is a test profile for stress testing");
}

/// F1210.2: Metadata preserved through save/load
#[test]
fn f1210_metadata_preserved() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    let profile = ProfileConfig::new("with_meta")
        .unwrap()
        .with_metadata("author", "test_user")
        .with_metadata("version", "2.0");

    manager.save_profile(&profile).unwrap();

    let loaded = manager.load_profile("with_meta").unwrap();
    assert_eq!(loaded.metadata.get("author"), Some(&"test_user".to_string()));
    assert_eq!(loaded.metadata.get("version"), Some(&"2.0".to_string()));
}

// =============================================================================
// Additional Tests
// =============================================================================

/// Test profile deletion
#[test]
fn test_profile_deletion() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    manager.save_profile(&ProfileConfig::new("deletable").unwrap()).unwrap();
    assert!(manager.profile_exists("deletable"));

    manager.delete_profile("deletable").unwrap();
    assert!(!manager.profile_exists("deletable"));
}

/// Test profile caching
#[test]
fn test_profile_caching() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    let profile = ProfileConfig::new("cached").unwrap();
    manager.save_profile(&profile).unwrap();

    // First load populates cache
    let _ = manager.load_profile("cached").unwrap();

    // Delete from disk but cache remains
    let path = temp_dir.path().join("cached.toml");
    std::fs::remove_file(&path).unwrap();

    // Still loads from cache
    let loaded = manager.load_profile("cached").unwrap();
    assert_eq!(loaded.name, "cached");
}

/// Test profile templates
#[test]
fn test_profile_templates() {
    let ml = templates::ml_training();
    assert_eq!(ml.name, "ml_training");
    assert_eq!(ml.backend, BackendConfig::Cuda);
    assert!(ml.metadata.contains_key("use_case"));

    let inference = templates::inference();
    assert_eq!(inference.name, "inference");
    assert!(inference.deterministic);

    let stress = templates::stress_test();
    assert_eq!(stress.load_intensity, 1.0);
    assert_eq!(stress.workload, WorkloadConfig::All);

    let simd = templates::simd_only();
    assert_eq!(simd.backend, BackendConfig::Simd);
}

/// Test profile import
#[test]
fn test_profile_import() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().join("profiles"));

    // Create an external profile file
    let external_path = temp_dir.path().join("external.toml");
    let profile = ProfileConfig::new("imported").unwrap().backend(BackendConfig::Wgpu);
    std::fs::write(&external_path, profile.to_toml().unwrap()).unwrap();

    // Import it
    let imported = manager.import_profile(&external_path).unwrap();
    assert_eq!(imported.name, "imported");

    // Should now be available locally
    let loaded = manager.load_profile("imported").unwrap();
    assert_eq!(loaded.backend, BackendConfig::Wgpu);
}

/// Test profile count
#[test]
fn test_profile_count() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    assert_eq!(manager.profile_count().unwrap(), 0);

    manager.save_profile(&ProfileConfig::new("one").unwrap()).unwrap();
    assert_eq!(manager.profile_count().unwrap(), 1);

    manager.save_profile(&ProfileConfig::new("two").unwrap()).unwrap();
    assert_eq!(manager.profile_count().unwrap(), 2);
}

/// Test builder pattern
#[test]
fn test_builder_pattern() {
    let profile = ProfileConfig::new("builder_test")
        .unwrap()
        .backend(BackendConfig::Cuda)
        .workload(WorkloadConfig::Gemm)
        .problem_size(2048)
        .load_intensity(0.75)
        .threads(8);

    assert_eq!(profile.backend, BackendConfig::Cuda);
    assert_eq!(profile.workload, WorkloadConfig::Gemm);
    assert_eq!(profile.problem_size, 2048);
    assert_eq!(profile.load_intensity, 0.75);
    assert_eq!(profile.threads, 8);
}

/// Test load intensity clamping
#[test]
fn test_load_intensity_clamping() {
    let profile = ProfileConfig::new("clamp_test").unwrap().load_intensity(1.5); // Should be clamped to 1.0
    assert_eq!(profile.load_intensity, 1.0);

    let profile2 = ProfileConfig::new("clamp_test2").unwrap().load_intensity(-0.5); // Should be clamped to 0.0
    assert_eq!(profile2.load_intensity, 0.0);
}
