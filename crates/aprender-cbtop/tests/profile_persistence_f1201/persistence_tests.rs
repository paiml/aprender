//! profile_persistence_f1201 - Part 1

use cbtop::{
    ProfileBackend as BackendConfig, ProfileConfig, ProfileError, ProfileManager, ProfileOverlay,
    ProfileWorkload as WorkloadConfig,
};
use tempfile::TempDir;

// =============================================================================
// F1201: Profile Loading Tests
// =============================================================================

/// F1201.1: Profile loaded by name
#[test]
fn f1201_profile_loaded_by_name() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    // Save a profile
    let profile =
        ProfileConfig::new("my_profile").unwrap().backend(BackendConfig::Cuda).problem_size(2048);
    manager.save_profile(&profile).unwrap();

    // Load by name
    let loaded = manager.load_profile("my_profile").unwrap();
    assert_eq!(loaded.name, "my_profile");
    assert_eq!(loaded.backend, BackendConfig::Cuda);
    assert_eq!(loaded.problem_size, 2048);
}

/// F1201.2: Profile not found returns error
#[test]
fn f1201_profile_not_found() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());
    manager.ensure_directory().unwrap();

    let result = manager.load_profile("nonexistent");
    assert!(matches!(result, Err(ProfileError::NotFound(_))));
}

// =============================================================================
// F1202: Profile Saving Tests
// =============================================================================

/// F1202.1: Profile saved to disk
#[test]
fn f1202_profile_saved_to_disk() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    let profile = ProfileConfig::new("saved_profile").unwrap();
    let path = manager.save_profile(&profile).unwrap();

    // Verify file exists
    assert!(path.exists());
    assert!(path.to_str().unwrap().ends_with(".toml"));

    // Verify content is valid TOML
    let content = std::fs::read_to_string(&path).unwrap();
    assert!(content.contains("name = \"saved_profile\""));
}

/// F1202.2: Profile content is valid TOML
#[test]
fn f1202_profile_content_valid_toml() {
    let profile = ProfileConfig::new("toml_test")
        .unwrap()
        .backend(BackendConfig::Simd)
        .workload(WorkloadConfig::Attention);

    let toml = profile.to_toml().unwrap();

    // Should be parseable
    let parsed: ProfileConfig = toml::from_str(&toml).unwrap();
    assert_eq!(parsed.name, "toml_test");
    assert_eq!(parsed.backend, BackendConfig::Simd);
    assert_eq!(parsed.workload, WorkloadConfig::Attention);
}

// =============================================================================
// F1203: Profile Listing Tests
// =============================================================================

/// F1203.1: Profile listing works
#[test]
fn f1203_profile_listing() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    // Save multiple profiles
    manager.save_profile(&ProfileConfig::new("alpha").unwrap()).unwrap();
    manager.save_profile(&ProfileConfig::new("beta").unwrap()).unwrap();
    manager.save_profile(&ProfileConfig::new("gamma").unwrap()).unwrap();

    let profiles = manager.list_profiles().unwrap();
    assert_eq!(profiles.len(), 3);
    assert!(profiles.contains(&"alpha".to_string()));
    assert!(profiles.contains(&"beta".to_string()));
    assert!(profiles.contains(&"gamma".to_string()));
}

/// F1203.2: Empty directory returns empty list
#[test]
fn f1203_empty_listing() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ProfileManager::new(temp_dir.path().to_path_buf());

    let profiles = manager.list_profiles().unwrap();
    assert!(profiles.is_empty());
}

// =============================================================================
// F1204: CLI Overlay Merging Tests
// =============================================================================

/// F1204.1: CLI overlay merges correctly
#[test]
fn f1204_overlay_merges() {
    let profile =
        ProfileConfig::new("base").unwrap().backend(BackendConfig::Simd).problem_size(1024);

    let overlay = ProfileOverlay::new()
        .backend(BackendConfig::Cuda) // Override backend
        .refresh_ms(50); // Override refresh

    let merged = overlay.apply(profile);

    // CLI overrides should take precedence
    assert_eq!(merged.backend, BackendConfig::Cuda);
    assert_eq!(merged.refresh_ms, 50);
    // Profile value should be preserved
    assert_eq!(merged.problem_size, 1024);
}

/// F1204.2: Empty overlay preserves profile
#[test]
fn f1204_empty_overlay() {
    let profile =
        ProfileConfig::new("unchanged").unwrap().backend(BackendConfig::Wgpu).problem_size(4096);

    let overlay = ProfileOverlay::new();
    assert!(!overlay.has_overrides());

    let merged = overlay.apply(profile.clone());
    assert_eq!(merged.backend, profile.backend);
    assert_eq!(merged.problem_size, profile.problem_size);
}

// =============================================================================
// F1205: Invalid Profile Handling Tests
// =============================================================================

/// F1205.1: Invalid profile returns error
#[test]
fn f1205_invalid_profile_error() {
    // Invalid name with special characters
    let result = ProfileConfig::new("inv@lid!");
    assert!(matches!(result, Err(ProfileError::InvalidName(_))));

    // Empty name
    let result = ProfileConfig::new("");
    assert!(matches!(result, Err(ProfileError::InvalidName(_))));

    // Starting with number
    let result = ProfileConfig::new("123invalid");
    assert!(matches!(result, Err(ProfileError::InvalidName(_))));
}

/// F1205.2: Invalid TOML returns parse error
#[test]
fn f1205_invalid_toml() {
    let invalid_toml = "this is not { valid toml";
    let result = ProfileConfig::from_toml(invalid_toml);
    assert!(matches!(result, Err(ProfileError::ParseError(_))));
}

// =============================================================================
// F1206: Profile Directory Creation Tests
// =============================================================================

/// F1206.1: Profile directory created automatically
#[test]
fn f1206_directory_created() {
    let temp_dir = TempDir::new().unwrap();
    let subdir = temp_dir.path().join("nested").join("profiles");
    let mut manager = ProfileManager::new(subdir.clone());

    // Directory doesn't exist yet
    assert!(!subdir.exists());

    // Save creates directory
    manager.save_profile(&ProfileConfig::new("test").unwrap()).unwrap();

    // Now it exists
    assert!(subdir.exists());
}

/// F1206.2: ensure_directory is idempotent
#[test]
fn f1206_ensure_directory_idempotent() {
    let temp_dir = TempDir::new().unwrap();
    let manager = ProfileManager::new(temp_dir.path().join("profiles"));

    // Call multiple times - should not fail
    manager.ensure_directory().unwrap();
    manager.ensure_directory().unwrap();
    manager.ensure_directory().unwrap();
}

// =============================================================================
// F1207: Profile Name Validation Tests
// =============================================================================

/// F1207.1: Valid names accepted
#[test]
fn f1207_valid_names() {
    assert!(ProfileConfig::new("valid_name").is_ok());
    assert!(ProfileConfig::new("valid-name").is_ok());
    assert!(ProfileConfig::new("ValidName").is_ok());
    assert!(ProfileConfig::new("valid123").is_ok());
    assert!(ProfileConfig::new("a").is_ok());
}

/// F1207.2: Invalid characters rejected
#[test]
fn f1207_invalid_chars_rejected() {
    assert!(ProfileConfig::new("no spaces").is_err());
    assert!(ProfileConfig::new("no.dots").is_err());
    assert!(ProfileConfig::new("no/slashes").is_err());
    assert!(ProfileConfig::new("no\\backslash").is_err());
    assert!(ProfileConfig::new("no:colons").is_err());
}

/// F1207.3: Name length limits enforced
#[test]
fn f1207_name_length_limits() {
    // Too long (>64 chars)
    let long_name = "a".repeat(65);
    assert!(ProfileConfig::new(&long_name).is_err());

    // Just at limit (64 chars)
    let max_name = "a".repeat(64);
    assert!(ProfileConfig::new(&max_name).is_ok());
}
