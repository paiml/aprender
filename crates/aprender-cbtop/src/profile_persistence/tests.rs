use super::*;
use tempfile::TempDir;

#[test]
fn test_profile_config_default() {
    let config = ProfileConfig::default();
    assert_eq!(config.name, "default");
    assert_eq!(config.refresh_ms, 100);
    assert_eq!(config.backend, BackendConfig::All);
}

#[test]
fn test_profile_config_new() {
    let config = ProfileConfig::new("my_profile").unwrap();
    assert_eq!(config.name, "my_profile");
}

#[test]
fn test_profile_name_validation() {
    assert!(validate_profile_name("valid_name").is_ok());
    assert!(validate_profile_name("valid-name").is_ok());
    assert!(validate_profile_name("valid123").is_ok());
    assert!(validate_profile_name("").is_err());
    assert!(validate_profile_name("-invalid").is_err());
    assert!(validate_profile_name("1invalid").is_err());
    assert!(validate_profile_name("inv@lid").is_err());
}

#[test]
fn test_profile_toml_serialization() {
    let config = ProfileConfig::new("test").unwrap();
    let toml = config.to_toml().unwrap();
    assert!(toml.contains("name = \"test\""));

    let parsed = ProfileConfig::from_toml(&toml).unwrap();
    assert_eq!(parsed.name, "test");
}

#[test]
fn test_profile_manager_save_load() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    let profile = ProfileConfig::new("test_profile").unwrap();
    manager.save_profile(&profile).unwrap();

    let loaded = manager.load_profile("test_profile").unwrap();
    assert_eq!(loaded.name, "test_profile");
}

#[test]
fn test_profile_manager_list() {
    let temp_dir = TempDir::new().unwrap();
    let mut manager = ProfileManager::new(temp_dir.path().to_path_buf());

    manager.save_profile(&ProfileConfig::new("profile_a").unwrap()).unwrap();
    manager.save_profile(&ProfileConfig::new("profile_b").unwrap()).unwrap();

    let profiles = manager.list_profiles().unwrap();
    assert_eq!(profiles.len(), 2);
    assert!(profiles.contains(&"profile_a".to_string()));
    assert!(profiles.contains(&"profile_b".to_string()));
}

#[test]
fn test_profile_overlay() {
    let profile = ProfileConfig::default();
    let overlay = ProfileOverlay::new().refresh_ms(200).backend(BackendConfig::Cuda);

    let merged = overlay.apply(profile);
    assert_eq!(merged.refresh_ms, 200);
    assert_eq!(merged.backend, BackendConfig::Cuda);
}

#[test]
fn test_templates() {
    let ml = templates::ml_training();
    assert_eq!(ml.name, "ml_training");
    assert_eq!(ml.backend, BackendConfig::Cuda);

    let inference = templates::inference();
    assert_eq!(inference.name, "inference");

    let stress = templates::stress_test();
    assert_eq!(stress.load_intensity, 1.0);
}
