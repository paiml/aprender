//! Tests for Nix flake configuration

use super::*;

#[test]
fn test_crate_spec_local() {
    let spec = CrateSpec::local("test", "crates/test");

    assert_eq!(spec.name, "test");
    assert!(spec.is_local());
    assert!(!spec.is_crates_io());
    assert!(!spec.is_git());
}

#[test]
fn test_crate_spec_crates_io() {
    let spec = CrateSpec::crates_io("test", "1.0.0");

    assert_eq!(spec.name, "test");
    assert!(!spec.is_local());
    assert!(spec.is_crates_io());
    assert!(!spec.is_git());
}

#[test]
fn test_crate_spec_git() {
    let spec = CrateSpec::git("test", "https://github.com/test/repo", "abc123");

    assert_eq!(spec.name, "test");
    assert!(!spec.is_local());
    assert!(!spec.is_crates_io());
    assert!(spec.is_git());
}

#[test]
fn test_crate_spec_nix_source() {
    assert!(CrateSpec::local("a", "path").nix_source().contains("path"));
    assert!(CrateSpec::crates_io("b", "1.0").nix_source().contains("1.0"));
    assert!(CrateSpec::git("c", "url", "rev").nix_source().contains("rev"));
}

#[test]
fn test_nix_system_as_str() {
    assert_eq!(NixSystem::X86_64Linux.as_str(), "x86_64-linux");
    assert_eq!(NixSystem::Aarch64Linux.as_str(), "aarch64-linux");
    assert_eq!(NixSystem::X86_64Darwin.as_str(), "x86_64-darwin");
    assert_eq!(NixSystem::Aarch64Darwin.as_str(), "aarch64-darwin");
}

#[test]
fn test_nix_system_all() {
    let all = NixSystem::all();
    assert_eq!(all.len(), 4);
}

#[test]
fn test_nix_system_linux_only() {
    let linux = NixSystem::linux_only();
    assert_eq!(linux.len(), 2);
    assert!(linux.contains(&NixSystem::X86_64Linux));
    assert!(linux.contains(&NixSystem::Aarch64Linux));
}

#[test]
fn test_nix_flake_config_new() {
    let config = NixFlakeConfig::new("Test flake");

    assert_eq!(config.description, "Test flake");
    assert!(config.crates.is_empty());
    assert_eq!(config.rust_version, "1.75.0");
}

#[test]
fn test_nix_flake_config_sovereign_stack() {
    let config = NixFlakeConfig::sovereign_stack();

    assert!(!config.crates.is_empty());
    assert!(config.crates.iter().any(|c| c.name == "trueno"));
    assert!(config.crates.iter().any(|c| c.name == "entrenar"));
    assert!(config.features.contains_key("trueno"));
}

#[test]
fn test_nix_flake_config_add_crate() {
    let config = NixFlakeConfig::new("Test").add_crate(CrateSpec::crates_io("test", "1.0"));

    assert_eq!(config.crates.len(), 1);
    assert_eq!(config.crates[0].name, "test");
}

#[test]
fn test_nix_flake_config_with_features() {
    let config = NixFlakeConfig::new("Test").with_features("test", ["feat1", "feat2"]);

    let features = config.features.get("test").expect("key should exist");
    assert_eq!(features.len(), 2);
}

#[test]
fn test_nix_flake_config_with_systems() {
    let config = NixFlakeConfig::new("Test").with_systems(NixSystem::linux_only());

    assert_eq!(config.systems.len(), 2);
}

#[test]
fn test_nix_flake_config_generate_flake_nix() {
    let config = NixFlakeConfig::sovereign_stack();
    let flake = config.generate_flake_nix();

    assert!(flake.contains("description ="));
    assert!(flake.contains("nixpkgs.url"));
    assert!(flake.contains("rust-overlay"));
    assert!(flake.contains("crane"));
    assert!(flake.contains("packages"));
    assert!(flake.contains("devShells"));
}

#[test]
fn test_nix_flake_config_generate_flake_nix_with_gpu() {
    let config = NixFlakeConfig::new("Test")
        .add_crate(CrateSpec::crates_io("test", "1.0"))
        .with_gpu_support(true);

    let flake = config.generate_flake_nix();
    assert!(flake.contains("cudatoolkit"));
}

#[test]
fn test_nix_flake_config_generate_cachix_config() {
    let config = NixFlakeConfig::sovereign_stack();
    let cachix = config.generate_cachix_config();

    assert!(cachix.contains("paiml"));
    assert!(cachix.contains("binary_caches"));
}

#[test]
fn test_nix_flake_config_minimal_flake() {
    let flake = NixFlakeConfig::minimal_flake("mylib", "crates/mylib");

    assert!(flake.contains("mylib"));
    assert!(flake.contains("crates/mylib"));
    assert!(flake.contains("packages.default"));
}

#[test]
fn test_nix_flake_config_default() {
    let config = NixFlakeConfig::default();

    // Default should be sovereign_stack
    assert!(!config.crates.is_empty());
    assert!(config.description.contains("Sovereign"));
}

#[test]
fn test_crate_spec_serialization() {
    let spec = CrateSpec::crates_io("test", "1.0.0");
    let json = serde_json::to_string(&spec).expect("JSON serialization should succeed");
    let parsed: CrateSpec =
        serde_json::from_str(&json).expect("JSON deserialization should succeed");

    assert_eq!(spec.name, parsed.name);
    assert_eq!(spec.version, parsed.version);
}

#[test]
fn test_nix_system_display() {
    assert_eq!(format!("{}", NixSystem::X86_64Linux), "x86_64-linux");
}
