//! Smoke tests for basic functionality

#[test]
fn test_version_exists() {
    // Verify the crate version string is valid semver
    let version = env!("CARGO_PKG_VERSION");
    assert!(!version.is_empty());
    let parts: Vec<&str> = version.split('.').collect();
    assert_eq!(parts.len(), 3, "Version should be semver: {version}");
}

#[test]
fn test_package_name() {
    let name = env!("CARGO_PKG_NAME");
    assert!(!name.is_empty());
}
