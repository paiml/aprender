use super::*;

#[test]
fn test_opt_level_copy() {
    let l1 = OptLevel::Debug;
    let l2 = l1; // Copy
    assert_eq!(l1, l2);
}

// ==================== Coverage: RustEdition PartialEq/Copy ====================

#[test]
fn test_rust_edition_eq() {
    assert_eq!(RustEdition::E2015, RustEdition::E2015);
    assert_ne!(RustEdition::E2015, RustEdition::E2018);
}

#[test]
fn test_rust_edition_copy() {
    let e1 = RustEdition::E2024;
    let e2 = e1; // Copy
    assert_eq!(e1, e2);
}

// ==================== Coverage: CompilationMode Clone ====================

#[test]
fn test_compilation_mode_clone() {
    let mode = CompilationMode::CargoCheck {
        manifest_path: PathBuf::from("/test/Cargo.toml"),
    };
    let cloned = mode.clone();
    match cloned {
        CompilationMode::CargoCheck { manifest_path } => {
            assert_eq!(manifest_path, PathBuf::from("/test/Cargo.toml"));
        }
        _ => panic!("Expected CargoCheck"),
    }
}
