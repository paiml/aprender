//! Manifest system for immutability enforcement.
//!
//! # Purpose
//!
//! Ensures generated JavaScript files cannot be manually modified.
//! Any modification is detected via hash verification.
//!
//! # Workflow
//!
//! 1. Generate JS → write .js file + .js.manifest.json
//! 2. On load → verify hash in manifest matches file
//! 3. If mismatch → error with regeneration instructions
//!
//! # References
//! - DO-178C (2011) Section 6.3.5: Configuration management
//! - Leveson (2012) "Engineering a Safer World" - Traceability

use crate::error::{JsGenError, Result};
use crate::hir::GenerationMetadata;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Manifest for a generated file.
///
/// Stored alongside generated files as `<filename>.manifest.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileManifest {
    /// Version of manifest format
    pub manifest_version: u32,
    /// Path to generated file (relative)
    pub output_path: String,
    /// Blake3 hash of generated file contents
    pub output_hash: String,
    /// Generation metadata
    pub generation: GenerationMetadata,
}

impl FileManifest {
    /// Current manifest format version.
    pub const VERSION: u32 = 1;

    /// Create a new manifest.
    #[must_use]
    pub fn new(
        output_path: impl Into<String>,
        output_hash: impl Into<String>,
        generation: GenerationMetadata,
    ) -> Self {
        Self {
            manifest_version: Self::VERSION,
            output_path: output_path.into(),
            output_hash: output_hash.into(),
            generation,
        }
    }

    /// Get the manifest file path for a generated file.
    #[must_use]
    pub fn manifest_path(generated_path: &Path) -> std::path::PathBuf {
        let mut path = generated_path.to_path_buf();
        let mut filename = generated_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        filename.push_str(".manifest.json");
        path.set_file_name(filename);
        path
    }

    /// Write manifest to file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be written.
    pub fn write(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Read manifest from file.
    ///
    /// # Errors
    ///
    /// Returns error if file cannot be read or parsed.
    pub fn read(path: &Path) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let manifest: Self = serde_json::from_str(&json)?;
        Ok(manifest)
    }
}

/// Compute Blake3 hash of file contents.
#[must_use]
pub fn hash_file_contents(contents: &str) -> String {
    blake3::hash(contents.as_bytes()).to_hex().to_string()
}

/// Verify a generated file matches its manifest.
///
/// # Errors
///
/// Returns error if:
/// - Manifest file doesn't exist
/// - Manifest cannot be parsed
/// - Hash mismatch (file was modified)
pub fn verify(generated_path: &Path) -> Result<()> {
    let manifest_path = FileManifest::manifest_path(generated_path);

    // Read manifest
    let manifest = FileManifest::read(&manifest_path).map_err(|_| JsGenError::ManifestError {
        path: generated_path.display().to_string(),
        reason: format!("manifest not found at {}", manifest_path.display()),
    })?;

    // Read generated file
    let contents = std::fs::read_to_string(generated_path)?;
    let actual_hash = hash_file_contents(&contents);

    // Compare hashes
    if actual_hash != manifest.output_hash {
        return Err(JsGenError::HashMismatch {
            path: generated_path.display().to_string(),
            expected: manifest.output_hash,
            actual: actual_hash,
        });
    }

    Ok(())
}

/// Write generated JavaScript with manifest.
///
/// Creates both the .js file and .js.manifest.json file.
///
/// # Errors
///
/// Returns error if files cannot be written.
pub fn write_with_manifest(
    path: &Path,
    contents: &str,
    metadata: GenerationMetadata,
) -> Result<()> {
    // Compute hash
    let hash = hash_file_contents(contents);

    // Write JS file
    std::fs::write(path, contents)?;

    // Create and write manifest
    let manifest = FileManifest::new(
        path.file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default(),
        hash,
        metadata,
    );
    let manifest_path = FileManifest::manifest_path(path);
    manifest.write(&manifest_path)?;

    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn test_metadata() -> GenerationMetadata {
        GenerationMetadata {
            tool: "probar-js-gen".to_string(),
            version: "0.1.0".to_string(),
            input_hash: "input123".to_string(),
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            regenerate_cmd: "probar gen js".to_string(),
        }
    }

    #[test]
    fn manifest_path_generation() {
        let path = Path::new("/foo/bar/worker.js");
        let manifest = FileManifest::manifest_path(path);
        assert_eq!(manifest.file_name().unwrap(), "worker.js.manifest.json");
    }

    #[test]
    fn hash_deterministic() {
        let content = "const x = 42;";
        let hash1 = hash_file_contents(content);
        let hash2 = hash_file_contents(content);
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn hash_changes_with_content() {
        let hash1 = hash_file_contents("const x = 42;");
        let hash2 = hash_file_contents("const x = 43;");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn write_and_verify_success() -> Result<()> {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.js");
        let contents = "const x = 42;";

        write_with_manifest(&path, contents, test_metadata())?;
        verify(&path)?;

        Ok(())
    }

    #[test]
    fn verify_detects_modification() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.js");
        let contents = "const x = 42;";

        write_with_manifest(&path, contents, test_metadata()).unwrap();

        // Modify the file
        std::fs::write(&path, "const x = 999;").unwrap();

        // Verification should fail
        let result = verify(&path);
        assert!(result.is_err());

        match result.unwrap_err() {
            JsGenError::HashMismatch {
                path: _,
                expected,
                actual,
            } => {
                assert_ne!(expected, actual);
            }
            e => panic!("Expected HashMismatch, got {:?}", e),
        }
    }

    #[test]
    fn verify_missing_manifest() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.js");
        std::fs::write(&path, "const x = 42;").unwrap();

        // No manifest file
        let result = verify(&path);
        assert!(result.is_err());

        match result.unwrap_err() {
            JsGenError::ManifestError { .. } => {}
            e => panic!("Expected ManifestError, got {:?}", e),
        }
    }

    #[test]
    fn manifest_serialization() {
        let manifest = FileManifest::new("test.js", "abc123", test_metadata());

        let json = serde_json::to_string(&manifest).unwrap();
        let parsed: FileManifest = serde_json::from_str(&json).unwrap();

        assert_eq!(manifest.output_path, parsed.output_path);
        assert_eq!(manifest.output_hash, parsed.output_hash);
        assert_eq!(manifest.manifest_version, FileManifest::VERSION);
    }
}
