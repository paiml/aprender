//! Offline model registry implementation.

use sha2::{Digest, Sha256};
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

use super::manifest::RegistryManifest;
use super::types::{ModelEntry, ModelSource};

/// Offline model registry
#[derive(Debug)]
pub struct OfflineModelRegistry {
    /// Root path for model storage
    pub root_path: PathBuf,
    /// Registry manifest
    pub manifest: RegistryManifest,
    /// Manifest file path
    manifest_path: PathBuf,
}

impl OfflineModelRegistry {
    /// Create a new registry at the given root path
    pub fn new(root: PathBuf) -> Self {
        let manifest_path = root.join("manifest.json");
        let manifest = if manifest_path.exists() {
            Self::load_manifest(&manifest_path).unwrap_or_default()
        } else {
            RegistryManifest::new()
        };

        Self { root_path: root, manifest, manifest_path }
    }

    /// Create registry at default location (~/.entrenar/models/)
    pub fn default_location() -> Self {
        let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("."));
        Self::new(home.join(".entrenar").join("models"))
    }

    /// Load manifest from file
    fn load_manifest(path: &Path) -> Result<RegistryManifest> {
        let content = fs::read_to_string(path)?;
        serde_json::from_str(&content).map_err(|e| Error::Io(format!("Invalid manifest data: {e}")))
    }

    /// Save manifest to file
    pub fn save_manifest(&self) -> Result<()> {
        // Ensure parent directory exists
        if let Some(parent) = self.manifest_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let content = serde_json::to_string_pretty(&self.manifest)
            .map_err(|e| Error::Io(format!("Failed to serialize manifest: {e}")))?;
        fs::write(&self.manifest_path, content)?;
        Ok(())
    }

    /// Add a model entry to the registry
    pub fn add_model(&mut self, entry: ModelEntry) {
        self.manifest.add(entry);
    }

    /// Mirror a model from HuggingFace Hub (simulated for offline scenarios)
    ///
    /// In a real implementation, this would download the model.
    /// For air-gapped scenarios, models are pre-downloaded and registered.
    pub fn mirror_from_hub(&mut self, repo_id: &str) -> Result<ModelEntry> {
        // Create model entry with HuggingFace source
        let name = repo_id.split('/').next_back().unwrap_or(repo_id);
        let local_path = self.root_path.join(name);

        let entry = ModelEntry::new(
            name,
            "1.0",
            "", // Checksum computed after download
            0,  // Size computed after download
            ModelSource::huggingface(repo_id),
        )
        .with_local_path(&local_path);

        self.manifest.add(entry.clone());
        Ok(entry)
    }

    /// Register a local model file
    pub fn register_local(&mut self, name: &str, path: &Path) -> Result<ModelEntry> {
        if !path.exists() {
            return Err(Error::ConfigError(format!("Model file not found: {}", path.display())));
        }

        let metadata = fs::metadata(path)?;
        let size_bytes = metadata.len();

        // Compute SHA-256
        let sha256 = Self::compute_file_sha256(path)?;

        // Determine format from extension
        let format = path.extension().and_then(|e| e.to_str()).map(String::from);

        let entry = ModelEntry::new(name, "local", sha256, size_bytes, ModelSource::local(path))
            .with_local_path(path);

        let entry = if let Some(fmt) = format { entry.with_format(fmt) } else { entry };

        self.manifest.add(entry.clone());
        self.manifest.mark_synced();
        self.save_manifest()?;

        Ok(entry)
    }

    /// Compute SHA-256 hash of a file
    fn compute_file_sha256(path: &Path) -> Result<String> {
        let mut file = fs::File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];

        loop {
            let bytes_read = file.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        Ok(format!("{:x}", hasher.finalize()))
    }

    /// Load a model by name, returning its local path
    pub fn load(&self, name: &str) -> Result<PathBuf> {
        let entry = self
            .manifest
            .find(name)
            .ok_or_else(|| Error::ConfigError(format!("Model not found: {name}")))?;

        let path = entry
            .local_path
            .as_ref()
            .ok_or_else(|| Error::ConfigError(format!("Model not available locally: {name}")))?;

        if !path.exists() {
            return Err(Error::ConfigError(format!("Model file missing: {}", path.display())));
        }

        Ok(path.clone())
    }

    /// Verify a model entry's checksum
    pub fn verify(&self, entry: &ModelEntry) -> Result<bool> {
        let path = entry
            .local_path
            .as_ref()
            .ok_or_else(|| Error::ConfigError("Model has no local path".into()))?;

        if !path.exists() {
            return Ok(false);
        }

        if entry.sha256.is_empty() {
            // No checksum to verify against
            return Ok(true);
        }

        let computed = Self::compute_file_sha256(path)?;
        Ok(computed == entry.sha256)
    }

    /// List all available (locally cached) models
    pub fn list_available(&self) -> Vec<&ModelEntry> {
        self.manifest.available()
    }

    /// List all models in registry
    pub fn list_all(&self) -> &[ModelEntry] {
        &self.manifest.models
    }

    /// Get a model entry by name
    pub fn get(&self, name: &str) -> Option<&ModelEntry> {
        self.manifest.find(name)
    }

    /// Remove a model from registry (does not delete files)
    pub fn remove(&mut self, name: &str) -> Option<ModelEntry> {
        let pos = self.manifest.models.iter().position(|m| m.name == name)?;
        Some(self.manifest.models.remove(pos))
    }

    /// Get total size of all models
    pub fn total_size(&self) -> u64 {
        self.manifest.total_size_bytes()
    }

    /// Get root path
    pub fn root(&self) -> &Path {
        &self.root_path
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn temp_registry_dir() -> PathBuf {
        let id = TEST_COUNTER.fetch_add(1, Ordering::SeqCst);
        let dir =
            std::env::temp_dir().join(format!("entrenar_offline_test_{}_{id}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_new_empty_registry() {
        let dir = temp_registry_dir();
        let reg = OfflineModelRegistry::new(dir.clone());
        assert!(reg.manifest.models.is_empty());
        assert_eq!(reg.root(), dir.as_path());
        assert_eq!(reg.total_size(), 0);
        assert!(reg.list_all().is_empty());
        assert!(reg.list_available().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_new_loads_existing_manifest() {
        let dir = temp_registry_dir();
        let manifest_path = dir.join("manifest.json");
        let manifest = RegistryManifest::new();
        let content = serde_json::to_string_pretty(&manifest).unwrap();
        std::fs::write(&manifest_path, content).unwrap();

        let reg = OfflineModelRegistry::new(dir.clone());
        assert!(reg.manifest.models.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_new_with_corrupted_manifest_falls_back() {
        let dir = temp_registry_dir();
        let manifest_path = dir.join("manifest.json");
        std::fs::write(&manifest_path, "not valid json").unwrap();

        let reg = OfflineModelRegistry::new(dir.clone());
        assert!(reg.manifest.models.is_empty()); // falls back to default
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_add_model() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry =
            ModelEntry::new("test-model", "1.0", "abc123", 1024, ModelSource::local("/tmp/model"));
        reg.add_model(entry);
        assert_eq!(reg.list_all().len(), 1);
        assert_eq!(reg.list_all()[0].name, "test-model");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_get_model() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = ModelEntry::new("mymodel", "2.0", "sha", 2048, ModelSource::local("/tmp/m"));
        reg.add_model(entry);

        assert!(reg.get("mymodel").is_some());
        assert_eq!(reg.get("mymodel").unwrap().version, "2.0");
        assert!(reg.get("nonexistent").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_remove_model() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = ModelEntry::new("removeme", "1.0", "hash", 512, ModelSource::local("/tmp"));
        reg.add_model(entry);
        assert_eq!(reg.list_all().len(), 1);

        let removed = reg.remove("removeme");
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "removeme");
        assert!(reg.list_all().is_empty());

        // Remove nonexistent
        assert!(reg.remove("nonexistent").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_save_manifest() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = ModelEntry::new("saved", "1.0", "sha256", 100, ModelSource::local("/tmp"));
        reg.add_model(entry);
        reg.save_manifest().unwrap();

        // Verify file was written
        let manifest_path = dir.join("manifest.json");
        assert!(manifest_path.exists());

        // Load it back
        let reg2 = OfflineModelRegistry::new(dir.clone());
        assert_eq!(reg2.list_all().len(), 1);
        assert_eq!(reg2.list_all()[0].name, "saved");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mirror_from_hub() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = reg.mirror_from_hub("org/my-model").unwrap();
        assert_eq!(entry.name, "my-model");
        assert_eq!(reg.list_all().len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_mirror_from_hub_no_slash() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = reg.mirror_from_hub("simple-model").unwrap();
        assert_eq!(entry.name, "simple-model");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_register_local_file() {
        let dir = temp_registry_dir();
        let model_file = dir.join("model.safetensors");
        let mut f = std::fs::File::create(&model_file).unwrap();
        f.write_all(b"fake model data for testing").unwrap();

        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = reg.register_local("local-model", &model_file).unwrap();
        assert_eq!(entry.name, "local-model");
        assert_eq!(entry.version, "local");
        assert!(!entry.sha256.is_empty());
        assert!(entry.size_bytes > 0);
        assert_eq!(entry.format, Some("safetensors".to_string()));
        assert!(reg.list_all().len() == 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_register_local_file_not_found() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        let result = reg.register_local("missing", Path::new("/tmp/nonexistent_model_xyz"));
        assert!(result.is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_register_local_no_extension() {
        let dir = temp_registry_dir();
        let model_file = dir.join("model_no_ext");
        std::fs::write(&model_file, b"data").unwrap();

        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = reg.register_local("noext", &model_file).unwrap();
        assert!(entry.format.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_model_found() {
        let dir = temp_registry_dir();
        let model_file = dir.join("loadable.bin");
        std::fs::write(&model_file, b"model content").unwrap();

        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = ModelEntry::new("loadable", "1.0", "", 100, ModelSource::local(&model_file))
            .with_local_path(&model_file);
        reg.add_model(entry);

        let path = reg.load("loadable").unwrap();
        assert_eq!(path, model_file);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_model_not_found() {
        let dir = temp_registry_dir();
        let reg = OfflineModelRegistry::new(dir.clone());
        assert!(reg.load("nonexistent").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_model_no_local_path() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = ModelEntry::new("no-path", "1.0", "", 0, ModelSource::huggingface("org/model"));
        reg.add_model(entry);
        assert!(reg.load("no-path").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_model_file_missing() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        let entry = ModelEntry::new("gone", "1.0", "", 0, ModelSource::local("/tmp/gone_xyz"))
            .with_local_path("/tmp/gone_xyz");
        reg.add_model(entry);
        assert!(reg.load("gone").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_no_local_path() {
        let dir = temp_registry_dir();
        let reg = OfflineModelRegistry::new(dir.clone());
        let entry = ModelEntry::new("no-path", "1.0", "sha", 0, ModelSource::huggingface("org/m"));
        assert!(reg.verify(&entry).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_file_missing() {
        let dir = temp_registry_dir();
        let reg = OfflineModelRegistry::new(dir.clone());
        let entry = ModelEntry::new("missing", "1.0", "sha", 0, ModelSource::local("/tmp/nope"))
            .with_local_path("/tmp/nope_xyz_verify");
        let result = reg.verify(&entry).unwrap();
        assert!(!result); // file doesn't exist
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_empty_checksum() {
        let dir = temp_registry_dir();
        let model_file = dir.join("verify_empty.bin");
        std::fs::write(&model_file, b"data").unwrap();

        let reg = OfflineModelRegistry::new(dir.clone());
        let entry = ModelEntry::new("verify-empty", "1.0", "", 0, ModelSource::local(&model_file))
            .with_local_path(&model_file);
        let result = reg.verify(&entry).unwrap();
        assert!(result); // empty checksum always passes
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_checksum_match() {
        let dir = temp_registry_dir();
        let model_file = dir.join("verify_match.bin");
        std::fs::write(&model_file, b"test content for sha256").unwrap();

        // Compute actual sha256
        let computed = OfflineModelRegistry::compute_file_sha256(&model_file).unwrap();

        let reg = OfflineModelRegistry::new(dir.clone());
        let entry =
            ModelEntry::new("verify-match", "1.0", &computed, 0, ModelSource::local(&model_file))
                .with_local_path(&model_file);
        let result = reg.verify(&entry).unwrap();
        assert!(result);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_checksum_mismatch() {
        let dir = temp_registry_dir();
        let model_file = dir.join("verify_mismatch.bin");
        std::fs::write(&model_file, b"some data").unwrap();

        let reg = OfflineModelRegistry::new(dir.clone());
        let entry =
            ModelEntry::new("mismatch", "1.0", "wrong_hash", 0, ModelSource::local(&model_file))
                .with_local_path(&model_file);
        let result = reg.verify(&entry).unwrap();
        assert!(!result);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_total_size() {
        let dir = temp_registry_dir();
        let mut reg = OfflineModelRegistry::new(dir.clone());
        reg.add_model(ModelEntry::new("m1", "1.0", "", 100, ModelSource::local("/tmp")));
        reg.add_model(ModelEntry::new("m2", "1.0", "", 200, ModelSource::local("/tmp")));
        reg.add_model(ModelEntry::new("m3", "1.0", "", 300, ModelSource::local("/tmp")));
        assert_eq!(reg.total_size(), 600);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
