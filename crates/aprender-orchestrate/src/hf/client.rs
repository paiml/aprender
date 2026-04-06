//! HuggingFace Hub Client
//!
//! Production-hardened client addressing review findings:
//! - Rate limit handling (429 with exponential backoff)
//! - SafeTensors enforcement (--safe-only default)
//! - Model card auto-generation
//! - Differential uploads (content-addressable)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// ============================================================================
// HF-CLIENT-001: Rate Limiting
// ============================================================================

/// Rate limit configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Initial backoff duration
    pub initial_backoff: Duration,
    /// Maximum backoff duration
    pub max_backoff: Duration,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Backoff multiplier
    pub multiplier: f64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            initial_backoff: Duration::from_secs(1),
            max_backoff: Duration::from_secs(60),
            max_retries: 5,
            multiplier: 2.0,
        }
    }
}

/// Rate limit state for tracking backoff
#[derive(Debug, Clone)]
pub struct RateLimitState {
    pub retry_count: u32,
    pub current_backoff: Duration,
    pub retry_after: Option<Duration>,
}

impl RateLimitState {
    pub fn new() -> Self {
        Self { retry_count: 0, current_backoff: Duration::from_secs(1), retry_after: None }
    }

    /// Calculate next backoff duration with exponential increase
    pub fn next_backoff(&mut self, config: &RateLimitConfig) -> Option<Duration> {
        if self.retry_count >= config.max_retries {
            return None; // Give up
        }

        self.retry_count += 1;

        // Use Retry-After header if provided, otherwise exponential backoff
        let backoff = self.retry_after.unwrap_or_else(|| {
            let backoff_secs = config.initial_backoff.as_secs_f64()
                * config.multiplier.powi(self.retry_count as i32 - 1);
            Duration::from_secs_f64(backoff_secs.min(config.max_backoff.as_secs_f64()))
        });

        self.current_backoff = backoff;
        Some(backoff)
    }

    /// Reset state after successful request
    pub fn reset(&mut self) {
        self.retry_count = 0;
        self.current_backoff = Duration::from_secs(1);
        self.retry_after = None;
    }

    /// Check if we should retry
    pub fn should_retry(&self, config: &RateLimitConfig) -> bool {
        self.retry_count < config.max_retries
    }
}

impl Default for RateLimitState {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HF-CLIENT-002: SafeTensors Enforcement
// ============================================================================

/// Safety policy for model downloads
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SafetyPolicy {
    /// Only allow SafeTensors format (default, secure)
    #[default]
    SafeOnly,
    /// Allow unsafe formats with explicit consent
    AllowUnsafe,
}

/// File safety classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileSafety {
    /// Safe format (SafeTensors, JSON, etc.)
    Safe,
    /// Unsafe format (pickle, PyTorch .bin)
    Unsafe,
    /// Unknown format
    Unknown,
}

/// Classify file safety based on extension
pub fn classify_file_safety(filename: &str) -> FileSafety {
    const SAFE_EXTENSIONS: &[&str] =
        &[".safetensors", ".json", ".txt", ".md", ".gguf", ".ggml", ".yaml", ".yml", ".toml"];
    const UNSAFE_EXTENSIONS: &[&str] = &[".bin", ".pt", ".pth", ".pkl", ".pickle"];

    let lower = filename.to_lowercase();
    if SAFE_EXTENSIONS.iter().any(|ext| lower.ends_with(ext)) {
        FileSafety::Safe
    } else if UNSAFE_EXTENSIONS.iter().any(|ext| lower.ends_with(ext)) {
        FileSafety::Unsafe
    } else {
        FileSafety::Unknown
    }
}

/// Check if download should be allowed based on policy
pub fn check_download_allowed(files: &[&str], policy: SafetyPolicy) -> Result<(), Vec<String>> {
    if policy == SafetyPolicy::AllowUnsafe {
        return Ok(());
    }

    let unsafe_files: Vec<String> = files
        .iter()
        .filter(|f| classify_file_safety(f) == FileSafety::Unsafe)
        .map(|f| (*f).to_string())
        .collect();

    // SAFETY: no actual unsafe code -- variable tracks files classified as unsafe for download
    if unsafe_files.is_empty() {
        Ok(())
    } else {
        Err(unsafe_files)
    }
}

// ============================================================================
// HF-CLIENT-003: Model Card Generation
// ============================================================================

/// Model card metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCardMetadata {
    pub model_name: String,
    pub language: Option<String>,
    pub license: Option<String>,
    pub tags: Vec<String>,
    pub library_name: Option<String>,
    pub pipeline_tag: Option<String>,
    pub datasets: Vec<String>,
    pub metrics: HashMap<String, f64>,
}

impl ModelCardMetadata {
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            language: None,
            license: None,
            tags: Vec::new(),
            library_name: Some("paiml".to_string()),
            pipeline_tag: None,
            datasets: Vec::new(),
            metrics: HashMap::new(),
        }
    }

    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    pub fn with_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.metrics.insert(name.into(), value);
        self
    }
}

/// Generate model card content
pub fn generate_model_card(metadata: &ModelCardMetadata) -> String {
    let mut card = String::new();

    // YAML frontmatter
    card.push_str("---\n");
    let optional_fields: &[(&str, Option<&str>)] = &[
        ("license", metadata.license.as_deref()),
        ("language", metadata.language.as_deref()),
        ("library_name", metadata.library_name.as_deref()),
        ("pipeline_tag", metadata.pipeline_tag.as_deref()),
    ];
    for (key, value) in optional_fields {
        if let Some(v) = value {
            card.push_str(&format!("{}: {}\n", key, v));
        }
    }
    if !metadata.tags.is_empty() {
        card.push_str("tags:\n");
        for tag in &metadata.tags {
            card.push_str(&format!("  - {}\n", tag));
        }
    }
    card.push_str("---\n\n");

    // Title
    card.push_str(&format!("# {}\n\n", metadata.model_name));

    // Description
    card.push_str("## Model Description\n\n");
    card.push_str("This model was trained using the PAIML stack.\n\n");

    // Metrics
    if !metadata.metrics.is_empty() {
        card.push_str("## Evaluation Results\n\n");
        card.push_str("| Metric | Value |\n");
        card.push_str("|--------|-------|\n");
        for (name, value) in &metadata.metrics {
            card.push_str(&format!("| {} | {:.4} |\n", name, value));
        }
        card.push('\n');
    }

    // Footer
    card.push_str("## Training Details\n\n");
    card.push_str("Trained with [PAIML Stack](https://github.com/paiml).\n");

    card
}

// ============================================================================
// HF-CLIENT-004: Differential Uploads
// ============================================================================

/// File hash for content-addressable storage
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileHash {
    pub sha256: String,
    pub size: u64,
}

impl FileHash {
    pub fn new(sha256: impl Into<String>, size: u64) -> Self {
        Self { sha256: sha256.into(), size }
    }

    /// Compute hash from content
    pub fn from_content(content: &[u8]) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple hash for demo (in production use sha2 crate)
        let mut hasher = DefaultHasher::new();
        content.hash(&mut hasher);
        let hash = hasher.finish();

        Self { sha256: format!("{:016x}", hash), size: content.len() as u64 }
    }
}

/// Manifest of files for differential upload
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadManifest {
    pub files: HashMap<String, FileHash>,
}

impl UploadManifest {
    pub fn new() -> Self {
        Self { files: HashMap::new() }
    }

    pub fn add_file(&mut self, path: impl Into<String>, hash: FileHash) {
        self.files.insert(path.into(), hash);
    }

    /// Compare with remote manifest to find changed files
    pub fn diff(&self, remote: &UploadManifest) -> Vec<String> {
        self.files
            .iter()
            .filter(|(path, hash)| remote.files.get(*path) != Some(hash))
            .map(|(path, _)| path.clone())
            .collect()
    }

    /// Get total size of files to upload
    pub fn total_size(&self, files: &[String]) -> u64 {
        files.iter().filter_map(|f| self.files.get(f)).map(|h| h.size).sum()
    }
}

impl Default for UploadManifest {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// HF-CLIENT-005: Secret Scanning (Poka-Yoke)
// ============================================================================

/// Detected secret type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecretType {
    ApiKey,
    EnvFile,
    PrivateKey,
    Password,
}

/// Secret detection result
#[derive(Debug, Clone)]
pub struct SecretDetection {
    pub file: String,
    pub secret_type: SecretType,
    pub line: Option<usize>,
}

/// Scan files for secrets before push
/// Detect the secret type for a filename, if any.
fn detect_secret_type(lower: &str) -> Option<SecretType> {
    const RULES: &[(&[&str], SecretType)] = &[
        (&[".env", ".env.", "env"], SecretType::EnvFile),
        (&[".pem", ".key", "id_rsa", "id_ed25519"], SecretType::PrivateKey),
        (&["credentials", "secrets", "password"], SecretType::Password),
    ];
    RULES.iter().find_map(|(patterns, secret_type)| {
        patterns.iter().any(|p| lower.contains(p)).then_some(*secret_type)
    })
}

pub fn scan_for_secrets(files: &[&str]) -> Vec<SecretDetection> {
    files
        .iter()
        .filter_map(|file| {
            detect_secret_type(&file.to_lowercase()).map(|secret_type| SecretDetection {
                file: (*file).to_string(),
                secret_type,
                line: None,
            })
        })
        .collect()
}

/// Check if push should be blocked due to secrets
pub fn check_push_allowed(files: &[&str]) -> Result<(), Vec<SecretDetection>> {
    let secrets = scan_for_secrets(files);
    if secrets.is_empty() {
        Ok(())
    } else {
        Err(secrets)
    }
}

// ============================================================================
// Tests - Extreme TDD
// ============================================================================

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    // ========================================================================
    // Test Helpers
    // ========================================================================

    /// Assert that a filename is classified with the expected safety level.
    fn assert_file_safety(filename: &str, expected: FileSafety) {
        assert_eq!(
            classify_file_safety(filename),
            expected,
            "Expected {filename} to be {expected:?}"
        );
    }

    /// Build an UploadManifest from a slice of (path, sha256, size) tuples.
    fn test_manifest(files: &[(&str, &str, u64)]) -> UploadManifest {
        let mut manifest = UploadManifest::new();
        for &(path, sha, size) in files {
            manifest.add_file(path, FileHash::new(sha, size));
        }
        manifest
    }

    /// Build a ModelCardMetadata with common test defaults.
    fn make_metadata(
        license: Option<&str>,
        tags: &[&str],
        metrics: &[(&str, f64)],
    ) -> ModelCardMetadata {
        let mut meta = ModelCardMetadata::new("test-model");
        if let Some(lic) = license {
            meta = meta.with_license(lic);
        }
        for tag in tags {
            meta = meta.with_tag(*tag);
        }
        for &(name, value) in metrics {
            meta = meta.with_metric(name, value);
        }
        meta
    }

    /// Generate a model card from metadata built via `make_metadata` and assert
    /// the rendered card contains every string in `expected`.
    fn assert_card_contains(
        license: Option<&str>,
        tags: &[&str],
        metrics: &[(&str, f64)],
        expected: &[&str],
    ) {
        let meta = make_metadata(license, tags, metrics);
        let card = generate_model_card(&meta);
        for s in expected {
            assert!(card.contains(s), "Card missing expected string: {s:?}");
        }
    }

    // ========================================================================
    // HF-CLIENT-001: Rate Limit Tests
    // ========================================================================

    #[test]
    fn test_HF_CLIENT_001_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.initial_backoff, Duration::from_secs(1));
        assert_eq!(config.max_retries, 5);
        assert_eq!(config.multiplier, 2.0);
    }

    #[test]
    fn test_HF_CLIENT_001_rate_limit_state_new() {
        let state = RateLimitState::new();
        assert_eq!(state.retry_count, 0);
        assert!(state.retry_after.is_none());
    }

    #[test]
    fn test_HF_CLIENT_001_rate_limit_exponential_backoff() {
        let config = RateLimitConfig::default();
        let mut state = RateLimitState::new();

        // First retry: 1s
        let backoff1 = state.next_backoff(&config).expect("unexpected failure");
        assert_eq!(backoff1, Duration::from_secs(1));

        // Second retry: 2s
        let backoff2 = state.next_backoff(&config).expect("unexpected failure");
        assert_eq!(backoff2, Duration::from_secs(2));

        // Third retry: 4s
        let backoff3 = state.next_backoff(&config).expect("unexpected failure");
        assert_eq!(backoff3, Duration::from_secs(4));
    }

    #[test]
    fn test_HF_CLIENT_001_rate_limit_max_backoff() {
        let config = RateLimitConfig { max_backoff: Duration::from_secs(10), ..Default::default() };
        let mut state = RateLimitState::new();

        // Exhaust retries to hit max
        for _ in 0..4 {
            state.next_backoff(&config);
        }

        let backoff = state.next_backoff(&config).expect("unexpected failure");
        assert!(backoff <= config.max_backoff);
    }

    #[test]
    fn test_HF_CLIENT_001_rate_limit_max_retries() {
        let config = RateLimitConfig { max_retries: 2, ..Default::default() };
        let mut state = RateLimitState::new();

        assert!(state.next_backoff(&config).is_some());
        assert!(state.next_backoff(&config).is_some());
        assert!(state.next_backoff(&config).is_none()); // Exhausted
    }

    #[test]
    fn test_HF_CLIENT_001_rate_limit_reset() {
        let config = RateLimitConfig::default();
        let mut state = RateLimitState::new();

        state.next_backoff(&config);
        state.next_backoff(&config);
        assert_eq!(state.retry_count, 2);

        state.reset();
        assert_eq!(state.retry_count, 0);
    }

    #[test]
    fn test_HF_CLIENT_001_rate_limit_retry_after_header() {
        let config = RateLimitConfig::default();
        let mut state = RateLimitState::new();
        state.retry_after = Some(Duration::from_secs(30));

        let backoff = state.next_backoff(&config).expect("unexpected failure");
        assert_eq!(backoff, Duration::from_secs(30));
    }

    // ========================================================================
    // HF-CLIENT-002: SafeTensors Enforcement Tests
    // ========================================================================

    #[test]
    fn test_HF_CLIENT_002_classify_safetensors_safe() {
        assert_file_safety("model.safetensors", FileSafety::Safe);
    }

    #[test]
    fn test_HF_CLIENT_002_classify_json_safe() {
        assert_file_safety("config.json", FileSafety::Safe);
    }

    #[test]
    fn test_HF_CLIENT_002_classify_gguf_safe() {
        assert_file_safety("model.gguf", FileSafety::Safe);
    }

    // SAFETY: no actual unsafe code -- tests file classification as Unsafe safety category
    #[test]
    fn test_HF_CLIENT_002_classify_bin_unsafe() {
        assert_file_safety("pytorch_model.bin", FileSafety::Unsafe);
    }

    // SAFETY: no actual unsafe code -- tests pickle file classified as Unsafe
    #[test]
    fn test_HF_CLIENT_002_classify_pickle_unsafe() {
        assert_file_safety("model.pkl", FileSafety::Unsafe);
        assert_file_safety("model.pickle", FileSafety::Unsafe);
    }

    // SAFETY: no actual unsafe code -- tests .pt file classified as Unsafe
    #[test]
    fn test_HF_CLIENT_002_classify_pt_unsafe() {
        assert_file_safety("model.pt", FileSafety::Unsafe);
        assert_file_safety("model.pth", FileSafety::Unsafe);
    }

    #[test]
    fn test_HF_CLIENT_002_check_download_safe_only_pass() {
        let files = vec!["model.safetensors", "config.json"];
        assert!(check_download_allowed(&files, SafetyPolicy::SafeOnly).is_ok());
    }

    #[test]
    fn test_HF_CLIENT_002_check_download_safe_only_fail() {
        let files = vec!["model.safetensors", "pytorch_model.bin"];
        let result = check_download_allowed(&files, SafetyPolicy::SafeOnly);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), vec!["pytorch_model.bin".to_string()]);
    }

    // SAFETY: no actual unsafe code -- tests AllowUnsafe policy permits unsafe file downloads
    #[test]
    fn test_HF_CLIENT_002_check_download_allow_unsafe() {
        let files = vec!["model.safetensors", "pytorch_model.bin"];
        assert!(check_download_allowed(&files, SafetyPolicy::AllowUnsafe).is_ok());
    }

    // ========================================================================
    // HF-CLIENT-003: Model Card Tests
    // ========================================================================

    #[test]
    fn test_HF_CLIENT_003_model_card_metadata_new() {
        let meta = ModelCardMetadata::new("my-model");
        assert_eq!(meta.model_name, "my-model");
        assert_eq!(meta.library_name, Some("paiml".to_string()));
    }

    #[test]
    fn test_HF_CLIENT_003_model_card_with_license() {
        let meta = ModelCardMetadata::new("my-model").with_license("apache-2.0");
        assert_eq!(meta.license, Some("apache-2.0".to_string()));
    }

    #[test]
    fn test_HF_CLIENT_003_model_card_with_tags() {
        let meta = make_metadata(None, &["text-classification", "rust"], &[]);
        assert_eq!(meta.tags.len(), 2);
    }

    #[test]
    fn test_HF_CLIENT_003_model_card_with_metrics() {
        let meta = make_metadata(None, &[], &[("accuracy", 0.95), ("f1", 0.92)]);
        assert_eq!(meta.metrics.len(), 2);
        assert_eq!(meta.metrics.get("accuracy"), Some(&0.95));
    }

    #[test]
    fn test_HF_CLIENT_003_generate_model_card_header() {
        let meta = make_metadata(None, &[], &[]);
        let card = generate_model_card(&meta);
        assert!(card.starts_with("---\n"));
        assert!(card.contains("# test-model"));
    }

    #[test]
    fn test_HF_CLIENT_003_generate_model_card_license() {
        assert_card_contains(Some("mit"), &[], &[], &["license: mit"]);
    }

    #[test]
    fn test_HF_CLIENT_003_generate_model_card_metrics() {
        assert_card_contains(None, &[], &[("acc", 0.9)], &["| acc |", "0.9"]);
    }

    #[test]
    fn test_HF_CLIENT_003_generate_model_card_paiml_footer() {
        assert_card_contains(None, &[], &[], &["PAIML Stack"]);
    }

    // ========================================================================
    // HF-CLIENT-004: Differential Upload Tests
    // ========================================================================

    #[test]
    fn test_HF_CLIENT_004_file_hash_new() {
        let hash = FileHash::new("abc123", 1024);
        assert_eq!(hash.sha256, "abc123");
        assert_eq!(hash.size, 1024);
    }

    #[test]
    fn test_HF_CLIENT_004_file_hash_from_content() {
        let hash = FileHash::from_content(b"hello world");
        assert!(!hash.sha256.is_empty());
        assert_eq!(hash.size, 11);
    }

    #[test]
    fn test_HF_CLIENT_004_file_hash_deterministic() {
        let hash1 = FileHash::from_content(b"test");
        let hash2 = FileHash::from_content(b"test");
        assert_eq!(hash1.sha256, hash2.sha256);
    }

    #[test]
    fn test_HF_CLIENT_004_upload_manifest_new() {
        let manifest = UploadManifest::new();
        assert!(manifest.files.is_empty());
    }

    #[test]
    fn test_HF_CLIENT_004_upload_manifest_add_file() {
        let manifest = test_manifest(&[("model.safetensors", "abc", 1000)]);
        assert_eq!(manifest.files.len(), 1);
    }

    #[test]
    fn test_HF_CLIENT_004_upload_manifest_diff_new_file() {
        let local = test_manifest(&[("new.txt", "abc", 100)]);
        let remote = test_manifest(&[]);

        let diff = local.diff(&remote);
        assert_eq!(diff, vec!["new.txt".to_string()]);
    }

    #[test]
    fn test_HF_CLIENT_004_upload_manifest_diff_changed_file() {
        let local = test_manifest(&[("file.txt", "new_hash", 100)]);
        let remote = test_manifest(&[("file.txt", "old_hash", 100)]);

        let diff = local.diff(&remote);
        assert_eq!(diff, vec!["file.txt".to_string()]);
    }

    #[test]
    fn test_HF_CLIENT_004_upload_manifest_diff_unchanged() {
        let local = test_manifest(&[("file.txt", "same", 100)]);
        let remote = test_manifest(&[("file.txt", "same", 100)]);

        let diff = local.diff(&remote);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_HF_CLIENT_004_upload_manifest_total_size() {
        let manifest = test_manifest(&[("a.txt", "a", 100), ("b.txt", "b", 200)]);

        let files = vec!["a.txt".to_string(), "b.txt".to_string()];
        assert_eq!(manifest.total_size(&files), 300);
    }

    // ========================================================================
    // HF-CLIENT-005: Secret Scanning Tests
    // ========================================================================

    #[test]
    fn test_HF_CLIENT_005_scan_env_file() {
        let files = vec![".env", "model.safetensors"];
        let secrets = scan_for_secrets(&files);
        assert_eq!(secrets.len(), 1);
        assert_eq!(secrets[0].secret_type, SecretType::EnvFile);
    }

    #[test]
    fn test_HF_CLIENT_005_scan_env_local() {
        let files = vec![".env.local"];
        let secrets = scan_for_secrets(&files);
        assert_eq!(secrets.len(), 1);
    }

    #[test]
    fn test_HF_CLIENT_005_scan_private_key() {
        let files = vec!["id_rsa", "key.pem"];
        let secrets = scan_for_secrets(&files);
        assert_eq!(secrets.len(), 2);
        assert!(secrets.iter().all(|s| s.secret_type == SecretType::PrivateKey));
    }

    #[test]
    fn test_HF_CLIENT_005_scan_credentials() {
        let files = vec!["credentials.json"];
        let secrets = scan_for_secrets(&files);
        assert_eq!(secrets.len(), 1);
        assert_eq!(secrets[0].secret_type, SecretType::Password);
    }

    #[test]
    fn test_HF_CLIENT_005_scan_no_secrets() {
        let files = vec!["model.safetensors", "config.json", "README.md"];
        let secrets = scan_for_secrets(&files);
        assert!(secrets.is_empty());
    }

    #[test]
    fn test_HF_CLIENT_005_check_push_allowed_clean() {
        let files = vec!["model.safetensors", "config.json"];
        assert!(check_push_allowed(&files).is_ok());
    }

    #[test]
    fn test_HF_CLIENT_005_check_push_blocked() {
        let files = vec!["model.safetensors", ".env"];
        let result = check_push_allowed(&files);
        assert!(result.is_err());
    }
}
