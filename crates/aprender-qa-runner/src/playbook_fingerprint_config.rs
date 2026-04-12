
/// Fingerprint configuration (PMAT-201, JAX-STAT-001)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FingerprintConfig {
    /// Enable fingerprint testing
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Tensors to fingerprint ("all" or comma-separated list)
    #[serde(default = "default_fingerprint_tensors")]
    pub tensors: String,
    /// Statistics to compute
    #[serde(default = "default_fingerprint_stats")]
    pub stats: Vec<String>,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Return the default fingerprint tensor selection ("all")
fn default_fingerprint_tensors() -> String {
    "all".to_string()
}

/// Return the default fingerprint statistics to compute
fn default_fingerprint_stats() -> Vec<String> {
    vec![
        "mean".to_string(),
        "std".to_string(),
        "min".to_string(),
        "max".to_string(),
        "checksum".to_string(),
    ]
}

/// Validate stats configuration (PMAT-202)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidateStatsConfig {
    /// Enable stats validation
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Reference file for comparison
    #[serde(default)]
    pub reference: Option<String>,
    /// Role-specific tolerances
    #[serde(default)]
    pub tolerance: StatsToleranceConfig,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Per-role tolerance configuration for validate-stats
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatsToleranceConfig {
    /// Tolerance for LayerNorm tensors (strict)
    #[serde(default = "default_layernorm_tolerance")]
    pub layernorm: f64,
    /// Tolerance for embedding tensors (loose)
    #[serde(default = "default_embedding_tolerance")]
    pub embedding: f64,
    /// Tolerance for attention tensors (medium)
    #[serde(default = "default_attention_tolerance")]
    pub attention: f64,
}

/// Return the default LayerNorm tolerance (0.001)
fn default_layernorm_tolerance() -> f64 {
    0.001
}

/// Return the default embedding tolerance (0.1)
fn default_embedding_tolerance() -> f64 {
    0.1
}

/// Return the default attention tolerance (0.01)
fn default_attention_tolerance() -> f64 {
    0.01
}

/// Profile CI configuration (PMAT-192)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileCiConfig {
    /// Enable profile CI
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Warmup iterations
    #[serde(default = "default_warmup")]
    pub warmup: usize,
    /// Measurement iterations
    #[serde(default = "default_measure")]
    pub measure: usize,
    /// Formats to profile (default: all available)
    #[serde(default = "default_profile_formats")]
    pub formats: Vec<String>,
    /// Backends to profile (default: [cpu, gpu])
    #[serde(default = "default_profile_backends")]
    pub backends: Vec<String>,
    /// Assertions to verify
    #[serde(default)]
    pub assertions: ProfileCiAssertions,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Return the default profile formats (gguf, apr, safetensors)
fn default_profile_formats() -> Vec<String> {
    vec![
        "gguf".to_string(),
        "apr".to_string(),
        "safetensors".to_string(),
    ]
}

/// Return the default profile backends (cpu, gpu)
fn default_profile_backends() -> Vec<String> {
    vec!["cpu".to_string(), "gpu".to_string()]
}

/// Return the default warmup iteration count (3)
fn default_warmup() -> usize {
    3
}

/// Return the default measurement iteration count (10)
fn default_measure() -> usize {
    10
}

/// Profile CI assertions
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileCiAssertions {
    /// Minimum throughput in tok/s (legacy, applies to all)
    #[serde(default)]
    pub min_throughput: Option<f64>,
    /// Minimum CPU throughput in tok/s
    #[serde(default)]
    pub min_throughput_cpu: Option<f64>,
    /// Minimum GPU throughput in tok/s
    #[serde(default)]
    pub min_throughput_gpu: Option<f64>,
    /// Maximum p99 latency in ms
    #[serde(default)]
    pub max_p99_ms: Option<f64>,
    /// Maximum p50 latency in ms
    #[serde(default)]
    pub max_p50_ms: Option<f64>,
}

impl ProfileCiAssertions {
    /// Get minimum throughput for a given backend
    #[must_use]
    pub fn min_throughput_for(&self, backend: &str) -> Option<f64> {
        match backend {
            "cpu" => self.min_throughput_cpu.or(self.min_throughput),
            "gpu" => self.min_throughput_gpu.or(self.min_throughput),
            _ => self.min_throughput,
        }
    }
}

/// Trace payload configuration (APR-TRACE-001)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracePayloadConfig {
    /// Enable trace payload
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Prompt for trace
    #[serde(default)]
    pub prompt: Option<String>,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Ollama parity configuration (GH-6/AC-2)
///
/// Tests that APR inference output matches ollama for the same model/quant.
/// Catches format-specific regressions by comparing against an independent runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OllamaParityConfig {
    /// Enable ollama parity testing
    #[serde(default, deserialize_with = "deserialize_bool_or_string")]
    pub enabled: bool,
    /// Ollama model tag (e.g., "qwen2.5-coder:7b-instruct-q4_k_m")
    #[serde(default)]
    pub model_tag: Option<String>,
    /// Quantizations to test (each maps to an ollama tag suffix)
    #[serde(default = "default_ollama_quantizations")]
    pub quantizations: Vec<String>,
    /// Prompts to test parity on
    #[serde(default = "default_ollama_prompts")]
    pub prompts: Vec<String>,
    /// Inference temperature (0.0 for deterministic)
    #[serde(default)]
    pub temperature: f64,
    /// Minimum performance ratio (APR tok/s / ollama tok/s)
    #[serde(default = "default_min_perf_ratio")]
    pub min_perf_ratio: f64,
    /// Gates to verify
    #[serde(default)]
    pub gates: Vec<String>,
}

/// Return the default ollama quantization list (q4_k_m)
fn default_ollama_quantizations() -> Vec<String> {
    vec!["q4_k_m".to_string()]
}

/// Return the default ollama parity test prompts
fn default_ollama_prompts() -> Vec<String> {
    vec!["What is 2+2?".to_string()]
}

/// Return the default minimum performance ratio (0.8)
fn default_min_perf_ratio() -> f64 {
    0.8
}

// ── Playbook Integrity Lock (§3.1) ──────────────────────────────────────

/// A single entry in the playbook lock file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookLockEntry {
    /// SHA-256 hash of the playbook file
    pub sha256: String,
    /// Fields that are locked (changing them requires re-approval)
    pub locked_fields: Vec<String>,
}

/// Lock file mapping playbook names to their integrity entries
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PlaybookLockFile {
    /// Map of playbook name → lock entry
    pub entries: HashMap<String, PlaybookLockEntry>,
}

/// Compute SHA-256 hash of a playbook file
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn compute_playbook_hash(path: impl AsRef<Path>) -> Result<String> {
    use sha2::{Digest, Sha256};
    let content = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&content);
    Ok(format!("{:x}", hasher.finalize()))
}

/// Load a lock file from YAML
///
/// # Errors
///
/// Returns an error if the file cannot be read or parsed.
pub fn load_lock_file(path: impl AsRef<Path>) -> Result<PlaybookLockFile> {
    let content = std::fs::read_to_string(path)?;
    serde_yaml::from_str(&content).map_err(Error::from)
}

/// Save a lock file to YAML
///
/// # Errors
///
/// Returns an error if serialization or writing fails.
pub fn save_lock_file(lock: &PlaybookLockFile, path: impl AsRef<Path>) -> Result<()> {
    let yaml = serde_yaml::to_string(lock).map_err(Error::from)?;
    std::fs::write(path, yaml)?;
    Ok(())
}

/// Verify a playbook's integrity against the lock file
///
/// # Errors
///
/// Returns an error if the hash does not match or if file operations fail.
pub fn verify_playbook_integrity(
    playbook_path: impl AsRef<Path>,
    lock_file: &PlaybookLockFile,
    name: &str,
) -> Result<()> {
    let entry = lock_file
        .entries
        .get(name)
        .ok_or_else(|| Error::Execution(format!("Playbook '{name}' not found in lock file")))?;

    let current_hash = compute_playbook_hash(&playbook_path)?;
    if current_hash != entry.sha256 {
        return Err(Error::Execution(format!(
            "Integrity check failed for '{name}': expected {}, got {current_hash}",
            entry.sha256
        )));
    }

    Ok(())
}

/// Generate a lock entry for a playbook file
///
/// # Errors
///
/// Returns an error if the file cannot be read.
pub fn generate_lock_entry(path: impl AsRef<Path>) -> Result<(String, PlaybookLockEntry)> {
    let path_ref = path.as_ref();
    let name = path_ref
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    // Strip common suffixes like ".playbook"
    let name = name.strip_suffix(".playbook").unwrap_or(&name).to_string();

    let sha256 = compute_playbook_hash(path_ref)?;

    let entry = PlaybookLockEntry {
        sha256,
        locked_fields: vec![
            "model.hf_repo".to_string(),
            "model.formats".to_string(),
            "test_matrix".to_string(),
            "falsification_gates".to_string(),
        ],
    };

    Ok((name, entry))
}

// ── Skip Mechanism (§3.3) ──────────────────────────────────────────────

/// Reason for skipping a test
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkipReason {
    /// Format or backend being skipped
    pub format_or_backend: String,
    /// Why it's skipped
    pub reason: String,
    /// Tracking issue (e.g., "GH-123")
    pub tracking_issue: Option<String>,
}

/// Type of skip
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SkipType {
    /// Explicitly declared via .skip file
    Explicit,
    /// Implicitly missing from the format list
    Implicit,
}

/// Find skip files for a given playbook
///
/// Looks for `<playbook_dir>/<name>.skip.yaml` files.
#[must_use]
pub fn find_skip_files(playbook_dir: &Path, name: &str) -> Vec<SkipReason> {
    let skip_path = playbook_dir.join(format!("{name}.skip.yaml"));
    if !skip_path.exists() {
        return Vec::new();
    }

    let Ok(content) = std::fs::read_to_string(&skip_path) else {
        eprintln!(
            "[WARN] Cannot read skip file: {}",
            skip_path.display()
        );
        return Vec::new();
    };

    match serde_yaml::from_str(&content) {
        Ok(reasons) => reasons,
        Err(e) => {
            eprintln!(
                "[WARN] Invalid skip file {}: {e}",
                skip_path.display()
            );
            Vec::new()
        }
    }
}

/// Detect implicit skips by comparing playbook formats against all known formats
#[must_use]
pub fn detect_implicit_skips(
    playbook: &Playbook,
    all_formats: &[Format],
    skip_files: &[SkipReason],
) -> Vec<String> {
    let mut implicit = Vec::new();
    let explicit_formats: Vec<&str> = skip_files
        .iter()
        .map(|s| s.format_or_backend.as_str())
        .collect();

    for format in all_formats {
        let format_str = format!("{format:?}").to_lowercase();
        if !playbook.model.formats.contains(format)
            && !explicit_formats.contains(&format_str.as_str())
        {
            implicit.push(format_str);
        }
    }

    implicit
}

#[cfg(test)]
#[path = "playbook_tests.rs"]
mod tests;
