/// Get the HuggingFace cache directory respecting environment variables.
///
/// Priority: `$HUGGINGFACE_HUB_CACHE` > `$HF_HOME/hub` > `~/.cache/huggingface/hub`
#[must_use]
pub fn get_hf_cache_dir() -> std::path::PathBuf {
    use std::path::PathBuf;

    if let Ok(cache) = std::env::var("HUGGINGFACE_HUB_CACHE") {
        return PathBuf::from(cache);
    }
    if let Ok(home) = std::env::var("HF_HOME") {
        return PathBuf::from(home).join("hub");
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/huggingface/hub")
}

/// Split a HuggingFace repo ID into (org, repo).
///
/// # Examples
///
/// ```
/// use apr_qa_runner::split_hf_repo;
///
/// assert_eq!(split_hf_repo("Qwen/Qwen2.5-Coder-0.5B"), ("Qwen", "Qwen2.5-Coder-0.5B"));
/// assert_eq!(split_hf_repo("model-only"), ("unknown", "model-only"));
/// ```
#[must_use]
pub fn split_hf_repo(hf_repo: &str) -> (&str, &str) {
    hf_repo.split_once('/').unwrap_or(("unknown", hf_repo))
}

/// Find a snapshot containing SafeTensors files in the HuggingFace cache.
///
/// Detects both monolithic (`model.safetensors`) and sharded
/// (`model.safetensors.index.json` + `model-NNNNN-of-NNNNN.safetensors`) layouts.
fn find_hf_snapshot(
    hf_cache: &std::path::Path,
    org: &str,
    repo: &str,
) -> Option<std::path::PathBuf> {
    let hf_model_dir = hf_cache
        .join(format!("models--{org}--{repo}"))
        .join("snapshots");

    if !hf_model_dir.exists() {
        return None;
    }

    let entries = std::fs::read_dir(&hf_model_dir).ok()?;
    for entry in entries.flatten() {
        let snapshot = entry.path();
        if snapshot.is_dir() && snapshot_has_safetensors(&snapshot) {
            return Some(snapshot);
        }
    }
    None
}

/// Check whether a snapshot directory contains SafeTensors model files.
///
/// Returns `true` for monolithic (`model.safetensors`) or sharded
/// (`model.safetensors.index.json`) layouts.
fn snapshot_has_safetensors(snapshot: &std::path::Path) -> bool {
    // Monolithic: single model.safetensors file
    if snapshot.join("model.safetensors").exists() {
        return true;
    }
    // Sharded: index file indicates sharded layout
    snapshot.join("model.safetensors.index.json").exists()
}

/// Find a model in the APR cache directory.
///
/// Internal helper that checks if a model exists in the APR cache.
fn find_apr_cache(home: &std::path::Path, org: &str, repo: &str) -> Option<std::path::PathBuf> {
    // apr pull stores sharded models at ~/.apr/cache/hf/{org}/{repo}/
    let apr_cache = home.join(".apr/cache/hf").join(org).join(repo);
    if apr_cache.exists() {
        return Some(apr_cache);
    }
    // Legacy path (deprecated, kept for backwards compat with manually created caches)
    let legacy_cache = home.join(".cache/apr-models").join(org).join(repo);
    if legacy_cache.exists() {
        return Some(legacy_cache);
    }
    None
}

/// Resolve HuggingFace repo to cache with explicit cache directories.
///
/// Internal helper for testing that doesn't depend on environment variables.
fn resolve_hf_repo_with_dirs(
    hf_repo: &str,
    hf_cache: &std::path::Path,
    home: &std::path::Path,
) -> Result<std::path::PathBuf> {
    let (org, repo) = split_hf_repo(hf_repo);

    // Try HuggingFace cache first
    if let Some(snapshot) = find_hf_snapshot(hf_cache, org, repo) {
        return Ok(snapshot);
    }

    // Try APR cache
    if let Some(apr_path) = find_apr_cache(home, org, repo) {
        return Ok(apr_path);
    }

    let hf_model_dir = hf_cache
        .join(format!("models--{org}--{repo}"))
        .join("snapshots");
    let apr_cache = home.join(".apr/cache/hf").join(org).join(repo);

    Err(Error::Execution(format!(
        "Model not found in cache: {hf_repo}\nSearched:\n  - {}\n  - {}",
        hf_model_dir.display(),
        apr_cache.display()
    )))
}

/// Resolve a HuggingFace repo ID to a local cache directory.
///
/// Searches for the model in the following locations (in order):
/// 1. HuggingFace cache: `$HUGGINGFACE_HUB_CACHE` or `$HF_HOME/hub` or `~/.cache/huggingface/hub`
/// 2. APR cache: `~/.cache/apr-models/{org}/{repo}/`
///
/// Returns the snapshot directory containing `model.safetensors` (for HF cache)
/// or the APR cache directory.
///
/// # Specification
///
/// Implements HF-CACHE-001: Automatic HuggingFace Cache Resolution.
///
/// # Errors
///
/// Returns an error if the model is not found in any cache location.
/// The error message lists all searched paths for debugging.
pub fn resolve_hf_repo_to_cache(hf_repo: &str) -> Result<std::path::PathBuf> {
    use std::path::PathBuf;

    let hf_cache = get_hf_cache_dir();
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let home = PathBuf::from(home);

    resolve_hf_repo_with_dirs(hf_repo, &hf_cache, &home)
}

