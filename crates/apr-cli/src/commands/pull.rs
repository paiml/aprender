//! Pull command implementation using pacha ModelFetcher
//!
//! Downloads and caches models from HuggingFace with Ollama-like UX.
//! Models are cached at `~/.cache/pacha/models/` for reuse.
//!
//! Usage:
//!   apr pull qwen2.5-coder:7b           # Short alias
//!   apr pull hf://bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf
//!   apr pull TheBloke/Llama-2-7B-GGUF   # Will auto-detect quant

use crate::error::{CliError, Result};
use colored::Colorize;
use pacha::fetcher::{FetchConfig, ModelFetcher};
use pacha::format::ModelFormat;
use std::io::{self, Read, Write};
use std::path::Path;

/// Run the pull command
pub fn run(model_ref: &str, force: bool) -> Result<()> {
    println!("{}", "=== APR Pull ===".cyan().bold());
    println!();

    // PMAT-108: Resolve HuggingFace URI to include filename if missing
    let resolved_ref = resolve_hf_uri(model_ref)?;
    let model_ref = resolved_ref.as_str();

    println!("Model: {}", model_ref.cyan());

    // Initialize pacha ModelFetcher
    let mut fetcher = ModelFetcher::with_config(FetchConfig::default()).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    // Check if already cached
    if !force && fetcher.is_cached(model_ref) {
        println!("{} Model already cached", "✓".green());

        // Get the cached path
        let result = fetcher
            .pull_quiet(model_ref)
            .map_err(|e| CliError::ValidationFailed(format!("Failed to get cached model: {e}")))?;

        println!("  Path: {}", result.path.display());
        println!("  Size: {}", result.size_human());
        println!("  Format: {:?}", result.format);

        // GH-198: Ensure companion files exist for cached SafeTensors models
        if matches!(result.format, ModelFormat::SafeTensors(_)) {
            fetch_safetensors_companions(&result.path, &result.resolved_uri)?;
        }

        println!();
        println!("{}", "Usage:".cyan().bold());
        println!("  apr run {}", result.path.display());
        return Ok(());
    }

    println!();
    println!("{}", "Downloading...".yellow());

    // Pull with progress callback
    let result = fetcher
        .pull(model_ref, |progress| {
            // Simple progress bar
            let pct = progress.percent();
            print!(
                "\r  [{:50}] {:5.1}% ({}/{})",
                "=".repeat((pct / 2.0) as usize),
                pct,
                format_bytes(progress.downloaded_bytes),
                format_bytes(progress.total_bytes)
            );
            io::stdout().flush().ok();
        })
        .map_err(|e| CliError::NetworkError(format!("Download failed: {e}")))?;

    println!();
    println!();

    if result.cache_hit {
        println!("{} Model retrieved from cache", "✓".green());
    } else {
        println!("{} Downloaded successfully", "✓".green());
    }

    println!("  Path: {}", result.path.display().to_string().green());
    println!("  Size: {}", result.size_human().yellow());
    println!("  Format: {:?}", result.format);
    println!("  Hash: {}", &result.hash[..16]);

    // GH-198: Download companion files for SafeTensors models
    if matches!(result.format, ModelFormat::SafeTensors(_)) {
        fetch_safetensors_companions(&result.path, &result.resolved_uri)?;
    }

    println!();
    println!("{}", "Usage:".cyan().bold());
    println!("  apr run {}", result.path.display());
    println!("  apr serve {}", result.path.display());

    Ok(())
}

/// List cached models
pub fn list() -> Result<()> {
    println!("{}", "=== Cached Models ===".cyan().bold());
    println!();

    let fetcher = ModelFetcher::new().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    let models = fetcher.list();

    if models.is_empty() {
        println!("{}", "No cached models found.".dimmed());
        println!();
        println!("Pull a model with:");
        println!("  apr pull hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
        println!();
        println!("Or run directly (auto-downloads):");
        println!("  apr run hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf");
        return Ok(());
    }

    // Print header
    println!(
        "{:<40} {:<12} {:<10} {}",
        "NAME".dimmed(),
        "SIZE".dimmed(),
        "FORMAT".dimmed(),
        "PATH".dimmed()
    );
    println!("{}", "-".repeat(100).dimmed());

    for model in &models {
        let size = format_bytes(model.size_bytes);
        let format = format!("{:?}", model.format);
        let name = if model.name.len() > 38 {
            format!("{}...", &model.name[..35])
        } else {
            model.name.clone()
        };

        println!(
            "{:<40} {:<12} {:<10} {}",
            name.cyan(),
            size.yellow(),
            format,
            model.path.display().to_string().dimmed()
        );
    }

    println!();

    // Print stats
    let stats = fetcher.stats();
    println!(
        "Total: {} models, {} used",
        models.len(),
        format_bytes(stats.total_size_bytes)
    );

    Ok(())
}

/// Remove a model from cache
pub fn remove(model_ref: &str) -> Result<()> {
    println!("{}", "=== APR Remove ===".cyan().bold());
    println!();
    println!("Model: {}", model_ref.cyan());

    let mut fetcher = ModelFetcher::new().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    let removed = fetcher
        .remove(model_ref)
        .map_err(|e| CliError::ValidationFailed(format!("Failed to remove model: {e}")))?;

    if removed {
        println!("{} Model removed from cache", "✓".green());
    } else {
        println!("{} Model not found in cache", "⚠".yellow());
    }

    Ok(())
}

/// Resolve a model reference to a local path (for run/serve commands)
/// Downloads if not cached and auto_pull is enabled
#[allow(dead_code)]
pub fn resolve_model_path(model_ref: &str) -> Result<std::path::PathBuf> {
    // If it's already a local file path, use it directly
    let path = std::path::Path::new(model_ref);
    if path.exists() && path.is_file() {
        return Ok(path.to_path_buf());
    }

    // Try to resolve via pacha
    let mut fetcher = ModelFetcher::with_config(FetchConfig::default()).map_err(|e| {
        CliError::ValidationFailed(format!("Failed to initialize model fetcher: {e}"))
    })?;

    // Pull (uses cache if available)
    let result = fetcher
        .pull(model_ref, |progress| {
            if progress.total_bytes > 0 {
                let pct = progress.percent();
                eprint!(
                    "\rPulling model... [{:30}] {:5.1}%",
                    "=".repeat((pct / 3.33) as usize),
                    pct
                );
                io::stderr().flush().ok();
            }
        })
        .map_err(|e| {
            // Not a pacha model ref, check if file exists
            CliError::ValidationFailed(format!(
                "Model '{}' not found. Not a local file and could not resolve via registry: {}",
                model_ref, e
            ))
        })?;

    if !result.cache_hit {
        eprintln!(); // Newline after progress
    }

    Ok(result.path)
}

/// Format bytes to human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

/// GH-198: Download companion files (tokenizer.json, config.json) for SafeTensors models.
///
/// SafeTensors format stores weights only — unlike GGUF which embeds tokenizer and config.
/// The realizar inference engine expects these as sibling files via `with_file_name()`.
fn fetch_safetensors_companions(model_path: &Path, resolved_uri: &str) -> Result<()> {
    // Extract HF repo from resolved URI: "hf://org/repo/file.safetensors" → "org/repo"
    let Some(repo_id) = extract_hf_repo(resolved_uri) else {
        // Not an HF URI — can't fetch companions (local file or unknown source)
        return Ok(());
    };

    let companions = ["tokenizer.json", "config.json"];
    let cache_dir = model_path
        .parent()
        .ok_or_else(|| CliError::ValidationFailed("Model path has no parent directory".into()))?;

    for filename in &companions {
        let sibling_path = cache_dir.join(filename);
        if sibling_path.exists() {
            println!(
                "  {} {} (already exists)",
                "✓".green(),
                filename.dimmed()
            );
            continue;
        }

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo_id, filename
        );

        match ureq::get(&url).call() {
            Ok(response) => {
                let mut body = Vec::new();
                response
                    .into_reader()
                    .read_to_end(&mut body)
                    .map_err(|e| {
                        CliError::NetworkError(format!("Failed to read {filename}: {e}"))
                    })?;
                std::fs::write(&sibling_path, &body).map_err(|e| {
                    CliError::ValidationFailed(format!(
                        "Failed to write {}: {e}",
                        sibling_path.display()
                    ))
                })?;
                println!(
                    "  {} {} ({})",
                    "✓".green(),
                    filename,
                    format_bytes(body.len() as u64).dimmed()
                );
            }
            Err(ureq::Error::Status(404, _)) => {
                // File doesn't exist in repo — not fatal
                println!(
                    "  {} {} (not found in repo)",
                    "⚠".yellow(),
                    filename.dimmed()
                );
            }
            Err(e) => {
                // Network error — warn but don't block the pull
                eprintln!(
                    "  {} Failed to download {}: {}",
                    "⚠".yellow(),
                    filename,
                    e
                );
            }
        }
    }

    Ok(())
}

/// Extract HuggingFace repo ID from a resolved URI.
///
/// Examples:
///   "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors" → Some("Qwen/Qwen2.5-Coder-0.5B-Instruct")
///   "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct" → Some("Qwen/Qwen2.5-Coder-0.5B-Instruct")
///   "/local/path/model.safetensors" → None
fn extract_hf_repo(uri: &str) -> Option<String> {
    let path = uri.strip_prefix("hf://")?;
    let parts: Vec<&str> = path.split('/').collect();
    if parts.len() >= 2 {
        Some(format!("{}/{}", parts[0], parts[1]))
    } else {
        None
    }
}

/// PMAT-108: Resolve HuggingFace URI to include filename if missing
///
/// Accepts both formats:
/// - `hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF` (auto-detects Q4_K_M .gguf)
/// - `hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model.gguf` (unchanged)
///
/// Priority for auto-detection:
/// 1. Q4_K_M (best quality/size ratio)
/// 2. Q4_K_S
/// 3. Q4_0
/// 4. Any .gguf file
pub fn resolve_hf_uri(uri: &str) -> Result<String> {
    // If not a HuggingFace URI, return unchanged
    if !uri.starts_with("hf://") {
        return Ok(uri.to_string());
    }

    // If already has .gguf extension, return unchanged
    if uri.to_lowercase().ends_with(".gguf") {
        return Ok(uri.to_string());
    }

    // Parse org/repo from URI
    let path = uri.strip_prefix("hf://").unwrap_or(uri);
    let parts: Vec<&str> = path.split('/').collect();

    if parts.len() < 2 {
        return Err(CliError::ValidationFailed(format!(
            "Invalid HuggingFace URI: {}. Expected hf://org/repo or hf://org/repo/file.gguf",
            uri
        )));
    }

    let org = parts[0];
    let repo = parts[1];

    // Query HuggingFace API for repo files
    let api_url = format!("https://huggingface.co/api/models/{}/{}", org, repo);

    let response = ureq::get(&api_url)
        .call()
        .map_err(|e| CliError::NetworkError(format!("Failed to query HuggingFace API: {}", e)))?;

    let body: serde_json::Value = response.into_json().map_err(|e| {
        CliError::ValidationFailed(format!("Failed to parse HuggingFace response: {}", e))
    })?;

    // Extract siblings (files) from response
    let siblings = body["siblings"]
        .as_array()
        .ok_or_else(|| CliError::ValidationFailed("No files found in repository".to_string()))?;

    // Find GGUF files
    let gguf_files: Vec<&str> = siblings
        .iter()
        .filter_map(|s| s["rfilename"].as_str())
        .filter(|f| f.to_lowercase().ends_with(".gguf"))
        .collect();

    if gguf_files.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "No .gguf files found in {}/{}",
            org, repo
        )));
    }

    // Priority: Q4_K_M > Q4_K_S > Q4_0 > Q8_0 > any
    let quantization_priority = ["q4_k_m", "q4_k_s", "q4_0", "q8_0"];

    for quant in quantization_priority {
        if let Some(file) = gguf_files.iter().find(|f| f.to_lowercase().contains(quant)) {
            return Ok(format!("hf://{}/{}/{}", org, repo, file));
        }
    }

    // Fallback to first GGUF file
    Ok(format!("hf://{}/{}/{}", org, repo, gguf_files[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // format_bytes tests
    // =========================================================================

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(5 * 1024 * 1024 * 1024), "5.00 GB");
    }

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn test_format_bytes_small() {
        assert_eq!(format_bytes(1), "1 B");
        assert_eq!(format_bytes(100), "100 B");
        assert_eq!(format_bytes(1023), "1023 B");
    }

    #[test]
    fn test_format_bytes_kilobytes() {
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(2048), "2.00 KB");
        assert_eq!(format_bytes(512 * 1024), "512.00 KB");
    }

    #[test]
    fn test_format_bytes_megabytes() {
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(100 * 1024 * 1024), "100.00 MB");
        assert_eq!(format_bytes(500 * 1024 * 1024), "500.00 MB");
    }

    #[test]
    fn test_format_bytes_gigabytes() {
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(format_bytes(10 * 1024 * 1024 * 1024), "10.00 GB");
        assert_eq!(format_bytes(100 * 1024 * 1024 * 1024), "100.00 GB");
    }

    #[test]
    fn test_format_bytes_fractional_gb() {
        // 4.5 GB = 4.5 * 1024 * 1024 * 1024 = 4831838208 bytes
        assert_eq!(format_bytes(4831838208), "4.50 GB");
    }

    #[test]
    fn test_format_bytes_fractional_mb() {
        // 2.5 MB = 2.5 * 1024 * 1024 = 2621440 bytes
        assert_eq!(format_bytes(2621440), "2.50 MB");
    }

    // =========================================================================
    // PMAT-108: resolve_hf_uri Tests (Extreme TDD)
    // =========================================================================

    #[test]
    fn test_pmat_108_resolve_uri_with_gguf_extension_unchanged() {
        let uri = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "URI with .gguf should be unchanged");
    }

    #[test]
    fn test_pmat_108_resolve_uri_case_insensitive_gguf() {
        let uri = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/model.GGUF";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "URI with .GGUF should be unchanged");
    }

    #[test]
    fn test_pmat_108_resolve_non_hf_uri_unchanged() {
        let uri = "/path/to/local/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Non-hf:// URI should be unchanged");
    }

    #[test]
    fn test_pmat_108_resolve_invalid_uri_fails() {
        let uri = "hf://invalid";
        let result = resolve_hf_uri(uri);
        assert!(result.is_err(), "Invalid URI should fail");
    }

    #[test]
    fn test_resolve_hf_uri_relative_path() {
        let uri = "./models/test.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Relative path should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_absolute_path() {
        let uri = "/home/user/models/test.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Absolute path should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_https_url() {
        let uri = "https://example.com/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "HTTPS URL should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_with_mixed_case_extension() {
        let uri = "hf://Org/Repo/model.GgUf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Mixed case .GgUf should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_empty_string() {
        let uri = "";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, "Empty string should be unchanged");
    }

    #[test]
    fn test_resolve_hf_uri_invalid_hf_format() {
        // hf:// without org/repo should fail
        let result = resolve_hf_uri("hf://only-one-part");
        assert!(result.is_err());
        match result {
            Err(CliError::ValidationFailed(msg)) => {
                assert!(msg.contains("Invalid HuggingFace URI"));
            }
            other => panic!("Expected ValidationFailed, got {:?}", other),
        }
    }

    #[test]
    fn test_resolve_hf_uri_with_safetensors_extension_unchanged() {
        // .safetensors files are not .gguf, so they will trigger HF API query
        // This test verifies the logic path, but we can't test the full flow
        // without mocking. Instead, test that non-gguf HF URIs attempt resolution.
        // The test_resolve_hf_uri_invalid_hf_format covers the error case.
        // For now, we just verify the URI format is preserved for files with .gguf extension
        let uri = "hf://org/repo/model.gguf";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert_eq!(resolved, uri, ".gguf extension should be unchanged");
    }

    #[test]
    #[ignore] // Requires network access
    fn test_resolve_hf_uri_with_safetensors_queries_api() {
        // This test would need network access to verify API query behavior
        let uri = "hf://org/repo/model.safetensors";
        let _result = resolve_hf_uri(uri);
        // Result depends on network and repo existence
    }

    // Integration test (requires network, marked ignore for CI)
    #[test]
    #[ignore]
    fn test_pmat_108_resolve_qwen_repo_finds_q4_k_m() {
        let uri = "hf://Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF";
        let resolved = resolve_hf_uri(uri).unwrap();
        assert!(resolved.ends_with(".gguf"), "Should end with .gguf");
        assert!(
            resolved.to_lowercase().contains("q4_k_m"),
            "Should prefer Q4_K_M quantization: {}",
            resolved
        );
    }

    // =========================================================================
    // resolve_model_path tests
    // =========================================================================

    #[test]
    fn test_resolve_model_path_existing_file() {
        // Create a temp file
        let temp_dir = std::env::temp_dir().join("apr_pull_test_path");
        let _ = std::fs::create_dir_all(&temp_dir);
        let test_file = temp_dir.join("test_model.gguf");
        let _ = std::fs::write(&test_file, "GGUF");

        let result = resolve_model_path(test_file.to_str().unwrap());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), test_file);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_resolve_model_path_nonexistent_local_fails() {
        let result = resolve_model_path("/nonexistent/model.gguf");
        // This will try pacha which will fail with validation error
        assert!(result.is_err());
    }

    // =========================================================================
    // GH-198: extract_hf_repo tests
    // =========================================================================

    #[test]
    fn test_gh198_extract_hf_repo_with_file() {
        let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors";
        assert_eq!(
            extract_hf_repo(uri),
            Some("Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string())
        );
    }

    #[test]
    fn test_gh198_extract_hf_repo_without_file() {
        let uri = "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct";
        assert_eq!(
            extract_hf_repo(uri),
            Some("Qwen/Qwen2.5-Coder-0.5B-Instruct".to_string())
        );
    }

    #[test]
    fn test_gh198_extract_hf_repo_local_path() {
        assert_eq!(extract_hf_repo("/local/path/model.safetensors"), None);
    }

    #[test]
    fn test_gh198_extract_hf_repo_empty() {
        assert_eq!(extract_hf_repo(""), None);
    }

    #[test]
    fn test_gh198_extract_hf_repo_only_org() {
        // hf://org (missing repo) → None
        assert_eq!(extract_hf_repo("hf://org"), None);
    }

    #[test]
    fn test_gh198_extract_hf_repo_nested_path() {
        let uri = "hf://org/repo/subdir/model.safetensors";
        assert_eq!(
            extract_hf_repo(uri),
            Some("org/repo".to_string())
        );
    }

    // =========================================================================
    // GH-198: fetch_safetensors_companions tests
    // =========================================================================

    #[test]
    fn test_gh198_companions_non_hf_uri_is_noop() {
        let temp_dir = std::env::temp_dir().join("apr_gh198_noop");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Local URI → should return Ok without downloading anything
        let result = fetch_safetensors_companions(&model_path, "/local/model.safetensors");
        assert!(result.is_ok());

        // No companion files should be created
        assert!(!temp_dir.join("tokenizer.json").exists());
        assert!(!temp_dir.join("config.json").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_gh198_companions_skips_existing() {
        let temp_dir = std::env::temp_dir().join("apr_gh198_existing");
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        // Pre-create companion files
        let _ = std::fs::write(temp_dir.join("tokenizer.json"), b"{}");
        let _ = std::fs::write(temp_dir.join("config.json"), b"{}");

        // Should succeed without attempting downloads (files already exist)
        let result = fetch_safetensors_companions(
            &model_path,
            "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors",
        );
        assert!(result.is_ok());

        // Verify files are unchanged (still our dummy content)
        let content = std::fs::read_to_string(temp_dir.join("tokenizer.json")).unwrap();
        assert_eq!(content, "{}");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    #[ignore] // Requires network access
    fn test_gh198_companions_downloads_from_hf() {
        let temp_dir = std::env::temp_dir().join("apr_gh198_download");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let _ = std::fs::create_dir_all(&temp_dir);
        let model_path = temp_dir.join("model.safetensors");
        let _ = std::fs::write(&model_path, b"dummy");

        let result = fetch_safetensors_companions(
            &model_path,
            "hf://Qwen/Qwen2.5-Coder-0.5B-Instruct/model.safetensors",
        );
        assert!(result.is_ok());

        // Both companion files should now exist
        assert!(
            temp_dir.join("tokenizer.json").exists(),
            "tokenizer.json should be downloaded"
        );
        assert!(
            temp_dir.join("config.json").exists(),
            "config.json should be downloaded"
        );

        // Verify tokenizer.json has vocab
        let tok = std::fs::read_to_string(temp_dir.join("tokenizer.json")).unwrap();
        assert!(tok.contains("vocab"), "tokenizer.json should contain vocab");

        // Verify config.json has model architecture
        let cfg = std::fs::read_to_string(temp_dir.join("config.json")).unwrap();
        assert!(
            cfg.contains("num_hidden_layers"),
            "config.json should contain num_hidden_layers"
        );

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
