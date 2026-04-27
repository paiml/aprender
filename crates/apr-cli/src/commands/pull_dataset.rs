// Dataset puller for `apr pull dataset <REPO> --include <GLOB> --output <DIR>`
//
// Per SHIP-TWO-001 §26.8 and contract `apr-cli-pull-dataset-v1.yaml`, this
// file extends `apr pull` with the dataset asset-type. Listed via the
// HuggingFace `/api/datasets/{repo}/tree/{rev}` endpoint, filtered by
// glob patterns, and streamed via the existing `download_file_with_progress`
// helper. License-allowlist row filtering is deferred to P1.1.5 (requires
// parquet round-trip).
//
// NOTE: This file is `include!()`-ed by pull.rs; do NOT add `use` for
// crate::error::{CliError, Result}, colored::Colorize, std::path::Path —
// pull.rs already imports them. Add only NEW imports here.

use std::path::PathBuf;

/// Run `apr pull dataset <REPO>` per FALSIFY-APR-PULL-DATASET-001..006.
///
/// Lists files in the dataset repo via HF API, filters by `--include` globs
/// (fail-fast on no-match), then streams each matched file to `<output>/<path>`
/// reusing `download_file_with_progress`.
pub fn run_dataset(
    repo: &str,
    include: &[String],
    revision: Option<&str>,
    output: Option<&Path>,
) -> Result<()> {
    println!("{}", "=== APR Pull Dataset ===".cyan().bold());
    println!("Repo:    {}", repo.cyan());
    let rev = revision.unwrap_or("main");
    println!("Rev:     {}", rev);

    // Resolve output directory
    let out_dir: PathBuf = match output {
        Some(p) => p.to_path_buf(),
        None => default_dataset_cache_dir(repo)?,
    };
    println!("Output:  {}", out_dir.display());
    if !include.is_empty() {
        println!("Include: {include:?}");
    }
    println!();

    // 1. List all files in the repo
    let all_files = list_dataset_repo_files(repo, rev)?;
    println!("{} {} files in repo", "✓".green(), all_files.len());

    // 2. Filter by --include globs (or pass-through if empty)
    let matched = filter_files_by_globs(&all_files, include)?;
    println!("{} {} files match include globs", "✓".green(), matched.len());

    // 3. Fail-fast on no-match (per FALSIFY-APR-PULL-DATASET-003)
    if !include.is_empty() && matched.is_empty() {
        return Err(CliError::ValidationFailed(format!(
            "apr pull dataset: no files in {repo} matched any --include pattern: {include:?}"
        )));
    }

    // 4. Download each matched file
    std::fs::create_dir_all(&out_dir)?;
    for (i, path) in matched.iter().enumerate() {
        let url = format!("https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}");
        let dest = out_dir.join(path);
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        println!(
            "[{}/{}] {} -> {}",
            i + 1,
            matched.len(),
            path.cyan(),
            dest.display()
        );
        let _checksum = download_file_with_progress(&url, &dest)?;
        println!();
    }

    println!("{} Pulled {} files to {}", "✓".green(), matched.len(), out_dir.display());
    Ok(())
}

/// Default cache dir: `~/.cache/aprender/datasets/<repo>/` per contract invariant.
fn default_dataset_cache_dir(repo: &str) -> Result<PathBuf> {
    let base = if let Ok(cache_home) = std::env::var("XDG_CACHE_HOME") {
        PathBuf::from(cache_home)
    } else {
        dirs::home_dir()
            .ok_or_else(|| CliError::ValidationFailed("Cannot find home directory".to_string()))?
            .join(".cache")
    };
    Ok(base.join("aprender").join("datasets").join(repo))
}

/// HF API: list all files in a dataset repo.
fn list_dataset_repo_files(repo: &str, revision: &str) -> Result<Vec<String>> {
    let url = format!("https://huggingface.co/api/datasets/{repo}/tree/{revision}?recursive=1");
    let response = hf_get(&url).call().map_err(|e| match &e {
        ureq::Error::Status(404, _) => CliError::HttpNotFound(format!(
            "Dataset {repo} not found at revision {revision}"
        )),
        ureq::Error::Status(401, _) => CliError::NetworkError(format_gated_model_error(&url)),
        _ => CliError::NetworkError(format!("Dataset listing failed: {e}")),
    })?;
    let body = response
        .into_string()
        .map_err(|e| CliError::NetworkError(format!("Read body: {e}")))?;
    let v: serde_json::Value = serde_json::from_str(&body)
        .map_err(|e| CliError::ValidationFailed(format!("HF API JSON parse: {e}")))?;
    let mut paths = Vec::new();
    if let Some(items) = v.as_array() {
        for it in items {
            if let Some(t) = it.get("type").and_then(|x| x.as_str()) {
                if t == "file" {
                    if let Some(p) = it.get("path").and_then(|x| x.as_str()) {
                        paths.push(p.to_string());
                    }
                }
            }
        }
    }
    Ok(paths)
}

/// Apply `--include` globs (union semantics). Empty includes = pass-through.
fn filter_files_by_globs(all: &[String], include: &[String]) -> Result<Vec<String>> {
    if include.is_empty() {
        return Ok(all.to_vec());
    }
    let patterns: Vec<glob::Pattern> = include
        .iter()
        .map(|s| {
            glob::Pattern::new(s).map_err(|e| {
                CliError::ValidationFailed(format!("Invalid --include glob '{s}': {e}"))
            })
        })
        .collect::<Result<_>>()?;
    let matched: Vec<String> = all
        .iter()
        .filter(|f| patterns.iter().any(|p| p.matches(f)))
        .cloned()
        .collect();
    Ok(matched)
}

#[cfg(test)]
mod pull_dataset_tests {
    use super::*;

    #[test]
    fn test_filter_files_empty_include_passthrough() {
        let all = vec!["a.parquet".to_string(), "b.json".to_string()];
        let r = filter_files_by_globs(&all, &[]).unwrap();
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn test_filter_files_glob_matches_subset() {
        let all = vec![
            "data/train-00000.parquet".to_string(),
            "data/train-00001.parquet".to_string(),
            "data/test-00000.parquet".to_string(),
            "README.md".to_string(),
        ];
        let include = vec!["data/train-*.parquet".to_string()];
        let r = filter_files_by_globs(&all, &include).unwrap();
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|f| f.starts_with("data/train-")));
    }

    #[test]
    fn test_filter_files_no_match_returns_empty() {
        let all = vec!["data/train.parquet".to_string()];
        let include = vec!["no/such/file/*".to_string()];
        let r = filter_files_by_globs(&all, &include).unwrap();
        assert_eq!(r.len(), 0); // caller fails fast on empty
    }

    #[test]
    fn test_filter_files_multi_include_unions() {
        let all = vec![
            "data/train.parquet".to_string(),
            "data/test.parquet".to_string(),
            "README.md".to_string(),
        ];
        let include = vec!["*.parquet".to_string(), "*.md".to_string()];
        let r = filter_files_by_globs(&all, &include).unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn test_filter_files_invalid_glob_errors() {
        let all = vec!["a.parquet".to_string()];
        let include = vec!["[invalid".to_string()];
        let r = filter_files_by_globs(&all, &include);
        assert!(r.is_err());
    }
}
