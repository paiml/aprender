//! Result caching, RAG backlinks, and cross-project search for PMAT Query.

use super::pmat_query_types::{PmatQueryOptions, PmatQueryResult};

// ============================================================================
// Result Caching
// ============================================================================

/// Get the cache directory for pmat query results.
pub(super) fn cache_dir() -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("batuta")
        .join("pmat-query")
}

/// Compute a cache key from query + project path.
pub(super) fn cache_key(query: &str, project_path: Option<&str>) -> String {
    let input = format!("{}:{}", query, project_path.unwrap_or("."));
    // FNV-1a hash
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for byte in input.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{:016x}", hash)
}

/// Check if any .rs file in the project is newer than the cache file.
fn any_source_newer_than(
    project_path: &std::path::Path,
    cache_mtime: std::time::SystemTime,
) -> bool {
    let walker = match glob::glob(&format!("{}/**/*.rs", project_path.display())) {
        Ok(w) => w,
        Err(_) => return true,
    };
    for entry in walker.flatten() {
        if let Ok(meta) = entry.metadata() {
            if let Ok(mtime) = meta.modified() {
                if mtime > cache_mtime {
                    return true;
                }
            }
        }
    }
    false
}

/// Try to load cached results, returning None if cache miss or stale.
pub(super) fn load_cached_results(
    query: &str,
    project_path: Option<&str>,
) -> Option<Vec<PmatQueryResult>> {
    let key = cache_key(query, project_path);
    let path = cache_dir().join(format!("{key}.json"));

    let cache_mtime = path.metadata().ok()?.modified().ok()?;

    let proj = std::path::Path::new(project_path.unwrap_or("."));
    if any_source_newer_than(proj, cache_mtime) {
        return None;
    }

    let data = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Save results to cache.
pub(super) fn save_cache(query: &str, project_path: Option<&str>, results: &[PmatQueryResult]) {
    let dir = cache_dir();
    let _ = std::fs::create_dir_all(&dir);
    let key = cache_key(query, project_path);
    let path = dir.join(format!("{key}.json"));
    if let Ok(json) = serde_json::to_string(results) {
        let _ = std::fs::write(path, json);
    }
}

// ============================================================================
// RAG Backlinks
// ============================================================================

/// Attach RAG documentation backlinks to PMAT results by matching file paths.
pub(super) fn attach_rag_backlinks(
    pmat_results: &mut [PmatQueryResult],
    chunk_contents: &std::collections::HashMap<String, String>,
) {
    for result in pmat_results.iter_mut() {
        // Match RAG chunk keys that contain the pmat result's file path
        // RAG keys look like: "component/src/foo.rs#chunk42"
        let file_suffix = &result.file_path;
        for key in chunk_contents.keys() {
            if key.contains(file_suffix) {
                result.rag_backlinks.push(key.clone());
            }
        }
        // Deduplicate and limit
        result.rag_backlinks.sort();
        result.rag_backlinks.dedup();
        result.rag_backlinks.truncate(3);
    }
}

// ============================================================================
// Cross-Project Search
// ============================================================================

/// Run pmat query across all discovered local PAIML projects, merging results.
/// Uses parallel execution via std::thread::scope for better performance.
pub(super) fn run_cross_project_query(
    opts: &PmatQueryOptions,
) -> anyhow::Result<Vec<PmatQueryResult>> {
    use crate::ansi_colors::Colorize;
    use crate::oracle::LocalWorkspaceOracle;

    let mut oracle_ws = LocalWorkspaceOracle::new()?;
    oracle_ws.discover_projects()?;
    let projects: Vec<_> = oracle_ws.projects().iter().collect();

    eprintln!("  {} Searching {} projects in parallel...", "[pmat-all]".dimmed(), projects.len());

    // Parallel query across all projects
    let all_chunk_results: Vec<Vec<PmatQueryResult>> = std::thread::scope(|s| {
        let handles: Vec<_> = projects
            .iter()
            .map(|(name, project)| {
                let project_opts = PmatQueryOptions {
                    query: opts.query.clone(),
                    project_path: Some(project.path.to_string_lossy().to_string()),
                    limit: opts.limit,
                    min_grade: opts.min_grade.clone(),
                    max_complexity: opts.max_complexity,
                    include_source: opts.include_source,
                };
                let name = (*name).clone();
                s.spawn(move || match super::pmat_query_display::run_pmat_query(&project_opts) {
                    Ok(mut results) => {
                        for r in &mut results {
                            r.project = Some(name.clone());
                            r.file_path = format!("{}/{}", name, r.file_path);
                        }
                        results
                    }
                    Err(_) => Vec::new(),
                })
            })
            .collect();

        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    });

    // Merge all results
    let mut all_results: Vec<PmatQueryResult> = all_chunk_results.into_iter().flatten().collect();

    // Sort by relevance descending, truncate to limit
    all_results.sort_by(|a, b| {
        b.relevance_score.partial_cmp(&a.relevance_score).unwrap_or(std::cmp::Ordering::Equal)
    });
    all_results.truncate(opts.limit);

    Ok(all_results)
}
