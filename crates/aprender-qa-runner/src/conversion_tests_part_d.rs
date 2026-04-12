// ── HF Cache Resolution Tests (conversion_hf_cache.rs) ─────────────────────
//
// Tests for find_hf_snapshot, find_apr_cache, resolve_hf_repo_with_dirs,
// split_hf_repo, and get_hf_cache_dir.

/// Verify split_hf_repo splits org/repo correctly
#[test]
fn test_split_hf_repo_standard() {
    let (org, repo) = split_hf_repo("Qwen/Qwen2.5-Coder-0.5B");
    assert_eq!(org, "Qwen");
    assert_eq!(repo, "Qwen2.5-Coder-0.5B");
}

/// Verify split_hf_repo returns "unknown" org for bare model names
#[test]
fn test_split_hf_repo_no_slash() {
    let (org, repo) = split_hf_repo("model-only");
    assert_eq!(org, "unknown");
    assert_eq!(repo, "model-only");
}

/// Verify split_hf_repo handles multiple slashes (only first split)
#[test]
fn test_split_hf_repo_multiple_slashes() {
    let (org, repo) = split_hf_repo("org/repo/extra");
    assert_eq!(org, "org");
    assert_eq!(repo, "repo/extra");
}

/// Verify find_hf_snapshot returns None when snapshots dir does not exist
#[test]
fn test_find_hf_snapshot_no_dir() {
    let dir = tempfile::tempdir().unwrap();
    let result = find_hf_snapshot(dir.path(), "Qwen", "Qwen2.5-Coder-0.5B");
    assert!(result.is_none());
}

/// Verify find_hf_snapshot finds snapshot with model.safetensors
#[test]
fn test_find_hf_snapshot_found() {
    let dir = tempfile::tempdir().unwrap();
    let snapshot_dir = dir
        .path()
        .join("models--Qwen--Qwen2.5-Coder-0.5B")
        .join("snapshots")
        .join("abc123");
    std::fs::create_dir_all(&snapshot_dir).unwrap();
    std::fs::write(snapshot_dir.join("model.safetensors"), "fake").unwrap();

    let result = find_hf_snapshot(dir.path(), "Qwen", "Qwen2.5-Coder-0.5B");
    assert!(result.is_some());
    assert_eq!(result.unwrap(), snapshot_dir);
}

/// Verify find_hf_snapshot skips snapshots without model.safetensors
#[test]
fn test_find_hf_snapshot_no_safetensors() {
    let dir = tempfile::tempdir().unwrap();
    let snapshot_dir = dir
        .path()
        .join("models--Qwen--Qwen2.5-Coder-0.5B")
        .join("snapshots")
        .join("abc123");
    std::fs::create_dir_all(&snapshot_dir).unwrap();
    // No model.safetensors file

    let result = find_hf_snapshot(dir.path(), "Qwen", "Qwen2.5-Coder-0.5B");
    assert!(result.is_none());
}

/// Verify find_hf_snapshot finds sharded models via index file
#[test]
fn test_find_hf_snapshot_sharded() {
    let dir = tempfile::tempdir().unwrap();
    let snapshot_dir = dir
        .path()
        .join("models--microsoft--Phi-3-mini-4k-instruct")
        .join("snapshots")
        .join("abc123");
    std::fs::create_dir_all(&snapshot_dir).unwrap();
    // Sharded layout: index + shard files, no monolithic model.safetensors
    std::fs::write(snapshot_dir.join("model.safetensors.index.json"), "{}").unwrap();
    std::fs::write(
        snapshot_dir.join("model-00001-of-00002.safetensors"),
        "shard1",
    )
    .unwrap();
    std::fs::write(
        snapshot_dir.join("model-00002-of-00002.safetensors"),
        "shard2",
    )
    .unwrap();

    let result = find_hf_snapshot(dir.path(), "microsoft", "Phi-3-mini-4k-instruct");
    assert!(result.is_some());
    assert_eq!(result.unwrap(), snapshot_dir);
}

/// Verify resolve_hf_repo_with_dirs finds sharded HF cache models
#[test]
fn test_resolve_with_dirs_sharded_hf_cache() {
    let hf_cache = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();

    let snapshot = hf_cache
        .path()
        .join("models--microsoft--Phi-3-mini-4k-instruct")
        .join("snapshots")
        .join("f39ac1d2");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("model.safetensors.index.json"), "{}").unwrap();
    std::fs::write(snapshot.join("model-00001-of-00002.safetensors"), "s1").unwrap();

    let result = resolve_hf_repo_with_dirs(
        "microsoft/Phi-3-mini-4k-instruct",
        hf_cache.path(),
        home.path(),
    );
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), snapshot);
}

/// Verify find_apr_cache returns None for nonexistent cache
#[test]
fn test_find_apr_cache_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let result = find_apr_cache(dir.path(), "Qwen", "Qwen2.5-Coder-0.5B");
    assert!(result.is_none());
}

/// Verify find_apr_cache finds existing APR cache directory
#[test]
fn test_find_apr_cache_found() {
    let dir = tempfile::tempdir().unwrap();
    let cache_dir = dir
        .path()
        .join(".cache/apr-models/Qwen/Qwen2.5-Coder-0.5B");
    std::fs::create_dir_all(&cache_dir).unwrap();

    let result = find_apr_cache(dir.path(), "Qwen", "Qwen2.5-Coder-0.5B");
    assert!(result.is_some());
    assert_eq!(result.unwrap(), cache_dir);
}

/// Verify resolve_hf_repo_with_dirs finds model in HF cache
#[test]
fn test_resolve_with_dirs_hf_cache() {
    let hf_cache = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();

    // Set up HF cache snapshot
    let snapshot = hf_cache
        .path()
        .join("models--Qwen--Qwen2.5-Coder-0.5B")
        .join("snapshots")
        .join("abc123");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("model.safetensors"), "fake").unwrap();

    let result =
        resolve_hf_repo_with_dirs("Qwen/Qwen2.5-Coder-0.5B", hf_cache.path(), home.path());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), snapshot);
}

/// Verify resolve_hf_repo_with_dirs falls back to APR cache
#[test]
fn test_resolve_with_dirs_apr_cache() {
    let hf_cache = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();

    // No HF cache, but APR cache exists
    let apr_dir = home
        .path()
        .join(".cache/apr-models/Qwen/Qwen2.5-Coder-0.5B");
    std::fs::create_dir_all(&apr_dir).unwrap();

    let result =
        resolve_hf_repo_with_dirs("Qwen/Qwen2.5-Coder-0.5B", hf_cache.path(), home.path());
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), apr_dir);
}

/// Verify resolve_hf_repo_with_dirs returns error when model not in any cache
#[test]
fn test_resolve_with_dirs_not_found() {
    let hf_cache = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();

    let result =
        resolve_hf_repo_with_dirs("Qwen/Qwen2.5-Coder-0.5B", hf_cache.path(), home.path());
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    assert!(err.contains("Model not found in cache"));
    assert!(err.contains("Qwen/Qwen2.5-Coder-0.5B"));
}

/// Verify resolve_hf_repo_with_dirs prefers HF cache over APR cache
#[test]
fn test_resolve_with_dirs_hf_preferred() {
    let hf_cache = tempfile::tempdir().unwrap();
    let home = tempfile::tempdir().unwrap();

    // Both caches exist
    let snapshot = hf_cache
        .path()
        .join("models--org--repo")
        .join("snapshots")
        .join("abc");
    std::fs::create_dir_all(&snapshot).unwrap();
    std::fs::write(snapshot.join("model.safetensors"), "hf").unwrap();

    let apr_dir = home.path().join(".cache/apr-models/org/repo");
    std::fs::create_dir_all(&apr_dir).unwrap();

    let result = resolve_hf_repo_with_dirs("org/repo", hf_cache.path(), home.path());
    assert!(result.is_ok());
    // Should resolve to HF snapshot, not APR cache
    assert_eq!(result.unwrap(), snapshot);
}

/// Verify get_hf_cache_dir returns a non-empty path
#[test]
fn test_get_hf_cache_dir_returns_path() {
    let path = get_hf_cache_dir();
    assert!(!path.as_os_str().is_empty());
}
