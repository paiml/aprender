//! Finding cache with mtime-based invalidation for bug-hunter.
//!
//! Caches findings to disk using FNV-1a hashed keys. Invalidates when any
//! source file under the project is newer than the cache file.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use super::types::{Finding, HuntMode};

/// Cached findings stored on disk.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct CachedFindings {
    pub findings: Vec<Finding>,
    pub mode: HuntMode,
    pub config_hash: String,
}

/// Compute an FNV-1a hash of the cache-relevant config fields.
///
/// Key components: project path, mode, targets, min_suspiciousness, use_pmat_quality.
pub fn cache_key(project_path: &Path, config: &super::types::HuntConfig) -> String {
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    let prime: u64 = 0x00000100000001B3;

    let input = format!(
        "{}:{}:{:?}:{:.4}:{}:{}:{}:{}:{}",
        project_path.display(),
        config.mode,
        config.targets,
        config.min_suspiciousness,
        config.use_pmat_quality,
        config.contracts_auto,
        config.contracts_path.as_deref().map(|p| p.display().to_string()).unwrap_or_default(),
        config.model_parity_auto,
        config.model_parity_path.as_deref().map(|p| p.display().to_string()).unwrap_or_default(),
    );

    for byte in input.bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(prime);
    }

    format!("{:016x}", hash)
}

/// Check if any `.rs` source file under `project_path` is newer than `cache_mtime`.
pub fn any_source_newer_than(project_path: &Path, cache_mtime: SystemTime) -> bool {
    let pattern = format!("{}/**/*.rs", project_path.display());
    if let Ok(entries) = glob::glob(&pattern) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if let Ok(mtime) = meta.modified() {
                    if mtime > cache_mtime {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Cache directory for bug-hunter findings.
fn cache_dir(project_path: &Path) -> PathBuf {
    project_path.join(".pmat").join("bug-hunter-cache")
}

/// Load cached findings if the cache exists and is still valid.
///
/// Returns `None` if the cache file doesn't exist, is corrupt, or if any
/// source file has been modified since the cache was written.
pub fn load_cached(
    project_path: &Path,
    config: &super::types::HuntConfig,
) -> Option<CachedFindings> {
    let key = cache_key(project_path, config);
    let cache_file = cache_dir(project_path).join(format!("{}.json", key));

    if !cache_file.exists() {
        return None;
    }

    let cache_mtime = cache_file.metadata().ok()?.modified().ok()?;

    if any_source_newer_than(project_path, cache_mtime) {
        return None;
    }

    let content = std::fs::read_to_string(&cache_file).ok()?;
    let cached: CachedFindings = serde_json::from_str(&content).ok()?;

    // Verify config hash matches
    if cached.config_hash != key {
        return None;
    }

    Some(cached)
}

/// Save findings to the cache.
pub fn save_cache(
    project_path: &Path,
    config: &super::types::HuntConfig,
    findings: &[Finding],
    mode: HuntMode,
) {
    let key = cache_key(project_path, config);
    let dir = cache_dir(project_path);

    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }

    let cached = CachedFindings { findings: findings.to_vec(), mode, config_hash: key.clone() };

    let cache_file = dir.join(format!("{}.json", key));
    if let Ok(json) = serde_json::to_string(&cached) {
        let _ = std::fs::write(&cache_file, json);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bug_hunter::types::HuntConfig;

    #[test]
    fn test_cache_key_deterministic() {
        let config = HuntConfig::default();
        let k1 = cache_key(Path::new("/tmp/proj"), &config);
        let k2 = cache_key(Path::new("/tmp/proj"), &config);
        assert_eq!(k1, k2, "Same inputs must produce same key");
    }

    #[test]
    fn test_cache_key_varies() {
        let c1 = HuntConfig::default();
        let c2 = HuntConfig { mode: HuntMode::Quick, ..Default::default() };

        let k1 = cache_key(Path::new("/tmp/proj"), &c1);
        let k2 = cache_key(Path::new("/tmp/proj"), &c2);
        assert_ne!(k1, k2, "Different modes must produce different keys");

        let k3 = cache_key(Path::new("/tmp/other"), &c1);
        assert_ne!(k1, k3, "Different paths must produce different keys");
    }

    #[test]
    fn test_load_cached_empty_dir() {
        let temp = std::env::temp_dir().join("test_bh_cache_empty");
        let _ = std::fs::create_dir_all(&temp);
        let config = HuntConfig::default();

        let result = load_cached(&temp, &config);
        assert!(result.is_none(), "Empty dir should return None");

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let temp = std::env::temp_dir().join("test_bh_cache_roundtrip");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(&temp);

        let config = HuntConfig { mode: HuntMode::Quick, ..Default::default() };

        let findings = vec![Finding::new("BH-001", "src/lib.rs", 42, "Test finding")];

        save_cache(&temp, &config, &findings, HuntMode::Quick);

        let cached = load_cached(&temp, &config);
        assert!(cached.is_some(), "Should load cached findings");

        let cached = cached.expect("unexpected failure");
        assert_eq!(cached.findings.len(), 1);
        assert_eq!(cached.findings[0].id, "BH-001");
        assert_eq!(cached.mode, HuntMode::Quick);

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_any_source_newer_than_returns_true() {
        // Create a temp dir with a .rs file, then use an old timestamp
        let temp = std::env::temp_dir().join("test_bh_cache_newer");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(&temp);
        std::fs::write(temp.join("test.rs"), "fn main() {}").expect("write");

        // Use UNIX_EPOCH as cache_mtime; any file will be newer than epoch
        let old_time = std::time::UNIX_EPOCH;
        assert!(
            any_source_newer_than(&temp, old_time),
            "Source file should be newer than UNIX_EPOCH"
        );

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_load_cached_invalidated_by_newer_source() {
        // Save a cache, then create a .rs file newer than the cache
        let temp = std::env::temp_dir().join("test_bh_cache_invalidate");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(&temp);

        let config = HuntConfig { mode: HuntMode::Quick, ..Default::default() };

        let findings = vec![Finding::new("BH-002", "src/lib.rs", 1, "Finding")];
        save_cache(&temp, &config, &findings, HuntMode::Quick);

        // Now create a .rs file so it's newer than the cache
        std::thread::sleep(std::time::Duration::from_millis(50));
        std::fs::write(temp.join("new_file.rs"), "// new").expect("write");

        let cached = load_cached(&temp, &config);
        assert!(cached.is_none(), "Cache should be invalidated by newer source");

        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_cache_key_varies_with_contracts_auto() {
        let c1 = HuntConfig::default();
        let c2 = HuntConfig { contracts_auto: true, ..Default::default() };

        let k1 = cache_key(Path::new("/tmp/proj"), &c1);
        let k2 = cache_key(Path::new("/tmp/proj"), &c2);
        assert_ne!(k1, k2, "contracts_auto must change cache key");
    }

    #[test]
    fn test_cache_key_varies_with_model_parity_auto() {
        let c1 = HuntConfig::default();
        let c2 = HuntConfig { model_parity_auto: true, ..Default::default() };

        let k1 = cache_key(Path::new("/tmp/proj"), &c1);
        let k2 = cache_key(Path::new("/tmp/proj"), &c2);
        assert_ne!(k1, k2, "model_parity_auto must change cache key");
    }

    #[test]
    fn test_cache_key_varies_with_contracts_path() {
        let c1 = HuntConfig::default();
        let c2 = HuntConfig {
            contracts_path: Some(PathBuf::from("/tmp/contracts")),
            ..Default::default()
        };

        let k1 = cache_key(Path::new("/tmp/proj"), &c1);
        let k2 = cache_key(Path::new("/tmp/proj"), &c2);
        assert_ne!(k1, k2, "contracts_path must change cache key");
    }

    #[test]
    fn test_load_cached_config_hash_mismatch() {
        // Manually write a cache file with a wrong config_hash
        let temp = std::env::temp_dir().join("test_bh_cache_hash_mismatch");
        let _ = std::fs::remove_dir_all(&temp);
        let _ = std::fs::create_dir_all(&temp);

        let config = HuntConfig { mode: HuntMode::Quick, ..Default::default() };

        let key = cache_key(&temp, &config);
        let dir = temp.join(".pmat").join("bug-hunter-cache");
        let _ = std::fs::create_dir_all(&dir);

        let cached = CachedFindings {
            findings: vec![],
            mode: HuntMode::Quick,
            config_hash: "wrong_hash_value".to_string(),
        };
        let cache_file = dir.join(format!("{}.json", key));
        let json = serde_json::to_string(&cached).expect("serialize");
        std::fs::write(&cache_file, json).expect("write");

        let result = load_cached(&temp, &config);
        assert!(result.is_none(), "Should return None on config hash mismatch");

        let _ = std::fs::remove_dir_all(&temp);
    }
}
