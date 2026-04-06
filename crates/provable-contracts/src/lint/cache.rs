//! Content-addressable lint cache.
//!
//! Each contract's lint result is cached using a hash of (YAML content +
//! rule config). Cache is stored in `.pv/cache/lint/`. Automatically
//! invalidated when any input changes.
//!
//! Spec: `docs/specifications/sub/lint.md` Section 10

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::finding::LintFinding;

/// Cache entry for one contract's lint results.
#[derive(Debug, Serialize, Deserialize)]
pub struct CacheEntry {
    pub content_hash: String,
    pub findings: Vec<LintFinding>,
}

/// Cache statistics.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub total: usize,
    pub hits: usize,
    pub misses: usize,
}

impl CacheStats {
    #[allow(clippy::cast_precision_loss)]
    pub fn hit_rate(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.hits as f64 / self.total as f64
        }
    }
}

/// Compute a content hash for cache key.
pub fn content_hash(yaml_content: &str, rule_config: &str) -> String {
    let mut hasher = DefaultHasher::new();
    yaml_content.hash(&mut hasher);
    rule_config.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Get the cache directory path.
pub fn cache_dir(base: &Path) -> PathBuf {
    base.join(".pv").join("cache").join("lint")
}

/// Look up a cached result.
pub fn cache_get(cache_root: &Path, hash: &str) -> Option<CacheEntry> {
    let path = cache_root.join(format!("{hash}.json"));
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Store a lint result in the cache.
pub fn cache_put(cache_root: &Path, hash: &str, findings: &[LintFinding]) -> Result<(), String> {
    std::fs::create_dir_all(cache_root).map_err(|e| format!("Failed to create cache dir: {e}"))?;

    let entry = CacheEntry {
        content_hash: hash.to_string(),
        findings: findings.to_vec(),
    };

    let json = serde_json::to_string(&entry)
        .map_err(|e| format!("Failed to serialize cache entry: {e}"))?;

    let path = cache_root.join(format!("{hash}.json"));
    std::fs::write(path, json).map_err(|e| format!("Failed to write cache entry: {e}"))?;

    Ok(())
}

/// Clear all cached results.
pub fn cache_clear(cache_root: &Path) -> Result<usize, String> {
    let mut count = 0;
    if cache_root.is_dir() {
        let entries =
            std::fs::read_dir(cache_root).map_err(|e| format!("Failed to read cache dir: {e}"))?;
        for entry in entries.flatten() {
            if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                let _ = std::fs::remove_file(entry.path());
                count += 1;
            }
        }
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lint::rules::RuleSeverity;

    #[test]
    fn content_hash_deterministic() {
        let h1 = content_hash("yaml content", "rules");
        let h2 = content_hash("yaml content", "rules");
        assert_eq!(h1, h2);
    }

    #[test]
    fn content_hash_differs_on_content() {
        let h1 = content_hash("yaml A", "rules");
        let h2 = content_hash("yaml B", "rules");
        assert_ne!(h1, h2);
    }

    #[test]
    fn content_hash_differs_on_rules() {
        let h1 = content_hash("yaml", "rules A");
        let h2 = content_hash("yaml", "rules B");
        assert_ne!(h1, h2);
    }

    #[test]
    fn cache_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");

        let finding = LintFinding::new("PV-VAL-001", RuleSeverity::Error, "test", "file.yaml");
        cache_put(&cache, "abc123", &[finding]).unwrap();

        let entry = cache_get(&cache, "abc123").unwrap();
        assert_eq!(entry.content_hash, "abc123");
        assert_eq!(entry.findings.len(), 1);
        assert_eq!(entry.findings[0].rule_id, "PV-VAL-001");
    }

    #[test]
    fn cache_miss() {
        let tmp = tempfile::tempdir().unwrap();
        assert!(cache_get(tmp.path(), "nonexistent").is_none());
    }

    #[test]
    fn cache_clear_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let count = cache_clear(tmp.path()).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn cache_clear_removes_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = tmp.path().join("cache");

        let finding = LintFinding::new("PV-VAL-001", RuleSeverity::Error, "t", "f.yaml");
        cache_put(&cache, "hash1", std::slice::from_ref(&finding)).unwrap();
        cache_put(&cache, "hash2", std::slice::from_ref(&finding)).unwrap();

        let count = cache_clear(&cache).unwrap();
        assert_eq!(count, 2);
        assert!(cache_get(&cache, "hash1").is_none());
    }

    #[test]
    fn cache_stats_default() {
        let stats = CacheStats::default();
        assert_eq!(stats.total, 0);
        assert!(stats.hit_rate().abs() < f64::EPSILON);
    }

    #[test]
    fn cache_stats_hit_rate() {
        let stats = CacheStats {
            total: 10,
            hits: 7,
            misses: 3,
        };
        assert!((stats.hit_rate() - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn cache_dir_path() {
        let base = Path::new("/repo");
        let dir = cache_dir(base);
        assert_eq!(dir, PathBuf::from("/repo/.pv/cache/lint"));
    }
}
