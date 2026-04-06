//! Index persistence — save/load `ContractIndex` to `.pv/contracts.idx`.
//!
//! Auto-rebuilds when `contracts/` mtime > stored mtime.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::types::ContractEntry;

/// Persisted index data (JSON-serializable subset of `ContractIndex`).
#[derive(Serialize, Deserialize)]
pub struct PersistedIndex {
    pub entries: Vec<ContractEntry>,
    pub score_cache: std::collections::HashMap<String, f64>,
    pub pagerank_cache: std::collections::HashMap<String, f64>,
}

/// Get the `.pv` directory path relative to a contracts directory.
///
/// If `contracts_dir` is `foo/contracts`, the `.pv` dir is `foo/.pv`.
fn pv_dir(contracts_dir: &Path) -> PathBuf {
    contracts_dir.parent().unwrap_or(Path::new(".")).join(".pv")
}

fn index_path(contracts_dir: &Path) -> PathBuf {
    pv_dir(contracts_dir).join("contracts.idx")
}

fn mtime_path(contracts_dir: &Path) -> PathBuf {
    pv_dir(contracts_dir).join("contracts.idx.mtime")
}

/// Get the max mtime of all YAML files in a directory (recursive).
pub fn dir_max_mtime(dir: &Path) -> Option<SystemTime> {
    let mut paths = vec![dir.to_path_buf()];
    let mut max_mtime: Option<SystemTime> = None;

    while let Some(current_dir) = paths.pop() {
        let entries = std::fs::read_dir(&current_dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                paths.push(path);
            } else if path.extension().and_then(|x| x.to_str()) == Some("yaml") {
                let mtime = std::fs::metadata(&path).ok()?.modified().ok()?;
                max_mtime = Some(max_mtime.map_or(mtime, |cur: SystemTime| cur.max(mtime)));
            }
        }
    }
    max_mtime
}

/// Try to load a cached index if it exists and is fresh.
///
/// Returns `None` if the cache is missing, stale, or corrupt.
pub fn load_cached(contracts_dir: &Path) -> Option<PersistedIndex> {
    let idx_path = index_path(contracts_dir);
    let mt_path = mtime_path(contracts_dir);

    // Read stored mtime
    let stored_mtime_str = std::fs::read_to_string(&mt_path).ok()?;
    let stored_epoch: u64 = stored_mtime_str.trim().parse().ok()?;

    // Get current max mtime
    let current_mtime = dir_max_mtime(contracts_dir)?;
    let current_epoch = current_mtime
        .duration_since(SystemTime::UNIX_EPOCH)
        .ok()?
        .as_secs();

    // Stale check
    if current_epoch > stored_epoch {
        return None;
    }

    // Load and deserialize
    let data = std::fs::read_to_string(&idx_path).ok()?;
    serde_json::from_str(&data).ok()
}

/// Save an index to the `.pv` cache directory.
pub fn save_cached(
    contracts_dir: &Path,
    index: &PersistedIndex,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = pv_dir(contracts_dir);
    std::fs::create_dir_all(&dir)?;

    let idx_path = index_path(contracts_dir);
    let mt_path = mtime_path(contracts_dir);

    // Serialize index
    let data = serde_json::to_string(index)?;
    std::fs::write(&idx_path, data)?;

    // Store current mtime
    let mtime = dir_max_mtime(contracts_dir)
        .and_then(|t| t.duration_since(SystemTime::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    std::fs::write(&mt_path, mtime.to_string())?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn dir_max_mtime_contracts() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let mtime = dir_max_mtime(&dir);
        assert!(
            mtime.is_some(),
            "contracts/ should have YAML files with mtimes"
        );
    }

    #[test]
    fn dir_max_mtime_empty() {
        let mtime = dir_max_mtime(Path::new("/nonexistent"));
        assert!(mtime.is_none());
    }

    #[test]
    fn roundtrip_persisted_index() {
        let tmp = std::env::temp_dir().join("pv_persist_test");
        std::fs::create_dir_all(&tmp).unwrap();

        // Create a fake contracts dir with a yaml file
        let contracts = tmp.join("contracts");
        std::fs::create_dir_all(&contracts).unwrap();
        std::fs::write(contracts.join("test.yaml"), "metadata:\n  version: 1").unwrap();

        let idx = PersistedIndex {
            entries: vec![ContractEntry {
                stem: "test-v1".into(),
                path: "contracts/test-v1.yaml".into(),
                description: "Test".into(),
                equations: vec!["f".into()],
                obligation_types: vec!["invariant".into()],
                properties: vec!["finite".into()],
                references: vec![],
                depends_on: vec![],
                is_registry: false,
                kind: crate::schema::ContractKind::default(),
                obligation_count: 1,
                falsification_count: 0,
                kani_count: 0,
                corpus_text: "test-v1 test f finite".into(),
            }],
            score_cache: {
                let mut m = HashMap::new();
                m.insert("test-v1".into(), 0.5);
                m
            },
            pagerank_cache: HashMap::new(),
        };

        save_cached(&contracts, &idx).unwrap();
        let loaded = load_cached(&contracts).unwrap();
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].stem, "test-v1");
        assert_eq!(loaded.score_cache.get("test-v1").copied(), Some(0.5));

        // Cleanup
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn stale_cache_returns_none() {
        let tmp = std::env::temp_dir().join("pv_persist_stale_test");
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(&tmp).unwrap();

        let contracts = tmp.join("contracts");
        std::fs::create_dir_all(&contracts).unwrap();
        std::fs::write(contracts.join("test.yaml"), "metadata:\n  version: 1").unwrap();

        let idx = PersistedIndex {
            entries: vec![],
            score_cache: HashMap::new(),
            pagerank_cache: HashMap::new(),
        };
        save_cached(&contracts, &idx).unwrap();

        // Write a stale mtime (1 second in the past) to force staleness
        let mt = mtime_path(&contracts);
        let current: u64 = std::fs::read_to_string(&mt)
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        std::fs::write(&mt, (current - 1).to_string()).unwrap();

        // Touch a file to ensure current mtime > stored
        std::fs::write(contracts.join("new.yaml"), "metadata:\n  version: 2").unwrap();

        let loaded = load_cached(&contracts);
        assert!(loaded.is_none(), "Should be stale after adding new file");

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
