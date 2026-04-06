//! Lock file management and cache decision logic (PB-004)
//!
//! Handles lock file persistence (atomic write via temp+rename) and cache
//! hit/miss determination with detailed invalidation reasons.

use super::types::*;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Cache decision for a stage
#[derive(Debug, Clone)]
pub enum CacheDecision {
    /// Cache hit — skip execution
    Hit,
    /// Cache miss — must execute
    Miss { reasons: Vec<InvalidationReason> },
}

/// Derive the lock file path from a playbook path: `.yaml` → `.lock.yaml`
pub fn lock_file_path(playbook_path: &Path) -> PathBuf {
    let stem = playbook_path.file_stem().unwrap_or_default().to_string_lossy();
    playbook_path.with_file_name(format!("{}.lock.yaml", stem))
}

/// Load a lock file if it exists
pub fn load_lock_file(playbook_path: &Path) -> Result<Option<LockFile>> {
    let path = lock_file_path(playbook_path);
    if !path.exists() {
        return Ok(None);
    }
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("failed to read lock file: {}", path.display()))?;
    let lock: LockFile = serde_yaml_ng::from_str(&content)
        .with_context(|| format!("failed to parse lock file: {}", path.display()))?;
    Ok(Some(lock))
}

/// Save a lock file atomically (write to temp, then rename)
pub fn save_lock_file(lock: &LockFile, playbook_path: &Path) -> Result<()> {
    let path = lock_file_path(playbook_path);
    let yaml = serde_yaml_ng::to_string(lock).context("failed to serialize lock file")?;

    // Write-then-rename for crash-safe persistence
    let parent = path.parent().unwrap_or(Path::new("."));
    let temp_path = parent
        .join(format!(".{}.tmp", path.file_name().expect("file_name missing").to_string_lossy()));

    std::fs::write(&temp_path, yaml.as_bytes())
        .with_context(|| format!("failed to write temp lock file: {}", temp_path.display()))?;

    std::fs::rename(&temp_path, &path)
        .with_context(|| format!("failed to rename lock file: {}", path.display()))?;

    Ok(())
}

/// Check cache status for a stage
pub fn check_cache(
    stage_name: &str,
    current_cache_key: &str,
    current_cmd_hash: &str,
    current_deps_hashes: &[(String, String)], // (path, hash)
    current_params_hash: &str,
    lock: &Option<LockFile>,
    forced: bool,
    upstream_rerun: &[String],
) -> CacheDecision {
    let mut reasons = Vec::new();

    if forced {
        reasons.push(InvalidationReason::Forced);
        return CacheDecision::Miss { reasons };
    }

    // Check upstream reruns
    for stage in upstream_rerun {
        reasons.push(InvalidationReason::UpstreamRerun { stage: stage.clone() });
    }
    if !reasons.is_empty() {
        return CacheDecision::Miss { reasons };
    }

    let lock = match lock {
        Some(l) => l,
        None => {
            reasons.push(InvalidationReason::NoLockFile);
            return CacheDecision::Miss { reasons };
        }
    };

    let stage_lock = match lock.stages.get(stage_name) {
        Some(sl) => sl,
        None => {
            reasons.push(InvalidationReason::StageNotInLock);
            return CacheDecision::Miss { reasons };
        }
    };

    // Only completed stages can be cached
    if stage_lock.status != StageStatus::Completed {
        reasons.push(InvalidationReason::PreviousRunIncomplete {
            status: format!("{:?}", stage_lock.status).to_lowercase(),
        });
        return CacheDecision::Miss { reasons };
    }

    // Check cache_key match (primary check)
    let Some(ref old_key) = stage_lock.cache_key else {
        reasons.push(InvalidationReason::StageNotInLock);
        return CacheDecision::Miss { reasons };
    };
    if old_key != current_cache_key {
        diagnose_key_mismatch(
            stage_lock,
            current_cmd_hash,
            current_deps_hashes,
            current_params_hash,
            old_key,
            current_cache_key,
            &mut reasons,
        );
        return CacheDecision::Miss { reasons };
    }

    // Check output files still exist
    for out in &stage_lock.outs {
        if !Path::new(&out.path).exists() {
            reasons.push(InvalidationReason::OutputMissing { path: out.path.clone() });
        }
    }

    if reasons.is_empty() {
        CacheDecision::Hit
    } else {
        CacheDecision::Miss { reasons }
    }
}

/// Diagnose why a cache key changed by comparing component hashes.
fn diagnose_key_mismatch(
    stage_lock: &StageLock,
    current_cmd_hash: &str,
    current_deps_hashes: &[(String, String)],
    current_params_hash: &str,
    old_key: &str,
    new_key: &str,
    reasons: &mut Vec<InvalidationReason>,
) {
    if let Some(ref old_cmd) = stage_lock.cmd_hash {
        if old_cmd != current_cmd_hash {
            reasons.push(InvalidationReason::CmdChanged {
                old: old_cmd.clone(),
                new: current_cmd_hash.to_string(),
            });
        }
    }

    for (path, new_hash) in current_deps_hashes {
        let old_hash =
            stage_lock.deps.iter().find(|d| d.path == *path).map(|d| d.hash.as_str()).unwrap_or("");
        if old_hash != new_hash {
            reasons.push(InvalidationReason::DepChanged {
                path: path.clone(),
                old_hash: old_hash.to_string(),
                new_hash: new_hash.clone(),
            });
        }
    }

    if let Some(ref old_params) = stage_lock.params_hash {
        if old_params != current_params_hash {
            reasons.push(InvalidationReason::ParamsChanged {
                old: old_params.clone(),
                new: current_params_hash.to_string(),
            });
        }
    }

    if reasons.is_empty() {
        reasons.push(InvalidationReason::CacheKeyMismatch {
            old: old_key.to_string(),
            new: new_key.to_string(),
        });
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use indexmap::IndexMap;

    fn make_lock_file(stage_name: &str, cache_key: &str) -> LockFile {
        LockFile {
            schema: "1.0".to_string(),
            playbook: "test".to_string(),
            generated_at: "2026-02-16T14:00:00Z".to_string(),
            generator: "batuta 0.6.5".to_string(),
            blake3_version: "1.8".to_string(),
            params_hash: Some("blake3:params".to_string()),
            stages: IndexMap::from([(
                stage_name.to_string(),
                StageLock {
                    status: StageStatus::Completed,
                    started_at: Some("2026-02-16T14:00:00Z".to_string()),
                    completed_at: Some("2026-02-16T14:00:01Z".to_string()),
                    duration_seconds: Some(1.0),
                    target: None,
                    deps: vec![DepLock {
                        path: "/tmp/in.txt".to_string(),
                        hash: "blake3:dep_hash".to_string(),
                        file_count: Some(1),
                        total_bytes: Some(100),
                    }],
                    params_hash: Some("blake3:stage_params".to_string()),
                    outs: vec![],
                    cmd_hash: Some("blake3:cmd_hash".to_string()),
                    cache_key: Some(cache_key.to_string()),
                },
            )]),
        }
    }

    #[test]
    fn test_PB004_lock_file_path_derivation() {
        let path = lock_file_path(Path::new("/tmp/pipeline.yaml"));
        assert_eq!(path, PathBuf::from("/tmp/pipeline.lock.yaml"));
    }

    #[test]
    fn test_PB004_lock_file_path_nested() {
        let path = lock_file_path(Path::new("/home/user/playbooks/build.yaml"));
        assert_eq!(path, PathBuf::from("/home/user/playbooks/build.lock.yaml"));
    }

    #[test]
    fn test_PB004_lock_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let playbook_path = dir.path().join("test.yaml");
        std::fs::write(&playbook_path, "").expect("fs write failed");

        let lock = make_lock_file("hello", "blake3:key123");
        save_lock_file(&lock, &playbook_path).expect("unexpected failure");

        let loaded = load_lock_file(&playbook_path)
            .expect("unexpected failure")
            .expect("unexpected failure");
        assert_eq!(loaded.playbook, "test");
        assert_eq!(loaded.stages["hello"].cache_key.as_deref(), Some("blake3:key123"));
    }

    #[test]
    fn test_PB004_load_nonexistent() {
        let result = load_lock_file(Path::new("/tmp/nonexistent_playbook.yaml"))
            .expect("unexpected failure");
        assert!(result.is_none());
    }

    #[test]
    fn test_PB004_cache_hit() {
        let lock = make_lock_file("hello", "blake3:key123");
        let decision = check_cache(
            "hello",
            "blake3:key123",
            "blake3:cmd_hash",
            &[("/tmp/in.txt".to_string(), "blake3:dep_hash".to_string())],
            "blake3:stage_params",
            &Some(lock),
            false,
            &[],
        );
        assert!(matches!(decision, CacheDecision::Hit));
    }

    #[test]
    fn test_PB004_cache_miss_no_lock() {
        let decision = check_cache(
            "hello",
            "blake3:key123",
            "blake3:cmd",
            &[],
            "blake3:params",
            &None,
            false,
            &[],
        );
        match decision {
            CacheDecision::Miss { reasons } => {
                assert_eq!(reasons.len(), 1);
                assert!(matches!(reasons[0], InvalidationReason::NoLockFile));
            }
            _ => panic!("expected miss"),
        }
    }

    #[test]
    fn test_PB004_cache_miss_stage_not_in_lock() {
        let lock = make_lock_file("hello", "blake3:key123");
        let decision = check_cache(
            "other_stage",
            "blake3:key123",
            "blake3:cmd",
            &[],
            "blake3:params",
            &Some(lock),
            false,
            &[],
        );
        match decision {
            CacheDecision::Miss { reasons } => {
                assert!(matches!(reasons[0], InvalidationReason::StageNotInLock));
            }
            _ => panic!("expected miss"),
        }
    }

    #[test]
    fn test_PB004_cache_miss_cmd_changed() {
        let lock = make_lock_file("hello", "blake3:old_key");
        let decision = check_cache(
            "hello",
            "blake3:new_key",
            "blake3:new_cmd",
            &[("/tmp/in.txt".to_string(), "blake3:dep_hash".to_string())],
            "blake3:stage_params",
            &Some(lock),
            false,
            &[],
        );
        match decision {
            CacheDecision::Miss { reasons } => {
                assert!(reasons.iter().any(|r| matches!(r, InvalidationReason::CmdChanged { .. })));
            }
            _ => panic!("expected miss"),
        }
    }

    #[test]
    fn test_PB004_cache_miss_forced() {
        let lock = make_lock_file("hello", "blake3:key123");
        let decision = check_cache(
            "hello",
            "blake3:key123",
            "blake3:cmd",
            &[],
            "blake3:params",
            &Some(lock),
            true, // forced
            &[],
        );
        match decision {
            CacheDecision::Miss { reasons } => {
                assert!(matches!(reasons[0], InvalidationReason::Forced));
            }
            _ => panic!("expected miss"),
        }
    }

    #[test]
    fn test_PB004_cache_miss_upstream_rerun() {
        let lock = make_lock_file("hello", "blake3:key123");
        let decision = check_cache(
            "hello",
            "blake3:key123",
            "blake3:cmd",
            &[],
            "blake3:params",
            &Some(lock),
            false,
            &["upstream_stage".to_string()],
        );
        match decision {
            CacheDecision::Miss { reasons } => {
                assert!(matches!(reasons[0], InvalidationReason::UpstreamRerun { .. }));
            }
            _ => panic!("expected miss"),
        }
    }

    #[test]
    fn test_PB004_cache_miss_dep_changed() {
        let lock = make_lock_file("hello", "blake3:old_key");
        let decision = check_cache(
            "hello",
            "blake3:new_key",
            "blake3:cmd_hash", // cmd unchanged
            &[("/tmp/in.txt".to_string(), "blake3:new_dep_hash".to_string())],
            "blake3:stage_params", // params unchanged
            &Some(lock),
            false,
            &[],
        );
        match decision {
            CacheDecision::Miss { reasons } => {
                assert!(reasons.iter().any(|r| matches!(r, InvalidationReason::DepChanged { .. })));
            }
            _ => panic!("expected miss"),
        }
    }

    #[test]
    fn test_PB004_atomic_write_survives_crash() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let playbook_path = dir.path().join("test.yaml");
        std::fs::write(&playbook_path, "").expect("fs write failed");

        // Write initial lock
        let lock1 = make_lock_file("hello", "blake3:key1");
        save_lock_file(&lock1, &playbook_path).expect("unexpected failure");

        // Write updated lock
        let lock2 = make_lock_file("hello", "blake3:key2");
        save_lock_file(&lock2, &playbook_path).expect("unexpected failure");

        // Verify the latest write persisted
        let loaded = load_lock_file(&playbook_path)
            .expect("unexpected failure")
            .expect("unexpected failure");
        assert_eq!(loaded.stages["hello"].cache_key.as_deref(), Some("blake3:key2"));

        // Verify no temp files remain
        let entries: Vec<_> = std::fs::read_dir(dir.path())
            .expect("unexpected failure")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().contains(".tmp"))
            .collect();
        assert!(entries.is_empty());
    }
}
