//! Drift detection — detect stale contracts via git timestamps.
//!
//! A contract is "stale" if it was modified after the binding file was
//! last updated, meaning the binding may reference an outdated version.
//!
//! `drift = 1.0 - (stale_contracts / total_bound_contracts)`

use std::collections::HashSet;
use std::path::Path;

/// Detect which bound contracts are stale relative to a binding file.
///
/// Uses `git log -1 --format=%ct -- <file>` to get the last-commit
/// timestamp for each file. A contract is stale if its commit timestamp
/// is more recent than the binding file's commit timestamp.
///
/// Returns the set of stale contract stems (filenames like "softmax-kernel-v1.yaml").
pub fn detect_stale_contracts<S: std::hash::BuildHasher>(
    contract_dir: &Path,
    binding_path: &Path,
    bound_stems: &HashSet<&str, S>,
) -> HashSet<String> {
    let mut stale = HashSet::new();

    let binding_ts = git_last_commit_ts(binding_path);
    if binding_ts == 0 {
        // No git history for binding — can't detect drift
        return stale;
    }

    for stem in bound_stems {
        let contract_path = contract_dir.join(stem);
        if !contract_path.exists() {
            continue;
        }
        let contract_ts = git_last_commit_ts(&contract_path);
        if contract_ts > binding_ts {
            stale.insert((*stem).to_string());
        }
    }

    stale
}

/// Compute drift score from stale contract count.
///
/// `drift = 1.0 - (stale / total)` where:
/// - `stale` = number of bound contracts modified after binding
/// - `total` = total number of bound contracts
///
/// Returns 1.0 (perfect freshness) when no contracts are stale,
/// and 0.0 when all bound contracts are stale.
#[allow(clippy::cast_precision_loss)]
pub fn compute_drift(stale_count: usize, total_bound: usize) -> f64 {
    if total_bound == 0 {
        return 1.0;
    }
    1.0 - (stale_count as f64 / total_bound as f64)
}

/// Get the unix timestamp of the last git commit that touched a file.
///
/// Returns 0 if the file is not tracked by git or git is unavailable.
fn git_last_commit_ts(path: &Path) -> u64 {
    let output = std::process::Command::new("git")
        .args(["log", "-1", "--format=%ct", "--"])
        .arg(path)
        .output();

    match output {
        Ok(o) if o.status.success() => {
            let s = String::from_utf8_lossy(&o.stdout);
            s.trim().parse().unwrap_or(0)
        }
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_drift_no_stale() {
        assert!((compute_drift(0, 10) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn compute_drift_all_stale() {
        assert!((compute_drift(5, 5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn compute_drift_partial() {
        assert!((compute_drift(2, 10) - 0.8).abs() < 1e-9);
    }

    #[test]
    fn compute_drift_empty_total() {
        assert!((compute_drift(0, 0) - 1.0).abs() < 1e-9);
    }

    #[test]
    #[ignore] // Requires git repo — skipped in container CI
    fn git_ts_for_real_contract() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../contracts/softmax-kernel-v1.yaml");
        let ts = git_last_commit_ts(&path);
        assert!(ts > 0, "softmax contract should have a git timestamp");
    }

    #[test]
    fn git_ts_nonexistent_file() {
        let ts = git_last_commit_ts(Path::new("/nonexistent/file.yaml"));
        assert_eq!(ts, 0);
    }

    #[test]
    fn detect_stale_with_real_files() {
        let contract_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let binding_path = contract_dir.join("aprender/binding.yaml");

        let mut bound = HashSet::new();
        bound.insert("softmax-kernel-v1.yaml");

        let stale = detect_stale_contracts(&contract_dir, &binding_path, &bound);
        // Result depends on actual git history — just check it doesn't panic
        assert!(stale.len() <= 1);
    }

    #[test]
    fn detect_stale_no_git_binding() {
        // Binding path with no git history → early return, no stale contracts
        let contract_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let mut bound = HashSet::new();
        bound.insert("softmax-kernel-v1.yaml");
        let stale = detect_stale_contracts(
            &contract_dir,
            Path::new("/tmp/nonexistent-binding.yaml"),
            &bound,
        );
        assert!(stale.is_empty());
    }

    #[test]
    fn detect_stale_missing_contract_file() {
        // Bound stem that doesn't exist on disk → skip (continue branch)
        let contract_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../contracts");
        let binding_path = contract_dir.join("aprender/binding.yaml");
        let mut bound = HashSet::new();
        bound.insert("nonexistent-contract-v1.yaml");
        let stale = detect_stale_contracts(&contract_dir, &binding_path, &bound);
        assert!(stale.is_empty());
    }
}
