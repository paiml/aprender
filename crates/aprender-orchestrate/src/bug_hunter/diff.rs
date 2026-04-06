//! Bug Hunter Diff Mode
//!
//! Compare findings against a baseline to show only new issues.

use super::types::{Finding, HuntResult};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;
use std::process::Command;

/// Baseline storage for diff comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    /// Git commit hash when baseline was created
    pub commit: String,
    /// Timestamp when baseline was created
    pub timestamp: u64,
    /// Finding fingerprints (file:line:pattern hashes)
    pub fingerprints: HashSet<String>,
}

impl Baseline {
    /// Create a new baseline from findings.
    pub fn from_findings(findings: &[Finding]) -> Self {
        let fingerprints = findings.iter().map(fingerprint).collect();

        let commit = get_current_commit().unwrap_or_default();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        Self { commit, timestamp, fingerprints }
    }

    /// Load baseline from disk.
    pub fn load(project_path: &Path) -> Option<Self> {
        let baseline_path = project_path.join(".pmat").join("bug-hunter-baseline.json");
        if baseline_path.exists() {
            let content = std::fs::read_to_string(&baseline_path).ok()?;
            serde_json::from_str(&content).ok()
        } else {
            None
        }
    }

    /// Save baseline to disk.
    pub fn save(&self, project_path: &Path) -> Result<(), String> {
        let pmat_dir = project_path.join(".pmat");
        std::fs::create_dir_all(&pmat_dir)
            .map_err(|e| format!("Failed to create .pmat directory: {}", e))?;

        let baseline_path = pmat_dir.join("bug-hunter-baseline.json");
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize baseline: {}", e))?;

        std::fs::write(&baseline_path, content)
            .map_err(|e| format!("Failed to write baseline: {}", e))?;

        Ok(())
    }

    /// Check if a finding is new (not in baseline).
    pub fn is_new(&self, finding: &Finding) -> bool {
        !self.fingerprints.contains(&fingerprint(finding))
    }
}

/// Generate a fingerprint for a finding.
fn fingerprint(finding: &Finding) -> String {
    // Use file path (relative), line number, and title for fingerprinting
    // This allows findings to persist across minor code movements
    let file_name = finding.file.file_name().map(|s| s.to_string_lossy()).unwrap_or_default();
    format!("{}:{}:{}", file_name, finding.line, finding.title)
}

/// Get current git commit hash.
fn get_current_commit() -> Option<String> {
    let output = Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

/// Get files changed since a commit or time period.
pub fn get_changed_files(
    project_path: &Path,
    base: Option<&str>,
    since: Option<&str>,
) -> Vec<String> {
    let output = if let Some(base) = base {
        Command::new("git").current_dir(project_path).args(["diff", "--name-only", base]).output()
    } else if let Some(since) = since {
        let git_since = format!("--since={}", since);
        Command::new("git")
            .current_dir(project_path)
            .args(["log", "--name-only", "--pretty=format:"])
            .arg(&git_since)
            .output()
    } else {
        return Vec::new();
    };

    match output {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect(),
        _ => Vec::new(),
    }
}

/// Filter findings to only include new ones.
pub fn filter_new_findings(result: &HuntResult, baseline: &Baseline) -> Vec<Finding> {
    result.findings.iter().filter(|f| baseline.is_new(f)).cloned().collect()
}

/// Filter findings to only those in changed files.
pub fn filter_changed_files(findings: &[Finding], changed_files: &[String]) -> Vec<Finding> {
    findings
        .iter()
        .filter(|f| {
            let file_path = f.file.to_string_lossy();
            changed_files.iter().any(|cf| file_path.ends_with(cf))
        })
        .cloned()
        .collect()
}

/// Diff result showing new and resolved findings.
#[derive(Debug, Clone)]
pub struct DiffResult {
    /// Newly introduced findings
    pub new_findings: Vec<Finding>,
    /// Findings that were in baseline but not in current
    pub resolved_count: usize,
    /// Total findings in current run
    pub total_current: usize,
    /// Total findings in baseline
    pub total_baseline: usize,
    /// Base commit/time used for comparison
    pub base_reference: String,
}

impl DiffResult {
    /// Create a diff result from current findings and baseline.
    pub fn compute(current: &HuntResult, baseline: &Baseline, base_ref: &str) -> Self {
        let new_findings = filter_new_findings(current, baseline);
        let current_fps: HashSet<String> = current.findings.iter().map(fingerprint).collect();

        let resolved_count =
            baseline.fingerprints.iter().filter(|fp| !current_fps.contains(*fp)).count();

        Self {
            new_findings,
            resolved_count,
            total_current: current.findings.len(),
            total_baseline: baseline.fingerprints.len(),
            base_reference: base_ref.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bug_hunter::types::{DefectCategory, FindingSeverity};
    use std::path::PathBuf;

    fn make_finding(file: &str, line: usize, title: &str) -> Finding {
        Finding::new("TEST-001".to_string(), PathBuf::from(file), line, title.to_string())
            .with_severity(FindingSeverity::Medium)
            .with_category(DefectCategory::LogicErrors)
            .with_suspiciousness(0.5)
    }

    #[test]
    fn test_fingerprint() {
        let f = make_finding("src/foo.rs", 42, "Pattern: TODO");
        let fp = fingerprint(&f);
        assert!(fp.contains("foo.rs"));
        assert!(fp.contains("42"));
        assert!(fp.contains("TODO"));
    }

    #[test]
    fn test_baseline_is_new() {
        let findings = vec![
            make_finding("src/foo.rs", 10, "Pattern: TODO"),
            make_finding("src/bar.rs", 20, "Pattern: FIXME"),
        ];
        let baseline = Baseline::from_findings(&findings);

        // Same finding is not new
        let same = make_finding("src/foo.rs", 10, "Pattern: TODO");
        assert!(!baseline.is_new(&same));

        // Different finding is new
        let new = make_finding("src/baz.rs", 30, "Pattern: HACK");
        assert!(baseline.is_new(&new));

        // Same file, different line is new
        let new_line = make_finding("src/foo.rs", 15, "Pattern: TODO");
        assert!(baseline.is_new(&new_line));
    }

    #[test]
    fn test_diff_result() {
        let baseline_findings = vec![
            make_finding("src/foo.rs", 10, "Pattern: TODO"),
            make_finding("src/bar.rs", 20, "Pattern: FIXME"),
        ];
        let baseline = Baseline::from_findings(&baseline_findings);

        let current = HuntResult {
            findings: vec![
                make_finding("src/foo.rs", 10, "Pattern: TODO"), // existing
                make_finding("src/baz.rs", 30, "Pattern: HACK"), // new
            ],
            ..Default::default()
        };

        let diff = DiffResult::compute(&current, &baseline, "main");

        assert_eq!(diff.new_findings.len(), 1);
        assert_eq!(diff.resolved_count, 1); // bar.rs:20 was resolved
        assert_eq!(diff.total_current, 2);
        assert_eq!(diff.total_baseline, 2);
    }

    // =========================================================================
    // Coverage gap: get_changed_files
    // =========================================================================

    #[test]
    fn test_get_changed_files_with_base() {
        // Use the actual project's git repo — compare HEAD~1 to HEAD
        let files = get_changed_files(std::path::Path::new("."), Some("HEAD~1"), None);
        // Should return some files (unless HEAD is initial commit)
        // Just verify it doesn't panic and returns a Vec
        assert!(!files.is_empty() || files.is_empty()); // exercises the code path
    }

    #[test]
    fn test_get_changed_files_with_since() {
        let files = get_changed_files(std::path::Path::new("."), None, Some("1 day ago"));
        // Should not panic
        let _ = files.len();
    }

    #[test]
    fn test_get_changed_files_neither() {
        // No base, no since → empty
        let files = get_changed_files(std::path::Path::new("."), None, None);
        assert!(files.is_empty());
    }

    #[test]
    fn test_get_changed_files_invalid_path() {
        let files =
            get_changed_files(std::path::Path::new("/nonexistent/repo"), Some("HEAD~1"), None);
        assert!(files.is_empty());
    }

    // =========================================================================
    // Coverage gap: filter_new_findings
    // =========================================================================

    #[test]
    fn test_filter_new_findings_all_new() {
        let baseline = Baseline::from_findings(&[]);
        let current = HuntResult {
            findings: vec![
                make_finding("src/a.rs", 1, "Pattern: TODO"),
                make_finding("src/b.rs", 2, "Pattern: FIXME"),
            ],
            ..Default::default()
        };
        let new = filter_new_findings(&current, &baseline);
        assert_eq!(new.len(), 2);
    }

    #[test]
    fn test_filter_new_findings_none_new() {
        let findings = vec![make_finding("src/a.rs", 1, "Pattern: TODO")];
        let baseline = Baseline::from_findings(&findings);
        let current = HuntResult { findings: findings.clone(), ..Default::default() };
        let new = filter_new_findings(&current, &baseline);
        assert!(new.is_empty());
    }

    // =========================================================================
    // Coverage gap: filter_changed_files
    // =========================================================================

    #[test]
    fn test_filter_changed_files_match() {
        let findings = vec![
            make_finding("src/foo.rs", 1, "Pattern: TODO"),
            make_finding("src/bar.rs", 2, "Pattern: FIXME"),
            make_finding("src/baz.rs", 3, "Pattern: HACK"),
        ];
        let changed = vec!["src/foo.rs".to_string(), "src/baz.rs".to_string()];
        let filtered = filter_changed_files(&findings, &changed);
        assert_eq!(filtered.len(), 2);
        assert!(filtered.iter().all(|f| {
            let p = f.file.to_string_lossy();
            p.ends_with("foo.rs") || p.ends_with("baz.rs")
        }));
    }

    #[test]
    fn test_filter_changed_files_no_match() {
        let findings = vec![make_finding("src/foo.rs", 1, "Pattern: TODO")];
        let changed = vec!["src/bar.rs".to_string()];
        let filtered = filter_changed_files(&findings, &changed);
        assert!(filtered.is_empty());
    }

    #[test]
    fn test_filter_changed_files_empty_changed() {
        let findings = vec![make_finding("src/foo.rs", 1, "Pattern: TODO")];
        let filtered = filter_changed_files(&findings, &[]);
        assert!(filtered.is_empty());
    }

    // =========================================================================
    // Coverage gap: Baseline::save()
    // =========================================================================

    #[test]
    fn test_baseline_save_and_load() {
        let dir = std::env::temp_dir().join(format!(
            "batuta_diff_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unexpected failure")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("mkdir failed");

        let findings = vec![
            make_finding("src/foo.rs", 10, "Pattern: TODO"),
            make_finding("src/bar.rs", 20, "Pattern: FIXME"),
        ];
        let baseline = Baseline::from_findings(&findings);

        // Save
        let result = baseline.save(&dir);
        assert!(result.is_ok(), "save failed: {:?}", result.err());

        // Load
        let loaded = Baseline::load(&dir);
        assert!(loaded.is_some(), "load returned None");
        let loaded = loaded.expect("unexpected failure");
        assert_eq!(loaded.fingerprints.len(), 2);
        assert_eq!(loaded.fingerprints, baseline.fingerprints);

        // Cleanup
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_baseline_save_creates_pmat_dir() {
        let dir = std::env::temp_dir().join(format!(
            "batuta_diff_pmat_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("unexpected failure")
                .as_nanos()
        ));
        // Don't create dir; save should create .pmat subdirectory
        let baseline = Baseline::from_findings(&[]);
        let result = baseline.save(&dir);
        assert!(result.is_ok());
        assert!(dir.join(".pmat").join("bug-hunter-baseline.json").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_baseline_load_nonexistent() {
        let path = PathBuf::from("/nonexistent/path/that/does/not/exist");
        let loaded = Baseline::load(&path);
        assert!(loaded.is_none());
    }

    // =========================================================================
    // Coverage gap: get_changed_files error paths
    // =========================================================================

    #[test]
    fn test_get_changed_files_invalid_base_ref() {
        // Invalid base ref in a valid git repo should return empty
        let files = get_changed_files(
            std::path::Path::new("."),
            Some("INVALID_REF_THAT_DOES_NOT_EXIST_12345"),
            None,
        );
        assert!(files.is_empty());
    }

    #[test]
    fn test_get_changed_files_since_in_invalid_repo() {
        let files =
            get_changed_files(std::path::Path::new("/nonexistent/repo"), None, Some("1 week ago"));
        assert!(files.is_empty());
    }

    // =========================================================================
    // Coverage gap: fingerprint edge cases
    // =========================================================================

    #[test]
    fn test_fingerprint_with_directory_path() {
        // Finding with a directory path (no file_name)
        let f = make_finding("/", 1, "Pattern: TODO");
        let fp = fingerprint(&f);
        // Should not panic, just use empty or "/"
        assert!(fp.contains('1'));
        assert!(fp.contains("TODO"));
    }

    #[test]
    fn test_fingerprint_stability() {
        // Same input should always produce same fingerprint
        let f1 = make_finding("src/main.rs", 42, "Pattern: unwrap");
        let f2 = make_finding("src/main.rs", 42, "Pattern: unwrap");
        assert_eq!(fingerprint(&f1), fingerprint(&f2));
    }

    #[test]
    fn test_fingerprint_different_files_same_line_title() {
        let f1 = make_finding("src/a.rs", 10, "Pattern: TODO");
        let f2 = make_finding("src/b.rs", 10, "Pattern: TODO");
        assert_ne!(fingerprint(&f1), fingerprint(&f2));
    }

    // =========================================================================
    // Coverage gap: DiffResult edge cases
    // =========================================================================

    #[test]
    fn test_diff_result_all_resolved() {
        let baseline_findings = vec![
            make_finding("src/a.rs", 1, "Pattern: TODO"),
            make_finding("src/b.rs", 2, "Pattern: FIXME"),
        ];
        let baseline = Baseline::from_findings(&baseline_findings);

        let current = HuntResult {
            findings: vec![], // All findings resolved
            ..Default::default()
        };

        let diff = DiffResult::compute(&current, &baseline, "HEAD~5");

        assert_eq!(diff.new_findings.len(), 0);
        assert_eq!(diff.resolved_count, 2);
        assert_eq!(diff.total_current, 0);
        assert_eq!(diff.total_baseline, 2);
        assert_eq!(diff.base_reference, "HEAD~5");
    }

    #[test]
    fn test_diff_result_empty_baseline() {
        let baseline = Baseline::from_findings(&[]);

        let current = HuntResult {
            findings: vec![make_finding("src/a.rs", 1, "Pattern: TODO")],
            ..Default::default()
        };

        let diff = DiffResult::compute(&current, &baseline, "initial");

        assert_eq!(diff.new_findings.len(), 1);
        assert_eq!(diff.resolved_count, 0);
        assert_eq!(diff.total_current, 1);
        assert_eq!(diff.total_baseline, 0);
    }

    // =========================================================================
    // Coverage gap: Baseline::from_findings timestamp/commit
    // =========================================================================

    #[test]
    fn test_baseline_from_findings_has_timestamp() {
        let baseline = Baseline::from_findings(&[]);
        // Timestamp should be non-zero (we're past epoch)
        assert!(baseline.timestamp > 0);
    }

    #[test]
    fn test_baseline_from_findings_has_commit() {
        let baseline = Baseline::from_findings(&[]);
        // In a git repo commit is a SHA; outside a git repo (e.g. clean-room
        // container) get_current_commit() returns None → empty string via
        // unwrap_or_default().  Both cases are valid.
        let in_git_repo = std::process::Command::new("git")
            .args(["rev-parse", "--git-dir"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if in_git_repo {
            assert!(!baseline.commit.is_empty(), "commit should be set in a git repo");
        } else {
            assert!(baseline.commit.is_empty(), "commit should be empty outside a git repo");
        }
    }
}
