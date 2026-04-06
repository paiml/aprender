//! Commit-level feature extraction for defect prediction
//!
//! Implements feature extraction from git commits based on organizational
//! intelligence analysis (D'Ambros et al. 2012, Zimmermann et al. 2009).
//!
//! # Feature Vector (8-dim)
//!
//! 1. **lines_added**: Total lines added in commit
//! 2. **lines_deleted**: Total lines removed
//! 3. **files_changed**: Number of files modified
//! 4. **churn_ratio**: added / (added + deleted + 1)
//! 5. **has_test_changes**: Whether test files were modified
//! 6. **complexity_delta**: Estimated cyclomatic complexity change
//! 7. **author_experience**: Author's commit count (normalized)
//! 8. **days_since_last_change**: Time since last commit to same files

use std::collections::HashMap;

/// 8-dimensional feature vector for commit-level defect prediction
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CommitFeatures {
    /// Total lines added in the commit
    pub lines_added: u32,
    /// Total lines deleted in the commit
    pub lines_deleted: u32,
    /// Number of files changed
    pub files_changed: u32,
    /// Churn ratio: added / (added + deleted + 1)
    pub churn_ratio: f32,
    /// Whether any test files were modified
    pub has_test_changes: bool,
    /// Estimated cyclomatic complexity change
    pub complexity_delta: f32,
    /// Author's normalized experience (0.0 = new, 1.0 = experienced)
    pub author_experience: f32,
    /// Days since last change to affected files
    pub days_since_last_change: f32,
}

impl CommitFeatures {
    /// Convert to 8-dimensional feature array for ML models
    #[must_use]
    pub fn to_array(&self) -> [f32; 8] {
        [
            self.lines_added as f32,
            self.lines_deleted as f32,
            self.files_changed as f32,
            self.churn_ratio,
            if self.has_test_changes { 1.0 } else { 0.0 },
            self.complexity_delta,
            self.author_experience,
            self.days_since_last_change,
        ]
    }

    /// Create from a feature array
    #[must_use]
    #[allow(clippy::cast_sign_loss)]
    pub fn from_array(arr: [f32; 8]) -> Self {
        Self {
            lines_added: arr[0].max(0.0) as u32,
            lines_deleted: arr[1].max(0.0) as u32,
            files_changed: arr[2].max(0.0) as u32,
            churn_ratio: arr[3],
            has_test_changes: arr[4] > 0.5,
            complexity_delta: arr[5],
            author_experience: arr[6],
            days_since_last_change: arr[7],
        }
    }

    /// Normalize features for ML models
    ///
    /// Uses the provided statistics to z-score normalize numeric features.
    #[must_use]
    pub fn normalize(&self, stats: &FeatureStats) -> [f32; 8] {
        let raw = self.to_array();
        let mut normalized = [0.0f32; 8];

        for (i, &val) in raw.iter().enumerate() {
            if stats.std[i] > f32::EPSILON {
                normalized[i] = (val - stats.mean[i]) / stats.std[i];
            } else {
                normalized[i] = 0.0;
            }
        }

        normalized
    }
}

/// Statistics for feature normalization
#[derive(Debug, Clone, Default)]
pub struct FeatureStats {
    /// Mean values for each feature dimension
    pub mean: [f32; 8],
    /// Standard deviation for each feature dimension
    pub std: [f32; 8],
}

impl FeatureStats {
    /// Compute statistics from a collection of features
    #[must_use]
    pub fn from_features(features: &[CommitFeatures]) -> Self {
        if features.is_empty() {
            return Self::default();
        }

        let n = features.len() as f32;

        // Compute means
        let mut mean = [0.0f32; 8];
        for f in features {
            let arr = f.to_array();
            for (i, &val) in arr.iter().enumerate() {
                mean[i] += val;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        // Compute standard deviations
        let mut std = [0.0f32; 8];
        for f in features {
            let arr = f.to_array();
            for (i, &val) in arr.iter().enumerate() {
                let diff = val - mean[i];
                std[i] += diff * diff;
            }
        }
        for s in &mut std {
            *s = (*s / n).sqrt();
        }

        Self { mean, std }
    }
}

/// Extract commit features from a git diff output
#[derive(Debug, Default)]
pub struct CommitFeatureExtractor {
    /// Historical author commit counts
    author_commits: HashMap<String, u32>,
    /// Last modification timestamps by file
    file_last_modified: HashMap<String, f64>,
    /// Total commits seen (for normalization)
    total_commits: u32,
}

impl CommitFeatureExtractor {
    /// Create a new feature extractor
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Extract features from a diff string
    ///
    /// # Arguments
    ///
    /// * `diff` - Git diff output (unified format)
    /// * `author` - Commit author name
    /// * `timestamp` - Commit timestamp (Unix epoch seconds)
    #[must_use]
    pub fn extract(&mut self, diff: &str, author: &str, timestamp: f64) -> CommitFeatures {
        let mut features = CommitFeatures::default();

        // Parse diff statistics
        let (added, deleted, files) = self.parse_diff_stats(diff);
        features.lines_added = added;
        features.lines_deleted = deleted;
        features.files_changed = files;

        // Compute churn ratio
        let total = added + deleted;
        features.churn_ratio = if total > 0 {
            added as f32 / (total as f32 + 1.0)
        } else {
            0.5
        };

        // Check for test changes
        features.has_test_changes = self.detect_test_changes(diff);

        // Estimate complexity delta
        features.complexity_delta = self.estimate_complexity_delta(diff);

        // Author experience
        let author_count = self.author_commits.entry(author.to_string()).or_insert(0);
        *author_count += 1;
        self.total_commits += 1;

        // Normalize experience: log scale, capped at 1.0
        features.author_experience = if self.total_commits > 0 {
            ((*author_count as f32).ln() / (self.total_commits as f32).ln().max(1.0)).min(1.0)
        } else {
            0.0
        };

        // Days since last change to affected files
        let affected_files = self.extract_affected_files(diff);
        let mut min_days = f64::MAX;
        let seconds_per_day = 86400.0;

        for file in &affected_files {
            if let Some(&last_mod) = self.file_last_modified.get(file) {
                let days = (timestamp - last_mod) / seconds_per_day;
                if days < min_days && days >= 0.0 {
                    min_days = days;
                }
            }
            self.file_last_modified.insert(file.clone(), timestamp);
        }

        features.days_since_last_change = if min_days >= f64::MAX - 1.0 {
            365.0 // Default for new files
        } else {
            (min_days as f32).min(365.0)
        };

        features
    }

    /// Parse diff statistics (lines added/deleted, files changed)
    fn parse_diff_stats(&self, diff: &str) -> (u32, u32, u32) {
        let mut added = 0u32;
        let mut deleted = 0u32;
        let mut files = 0u32;

        for line in diff.lines() {
            if line.starts_with("diff --git") || line.starts_with("--- ") {
                if line.starts_with("diff --git") {
                    files += 1;
                }
            } else if line.starts_with('+') && !line.starts_with("+++") {
                added += 1;
            } else if line.starts_with('-') && !line.starts_with("---") {
                deleted += 1;
            }
        }

        (added, deleted, files.max(1))
    }

    /// Detect if test files were changed
    fn detect_test_changes(&self, diff: &str) -> bool {
        for line in diff.lines() {
            if line.starts_with("diff --git")
                || line.starts_with("--- ")
                || line.starts_with("+++ ")
            {
                let lower = line.to_lowercase();
                if lower.contains("test")
                    || lower.contains("spec")
                    || lower.contains("_test.")
                    || lower.contains(".test.")
                {
                    return true;
                }
            }
        }
        false
    }

    /// Estimate cyclomatic complexity change from diff
    fn estimate_complexity_delta(&self, diff: &str) -> f32 {
        let mut delta = 0i32;

        for line in diff.lines() {
            let trimmed = line.trim();
            let is_addition = line.starts_with('+') && !line.starts_with("+++");
            let is_deletion = line.starts_with('-') && !line.starts_with("---");

            // Count control flow keywords
            let control_flow = ["if ", "elif ", "else:", "for ", "while ", "match ", "case "];
            for kw in control_flow {
                if trimmed.contains(kw) {
                    if is_addition {
                        delta += 1;
                    } else if is_deletion {
                        delta -= 1;
                    }
                }
            }
        }

        delta as f32
    }

    /// Extract list of affected files from diff
    fn extract_affected_files(&self, diff: &str) -> Vec<String> {
        let mut files = Vec::new();

        for line in diff.lines() {
            if line.starts_with("diff --git a/") {
                // diff --git a/src/foo.rs b/src/foo.rs
                if let Some(path) = line.strip_prefix("diff --git a/") {
                    if let Some(space_pos) = path.find(" b/") {
                        files.push(path[..space_pos].to_string());
                    }
                }
            } else if line.starts_with("+++ b/") {
                if let Some(path) = line.strip_prefix("+++ b/") {
                    if !files.contains(&path.to_string()) {
                        files.push(path.to_string());
                    }
                }
            }
        }

        files
    }

    /// Reset extractor state (for testing or batch processing)
    pub fn reset(&mut self) {
        self.author_commits.clear();
        self.file_last_modified.clear();
        self.total_commits = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_features_default() {
        let features = CommitFeatures::default();
        assert_eq!(features.lines_added, 0);
        assert_eq!(features.lines_deleted, 0);
        assert_eq!(features.files_changed, 0);
    }

    #[test]
    fn test_commit_features_to_array() {
        let features = CommitFeatures {
            lines_added: 10,
            lines_deleted: 5,
            files_changed: 2,
            churn_ratio: 0.67,
            has_test_changes: true,
            complexity_delta: 3.0,
            author_experience: 0.5,
            days_since_last_change: 7.0,
        };

        let arr = features.to_array();
        assert_eq!(arr[0], 10.0);
        assert_eq!(arr[1], 5.0);
        assert_eq!(arr[2], 2.0);
        assert!((arr[3] - 0.67).abs() < 0.01);
        assert_eq!(arr[4], 1.0); // has_test_changes = true
        assert_eq!(arr[5], 3.0);
        assert_eq!(arr[6], 0.5);
        assert_eq!(arr[7], 7.0);
    }

    #[test]
    fn test_commit_features_from_array() {
        let arr = [10.0, 5.0, 2.0, 0.67, 1.0, 3.0, 0.5, 7.0];
        let features = CommitFeatures::from_array(arr);

        assert_eq!(features.lines_added, 10);
        assert_eq!(features.lines_deleted, 5);
        assert_eq!(features.files_changed, 2);
        assert!(features.has_test_changes);
    }

    #[test]
    fn test_feature_stats_from_features() {
        let features = vec![
            CommitFeatures {
                lines_added: 10,
                lines_deleted: 5,
                ..Default::default()
            },
            CommitFeatures {
                lines_added: 20,
                lines_deleted: 15,
                ..Default::default()
            },
        ];

        let stats = FeatureStats::from_features(&features);

        // Mean of lines_added: (10 + 20) / 2 = 15
        assert!((stats.mean[0] - 15.0).abs() < 0.01);
        // Mean of lines_deleted: (5 + 15) / 2 = 10
        assert!((stats.mean[1] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_feature_stats_empty() {
        let features: Vec<CommitFeatures> = vec![];
        let stats = FeatureStats::from_features(&features);
        assert_eq!(stats.mean[0], 0.0);
        assert_eq!(stats.std[0], 0.0);
    }

    #[test]
    fn test_normalize_features() {
        let features = CommitFeatures {
            lines_added: 20,
            ..Default::default()
        };

        let stats = FeatureStats {
            mean: [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            std: [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        };

        let normalized = features.normalize(&stats);
        // (20 - 10) / 5 = 2.0
        assert!((normalized[0] - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_extractor_parse_diff() {
        let mut extractor = CommitFeatureExtractor::new();

        let diff = r#"diff --git a/src/main.rs b/src/main.rs
--- a/src/main.rs
+++ b/src/main.rs
@@ -1,3 +1,5 @@
 fn main() {
+    let x = 1;
+    let y = 2;
-    println!("hello");
 }
"#;

        let features = extractor.extract(diff, "alice", 1700000000.0);

        assert_eq!(features.lines_added, 2);
        assert_eq!(features.lines_deleted, 1);
        assert_eq!(features.files_changed, 1);
        assert!(!features.has_test_changes);
    }

    #[test]
    fn test_extractor_test_changes() {
        let mut extractor = CommitFeatureExtractor::new();

        let diff = r#"diff --git a/tests/test_main.rs b/tests/test_main.rs
+++ b/tests/test_main.rs
+#[test]
+fn test_foo() {}
"#;

        let features = extractor.extract(diff, "bob", 1700000000.0);
        assert!(features.has_test_changes);
    }

    #[test]
    fn test_extractor_complexity_delta() {
        let mut extractor = CommitFeatureExtractor::new();

        let diff = r#"diff --git a/src/lib.rs b/src/lib.rs
+if x > 0 {
+    for i in 0..10 {
+        while running {
-    println!("simple");
"#;

        let features = extractor.extract(diff, "carol", 1700000000.0);
        // Added: if, for, while = +3
        // No deletions with control flow
        assert!((features.complexity_delta - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_extractor_author_experience() {
        let mut extractor = CommitFeatureExtractor::new();

        let diff = "diff --git a/foo.rs b/foo.rs\n+line";

        // First commit
        let f1 = extractor.extract(diff, "alice", 1700000000.0);
        assert!(f1.author_experience >= 0.0);

        // Second commit
        let f2 = extractor.extract(diff, "alice", 1700001000.0);
        assert!(f2.author_experience >= f1.author_experience);

        // Different author starts fresh
        let f3 = extractor.extract(diff, "bob", 1700002000.0);
        assert!(f3.author_experience <= f2.author_experience);
    }

    #[test]
    fn test_extractor_days_since_last_change() {
        let mut extractor = CommitFeatureExtractor::new();

        let diff = "diff --git a/src/foo.rs b/src/foo.rs\n+++ b/src/foo.rs\n+line";

        // First commit to file
        let f1 = extractor.extract(diff, "alice", 1700000000.0);
        assert!((f1.days_since_last_change - 365.0).abs() < 0.01); // Default for new file

        // Second commit 7 days later
        let seconds_per_day = 86400.0;
        let f2 = extractor.extract(diff, "alice", 1700000000.0 + 7.0 * seconds_per_day);
        assert!((f2.days_since_last_change - 7.0).abs() < 0.01);
    }

    #[test]
    fn test_extractor_reset() {
        let mut extractor = CommitFeatureExtractor::new();

        let diff = "diff --git a/foo.rs b/foo.rs\n+line";
        extractor.extract(diff, "alice", 1700000000.0);

        assert!(extractor.total_commits > 0);

        extractor.reset();

        assert_eq!(extractor.total_commits, 0);
        assert!(extractor.author_commits.is_empty());
    }

    #[test]
    fn test_extractor_churn_ratio() {
        let mut extractor = CommitFeatureExtractor::new();

        // All additions
        let diff1 = "diff --git a/f.rs b/f.rs\n+a\n+b\n+c";
        let f1 = extractor.extract(diff1, "alice", 1.0);
        assert!(f1.churn_ratio > 0.5); // More additions than deletions

        extractor.reset();

        // All deletions
        let diff2 = "diff --git a/f.rs b/f.rs\n-a\n-b\n-c";
        let f2 = extractor.extract(diff2, "alice", 1.0);
        assert!(f2.churn_ratio < 0.5); // More deletions than additions
    }

    #[test]
    fn test_extractor_empty_diff() {
        let mut extractor = CommitFeatureExtractor::new();
        let features = extractor.extract("", "alice", 1.0);

        assert_eq!(features.lines_added, 0);
        assert_eq!(features.lines_deleted, 0);
        assert_eq!(features.files_changed, 1); // At least 1
    }

    #[test]
    fn test_commit_features_debug() {
        let features = CommitFeatures::default();
        let debug = format!("{features:?}");
        assert!(debug.contains("CommitFeatures"));
    }

    #[test]
    fn test_feature_stats_debug() {
        let stats = FeatureStats::default();
        let debug = format!("{stats:?}");
        assert!(debug.contains("FeatureStats"));
    }

    #[test]
    fn test_extractor_debug() {
        let extractor = CommitFeatureExtractor::new();
        let debug = format!("{extractor:?}");
        assert!(debug.contains("CommitFeatureExtractor"));
    }

    #[test]
    fn test_commit_features_clone_eq() {
        let f1 = CommitFeatures {
            lines_added: 10,
            ..Default::default()
        };
        let f2 = f1.clone();
        assert_eq!(f1, f2);
    }
}
