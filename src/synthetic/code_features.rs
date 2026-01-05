//! Code Feature Extraction for Commit-Level Analysis.
//!
//! Extracts 8-dimensional feature vectors from code commits for
//! defect prediction and code quality analysis. Based on D'Ambros et al. (2012)
//! benchmark methodology for software defect prediction.
//!
//! # Feature Vector
//!
//! The `CommitFeatures` struct contains:
//! 1. `defect_category` - Predicted defect category (0-255)
//! 2. `files_changed` - Number of files modified
//! 3. `lines_added` - Lines of code added
//! 4. `lines_deleted` - Lines of code removed
//! 5. `complexity_delta` - Change in cyclomatic complexity
//! 6. `timestamp` - Unix timestamp of commit
//! 7. `hour_of_day` - Hour when commit was made (0-23)
//! 8. `day_of_week` - Day of week (0=Sunday, 6=Saturday)
//!
//! # Example
//!
//! ```
//! use aprender::synthetic::code_features::{CodeFeatureExtractor, CommitFeatures, CommitDiff};
//!
//! let extractor = CodeFeatureExtractor::new();
//!
//! let diff = CommitDiff {
//!     files_changed: 3,
//!     lines_added: 150,
//!     lines_deleted: 50,
//!     timestamp: 1700000000,
//!     message: "fix: resolve memory leak".to_string(),
//! };
//!
//! let features = extractor.extract(&diff);
//! assert_eq!(features.files_changed, 3.0);
//! ```
//!
//! # References
//!
//! - D'Ambros et al. (2012). "Evaluating Defect Prediction Approaches"

use std::collections::HashSet;

/// Commit-level features for defect prediction (8-dimensional).
///
/// This structure matches the org-intel `CommitFeatures` format for
/// compatibility with defect prediction pipelines.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CommitFeatures {
    /// Predicted defect category (0-255).
    /// Categories include: 0=clean, 1=bug, 2=security, 3=perf, etc.
    pub defect_category: u8,
    /// Number of files changed in the commit.
    pub files_changed: f32,
    /// Total lines of code added.
    pub lines_added: f32,
    /// Total lines of code deleted.
    pub lines_deleted: f32,
    /// Estimated change in cyclomatic complexity.
    pub complexity_delta: f32,
    /// Unix timestamp of the commit.
    pub timestamp: f64,
    /// Hour of day the commit was made (0-23).
    pub hour_of_day: u8,
    /// Day of week (0=Sunday, 6=Saturday).
    pub day_of_week: u8,
}

impl Default for CommitFeatures {
    fn default() -> Self {
        Self {
            defect_category: 0,
            files_changed: 0.0,
            lines_added: 0.0,
            lines_deleted: 0.0,
            complexity_delta: 0.0,
            timestamp: 0.0,
            hour_of_day: 12,
            day_of_week: 0,
        }
    }
}

impl CommitFeatures {
    /// Create new commit features with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert features to an 8-element f32 vector for ML pipelines.
    #[must_use]
    pub fn to_vec(&self) -> Vec<f32> {
        vec![
            f32::from(self.defect_category),
            self.files_changed,
            self.lines_added,
            self.lines_deleted,
            self.complexity_delta,
            self.timestamp as f32,
            f32::from(self.hour_of_day),
            f32::from(self.day_of_week),
        ]
    }

    /// Create features from a raw vector.
    ///
    /// # Panics
    ///
    /// Panics if vector has fewer than 8 elements.
    #[must_use]
    pub fn from_vec(v: &[f32]) -> Self {
        assert!(v.len() >= 8, "Feature vector must have at least 8 elements");
        Self {
            defect_category: v[0] as u8,
            files_changed: v[1],
            lines_added: v[2],
            lines_deleted: v[3],
            complexity_delta: v[4],
            timestamp: f64::from(v[5]),
            hour_of_day: v[6] as u8,
            day_of_week: v[7] as u8,
        }
    }

    /// Calculate churn (lines added + deleted).
    #[must_use]
    pub fn churn(&self) -> f32 {
        self.lines_added + self.lines_deleted
    }

    /// Calculate net change (lines added - deleted).
    #[must_use]
    pub fn net_change(&self) -> f32 {
        self.lines_added - self.lines_deleted
    }

    /// Check if commit is a fix based on category.
    #[must_use]
    pub fn is_fix(&self) -> bool {
        self.defect_category == 1
    }
}

/// Input data for feature extraction - minimal commit diff information.
#[derive(Debug, Clone, Default)]
pub struct CommitDiff {
    /// Number of files changed.
    pub files_changed: u32,
    /// Lines added.
    pub lines_added: u32,
    /// Lines deleted.
    pub lines_deleted: u32,
    /// Unix timestamp.
    pub timestamp: u64,
    /// Commit message.
    pub message: String,
}

impl CommitDiff {
    /// Create a new commit diff.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set files changed.
    #[must_use]
    pub fn with_files_changed(mut self, n: u32) -> Self {
        self.files_changed = n;
        self
    }

    /// Builder: set lines added.
    #[must_use]
    pub fn with_lines_added(mut self, n: u32) -> Self {
        self.lines_added = n;
        self
    }

    /// Builder: set lines deleted.
    #[must_use]
    pub fn with_lines_deleted(mut self, n: u32) -> Self {
        self.lines_deleted = n;
        self
    }

    /// Builder: set timestamp.
    #[must_use]
    pub fn with_timestamp(mut self, ts: u64) -> Self {
        self.timestamp = ts;
        self
    }

    /// Builder: set commit message.
    #[must_use]
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = msg.into();
        self
    }
}

/// Feature extractor for commit-level defect prediction.
///
/// Extracts 8-dimensional feature vectors from commit diffs,
/// suitable for training defect prediction models.
#[derive(Debug, Clone)]
pub struct CodeFeatureExtractor {
    /// Keywords indicating bug fixes.
    bug_keywords: HashSet<String>,
    /// Keywords indicating security issues.
    security_keywords: HashSet<String>,
    /// Keywords indicating performance changes.
    perf_keywords: HashSet<String>,
    /// Keywords indicating refactoring.
    refactor_keywords: HashSet<String>,
    /// Complexity estimation factor (lines per complexity point).
    complexity_factor: f32,
}

impl Default for CodeFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeFeatureExtractor {
    /// Create a new feature extractor with default keyword sets.
    #[must_use]
    pub fn new() -> Self {
        let bug_keywords: HashSet<_> = [
            "fix",
            "bug",
            "error",
            "issue",
            "crash",
            "fault",
            "defect",
            "problem",
            "wrong",
            "broken",
            "fail",
            "mistake",
            "incorrect",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        let security_keywords: HashSet<_> = [
            "security",
            "vulnerability",
            "cve",
            "exploit",
            "injection",
            "xss",
            "csrf",
            "auth",
            "permission",
            "sanitize",
            "escape",
            "unsafe",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        let perf_keywords: HashSet<_> = [
            "performance",
            "perf",
            "optimize",
            "speed",
            "fast",
            "slow",
            "memory",
            "cache",
            "efficient",
            "latency",
            "throughput",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        let refactor_keywords: HashSet<_> = [
            "refactor",
            "clean",
            "rename",
            "move",
            "reorganize",
            "restructure",
            "simplify",
            "extract",
            "inline",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect();

        Self {
            bug_keywords,
            security_keywords,
            perf_keywords,
            refactor_keywords,
            complexity_factor: 10.0, // ~10 lines per complexity point
        }
    }

    /// Set complexity estimation factor.
    #[must_use]
    pub fn with_complexity_factor(mut self, factor: f32) -> Self {
        self.complexity_factor = factor.max(1.0);
        self
    }

    /// Add custom bug keywords.
    pub fn add_bug_keywords(&mut self, keywords: &[&str]) {
        for kw in keywords {
            self.bug_keywords.insert((*kw).to_string());
        }
    }

    /// Add custom security keywords.
    pub fn add_security_keywords(&mut self, keywords: &[&str]) {
        for kw in keywords {
            self.security_keywords.insert((*kw).to_string());
        }
    }

    /// Extract features from a commit diff.
    #[must_use]
    pub fn extract(&self, diff: &CommitDiff) -> CommitFeatures {
        let defect_category = self.classify_commit(&diff.message);
        let complexity_delta = self.estimate_complexity_delta(diff);
        let (hour_of_day, day_of_week) = self.extract_time_features(diff.timestamp);

        CommitFeatures {
            defect_category,
            files_changed: diff.files_changed as f32,
            lines_added: diff.lines_added as f32,
            lines_deleted: diff.lines_deleted as f32,
            complexity_delta,
            timestamp: diff.timestamp as f64,
            hour_of_day,
            day_of_week,
        }
    }

    /// Extract features from multiple diffs.
    #[must_use]
    pub fn extract_batch(&self, diffs: &[CommitDiff]) -> Vec<CommitFeatures> {
        diffs.iter().map(|d| self.extract(d)).collect()
    }

    /// Classify commit based on message keywords.
    ///
    /// Returns:
    /// - 0: Clean/unknown
    /// - 1: Bug fix
    /// - 2: Security fix
    /// - 3: Performance improvement
    /// - 4: Refactoring
    fn classify_commit(&self, message: &str) -> u8 {
        let lower = message.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();

        // Check in priority order
        for word in &words {
            if self.security_keywords.contains(*word) {
                return 2; // Security
            }
        }

        for word in &words {
            if self.bug_keywords.contains(*word) {
                return 1; // Bug
            }
        }

        for word in &words {
            if self.perf_keywords.contains(*word) {
                return 3; // Performance
            }
        }

        for word in &words {
            if self.refactor_keywords.contains(*word) {
                return 4; // Refactor
            }
        }

        0 // Clean/unknown
    }

    /// Estimate complexity delta from diff statistics.
    ///
    /// Uses a simple heuristic: net lines changed / `complexity_factor`.
    /// Positive = complexity increased, negative = decreased.
    fn estimate_complexity_delta(&self, diff: &CommitDiff) -> f32 {
        let net = diff.lines_added as f32 - diff.lines_deleted as f32;
        net / self.complexity_factor
    }

    /// Extract time-based features from timestamp.
    ///
    /// Returns (`hour_of_day`, `day_of_week`).
    #[allow(clippy::unused_self)]
    fn extract_time_features(&self, timestamp: u64) -> (u8, u8) {
        // Simple calculation assuming UTC
        // Seconds since epoch
        let seconds_in_day: u64 = 86400;
        let seconds_in_hour: u64 = 3600;

        // Days since Thursday, Jan 1, 1970 (epoch was a Thursday)
        let days_since_epoch = timestamp / seconds_in_day;
        // Thursday = 4, so (days + 4) % 7 gives correct day
        let day_of_week = ((days_since_epoch + 4) % 7) as u8;

        // Hour of day
        let seconds_today = timestamp % seconds_in_day;
        let hour_of_day = (seconds_today / seconds_in_hour) as u8;

        (hour_of_day, day_of_week)
    }

    /// Normalize features to [0, 1] range using provided statistics.
    #[must_use]
    pub fn normalize(&self, features: &CommitFeatures, stats: &FeatureStats) -> CommitFeatures {
        CommitFeatures {
            defect_category: features.defect_category, // Categorical, don't normalize
            files_changed: Self::normalize_value(features.files_changed, stats.files_changed_max),
            lines_added: Self::normalize_value(features.lines_added, stats.lines_added_max),
            lines_deleted: Self::normalize_value(features.lines_deleted, stats.lines_deleted_max),
            complexity_delta: Self::normalize_value(
                features.complexity_delta,
                stats.complexity_max,
            ),
            timestamp: features.timestamp, // Keep raw timestamp
            hour_of_day: features.hour_of_day,
            day_of_week: features.day_of_week,
        }
    }

    /// Normalize a single value to [0, 1].
    fn normalize_value(value: f32, max: f32) -> f32 {
        if max <= 0.0 {
            0.0
        } else {
            (value / max).clamp(0.0, 1.0)
        }
    }
}

/// Statistics for feature normalization.
#[derive(Debug, Clone, Copy, Default)]
pub struct FeatureStats {
    /// Maximum files changed in dataset.
    pub files_changed_max: f32,
    /// Maximum lines added.
    pub lines_added_max: f32,
    /// Maximum lines deleted.
    pub lines_deleted_max: f32,
    /// Maximum absolute complexity delta.
    pub complexity_max: f32,
}

impl FeatureStats {
    /// Create stats from a set of features.
    #[must_use]
    pub fn from_features(features: &[CommitFeatures]) -> Self {
        let mut stats = Self::default();

        for f in features {
            stats.files_changed_max = stats.files_changed_max.max(f.files_changed);
            stats.lines_added_max = stats.lines_added_max.max(f.lines_added);
            stats.lines_deleted_max = stats.lines_deleted_max.max(f.lines_deleted);
            stats.complexity_max = stats.complexity_max.max(f.complexity_delta.abs());
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_commit_features_default() {
        let f = CommitFeatures::default();
        assert_eq!(f.defect_category, 0);
        assert!((f.files_changed - 0.0).abs() < f32::EPSILON);
        assert_eq!(f.hour_of_day, 12);
    }

    #[test]
    fn test_commit_features_to_vec() {
        let f = CommitFeatures {
            defect_category: 1,
            files_changed: 5.0,
            lines_added: 100.0,
            lines_deleted: 20.0,
            complexity_delta: 8.0,
            timestamp: 1700000000.0,
            hour_of_day: 14,
            day_of_week: 2,
        };

        let v = f.to_vec();
        assert_eq!(v.len(), 8);
        assert!((v[0] - 1.0).abs() < f32::EPSILON);
        assert!((v[1] - 5.0).abs() < f32::EPSILON);
        assert!((v[2] - 100.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_commit_features_from_vec() {
        let v = vec![2.0, 3.0, 50.0, 10.0, 4.0, 1700000000.0, 10.0, 5.0];
        let f = CommitFeatures::from_vec(&v);

        assert_eq!(f.defect_category, 2);
        assert!((f.files_changed - 3.0).abs() < f32::EPSILON);
        assert!((f.lines_added - 50.0).abs() < f32::EPSILON);
        assert_eq!(f.hour_of_day, 10);
        assert_eq!(f.day_of_week, 5);
    }

    #[test]
    fn test_commit_features_churn() {
        let f = CommitFeatures {
            lines_added: 100.0,
            lines_deleted: 50.0,
            ..Default::default()
        };
        assert!((f.churn() - 150.0).abs() < f32::EPSILON);
        assert!((f.net_change() - 50.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_commit_features_is_fix() {
        let fix = CommitFeatures {
            defect_category: 1,
            ..Default::default()
        };
        let clean = CommitFeatures::default();

        assert!(fix.is_fix());
        assert!(!clean.is_fix());
    }

    #[test]
    fn test_commit_diff_builder() {
        let diff = CommitDiff::new()
            .with_files_changed(3)
            .with_lines_added(100)
            .with_lines_deleted(50)
            .with_timestamp(1700000000)
            .with_message("fix: resolve bug");

        assert_eq!(diff.files_changed, 3);
        assert_eq!(diff.lines_added, 100);
        assert_eq!(diff.lines_deleted, 50);
        assert_eq!(diff.timestamp, 1700000000);
        assert_eq!(diff.message, "fix: resolve bug");
    }

    #[test]
    fn test_extractor_new() {
        let extractor = CodeFeatureExtractor::new();
        assert!(extractor.bug_keywords.contains("fix"));
        assert!(extractor.security_keywords.contains("vulnerability"));
        assert!(extractor.perf_keywords.contains("optimize"));
    }

    #[test]
    fn test_extractor_extract_basic() {
        let extractor = CodeFeatureExtractor::new();
        let diff = CommitDiff {
            files_changed: 3,
            lines_added: 150,
            lines_deleted: 50,
            timestamp: 1700000000,
            message: "Add new feature".to_string(),
        };

        let features = extractor.extract(&diff);

        assert!((features.files_changed - 3.0).abs() < f32::EPSILON);
        assert!((features.lines_added - 150.0).abs() < f32::EPSILON);
        assert!((features.lines_deleted - 50.0).abs() < f32::EPSILON);
        assert!((features.timestamp - 1700000000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_extractor_classify_bug() {
        let extractor = CodeFeatureExtractor::new();
        let diff = CommitDiff {
            message: "fix: resolve memory leak bug".to_string(),
            ..Default::default()
        };

        let features = extractor.extract(&diff);
        assert_eq!(features.defect_category, 1); // Bug
    }

    #[test]
    fn test_extractor_classify_security() {
        let extractor = CodeFeatureExtractor::new();
        let diff = CommitDiff {
            message: "security: patch CVE-2024-1234 vulnerability".to_string(),
            ..Default::default()
        };

        let features = extractor.extract(&diff);
        assert_eq!(features.defect_category, 2); // Security
    }

    #[test]
    fn test_extractor_classify_performance() {
        let extractor = CodeFeatureExtractor::new();
        let diff = CommitDiff {
            message: "perf: optimize database queries".to_string(),
            ..Default::default()
        };

        let features = extractor.extract(&diff);
        assert_eq!(features.defect_category, 3); // Performance
    }

    #[test]
    fn test_extractor_classify_refactor() {
        let extractor = CodeFeatureExtractor::new();
        let diff = CommitDiff {
            message: "refactor: clean up legacy code".to_string(),
            ..Default::default()
        };

        let features = extractor.extract(&diff);
        assert_eq!(features.defect_category, 4); // Refactor
    }

    #[test]
    fn test_extractor_classify_clean() {
        let extractor = CodeFeatureExtractor::new();
        let diff = CommitDiff {
            message: "Add new dashboard component".to_string(),
            ..Default::default()
        };

        let features = extractor.extract(&diff);
        assert_eq!(features.defect_category, 0); // Clean
    }

    #[test]
    fn test_extractor_complexity_delta() {
        let extractor = CodeFeatureExtractor::new().with_complexity_factor(10.0);
        let diff = CommitDiff {
            lines_added: 100,
            lines_deleted: 20,
            ..Default::default()
        };

        let features = extractor.extract(&diff);
        // (100 - 20) / 10 = 8.0
        assert!((features.complexity_delta - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extractor_time_features() {
        let extractor = CodeFeatureExtractor::new();
        // 1700000000 = Tuesday, November 14, 2023 22:13:20 UTC
        let diff = CommitDiff {
            timestamp: 1700000000,
            ..Default::default()
        };

        let features = extractor.extract(&diff);
        assert_eq!(features.hour_of_day, 22);
        assert_eq!(features.day_of_week, 2); // Tuesday
    }

    #[test]
    fn test_extractor_batch() {
        let extractor = CodeFeatureExtractor::new();
        let diffs = vec![
            CommitDiff {
                files_changed: 1,
                ..Default::default()
            },
            CommitDiff {
                files_changed: 2,
                ..Default::default()
            },
            CommitDiff {
                files_changed: 3,
                ..Default::default()
            },
        ];

        let features = extractor.extract_batch(&diffs);
        assert_eq!(features.len(), 3);
        assert!((features[0].files_changed - 1.0).abs() < f32::EPSILON);
        assert!((features[1].files_changed - 2.0).abs() < f32::EPSILON);
        assert!((features[2].files_changed - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_feature_stats_from_features() {
        let features = vec![
            CommitFeatures {
                files_changed: 5.0,
                lines_added: 100.0,
                lines_deleted: 20.0,
                complexity_delta: 8.0,
                ..Default::default()
            },
            CommitFeatures {
                files_changed: 10.0,
                lines_added: 200.0,
                lines_deleted: 50.0,
                complexity_delta: -5.0,
                ..Default::default()
            },
        ];

        let stats = FeatureStats::from_features(&features);
        assert!((stats.files_changed_max - 10.0).abs() < f32::EPSILON);
        assert!((stats.lines_added_max - 200.0).abs() < f32::EPSILON);
        assert!((stats.lines_deleted_max - 50.0).abs() < f32::EPSILON);
        assert!((stats.complexity_max - 8.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extractor_normalize() {
        let extractor = CodeFeatureExtractor::new();
        let features = CommitFeatures {
            files_changed: 5.0,
            lines_added: 100.0,
            lines_deleted: 25.0,
            complexity_delta: 4.0,
            ..Default::default()
        };
        let stats = FeatureStats {
            files_changed_max: 10.0,
            lines_added_max: 200.0,
            lines_deleted_max: 50.0,
            complexity_max: 8.0,
        };

        let normalized = extractor.normalize(&features, &stats);
        assert!((normalized.files_changed - 0.5).abs() < f32::EPSILON);
        assert!((normalized.lines_added - 0.5).abs() < f32::EPSILON);
        assert!((normalized.lines_deleted - 0.5).abs() < f32::EPSILON);
        assert!((normalized.complexity_delta - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extractor_normalize_zero_max() {
        let extractor = CodeFeatureExtractor::new();
        let features = CommitFeatures {
            files_changed: 5.0,
            ..Default::default()
        };
        let stats = FeatureStats::default(); // All zeros

        let normalized = extractor.normalize(&features, &stats);
        assert!((normalized.files_changed - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_extractor_add_keywords() {
        let mut extractor = CodeFeatureExtractor::new();
        extractor.add_bug_keywords(&["glitch", "oops"]);
        extractor.add_security_keywords(&["hack"]);

        assert!(extractor.bug_keywords.contains("glitch"));
        assert!(extractor.bug_keywords.contains("oops"));
        assert!(extractor.security_keywords.contains("hack"));
    }

    #[test]
    fn test_security_priority_over_bug() {
        let extractor = CodeFeatureExtractor::new();
        // Message contains both security and bug keywords
        let diff = CommitDiff {
            message: "fix security vulnerability bug".to_string(),
            ..Default::default()
        };

        let features = extractor.extract(&diff);
        // Security should take priority
        assert_eq!(features.defect_category, 2);
    }

    #[test]
    fn test_epoch_thursday() {
        let extractor = CodeFeatureExtractor::new();
        // Unix epoch (0) was Thursday, January 1, 1970
        let diff = CommitDiff {
            timestamp: 0,
            ..Default::default()
        };

        let features = extractor.extract(&diff);
        assert_eq!(features.day_of_week, 4); // Thursday
        assert_eq!(features.hour_of_day, 0);
    }
}
