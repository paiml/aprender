//! Visual snapshot testing.
//!
//! Per spec Section 6.2: Visual Regression Testing

/// Configuration for snapshot testing
#[derive(Debug, Clone)]
pub struct SnapshotConfig {
    /// Whether to update snapshots on mismatch
    pub update_snapshots: bool,
    /// Difference threshold (0.0-1.0)
    pub threshold: f64,
    /// Directory to store snapshots
    pub snapshot_dir: String,
}

impl Default for SnapshotConfig {
    fn default() -> Self {
        Self {
            update_snapshots: false,
            threshold: 0.01, // 1% difference allowed
            snapshot_dir: String::from("__snapshots__"),
        }
    }
}

impl SnapshotConfig {
    /// Set update mode
    #[must_use]
    pub const fn with_update(mut self, update: bool) -> Self {
        self.update_snapshots = update;
        self
    }

    /// Set threshold
    #[must_use]
    pub const fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set snapshot directory
    #[must_use]
    pub fn with_dir(mut self, dir: impl Into<String>) -> Self {
        self.snapshot_dir = dir.into();
        self
    }
}

/// A visual snapshot
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Snapshot name/identifier
    pub name: String,
    /// Raw image data
    pub data: Vec<u8>,
    /// Image width
    pub width: u32,
    /// Image height
    pub height: u32,
}

impl Snapshot {
    /// Create a new snapshot
    #[must_use]
    pub fn new(name: impl Into<String>, data: Vec<u8>) -> Self {
        Self {
            name: name.into(),
            data,
            width: 0,
            height: 0,
        }
    }

    /// Create with dimensions
    #[must_use]
    pub const fn with_dimensions(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Compare this snapshot to another
    #[must_use]
    pub fn diff(&self, other: &Self) -> SnapshotDiff {
        // Simple byte-by-byte comparison
        let mut difference_count = 0;
        let max_len = self.data.len().max(other.data.len());

        if max_len == 0 {
            return SnapshotDiff {
                identical: true,
                difference_count: 0,
                difference_percent: 0.0,
                diff_data: Vec::new(),
            };
        }

        for i in 0..max_len {
            let a = self.data.get(i).copied().unwrap_or(0);
            let b = other.data.get(i).copied().unwrap_or(0);
            if a != b {
                difference_count += 1;
            }
        }

        #[allow(clippy::cast_precision_loss)] // Acceptable for percentage calculation
        let difference_percent = (difference_count as f64 / max_len as f64) * 100.0;

        SnapshotDiff {
            identical: difference_count == 0,
            difference_count,
            difference_percent,
            diff_data: Vec::new(), // Would contain visual diff in full impl
        }
    }

    /// Get snapshot size in bytes
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }
}

/// Result of comparing two snapshots
#[derive(Debug, Clone)]
pub struct SnapshotDiff {
    /// Whether snapshots are identical
    pub identical: bool,
    /// Number of differing bytes/pixels
    pub difference_count: usize,
    /// Percentage of difference
    pub difference_percent: f64,
    /// Visual diff data (highlighted differences)
    pub diff_data: Vec<u8>,
}

impl SnapshotDiff {
    /// Check if snapshots are identical
    #[must_use]
    pub const fn is_identical(&self) -> bool {
        self.identical
    }

    /// Check if difference is within threshold
    #[must_use]
    pub fn within_threshold(&self, threshold: f64) -> bool {
        self.difference_percent <= threshold * 100.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    // =========================================================================
    // H₀ EXTREME TDD: Snapshot Tests (Feature F P0)
    // =========================================================================

    mod h0_snapshot_config_tests {
        use super::*;

        #[test]
        fn h0_snap_01_config_default_update_false() {
            let config = SnapshotConfig::default();
            assert!(!config.update_snapshots);
        }

        #[test]
        fn h0_snap_02_config_default_threshold() {
            let config = SnapshotConfig::default();
            assert!((config.threshold - 0.01).abs() < 0.001);
        }

        #[test]
        fn h0_snap_03_config_default_dir() {
            let config = SnapshotConfig::default();
            assert_eq!(config.snapshot_dir, "__snapshots__");
        }

        #[test]
        fn h0_snap_04_config_with_update_true() {
            let config = SnapshotConfig::default().with_update(true);
            assert!(config.update_snapshots);
        }

        #[test]
        fn h0_snap_05_config_with_update_false() {
            let config = SnapshotConfig::default().with_update(false);
            assert!(!config.update_snapshots);
        }

        #[test]
        fn h0_snap_06_config_with_threshold() {
            let config = SnapshotConfig::default().with_threshold(0.05);
            assert!((config.threshold - 0.05).abs() < 0.001);
        }

        #[test]
        fn h0_snap_07_config_with_dir() {
            let config = SnapshotConfig::default().with_dir("custom_dir");
            assert_eq!(config.snapshot_dir, "custom_dir");
        }

        #[test]
        fn h0_snap_08_config_builder_chain() {
            let config = SnapshotConfig::default()
                .with_update(true)
                .with_threshold(0.1)
                .with_dir("test_snaps");
            assert!(config.update_snapshots);
            assert!((config.threshold - 0.1).abs() < 0.001);
            assert_eq!(config.snapshot_dir, "test_snaps");
        }

        #[test]
        fn h0_snap_09_config_clone() {
            let config = SnapshotConfig::default().with_threshold(0.02);
            let cloned = config;
            assert!((cloned.threshold - 0.02).abs() < 0.001);
        }

        #[test]
        fn h0_snap_10_config_zero_threshold() {
            let config = SnapshotConfig::default().with_threshold(0.0);
            assert!((config.threshold - 0.0).abs() < f64::EPSILON);
        }
    }

    mod h0_snapshot_tests {
        use super::*;

        #[test]
        fn h0_snap_11_snapshot_new() {
            let snap = Snapshot::new("test", vec![1, 2, 3]);
            assert_eq!(snap.name, "test");
        }

        #[test]
        fn h0_snap_12_snapshot_new_data() {
            let snap = Snapshot::new("test", vec![10, 20, 30]);
            assert_eq!(snap.data, vec![10, 20, 30]);
        }

        #[test]
        fn h0_snap_13_snapshot_default_dimensions() {
            let snap = Snapshot::new("test", vec![]);
            assert_eq!(snap.width, 0);
            assert_eq!(snap.height, 0);
        }

        #[test]
        fn h0_snap_14_snapshot_with_dimensions() {
            let snap = Snapshot::new("test", vec![]).with_dimensions(100, 200);
            assert_eq!(snap.width, 100);
            assert_eq!(snap.height, 200);
        }

        #[test]
        fn h0_snap_15_snapshot_size() {
            let snap = Snapshot::new("test", vec![1, 2, 3, 4, 5]);
            assert_eq!(snap.size(), 5);
        }

        #[test]
        fn h0_snap_16_snapshot_size_empty() {
            let snap = Snapshot::new("test", vec![]);
            assert_eq!(snap.size(), 0);
        }

        #[test]
        fn h0_snap_17_snapshot_clone() {
            let snap = Snapshot::new("original", vec![1, 2, 3]);
            let cloned = snap;
            assert_eq!(cloned.name, "original");
            assert_eq!(cloned.data, vec![1, 2, 3]);
        }

        #[test]
        fn h0_snap_18_snapshot_string_name() {
            let snap = Snapshot::new(String::from("string_name"), vec![]);
            assert_eq!(snap.name, "string_name");
        }

        #[test]
        fn h0_snap_19_snapshot_large_data() {
            let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();
            let snap = Snapshot::new("large", data);
            assert_eq!(snap.size(), 1000);
        }

        #[test]
        fn h0_snap_20_snapshot_dimensions_chain() {
            let snap = Snapshot::new("test", vec![0; 100]).with_dimensions(10, 10);
            assert_eq!(snap.width * snap.height, 100);
        }
    }

    mod h0_snapshot_diff_tests {
        use super::*;

        #[test]
        fn h0_snap_21_diff_identical() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![1, 2, 3]);
            let diff = a.diff(&b);
            assert!(diff.identical);
        }

        #[test]
        fn h0_snap_22_diff_not_identical() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![1, 2, 4]);
            let diff = a.diff(&b);
            assert!(!diff.identical);
        }

        #[test]
        fn h0_snap_23_diff_difference_count() {
            let a = Snapshot::new("a", vec![1, 2, 3, 4]);
            let b = Snapshot::new("b", vec![1, 0, 3, 0]);
            let diff = a.diff(&b);
            assert_eq!(diff.difference_count, 2);
        }

        #[test]
        fn h0_snap_24_diff_difference_percent() {
            let a = Snapshot::new("a", vec![1, 2, 3, 4]);
            let b = Snapshot::new("b", vec![0, 0, 0, 0]);
            let diff = a.diff(&b);
            assert!((diff.difference_percent - 100.0).abs() < 0.001);
        }

        #[test]
        fn h0_snap_25_diff_empty_snapshots() {
            let a = Snapshot::new("a", vec![]);
            let b = Snapshot::new("b", vec![]);
            let diff = a.diff(&b);
            assert!(diff.identical);
        }

        #[test]
        fn h0_snap_26_diff_is_identical() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![1, 2, 3]);
            let diff = a.diff(&b);
            assert!(diff.is_identical());
        }

        #[test]
        fn h0_snap_27_diff_not_is_identical() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![4, 5, 6]);
            let diff = a.diff(&b);
            assert!(!diff.is_identical());
        }

        #[test]
        fn h0_snap_28_diff_within_threshold_true() {
            let a = Snapshot::new("a", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
            let b = Snapshot::new("b", vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 11]);
            let diff = a.diff(&b);
            // 10% diff, threshold 0.15 (15%)
            assert!(diff.within_threshold(0.15));
        }

        #[test]
        fn h0_snap_29_diff_within_threshold_false() {
            let a = Snapshot::new("a", vec![1, 2, 3, 4, 5]);
            let b = Snapshot::new("b", vec![0, 0, 0, 0, 0]);
            let diff = a.diff(&b);
            // 100% diff, threshold 0.5 (50%)
            assert!(!diff.within_threshold(0.5));
        }

        #[test]
        fn h0_snap_30_diff_clone() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![1, 2, 4]);
            let diff = a.diff(&b);
            let cloned = diff.clone();
            assert_eq!(cloned.difference_count, diff.difference_count);
        }
    }

    mod h0_snapshot_comparison_tests {
        use super::*;

        #[test]
        fn h0_snap_31_diff_different_lengths_longer_b() {
            let a = Snapshot::new("a", vec![1, 2]);
            let b = Snapshot::new("b", vec![1, 2, 3, 4]);
            let diff = a.diff(&b);
            assert!(!diff.identical);
        }

        #[test]
        fn h0_snap_32_diff_different_lengths_longer_a() {
            let a = Snapshot::new("a", vec![1, 2, 3, 4]);
            let b = Snapshot::new("b", vec![1, 2]);
            let diff = a.diff(&b);
            assert!(!diff.identical);
        }

        #[test]
        fn h0_snap_33_diff_zero_percent_identical() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![1, 2, 3]);
            let diff = a.diff(&b);
            assert!(diff.difference_percent < 0.001);
        }

        #[test]
        fn h0_snap_34_diff_fifty_percent() {
            let a = Snapshot::new("a", vec![1, 1]);
            let b = Snapshot::new("b", vec![1, 2]);
            let diff = a.diff(&b);
            assert!((diff.difference_percent - 50.0).abs() < 0.001);
        }

        #[test]
        fn h0_snap_35_diff_data_empty() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![1, 2, 4]);
            let diff = a.diff(&b);
            assert!(diff.diff_data.is_empty());
        }

        #[test]
        fn h0_snap_36_within_threshold_zero() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![1, 2, 3]);
            let diff = a.diff(&b);
            assert!(diff.within_threshold(0.0));
        }

        #[test]
        fn h0_snap_37_within_threshold_one() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![4, 5, 6]);
            let diff = a.diff(&b);
            assert!(diff.within_threshold(1.0));
        }

        #[test]
        fn h0_snap_38_single_byte_diff() {
            let a = Snapshot::new("a", vec![255]);
            let b = Snapshot::new("b", vec![0]);
            let diff = a.diff(&b);
            assert_eq!(diff.difference_count, 1);
            assert!((diff.difference_percent - 100.0).abs() < 0.001);
        }

        #[test]
        fn h0_snap_39_single_byte_same() {
            let a = Snapshot::new("a", vec![128]);
            let b = Snapshot::new("b", vec![128]);
            let diff = a.diff(&b);
            assert!(diff.identical);
        }

        #[test]
        fn h0_snap_40_large_snapshot_identical() {
            let data: Vec<u8> = vec![100; 10000];
            let a = Snapshot::new("a", data.clone());
            let b = Snapshot::new("b", data);
            let diff = a.diff(&b);
            assert!(diff.identical);
        }
    }

    mod h0_snapshot_edge_cases {
        use super::*;

        #[test]
        fn h0_snap_41_config_high_threshold() {
            let config = SnapshotConfig::default().with_threshold(1.0);
            assert!((config.threshold - 1.0).abs() < f64::EPSILON);
        }

        #[test]
        fn h0_snap_42_snapshot_with_zero_dimensions() {
            let snap = Snapshot::new("test", vec![1, 2, 3]).with_dimensions(0, 0);
            assert_eq!(snap.width, 0);
            assert_eq!(snap.height, 0);
        }

        #[test]
        fn h0_snap_43_snapshot_dimension_overflow_check() {
            let snap = Snapshot::new("test", vec![]).with_dimensions(u32::MAX, 1);
            assert_eq!(snap.width, u32::MAX);
        }

        #[test]
        fn h0_snap_44_diff_all_zeros() {
            let a = Snapshot::new("a", vec![0, 0, 0]);
            let b = Snapshot::new("b", vec![0, 0, 0]);
            let diff = a.diff(&b);
            assert!(diff.identical);
        }

        #[test]
        fn h0_snap_45_diff_all_max() {
            let a = Snapshot::new("a", vec![255, 255, 255]);
            let b = Snapshot::new("b", vec![255, 255, 255]);
            let diff = a.diff(&b);
            assert!(diff.identical);
        }

        #[test]
        fn h0_snap_46_snapshot_name_empty() {
            let snap = Snapshot::new("", vec![1, 2, 3]);
            assert_eq!(snap.name, "");
        }

        #[test]
        fn h0_snap_47_snapshot_name_unicode() {
            let snap = Snapshot::new("テスト_스냅샷", vec![1, 2, 3]);
            assert_eq!(snap.name, "テスト_스냅샷");
        }

        #[test]
        fn h0_snap_48_config_empty_dir() {
            let config = SnapshotConfig::default().with_dir("");
            assert_eq!(config.snapshot_dir, "");
        }

        #[test]
        fn h0_snap_49_diff_one_empty_one_full() {
            let a = Snapshot::new("a", vec![]);
            let b = Snapshot::new("b", vec![1, 2, 3]);
            let diff = a.diff(&b);
            assert!(!diff.identical);
            assert_eq!(diff.difference_count, 3);
        }

        #[test]
        fn h0_snap_50_diff_full_one_empty() {
            let a = Snapshot::new("a", vec![1, 2, 3]);
            let b = Snapshot::new("b", vec![]);
            let diff = a.diff(&b);
            assert!(!diff.identical);
            assert_eq!(diff.difference_count, 3);
        }
    }
}
