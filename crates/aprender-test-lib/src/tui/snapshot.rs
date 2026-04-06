//! TUI Snapshot Testing (Feature 21 - EDD Compliance)
//!
//! Provides snapshot testing for TUI frames with YAML serialization.
//!
//! ## EXTREME TDD: Tests written FIRST per spec
//!
//! ## Toyota Way Application
//!
//! - **Poka-Yoke**: Content-addressable snapshots prevent mismatches
//! - **Muda**: YAML format for human-readable diffs
//! - **Genchi Genbutsu**: Snapshot files are source of truth

use super::backend::TuiFrame;
use crate::result::{ProbarError, ProbarResult};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

/// TUI Snapshot for golden file testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuiSnapshot {
    /// Snapshot name/identifier
    pub name: String,
    /// Content hash for quick comparison
    pub hash: String,
    /// Frame width
    pub width: u16,
    /// Frame height
    pub height: u16,
    /// Frame content as lines
    pub content: Vec<String>,
    /// Optional metadata
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

impl TuiSnapshot {
    /// Create a snapshot from a TUI frame
    #[must_use]
    pub fn from_frame(name: &str, frame: &TuiFrame) -> Self {
        let content: Vec<String> = frame.lines().iter().map(ToString::to_string).collect();
        let hash = Self::compute_hash(&content);

        Self {
            name: name.to_string(),
            hash,
            width: frame.width(),
            height: frame.height(),
            content,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Create a snapshot from raw lines
    #[must_use]
    pub fn from_lines(name: &str, lines: &[&str]) -> Self {
        let content: Vec<String> = lines.iter().map(|s| (*s).to_string()).collect();
        let hash = Self::compute_hash(&content);
        let width = content.iter().map(|l| l.len()).max().unwrap_or(0) as u16;
        let height = content.len() as u16;

        Self {
            name: name.to_string(),
            hash,
            width,
            height,
            content,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add metadata to the snapshot
    #[must_use]
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }

    /// Compute content hash
    fn compute_hash(content: &[String]) -> String {
        let mut hasher = Sha256::new();
        for line in content {
            hasher.update(line.as_bytes());
            hasher.update(b"\n");
        }
        let result = hasher.finalize();
        format!("{result:x}")
    }

    /// Check if this snapshot matches another
    #[must_use]
    pub fn matches(&self, other: &TuiSnapshot) -> bool {
        self.hash == other.hash
    }

    /// Convert to TUI frame
    #[must_use]
    pub fn to_frame(&self) -> TuiFrame {
        let lines: Vec<&str> = self.content.iter().map(String::as_str).collect();
        TuiFrame::from_lines(&lines)
    }

    /// Save snapshot to a YAML file
    pub fn save(&self, path: &Path) -> ProbarResult<()> {
        let yaml = serde_yaml_ng::to_string(self).map_err(|e| {
            ProbarError::SnapshotSerializationError {
                message: format!("Failed to serialize snapshot: {e}"),
            }
        })?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(path, yaml)?;
        Ok(())
    }

    /// Load snapshot from a YAML file
    pub fn load(path: &Path) -> ProbarResult<Self> {
        let yaml = fs::read_to_string(path)?;
        let snapshot: TuiSnapshot = serde_yaml_ng::from_str(&yaml).map_err(|e| {
            ProbarError::SnapshotSerializationError {
                message: format!("Failed to deserialize snapshot: {e}"),
            }
        })?;
        Ok(snapshot)
    }

    /// Assert this snapshot matches an expected snapshot
    pub fn assert_matches(&self, expected: &TuiSnapshot) -> ProbarResult<()> {
        if self.matches(expected) {
            Ok(())
        } else {
            let actual_frame = self.to_frame();
            let expected_frame = expected.to_frame();
            let diff = actual_frame.diff(&expected_frame);

            Err(ProbarError::AssertionFailed {
                message: format!(
                    "Snapshot '{}' does not match expected:\n{}",
                    self.name, diff
                ),
            })
        }
    }
}

/// Snapshot manager for organizing and comparing snapshots
#[derive(Debug)]
pub struct SnapshotManager {
    /// Directory for snapshot files
    snapshot_dir: PathBuf,
    /// Whether to update snapshots on mismatch
    update_mode: bool,
}

impl SnapshotManager {
    /// Create a new snapshot manager
    #[must_use]
    pub fn new(snapshot_dir: &Path) -> Self {
        Self {
            snapshot_dir: snapshot_dir.to_path_buf(),
            update_mode: false,
        }
    }

    /// Enable update mode (overwrite snapshots on mismatch)
    #[must_use]
    pub fn with_update_mode(mut self, update: bool) -> Self {
        self.update_mode = update;
        self
    }

    /// Get the snapshot directory
    #[must_use]
    pub fn snapshot_dir(&self) -> &Path {
        &self.snapshot_dir
    }

    /// Get the path for a named snapshot
    #[must_use]
    pub fn snapshot_path(&self, name: &str) -> PathBuf {
        self.snapshot_dir.join(format!("{name}.snap.yaml"))
    }

    /// Check if a snapshot exists
    #[must_use]
    pub fn exists(&self, name: &str) -> bool {
        self.snapshot_path(name).exists()
    }

    /// Save a snapshot
    pub fn save(&self, snapshot: &TuiSnapshot) -> ProbarResult<()> {
        let path = self.snapshot_path(&snapshot.name);
        snapshot.save(&path)
    }

    /// Load a snapshot
    pub fn load(&self, name: &str) -> ProbarResult<TuiSnapshot> {
        let path = self.snapshot_path(name);
        TuiSnapshot::load(&path)
    }

    /// Assert a frame matches a snapshot (or create if missing)
    pub fn assert_snapshot(&self, name: &str, frame: &TuiFrame) -> ProbarResult<()> {
        let actual = TuiSnapshot::from_frame(name, frame);
        let path = self.snapshot_path(name);

        if path.exists() {
            let expected = TuiSnapshot::load(&path)?;

            if actual.matches(&expected) {
                Ok(())
            } else if self.update_mode {
                actual.save(&path)?;
                Ok(())
            } else {
                actual.assert_matches(&expected)
            }
        } else {
            // First run - create the snapshot
            actual.save(&path)?;
            Ok(())
        }
    }

    /// List all snapshots in the directory
    pub fn list(&self) -> ProbarResult<Vec<String>> {
        if !self.snapshot_dir.exists() {
            return Ok(Vec::new());
        }

        let mut names = Vec::new();
        for entry in fs::read_dir(&self.snapshot_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("yaml") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Some(name) = stem.strip_suffix(".snap") {
                        names.push(name.to_string());
                    }
                }
            }
        }
        names.sort();
        Ok(names)
    }

    /// Delete a snapshot
    pub fn delete(&self, name: &str) -> ProbarResult<()> {
        let path = self.snapshot_path(name);
        if path.exists() {
            fs::remove_file(path)?;
        }
        Ok(())
    }
}

impl Default for SnapshotManager {
    fn default() -> Self {
        Self::new(Path::new("__tui_snapshots__"))
    }
}

/// Sequence of frames for animation testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameSequence {
    /// Sequence name
    pub name: String,
    /// Ordered list of frame snapshots
    pub frames: Vec<TuiSnapshot>,
    /// Total duration in milliseconds
    pub duration_ms: u64,
}

impl FrameSequence {
    /// Create a new frame sequence
    #[must_use]
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            frames: Vec::new(),
            duration_ms: 0,
        }
    }

    /// Add a frame to the sequence
    pub fn add_frame(&mut self, frame: &TuiFrame) {
        let index = self.frames.len();
        let snapshot = TuiSnapshot::from_frame(&format!("{}_{}", self.name, index), frame);
        self.frames.push(snapshot);
        self.duration_ms = frame.timestamp_ms();
    }

    /// Get the number of frames
    #[must_use]
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Check if sequence is empty
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Get frame at index
    #[must_use]
    pub fn frame_at(&self, index: usize) -> Option<&TuiSnapshot> {
        self.frames.get(index)
    }

    /// Get the first frame
    #[must_use]
    pub fn first(&self) -> Option<&TuiSnapshot> {
        self.frames.first()
    }

    /// Get the last frame
    #[must_use]
    pub fn last(&self) -> Option<&TuiSnapshot> {
        self.frames.last()
    }

    /// Check if all frames in sequence match another sequence
    #[must_use]
    pub fn matches(&self, other: &FrameSequence) -> bool {
        if self.frames.len() != other.frames.len() {
            return false;
        }
        self.frames
            .iter()
            .zip(other.frames.iter())
            .all(|(a, b)| a.matches(b))
    }

    /// Get frames that differ between sequences
    #[must_use]
    pub fn diff_frames(&self, other: &FrameSequence) -> Vec<usize> {
        let mut diffs = Vec::new();
        let max_len = self.frames.len().max(other.frames.len());

        for i in 0..max_len {
            let self_frame = self.frames.get(i);
            let other_frame = other.frames.get(i);

            match (self_frame, other_frame) {
                (Some(a), Some(b)) if !a.matches(b) => diffs.push(i),
                (Some(_), None) | (None, Some(_)) => diffs.push(i),
                _ => {}
            }
        }
        diffs
    }

    /// Save sequence to YAML file
    pub fn save(&self, path: &Path) -> ProbarResult<()> {
        let yaml = serde_yaml_ng::to_string(self).map_err(|e| {
            ProbarError::SnapshotSerializationError {
                message: format!("Failed to serialize frame sequence: {e}"),
            }
        })?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(path, yaml)?;
        Ok(())
    }

    /// Load sequence from YAML file
    pub fn load(path: &Path) -> ProbarResult<Self> {
        let yaml = fs::read_to_string(path)?;
        let sequence: FrameSequence = serde_yaml_ng::from_str(&yaml).map_err(|e| {
            ProbarError::SnapshotSerializationError {
                message: format!("Failed to deserialize frame sequence: {e}"),
            }
        })?;
        Ok(sequence)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    mod tui_snapshot_tests {
        use super::*;

        #[test]
        fn test_from_lines() {
            let snap = TuiSnapshot::from_lines("test", &["Hello", "World"]);
            assert_eq!(snap.name, "test");
            assert_eq!(snap.width, 5);
            assert_eq!(snap.height, 2);
            assert_eq!(snap.content, vec!["Hello", "World"]);
            assert!(!snap.hash.is_empty());
        }

        #[test]
        fn test_from_frame() {
            let frame = TuiFrame::from_lines(&["Test", "Frame"]);
            let snap = TuiSnapshot::from_frame("test_snap", &frame);

            assert_eq!(snap.name, "test_snap");
            assert_eq!(snap.width, frame.width());
            assert_eq!(snap.height, frame.height());
        }

        #[test]
        fn test_with_metadata() {
            let snap = TuiSnapshot::from_lines("test", &["Hello"])
                .with_metadata("version", "1.0")
                .with_metadata("author", "test");

            assert_eq!(snap.metadata.get("version"), Some(&"1.0".to_string()));
            assert_eq!(snap.metadata.get("author"), Some(&"test".to_string()));
        }

        #[test]
        fn test_hash_consistency() {
            let snap1 = TuiSnapshot::from_lines("test1", &["Same", "Content"]);
            let snap2 = TuiSnapshot::from_lines("test2", &["Same", "Content"]);

            assert_eq!(snap1.hash, snap2.hash);
        }

        #[test]
        fn test_hash_different() {
            let snap1 = TuiSnapshot::from_lines("test", &["Content A"]);
            let snap2 = TuiSnapshot::from_lines("test", &["Content B"]);

            assert_ne!(snap1.hash, snap2.hash);
        }

        #[test]
        fn test_matches() {
            let snap1 = TuiSnapshot::from_lines("test1", &["Same"]);
            let snap2 = TuiSnapshot::from_lines("test2", &["Same"]);
            let snap3 = TuiSnapshot::from_lines("test3", &["Different"]);

            assert!(snap1.matches(&snap2));
            assert!(!snap1.matches(&snap3));
        }

        #[test]
        fn test_to_frame() {
            let snap = TuiSnapshot::from_lines("test", &["Hello", "World"]);
            let frame = snap.to_frame();

            assert_eq!(frame.lines(), &["Hello", "World"]);
        }

        #[test]
        fn test_save_and_load() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("test.snap.yaml");

            let snap =
                TuiSnapshot::from_lines("test", &["Hello", "World"]).with_metadata("key", "value");

            snap.save(&path).unwrap();
            assert!(path.exists());

            let loaded = TuiSnapshot::load(&path).unwrap();
            assert_eq!(loaded.name, snap.name);
            assert_eq!(loaded.hash, snap.hash);
            assert_eq!(loaded.content, snap.content);
            assert_eq!(loaded.metadata.get("key"), Some(&"value".to_string()));
        }

        #[test]
        fn test_assert_matches_pass() {
            let snap1 = TuiSnapshot::from_lines("test", &["Same"]);
            let snap2 = TuiSnapshot::from_lines("test", &["Same"]);

            assert!(snap1.assert_matches(&snap2).is_ok());
        }

        #[test]
        fn test_assert_matches_fail() {
            let snap1 = TuiSnapshot::from_lines("test", &["Actual"]);
            let snap2 = TuiSnapshot::from_lines("test", &["Expected"]);

            assert!(snap1.assert_matches(&snap2).is_err());
        }
    }

    mod snapshot_manager_tests {
        use super::*;

        #[test]
        fn test_new() {
            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());

            assert_eq!(manager.snapshot_dir(), temp_dir.path());
        }

        #[test]
        fn test_snapshot_path() {
            let manager = SnapshotManager::new(Path::new("/tmp/snaps"));
            let path = manager.snapshot_path("test");

            assert_eq!(path, PathBuf::from("/tmp/snaps/test.snap.yaml"));
        }

        #[test]
        fn test_save_and_load() {
            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());

            let snap = TuiSnapshot::from_lines("test", &["Content"]);
            manager.save(&snap).unwrap();

            assert!(manager.exists("test"));

            let loaded = manager.load("test").unwrap();
            assert_eq!(loaded.hash, snap.hash);
        }

        #[test]
        fn test_assert_snapshot_create() {
            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());
            let frame = TuiFrame::from_lines(&["Initial"]);

            // First call creates the snapshot
            assert!(manager.assert_snapshot("new_snap", &frame).is_ok());
            assert!(manager.exists("new_snap"));
        }

        #[test]
        fn test_assert_snapshot_match() {
            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());
            let frame = TuiFrame::from_lines(&["Same Content"]);

            // Create snapshot
            manager.assert_snapshot("test", &frame).unwrap();

            // Assert same content matches
            assert!(manager.assert_snapshot("test", &frame).is_ok());
        }

        #[test]
        fn test_assert_snapshot_mismatch() {
            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());

            // Create with one content
            let frame1 = TuiFrame::from_lines(&["Original"]);
            manager.assert_snapshot("test", &frame1).unwrap();

            // Assert different content fails
            let frame2 = TuiFrame::from_lines(&["Changed"]);
            assert!(manager.assert_snapshot("test", &frame2).is_err());
        }

        #[test]
        fn test_assert_snapshot_update_mode() {
            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path()).with_update_mode(true);

            // Create with one content
            let frame1 = TuiFrame::from_lines(&["Original"]);
            manager.assert_snapshot("test", &frame1).unwrap();

            // Update mode allows different content
            let frame2 = TuiFrame::from_lines(&["Updated"]);
            assert!(manager.assert_snapshot("test", &frame2).is_ok());

            // Verify it was updated
            let loaded = manager.load("test").unwrap();
            assert_eq!(loaded.content, vec!["Updated"]);
        }

        #[test]
        fn test_list() {
            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());

            manager
                .save(&TuiSnapshot::from_lines("alpha", &["a"]))
                .unwrap();
            manager
                .save(&TuiSnapshot::from_lines("beta", &["b"]))
                .unwrap();
            manager
                .save(&TuiSnapshot::from_lines("gamma", &["c"]))
                .unwrap();

            let list = manager.list().unwrap();
            assert_eq!(list, vec!["alpha", "beta", "gamma"]);
        }

        #[test]
        fn test_delete() {
            let temp_dir = TempDir::new().unwrap();
            let manager = SnapshotManager::new(temp_dir.path());

            manager
                .save(&TuiSnapshot::from_lines("test", &["content"]))
                .unwrap();
            assert!(manager.exists("test"));

            manager.delete("test").unwrap();
            assert!(!manager.exists("test"));
        }
    }

    mod frame_sequence_tests {
        use super::*;

        #[test]
        fn test_new() {
            let seq = FrameSequence::new("animation");
            assert_eq!(seq.name, "animation");
            assert!(seq.is_empty());
            assert_eq!(seq.len(), 0);
        }

        #[test]
        fn test_add_frame() {
            let mut seq = FrameSequence::new("test");
            let frame1 = TuiFrame::from_lines(&["Frame 1"]);
            let frame2 = TuiFrame::from_lines(&["Frame 2"]);

            seq.add_frame(&frame1);
            seq.add_frame(&frame2);

            assert_eq!(seq.len(), 2);
            assert!(!seq.is_empty());
        }

        #[test]
        fn test_frame_at() {
            let mut seq = FrameSequence::new("test");
            seq.add_frame(&TuiFrame::from_lines(&["First"]));
            seq.add_frame(&TuiFrame::from_lines(&["Second"]));

            assert!(seq.frame_at(0).is_some());
            assert!(seq.frame_at(1).is_some());
            assert!(seq.frame_at(2).is_none());
        }

        #[test]
        fn test_first_and_last() {
            let mut seq = FrameSequence::new("test");
            seq.add_frame(&TuiFrame::from_lines(&["First"]));
            seq.add_frame(&TuiFrame::from_lines(&["Middle"]));
            seq.add_frame(&TuiFrame::from_lines(&["Last"]));

            assert_eq!(seq.first().unwrap().content, vec!["First"]);
            assert_eq!(seq.last().unwrap().content, vec!["Last"]);
        }

        #[test]
        fn test_matches() {
            let mut seq1 = FrameSequence::new("seq1");
            let mut seq2 = FrameSequence::new("seq2");

            seq1.add_frame(&TuiFrame::from_lines(&["Same"]));
            seq2.add_frame(&TuiFrame::from_lines(&["Same"]));

            assert!(seq1.matches(&seq2));
        }

        #[test]
        fn test_matches_different() {
            let mut seq1 = FrameSequence::new("seq1");
            let mut seq2 = FrameSequence::new("seq2");

            seq1.add_frame(&TuiFrame::from_lines(&["A"]));
            seq2.add_frame(&TuiFrame::from_lines(&["B"]));

            assert!(!seq1.matches(&seq2));
        }

        #[test]
        fn test_matches_different_length() {
            let mut seq1 = FrameSequence::new("seq1");
            let mut seq2 = FrameSequence::new("seq2");

            seq1.add_frame(&TuiFrame::from_lines(&["A"]));
            seq1.add_frame(&TuiFrame::from_lines(&["B"]));
            seq2.add_frame(&TuiFrame::from_lines(&["A"]));

            assert!(!seq1.matches(&seq2));
        }

        #[test]
        fn test_diff_frames() {
            let mut seq1 = FrameSequence::new("seq1");
            let mut seq2 = FrameSequence::new("seq2");

            seq1.add_frame(&TuiFrame::from_lines(&["Same"]));
            seq1.add_frame(&TuiFrame::from_lines(&["Diff1"]));
            seq1.add_frame(&TuiFrame::from_lines(&["Same"]));

            seq2.add_frame(&TuiFrame::from_lines(&["Same"]));
            seq2.add_frame(&TuiFrame::from_lines(&["Diff2"]));
            seq2.add_frame(&TuiFrame::from_lines(&["Same"]));

            let diffs = seq1.diff_frames(&seq2);
            assert_eq!(diffs, vec![1]); // Only frame 1 differs
        }

        #[test]
        fn test_save_and_load() {
            let temp_dir = TempDir::new().unwrap();
            let path = temp_dir.path().join("sequence.yaml");

            let mut seq = FrameSequence::new("test_seq");
            seq.add_frame(&TuiFrame::from_lines(&["Frame 1"]));
            seq.add_frame(&TuiFrame::from_lines(&["Frame 2"]));

            seq.save(&path).unwrap();
            assert!(path.exists());

            let loaded = FrameSequence::load(&path).unwrap();
            assert_eq!(loaded.name, seq.name);
            assert_eq!(loaded.len(), seq.len());
            assert!(loaded.matches(&seq));
        }
    }
}
