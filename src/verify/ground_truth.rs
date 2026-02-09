//! Ground Truth data structures for pipeline verification.
//!
//! Ground truth represents the expected output at a pipeline stage,
//! extracted from a reference implementation (whisper.cpp, `HuggingFace`, etc.)

use std::io;
use std::path::Path;

/// Ground truth statistics for a tensor at a pipeline stage.
#[derive(Debug, Clone)]
pub struct GroundTruth {
    /// Mean value
    mean: f32,
    /// Standard deviation
    std: f32,
    /// Minimum value
    min: f32,
    /// Maximum value
    max: f32,
    /// Optional raw data for detailed comparison
    data: Option<Vec<f32>>,
    /// Shape information
    shape: Vec<usize>,
}

impl GroundTruth {
    /// Create ground truth from precomputed statistics.
    ///
    /// # Arguments
    /// * `mean` - Expected mean value
    /// * `std` - Expected standard deviation
    #[must_use]
    pub fn from_stats(mean: f32, std: f32) -> Self {
        Self {
            mean,
            std,
            min: f32::NEG_INFINITY,
            max: f32::INFINITY,
            data: None,
            shape: vec![],
        }
    }

    /// Create ground truth from a data slice.
    ///
    /// Computes mean, std, min, max from the data.
    pub fn from_slice(data: &[f32]) -> Self {
        if data.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                data: None,
                shape: vec![0],
            };
        }

        let n = data.len() as f32;
        let mean = data.iter().sum::<f32>() / n;
        let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = variance.sqrt();
        let min = data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        Self {
            mean,
            std,
            min,
            max,
            data: Some(data.to_vec()),
            shape: vec![data.len()],
        }
    }

    /// Create ground truth with shape information.
    #[must_use]
    pub fn from_slice_with_shape(data: &[f32], shape: Vec<usize>) -> Self {
        let mut gt = Self::from_slice(data);
        gt.shape = shape;
        gt
    }

    /// Load ground truth from a binary file.
    ///
    /// Format: raw f32 values in little-endian byte order.
    pub fn from_bin_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        let data: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        Ok(Self::from_slice(&data))
    }

    /// Load ground truth statistics from a JSON file.
    ///
    /// Expected format: `{"mean": f32, "std": f32, "min": f32, "max": f32}`
    pub fn from_json_file<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        // Simple JSON parsing without serde dependency
        let mean = Self::extract_json_f32(&content, "mean")?;
        let std = Self::extract_json_f32(&content, "std")?;
        let min = Self::extract_json_f32(&content, "min").unwrap_or(f32::NEG_INFINITY);
        let max = Self::extract_json_f32(&content, "max").unwrap_or(f32::INFINITY);

        Ok(Self {
            mean,
            std,
            min,
            max,
            data: None,
            shape: vec![],
        })
    }

    /// Helper to extract a f32 value from JSON content.
    fn extract_json_f32(content: &str, key: &str) -> io::Result<f32> {
        let pattern = format!("\"{key}\":");
        let start = content.find(&pattern).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("key '{key}' not found in JSON"),
            )
        })?;
        let after_key = &content[start + pattern.len()..];
        let value_start = after_key.trim_start();
        let value_end = value_start
            .find([',', '}', '\n'])
            .unwrap_or(value_start.len());
        let value_str = value_start[..value_end].trim();
        value_str.parse::<f32>().map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("could not parse '{value_str}' as f32"),
            )
        })
    }

    /// Get the mean value.
    #[must_use]
    pub fn mean(&self) -> f32 {
        self.mean
    }

    /// Get the standard deviation.
    #[must_use]
    pub fn std(&self) -> f32 {
        self.std
    }

    /// Get the minimum value.
    #[must_use]
    pub fn min(&self) -> f32 {
        self.min
    }

    /// Get the maximum value.
    #[must_use]
    pub fn max(&self) -> f32 {
        self.max
    }

    /// Get the raw data if available.
    #[must_use]
    pub fn data(&self) -> Option<&[f32]> {
        self.data.as_deref()
    }

    /// Get the shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Check if raw data is available for detailed comparison.
    #[must_use]
    pub fn has_data(&self) -> bool {
        self.data.is_some()
    }
}

impl Default for GroundTruth {
    fn default() -> Self {
        Self::from_stats(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_stats() {
        let gt = GroundTruth::from_stats(-0.215, 0.448);
        assert!((gt.mean() - (-0.215)).abs() < 1e-6);
        assert!((gt.std() - 0.448).abs() < 1e-6);
    }

    #[test]
    fn test_from_slice_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let gt = GroundTruth::from_slice(&data);
        assert!((gt.mean() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_slice_std() {
        // Data with known population std
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let gt = GroundTruth::from_slice(&data);
        // Mean = 5.0, Variance = 4.0, Std = 2.0
        assert!((gt.std() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_slice_min_max() {
        let data = vec![-5.0, 0.0, 10.0, 3.0];
        let gt = GroundTruth::from_slice(&data);
        assert!((gt.min() - (-5.0)).abs() < 1e-6);
        assert!((gt.max() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_slice() {
        let data: Vec<f32> = vec![];
        let gt = GroundTruth::from_slice(&data);
        assert_eq!(gt.mean(), 0.0);
        assert_eq!(gt.std(), 0.0);
    }

    #[test]
    fn test_from_slice_with_shape() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gt = GroundTruth::from_slice_with_shape(&data, vec![2, 3]);
        assert_eq!(gt.shape(), &[2, 3]);
        assert!(gt.has_data());
        assert_eq!(
            gt.data()
                .expect("data should be present for slice-constructed GroundTruth")
                .len(),
            6
        );
    }

    #[test]
    fn test_has_data() {
        let gt_with_data = GroundTruth::from_slice(&[1.0, 2.0, 3.0]);
        let gt_stats_only = GroundTruth::from_stats(1.0, 0.5);
        assert!(gt_with_data.has_data());
        assert!(!gt_stats_only.has_data());
    }

    #[test]
    fn test_data_accessor() {
        let data = vec![1.0, 2.0, 3.0];
        let gt = GroundTruth::from_slice(&data);
        assert!(gt.data().is_some());
        assert_eq!(
            gt.data()
                .expect("data should be present for slice-constructed GroundTruth"),
            &[1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn test_shape_accessor() {
        let gt = GroundTruth::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(gt.shape(), &[4]);
    }

    #[test]
    fn test_default() {
        let gt = GroundTruth::default();
        assert!((gt.mean() - 0.0).abs() < 1e-6);
        assert!((gt.std() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_stats_min_max_defaults() {
        let gt = GroundTruth::from_stats(0.0, 1.0);
        assert!(gt.min().is_infinite() && gt.min().is_sign_negative());
        assert!(gt.max().is_infinite() && gt.max().is_sign_positive());
    }

    #[test]
    fn test_from_bin_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir creation should succeed");
        let path = dir.path().join("test.bin");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::File::create(&path)
            .expect("test file creation should succeed")
            .write_all(&bytes)
            .expect("test file write should succeed");

        let gt = GroundTruth::from_bin_file(&path)
            .expect("from_bin_file should parse valid binary data");
        assert!((gt.mean() - 3.0).abs() < 1e-6);
        assert!(gt.has_data());
    }

    #[test]
    fn test_from_bin_file_not_found() {
        let result = GroundTruth::from_bin_file("/nonexistent/path.bin");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_file() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir creation should succeed");
        let path = dir.path().join("test.json");
        let json = r#"{"mean": 0.5, "std": 1.2, "min": -0.1, "max": 2.0}"#;
        std::fs::File::create(&path)
            .expect("test file creation should succeed")
            .write_all(json.as_bytes())
            .expect("test file write should succeed");

        let gt =
            GroundTruth::from_json_file(&path).expect("from_json_file should parse valid JSON");
        assert!((gt.mean() - 0.5).abs() < 1e-6);
        assert!((gt.std() - 1.2).abs() < 1e-6);
        assert!((gt.min() - (-0.1)).abs() < 1e-6);
        assert!((gt.max() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_from_json_file_partial() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir creation should succeed");
        let path = dir.path().join("partial.json");
        let json = r#"{"mean": 0.5, "std": 1.2}"#;
        std::fs::File::create(&path)
            .expect("test file creation should succeed")
            .write_all(json.as_bytes())
            .expect("test file write should succeed");

        let gt = GroundTruth::from_json_file(&path)
            .expect("from_json_file should parse partial JSON with defaults");
        assert!((gt.mean() - 0.5).abs() < 1e-6);
        assert!(gt.min().is_infinite()); // Default
        assert!(gt.max().is_infinite()); // Default
    }

    #[test]
    fn test_from_json_file_not_found() {
        let result = GroundTruth::from_json_file("/nonexistent/path.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_file_missing_key() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir creation should succeed");
        let path = dir.path().join("missing.json");
        let json = r#"{"std": 1.2}"#; // Missing "mean"
        std::fs::File::create(&path)
            .expect("test file creation should succeed")
            .write_all(json.as_bytes())
            .expect("test file write should succeed");

        let result = GroundTruth::from_json_file(&path);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_json_file_invalid_value() {
        use std::io::Write;
        let dir = tempfile::tempdir().expect("tempdir creation should succeed");
        let path = dir.path().join("invalid.json");
        let json = r#"{"mean": "not_a_number", "std": 1.2}"#;
        std::fs::File::create(&path)
            .expect("test file creation should succeed")
            .write_all(json.as_bytes())
            .expect("test file write should succeed");

        let result = GroundTruth::from_json_file(&path);
        assert!(result.is_err());
    }
}
