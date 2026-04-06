
// ============================================================================
// Tensor Statistics Canary (Regression Detection)
// ============================================================================

/// Tensor statistics for canary regression testing
#[derive(Debug, Clone)]
pub struct TensorCanary {
    /// Tensor name
    pub name: String,
    /// Shape
    pub shape: Vec<usize>,
    /// Data type
    pub dtype: String,
    /// Mean value
    pub mean: f32,
    /// Standard deviation
    pub std: f32,
    /// Minimum value
    pub min: f32,
    /// Maximum value
    pub max: f32,
    /// CRC32 checksum of first 1024 bytes
    pub checksum: u32,
}

/// Regression type detected by canary
#[derive(Debug, Clone)]
pub enum Regression {
    /// Shape does not match
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Mean drifted beyond tolerance
    MeanDrift {
        expected: f32,
        actual: f32,
        error: f32,
    },
    /// Standard deviation drifted
    StdDrift {
        expected: f32,
        actual: f32,
        error: f32,
    },
    /// Range (min/max) drifted
    RangeDrift {
        expected_min: f32,
        expected_max: f32,
        actual_min: f32,
        actual_max: f32,
    },
    /// Checksum mismatch
    ChecksumMismatch { expected: u32, actual: u32 },
}

impl TensorCanary {
    /// Create canary from tensor data
    #[must_use]
    pub fn from_data(
        name: impl Into<String>,
        shape: Vec<usize>,
        dtype: impl Into<String>,
        data: &[f32],
    ) -> Self {
        let features = TensorFeatures::from_data(data);

        // CRC32 of first 1024 bytes
        let bytes: Vec<u8> = data
            .iter()
            .take(256) // 256 floats = 1024 bytes
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let checksum = crc32_simple(&bytes);

        Self {
            name: name.into(),
            shape,
            dtype: dtype.into(),
            mean: features.mean,
            std: features.std,
            min: features.min,
            max: features.max,
            checksum,
        }
    }

    /// Detect regression against this canary
    ///
    /// # Falsification Criterion (F-CANARY-001)
    ///
    /// "If canary test produces false positive on identical model files,
    /// the tolerance thresholds are falsified."
    #[must_use]
    pub fn detect_regression(&self, current: &TensorCanary) -> Option<Regression> {
        // Shape MUST match exactly
        if self.shape != current.shape {
            return Some(Regression::ShapeMismatch {
                expected: self.shape.clone(),
                actual: current.shape.clone(),
            });
        }

        // Mean: 1% relative error tolerance
        let mean_base = self.mean.abs().max(1e-7);
        let mean_error = (current.mean - self.mean).abs() / mean_base;
        if mean_error > 0.01 {
            return Some(Regression::MeanDrift {
                expected: self.mean,
                actual: current.mean,
                error: mean_error,
            });
        }

        // Std: 5% relative error tolerance
        let std_base = self.std.abs().max(1e-7);
        let std_error = (current.std - self.std).abs() / std_base;
        if std_error > 0.05 {
            return Some(Regression::StdDrift {
                expected: self.std,
                actual: current.std,
                error: std_error,
            });
        }

        // Range: 10% absolute tolerance
        let range_tolerance = (self.max - self.min).abs() * 0.1;
        if current.min < self.min - range_tolerance || current.max > self.max + range_tolerance {
            return Some(Regression::RangeDrift {
                expected_min: self.min,
                expected_max: self.max,
                actual_min: current.min,
                actual_max: current.max,
            });
        }

        // Checksum: exact match (only first 1024 bytes)
        if self.checksum != current.checksum {
            return Some(Regression::ChecksumMismatch {
                expected: self.checksum,
                actual: current.checksum,
            });
        }

        None
    }
}

/// Canary file for a model
#[derive(Debug, Clone)]
pub struct CanaryFile {
    /// Model name
    pub model_name: String,
    /// Creation timestamp
    pub created_at: String,
    /// Tensor canaries
    pub tensors: Vec<TensorCanary>,
}

impl CanaryFile {
    /// Create new canary file
    #[must_use]
    pub fn new(model_name: impl Into<String>) -> Self {
        Self {
            model_name: model_name.into(),
            created_at: chrono_now(),
            tensors: Vec::new(),
        }
    }

    /// Add tensor canary
    pub fn add_tensor(&mut self, canary: TensorCanary) {
        self.tensors.push(canary);
    }

    /// Verify current model against canary
    #[must_use]
    pub fn verify(&self, current_tensors: &[TensorCanary]) -> Vec<(String, Regression)> {
        let mut regressions = Vec::new();

        for canary in &self.tensors {
            if let Some(current) = current_tensors.iter().find(|t| t.name == canary.name) {
                if let Some(regression) = canary.detect_regression(current) {
                    regressions.push((canary.name.clone(), regression));
                }
            } else {
                // Tensor missing - this is also a regression
                regressions.push((
                    canary.name.clone(),
                    Regression::ShapeMismatch {
                        expected: canary.shape.clone(),
                        actual: vec![],
                    },
                ));
            }
        }

        regressions
    }
}

/// Simple CRC32 implementation (for checksums)
fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB8_8320;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Get current timestamp (ISO 8601 format)
fn chrono_now() -> String {
    // Simple timestamp without chrono dependency
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests;
