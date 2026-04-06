//! Decision Path Trait, Error Types, and ByteReader
//!
//! Shared foundation for all decision path types.

use serde::{Deserialize, Serialize};

/// Common interface for all decision paths
pub trait DecisionPath: Clone + Send + Sync + 'static {
    /// Human-readable explanation
    fn explain(&self) -> String;

    /// Feature importance scores (contribution of each feature)
    fn feature_contributions(&self) -> &[f32];

    /// Confidence in this decision (0.0 - 1.0)
    fn confidence(&self) -> f32;

    /// Compact binary representation
    fn to_bytes(&self) -> Vec<u8>;

    /// Reconstruct from binary representation
    fn from_bytes(bytes: &[u8]) -> Result<Self, PathError>
    where
        Self: Sized;
}

/// Error type for path operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PathError {
    /// Invalid binary format
    InvalidFormat(String),
    /// Insufficient data
    InsufficientData { expected: usize, actual: usize },
    /// Version mismatch
    VersionMismatch { expected: u8, actual: u8 },
}

impl std::fmt::Display for PathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PathError::InvalidFormat(msg) => write!(f, "Invalid format: {msg}"),
            PathError::InsufficientData { expected, actual } => {
                write!(f, "Insufficient data: expected {expected}, got {actual}")
            }
            PathError::VersionMismatch { expected, actual } => {
                write!(f, "Version mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

impl std::error::Error for PathError {}

/// Stateful byte reader that tracks offset and validates bounds.
///
/// Consolidates duplicated readers from tree/forest/knn/neural paths.
pub(crate) struct ByteReader<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self { data, offset: 0 }
    }

    pub(crate) fn read_u8(&mut self) -> Result<u8, PathError> {
        self.ensure_available(1)?;
        let val = self.data[self.offset];
        self.offset += 1;
        Ok(val)
    }

    pub(crate) fn read_bool(&mut self) -> Result<bool, PathError> {
        Ok(self.read_u8()? != 0)
    }

    pub(crate) fn read_u32(&mut self) -> Result<u32, PathError> {
        self.ensure_available(4)?;
        let o = self.offset;
        let val = u32::from_le_bytes([
            self.data[o],
            self.data[o + 1],
            self.data[o + 2],
            self.data[o + 3],
        ]);
        self.offset += 4;
        Ok(val)
    }

    pub(crate) fn read_u32_as_usize(&mut self) -> Result<usize, PathError> {
        Ok(self.read_u32()? as usize)
    }

    pub(crate) fn read_f32(&mut self) -> Result<f32, PathError> {
        self.ensure_available(4)?;
        let o = self.offset;
        let val = f32::from_le_bytes([
            self.data[o],
            self.data[o + 1],
            self.data[o + 2],
            self.data[o + 3],
        ]);
        self.offset += 4;
        Ok(val)
    }

    pub(crate) fn read_f32_vec(&mut self) -> Result<Vec<f32>, PathError> {
        let len = self.read_u32()? as usize;
        let mut vec = Vec::with_capacity(len);
        for _ in 0..len {
            vec.push(self.read_f32()?);
        }
        Ok(vec)
    }

    /// Read a length-prefixed sub-message: u32 byte-length, then delegate to a parser.
    pub(crate) fn read_sub_message<T>(
        &mut self,
        parser: impl FnOnce(&[u8]) -> Result<T, PathError>,
    ) -> Result<T, PathError> {
        let len = self.read_u32()? as usize;
        self.ensure_available(len)?;
        let result = parser(&self.data[self.offset..self.offset + len])?;
        self.offset += len;
        Ok(result)
    }

    pub(crate) fn read_optional<T>(
        &mut self,
        read_value: impl FnOnce(&mut Self) -> Result<T, PathError>,
    ) -> Result<Option<T>, PathError> {
        let present = self.read_bool()?;
        if present {
            Ok(Some(read_value(self)?))
        } else {
            Ok(None)
        }
    }

    /// Read a length-prefixed `Vec<Vec<f32>>` (nested layers).
    pub(crate) fn read_nested_f32_vecs(&mut self) -> Result<Vec<Vec<f32>>, PathError> {
        let n_layers = self.read_u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(self.read_f32_vec()?);
        }
        Ok(layers)
    }

    pub(crate) fn ensure_available(&self, needed: usize) -> Result<(), PathError> {
        if self.offset + needed > self.data.len() {
            return Err(PathError::InsufficientData {
                expected: self.offset + needed,
                actual: self.data.len(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_error_display_invalid_format() {
        let err = PathError::InvalidFormat("test error".to_string());
        assert_eq!(format!("{err}"), "Invalid format: test error");
    }

    #[test]
    fn test_path_error_display_insufficient_data() {
        let err = PathError::InsufficientData {
            expected: 100,
            actual: 50,
        };
        assert_eq!(format!("{err}"), "Insufficient data: expected 100, got 50");
    }

    #[test]
    fn test_path_error_display_version_mismatch() {
        let err = PathError::VersionMismatch {
            expected: 1,
            actual: 2,
        };
        assert_eq!(format!("{err}"), "Version mismatch: expected 1, got 2");
    }

    #[test]
    fn test_path_error_clone() {
        let err = PathError::InvalidFormat("test".to_string());
        let cloned = err.clone();
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_path_error_partial_eq() {
        let err1 = PathError::InsufficientData {
            expected: 10,
            actual: 5,
        };
        let err2 = PathError::InsufficientData {
            expected: 10,
            actual: 5,
        };
        let err3 = PathError::InsufficientData {
            expected: 10,
            actual: 6,
        };
        assert_eq!(err1, err2);
        assert_ne!(err1, err3);
    }
}
