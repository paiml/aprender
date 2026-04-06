//! Tensor implementation
//!
//! This module provides the core `Tensor` type, which is an N-dimensional array
//! with automatic backend selection for optimal performance.

use std::fmt;

use num_traits::Num;
use serde::{Deserialize, Serialize};

use crate::error::{RealizarError, Result};

/// N-dimensional tensor with automatic backend dispatch
///
/// The tensor automatically selects the optimal execution backend (SIMD, GPU, WASM)
/// based on operation type, data size, and available hardware.
///
/// # Examples
///
/// ```
/// use realizar::Tensor;
///
/// // Create a 2×3 tensor
/// let t = Tensor::from_vec(vec![2, 3], vec![
///     1.0, 2.0, 3.0,
///     4.0, 5.0, 6.0,
/// ]).expect("test");
///
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.ndim(), 2);
/// assert_eq!(t.size(), 6);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T: Num> {
    /// Flattened data in row-major order
    data: Vec<T>,
    /// Shape of the tensor
    shape: Vec<usize>,
}

impl<T: Num + Clone> Tensor<T> {
    /// Create a new tensor from a vector and shape
    ///
    /// # Arguments
    ///
    /// * `shape` - Dimensions of the tensor
    /// * `data` - Flattened data in row-major order
    ///
    /// # Errors
    ///
    /// Returns `Err` if:
    /// - Shape is empty
    /// - Data size doesn't match shape
    /// - Shape contains zero
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
    /// assert_eq!(t.shape(), &[2, 2]);
    /// ```
    pub fn from_vec(shape: Vec<usize>, data: Vec<T>) -> Result<Self> {
        // Validate shape
        if shape.is_empty() {
            return Err(RealizarError::InvalidShape {
                reason: "Shape cannot be empty".to_string(),
            });
        }

        if shape.contains(&0) {
            return Err(RealizarError::InvalidShape {
                reason: "Shape dimensions cannot be zero".to_string(),
            });
        }

        // Calculate expected size
        let expected_size = shape.iter().product();

        // Validate data size
        if data.len() != expected_size {
            return Err(RealizarError::DataShapeMismatch {
                data_size: data.len(),
                shape,
                expected: expected_size,
            });
        }

        Ok(Self { data, shape })
    }

    /// Get the shape of the tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![3, 4], vec![0.0; 12]).expect("test");
    /// assert_eq!(t.shape(), &[3, 4]);
    /// ```
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the number of dimensions
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2, 3, 4], vec![0.0; 24]).expect("test");
    /// assert_eq!(t.ndim(), 3);
    /// ```
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2, 3], vec![0.0; 6]).expect("test");
    /// assert_eq!(t.size(), 6);
    /// ```
    #[must_use]
    pub fn size(&self) -> usize {
        self.data.len()
    }

    /// Get a reference to the underlying data
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test");
    /// assert_eq!(t.data(), &[1.0, 2.0]);
    /// ```
    #[must_use]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Get a mutable reference to the underlying data
    ///
    /// # Examples
    ///
    /// ```
    /// use realizar::Tensor;
    ///
    /// let mut t = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test");
    /// t.data_mut()[0] = 9.0;
    /// assert_eq!(t.data(), &[9.0, 2.0]);
    /// ```
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T: Num + Clone + fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={:?}, data=[", self.shape)?;
        for (i, val) in self.data.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{val}")?;
        }
        write!(f, "])")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_tensor() {
        let t = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.size(), 6);
    }

    #[test]
    fn test_empty_shape_error() {
        let result = Tensor::from_vec(vec![], vec![1.0, 2.0]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::InvalidShape { .. }
        ));
    }

    #[test]
    fn test_zero_dimension_error() {
        let result = Tensor::<f32>::from_vec(vec![2, 0], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_size_mismatch_error() {
        let result = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0]);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RealizarError::DataShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_display() {
        let t = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test");
        let display = format!("{t}");
        assert!(display.contains("shape=[2]"));
        assert!(display.contains('1'));
        assert!(display.contains('2'));
    }

    #[test]
    fn test_tensor_1d() {
        let t = Tensor::from_vec(vec![5], vec![1.0, 2.0, 3.0, 4.0, 5.0]).expect("test");
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.ndim(), 1);
        assert_eq!(t.size(), 5);
        assert_eq!(t.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_tensor_3d() {
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = Tensor::from_vec(vec![2, 3, 4], data.clone()).expect("test");
        assert_eq!(t.shape(), &[2, 3, 4]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.size(), 24);
        assert_eq!(t.data(), data.as_slice());
    }

    #[test]
    fn test_tensor_4d() {
        let data: Vec<f64> = (0..120).map(|x| x as f64).collect();
        let t = Tensor::from_vec(vec![2, 3, 4, 5], data).expect("test");
        assert_eq!(t.shape(), &[2, 3, 4, 5]);
        assert_eq!(t.ndim(), 4);
        assert_eq!(t.size(), 120);
    }

    #[test]
    fn test_tensor_single_element() {
        let t = Tensor::from_vec(vec![1], vec![42.0]).expect("test");
        assert_eq!(t.shape(), &[1]);
        assert_eq!(t.ndim(), 1);
        assert_eq!(t.size(), 1);
        assert_eq!(t.data(), &[42.0]);
    }

    #[test]
    fn test_tensor_with_integers() {
        let t = Tensor::from_vec(vec![2, 2], vec![1, 2, 3, 4]).expect("test");
        assert_eq!(t.shape(), &[2, 2]);
        assert_eq!(t.size(), 4);
        assert_eq!(t.data(), &[1, 2, 3, 4]);
    }

    #[test]
    fn test_tensor_clone() {
        let t1 = Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
        let t2 = t1.clone();
        assert_eq!(t1.shape(), t2.shape());
        assert_eq!(t1.data(), t2.data());
    }

    #[test]
    fn test_tensor_debug() {
        let t = Tensor::from_vec(vec![2], vec![1.0, 2.0]).expect("test");
        let debug = format!("{:?}", t);
        assert!(debug.contains("Tensor"));
        assert!(debug.contains("data"));
        assert!(debug.contains("shape"));
    }

    #[test]
    fn test_display_multiple_elements() {
        let t = Tensor::from_vec(vec![3], vec![1.0, 2.0, 3.0]).expect("test");
        let display = format!("{t}");
        assert!(display.contains("shape=[3]"));
        assert!(display.contains("1"));
        assert!(display.contains("2"));
        assert!(display.contains("3"));
        // Check comma separators
        assert!(display.contains(", "));
    }

    #[test]
    fn test_display_2d() {
        let t = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("test");
        let display = format!("{t}");
        assert!(display.contains("shape=[2, 2]"));
    }

    #[test]
    fn test_zero_in_middle_dimension() {
        let result = Tensor::<f32>::from_vec(vec![2, 0, 3], vec![]);
        assert!(result.is_err());
        match result.unwrap_err() {
            RealizarError::InvalidShape { reason } => {
                assert!(reason.contains("zero"));
            },
            _ => panic!("Expected InvalidShape error"),
        }
    }

    #[test]
    fn test_shape_with_ones() {
        // Test with all 1s - should create scalar-like tensor
        let t = Tensor::from_vec(vec![1, 1, 1], vec![42.0]).expect("test");
        assert_eq!(t.shape(), &[1, 1, 1]);
        assert_eq!(t.ndim(), 3);
        assert_eq!(t.size(), 1);
    }

    #[test]
    fn test_data_size_too_large() {
        // More data than shape expects
        let result = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(result.is_err());
        match result.unwrap_err() {
            RealizarError::DataShapeMismatch {
                data_size,
                expected,
                ..
            } => {
                assert_eq!(data_size, 5);
                assert_eq!(expected, 4);
            },
            _ => panic!("Expected DataShapeMismatch error"),
        }
    }

    #[test]
    fn test_data_size_too_small() {
        // Less data than shape expects
        let result = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 3.0]);
        assert!(result.is_err());
        match result.unwrap_err() {
            RealizarError::DataShapeMismatch {
                data_size,
                expected,
                ..
            } => {
                assert_eq!(data_size, 3);
                assert_eq!(expected, 4);
            },
            _ => panic!("Expected DataShapeMismatch error"),
        }
    }

    #[test]
    fn test_shape_returns_reference() {
        let t = Tensor::from_vec(vec![2, 3, 4], vec![0.0f32; 24]).expect("test");
        let shape = t.shape();
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0], 2);
        assert_eq!(shape[1], 3);
        assert_eq!(shape[2], 4);
    }

    #[test]
    fn test_serde_roundtrip() {
        let original =
            Tensor::from_vec(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("test");
        let serialized = serde_json::to_string(&original).expect("serialize");
        let deserialized: Tensor<f64> = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(original.shape(), deserialized.shape());
        assert_eq!(original.data(), deserialized.data());
    }

    #[test]
    fn test_large_tensor() {
        // Test with a moderately large tensor
        let size = 1000;
        let data: Vec<f32> = (0..size).map(|x| x as f32).collect();
        let t = Tensor::from_vec(vec![10, 10, 10], data).expect("test");
        assert_eq!(t.size(), 1000);
        assert_eq!(t.ndim(), 3);
    }
}
