//! Vector type for 1D numeric data.

use std::ops::{Add, Index, IndexMut, Mul, Sub};

/// A 1D vector of floating-point values.
///
/// # Examples
///
/// ```
/// use aprender::primitives::Vector;
///
/// let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// assert_eq!(v.len(), 3);
/// assert!((v.sum() - 6.0).abs() < 1e-6);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Vector<T> {
    data: Vec<T>,
}

impl<T: Copy> Vector<T> {
    /// Creates a new vector from a slice.
    #[must_use]
    pub fn from_slice(data: &[T]) -> Self {
        Self {
            data: data.to_vec(),
        }
    }

    /// Creates a new vector from a Vec.
    #[must_use]
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Returns the number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the vector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a slice of the underlying data.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns a mutable slice of the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Returns a slice from start to end indices.
    #[must_use]
    pub fn slice(&self, start: usize, end: usize) -> Self {
        Self::from_slice(&self.data[start..end])
    }
}

impl<T> Index<usize> for Vector<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Vector<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl Vector<f32> {
    /// Creates a vector of zeros.
    #[must_use]
    pub fn zeros(len: usize) -> Self {
        Self {
            data: vec![0.0; len],
        }
    }

    /// Creates a vector of ones.
    #[must_use]
    pub fn ones(len: usize) -> Self {
        Self {
            data: vec![1.0; len],
        }
    }

    /// Computes the sum of all elements.
    #[must_use]
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Computes the mean of all elements.
    #[must_use]
    pub fn mean(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.sum() / self.data.len() as f32
    }

    /// Computes the dot product with another vector.
    ///
    /// # Panics
    ///
    /// Panics if vectors have different lengths.
    #[must_use]
    pub fn dot(&self, other: &Self) -> f32 {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector lengths must match for dot product"
        );
        self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Adds a scalar to each element.
    #[must_use]
    pub fn add_scalar(&self, scalar: f32) -> Self {
        Self {
            data: self.data.iter().map(|x| x + scalar).collect(),
        }
    }

    /// Multiplies each element by a scalar.
    #[must_use]
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        Self {
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }

    /// Computes the squared L2 norm.
    #[must_use]
    pub fn norm_squared(&self) -> f32 {
        self.dot(self)
    }

    /// Computes the L2 norm.
    #[must_use]
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }

    /// Returns the index of the minimum element.
    #[must_use]
    pub fn argmin(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Returns the index of the maximum element.
    #[must_use]
    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Computes variance of all elements.
    #[must_use]
    pub fn variance(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        let mean = self.mean();
        self.data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / self.data.len() as f32
    }
}

impl Add for &Vector<f32> {
    type Output = Vector<f32>;

    fn add(self, other: Self) -> Self::Output {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector lengths must match for addition"
        );
        Vector {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a + b)
                .collect(),
        }
    }
}

impl Sub for &Vector<f32> {
    type Output = Vector<f32>;

    fn sub(self, other: Self) -> Self::Output {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector lengths must match for subtraction"
        );
        Vector {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a - b)
                .collect(),
        }
    }
}

impl Mul for &Vector<f32> {
    type Output = Vector<f32>;

    fn mul(self, other: Self) -> Self::Output {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector lengths must match for multiplication"
        );
        Vector {
            data: self
                .data
                .iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_slice() {
        let v = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_zeros() {
        let v = Vector::<f32>::zeros(5);
        assert_eq!(v.len(), 5);
        assert!(v.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_ones() {
        let v = Vector::<f32>::ones(5);
        assert_eq!(v.len(), 5);
        assert!(v.as_slice().iter().all(|&x| (x - 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_sum() {
        let v = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        assert!((v.sum() - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean() {
        let v = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        assert!((v.mean() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot() {
        let a = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0_f32, 5.0, 6.0]);
        assert!((a.dot(&b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_scalar() {
        let v = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        let result = v.add_scalar(10.0);
        assert!((result[0] - 11.0).abs() < 1e-6);
        assert!((result[1] - 12.0).abs() < 1e-6);
        assert!((result[2] - 13.0).abs() < 1e-6);
    }

    #[test]
    fn test_mul_scalar() {
        let v = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        let result = v.mul_scalar(2.0);
        assert!((result[0] - 2.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
        assert!((result[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm() {
        let v = Vector::from_slice(&[3.0_f32, 4.0]);
        assert!((v.norm() - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmin() {
        let v = Vector::from_slice(&[3.0_f32, 1.0, 2.0]);
        assert_eq!(v.argmin(), 1);
    }

    #[test]
    fn test_argmax() {
        let v = Vector::from_slice(&[3.0_f32, 1.0, 2.0]);
        assert_eq!(v.argmax(), 0);
    }

    #[test]
    fn test_variance() {
        let v = Vector::from_slice(&[2.0_f32, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        assert!((v.variance() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_add_vectors() {
        let a = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        let b = Vector::from_slice(&[4.0_f32, 5.0, 6.0]);
        let result = &a + &b;
        assert!((result[0] - 5.0).abs() < 1e-6);
        assert!((result[1] - 7.0).abs() < 1e-6);
        assert!((result[2] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_sub_vectors() {
        let a = Vector::from_slice(&[4.0_f32, 5.0, 6.0]);
        let b = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        let result = &a - &b;
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 3.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_slice() {
        let v = Vector::from_slice(&[1.0_f32, 2.0, 3.0, 4.0, 5.0]);
        let sliced = v.slice(1, 4);
        assert_eq!(sliced.len(), 3);
        assert!((sliced[0] - 2.0).abs() < 1e-6);
        assert!((sliced[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_mean() {
        let v = Vector::<f32>::from_vec(vec![]);
        assert!((v.mean() - 0.0).abs() < 1e-6);
    }
}
