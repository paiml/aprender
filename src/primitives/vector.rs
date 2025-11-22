//! Vector type for 1D numeric data.

use serde::{Deserialize, Serialize};
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
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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
            .map_or(0, |(i, _)| i)
    }

    /// Returns the index of the maximum element.
    #[must_use]
    pub fn argmax(&self) -> usize {
        self.data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(i, _)| i)
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

    /// Computes standard deviation of all elements.
    ///
    /// Standard deviation is the square root of variance.
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::primitives::Vector;
    ///
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let std = v.std();
    /// assert!((std - 1.414).abs() < 0.01);
    /// ```
    #[must_use]
    pub fn std(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Computes Gini coefficient (inequality measure).
    ///
    /// The Gini coefficient measures inequality in a distribution.
    /// Formula: G = Σ Σ |x_i - x_j| / (2n² * mean)
    ///
    /// # Returns
    /// - 0.0: Perfect equality (all values are the same)
    /// - 1.0: Maximum inequality (one value has everything)
    ///
    /// # Examples
    ///
    /// ```
    /// use aprender::primitives::Vector;
    ///
    /// // Perfect equality
    /// let v = Vector::from_slice(&[5.0, 5.0, 5.0]);
    /// assert!((v.gini_coefficient() - 0.0).abs() < 0.01);
    ///
    /// // Some inequality
    /// let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    /// let gini = v.gini_coefficient();
    /// assert!(gini > 0.0 && gini < 1.0);
    /// ```
    #[must_use]
    pub fn gini_coefficient(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }

        let mean = self.mean();
        if mean == 0.0 {
            return 0.0;
        }

        let n = self.data.len() as f32;
        let mut sum_abs_diff = 0.0;

        for i in 0..self.data.len() {
            for j in 0..self.data.len() {
                sum_abs_diff += (self.data[i] - self.data[j]).abs();
            }
        }

        sum_abs_diff / (2.0 * n * n * mean)
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

    #[test]
    fn test_is_empty() {
        let empty = Vector::<f32>::from_vec(vec![]);
        assert!(empty.is_empty());

        let non_empty = Vector::from_slice(&[1.0_f32]);
        assert!(!non_empty.is_empty());
    }

    #[test]
    fn test_argmax_single_element() {
        let v = Vector::from_slice(&[42.0_f32]);
        assert_eq!(v.argmax(), 0);
    }

    #[test]
    fn test_argmax_all_equal() {
        let v = Vector::from_slice(&[5.0_f32, 5.0, 5.0]);
        let idx = v.argmax();
        // When all equal, any valid index is acceptable
        assert!(idx < v.len());
        assert!((v[idx] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmin_single_element() {
        let v = Vector::from_slice(&[42.0_f32]);
        assert_eq!(v.argmin(), 0);
    }

    #[test]
    fn test_argmin_all_equal() {
        let v = Vector::from_slice(&[5.0_f32, 5.0, 5.0]);
        let idx = v.argmin();
        // When all equal, any valid index is acceptable
        assert!(idx < v.len());
        assert!((v[idx] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_argmax_not_at_zero() {
        // Max at index 2, not 0 - catches "argmax -> 0" mutation
        let v = Vector::from_slice(&[1.0_f32, 2.0, 10.0]);
        assert_eq!(v.argmax(), 2);
    }

    #[test]
    fn test_mul_vectors() {
        // Element-wise multiplication - catches operator mutations
        let a = Vector::from_slice(&[2.0_f32, 3.0, 4.0]);
        let b = Vector::from_slice(&[5.0_f32, 6.0, 7.0]);
        let result = &a * &b;
        // 2*5=10, 3*6=18, 4*7=28
        assert!((result[0] - 10.0).abs() < 1e-6);
        assert!((result[1] - 18.0).abs() < 1e-6);
        assert!((result[2] - 28.0).abs() < 1e-6);

        // Verify it's not addition: would be 7, 9, 11
        assert!((result[0] - 7.0).abs() > 0.1);
        // Verify it's not division: would be 0.4, 0.5, 0.571...
        assert!((result[1] - 0.5).abs() > 1.0);
    }

    #[test]
    fn test_is_empty_true() {
        // Test that empty vector returns true for is_empty
        // Catches is_empty -> false mutation
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert!(v.is_empty(), "Empty vector should return true for is_empty");
        assert_eq!(v.len(), 0, "Empty vector should have len 0");
    }

    #[test]
    fn test_is_empty_false() {
        // Test that non-empty vector returns false for is_empty
        let v = Vector::from_slice(&[1.0_f32]);
        assert!(
            !v.is_empty(),
            "Non-empty vector should return false for is_empty"
        );
    }

    #[test]
    fn test_argmin_not_at_one() {
        // Min at index 0, not 1 - catches "argmin -> 1" mutation
        let v = Vector::from_slice(&[1.0_f32, 5.0, 3.0]);
        assert_eq!(v.argmin(), 0, "Minimum should be at index 0, not 1");
    }

    #[test]
    fn test_argmin_at_end() {
        // Min at last index - catches various index mutations
        let v = Vector::from_slice(&[5.0_f32, 3.0, 1.0]);
        assert_eq!(v.argmin(), 2, "Minimum should be at index 2");
    }

    #[test]
    fn test_as_mut_slice_modifies() {
        // Test that as_mut_slice actually allows modification
        // Catches as_mut_slice -> empty slice mutation
        let mut v = Vector::from_slice(&[1.0_f32, 2.0, 3.0]);
        {
            let slice = v.as_mut_slice();
            slice[0] = 10.0;
            slice[1] = 20.0;
        }
        assert!(
            (v[0] - 10.0).abs() < 1e-6,
            "First element should be modified to 10.0"
        );
        assert!(
            (v[1] - 20.0).abs() < 1e-6,
            "Second element should be modified to 20.0"
        );
    }

    #[test]
    fn test_as_mut_slice_length() {
        // Verify as_mut_slice returns correct length
        let mut v = Vector::from_slice(&[1.0_f32, 2.0, 3.0, 4.0]);
        let slice = v.as_mut_slice();
        assert_eq!(slice.len(), 4, "Mutable slice should have correct length");
    }

    // EXTREME TDD: Additional mutation-killing tests
    // These tests explicitly target MISSED mutants reported by cargo-mutants

    #[test]
    fn test_argmax_f32_returns_nonzero() {
        // MUTATION TARGET: "replace Vector<f32>::argmax -> usize with 0"
        // This test ensures argmax() does NOT always return 0
        let v: Vector<f32> = Vector::from_slice(&[1.0, 2.0, 999.0, 3.0]);
        assert_eq!(
            v.argmax(),
            2,
            "argmax must return 2 (position of max 999.0), not 0"
        );
        // Double check: if mutation makes argmax() return 0, this fails
        assert_ne!(v.argmax(), 0, "argmax must not always return 0");
    }

    #[test]
    fn test_as_mut_slice_f32_not_empty() {
        // MUTATION TARGET: "replace as_mut_slice -> &mut[T] with Vec::leak(Vec::new())"
        // This test ensures as_mut_slice() does NOT return empty slice
        let mut v: Vector<f32> = Vector::from_slice(&[10.0, 20.0, 30.0]);
        let slice = v.as_mut_slice();

        // If mutation returns empty slice, len check fails
        assert_eq!(
            slice.len(),
            3,
            "as_mut_slice must return slice with 3 elements, not empty"
        );

        // If mutation returns empty slice, modification has no effect
        slice[0] = 100.0;
        assert!(
            (v[0] - 100.0).abs() < 1e-6,
            "as_mut_slice must allow mutation of original data"
        );
    }

    #[test]
    fn test_mul_f32_not_addition() {
        // MUTATION TARGET: "replace * with + in <impl Mul for &Vector<f32>>::mul"
        // This test ensures Mul uses *, not +
        let a: Vector<f32> = Vector::from_slice(&[3.0, 4.0]);
        let b: Vector<f32> = Vector::from_slice(&[5.0, 6.0]);
        let result = &a * &b;

        // Multiplication: [3*5=15, 4*6=24]
        assert!(
            (result[0] - 15.0).abs() < 1e-6,
            "3*5 must equal 15, not 3+5=8"
        );
        assert!(
            (result[1] - 24.0).abs() < 1e-6,
            "4*6 must equal 24, not 4+6=10"
        );

        // If mutation uses +, we get [8, 10] instead
        assert!((result[0] - 8.0).abs() > 1.0, "Must not be addition");
        assert!((result[1] - 10.0).abs() > 1.0, "Must not be addition");
    }

    #[test]
    fn test_mul_f32_not_division() {
        // MUTATION TARGET: "replace * with / in <impl Mul for &Vector<f32>>::mul"
        // This test ensures Mul uses *, not /
        let a: Vector<f32> = Vector::from_slice(&[12.0, 20.0]);
        let b: Vector<f32> = Vector::from_slice(&[3.0, 4.0]);
        let result = &a * &b;

        // Multiplication: [12*3=36, 20*4=80]
        assert!(
            (result[0] - 36.0).abs() < 1e-6,
            "12*3 must equal 36, not 12/3=4"
        );
        assert!(
            (result[1] - 80.0).abs() < 1e-6,
            "20*4 must equal 80, not 20/4=5"
        );

        // If mutation uses /, we get [4, 5] instead
        assert!((result[0] - 4.0).abs() > 1.0, "Must not be division");
        assert!((result[1] - 5.0).abs() > 1.0, "Must not be division");
    }

    #[test]
    fn test_std() {
        // Test standard deviation calculation
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let std = v.std();

        // Expected: sqrt(variance) = sqrt(2.0) ≈ 1.414
        assert!((std - 1.414).abs() < 0.01, "std = {std}");
    }

    #[test]
    fn test_std_uniform() {
        // Uniform values should have std = 0
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
        assert!((v.std() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_gini_coefficient_perfect_equality() {
        // All values equal -> Gini = 0
        let v = Vector::from_slice(&[5.0, 5.0, 5.0, 5.0]);
        assert!((v.gini_coefficient() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_gini_coefficient_inequality() {
        // Some inequality
        let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let gini = v.gini_coefficient();

        // Should be between 0 and 1
        assert!(gini > 0.0 && gini < 1.0, "Gini = {gini}");

        // For this specific distribution: Gini ≈ 0.267
        assert!((gini - 0.267).abs() < 0.01, "Gini = {gini}");
    }

    #[test]
    fn test_gini_coefficient_maximum_inequality() {
        // Maximum inequality: one person has everything
        let v = Vector::from_slice(&[0.0, 0.0, 0.0, 100.0]);
        let gini = v.gini_coefficient();

        // Should approach 1.0 (but exact value depends on n)
        // For n=4: Gini = 0.75
        assert!(gini > 0.7 && gini < 0.8, "Gini = {gini}");
    }

    #[test]
    fn test_gini_coefficient_empty() {
        let v: Vector<f32> = Vector::from_slice(&[]);
        assert_eq!(v.gini_coefficient(), 0.0);
    }
}
