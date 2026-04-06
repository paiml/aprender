//! Sovereign array types — Vec-backed replacements for ndarray Array1/Array2.
//!
//! These types provide the same API surface as ndarray but use plain Vec<f32>
//! storage, eliminating the ndarray dependency while maintaining identical behavior.

use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

/// 1-D array backed by `Vec<f32>`. Drop-in replacement for `ndarray::Array1<f32>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Array1<T = f32> {
    data: Vec<T>,
}

impl<T: Clone + Default> Array1<T> {
    pub fn zeros(n: usize) -> Self
    where
        T: From<f32>,
    {
        Self { data: vec![T::from(0.0f32); n] }
    }

    pub fn from_vec(v: Vec<T>) -> Self {
        Self { data: v }
    }
}

// Specialization for f32
impl Array1<f32> {
    pub fn ones(n: usize) -> Self {
        Self { data: vec![1.0; n] }
    }

    pub fn zeros_f32(n: usize) -> Self {
        Self { data: vec![0.0; n] }
    }

    pub fn mapv<F: Fn(f32) -> f32>(&self, f: F) -> Self {
        Self { data: self.data.iter().map(|&x| f(x)).collect() }
    }

    pub fn mapv_inplace<F: Fn(f32) -> f32>(&mut self, f: F) {
        for v in &mut self.data {
            *v = f(*v);
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn dot(&self, other: &Self) -> f32 {
        self.data.iter().zip(other.data.iter()).map(|(&a, &b)| a * b).sum()
    }

    pub fn mean(&self) -> Option<f32> {
        if self.data.is_empty() {
            None
        } else {
            Some(self.sum() / self.data.len() as f32)
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_slice_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    pub fn into_raw_vec(self) -> Vec<f32> {
        self.data
    }

    /// Vector-matrix multiply: self [m] dot other [m, n] -> [n]
    pub fn dot_mat(&self, mat: &Array2<f32>) -> Array1<f32> {
        assert_eq!(
            self.data.len(),
            mat.nrows(),
            "dot_mat: vector length {} != matrix rows {}",
            self.data.len(),
            mat.nrows()
        );
        let n = mat.ncols();
        let m = mat.nrows();
        let mut out = vec![0.0f32; n];
        for j in 0..n {
            let mut sum = 0.0f32;
            for i in 0..m {
                sum += self.data[i] * mat[[i, j]];
            }
            out[j] = sum;
        }
        Array1::from(out)
    }
}

impl<T> Array1<T> {
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    pub fn iter(&self) -> std::slice::Iter<'_, T> {
        self.data.iter()
    }
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, T> {
        self.data.iter_mut()
    }
}

impl Array1<f32> {
    pub fn assign(&mut self, other: &Array1<f32>) {
        self.data.clear();
        self.data.extend_from_slice(&other.data);
    }
}

impl<T> From<Vec<T>> for Array1<T> {
    fn from(v: Vec<T>) -> Self {
        Self { data: v }
    }
}

impl<T: Clone> From<&[T]> for Array1<T> {
    fn from(s: &[T]) -> Self {
        Self { data: s.to_vec() }
    }
}

// FromIterator for .collect()
impl std::iter::FromIterator<f32> for Array1<f32> {
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        Self { data: iter.into_iter().collect() }
    }
}

// += for Array1
impl std::ops::AddAssign<&Array1<f32>> for Array1<f32> {
    fn add_assign(&mut self, rhs: &Array1<f32>) {
        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            *a += b;
        }
    }
}

impl<T> Index<usize> for Array1<T> {
    type Output = T;
    fn index(&self, i: usize) -> &T {
        &self.data[i]
    }
}

impl<T> IndexMut<usize> for Array1<T> {
    fn index_mut(&mut self, i: usize) -> &mut T {
        &mut self.data[i]
    }
}

// &Array1 + &Array1
impl<'b> Add<&'b Array1<f32>> for &Array1<f32> {
    type Output = Array1<f32>;
    fn add(self, rhs: &'b Array1<f32>) -> Array1<f32> {
        Array1 { data: self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a + b).collect() }
    }
}

// &Array1 - &Array1
impl<'b> Sub<&'b Array1<f32>> for &Array1<f32> {
    type Output = Array1<f32>;
    fn sub(self, rhs: &'b Array1<f32>) -> Array1<f32> {
        Array1 { data: self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a - b).collect() }
    }
}

// &Array1 * &Array1
impl<'b> Mul<&'b Array1<f32>> for &Array1<f32> {
    type Output = Array1<f32>;
    fn mul(self, rhs: &'b Array1<f32>) -> Array1<f32> {
        Array1 { data: self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a * b).collect() }
    }
}

// &Array1 / &Array1
impl<'b> Div<&'b Array1<f32>> for &Array1<f32> {
    type Output = Array1<f32>;
    fn div(self, rhs: &'b Array1<f32>) -> Array1<f32> {
        Array1 { data: self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a / b).collect() }
    }
}

// Array1 + f32 (scalar broadcast)
impl Add<f32> for Array1<f32> {
    type Output = Array1<f32>;
    fn add(self, rhs: f32) -> Array1<f32> {
        Array1 { data: self.data.iter().map(|&a| a + rhs).collect() }
    }
}

impl Add<f32> for &Array1<f32> {
    type Output = Array1<f32>;
    fn add(self, rhs: f32) -> Array1<f32> {
        Array1 { data: self.data.iter().map(|&a| a + rhs).collect() }
    }
}

// Array1 * f32 (scalar broadcast)
impl Mul<f32> for Array1<f32> {
    type Output = Array1<f32>;
    fn mul(self, rhs: f32) -> Array1<f32> {
        Array1 { data: self.data.iter().map(|&a| a * rhs).collect() }
    }
}

impl Mul<f32> for &Array1<f32> {
    type Output = Array1<f32>;
    fn mul(self, rhs: f32) -> Array1<f32> {
        Array1 { data: self.data.iter().map(|&a| a * rhs).collect() }
    }
}

// Array1 / f32 (scalar broadcast)
impl Div<f32> for Array1<f32> {
    type Output = Array1<f32>;
    fn div(self, rhs: f32) -> Array1<f32> {
        Array1 { data: self.data.iter().map(|&a| a / rhs).collect() }
    }
}

// Array1 - f32 (scalar broadcast)
impl Sub<f32> for Array1<f32> {
    type Output = Array1<f32>;
    fn sub(self, rhs: f32) -> Array1<f32> {
        Array1 { data: self.data.iter().map(|&a| a - rhs).collect() }
    }
}

impl Sub<f32> for &Array1<f32> {
    type Output = Array1<f32>;
    fn sub(self, rhs: f32) -> Array1<f32> {
        Array1 { data: self.data.iter().map(|&a| a - rhs).collect() }
    }
}

// f32 * &Array1 (scalar on left)
impl Mul<&Array1<f32>> for f32 {
    type Output = Array1<f32>;
    fn mul(self, rhs: &Array1<f32>) -> Array1<f32> {
        Array1 { data: rhs.data.iter().map(|&a| self * a).collect() }
    }
}

// IntoIterator support
impl<T> IntoIterator for Array1<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T> IntoIterator for &'a Array1<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

/// Free function: create Array1 from slice (replaces `ndarray::arr1`)
pub fn arr1(data: &[f32]) -> Array1<f32> {
    Array1 { data: data.to_vec() }
}

/// Axis for Array2 operations (replaces `ndarray::Axis`)
#[derive(Clone, Copy, Debug)]
pub struct Axis(pub usize);

/// 2-D array backed by `Vec<T>` in row-major order.
/// Drop-in replacement for `ndarray::Array2<T>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Array2<T = f32> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

impl<T: Clone + Default> Array2<T> {
    pub fn zeros(shape: (usize, usize)) -> Self {
        Self { data: vec![T::default(); shape.0 * shape.1], rows: shape.0, cols: shape.1 }
    }

    pub fn from_elem(shape: (usize, usize), val: T) -> Self {
        Self { data: vec![val; shape.0 * shape.1], rows: shape.0, cols: shape.1 }
    }
}

impl Array2<f32> {
    pub fn ones(shape: (usize, usize)) -> Self {
        Self { data: vec![1.0; shape.0 * shape.1], rows: shape.0, cols: shape.1 }
    }

    pub fn from_shape_fn<F: Fn((usize, usize)) -> f32>(shape: (usize, usize), f: F) -> Self {
        let mut data = Vec::with_capacity(shape.0 * shape.1);
        for r in 0..shape.0 {
            for c in 0..shape.1 {
                data.push(f((r, c)));
            }
        }
        Self { data, rows: shape.0, cols: shape.1 }
    }

    pub fn from_shape_vec(
        shape: (usize, usize),
        data: Vec<f32>,
    ) -> std::result::Result<Self, String> {
        if data.len() != shape.0 * shape.1 {
            return Err(format!(
                "shape mismatch: expected {} elements, got {}",
                shape.0 * shape.1,
                data.len()
            ));
        }
        Ok(Self { data, rows: shape.0, cols: shape.1 })
    }

    pub fn nrows(&self) -> usize {
        self.rows
    }

    pub fn ncols(&self) -> usize {
        self.cols
    }

    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }

    pub fn row(&self, r: usize) -> ArrayView1<'_> {
        let start = r * self.cols;
        ArrayView1 { data: &self.data[start..start + self.cols] }
    }

    pub fn mapv<F: Fn(f32) -> f32>(&self, f: F) -> Self {
        Self { data: self.data.iter().map(|&x| f(x)).collect(), rows: self.rows, cols: self.cols }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> Option<f32> {
        if self.data.is_empty() {
            None
        } else {
            Some(self.data.iter().sum::<f32>() / self.data.len() as f32)
        }
    }

    pub fn t(&self) -> Self {
        let mut result = vec![0.0f32; self.data.len()];
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[c * self.rows + r] = self.data[r * self.cols + c];
            }
        }
        Self { data: result, rows: self.cols, cols: self.rows }
    }

    /// Matrix multiply: self [m,k] dot other [k,n] -> [m,n]
    pub fn dot(&self, other: &Self) -> Self {
        assert_eq!(
            self.cols, other.rows,
            "dot: incompatible shapes [{},{}] x [{},{}]",
            self.rows, self.cols, other.rows, other.cols
        );
        let m = self.rows;
        let k = self.cols;
        let n = other.cols;
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += self.data[i * k + p] * other.data[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        Self { data: out, rows: m, cols: n }
    }

    /// Matrix-vector multiply: self [m,k] dot vec [k] -> [m]
    pub fn dot_vec(&self, vec: &Array1<f32>) -> Array1<f32> {
        assert_eq!(
            self.cols,
            vec.len(),
            "dot_vec: matrix cols {} != vector length {}",
            self.cols,
            vec.len()
        );
        let m = self.rows;
        let k = self.cols;
        let mut out = vec![0.0f32; m];
        for i in 0..m {
            let mut sum = 0.0f32;
            for j in 0..k {
                sum += self.data[i * k + j] * vec[j];
            }
            out[i] = sum;
        }
        Array1::from(out)
    }

    /// Iterate rows along Axis(0)
    pub fn axis_iter(&self, _axis: Axis) -> AxisIter<'_> {
        AxisIter { data: &self.data, cols: self.cols, row: 0, total_rows: self.rows }
    }

    /// Iterate mutable rows along Axis(0)
    pub fn axis_iter_mut(&mut self, _axis: Axis) -> impl Iterator<Item = ArrayViewMut1<'_>> {
        let cols = self.cols;
        self.data.chunks_mut(cols).map(move |chunk| ArrayViewMut1 { data: chunk })
    }

    /// Sum along an axis
    pub fn sum_axis(&self, axis: Axis) -> Array1<f32> {
        match axis.0 {
            0 => {
                // Sum each column -> Array1 of length cols
                let mut result = vec![0.0f32; self.cols];
                for r in 0..self.rows {
                    for c in 0..self.cols {
                        result[c] += self.data[r * self.cols + c];
                    }
                }
                Array1::from(result)
            }
            1 => {
                // Sum each row -> Array1 of length rows
                let mut result = vec![0.0f32; self.rows];
                for r in 0..self.rows {
                    for c in 0..self.cols {
                        result[r] += self.data[r * self.cols + c];
                    }
                }
                Array1::from(result)
            }
            _ => panic!("Axis({}) not supported for 2D array", axis.0),
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.data.iter()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.clone()
    }

    pub fn row_mut(&mut self, r: usize) -> &mut [f32] {
        let start = r * self.cols;
        &mut self.data[start..start + self.cols]
    }

    pub fn rows_mut(&mut self) -> impl Iterator<Item = &mut [f32]> {
        let cols = self.cols;
        self.data.chunks_mut(cols)
    }
}

impl Index<[usize; 2]> for Array2<f32> {
    type Output = f32;
    fn index(&self, idx: [usize; 2]) -> &f32 {
        &self.data[idx[0] * self.cols + idx[1]]
    }
}

impl IndexMut<[usize; 2]> for Array2<f32> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut f32 {
        &mut self.data[idx[0] * self.cols + idx[1]]
    }
}

// &Array2 + &Array2
impl<'b> Add<&'b Array2<f32>> for &Array2<f32> {
    type Output = Array2<f32>;
    fn add(self, rhs: &'b Array2<f32>) -> Array2<f32> {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        Array2 {
            data: self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a + b).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// &Array2 - &Array2
impl<'b> Sub<&'b Array2<f32>> for &Array2<f32> {
    type Output = Array2<f32>;
    fn sub(self, rhs: &'b Array2<f32>) -> Array2<f32> {
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        Array2 {
            data: self.data.iter().zip(rhs.data.iter()).map(|(&a, &b)| a - b).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// Array2 * f32
impl Mul<f32> for &Array2<f32> {
    type Output = Array2<f32>;
    fn mul(self, rhs: f32) -> Array2<f32> {
        Array2 {
            data: self.data.iter().map(|&a| a * rhs).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// Array2 / f32
impl Div<f32> for Array2<f32> {
    type Output = Array2<f32>;
    fn div(self, rhs: f32) -> Array2<f32> {
        Array2 {
            data: self.data.iter().map(|&a| a / rhs).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

/// Read-only 1-D view (returned by `Array2::row`)
pub struct ArrayView1<'a> {
    data: &'a [f32],
}

impl ArrayView1<'_> {
    pub fn to_owned(&self) -> Array1<f32> {
        Array1::from(self.data.to_vec())
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.data.iter()
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mapv<F: Fn(f32) -> f32>(&self, f: F) -> Array1<f32> {
        Array1 { data: self.data.iter().map(|&x| f(x)).collect() }
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.data.to_vec()
    }
}

impl Index<usize> for ArrayView1<'_> {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        &self.data[i]
    }
}

/// Mutable 1-D view (returned by `axis_iter_mut`)
pub struct ArrayViewMut1<'a> {
    data: &'a mut [f32],
}

impl ArrayViewMut1<'_> {
    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[allow(clippy::iter_without_into_iter)]
    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.data.iter()
    }

    #[allow(clippy::iter_without_into_iter)]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f32> {
        self.data.iter_mut()
    }

    pub fn mapv_inplace<F: Fn(f32) -> f32>(&mut self, f: F) {
        for v in self.data.iter_mut() {
            *v = f(*v);
        }
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Assign values from an Array1
    pub fn assign(&mut self, src: &Array1<f32>) {
        self.data.copy_from_slice(src.as_slice());
    }
}

impl Index<usize> for ArrayViewMut1<'_> {
    type Output = f32;
    fn index(&self, i: usize) -> &f32 {
        &self.data[i]
    }
}

impl IndexMut<usize> for ArrayViewMut1<'_> {
    fn index_mut(&mut self, i: usize) -> &mut f32 {
        &mut self.data[i]
    }
}

pub struct AxisIter<'a> {
    data: &'a [f32],
    cols: usize,
    row: usize,
    total_rows: usize,
}

impl<'a> Iterator for AxisIter<'a> {
    type Item = ArrayView1<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.row >= self.total_rows {
            return None;
        }
        let start = self.row * self.cols;
        self.row += 1;
        Some(ArrayView1 { data: &self.data[start..start + self.cols] })
    }
}

/// Macro to create Array2 from nested arrays (replaces `ndarray::array!`)
#[macro_export]
macro_rules! array {
    [$([$($val:expr),* $(,)?]),* $(,)?] => {{
        let rows_data: Vec<Vec<f32>> = vec![$(vec![$($val as f32),*]),*];
        let rows = rows_data.len();
        let cols = if rows > 0 { rows_data[0].len() } else { 0 };
        let flat: Vec<f32> = rows_data.into_iter().flatten().collect();
        $crate::sovereign_array::Array2::from_shape_vec((rows, cols), flat).unwrap()
    }};
}

impl Array2<u32> {
    pub fn nrows(&self) -> usize {
        self.rows
    }
    pub fn ncols(&self) -> usize {
        self.cols
    }
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
    pub fn row(&self, r: usize) -> &[u32] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }
}

impl Array2<u8> {
    pub fn nrows(&self) -> usize {
        self.rows
    }
    pub fn ncols(&self) -> usize {
        self.cols
    }
    pub fn shape(&self) -> [usize; 2] {
        [self.rows, self.cols]
    }
    pub fn row(&self, r: usize) -> &[u8] {
        let start = r * self.cols;
        &self.data[start..start + self.cols]
    }
}

macro_rules! impl_array2_index {
    ($T:ty) => {
        impl Index<[usize; 2]> for Array2<$T> {
            type Output = $T;
            fn index(&self, idx: [usize; 2]) -> &$T {
                &self.data[idx[0] * self.cols + idx[1]]
            }
        }
        impl IndexMut<[usize; 2]> for Array2<$T> {
            fn index_mut(&mut self, idx: [usize; 2]) -> &mut $T {
                &mut self.data[idx[0] * self.cols + idx[1]]
            }
        }
    };
}
impl_array2_index!(u32);
impl_array2_index!(u8);

impl<'a, T> IntoIterator for &'a mut Array1<T> {
    type Item = &'a mut T;
    type IntoIter = std::slice::IterMut<'a, T>;
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

/// Generate delegating operator impls that forward owned args to &-& impl.
macro_rules! delegate_binop {
    ($Op:ident, $method:ident, $Lhs:ty, $Rhs:ty, $Out:ty) => {
        impl $Op<$Rhs> for $Lhs {
            type Output = $Out;
            fn $method(self, rhs: $Rhs) -> $Out {
                (&self).$method(&rhs)
            }
        }
    };
    (lref, $Op:ident, $method:ident, $Lhs:ty, $Rhs:ty, $Out:ty) => {
        impl $Op<$Rhs> for $Lhs {
            type Output = $Out;
            fn $method(self, rhs: $Rhs) -> $Out {
                self.$method(&rhs)
            }
        }
    };
    (rref, $Op:ident, $method:ident, $Lhs:ty, $Rhs:ty, $Out:ty) => {
        impl $Op<$Rhs> for $Lhs {
            type Output = $Out;
            fn $method(self, rhs: $Rhs) -> $Out {
                (&self).$method(rhs)
            }
        }
    };
}
delegate_binop!(Add, add, Array2<f32>, Array2<f32>, Array2<f32>);
delegate_binop!(rref, Add, add, Array2<f32>, &Array2<f32>, Array2<f32>);
delegate_binop!(lref, Add, add, &Array1<f32>, Array1<f32>, Array1<f32>);
delegate_binop!(rref, Add, add, Array1<f32>, &Array1<f32>, Array1<f32>);
delegate_binop!(Add, add, Array1<f32>, Array1<f32>, Array1<f32>);
delegate_binop!(rref, Sub, sub, Array1<f32>, &Array1<f32>, Array1<f32>);
delegate_binop!(Sub, sub, Array1<f32>, Array1<f32>, Array1<f32>);

impl Div<f32> for &Array2<f32> {
    type Output = Array2<f32>;
    fn div(self, rhs: f32) -> Array2<f32> {
        Array2 {
            data: self.data.iter().map(|&a| a / rhs).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn array1_zeros() {
        let a = Array1::<f32>::zeros(4);
        assert_eq!(a.len(), 4);
        assert!(a.as_slice().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn array1_from_vec_and_index() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(a[0], 1.0);
        assert_eq!(a[2], 3.0);
    }

    #[test]
    fn array1_arithmetic() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);
        assert_eq!((&a + &b)[0], 4.0);
        assert_eq!((&a - &b)[0], -2.0);
    }

    #[test]
    fn array1_dot() {
        let a = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let b = Array1::from_vec(vec![4.0, 5.0, 6.0]);
        assert_eq!(a.dot(&b), 32.0);
    }

    #[test]
    fn array2_zeros_and_shape() {
        let m = Array2::<f32>::zeros((2, 3));
        assert_eq!(m.shape(), [2, 3]);
        assert_eq!(m.nrows(), 2);
        assert_eq!(m.ncols(), 3);
    }

    #[test]
    fn array2_index_and_row() {
        let mut m = Array2::<f32>::zeros((2, 2));
        m[[0, 1]] = 5.0;
        assert_eq!(m[[0, 1]], 5.0);
        let m2 = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(m2.row(0).len(), 3);
    }

    #[test]
    fn array2_transpose() {
        let m = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let t = m.t();
        assert_eq!(t.shape(), [3, 2]);
        assert_eq!(t[[1, 0]], 2.0);
        assert_eq!(t[[0, 1]], 4.0);
    }
}
