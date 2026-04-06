#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! SIMD-accelerated operations using Trueno.
//!
//! This module provides hardware-accelerated vector and matrix operations
//! for the Presentar rendering pipeline.
//!
//! When the `simd` feature is disabled, operations fall back to scalar
//! implementations.
//!
//! # Example
//!
//! ```
//! use presentar_core::simd::{Vec4, Mat4, batch_transform_points};
//! use presentar_core::Point;
//!
//! let transform = Mat4::identity();
//! let points = vec![Point::new(0.0, 0.0), Point::new(100.0, 100.0)];
//! let transformed = batch_transform_points(&points, &transform);
//! ```

use crate::{Point, Rect};

/// 4-component vector for SIMD operations.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Vec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Vec4 {
    /// Create a new Vec4.
    #[inline]
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    /// Create a zero vector.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Create from a Point (z=0, w=1).
    #[inline]
    #[must_use]
    pub const fn from_point(p: Point) -> Self {
        Self::new(p.x, p.y, 0.0, 1.0)
    }

    /// Convert to Point (ignoring z and w).
    #[inline]
    #[must_use]
    pub const fn to_point(self) -> Point {
        Point {
            x: self.x,
            y: self.y,
        }
    }

    /// Dot product.
    #[inline]
    #[must_use]
    pub fn dot(self, other: Self) -> f32 {
        self.w.mul_add(
            other.w,
            self.z
                .mul_add(other.z, self.x.mul_add(other.x, self.y * other.y)),
        )
    }

    /// Component-wise addition.
    #[inline]
    #[must_use]
    pub fn add(self, other: Self) -> Self {
        Self::new(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
            self.w + other.w,
        )
    }

    /// Component-wise subtraction.
    #[inline]
    #[must_use]
    pub fn sub(self, other: Self) -> Self {
        Self::new(
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
            self.w - other.w,
        )
    }

    /// Scalar multiplication.
    #[inline]
    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s, self.w * s)
    }

    /// Component-wise multiplication.
    #[inline]
    #[must_use]
    pub fn mul(self, other: Self) -> Self {
        Self::new(
            self.x * other.x,
            self.y * other.y,
            self.z * other.z,
            self.w * other.w,
        )
    }

    /// Linear interpolation.
    #[inline]
    #[must_use]
    pub fn lerp(self, other: Self, t: f32) -> Self {
        self.add(other.sub(self).scale(t))
    }

    /// Length (magnitude).
    #[inline]
    #[must_use]
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// Normalize to unit length.
    #[inline]
    #[must_use]
    pub fn normalize(self) -> Self {
        let len = self.length();
        if len > 0.0 {
            self.scale(1.0 / len)
        } else {
            self
        }
    }
}

impl Default for Vec4 {
    fn default() -> Self {
        Self::zero()
    }
}

impl From<Point> for Vec4 {
    fn from(p: Point) -> Self {
        Self::from_point(p)
    }
}

impl From<Vec4> for Point {
    fn from(v: Vec4) -> Self {
        v.to_point()
    }
}

/// 4x4 matrix for transforms.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Mat4 {
    /// Row-major matrix data [row][col]
    pub data: [[f32; 4]; 4],
}

impl Mat4 {
    /// Create from raw data (row-major).
    #[inline]
    #[must_use]
    pub const fn from_data(data: [[f32; 4]; 4]) -> Self {
        Self { data }
    }

    /// Create identity matrix.
    #[inline]
    #[must_use]
    pub const fn identity() -> Self {
        Self::from_data([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create zero matrix.
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self::from_data([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ])
    }

    /// Create translation matrix.
    #[inline]
    #[must_use]
    pub const fn translation(x: f32, y: f32, z: f32) -> Self {
        Self::from_data([
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, y],
            [0.0, 0.0, 1.0, z],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create 2D translation matrix.
    #[inline]
    #[must_use]
    pub const fn translation_2d(x: f32, y: f32) -> Self {
        Self::translation(x, y, 0.0)
    }

    /// Create scale matrix.
    #[inline]
    #[must_use]
    pub const fn scale(x: f32, y: f32, z: f32) -> Self {
        Self::from_data([
            [x, 0.0, 0.0, 0.0],
            [0.0, y, 0.0, 0.0],
            [0.0, 0.0, z, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create 2D scale matrix.
    #[inline]
    #[must_use]
    pub const fn scale_2d(x: f32, y: f32) -> Self {
        Self::scale(x, y, 1.0)
    }

    /// Create uniform scale matrix.
    #[inline]
    #[must_use]
    pub const fn scale_uniform(s: f32) -> Self {
        Self::scale(s, s, s)
    }

    /// Create rotation around Z axis (2D rotation).
    #[inline]
    #[must_use]
    pub fn rotation_z(angle_rad: f32) -> Self {
        let (sin, cos) = angle_rad.sin_cos();
        Self::from_data([
            [cos, -sin, 0.0, 0.0],
            [sin, cos, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create orthographic projection matrix.
    #[inline]
    #[must_use]
    pub fn ortho(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self {
        let width = right - left;
        let height = top - bottom;
        let depth = far - near;

        Self::from_data([
            [2.0 / width, 0.0, 0.0, -(right + left) / width],
            [0.0, 2.0 / height, 0.0, -(top + bottom) / height],
            [0.0, 0.0, -2.0 / depth, -(far + near) / depth],
            [0.0, 0.0, 0.0, 1.0],
        ])
    }

    /// Create orthographic projection for screen coordinates (Y down).
    #[inline]
    #[must_use]
    pub fn ortho_screen(width: f32, height: f32) -> Self {
        Self::ortho(0.0, width, height, 0.0, -1.0, 1.0)
    }

    /// Matrix multiplication.
    #[inline]
    #[must_use]
    pub fn mul(&self, other: &Self) -> Self {
        let mut result = Self::zero();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }

    /// Transform a Vec4.
    #[inline]
    #[must_use]
    pub fn transform_vec4(&self, v: Vec4) -> Vec4 {
        Vec4::new(
            self.data[0][3].mul_add(
                v.w,
                self.data[0][2].mul_add(v.z, self.data[0][0].mul_add(v.x, self.data[0][1] * v.y)),
            ),
            self.data[1][3].mul_add(
                v.w,
                self.data[1][2].mul_add(v.z, self.data[1][0].mul_add(v.x, self.data[1][1] * v.y)),
            ),
            self.data[2][3].mul_add(
                v.w,
                self.data[2][2].mul_add(v.z, self.data[2][0].mul_add(v.x, self.data[2][1] * v.y)),
            ),
            self.data[3][3].mul_add(
                v.w,
                self.data[3][2].mul_add(v.z, self.data[3][0].mul_add(v.x, self.data[3][1] * v.y)),
            ),
        )
    }

    /// Transform a 2D point (assumes z=0, w=1).
    #[inline]
    #[must_use]
    pub fn transform_point(&self, p: Point) -> Point {
        let v = self.transform_vec4(Vec4::from_point(p));
        Point::new(v.x, v.y)
    }

    /// Transform a rectangle.
    #[inline]
    #[must_use]
    pub fn transform_rect(&self, rect: &Rect) -> Rect {
        let corners = [
            Point::new(rect.x, rect.y),
            Point::new(rect.x + rect.width, rect.y),
            Point::new(rect.x + rect.width, rect.y + rect.height),
            Point::new(rect.x, rect.y + rect.height),
        ];

        let transformed: Vec<Point> = corners.iter().map(|&p| self.transform_point(p)).collect();

        let min_x = transformed
            .iter()
            .map(|p| p.x)
            .fold(f32::INFINITY, f32::min);
        let max_x = transformed
            .iter()
            .map(|p| p.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = transformed
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min);
        let max_y = transformed
            .iter()
            .map(|p| p.y)
            .fold(f32::NEG_INFINITY, f32::max);

        Rect::new(min_x, min_y, max_x - min_x, max_y - min_y)
    }

    /// Get column as Vec4.
    #[inline]
    #[must_use]
    pub const fn column(&self, idx: usize) -> Vec4 {
        Vec4::new(
            self.data[0][idx],
            self.data[1][idx],
            self.data[2][idx],
            self.data[3][idx],
        )
    }

    /// Get row as Vec4.
    #[inline]
    #[must_use]
    pub const fn row(&self, idx: usize) -> Vec4 {
        Vec4::new(
            self.data[idx][0],
            self.data[idx][1],
            self.data[idx][2],
            self.data[idx][3],
        )
    }

    /// Transpose the matrix.
    #[inline]
    #[must_use]
    pub const fn transpose(&self) -> Self {
        Self::from_data([
            [
                self.data[0][0],
                self.data[1][0],
                self.data[2][0],
                self.data[3][0],
            ],
            [
                self.data[0][1],
                self.data[1][1],
                self.data[2][1],
                self.data[3][1],
            ],
            [
                self.data[0][2],
                self.data[1][2],
                self.data[2][2],
                self.data[3][2],
            ],
            [
                self.data[0][3],
                self.data[1][3],
                self.data[2][3],
                self.data[3][3],
            ],
        ])
    }
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::identity()
    }
}

impl std::ops::Mul for Mat4 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self::mul(&self, &rhs)
    }
}

impl std::ops::Mul<Vec4> for Mat4 {
    type Output = Vec4;
    fn mul(self, rhs: Vec4) -> Vec4 {
        self.transform_vec4(rhs)
    }
}

/// Batch transform multiple points.
///
/// When SIMD is enabled, this uses vectorized operations for better performance.
#[inline]
#[must_use]
pub fn batch_transform_points(points: &[Point], transform: &Mat4) -> Vec<Point> {
    points
        .iter()
        .map(|&p| transform.transform_point(p))
        .collect()
}

/// Batch transform multiple Vec4s.
#[inline]
#[must_use]
pub fn batch_transform_vec4(vecs: &[Vec4], transform: &Mat4) -> Vec<Vec4> {
    vecs.iter().map(|&v| transform.transform_vec4(v)).collect()
}

/// Batch linear interpolation.
#[inline]
#[must_use]
pub fn batch_lerp_points(from: &[Point], to: &[Point], t: f32) -> Vec<Point> {
    debug_assert_eq!(from.len(), to.len());
    from.iter()
        .zip(to.iter())
        .map(|(a, b)| a.lerp(b, t))
        .collect()
}

/// Axis-aligned bounding box from points.
#[inline]
#[must_use]
pub fn bounding_box(points: &[Point]) -> Option<Rect> {
    if points.is_empty() {
        return None;
    }

    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for p in points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    Some(Rect::new(min_x, min_y, max_x - min_x, max_y - min_y))
}

/// Calculate the centroid of points.
#[inline]
#[must_use]
pub fn centroid(points: &[Point]) -> Option<Point> {
    if points.is_empty() {
        return None;
    }

    let sum: (f32, f32) = points
        .iter()
        .fold((0.0, 0.0), |acc, p| (acc.0 + p.x, acc.1 + p.y));
    let n = points.len() as f32;
    Some(Point::new(sum.0 / n, sum.1 / n))
}

/// Check if a point is inside a convex polygon.
#[must_use]
pub fn point_in_convex_polygon(point: Point, polygon: &[Point]) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let mut positive = false;
    let mut negative = false;

    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];

        let cross = (point.x - a.x).mul_add(b.y - a.y, -((point.y - a.y) * (b.x - a.x)));

        if cross > 0.0 {
            positive = true;
        } else if cross < 0.0 {
            negative = true;
        }

        if positive && negative {
            return false;
        }
    }

    true
}

/// Compute the area of a polygon using the shoelace formula.
#[must_use]
pub fn polygon_area(polygon: &[Point]) -> f32 {
    if polygon.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    for i in 0..polygon.len() {
        let j = (i + 1) % polygon.len();
        area += polygon[i].x * polygon[j].y;
        area -= polygon[j].x * polygon[i].y;
    }

    (area / 2.0).abs()
}

// =============================================================================
// SIMD-accelerated implementations when trueno is available
// =============================================================================

#[cfg(feature = "simd")]
mod simd_impl {
    use super::Vec4;

    /// SIMD vector type from trueno (f32).
    pub type SimdVectorF32 = trueno::Vector<f32>;

    /// Create a SIMD-backed vector from Vec4.
    #[inline]
    #[must_use]
    pub fn vec4_to_simd(v: Vec4) -> SimdVectorF32 {
        SimdVectorF32::from_slice(&[v.x, v.y, v.z, v.w])
    }

    /// Create Vec4 from SIMD vector.
    #[inline]
    #[must_use]
    pub fn simd_to_vec4(v: &SimdVectorF32) -> Vec4 {
        let slice = v.as_slice();
        Vec4::new(
            slice.first().copied().unwrap_or(0.0),
            slice.get(1).copied().unwrap_or(0.0),
            slice.get(2).copied().unwrap_or(0.0),
            slice.get(3).copied().unwrap_or(0.0),
        )
    }

    /// SIMD-accelerated batch add using trueno.
    pub fn batch_add_simd(a: &[f32], b: &[f32]) -> trueno::Result<Vec<f32>> {
        let va = SimdVectorF32::from_slice(a);
        let vb = SimdVectorF32::from_slice(b);
        let result = va.add(&vb)?;
        Ok(result.as_slice().to_vec())
    }

    /// SIMD-accelerated dot product using trueno.
    pub fn dot_simd(a: &[f32], b: &[f32]) -> trueno::Result<f32> {
        let va = SimdVectorF32::from_slice(a);
        let vb = SimdVectorF32::from_slice(b);
        va.dot(&vb)
    }

    /// SIMD-accelerated scale using trueno.
    pub fn scale_simd(a: &[f32], s: f32) -> trueno::Result<Vec<f32>> {
        let va = SimdVectorF32::from_slice(a);
        let result = va.scale(s)?;
        Ok(result.as_slice().to_vec())
    }

    /// Get the best available SIMD backend.
    #[must_use]
    pub fn best_backend() -> trueno::Backend {
        trueno::Backend::select_best()
    }

    /// SIMD-accelerated batch dot product.
    pub fn batch_dot_product(a: &[Vec4], b: &[Vec4]) -> Vec<f32> {
        debug_assert_eq!(a.len(), b.len());
        a.iter().zip(b.iter()).map(|(va, vb)| va.dot(*vb)).collect()
    }
}

#[cfg(feature = "simd")]
pub use simd_impl::*;

// =============================================================================
// ComputeBlocks: Auto-Accelerated Aggregation Primitives
// =============================================================================
//
// ComputeBlocks are data processing primitives with automatic acceleration.
// They select the optimal execution path at compile time:
//
// 1. **Auto-vectorization**: Tight 4-way loops enable compiler SIMD codegen
// 2. **ILP (Instruction-Level Parallelism)**: Multiple accumulators hide latency
// 3. **Cache-friendly**: Sequential access patterns for prefetcher efficiency
//
// These primitives form the foundation for widget data processing:
// - Charts: range calculation, normalization, statistics
// - Heatmaps: binning, histogram generation
// - Tables: aggregation, sorting support
//
// Reference: PROBAR-SPEC-009 (ComputeBlock Architecture)
// Reference: docs/specifications/computeblocks-refactor.md Section 2

/// SIMD-friendly sum of f64 values.
///
/// Uses 4-way accumulator for instruction-level parallelism and auto-vectorization.
/// For small slices (<4), falls back to direct sum.
///
/// # Example
/// ```
/// use presentar_core::simd::batch_sum_f64;
/// let values = vec![1.0, 2.0, 3.0, 4.0];
/// assert_eq!(batch_sum_f64(&values), 10.0);
/// ```
#[inline]
#[must_use]
pub fn batch_sum_f64(values: &[f64]) -> f64 {
    if values.len() < 4 {
        return values.iter().sum();
    }

    // 4-way accumulator for ILP and auto-vectorization
    let chunks = values.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut acc = [0.0f64; 4];
    for chunk in chunks {
        acc[0] += chunk[0];
        acc[1] += chunk[1];
        acc[2] += chunk[2];
        acc[3] += chunk[3];
    }

    let mut sum = acc[0] + acc[1] + acc[2] + acc[3];
    for &v in remainder {
        sum += v;
    }
    sum
}

/// SIMD-friendly mean of f64 values.
///
/// Returns 0.0 for empty slices.
///
/// # Example
/// ```
/// use presentar_core::simd::batch_mean_f64;
/// let values = vec![2.0, 4.0, 6.0, 8.0];
/// assert_eq!(batch_mean_f64(&values), 5.0);
/// ```
#[inline]
#[must_use]
pub fn batch_mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    batch_sum_f64(values) / values.len() as f64
}

/// SIMD-friendly min/max of f64 values.
///
/// Returns `None` for empty slices.
/// Uses 4-way comparison for auto-vectorization.
///
/// # Example
/// ```
/// use presentar_core::simd::batch_min_max_f64;
/// let values = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
/// let (min, max) = batch_min_max_f64(&values).unwrap();
/// assert_eq!(min, 1.0);
/// assert_eq!(max, 9.0);
/// ```
#[inline]
#[must_use]
pub fn batch_min_max_f64(values: &[f64]) -> Option<(f64, f64)> {
    if values.is_empty() {
        return None;
    }

    if values.len() < 4 {
        let mut min = values[0];
        let mut max = values[0];
        for &v in &values[1..] {
            min = min.min(v);
            max = max.max(v);
        }
        return Some((min, max));
    }

    // 4-way min/max for auto-vectorization
    let chunks = values.chunks_exact(4);
    let remainder = chunks.remainder();

    let mut min_acc = [f64::INFINITY; 4];
    let mut max_acc = [f64::NEG_INFINITY; 4];

    for chunk in chunks {
        min_acc[0] = min_acc[0].min(chunk[0]);
        min_acc[1] = min_acc[1].min(chunk[1]);
        min_acc[2] = min_acc[2].min(chunk[2]);
        min_acc[3] = min_acc[3].min(chunk[3]);

        max_acc[0] = max_acc[0].max(chunk[0]);
        max_acc[1] = max_acc[1].max(chunk[1]);
        max_acc[2] = max_acc[2].max(chunk[2]);
        max_acc[3] = max_acc[3].max(chunk[3]);
    }

    let mut min = min_acc[0].min(min_acc[1]).min(min_acc[2]).min(min_acc[3]);
    let mut max = max_acc[0].max(max_acc[1]).max(max_acc[2]).max(max_acc[3]);

    for &v in remainder {
        min = min.min(v);
        max = max.max(v);
    }

    Some((min, max))
}

/// SIMD-friendly normalization to [0, 1] range.
///
/// Normalizes values using `(v - min) / (max - min)`.
/// Returns empty vec for empty input, all zeros if min == max.
///
/// # Example
/// ```
/// use presentar_core::simd::normalize_f64;
/// let values = vec![0.0, 50.0, 100.0];
/// let normalized = normalize_f64(&values);
/// assert_eq!(normalized, vec![0.0, 0.5, 1.0]);
/// ```
#[inline]
#[must_use]
pub fn normalize_f64(values: &[f64]) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }

    let Some((min, max)) = batch_min_max_f64(values) else {
        return Vec::new();
    };

    let range = max - min;
    if range == 0.0 {
        return vec![0.0; values.len()];
    }

    let inv_range = 1.0 / range;
    values.iter().map(|&v| (v - min) * inv_range).collect()
}

/// SIMD-friendly normalization with provided range.
///
/// Normalizes values using `(v - min) / (max - min)` with caller-provided bounds.
/// More efficient when min/max are already known.
///
/// # Example
/// ```
/// use presentar_core::simd::normalize_with_range_f64;
/// let values = vec![25.0, 50.0, 75.0];
/// let normalized = normalize_with_range_f64(&values, 0.0, 100.0);
/// assert_eq!(normalized, vec![0.25, 0.5, 0.75]);
/// ```
#[inline]
#[must_use]
pub fn normalize_with_range_f64(values: &[f64], min: f64, max: f64) -> Vec<f64> {
    let range = max - min;
    if range == 0.0 {
        return vec![0.0; values.len()];
    }

    let inv_range = 1.0 / range;
    values.iter().map(|&v| (v - min) * inv_range).collect()
}

/// SIMD-friendly scale operation.
///
/// Multiplies all values by a scalar.
///
/// # Example
/// ```
/// use presentar_core::simd::batch_scale_f64;
/// let values = vec![1.0, 2.0, 3.0];
/// let scaled = batch_scale_f64(&values, 2.0);
/// assert_eq!(scaled, vec![2.0, 4.0, 6.0]);
/// ```
#[inline]
#[must_use]
pub fn batch_scale_f64(values: &[f64], scale: f64) -> Vec<f64> {
    values.iter().map(|&v| v * scale).collect()
}

/// SIMD-friendly scale and offset operation.
///
/// Applies `v * scale + offset` to all values.
/// Common operation for data transformation.
///
/// # Example
/// ```
/// use presentar_core::simd::batch_scale_offset_f64;
/// let values = vec![0.0, 0.5, 1.0];
/// // Map [0, 1] to [100, 200]
/// let transformed = batch_scale_offset_f64(&values, 100.0, 100.0);
/// assert_eq!(transformed, vec![100.0, 150.0, 200.0]);
/// ```
#[inline]
#[must_use]
pub fn batch_scale_offset_f64(values: &[f64], scale: f64, offset: f64) -> Vec<f64> {
    values.iter().map(|&v| v.mul_add(scale, offset)).collect()
}

/// SIMD-friendly variance calculation (population variance).
///
/// Uses two-pass algorithm: first calculates mean, then variance.
/// Returns 0.0 for empty or single-element slices.
///
/// # Example
/// ```
/// use presentar_core::simd::batch_variance_f64;
/// let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let variance = batch_variance_f64(&values);
/// assert!((variance - 4.0).abs() < 0.001);
/// ```
#[inline]
#[must_use]
pub fn batch_variance_f64(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean = batch_mean_f64(values);
    let sum_sq: f64 = values.iter().map(|&v| (v - mean) * (v - mean)).sum();
    sum_sq / values.len() as f64
}

/// SIMD-friendly standard deviation.
///
/// Returns square root of population variance.
///
/// # Example
/// ```
/// use presentar_core::simd::batch_stddev_f64;
/// let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let stddev = batch_stddev_f64(&values);
/// assert!((stddev - 2.0).abs() < 0.001);
/// ```
#[inline]
#[must_use]
pub fn batch_stddev_f64(values: &[f64]) -> f64 {
    batch_variance_f64(values).sqrt()
}

/// SIMD-friendly weighted sum.
///
/// Computes sum(values[i] * weights[i]).
/// Panics in debug mode if lengths differ.
///
/// # Example
/// ```
/// use presentar_core::simd::weighted_sum_f64;
/// let values = vec![1.0, 2.0, 3.0];
/// let weights = vec![0.5, 0.3, 0.2];
/// let result = weighted_sum_f64(&values, &weights);
/// assert!((result - 1.7).abs() < 0.001);
/// ```
#[inline]
#[must_use]
pub fn weighted_sum_f64(values: &[f64], weights: &[f64]) -> f64 {
    debug_assert_eq!(values.len(), weights.len());

    if values.len() < 4 {
        return values
            .iter()
            .zip(weights.iter())
            .map(|(&v, &w)| v * w)
            .sum();
    }

    // 4-way accumulator
    let v_chunks = values.chunks_exact(4);
    let w_chunks = weights.chunks_exact(4);
    let v_rem = v_chunks.remainder();
    let w_rem = w_chunks.remainder();

    let mut acc = [0.0f64; 4];
    for (vc, wc) in v_chunks.zip(w_chunks) {
        acc[0] = vc[0].mul_add(wc[0], acc[0]);
        acc[1] = vc[1].mul_add(wc[1], acc[1]);
        acc[2] = vc[2].mul_add(wc[2], acc[2]);
        acc[3] = vc[3].mul_add(wc[3], acc[3]);
    }

    let mut sum = acc[0] + acc[1] + acc[2] + acc[3];
    for (&v, &w) in v_rem.iter().zip(w_rem.iter()) {
        sum = v.mul_add(w, sum);
    }
    sum
}

/// SIMD-friendly percentile calculation.
///
/// Finds the value at the given percentile (0.0-1.0).
/// Uses linear interpolation for non-integer indices.
/// Values must be sorted in ascending order.
///
/// # Example
/// ```
/// use presentar_core::simd::percentile_sorted_f64;
/// let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// assert_eq!(percentile_sorted_f64(&sorted, 0.5), 3.0); // median
/// ```
#[inline]
#[must_use]
pub fn percentile_sorted_f64(sorted_values: &[f64], p: f64) -> f64 {
    if sorted_values.is_empty() {
        return 0.0;
    }
    if sorted_values.len() == 1 {
        return sorted_values[0];
    }

    let p = p.clamp(0.0, 1.0);
    let n = sorted_values.len() as f64;
    let idx = p * (n - 1.0);
    let lower = idx.floor() as usize;
    let upper = idx.ceil() as usize;

    if lower == upper {
        sorted_values[lower]
    } else {
        let frac = idx - lower as f64;
        sorted_values[lower].mul_add(1.0 - frac, sorted_values[upper] * frac)
    }
}

/// Compute histogram bin counts for f64 data.
///
/// Assigns values to bins and returns counts per bin.
/// Efficient for visualization (heatmaps, histograms).
///
/// # Example
/// ```
/// use presentar_core::simd::histogram_f64;
/// let values = vec![0.1, 0.5, 0.9, 1.5, 2.5, 3.5, 4.5];
/// let counts = histogram_f64(&values, 0.0, 5.0, 5);
/// assert_eq!(counts, vec![3, 1, 1, 1, 1]); // bins: [0-1), [1-2), [2-3), [3-4), [4-5]
/// ```
#[inline]
#[must_use]
pub fn histogram_f64(values: &[f64], min: f64, max: f64, num_bins: usize) -> Vec<usize> {
    if num_bins == 0 || values.is_empty() {
        return vec![0; num_bins];
    }

    let range = max - min;
    if range <= 0.0 {
        // All values in first bin
        return {
            let mut counts = vec![0; num_bins];
            counts[0] = values.len();
            counts
        };
    }

    let bin_width = range / num_bins as f64;
    let mut counts = vec![0usize; num_bins];

    for &v in values {
        let bin = ((v - min) / bin_width) as usize;
        let bin = bin.min(num_bins - 1); // Clamp to last bin
        counts[bin] += 1;
    }

    counts
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Vec4 Tests
    // =========================================================================

    #[test]
    fn test_vec4_new() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);
        assert_eq!(v.z, 3.0);
        assert_eq!(v.w, 4.0);
    }

    #[test]
    fn test_vec4_zero() {
        let v = Vec4::zero();
        assert_eq!(v, Vec4::new(0.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_vec4_default() {
        let v = Vec4::default();
        assert_eq!(v, Vec4::zero());
    }

    #[test]
    fn test_vec4_from_point() {
        let p = Point::new(10.0, 20.0);
        let v = Vec4::from_point(p);
        assert_eq!(v, Vec4::new(10.0, 20.0, 0.0, 1.0));
    }

    #[test]
    fn test_vec4_to_point() {
        let v = Vec4::new(5.0, 15.0, 25.0, 35.0);
        let p = v.to_point();
        assert_eq!(p, Point::new(5.0, 15.0));
    }

    #[test]
    fn test_vec4_dot() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(2.0, 3.0, 4.0, 5.0);
        assert_eq!(
            a.dot(b),
            4.0f32.mul_add(5.0, 3.0f32.mul_add(4.0, 1.0f32.mul_add(2.0, 2.0 * 3.0)))
        );
    }

    #[test]
    fn test_vec4_add() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(5.0, 6.0, 7.0, 8.0);
        let c = a.add(b);
        assert_eq!(c, Vec4::new(6.0, 8.0, 10.0, 12.0));
    }

    #[test]
    fn test_vec4_sub() {
        let a = Vec4::new(5.0, 6.0, 7.0, 8.0);
        let b = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let c = a.sub(b);
        assert_eq!(c, Vec4::new(4.0, 4.0, 4.0, 4.0));
    }

    #[test]
    fn test_vec4_scale() {
        let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let s = v.scale(2.0);
        assert_eq!(s, Vec4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_vec4_mul() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(2.0, 2.0, 2.0, 2.0);
        let c = a.mul(b);
        assert_eq!(c, Vec4::new(2.0, 4.0, 6.0, 8.0));
    }

    #[test]
    fn test_vec4_lerp() {
        let a = Vec4::new(0.0, 0.0, 0.0, 0.0);
        let b = Vec4::new(10.0, 10.0, 10.0, 10.0);
        let c = a.lerp(b, 0.5);
        assert_eq!(c, Vec4::new(5.0, 5.0, 5.0, 5.0));
    }

    #[test]
    fn test_vec4_length() {
        let v = Vec4::new(3.0, 4.0, 0.0, 0.0);
        assert!((v.length() - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec4_normalize() {
        let v = Vec4::new(3.0, 4.0, 0.0, 0.0);
        let n = v.normalize();
        assert!((n.length() - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_vec4_from_impl() {
        let p = Point::new(1.0, 2.0);
        let v: Vec4 = p.into();
        assert_eq!(v, Vec4::new(1.0, 2.0, 0.0, 1.0));
    }

    // =========================================================================
    // Mat4 Tests
    // =========================================================================

    #[test]
    fn test_mat4_identity() {
        let m = Mat4::identity();
        assert_eq!(m.data[0][0], 1.0);
        assert_eq!(m.data[1][1], 1.0);
        assert_eq!(m.data[2][2], 1.0);
        assert_eq!(m.data[3][3], 1.0);
        assert_eq!(m.data[0][1], 0.0);
    }

    #[test]
    fn test_mat4_zero() {
        let m = Mat4::zero();
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(m.data[i][j], 0.0);
            }
        }
    }

    #[test]
    fn test_mat4_default() {
        let m = Mat4::default();
        assert_eq!(m, Mat4::identity());
    }

    #[test]
    fn test_mat4_translation() {
        let m = Mat4::translation(10.0, 20.0, 30.0);
        let p = Point::new(0.0, 0.0);
        let t = m.transform_point(p);
        assert_eq!(t, Point::new(10.0, 20.0));
    }

    #[test]
    fn test_mat4_translation_2d() {
        let m = Mat4::translation_2d(5.0, 15.0);
        let p = Point::new(10.0, 10.0);
        let t = m.transform_point(p);
        assert_eq!(t, Point::new(15.0, 25.0));
    }

    #[test]
    fn test_mat4_scale() {
        let m = Mat4::scale(2.0, 3.0, 4.0);
        let p = Point::new(10.0, 10.0);
        let t = m.transform_point(p);
        assert_eq!(t, Point::new(20.0, 30.0));
    }

    #[test]
    fn test_mat4_scale_2d() {
        let m = Mat4::scale_2d(0.5, 2.0);
        let p = Point::new(10.0, 10.0);
        let t = m.transform_point(p);
        assert_eq!(t, Point::new(5.0, 20.0));
    }

    #[test]
    fn test_mat4_scale_uniform() {
        let m = Mat4::scale_uniform(2.0);
        let p = Point::new(5.0, 5.0);
        let t = m.transform_point(p);
        assert_eq!(t, Point::new(10.0, 10.0));
    }

    #[test]
    fn test_mat4_rotation_z() {
        use std::f32::consts::PI;
        let m = Mat4::rotation_z(PI / 2.0); // 90 degrees
        let p = Point::new(1.0, 0.0);
        let t = m.transform_point(p);
        // Should rotate (1, 0) to approximately (0, 1)
        assert!((t.x - 0.0).abs() < 0.0001);
        assert!((t.y - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_mat4_mul_identity() {
        let a = Mat4::identity();
        let b = Mat4::identity();
        let c = a.mul(&b);
        assert_eq!(c, Mat4::identity());
    }

    #[test]
    fn test_mat4_mul_combined_transform() {
        let translate = Mat4::translation_2d(10.0, 0.0);
        let scale = Mat4::scale_2d(2.0, 2.0);
        let combined = translate.mul(&scale);

        let p = Point::new(5.0, 5.0);
        let t = combined.transform_point(p);
        // First scale (5,5) -> (10, 10), then translate -> (20, 10)
        assert_eq!(t, Point::new(20.0, 10.0));
    }

    #[test]
    fn test_mat4_transform_rect() {
        let m = Mat4::scale_2d(2.0, 2.0);
        let rect = Rect::new(10.0, 10.0, 20.0, 30.0);
        let t = m.transform_rect(&rect);
        assert_eq!(t.x, 20.0);
        assert_eq!(t.y, 20.0);
        assert_eq!(t.width, 40.0);
        assert_eq!(t.height, 60.0);
    }

    #[test]
    fn test_mat4_transpose() {
        let m = Mat4::from_data([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let t = m.transpose();
        assert_eq!(t.data[0][1], 5.0);
        assert_eq!(t.data[1][0], 2.0);
    }

    #[test]
    fn test_mat4_column() {
        let m = Mat4::identity();
        let col = m.column(0);
        assert_eq!(col, Vec4::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_mat4_row() {
        let m = Mat4::identity();
        let row = m.row(0);
        assert_eq!(row, Vec4::new(1.0, 0.0, 0.0, 0.0));
    }

    #[test]
    fn test_mat4_ortho_screen() {
        let m = Mat4::ortho_screen(800.0, 600.0);
        // Point at top-left should map to (-1, 1) in NDC
        let p = m.transform_vec4(Vec4::new(0.0, 0.0, 0.0, 1.0));
        assert!((p.x - (-1.0)).abs() < 0.001);
        assert!((p.y - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mat4_mul_operator() {
        let a = Mat4::translation_2d(10.0, 20.0);
        let b = Mat4::scale_2d(2.0, 2.0);
        let c = a * b;
        assert_eq!(c, a.mul(&b));
    }

    #[test]
    fn test_mat4_mul_vec4_operator() {
        let m = Mat4::translation_2d(10.0, 20.0);
        let v = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let r = m * v;
        assert_eq!(r, Vec4::new(10.0, 20.0, 0.0, 1.0));
    }

    // =========================================================================
    // Batch Operation Tests
    // =========================================================================

    #[test]
    fn test_batch_transform_points() {
        let m = Mat4::translation_2d(10.0, 10.0);
        let points = vec![Point::new(0.0, 0.0), Point::new(5.0, 5.0)];
        let result = batch_transform_points(&points, &m);
        assert_eq!(result[0], Point::new(10.0, 10.0));
        assert_eq!(result[1], Point::new(15.0, 15.0));
    }

    #[test]
    fn test_batch_transform_vec4() {
        let m = Mat4::scale_uniform(2.0);
        let vecs = vec![Vec4::new(1.0, 1.0, 1.0, 0.0), Vec4::new(2.0, 2.0, 2.0, 0.0)];
        let result = batch_transform_vec4(&vecs, &m);
        assert_eq!(result[0], Vec4::new(2.0, 2.0, 2.0, 0.0));
        assert_eq!(result[1], Vec4::new(4.0, 4.0, 4.0, 0.0));
    }

    #[test]
    fn test_batch_lerp_points() {
        let from = vec![Point::new(0.0, 0.0), Point::new(10.0, 10.0)];
        let to = vec![Point::new(10.0, 10.0), Point::new(20.0, 20.0)];
        let result = batch_lerp_points(&from, &to, 0.5);
        assert_eq!(result[0], Point::new(5.0, 5.0));
        assert_eq!(result[1], Point::new(15.0, 15.0));
    }

    // =========================================================================
    // Geometry Tests
    // =========================================================================

    #[test]
    fn test_bounding_box() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 5.0),
            Point::new(5.0, 15.0),
        ];
        let bbox = bounding_box(&points).unwrap();
        assert_eq!(bbox.x, 0.0);
        assert_eq!(bbox.y, 0.0);
        assert_eq!(bbox.width, 10.0);
        assert_eq!(bbox.height, 15.0);
    }

    #[test]
    fn test_bounding_box_empty() {
        let points: Vec<Point> = vec![];
        assert!(bounding_box(&points).is_none());
    }

    #[test]
    fn test_centroid() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let c = centroid(&points).unwrap();
        assert_eq!(c, Point::new(5.0, 5.0));
    }

    #[test]
    fn test_centroid_empty() {
        let points: Vec<Point> = vec![];
        assert!(centroid(&points).is_none());
    }

    #[test]
    fn test_point_in_convex_polygon() {
        let square = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];

        assert!(point_in_convex_polygon(Point::new(5.0, 5.0), &square));
        assert!(!point_in_convex_polygon(Point::new(15.0, 5.0), &square));
    }

    #[test]
    fn test_point_in_convex_polygon_edge() {
        let triangle = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(5.0, 10.0),
        ];

        // On the edge
        assert!(point_in_convex_polygon(Point::new(5.0, 0.0), &triangle));
    }

    #[test]
    fn test_polygon_area_square() {
        let square = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ];
        let area = polygon_area(&square);
        assert!((area - 100.0).abs() < 0.0001);
    }

    #[test]
    fn test_polygon_area_triangle() {
        let triangle = vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(5.0, 10.0),
        ];
        let area = polygon_area(&triangle);
        assert!((area - 50.0).abs() < 0.0001);
    }

    #[test]
    fn test_polygon_area_too_few_points() {
        assert_eq!(polygon_area(&[]), 0.0);
        assert_eq!(polygon_area(&[Point::new(0.0, 0.0)]), 0.0);
        assert_eq!(
            polygon_area(&[Point::new(0.0, 0.0), Point::new(1.0, 1.0)]),
            0.0
        );
    }

    // =========================================================================
    // SIMD Tests (when feature enabled)
    // =========================================================================

    #[cfg(feature = "simd")]
    mod simd_tests {
        use super::*;

        #[test]
        fn test_vec4_to_simd_roundtrip() {
            let v = Vec4::new(1.0, 2.0, 3.0, 4.0);
            let simd = vec4_to_simd(v);
            let back = simd_to_vec4(&simd);
            assert_eq!(v, back);
        }

        #[test]
        fn test_batch_add_simd() {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![5.0, 6.0, 7.0, 8.0];
            let result = batch_add_simd(&a, &b).unwrap();
            assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
        }

        #[test]
        fn test_dot_simd() {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let b = vec![1.0, 1.0, 1.0, 1.0];
            let result = dot_simd(&a, &b).unwrap();
            assert_eq!(result, 10.0);
        }

        #[test]
        fn test_scale_simd() {
            let a = vec![1.0, 2.0, 3.0, 4.0];
            let result = scale_simd(&a, 2.0).unwrap();
            assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
        }

        #[test]
        fn test_best_backend() {
            let backend = best_backend();
            // Just check it doesn't panic and returns a valid backend
            assert!(matches!(
                backend,
                trueno::Backend::Scalar
                    | trueno::Backend::SSE2
                    | trueno::Backend::AVX
                    | trueno::Backend::AVX2
                    | trueno::Backend::AVX512
                    | trueno::Backend::NEON
                    | trueno::Backend::WasmSIMD
                    | trueno::Backend::GPU
                    | trueno::Backend::Auto
            ));
        }

        #[test]
        fn test_batch_dot_product() {
            let a = vec![Vec4::new(1.0, 0.0, 0.0, 0.0), Vec4::new(0.0, 1.0, 0.0, 0.0)];
            let b = vec![Vec4::new(1.0, 0.0, 0.0, 0.0), Vec4::new(0.0, 1.0, 0.0, 0.0)];
            let dots = batch_dot_product(&a, &b);
            assert_eq!(dots, vec![1.0, 1.0]);
        }
    }

    // =========================================================================
    // ComputeBlock Tests (f64 Aggregation)
    // =========================================================================

    mod compute_block_tests {
        use super::*;

        #[test]
        fn test_batch_sum_f64_empty() {
            assert_eq!(batch_sum_f64(&[]), 0.0);
        }

        #[test]
        fn test_batch_sum_f64_small() {
            assert_eq!(batch_sum_f64(&[1.0, 2.0, 3.0]), 6.0);
        }

        #[test]
        fn test_batch_sum_f64_large() {
            let values: Vec<f64> = (1..=100).map(f64::from).collect();
            assert_eq!(batch_sum_f64(&values), 5050.0);
        }

        #[test]
        fn test_batch_sum_f64_exact_chunks() {
            // 8 elements = 2 exact chunks of 4
            let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            assert_eq!(batch_sum_f64(&values), 36.0);
        }

        #[test]
        fn test_batch_mean_f64_empty() {
            assert_eq!(batch_mean_f64(&[]), 0.0);
        }

        #[test]
        fn test_batch_mean_f64() {
            assert_eq!(batch_mean_f64(&[2.0, 4.0, 6.0, 8.0]), 5.0);
        }

        #[test]
        fn test_batch_min_max_f64_empty() {
            assert!(batch_min_max_f64(&[]).is_none());
        }

        #[test]
        fn test_batch_min_max_f64_single() {
            assert_eq!(batch_min_max_f64(&[42.0]), Some((42.0, 42.0)));
        }

        #[test]
        fn test_batch_min_max_f64_small() {
            assert_eq!(batch_min_max_f64(&[3.0, 1.0, 2.0]), Some((1.0, 3.0)));
        }

        #[test]
        fn test_batch_min_max_f64_large() {
            let values: Vec<f64> = (1..=256).map(f64::from).collect();
            assert_eq!(batch_min_max_f64(&values), Some((1.0, 256.0)));
        }

        #[test]
        fn test_normalize_f64_empty() {
            assert!(normalize_f64(&[]).is_empty());
        }

        #[test]
        fn test_normalize_f64_constant() {
            let result = normalize_f64(&[5.0, 5.0, 5.0]);
            assert_eq!(result, vec![0.0, 0.0, 0.0]);
        }

        #[test]
        fn test_normalize_f64() {
            let result = normalize_f64(&[0.0, 50.0, 100.0]);
            assert_eq!(result, vec![0.0, 0.5, 1.0]);
        }

        #[test]
        fn test_normalize_with_range_f64() {
            let result = normalize_with_range_f64(&[25.0, 50.0, 75.0], 0.0, 100.0);
            assert_eq!(result, vec![0.25, 0.5, 0.75]);
        }

        #[test]
        fn test_batch_scale_f64() {
            let result = batch_scale_f64(&[1.0, 2.0, 3.0], 2.0);
            assert_eq!(result, vec![2.0, 4.0, 6.0]);
        }

        #[test]
        fn test_batch_scale_offset_f64() {
            let result = batch_scale_offset_f64(&[0.0, 0.5, 1.0], 100.0, 100.0);
            assert_eq!(result, vec![100.0, 150.0, 200.0]);
        }

        #[test]
        fn test_batch_variance_f64_empty() {
            assert_eq!(batch_variance_f64(&[]), 0.0);
        }

        #[test]
        fn test_batch_variance_f64_single() {
            assert_eq!(batch_variance_f64(&[42.0]), 0.0);
        }

        #[test]
        fn test_batch_variance_f64() {
            let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
            let variance = batch_variance_f64(&values);
            assert!((variance - 4.0).abs() < 0.001);
        }

        #[test]
        fn test_batch_stddev_f64() {
            let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
            let stddev = batch_stddev_f64(&values);
            assert!((stddev - 2.0).abs() < 0.001);
        }

        #[test]
        fn test_weighted_sum_f64_small() {
            let values = vec![1.0, 2.0];
            let weights = vec![0.5, 0.5];
            assert_eq!(weighted_sum_f64(&values, &weights), 1.5);
        }

        #[test]
        fn test_weighted_sum_f64_large() {
            let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let weights = vec![1.0; 8];
            assert_eq!(weighted_sum_f64(&values, &weights), 36.0);
        }

        #[test]
        fn test_percentile_sorted_f64_empty() {
            assert_eq!(percentile_sorted_f64(&[], 0.5), 0.0);
        }

        #[test]
        fn test_percentile_sorted_f64_single() {
            assert_eq!(percentile_sorted_f64(&[42.0], 0.5), 42.0);
        }

        #[test]
        fn test_percentile_sorted_f64_median() {
            let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            assert_eq!(percentile_sorted_f64(&sorted, 0.5), 3.0);
        }

        #[test]
        fn test_percentile_sorted_f64_quartiles() {
            let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
            assert_eq!(percentile_sorted_f64(&sorted, 0.0), 1.0);
            assert_eq!(percentile_sorted_f64(&sorted, 1.0), 5.0);
            assert_eq!(percentile_sorted_f64(&sorted, 0.25), 2.0);
            assert_eq!(percentile_sorted_f64(&sorted, 0.75), 4.0);
        }

        #[test]
        fn test_histogram_f64_empty() {
            assert_eq!(histogram_f64(&[], 0.0, 10.0, 5), vec![0; 5]);
        }

        #[test]
        fn test_histogram_f64() {
            let values = vec![0.1, 0.5, 0.9, 1.5, 2.5, 3.5, 4.5];
            let counts = histogram_f64(&values, 0.0, 5.0, 5);
            assert_eq!(counts, vec![3, 1, 1, 1, 1]);
        }

        #[test]
        fn test_histogram_f64_edge_values() {
            // Values exactly at bin boundaries
            let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 4.999];
            let counts = histogram_f64(&values, 0.0, 5.0, 5);
            assert_eq!(counts, vec![1, 1, 1, 1, 2]); // 4.0 and 4.999 in last bin
        }

        #[test]
        fn test_histogram_f64_uniform() {
            // Uniform distribution across bins
            let values: Vec<f64> = (0..100).map(f64::from).collect();
            let counts = histogram_f64(&values, 0.0, 100.0, 10);
            // Each bin should have 10 values
            assert!(counts.iter().all(|&c| c == 10));
        }
    }
}
