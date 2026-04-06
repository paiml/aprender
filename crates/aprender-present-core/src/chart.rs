#![allow(clippy::unwrap_used, clippy::disallowed_methods)]
//! Chart rendering algorithms for Presentar.
//!
//! This module provides mathematical algorithms for chart rendering:
//! - Interpolation (linear, cubic spline, Catmull-Rom, Bezier)
//! - Path tessellation for GPU rendering
//! - Histogram binning
//! - Arc geometry computation
//! - Data normalization and scaling
//!
//! # Example
//!
//! ```
//! use presentar_core::chart::{Interpolator, CubicSpline, Point2D};
//!
//! // Create a spline from control points
//! let points = vec![
//!     Point2D::new(0.0, 0.0),
//!     Point2D::new(1.0, 2.0),
//!     Point2D::new(2.0, 1.5),
//!     Point2D::new(3.0, 3.0),
//! ];
//! let spline = CubicSpline::from_points(&points);
//!
//! // Interpolate at any x value
//! let y = spline.interpolate(1.5);
//! ```

use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// 2D point for chart calculations.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Point2D {
    pub x: f64,
    pub y: f64,
}

impl Point2D {
    /// Create a new point.
    #[must_use]
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Origin point (0, 0).
    pub const ORIGIN: Self = Self { x: 0.0, y: 0.0 };

    /// Distance to another point.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        dx.hypot(dy)
    }

    /// Linear interpolation between two points.
    #[must_use]
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            x: (other.x - self.x).mul_add(t, self.x),
            y: (other.y - self.y).mul_add(t, self.y),
        }
    }
}

impl std::ops::Add for Point2D {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl std::ops::Sub for Point2D {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl std::ops::Mul<f64> for Point2D {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

/// Interpolation trait for different curve types.
pub trait Interpolator {
    /// Interpolate y value at given x.
    fn interpolate(&self, x: f64) -> f64;

    /// Generate points along the curve.
    fn sample(&self, start: f64, end: f64, num_points: usize) -> Vec<Point2D> {
        if num_points < 2 {
            return vec![];
        }
        let step = (end - start) / (num_points - 1) as f64;
        (0..num_points)
            .map(|i| {
                let x = (i as f64).mul_add(step, start);
                Point2D::new(x, self.interpolate(x))
            })
            .collect()
    }
}

/// Linear interpolation between points.
#[derive(Debug, Clone)]
pub struct LinearInterpolator {
    points: Vec<Point2D>,
}

impl LinearInterpolator {
    /// Create from points (must be sorted by x).
    #[must_use]
    pub fn from_points(points: &[Point2D]) -> Self {
        let mut sorted = points.to_vec();
        sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        Self { points: sorted }
    }

    /// Create from x,y data.
    #[must_use]
    pub fn from_xy(xs: &[f64], ys: &[f64]) -> Self {
        let points: Vec<_> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| Point2D::new(x, y))
            .collect();
        Self::from_points(&points)
    }

    /// Get the underlying points.
    #[must_use]
    pub fn points(&self) -> &[Point2D] {
        &self.points
    }

    /// Find the segment containing x.
    fn find_segment(&self, x: f64) -> Option<(usize, f64)> {
        if self.points.len() < 2 {
            return None;
        }
        for i in 0..self.points.len() - 1 {
            let p1 = &self.points[i];
            let p2 = &self.points[i + 1];
            if x >= p1.x && x <= p2.x {
                let t = if (p2.x - p1.x).abs() < 1e-10 {
                    0.0
                } else {
                    (x - p1.x) / (p2.x - p1.x)
                };
                return Some((i, t));
            }
        }
        // Extrapolate
        if x < self.points[0].x {
            Some((
                0,
                (x - self.points[0].x) / (self.points[1].x - self.points[0].x),
            ))
        } else {
            let n = self.points.len();
            Some((
                n - 2,
                (x - self.points[n - 2].x) / (self.points[n - 1].x - self.points[n - 2].x),
            ))
        }
    }
}

impl Interpolator for LinearInterpolator {
    fn interpolate(&self, x: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        if self.points.len() == 1 {
            return self.points[0].y;
        }

        if let Some((i, t)) = self.find_segment(x) {
            let p1 = &self.points[i];
            let p2 = &self.points[i + 1];
            (p2.y - p1.y).mul_add(t, p1.y)
        } else {
            0.0
        }
    }
}

/// Cubic spline interpolation (natural spline).
#[derive(Debug, Clone)]
pub struct CubicSpline {
    points: Vec<Point2D>,
    /// Second derivatives at each point
    y2: Vec<f64>,
}

impl CubicSpline {
    /// Create from points (must have at least 3 points).
    #[must_use]
    pub fn from_points(points: &[Point2D]) -> Self {
        let mut sorted = points.to_vec();
        sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted.len();
        if n < 3 {
            return Self {
                points: sorted,
                y2: vec![0.0; n],
            };
        }

        // Compute second derivatives using natural spline boundary conditions
        let mut y2 = vec![0.0; n];
        let mut u = vec![0.0; n];

        // Forward sweep
        for i in 1..n - 1 {
            let h_prev = sorted[i].x - sorted[i - 1].x;
            let h_next = sorted[i + 1].x - sorted[i].x;

            if h_prev.abs() < 1e-10 || h_next.abs() < 1e-10 {
                continue;
            }

            let sig = h_prev / (h_prev + h_next);
            let p = sig.mul_add(y2[i - 1], 2.0);
            y2[i] = (sig - 1.0) / p;
            u[i] =
                (sorted[i + 1].y - sorted[i].y) / h_next - (sorted[i].y - sorted[i - 1].y) / h_prev;
            u[i] = sig.mul_add(-u[i - 1], 6.0 * u[i] / (h_prev + h_next)) / p;
        }

        // Back substitution
        for k in (0..n - 1).rev() {
            y2[k] = y2[k].mul_add(y2[k + 1], u[k]);
        }

        Self { points: sorted, y2 }
    }

    /// Create from x,y data.
    #[must_use]
    pub fn from_xy(xs: &[f64], ys: &[f64]) -> Self {
        let points: Vec<_> = xs
            .iter()
            .zip(ys.iter())
            .map(|(&x, &y)| Point2D::new(x, y))
            .collect();
        Self::from_points(&points)
    }

    /// Get the underlying points.
    #[must_use]
    pub fn points(&self) -> &[Point2D] {
        &self.points
    }
}

impl Interpolator for CubicSpline {
    fn interpolate(&self, x: f64) -> f64 {
        let n = self.points.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return self.points[0].y;
        }
        if n == 2 {
            // Fall back to linear
            let t = (x - self.points[0].x) / (self.points[1].x - self.points[0].x);
            return (self.points[1].y - self.points[0].y).mul_add(t, self.points[0].y);
        }

        // Find segment
        let mut lo = 0;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (hi + lo) / 2;
            if self.points[mid].x > x {
                hi = mid;
            } else {
                lo = mid;
            }
        }

        let h = self.points[hi].x - self.points[lo].x;
        if h.abs() < 1e-10 {
            return self.points[lo].y;
        }

        let a = (self.points[hi].x - x) / h;
        let b = (x - self.points[lo].x) / h;

        a.mul_add(self.points[lo].y, b * self.points[hi].y)
            + (a * a)
                .mul_add(a, -a)
                .mul_add(self.y2[lo], (b * b).mul_add(b, -b) * self.y2[hi])
                * h
                * h
                / 6.0
    }
}

/// Catmull-Rom spline interpolation.
#[derive(Debug, Clone)]
pub struct CatmullRom {
    points: Vec<Point2D>,
    /// Tension parameter (0.0 to 1.0)
    tension: f64,
}

impl CatmullRom {
    /// Create from points with default tension.
    #[must_use]
    pub fn from_points(points: &[Point2D]) -> Self {
        Self::with_tension(points, 0.5)
    }

    /// Create from points with custom tension.
    #[must_use]
    pub fn with_tension(points: &[Point2D], tension: f64) -> Self {
        let mut sorted = points.to_vec();
        sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
        Self {
            points: sorted,
            tension: tension.clamp(0.0, 1.0),
        }
    }

    /// Get the underlying points.
    #[must_use]
    pub fn points(&self) -> &[Point2D] {
        &self.points
    }

    /// Generate a smooth path through the points.
    #[must_use]
    pub fn to_path(&self, segments_per_span: usize) -> Vec<Point2D> {
        if self.points.len() < 2 {
            return self.points.clone();
        }
        if self.points.len() == 2 {
            return self.points.clone();
        }

        let mut path = Vec::new();
        let n = self.points.len();

        for i in 0..n - 1 {
            let p0 = if i == 0 {
                self.points[0]
            } else {
                self.points[i - 1]
            };
            let p1 = self.points[i];
            let p2 = self.points[i + 1];
            let p3 = if i + 2 < n {
                self.points[i + 2]
            } else {
                self.points[n - 1]
            };

            for j in 0..segments_per_span {
                let t = j as f64 / segments_per_span as f64;
                let point = self.catmull_rom_point(p0, p1, p2, p3, t);
                path.push(point);
            }
        }

        // Add final point
        path.push(self.points[n - 1]);
        path
    }

    /// Compute point on Catmull-Rom curve.
    fn catmull_rom_point(
        &self,
        p0: Point2D,
        p1: Point2D,
        p2: Point2D,
        p3: Point2D,
        t: f64,
    ) -> Point2D {
        let t2 = t * t;
        let t3 = t2 * t;

        let tau = self.tension;

        // Catmull-Rom basis matrix with tension
        let c0 = tau.mul_add(-t, (-tau).mul_add(t3, 2.0 * tau * t2));
        let c1 = (2.0 - tau).mul_add(t3, (tau - 3.0) * t2) + 1.0;
        let c2 = tau.mul_add(t, (tau - 2.0).mul_add(t3, 2.0f64.mul_add(-tau, 3.0) * t2));
        let c3 = tau.mul_add(t3, -(tau * t2));

        Point2D::new(
            c3.mul_add(p3.x, c0 * p0.x + c1 * p1.x + c2 * p2.x),
            c3.mul_add(p3.y, c0 * p0.y + c1 * p1.y + c2 * p2.y),
        )
    }
}

impl Interpolator for CatmullRom {
    fn interpolate(&self, x: f64) -> f64 {
        // For Catmull-Rom, we need to find the segment and interpolate
        if self.points.is_empty() {
            return 0.0;
        }
        if self.points.len() == 1 {
            return self.points[0].y;
        }

        // Find segment containing x
        let mut idx = 0;
        for i in 0..self.points.len() - 1 {
            if x >= self.points[i].x && x <= self.points[i + 1].x {
                idx = i;
                break;
            }
            if x < self.points[i].x {
                idx = i.saturating_sub(1);
                break;
            }
            idx = i;
        }

        let p1 = &self.points[idx];
        let p2 = &self.points[(idx + 1).min(self.points.len() - 1)];

        let t = if (p2.x - p1.x).abs() < 1e-10 {
            0.0
        } else {
            ((x - p1.x) / (p2.x - p1.x)).clamp(0.0, 1.0)
        };

        let p0 = if idx == 0 { *p1 } else { self.points[idx - 1] };
        let p3 = if idx + 2 < self.points.len() {
            self.points[idx + 2]
        } else {
            *p2
        };

        self.catmull_rom_point(p0, *p1, *p2, p3, t).y
    }
}

/// Bezier curve segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubicBezier {
    /// Start point
    pub p0: Point2D,
    /// First control point
    pub p1: Point2D,
    /// Second control point
    pub p2: Point2D,
    /// End point
    pub p3: Point2D,
}

impl CubicBezier {
    /// Create a new cubic Bezier curve.
    #[must_use]
    pub const fn new(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D) -> Self {
        Self { p0, p1, p2, p3 }
    }

    /// Evaluate the curve at parameter t (0 to 1).
    #[must_use]
    pub fn evaluate(&self, t: f64) -> Point2D {
        let t = t.clamp(0.0, 1.0);
        let mt = 1.0 - t;
        let mt2 = mt * mt;
        let mt3 = mt2 * mt;
        let t2 = t * t;
        let t3 = t2 * t;

        Point2D::new(
            (3.0 * mt * t2).mul_add(self.p2.x, mt3 * self.p0.x + 3.0 * mt2 * t * self.p1.x)
                + t3 * self.p3.x,
            (3.0 * mt * t2).mul_add(self.p2.y, mt3 * self.p0.y + 3.0 * mt2 * t * self.p1.y)
                + t3 * self.p3.y,
        )
    }

    /// Convert to a polyline with given number of segments.
    #[must_use]
    pub fn to_polyline(&self, segments: usize) -> Vec<Point2D> {
        let segments = segments.max(1);
        (0..=segments)
            .map(|i| self.evaluate(i as f64 / segments as f64))
            .collect()
    }

    /// Compute approximate arc length.
    #[must_use]
    pub fn arc_length(&self, segments: usize) -> f64 {
        let points = self.to_polyline(segments);
        points.windows(2).map(|w| w[0].distance(&w[1])).sum()
    }

    /// Split curve at parameter t.
    #[must_use]
    pub fn split(&self, t: f64) -> (Self, Self) {
        let t = t.clamp(0.0, 1.0);

        // De Casteljau's algorithm
        let p01 = self.p0.lerp(&self.p1, t);
        let p12 = self.p1.lerp(&self.p2, t);
        let p23 = self.p2.lerp(&self.p3, t);

        let p012 = p01.lerp(&p12, t);
        let p123 = p12.lerp(&p23, t);

        let p0123 = p012.lerp(&p123, t);

        let left = Self::new(self.p0, p01, p012, p0123);
        let right = Self::new(p0123, p123, p23, self.p3);

        (left, right)
    }
}

/// Histogram binning configuration.
#[derive(Debug, Clone)]
pub struct HistogramBins {
    /// Bin edges (n+1 edges for n bins)
    pub edges: Vec<f64>,
    /// Bin counts
    pub counts: Vec<usize>,
    /// Bin densities (normalized)
    pub densities: Vec<f64>,
}

impl HistogramBins {
    /// Create histogram from data with specified number of bins.
    #[must_use]
    pub fn from_data(data: &[f64], num_bins: usize) -> Self {
        if data.is_empty() || num_bins == 0 {
            return Self {
                edges: vec![],
                counts: vec![],
                densities: vec![],
            };
        }

        let num_bins = num_bins.max(1);
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        Self::from_data_range(data, num_bins, min, max)
    }

    /// Create histogram from data with explicit range.
    #[must_use]
    pub fn from_data_range(data: &[f64], num_bins: usize, min: f64, max: f64) -> Self {
        let num_bins = num_bins.max(1);
        let range = (max - min).max(1e-10);
        let bin_width = range / num_bins as f64;

        // Create edges
        let edges: Vec<f64> = (0..=num_bins)
            .map(|i| (i as f64).mul_add(bin_width, min))
            .collect();

        // Count values in each bin
        let mut counts = vec![0usize; num_bins];
        for &value in data {
            let bin = ((value - min) / bin_width).floor() as usize;
            let bin = bin.min(num_bins - 1); // Handle edge case where value == max
            counts[bin] += 1;
        }

        // Compute densities (probability density)
        let total = data.len() as f64;
        let densities: Vec<f64> = counts
            .iter()
            .map(|&c| (c as f64) / (total * bin_width))
            .collect();

        Self {
            edges,
            counts,
            densities,
        }
    }

    /// Get number of bins.
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.counts.len()
    }

    /// Get bin width (assumes uniform bins).
    #[must_use]
    pub fn bin_width(&self) -> f64 {
        if self.edges.len() < 2 {
            return 0.0;
        }
        self.edges[1] - self.edges[0]
    }

    /// Get bin center for given index.
    #[must_use]
    pub fn bin_center(&self, index: usize) -> Option<f64> {
        if index >= self.counts.len() {
            return None;
        }
        Some((self.edges[index] + self.edges[index + 1]) / 2.0)
    }

    /// Get bin range for given index.
    #[must_use]
    pub fn bin_range(&self, index: usize) -> Option<(f64, f64)> {
        if index >= self.counts.len() {
            return None;
        }
        Some((self.edges[index], self.edges[index + 1]))
    }

    /// Total count across all bins.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.counts.iter().sum()
    }

    /// Maximum count in any bin.
    #[must_use]
    pub fn max_count(&self) -> usize {
        self.counts.iter().copied().max().unwrap_or(0)
    }
}

/// Arc geometry for pie charts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArcGeometry {
    /// Center point
    pub center: Point2D,
    /// Radius
    pub radius: f64,
    /// Start angle (radians)
    pub start_angle: f64,
    /// End angle (radians)
    pub end_angle: f64,
}

impl ArcGeometry {
    /// Create a new arc.
    #[must_use]
    pub const fn new(center: Point2D, radius: f64, start_angle: f64, end_angle: f64) -> Self {
        Self {
            center,
            radius,
            start_angle,
            end_angle,
        }
    }

    /// Create a full circle.
    #[must_use]
    pub fn circle(center: Point2D, radius: f64) -> Self {
        Self::new(center, radius, 0.0, 2.0 * PI)
    }

    /// Get the sweep angle.
    #[must_use]
    pub fn sweep(&self) -> f64 {
        self.end_angle - self.start_angle
    }

    /// Point on arc at given angle.
    #[must_use]
    pub fn point_at_angle(&self, angle: f64) -> Point2D {
        Point2D::new(
            self.radius.mul_add(angle.cos(), self.center.x),
            self.radius.mul_add(angle.sin(), self.center.y),
        )
    }

    /// Start point of arc.
    #[must_use]
    pub fn start_point(&self) -> Point2D {
        self.point_at_angle(self.start_angle)
    }

    /// End point of arc.
    #[must_use]
    pub fn end_point(&self) -> Point2D {
        self.point_at_angle(self.end_angle)
    }

    /// Midpoint of arc.
    #[must_use]
    pub fn mid_point(&self) -> Point2D {
        let mid_angle = (self.start_angle + self.end_angle) / 2.0;
        self.point_at_angle(mid_angle)
    }

    /// Arc length.
    #[must_use]
    pub fn arc_length(&self) -> f64 {
        self.radius * self.sweep().abs()
    }

    /// Convert arc to polyline for rendering.
    #[must_use]
    pub fn to_polyline(&self, segments: usize) -> Vec<Point2D> {
        let segments = segments.max(1);
        let sweep = self.sweep();
        (0..=segments)
            .map(|i| {
                let t = i as f64 / segments as f64;
                let angle = self.start_angle + t * sweep;
                self.point_at_angle(angle)
            })
            .collect()
    }

    /// Convert arc to pie slice (includes center point).
    #[must_use]
    pub fn to_pie_slice(&self, segments: usize) -> Vec<Point2D> {
        let mut points = vec![self.center];
        points.extend(self.to_polyline(segments));
        points.push(self.center);
        points
    }

    /// Check if angle is within arc sweep.
    #[must_use]
    pub fn contains_angle(&self, angle: f64) -> bool {
        let normalized = Self::normalize_angle(angle);
        let start = Self::normalize_angle(self.start_angle);
        let end = Self::normalize_angle(self.end_angle);

        if start <= end {
            normalized >= start && normalized <= end
        } else {
            normalized >= start || normalized <= end
        }
    }

    /// Normalize angle to [0, 2π).
    fn normalize_angle(angle: f64) -> f64 {
        let mut a = angle % (2.0 * PI);
        if a < 0.0 {
            a += 2.0 * PI;
        }
        a
    }
}

/// Data normalization for chart rendering.
#[derive(Debug, Clone, Copy)]
pub struct DataNormalizer {
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
}

impl DataNormalizer {
    /// Create normalizer from data range.
    #[must_use]
    pub fn new(min: f64, max: f64) -> Self {
        Self { min, max }
    }

    /// Create normalizer from data.
    #[must_use]
    pub fn from_data(data: &[f64]) -> Self {
        if data.is_empty() {
            return Self::new(0.0, 1.0);
        }
        let min = data.iter().copied().fold(f64::INFINITY, f64::min);
        let max = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::new(min, max)
    }

    /// Normalize a value to [0, 1].
    #[must_use]
    pub fn normalize(&self, value: f64) -> f64 {
        let range = self.max - self.min;
        if range.abs() < 1e-10 {
            return 0.5;
        }
        (value - self.min) / range
    }

    /// Denormalize a value from [0, 1] to original range.
    #[must_use]
    pub fn denormalize(&self, normalized: f64) -> f64 {
        normalized.mul_add(self.max - self.min, self.min)
    }

    /// Normalize all values in a slice.
    #[must_use]
    pub fn normalize_all(&self, data: &[f64]) -> Vec<f64> {
        data.iter().map(|&v| self.normalize(v)).collect()
    }

    /// Get nice axis bounds (rounded for display).
    #[must_use]
    pub fn nice_bounds(&self) -> (f64, f64) {
        let range = self.max - self.min;
        if range.abs() < 1e-10 {
            return (self.min - 1.0, self.max + 1.0);
        }

        let magnitude = 10.0_f64.powf(range.log10().floor());
        let nice_min = (self.min / magnitude).floor() * magnitude;
        let nice_max = (self.max / magnitude).ceil() * magnitude;

        (nice_min, nice_max)
    }
}

/// Path tessellation for GPU rendering.
#[derive(Debug, Clone, Default)]
pub struct PathTessellator {
    /// Tolerance for curve flattening
    pub tolerance: f64,
    /// Generated vertices (x, y)
    pub vertices: Vec<[f32; 2]>,
    /// Triangle indices
    pub indices: Vec<u32>,
}

impl PathTessellator {
    /// Create a new tessellator.
    #[must_use]
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance: tolerance.max(0.001),
            vertices: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Create with default tolerance.
    #[must_use]
    pub fn with_default_tolerance() -> Self {
        Self::new(0.25)
    }

    /// Clear the tessellator.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    /// Tessellate a filled polygon.
    pub fn tessellate_polygon(&mut self, points: &[Point2D]) {
        if points.len() < 3 {
            return;
        }

        let base_idx = self.vertices.len() as u32;

        // Add vertices
        for p in points {
            self.vertices.push([p.x as f32, p.y as f32]);
        }

        // Fan triangulation (simple, works for convex polygons)
        for i in 1..points.len() as u32 - 1 {
            self.indices.push(base_idx);
            self.indices.push(base_idx + i);
            self.indices.push(base_idx + i + 1);
        }
    }

    /// Tessellate a stroked polyline.
    pub fn tessellate_stroke(&mut self, points: &[Point2D], width: f64) {
        if points.len() < 2 {
            return;
        }

        let half_width = width / 2.0;

        for window in points.windows(2) {
            let p1 = window[0];
            let p2 = window[1];

            // Compute perpendicular direction
            let dx = p2.x - p1.x;
            let dy = p2.y - p1.y;
            let len = dx.hypot(dy);
            if len < 1e-10 {
                continue;
            }

            let nx = -dy / len * half_width;
            let ny = dx / len * half_width;

            let base_idx = self.vertices.len() as u32;

            // Add quad vertices
            self.vertices.push([(p1.x + nx) as f32, (p1.y + ny) as f32]);
            self.vertices.push([(p1.x - nx) as f32, (p1.y - ny) as f32]);
            self.vertices.push([(p2.x + nx) as f32, (p2.y + ny) as f32]);
            self.vertices.push([(p2.x - nx) as f32, (p2.y - ny) as f32]);

            // Two triangles for quad
            self.indices.push(base_idx);
            self.indices.push(base_idx + 1);
            self.indices.push(base_idx + 2);

            self.indices.push(base_idx + 1);
            self.indices.push(base_idx + 3);
            self.indices.push(base_idx + 2);
        }
    }

    /// Tessellate a circle.
    pub fn tessellate_circle(&mut self, center: Point2D, radius: f64, segments: usize) {
        let segments = segments.max(8);
        let base_idx = self.vertices.len() as u32;

        // Center vertex
        self.vertices.push([center.x as f32, center.y as f32]);

        // Perimeter vertices
        for i in 0..segments {
            let angle = 2.0 * PI * i as f64 / segments as f64;
            let x = radius.mul_add(angle.cos(), center.x);
            let y = radius.mul_add(angle.sin(), center.y);
            self.vertices.push([x as f32, y as f32]);
        }

        // Fan triangles
        for i in 0..segments as u32 {
            self.indices.push(base_idx); // Center
            self.indices.push(base_idx + 1 + i);
            self.indices.push(base_idx + 1 + (i + 1) % segments as u32);
        }
    }

    /// Tessellate a rectangle.
    pub fn tessellate_rect(&mut self, x: f64, y: f64, width: f64, height: f64) {
        let base_idx = self.vertices.len() as u32;

        self.vertices.push([x as f32, y as f32]);
        self.vertices.push([(x + width) as f32, y as f32]);
        self.vertices
            .push([(x + width) as f32, (y + height) as f32]);
        self.vertices.push([x as f32, (y + height) as f32]);

        // Two triangles
        self.indices.push(base_idx);
        self.indices.push(base_idx + 1);
        self.indices.push(base_idx + 2);

        self.indices.push(base_idx);
        self.indices.push(base_idx + 2);
        self.indices.push(base_idx + 3);
    }

    /// Get vertex count.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Get index count.
    #[must_use]
    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    /// Get triangle count.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }
}

/// Draw call batching for GPU efficiency.
#[derive(Debug, Clone, Default)]
pub struct DrawBatch {
    /// Batched circles (`center_x`, `center_y`, radius, `color_rgba`)
    pub circles: Vec<[f32; 7]>,
    /// Batched rectangles (x, y, w, h, `color_rgba`)
    pub rects: Vec<[f32; 8]>,
    /// Batched lines (x1, y1, x2, y2, width, `color_rgba`)
    pub lines: Vec<[f32; 9]>,
}

impl DrawBatch {
    /// Create a new batch.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a circle to the batch.
    pub fn add_circle(&mut self, x: f32, y: f32, radius: f32, r: f32, g: f32, b: f32, a: f32) {
        self.circles.push([x, y, radius, r, g, b, a]);
    }

    /// Add a rectangle to the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn add_rect(&mut self, x: f32, y: f32, w: f32, h: f32, r: f32, g: f32, b: f32, a: f32) {
        self.rects.push([x, y, w, h, r, g, b, a]);
    }

    /// Add a line to the batch.
    #[allow(clippy::too_many_arguments)]
    pub fn add_line(
        &mut self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        width: f32,
        r: f32,
        g: f32,
        b: f32,
        a: f32,
    ) {
        self.lines.push([x1, y1, x2, y2, width, r, g, b, a]);
    }

    /// Clear all batches.
    pub fn clear(&mut self) {
        self.circles.clear();
        self.rects.clear();
        self.lines.clear();
    }

    /// Total draw calls if not batched.
    #[must_use]
    pub fn unbatched_draw_calls(&self) -> usize {
        self.circles.len() + self.rects.len() + self.lines.len()
    }

    /// Actual draw calls with batching (3 max: circles, rects, lines).
    #[must_use]
    pub fn batched_draw_calls(&self) -> usize {
        let mut calls = 0;
        if !self.circles.is_empty() {
            calls += 1;
        }
        if !self.rects.is_empty() {
            calls += 1;
        }
        if !self.lines.is_empty() {
            calls += 1;
        }
        calls
    }
}
