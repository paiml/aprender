//! Minimal complex number type for FFT operations.

use std::ops::{Add, Mul, Sub};

/// Complex number (f32 real + f32 imaginary).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Complex {
    /// Real part.
    pub re: f32,
    /// Imaginary part.
    pub im: f32,
}

impl Complex {
    /// Create a new complex number.
    #[inline]
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Complex zero.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// Squared magnitude: |z|^2 = re^2 + im^2.
    #[inline]
    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Magnitude: |z|.
    #[inline]
    pub fn abs(self) -> f32 {
        self.norm_sq().sqrt()
    }

    /// Complex conjugate.
    #[inline]
    pub fn conj(self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    /// Create from polar form: r * e^(iθ).
    #[inline]
    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self { re: r * theta.cos(), im: r * theta.sin() }
    }

    /// Scale by a real number.
    #[inline]
    pub fn scale(self, s: f32) -> Self {
        Self { re: self.re * s, im: self.im * s }
    }
}

impl Add for Complex {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl Sub for Complex {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl Mul for Complex {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self { re: self.re * rhs.re - self.im * rhs.im, im: self.re * rhs.im + self.im * rhs.re }
    }
}
