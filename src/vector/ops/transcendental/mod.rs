//! Transcendental mathematical functions for Vector<f32>
//!
//! This module provides element-wise transcendental functions including:
//! - Exponentials: `exp`, `ln`, `log2`, `log10`
//! - Trigonometric: `sin`, `cos`, `tan`, `asin`, `acos`, `atan`
//! - Hyperbolic: `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

mod exp_log;
mod hyperbolic;
mod trigonometric;

#[cfg(test)]
mod tests;
