//! Vector operation modules
//!
//! This module organizes Vector operations into focused submodules:
//! - `activations`: Neural network activation functions (relu, sigmoid, gelu, etc.)
//! - `arithmetic`: Binary arithmetic operations (add, sub, mul, div, fma, scale)
//! - `normalization`: Normalization methods (zscore, minmax, layer_norm, normalize)
//! - `norms`: Vector norm calculations (norm_l1, norm_l2, norm_linf)
//! - `reductions`: Reduction operations (sum, dot, max, min, argmax, mean, variance, etc.)
//! - `rounding`: Rounding and sign functions (floor, ceil, round, trunc, etc.)
//! - `transcendental`: Mathematical functions (exp, log, sin, cos, etc.)
//! - `transforms`: Element-wise transforms (abs, clamp, clip, lerp, sqrt, recip, pow)

pub(super) mod activations;
pub(super) mod arithmetic;
pub(super) mod normalization;
pub(super) mod norms;
pub(super) mod reductions;
pub(super) mod rounding;
pub(super) mod transcendental;
pub(super) mod transforms;
