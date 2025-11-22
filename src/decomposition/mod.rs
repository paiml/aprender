//! Dimensionality reduction and matrix decomposition algorithms.
//!
//! This module provides techniques for reducing the dimensionality of data
//! and decomposing matrices into meaningful components.
//!
//! # Available Algorithms
//!
//! - **ICA** (Independent Component Analysis): Separates multivariate signals into
//!   independent, non-Gaussian components using the FastICA algorithm
//! - **PCA** (Principal Component Analysis): Available in the main crate

pub mod ica;

pub use ica::ICA;
