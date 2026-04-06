//! AVX2 backend tests
//!
//! Tests verifying AVX2 SIMD operations produce correct results and match scalar baseline.

#[cfg(test)]
mod basic_ops;
#[cfg(test)]
mod compound_ops;
#[cfg(test)]
mod math_ops;
