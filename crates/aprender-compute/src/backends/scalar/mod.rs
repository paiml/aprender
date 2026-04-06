//! Scalar (non-SIMD) backend implementation
//!
//! This is the portable baseline implementation that works on all platforms.
//! It uses simple loops without any SIMD instructions.
//!
//! # Performance
//!
//! This backend provides correctness reference but no SIMD acceleration.
//! Expected to be 8-32x slower than SIMD backends on operations with 1K+ elements.

use super::VectorBackend;

/// Scalar backend (portable, no SIMD)
pub struct ScalarBackend;

impl VectorBackend for ScalarBackend {
    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn add(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn sub(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] - b[i];
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn mul(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn div(a: &[f32], b: &[f32], result: &mut [f32]) {
        for i in 0..a.len() {
            result[i] = a[i] / b[i];
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/indexing
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    //
    // OPTIMIZATION: 4× unrolling with mul_add for better ILP and auto-vectorization.
    // This follows the cuda-tile pattern for improved throughput (spec: cuda-tile-behavior.md).
    // Using f32::mul_add provides FMA semantics where available, improving accuracy.
    #[inline(always)]
    // SAFETY: caller ensures preconditions are met for this unsafe function
    unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        contract_pre_dot_product!();
        let len = a.len();
        let chunks = len / 4;

        // 4 independent accumulators for better ILP (cuda-tile inspired optimization)
        let mut acc0 = 0.0f32;
        let mut acc1 = 0.0f32;
        let mut acc2 = 0.0f32;
        let mut acc3 = 0.0f32;

        // Process 4 elements at a time with independent accumulation chains
        for i in 0..chunks {
            let base = i * 4;
            acc0 = a[base].mul_add(b[base], acc0);
            acc1 = a[base + 1].mul_add(b[base + 1], acc1);
            acc2 = a[base + 2].mul_add(b[base + 2], acc2);
            acc3 = a[base + 3].mul_add(b[base + 3], acc3);
        }

        // Combine all 4 accumulators
        let mut sum = (acc0 + acc1) + (acc2 + acc3);

        // Handle remainder
        for i in (chunks * 4)..len {
            sum = a[i].mul_add(b[i], sum);
        }

        contract_post_dot_product_parity!(sum);
        sum
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn sum(a: &[f32]) -> f32 {
        let mut total = 0.0;
        for &val in a {
            total += val;
        }
        total
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust slicing/iteration
    // 2. Caller must ensure slice is non-empty (a[0] access)
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn max(a: &[f32]) -> f32 {
        let mut maximum = a[0];
        for &val in a.get(1..).unwrap_or(&[]) {
            if val > maximum {
                maximum = val;
            }
        }
        maximum
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust slicing/iteration
    // 2. Caller must ensure slice is non-empty (a[0] access)
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn min(a: &[f32]) -> f32 {
        let mut minimum = a[0];
        for &val in a.get(1..).unwrap_or(&[]) {
            if val < minimum {
                minimum = val;
            }
        }
        minimum
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Caller must ensure slice is non-empty (a[0] access)
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn argmax(a: &[f32]) -> usize {
        let mut max_value = a[0];
        let mut max_index = 0;
        for (i, &val) in a.iter().enumerate() {
            if val > max_value {
                max_value = val;
                max_index = i;
            }
        }
        max_index
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Caller must ensure slice is non-empty (a[0] access)
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn argmin(a: &[f32]) -> usize {
        let mut min_value = a[0];
        let mut min_index = 0;
        for (i, &val) in a.iter().enumerate() {
            if val < min_value {
                min_value = val;
                min_index = i;
            }
        }
        min_index
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Kahan summation uses only safe floating-point arithmetic
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn sum_kahan(a: &[f32]) -> f32 {
        let mut sum = 0.0;
        let mut c = 0.0; // Compensation for lost low-order bits

        for &value in a {
            let y = value - c; // Subtract the compensation
            let t = sum + y; // Add to sum
            c = (t - sum) - y; // Update compensation
            sum = t; // Update sum
        }

        sum
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Empty check prevents undefined behavior
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn norm_l2(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let mut sum_of_squares = 0.0;
        for &val in a {
            sum_of_squares += val * val;
        }
        sum_of_squares.sqrt()
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Empty check prevents undefined behavior
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn norm_l1(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let mut sum = 0.0;
        for &val in a {
            sum += val.abs();
        }
        sum
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator
    // 2. Empty check prevents undefined behavior
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn norm_linf(a: &[f32]) -> f32 {
        if a.is_empty() {
            return 0.0;
        }

        let mut max_val = 0.0_f32;
        for &val in a {
            let abs_val = val.abs();
            if abs_val > max_val {
                max_val = abs_val;
            }
        }
        max_val
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn scale(a: &[f32], scalar: f32, result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val * scalar;
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn abs(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.abs();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn clamp(a: &[f32], min_val: f32, max_val: f32, result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.max(min_val).min(max_val);
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate/zip
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn lerp(a: &[f32], b: &[f32], t: f32, result: &mut [f32]) {
        for (i, (&a_val, &b_val)) in a.iter().zip(b.iter()).enumerate() {
            // result = a + t * (b - a)
            result[i] = a_val + t * (b_val - a_val);
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate/zip
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) {
        for (i, ((&a_val, &b_val), &c_val)) in a.iter().zip(b.iter()).zip(c.iter()).enumerate() {
            // result = a * b + c
            result[i] = a_val * b_val + c_val;
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn relu(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = if val > 0.0 { val } else { 0.0 };
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn exp(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.exp();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. Clamping prevents exp() overflow
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn sigmoid(a: &[f32], result: &mut [f32]) {
        contract_pre_sigmoid!(a);
        for (i, &val) in a.iter().enumerate() {
            // Handle extreme values for numerical stability
            result[i] = if val < -50.0 {
                0.0 // exp(-x) would overflow, but sigmoid approaches 0
            } else if val > 50.0 {
                1.0 // exp(-x) underflows to 0, sigmoid approaches 1
            } else {
                1.0 / (1.0 + (-val).exp())
            };
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn gelu(a: &[f32], result: &mut [f32]) {
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        contract_pre_gelu!(a);
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        const COEFF: f32 = 0.044715;

        for (i, &x) in a.iter().enumerate() {
            let x3 = x * x * x;
            let inner = SQRT_2_OVER_PI * (x + COEFF * x3);
            result[i] = 0.5 * x * (1.0 + inner.tanh());
        }
        contract_post_gelu!(result);
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. Clamping prevents exp() overflow
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn swish(a: &[f32], result: &mut [f32]) {
        contract_pre_silu!();
        // Swish: x * sigmoid(x) = x / (1 + exp(-x))
        for (i, &x) in a.iter().enumerate() {
            if x < -50.0 {
                result[i] = 0.0; // x * 0 = 0
            } else if x > 50.0 {
                result[i] = x; // x * 1 = x
            } else {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                result[i] = x * sigmoid;
            }
        }
        contract_post_silu!(result);
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn tanh(a: &[f32], result: &mut [f32]) {
        // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
        for (i, &x) in a.iter().enumerate() {
            result[i] = x.tanh();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn sqrt(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.sqrt();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn recip(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.recip();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn ln(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.ln();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn log2(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.log2();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn log10(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.log10();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn sin(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.sin();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn cos(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.cos();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn tan(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.tan();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn floor(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.floor();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn ceil(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.ceil();
        }
    }

    // SAFETY: This function is safe because:
    // 1. All slice accesses are bounds-checked by Rust iterator/enumerate
    // 2. No raw pointer arithmetic is performed
    // 3. Marked unsafe only to match VectorBackend trait interface
    unsafe fn round(a: &[f32], result: &mut [f32]) {
        for (i, &val) in a.iter().enumerate() {
            result[i] = val.round();
        }
    }
}

#[cfg(test)]
mod tests;
