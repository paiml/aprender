//! Bias addition kernel.
//!
//! Matches `bias-add-v1.yaml`.
//! `y[b, i] = x[b, i] + bias[i]` — broadcast bias vector over batch dimension.
//!
//! Each function provides one of three backends:
//! - `fn bias_add_scalar(...)` -- Pure Rust scalar reference (ground truth)
//! - `unsafe fn bias_add_avx2(...)` -- AVX2 SIMD implementation
//! - `fn bias_add_ptx() -> &'static str` -- PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Broadcast bias addition (scalar reference).
///
/// `x` is `batch x dim` (row-major), `bias` is `dim`, output is `batch x dim`.
/// Each batch row gets the same bias vector added.
///
/// # Panics
/// Panics if dimensions don't match.
pub fn bias_add_scalar(x: &[f32], bias: &[f32], batch: usize, dim: usize, output: &mut [f32]) {
    assert_eq!(x.len(), batch * dim, "x dimension mismatch");
    assert_eq!(bias.len(), dim, "bias dimension mismatch");
    assert_eq!(output.len(), batch * dim, "output dimension mismatch");

    for b in 0..batch {
        for d in 0..dim {
            let idx = b * dim + d;
            output[idx] = x[idx] + bias[d];
        }
    }
}

/// In-place broadcast bias addition.
///
/// `x` is `batch x dim`, modified in place. `bias` is `dim`.
pub fn bias_add_inplace(x: &mut [f32], bias: &[f32], batch: usize, dim: usize) {
    assert_eq!(x.len(), batch * dim, "x dimension mismatch");
    assert_eq!(bias.len(), dim, "bias dimension mismatch");

    for b in 0..batch {
        for d in 0..dim {
            x[b * dim + d] += bias[d];
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 bias addition -- delegates to scalar (addition is exact).
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn bias_add_avx2(x: &[f32], bias: &[f32], batch: usize, dim: usize, output: &mut [f32]) {
    bias_add_scalar(x, bias, batch, dim, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for bias addition.
///
/// One thread per output element. bid = batch index, tid = dimension index.
pub fn bias_add_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry bias_add_kernel(
    .param .u64 X,
    .param .u64 BIAS,
    .param .u64 OUT,
    .param .u32 BATCH,
    .param .u32 DIM
) {
    .reg .u32 %tid, %bid, %dim, %batch, %offset;
    .reg .u64 %x_ptr, %bias_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %x_val, %bias_val, %result;
    .reg .pred %p_bound;

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    ld.param.u32 %dim, [DIM];
    ld.param.u32 %batch, [BATCH];
    ld.param.u64 %x_ptr, [X];
    ld.param.u64 %bias_ptr, [BIAS];
    ld.param.u64 %out_ptr, [OUT];

    // bid = batch index, tid = dim index
    setp.ge.u32 %p_bound, %tid, %dim;
    @%p_bound bra EXIT;

    // Load x[bid * dim + tid]
    mad.lo.u32 %offset, %bid, %dim, %tid;
    mul.wide.u32 %off64, %offset, 4;
    add.u64 %addr, %x_ptr, %off64;
    ld.global.f32 %x_val, [%addr];

    // Load bias[tid]
    mul.wide.u32 %off64, %tid, 4;
    add.u64 %addr, %bias_ptr, %off64;
    ld.global.f32 %bias_val, [%addr];

    // output = x + bias
    add.f32 %result, %x_val, %bias_val;

    // Store out[bid * dim + tid]
    mad.lo.u32 %offset, %bid, %dim, %tid;
    mul.wide.u32 %off64, %offset, 4;
    add.u64 %addr, %out_ptr, %off64;
    st.global.f32 [%addr], %result;

EXIT:
    ret;
}
"#
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_bias_add_basic() {
        let x = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let bias = [10.0, 20.0];
        let mut output = [0.0f32; 4];

        bias_add_scalar(&x, &bias, 2, 2, &mut output);
        assert_eq!(&output, &[11.0, 22.0, 13.0, 24.0]);
    }

    #[test]
    fn test_bias_add_zero_is_identity() {
        let x = [1.0, 2.0, 3.0];
        let bias = [0.0, 0.0, 0.0];
        let mut output = [0.0f32; 3];

        bias_add_scalar(&x, &bias, 1, 3, &mut output);
        assert_eq!(&output, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bias_add_additivity() {
        // bias_add(bias_add(x, b1), b2) = bias_add(x, b1 + b2)
        let x = [1.0, 2.0, 3.0, 4.0];
        let b1 = [0.5, 0.3];
        let b2 = [0.1, 0.2];
        let b12: Vec<f32> = b1.iter().zip(b2.iter()).map(|(a, b)| a + b).collect();

        let mut step1 = [0.0f32; 4];
        let mut step2 = [0.0f32; 4];
        let mut direct = [0.0f32; 4];

        bias_add_scalar(&x, &b1, 2, 2, &mut step1);
        bias_add_scalar(&step1, &b2, 2, 2, &mut step2);
        bias_add_scalar(&x, &b12, 2, 2, &mut direct);

        for i in 0..4 {
            assert!(
                (step2[i] - direct[i]).abs() < 1e-6,
                "additivity violated at {i}: {:.6} vs {:.6}",
                step2[i],
                direct[i]
            );
        }
    }

    #[test]
    fn test_bias_add_inplace_matches() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let bias = [10.0, 20.0];
        let mut output = [0.0f32; 4];
        let mut x_copy = x;

        bias_add_scalar(&x, &bias, 2, 2, &mut output);
        bias_add_inplace(&mut x_copy, &bias, 2, 2);
        assert_eq!(&output, &x_copy);
    }

    #[test]
    fn test_bias_add_broadcast() {
        // Same bias applied to each batch element
        let x = [0.0; 6]; // 3x2
        let bias = [5.0, 7.0];
        let mut output = [0.0f32; 6];

        bias_add_scalar(&x, &bias, 3, 2, &mut output);
        assert_eq!(&output, &[5.0, 7.0, 5.0, 7.0, 5.0, 7.0]);
    }

    proptest! {
        #[test]
        fn prop_bias_add_finite(
            batch in 1usize..4,
            dim in 1usize..6,
        ) {
            let x: Vec<f32> = (0..batch * dim).map(|i| (i as f32) * 0.1).collect();
            let bias: Vec<f32> = (0..dim).map(|i| (i as f32) * 0.01).collect();
            let mut output = vec![0.0f32; batch * dim];

            bias_add_scalar(&x, &bias, batch, dim, &mut output);

            for (idx, &val) in output.iter().enumerate() {
                prop_assert!(val.is_finite(), "output[{idx}] = {val} not finite");
            }
        }
    }

    #[test]
    fn test_bias_add_ptx_structure() {
        let ptx = bias_add_ptx();
        assert!(ptx.contains(".entry bias_add_kernel"));
        assert!(ptx.contains("add.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_bias_add_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let bias = [10.0, 20.0];
        let mut scalar_out = [0.0f32; 6];
        let mut avx2_out = [0.0f32; 6];
        bias_add_scalar(&x, &bias, 3, 2, &mut scalar_out);
        unsafe { bias_add_avx2(&x, &bias, 3, 2, &mut avx2_out) };
        assert_eq!(scalar_out, avx2_out);
    }
}
