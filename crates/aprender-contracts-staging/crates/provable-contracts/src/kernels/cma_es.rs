//! CMA-ES sampling kernel.
//!
//! Matches `cma-es-kernel-v1.yaml`.
//! x = mean + sigma * L * z where L is the Cholesky factor of the covariance matrix.

/// Scalar reference implementation of CMA-ES sample generation.
///
/// Transforms a standard normal sample `z` into a candidate solution `x`
/// using the current distribution parameters.
///
/// - `mean`: distribution mean, length `d`
/// - `sigma`: overall step size (scalar)
/// - `cholesky_l`: lower triangular Cholesky factor, flattened `d x d` (row-major)
/// - `d`: dimension of the search space
/// - `z`: standard normal sample, length `d`
/// - `output`: candidate solution `x = mean + sigma * L * z`, length `d`
///
/// # Panics
///
/// Panics if dimensions are inconsistent.
pub fn cma_sample_scalar(
    mean: &[f32],
    sigma: f32,
    cholesky_l: &[f32],
    d: usize,
    z: &[f32],
    output: &mut [f32],
) {
    assert_eq!(mean.len(), d, "mean length mismatch");
    assert_eq!(cholesky_l.len(), d * d, "cholesky_l length mismatch");
    assert_eq!(z.len(), d, "z length mismatch");
    assert_eq!(output.len(), d, "output length mismatch");

    // Compute L*z (lower triangular matrix-vector multiply)
    for i in 0..d {
        let mut sum = 0.0_f32;
        for j in 0..=i {
            sum += cholesky_l[i * d + j] * z[j];
        }
        output[i] = mean[i] + sigma * sum;
    }
}

/// AVX2 implementation of CMA-ES sample generation.
///
/// Delegates to scalar.
///
/// # Safety
///
/// Requires AVX2 support on the target CPU.
///
/// # Panics
///
/// Same as [`cma_sample_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn cma_sample_avx2(
    mean: &[f32],
    sigma: f32,
    cholesky_l: &[f32],
    d: usize,
    z: &[f32],
    output: &mut [f32],
) {
    cma_sample_scalar(mean, sigma, cholesky_l, d, z, output);
}

/// PTX assembly for the CMA-ES sampling kernel.
///
/// 1 thread per dimension. Each thread computes one row of L*z (dot product
/// of lower triangular row with z vector).
pub fn cma_sample_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// CMA-ES sample kernel: 1 thread per dimension.
// x[i] = mean[i] + sigma * sum_{j<=i} L[i*d+j] * z[j]
// Params: mean_ptr, sigma, cholesky_l_ptr, z_ptr, output_ptr, d
.visible .entry cma_sample_kernel(
    .param .u64 mean_ptr,
    .param .f32 sigma,
    .param .u64 cholesky_l_ptr,
    .param .u64 z_ptr,
    .param .u64 output_ptr,
    .param .u32 d
)
{
    .reg .u32 %tid, %ntid, %ctaid, %idx, %d, %j;
    .reg .u32 %row_off, %tmp;
    .reg .u64 %m_base, %l_base, %z_base, %o_base, %addr;
    .reg .f32 %sum, %lval, %zval, %mean_val, %sigma, %scaled, %result;
    .reg .pred %p, %p_j;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;
    ld.param.u32 %d, [d];
    setp.ge.u32 %p, %idx, %d;
    @%p bra DONE;

    ld.param.u64 %m_base, [mean_ptr];
    ld.param.f32 %sigma, [sigma];
    ld.param.u64 %l_base, [cholesky_l_ptr];
    ld.param.u64 %z_base, [z_ptr];
    ld.param.u64 %o_base, [output_ptr];

    // Load mean[idx]
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %m_base, %addr;
    ld.global.f32 %mean_val, [%addr];

    // Compute L[idx, :] . z (only j <= idx)
    mul.lo.u32 %row_off, %idx, %d;
    mov.f32 %sum, 0f00000000;
    mov.u32 %j, 0;
    // Loop limit: j <= idx, i.e., j < idx+1
    .reg .u32 %limit;
    add.u32 %limit, %idx, 1;
L_LOOP:
    setp.ge.u32 %p_j, %j, %limit;
    @%p_j bra L_DONE;

    // Load L[idx*d + j]
    add.u32 %tmp, %row_off, %j;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %l_base, %addr;
    ld.global.f32 %lval, [%addr];

    // Load z[j]
    cvt.u64.u32 %addr, %j;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %z_base, %addr;
    ld.global.f32 %zval, [%addr];

    fma.rn.f32 %sum, %lval, %zval, %sum;

    add.u32 %j, %j, 1;
    bra L_LOOP;
L_DONE:

    // output[idx] = mean[idx] + sigma * sum
    mul.f32 %scaled, %sigma, %sum;
    add.f32 %result, %mean_val, %scaled;

    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %o_base, %addr;
    st.global.f32 [%addr], %result;

DONE:
    ret;
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ulp::assert_ulp_eq;

    // ---------------------------------------------------------------
    // Scalar tests
    // ---------------------------------------------------------------

    #[test]
    fn test_cma_sigma_zero() {
        // sigma=0 -> output = mean regardless of z and L
        let mean = [1.0_f32, 2.0, 3.0];
        let cholesky_l = [1.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.3, 0.2, 1.0];
        let z = [10.0_f32, 20.0, 30.0];
        let mut output = [0.0_f32; 3];

        cma_sample_scalar(&mean, 0.0, &cholesky_l, 3, &z, &mut output);

        assert_eq!(output, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_cma_identity_cholesky() {
        // L=I -> output = mean + sigma*z
        let mean = [1.0_f32, 2.0, 3.0];
        let cholesky_l = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let z = [0.5_f32, -0.3, 0.8];
        let sigma = 2.0;
        let mut output = [0.0_f32; 3];

        cma_sample_scalar(&mean, sigma, &cholesky_l, 3, &z, &mut output);

        // output = [1 + 2*0.5, 2 + 2*(-0.3), 3 + 2*0.8] = [2, 1.4, 4.6]
        assert!((output[0] - 2.0).abs() < 1e-6, "output[0] = {}", output[0]);
        assert!((output[1] - 1.4).abs() < 1e-6, "output[1] = {}", output[1]);
        assert!((output[2] - 4.6).abs() < 1e-6, "output[2] = {}", output[2]);
    }

    #[test]
    fn test_cma_lower_triangular() {
        // Verify lower-triangular multiplication
        let mean = [0.0_f32; 2];
        let cholesky_l = [2.0, 0.0, 3.0, 4.0];
        let z = [1.0_f32, 1.0];
        let sigma = 1.0;
        let mut output = [0.0_f32; 2];

        cma_sample_scalar(&mean, sigma, &cholesky_l, 2, &z, &mut output);

        // L*z: row 0 = 2*1 = 2, row 1 = 3*1 + 4*1 = 7
        assert!((output[0] - 2.0).abs() < 1e-6, "output[0] = {}", output[0]);
        assert!((output[1] - 7.0).abs() < 1e-6, "output[1] = {}", output[1]);
    }

    #[test]
    fn test_cma_single_dimension() {
        let mean = [5.0_f32];
        let cholesky_l = [3.0_f32];
        let z = [2.0_f32];
        let sigma = 0.5;
        let mut output = [0.0_f32; 1];

        cma_sample_scalar(&mean, sigma, &cholesky_l, 1, &z, &mut output);

        // output = 5 + 0.5 * 3 * 2 = 5 + 3 = 8
        assert!((output[0] - 8.0).abs() < 1e-6, "output[0] = {}", output[0]);
    }

    #[test]
    #[should_panic(expected = "mean length mismatch")]
    fn test_cma_mean_mismatch() {
        let mut output = [0.0_f32; 3];
        cma_sample_scalar(&[1.0, 2.0], 1.0, &[0.0; 9], 3, &[0.0; 3], &mut output);
    }

    // ---------------------------------------------------------------
    // AVX2 tests
    // ---------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_cma_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let d = 4;
        let mean = [1.0_f32, 2.0, 3.0, 4.0];
        let cholesky_l = [
            1.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.3, 0.2, 1.0, 0.0, 0.1, 0.4, 0.6, 1.0,
        ];
        let z = [0.1_f32, -0.2, 0.3, -0.4];
        let sigma = 1.5;
        let mut scalar_out = [0.0_f32; 4];
        let mut avx2_out = [0.0_f32; 4];

        cma_sample_scalar(&mean, sigma, &cholesky_l, d, &z, &mut scalar_out);
        unsafe {
            cma_sample_avx2(&mean, sigma, &cholesky_l, d, &z, &mut avx2_out);
        }

        assert_ulp_eq(&scalar_out, &avx2_out, 0);
    }

    // ---------------------------------------------------------------
    // PTX structural tests
    // ---------------------------------------------------------------

    #[test]
    fn test_cma_ptx_version() {
        let ptx = cma_sample_ptx();
        assert!(
            ptx.contains(".version 8.5"),
            "PTX must declare .version 8.5"
        );
    }

    #[test]
    fn test_cma_ptx_target() {
        let ptx = cma_sample_ptx();
        assert!(ptx.contains(".target sm_90"), "PTX must target sm_90");
    }

    #[test]
    fn test_cma_ptx_entry() {
        let ptx = cma_sample_ptx();
        assert!(
            ptx.contains(".entry cma_sample_kernel"),
            "PTX must have .entry"
        );
    }

    #[test]
    fn test_cma_ptx_ret() {
        let ptx = cma_sample_ptx();
        assert!(ptx.contains("ret;"), "PTX must have ret;");
    }

    #[test]
    fn test_cma_ptx_balanced_braces() {
        let ptx = cma_sample_ptx();
        let opens = ptx.chars().filter(|&c| c == '{').count();
        let closes = ptx.chars().filter(|&c| c == '}').count();
        assert_eq!(
            opens, closes,
            "PTX must have balanced braces: {opens} opens vs {closes} closes"
        );
    }
}
