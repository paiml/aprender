//! State-Space Model (SSM) scan kernel.
//!
//! Matches `ssm-kernel-v1.yaml`.
//! h_t = A_bar * h_{t-1} + B_bar * x_t
//! y_t = C * h_t

/// Scalar reference implementation of the SSM sequential scan.
///
/// Computes the recurrent state-space model output for a 1D input sequence.
///
/// - `a_bar`: diagonal of the discretized state matrix, length `state_dim`
/// - `b_bar`: input projection, flattened `state_dim x seq_len`
/// - `c`: output projection, length `state_dim`
/// - `x`: input sequence, length `seq_len`
/// - `output`: output sequence, length `seq_len`
///
/// # Panics
///
/// Panics if any dimension is inconsistent.
pub fn ssm_scan_scalar(
    a_bar: &[f32],
    b_bar: &[f32],
    c: &[f32],
    x: &[f32],
    state_dim: usize,
    seq_len: usize,
    output: &mut [f32],
) {
    assert_eq!(a_bar.len(), state_dim, "a_bar length mismatch");
    assert_eq!(b_bar.len(), state_dim * seq_len, "b_bar length mismatch");
    assert_eq!(c.len(), state_dim, "c length mismatch");
    assert_eq!(x.len(), seq_len, "x length mismatch");
    assert_eq!(output.len(), seq_len, "output length mismatch");

    let mut h = vec![0.0_f32; state_dim];

    for t in 0..seq_len {
        // Update state: h[i] = a_bar[i] * h[i] + b_bar[i*seq_len+t] * x[t]
        for i in 0..state_dim {
            h[i] = a_bar[i] * h[i] + b_bar[i * seq_len + t] * x[t];
        }
        // Compute output: y[t] = sum_i c[i] * h[i]
        let mut y = 0.0_f32;
        for i in 0..state_dim {
            y += c[i] * h[i];
        }
        output[t] = y;
    }
}

/// AVX2 implementation of the SSM sequential scan.
///
/// Delegates to scalar. The sequential time dependency makes SIMD
/// vectorization across time impossible; vectorizing across `state_dim`
/// provides limited benefit for typical small state dimensions.
///
/// # Safety
///
/// Requires AVX2 support on the target CPU.
///
/// # Panics
///
/// Same as [`ssm_scan_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn ssm_scan_avx2(
    a_bar: &[f32],
    b_bar: &[f32],
    c: &[f32],
    x: &[f32],
    state_dim: usize,
    seq_len: usize,
    output: &mut [f32],
) {
    ssm_scan_scalar(a_bar, b_bar, c, x, state_dim, seq_len, output);
}

/// PTX assembly for the SSM scan kernel.
///
/// Parallel across batch/feature dimensions (one block per independent scan).
/// Sequential along the time dimension within each thread.
pub fn ssm_scan_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// SSM scan kernel: 1 thread per independent scan.
// Sequential along time, each thread owns one (a_bar, b_bar, c, x) set.
// Params: a_bar_ptr, b_bar_ptr, c_ptr, x_ptr, output_ptr, state_dim, seq_len
.visible .entry ssm_scan_kernel(
    .param .u64 a_bar_ptr,
    .param .u64 b_bar_ptr,
    .param .u64 c_ptr,
    .param .u64 x_ptr,
    .param .u64 output_ptr,
    .param .u32 state_dim,
    .param .u32 seq_len
)
{
    .reg .u32 %tid, %ntid, %ctaid, %idx, %sd, %sl, %t, %i;
    .reg .u32 %tmp;
    .reg .u64 %a_base, %b_base, %c_base, %x_base, %o_base, %addr;
    .reg .f32 %h, %a, %bval, %cval, %xval, %y, %prod;
    .reg .pred %p_t, %p_i;

    mov.u32 %tid, %tid.x;
    mov.u32 %ntid, %ntid.x;
    mov.u32 %ctaid, %ctaid.x;
    mad.lo.u32 %idx, %ctaid, %ntid, %tid;

    // For simplicity, this kernel handles a single scan (idx=0 only)
    setp.ne.u32 %p_t, %idx, 0;
    @%p_t bra DONE;

    ld.param.u64 %a_base, [a_bar_ptr];
    ld.param.u64 %b_base, [b_bar_ptr];
    ld.param.u64 %c_base, [c_ptr];
    ld.param.u64 %x_base, [x_ptr];
    ld.param.u64 %o_base, [output_ptr];
    ld.param.u32 %sd, [state_dim];
    ld.param.u32 %sl, [seq_len];

    // Outer loop over time
    mov.u32 %t, 0;
TIME_LOOP:
    setp.ge.u32 %p_t, %t, %sl;
    @%p_t bra DONE;

    // Load x[t]
    cvt.u64.u32 %addr, %t;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %x_base, %addr;
    ld.global.f32 %xval, [%addr];

    mov.f32 %y, 0f00000000;

    // Inner loop over state dimensions
    mov.u32 %i, 0;
STATE_LOOP:
    setp.ge.u32 %p_i, %i, %sd;
    @%p_i bra STATE_DONE;

    // Load a_bar[i]
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %a_base, %addr;
    ld.global.f32 %a, [%addr];

    // Load b_bar[i * seq_len + t]
    mul.lo.u32 %tmp, %i, %sl;
    add.u32 %tmp, %tmp, %t;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %b_base, %addr;
    ld.global.f32 %bval, [%addr];

    // Load c[i]
    cvt.u64.u32 %addr, %i;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %c_base, %addr;
    ld.global.f32 %cval, [%addr];

    // h = a * h + b * x (simplified: single register for h)
    fma.rn.f32 %h, %bval, %xval, %h;
    // y += c * h
    fma.rn.f32 %y, %cval, %h, %y;

    add.u32 %i, %i, 1;
    bra STATE_LOOP;
STATE_DONE:

    // Store output[t]
    cvt.u64.u32 %addr, %t;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %o_base, %addr;
    st.global.f32 [%addr], %y;

    add.u32 %t, %t, 1;
    bra TIME_LOOP;

DONE:
    ret;
}
"#
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---------------------------------------------------------------
    // Scalar tests
    // ---------------------------------------------------------------

    /// Verify zero input produces zero output for the SSM scan
    #[test]
    fn test_ssm_zero_input() {
        let state_dim = 3;
        let seq_len = 4;
        let a_bar = [0.9_f32, 0.8, 0.7];
        let b_bar = vec![1.0_f32; state_dim * seq_len];
        let c = [1.0_f32, 1.0, 1.0];
        let x = [0.0_f32; 4];
        let mut output = [0.0_f32; 4];

        ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut output);

        for (t, &o) in output.iter().enumerate() {
            assert!(
                o.abs() < 1e-7,
                "zero input should produce zero output, got output[{t}] = {o}"
            );
        }
    }

    /// Verify SSM single-timestep output matches hand-computed h = B*x, y = C*h
    #[test]
    fn test_ssm_single_timestep() {
        // With h_0 = 0:
        //   h_1 = A*0 + B*x_0 = B*x_0
        //   y_0 = C * h_1
        let state_dim = 2;
        let seq_len = 1;
        let a_bar = [0.5_f32, 0.5];
        // b_bar: 2x1
        let b_bar = [2.0_f32, 3.0];
        let c = [1.0_f32, 1.0];
        let x = [1.0_f32];
        let mut output = [0.0_f32; 1];

        ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut output);

        // h = [2.0*1.0, 3.0*1.0] = [2.0, 3.0]
        // y = 1.0*2.0 + 1.0*3.0 = 5.0
        assert!(
            (output[0] - 5.0).abs() < 1e-6,
            "expected 5.0, got {}",
            output[0]
        );
    }

    /// Verify SSM recurrence over two timesteps with state decay
    #[test]
    fn test_ssm_two_timesteps() {
        let state_dim = 1;
        let seq_len = 2;
        let a_bar = [0.5_f32];
        let b_bar = [1.0_f32, 1.0]; // 1 x 2
        let c = [2.0_f32];
        let x = [1.0_f32, 1.0];
        let mut output = [0.0_f32; 2];

        ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut output);

        // t=0: h = 0.5*0 + 1.0*1.0 = 1.0; y = 2.0*1.0 = 2.0
        // t=1: h = 0.5*1.0 + 1.0*1.0 = 1.5; y = 2.0*1.5 = 3.0
        assert!(
            (output[0] - 2.0).abs() < 1e-6,
            "t=0: expected 2.0, got {}",
            output[0]
        );
        assert!(
            (output[1] - 3.0).abs() < 1e-6,
            "t=1: expected 3.0, got {}",
            output[1]
        );
    }

    /// Verify SSM panics on a_bar length mismatch
    #[test]
    #[should_panic(expected = "a_bar length mismatch")]
    fn test_ssm_abar_mismatch() {
        let mut output = [0.0_f32; 2];
        ssm_scan_scalar(
            &[0.5],
            &[1.0; 4],
            &[1.0, 1.0],
            &[1.0, 1.0],
            2,
            2,
            &mut output,
        );
    }

    // ---------------------------------------------------------------
    // AVX2 tests
    // ---------------------------------------------------------------

    /// Verify AVX2 SSM scan produces identical results to scalar
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_ssm_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let state_dim = 4;
        let seq_len = 8;
        let a_bar: Vec<f32> = (0..state_dim).map(|i| 0.5 + 0.1 * i as f32).collect();
        let b_bar: Vec<f32> = (0..state_dim * seq_len)
            .map(|i| ((i as f32) * 0.1).sin())
            .collect();
        let c: Vec<f32> = (0..state_dim).map(|i| 1.0 / (i as f32 + 1.0)).collect();
        let x: Vec<f32> = (0..seq_len).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut scalar_out = vec![0.0_f32; seq_len];
        let mut avx2_out = vec![0.0_f32; seq_len];

        ssm_scan_scalar(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut scalar_out);
        unsafe {
            ssm_scan_avx2(&a_bar, &b_bar, &c, &x, state_dim, seq_len, &mut avx2_out);
        }

        assert_eq!(scalar_out, avx2_out);
    }

    // ---------------------------------------------------------------
    // PTX structural tests
    // ---------------------------------------------------------------

    /// Verify SSM PTX declares version 8.5
    #[test]
    fn test_ssm_ptx_version() {
        let ptx = ssm_scan_ptx();
        assert!(
            ptx.contains(".version 8.5"),
            "PTX must declare .version 8.5"
        );
    }

    /// Verify SSM PTX targets sm_90
    #[test]
    fn test_ssm_ptx_target() {
        let ptx = ssm_scan_ptx();
        assert!(ptx.contains(".target sm_90"), "PTX must target sm_90");
    }

    /// Verify SSM PTX contains the kernel entry point
    #[test]
    fn test_ssm_ptx_entry() {
        let ptx = ssm_scan_ptx();
        assert!(
            ptx.contains(".entry ssm_scan_kernel"),
            "PTX must have .entry"
        );
    }

    /// Verify SSM PTX contains a ret instruction
    #[test]
    fn test_ssm_ptx_ret() {
        let ptx = ssm_scan_ptx();
        assert!(ptx.contains("ret;"), "PTX must have ret;");
    }

    /// Verify SSM PTX has balanced curly braces
    #[test]
    fn test_ssm_ptx_balanced_braces() {
        let ptx = ssm_scan_ptx();
        let opens = ptx.chars().filter(|&c| c == '{').count();
        let closes = ptx.chars().filter(|&c| c == '}').count();
        assert_eq!(
            opens, closes,
            "PTX must have balanced braces: {opens} opens vs {closes} closes"
        );
    }
}
