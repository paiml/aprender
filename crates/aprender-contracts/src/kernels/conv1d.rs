//! 1D Convolution kernel.
//!
//! Matches `conv1d-kernel-v1.yaml`.
//!
//! Standard 1D convolution with configurable stride and padding.
//! Input layout: c_in x length (row-major).
//! Weight layout: c_out x c_in x kernel_size (row-major).
//! Output layout: c_out x out_length (row-major).

/// Scalar reference implementation of 1D convolution.
///
/// Computes the convolution of `input` with `weight` and optional `bias`.
///
/// - `input`: flattened `c_in x length`
/// - `weight`: flattened `c_out x c_in x kernel_size`
/// - `bias`: optional, length `c_out`
/// - `output`: flattened `c_out x out_length` where `out_length = (length + 2*padding - kernel_size) / stride + 1`
///
/// # Panics
///
/// Panics if dimensions are inconsistent or output buffer has wrong length.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_scalar(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    c_in: usize,
    c_out: usize,
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output: &mut [f32],
) {
    assert_eq!(input.len(), c_in * length, "input length mismatch");
    assert_eq!(
        weight.len(),
        c_out * c_in * kernel_size,
        "weight length mismatch"
    );
    if let Some(b) = bias {
        assert_eq!(b.len(), c_out, "bias length mismatch");
    }
    assert!(stride > 0, "stride must be > 0");
    let out_length = (length + 2 * padding - kernel_size) / stride + 1;
    assert_eq!(
        output.len(),
        c_out * out_length,
        "output length mismatch: expected {}",
        c_out * out_length
    );

    for oc in 0..c_out {
        for ol in 0..out_length {
            let sum = conv1d_output_element(
                input,
                weight,
                c_in,
                length,
                kernel_size,
                stride,
                padding,
                oc,
                ol,
            );
            let bias_val = bias.map_or(0.0, |b| b[oc]);
            output[oc * out_length + ol] = sum + bias_val;
        }
    }
}

/// Compute a single output element of the convolution.
#[allow(clippy::too_many_arguments)]
fn conv1d_output_element(
    input: &[f32],
    weight: &[f32],
    c_in: usize,
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    oc: usize,
    ol: usize,
) -> f32 {
    let mut sum = 0.0_f32;
    for ic in 0..c_in {
        for k in 0..kernel_size {
            let in_pos_signed = (ol * stride + k) as isize - padding as isize;
            if in_pos_signed >= 0 && (in_pos_signed as usize) < length {
                let in_pos = in_pos_signed as usize;
                let w_idx = oc * c_in * kernel_size + ic * kernel_size + k;
                let i_idx = ic * length + in_pos;
                sum += weight[w_idx] * input[i_idx];
            }
        }
    }
    sum
}

/// AVX2 implementation of 1D convolution.
///
/// Delegates to scalar due to irregular memory access patterns in convolution.
///
/// # Safety
///
/// Requires AVX2 support on the target CPU.
///
/// # Panics
///
/// Same as [`conv1d_scalar`].
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv1d_avx2(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    c_in: usize,
    c_out: usize,
    length: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output: &mut [f32],
) {
    conv1d_scalar(
        input,
        weight,
        bias,
        c_in,
        c_out,
        length,
        kernel_size,
        stride,
        padding,
        output,
    );
}

/// PTX assembly for the 1D convolution kernel.
///
/// One block per output channel, threads along output length.
/// Each thread computes one output position by summing over input
/// channels and kernel positions.
pub fn conv1d_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64

// Conv1D kernel: 1 block per output channel, threads along output length.
// Params: input_ptr, weight_ptr, bias_ptr, output_ptr,
//         c_in, length, kernel_size, stride, padding, out_length
.visible .entry conv1d_kernel(
    .param .u64 input_ptr,
    .param .u64 weight_ptr,
    .param .u64 bias_ptr,
    .param .u64 output_ptr,
    .param .u32 c_in,
    .param .u32 length,
    .param .u32 kernel_size,
    .param .u32 stride,
    .param .u32 padding,
    .param .u32 out_length
)
{
    .reg .u32 %tid, %oc, %ol, %ic, %k, %c_in, %len, %ks, %str, %pad, %olen;
    .reg .u32 %in_pos, %w_idx, %i_idx, %tmp, %w_base_oc;
    .reg .s32 %in_pos_signed;
    .reg .u64 %in_base, %w_base, %b_base, %out_base, %addr;
    .reg .f32 %sum, %wval, %ival, %bval;
    .reg .pred %p, %p_lo, %p_hi;

    mov.u32 %oc, %ctaid.x;
    mov.u32 %tid, %tid.x;
    ld.param.u32 %olen, [out_length];
    setp.ge.u32 %p, %tid, %olen;
    @%p bra DONE;

    mov.u32 %ol, %tid;
    ld.param.u64 %in_base, [input_ptr];
    ld.param.u64 %w_base, [weight_ptr];
    ld.param.u64 %b_base, [bias_ptr];
    ld.param.u64 %out_base, [output_ptr];
    ld.param.u32 %c_in, [c_in];
    ld.param.u32 %len, [length];
    ld.param.u32 %ks, [kernel_size];
    ld.param.u32 %str, [stride];
    ld.param.u32 %pad, [padding];

    mov.f32 %sum, 0f00000000;

    // weight base for this output channel: oc * c_in * kernel_size
    mul.lo.u32 %w_base_oc, %oc, %c_in;
    mul.lo.u32 %w_base_oc, %w_base_oc, %ks;

    mov.u32 %ic, 0;
IC_LOOP:
    setp.ge.u32 %p, %ic, %c_in;
    @%p bra IC_DONE;

    mov.u32 %k, 0;
K_LOOP:
    setp.ge.u32 %p, %k, %ks;
    @%p bra K_DONE;

    // in_pos_signed = ol * stride + k - padding
    mul.lo.u32 %in_pos, %ol, %str;
    add.u32 %in_pos, %in_pos, %k;
    sub.s32 %in_pos_signed, %in_pos, %pad;
    setp.lt.s32 %p_lo, %in_pos_signed, 0;
    @%p_lo bra SKIP;
    mov.u32 %in_pos, %in_pos_signed;
    setp.ge.u32 %p_hi, %in_pos, %len;
    @%p_hi bra SKIP;

    // w_idx = w_base_oc + ic * ks + k
    mul.lo.u32 %w_idx, %ic, %ks;
    add.u32 %w_idx, %w_idx, %w_base_oc;
    add.u32 %w_idx, %w_idx, %k;
    // Load weight
    cvt.u64.u32 %addr, %w_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %w_base, %addr;
    ld.global.f32 %wval, [%addr];

    // i_idx = ic * length + in_pos
    mul.lo.u32 %i_idx, %ic, %len;
    add.u32 %i_idx, %i_idx, %in_pos;
    cvt.u64.u32 %addr, %i_idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %in_base, %addr;
    ld.global.f32 %ival, [%addr];

    fma.rn.f32 %sum, %wval, %ival, %sum;

SKIP:
    add.u32 %k, %k, 1;
    bra K_LOOP;
K_DONE:
    add.u32 %ic, %ic, 1;
    bra IC_LOOP;
IC_DONE:

    // Add bias if present (bias_ptr != 0)
    setp.eq.u64 %p, %b_base, 0;
    @%p bra STORE;
    cvt.u64.u32 %addr, %oc;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %b_base, %addr;
    ld.global.f32 %bval, [%addr];
    add.f32 %sum, %sum, %bval;

STORE:
    // output[oc * out_length + ol]
    mul.lo.u32 %tmp, %oc, %olen;
    add.u32 %tmp, %tmp, %ol;
    cvt.u64.u32 %addr, %tmp;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %out_base, %addr;
    st.global.f32 [%addr], %sum;

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

    #[test]
    fn test_conv1d_identity() {
        // kernel_size=1, weight=identity matrix, c_in=c_out=2, length=4
        let c_in = 2;
        let c_out = 2;
        let length = 4;
        let kernel_size = 1;
        let stride = 1;
        let padding = 0;
        let out_length = (length + 2 * padding - kernel_size) / stride + 1;

        // Input: [[1,2,3,4],[5,6,7,8]]
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        // Weight: identity (2x2x1) = [[1,0],[0,1]]
        let weight = [1.0_f32, 0.0, 0.0, 1.0];
        let mut output = vec![0.0_f32; c_out * out_length];

        conv1d_scalar(
            &input,
            &weight,
            None,
            c_in,
            c_out,
            length,
            kernel_size,
            stride,
            padding,
            &mut output,
        );

        // Output should equal input
        assert_eq!(output, input.to_vec());
    }

    #[test]
    fn test_conv1d_known_values() {
        // Single input channel, single output channel
        // input = [1, 2, 3, 4, 5], kernel = [1, 0, -1], stride=1, padding=0
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0_f32, 0.0, -1.0];
        let out_length = (5 - 3) + 1; // 3
        let mut output = vec![0.0_f32; out_length];

        conv1d_scalar(&input, &weight, None, 1, 1, 5, 3, 1, 0, &mut output);

        // output[0] = 1*1 + 2*0 + 3*(-1) = -2
        // output[1] = 2*1 + 3*0 + 4*(-1) = -2
        // output[2] = 3*1 + 4*0 + 5*(-1) = -2
        assert_eq!(output, vec![-2.0, -2.0, -2.0]);
    }

    #[test]
    fn test_conv1d_with_bias() {
        let input = [1.0_f32, 2.0, 3.0];
        let weight = [1.0_f32, 1.0];
        let bias = [10.0_f32];
        let out_length = (3 - 2) + 1; // 2
        let mut output = vec![0.0_f32; out_length];

        conv1d_scalar(&input, &weight, Some(&bias), 1, 1, 3, 2, 1, 0, &mut output);

        // output[0] = 1+2 + 10 = 13
        // output[1] = 2+3 + 10 = 15
        assert_eq!(output, vec![13.0, 15.0]);
    }

    #[test]
    fn test_conv1d_with_padding() {
        // input = [1, 2, 3], kernel = [1, 1, 1], padding=1, stride=1
        let input = [1.0_f32, 2.0, 3.0];
        let weight = [1.0_f32, 1.0, 1.0];
        let out_length = (3 + 2 - 3) + 1; // 3
        let mut output = vec![0.0_f32; out_length];

        conv1d_scalar(&input, &weight, None, 1, 1, 3, 3, 1, 1, &mut output);

        // output[0]: pos 0..3, with padding: [0, 1, 2] -> 0+1+2 = 3
        // output[1]: pos 1..4, -> [1, 2, 3] -> 6
        // output[2]: pos 2..5, with padding: [2, 3, 0] -> 5
        assert_eq!(output, vec![3.0, 6.0, 5.0]);
    }

    #[test]
    fn test_conv1d_with_stride() {
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0_f32];
        let out_length = (5 - 1) / 2 + 1; // 3
        let mut output = vec![0.0_f32; out_length];

        conv1d_scalar(&input, &weight, None, 1, 1, 5, 1, 2, 0, &mut output);

        assert_eq!(output, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    #[should_panic(expected = "input length mismatch")]
    fn test_conv1d_input_mismatch() {
        let input = [1.0_f32; 5];
        let weight = [1.0_f32; 3];
        let mut output = [0.0_f32; 3];
        conv1d_scalar(&input, &weight, None, 2, 1, 5, 3, 1, 0, &mut output);
    }

    // ---------------------------------------------------------------
    // AVX2 tests
    // ---------------------------------------------------------------

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_conv1d_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let weight = [1.0_f32, 0.0, -1.0];
        let out_length = 3;
        let mut scalar_out = vec![0.0_f32; out_length];
        let mut avx2_out = vec![0.0_f32; out_length];

        conv1d_scalar(&input, &weight, None, 1, 1, 5, 3, 1, 0, &mut scalar_out);
        unsafe {
            conv1d_avx2(&input, &weight, None, 1, 1, 5, 3, 1, 0, &mut avx2_out);
        }
        assert_eq!(scalar_out, avx2_out);
    }

    // ---------------------------------------------------------------
    // PTX structural tests
    // ---------------------------------------------------------------

    #[test]
    fn test_conv1d_ptx_version() {
        let ptx = conv1d_ptx();
        assert!(
            ptx.contains(".version 8.5"),
            "PTX must declare .version 8.5"
        );
    }

    #[test]
    fn test_conv1d_ptx_target() {
        let ptx = conv1d_ptx();
        assert!(ptx.contains(".target sm_90"), "PTX must target sm_90");
    }

    #[test]
    fn test_conv1d_ptx_entry() {
        let ptx = conv1d_ptx();
        assert!(ptx.contains(".entry conv1d_kernel"), "PTX must have .entry");
    }

    #[test]
    fn test_conv1d_ptx_ret() {
        let ptx = conv1d_ptx();
        assert!(ptx.contains("ret;"), "PTX must have ret;");
    }

    #[test]
    fn test_conv1d_ptx_balanced_braces() {
        let ptx = conv1d_ptx();
        let opens = ptx.chars().filter(|&c| c == '{').count();
        let closes = ptx.chars().filter(|&c| c == '}').count();
        assert_eq!(
            opens, closes,
            "PTX must have balanced braces: {opens} opens vs {closes} closes"
        );
    }
}
