//! Dropout kernel.
//!
//! Matches `dropout-v1.yaml`.
//! Train: `y = mask * x / (1 - p)` where mask ~ Bernoulli(1 - p).
//! Eval: `y = x` (identity).
//!
//! Note: The mask is pre-computed and passed in (deterministic), rather than
//! using internal RNG. This makes the kernel verifiable and reproducible.
//!
//! Each function provides one of three backends:
//! - `fn dropout_{train,eval}_scalar(...)` -- Pure Rust scalar reference
//! - `unsafe fn dropout_{train,eval}_avx2(...)` -- AVX2 SIMD implementation
//! - `fn dropout_ptx() -> &'static str` -- PTX assembly source string

// ────────────────────────────────────────────────────────────────────────────
// Scalar implementation
// ────────────────────────────────────────────────────────────────────────────

/// Dropout in training mode (scalar reference).
///
/// `mask` is a pre-computed boolean mask (0.0 or 1.0), `p` is the drop probability.
/// `output[i] = mask[i] * input[i] / (1 - p)` (inverted dropout).
///
/// # Panics
/// Panics if dimensions don't match or `p >= 1.0`.
pub fn dropout_train_scalar(input: &[f32], mask: &[f32], p: f32, output: &mut [f32]) {
    assert_eq!(input.len(), mask.len(), "input/mask dimension mismatch");
    assert_eq!(input.len(), output.len(), "input/output dimension mismatch");
    assert!((0.0..1.0).contains(&p), "p must be in [0, 1), got {p}");

    let scale = 1.0 / (1.0 - p);
    for i in 0..input.len() {
        output[i] = mask[i] * input[i] * scale;
    }
}

/// Dropout in eval mode (scalar reference).
///
/// Identity function: `output[i] = input[i]`.
///
/// # Panics
/// Panics if `input.len() != output.len()`.
pub fn dropout_eval_scalar(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "input/output dimension mismatch");
    output.copy_from_slice(input);
}

// ────────────────────────────────────────────────────────────────────────────
// AVX2 implementation
// ────────────────────────────────────────────────────────────────────────────

/// AVX2 dropout (training) -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dropout_train_avx2(input: &[f32], mask: &[f32], p: f32, output: &mut [f32]) {
    dropout_train_scalar(input, mask, p, output);
}

/// AVX2 dropout (eval) -- delegates to scalar.
///
/// # Safety
/// Requires AVX2 support.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn dropout_eval_avx2(input: &[f32], output: &mut [f32]) {
    dropout_eval_scalar(input, output);
}

// ────────────────────────────────────────────────────────────────────────────
// PTX implementation
// ────────────────────────────────────────────────────────────────────────────

/// PTX assembly for dropout (training mode).
///
/// One thread per element. Each thread applies mask and scaling.
pub fn dropout_ptx() -> &'static str {
    r#".version 8.5
.target sm_90
.address_size 64
.visible .entry dropout_train_kernel(
    .param .u64 INPUT,
    .param .u64 MASK,
    .param .u64 OUT,
    .param .f32 SCALE,
    .param .u32 N
) {
    .reg .u32 %tid, %bid, %n, %idx;
    .reg .u64 %in_ptr, %mask_ptr, %out_ptr, %addr, %off64;
    .reg .f32 %in_val, %mask_val, %scale, %result;
    .reg .pred %p_bound;

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;

    ld.param.u32 %n, [N];
    ld.param.f32 %scale, [SCALE];
    ld.param.u64 %in_ptr, [INPUT];
    ld.param.u64 %mask_ptr, [MASK];
    ld.param.u64 %out_ptr, [OUT];

    // Global thread index
    mul.lo.u32 %idx, %bid, 256;
    add.u32 %idx, %idx, %tid;

    setp.ge.u32 %p_bound, %idx, %n;
    @%p_bound bra EXIT;

    mul.wide.u32 %off64, %idx, 4;

    // Load input[idx]
    add.u64 %addr, %in_ptr, %off64;
    ld.global.f32 %in_val, [%addr];

    // Load mask[idx]
    add.u64 %addr, %mask_ptr, %off64;
    ld.global.f32 %mask_val, [%addr];

    // result = mask * input * scale
    mul.f32 %result, %mask_val, %in_val;
    mul.f32 %result, %result, %scale;

    // Store output[idx]
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
    fn test_dropout_eval_is_identity() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0f32; 5];
        dropout_eval_scalar(&input, &mut output);
        assert_eq!(&output, &input);
    }

    #[test]
    fn test_dropout_train_all_kept() {
        let input = [1.0, 2.0, 3.0];
        let mask = [1.0, 1.0, 1.0]; // all kept
        let mut output = [0.0f32; 3];

        dropout_train_scalar(&input, &mask, 0.5, &mut output);
        // scale = 1 / (1 - 0.5) = 2.0
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[1] - 4.0).abs() < 1e-6);
        assert!((output[2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_dropout_train_all_dropped() {
        let input = [1.0, 2.0, 3.0];
        let mask = [0.0, 0.0, 0.0];
        let mut output = [99.0f32; 3];

        dropout_train_scalar(&input, &mask, 0.5, &mut output);
        assert_eq!(&output, &[0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dropout_train_zero_p() {
        // p=0 means no dropout, scale = 1/(1-0) = 1
        let input = [1.0, 2.0, 3.0];
        let mask = [1.0, 1.0, 1.0];
        let mut output = [0.0f32; 3];

        dropout_train_scalar(&input, &mask, 0.0, &mut output);
        assert_eq!(&output, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_dropout_dropped_units_are_zero() {
        let input = [5.0, 10.0, 15.0, 20.0];
        let mask = [1.0, 0.0, 1.0, 0.0]; // drop indices 1, 3
        let mut output = [0.0f32; 4];

        dropout_train_scalar(&input, &mask, 0.3, &mut output);
        assert_eq!(output[1], 0.0);
        assert_eq!(output[3], 0.0);
        assert!(output[0] > 0.0);
        assert!(output[2] > 0.0);
    }

    #[test]
    fn test_dropout_shape_preservation() {
        let n = 7;
        let input = vec![1.0f32; n];
        let mask = vec![1.0f32; n];
        let mut output = vec![0.0f32; n];

        dropout_train_scalar(&input, &mask, 0.1, &mut output);
        assert_eq!(output.len(), input.len());
    }

    proptest! {
        #[test]
        fn prop_dropout_eval_identity(n in 1usize..16) {
            let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3).collect();
            let mut output = vec![0.0f32; n];
            dropout_eval_scalar(&input, &mut output);

            for (i, (&a, &b)) in input.iter().zip(output.iter()).enumerate() {
                prop_assert_eq!(a, b, "eval not identity at {}", i);
            }
        }

        #[test]
        fn prop_dropout_train_finite(
            n in 1usize..10,
            p_int in 0u32..99,
        ) {
            let p = p_int as f32 / 100.0;
            let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
            let mask: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect();
            let mut output = vec![0.0f32; n];

            dropout_train_scalar(&input, &mask, p, &mut output);

            for (idx, &val) in output.iter().enumerate() {
                prop_assert!(val.is_finite(), "output[{idx}] = {val} not finite");
            }
        }
    }

    #[test]
    fn test_dropout_ptx_structure() {
        let ptx = dropout_ptx();
        assert!(ptx.contains(".entry dropout_train_kernel"));
        assert!(ptx.contains("mul.f32"));
        assert!(ptx.contains("ret;"));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_dropout_avx2_parity() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let input = [1.0, 2.0, 3.0, 4.0];
        let mask = [1.0, 0.0, 1.0, 0.0];
        let mut scalar_out = [0.0f32; 4];
        let mut avx2_out = [0.0f32; 4];
        dropout_train_scalar(&input, &mask, 0.5, &mut scalar_out);
        unsafe { dropout_train_avx2(&input, &mask, 0.5, &mut avx2_out) };
        assert_eq!(scalar_out, avx2_out);
    }
}
