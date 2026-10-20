
#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx2_q4k_q8k_dot_varying_scales() {
    if !is_x86_feature_detected!("avx2") {
        return;
    }

    // 2 super-blocks with different Q8K scales
    let block = build_q4k_test_block(1.0, 0.0, 8);
    let mut q4k_data = Vec::with_capacity(144 * 2);
    q4k_data.extend_from_slice(&block);
    q4k_data.extend_from_slice(&block);
    let q8k_scales = vec![0.5f32, 2.0f32];
    let q8k_quants = vec![3i8; 256 * 2];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    // SAFETY: Preconditions verified by caller or enclosing context
    let avx2 = unsafe { fused_q4k_q8k_dot_avx2(&q4k_data, &q8k_scales, &q8k_quants) }.expect("avx2");

    let diff = (scalar - avx2).abs();
    let rel_tolerance = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < rel_tolerance,
        "varying scales: scalar={scalar} vs avx2={avx2}, diff={diff}"
    );
}

// ============================================================================
// FUSED Q4_K × Q8_K DOT PRODUCT — AVX-512 VNNI COVERAGE TESTS
// ============================================================================
// These tests call the unsafe fused_q4k_q8k_dot_avx512vnni and
// fused_q4k_q8k_dot_avx512vnni_opt directly to cover both AVX-512 code paths.

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 3);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    let vnni =
        // SAFETY: Preconditions verified by caller or enclosing context
        unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) }.expect("expected value");

    let diff = (scalar - vnni).abs();
    assert!(
        diff < 1.0,
        "scalar={scalar} vs avx512vnni={vnni}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_zero_quants() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 0);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![0i8; 256];

    let result =
        // SAFETY: Preconditions verified by caller or enclosing context
        unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) }.expect("expected value");
    assert!(
        result.abs() < 1e-6,
        "zero × zero should produce ~0, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_multi_superblock() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 5);
    let mut q4k_data = Vec::with_capacity(144 * 3);
    for _ in 0..3 {
        q4k_data.extend_from_slice(&block);
    }
    let q8k_scales = vec![1.0f32; 3];
    let q8k_quants = vec![2i8; 256 * 3];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    let vnni =
        // SAFETY: Preconditions verified by caller or enclosing context
        unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) }.expect("expected value");

    let diff = (scalar - vnni).abs();
    let tol = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < tol,
        "3-block: scalar={scalar} vs avx512vnni={vnni}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_invalid_data_length() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let q4k_data = vec![0u8; 100]; // Not a multiple of 144
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err(), "should fail for invalid data length");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_q8k_dot_buffer_too_small() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 1);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 128]; // Too small

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_q8k_dot_avx512vnni(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err(), "should fail for too-small Q8K buffer");
}

// --- fused_q4k_q8k_dot_avx512vnni_opt tests ---

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_parity_with_scalar() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 3);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    let vnni_opt =
        // SAFETY: Preconditions verified by caller or enclosing context
        unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) }.expect("expected value");

    let diff = (scalar - vnni_opt).abs();
    assert!(
        diff < 1.0,
        "scalar={scalar} vs avx512vnni_opt={vnni_opt}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_zero() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 0);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![0i8; 256];

    let result =
        // SAFETY: Preconditions verified by caller or enclosing context
        unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) }.expect("expected value");
    assert!(
        result.abs() < 1e-6,
        "zero × zero should produce ~0, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_multi_superblock() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 8);
    let mut q4k_data = Vec::with_capacity(144 * 4);
    for _ in 0..4 {
        q4k_data.extend_from_slice(&block);
    }
    let q8k_scales = vec![0.5f32, 1.0, 2.0, 0.25];
    let q8k_quants = vec![3i8; 256 * 4];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    let vnni_opt =
        // SAFETY: Preconditions verified by caller or enclosing context
        unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) }.expect("expected value");

    let diff = (scalar - vnni_opt).abs();
    let tol = scalar.abs().max(1.0) * 0.01;
    assert!(
        diff < tol,
        "4-block varying: scalar={scalar} vs vnni_opt={vnni_opt}, diff={diff}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_invalid_data_length() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let q4k_data = vec![0u8; 100];
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 256];

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_buffer_too_small() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 1);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![1i8; 128]; // Too small

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) };
    assert!(result.is_err());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_opt_q4k_q8k_dot_negative_quants() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(2.0, 0.5, 7);
    let q4k_data = block.to_vec();
    let q8k_scales = vec![1.0f32];
    let q8k_quants = vec![-3i8; 256];

    let scalar = fused_q4k_q8k_dot(&q4k_data, &q8k_scales, &q8k_quants).expect("scalar");
    let vnni_opt =
        // SAFETY: Preconditions verified by caller or enclosing context
        unsafe { fused_q4k_q8k_dot_avx512vnni_opt(&q4k_data, &q8k_scales, &q8k_quants) }.expect("expected value");

    let diff = (scalar - vnni_opt).abs();
    let tol = scalar.abs().max(1.0) * 0.02;
    assert!(
        diff < tol,
        "negative quants: scalar={scalar} vs vnni_opt={vnni_opt}, diff={diff}"
    );
}

// --- fused_q4k_dot_avx512_vnni tests (Q4K × f32 activations) ---

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_dot_exercises_code_path() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 5);
    let q4k_data = block.to_vec();
    let activations = vec![1.0f32; 256];

    // AVX512 VNNI Q4K dot uses internal int8 quantization of activations,
    // so results may differ from the AVX2 public path. Just verify it runs
    // and produces a finite value.
    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_dot_avx512_vnni(&q4k_data, &activations) }.expect("result");
    assert!(
        result.is_finite(),
        "should produce finite result, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_dot_zero_activations() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(1.0, 0.0, 8);
    let q4k_data = block.to_vec();
    let activations = vec![0.0f32; 256];

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_dot_avx512_vnni(&q4k_data, &activations) }.expect("result");
    assert!(
        result.abs() < 1e-3,
        "zero activations should produce ~0, got {result}"
    );
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_dot_invalid_data_length() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let q4k_data = vec![0u8; 100]; // Not multiple of 144
    let activations = vec![1.0f32; 256];

    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_dot_avx512_vnni(&q4k_data, &activations) };
    assert!(result.is_err());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512vnni_q4k_dot_multi_superblock() {
    if !is_x86_feature_detected!("avx512vnni") {
        return;
    }

    let block = build_q4k_test_block(0.5, 0.1, 6);
    let mut q4k_data = Vec::with_capacity(144 * 2);
    q4k_data.extend_from_slice(&block);
    q4k_data.extend_from_slice(&block);
    let activations: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) * 0.01).collect();

    // Exercises multi-block AVX512 VNNI path with varied activations
    // SAFETY: Preconditions verified by caller or enclosing context
    let result = unsafe { fused_q4k_dot_avx512_vnni(&q4k_data, &activations) }.expect("result");
    assert!(
        result.is_finite(),
        "should produce finite result, got {result}"
    );
}

// ============================================================================
// avx512_quantize_dot (#5652): exact integer result per branch
// ============================================================================

/// Scalar model of `avx512_quantize_dot`: u8 nibbles times activations quantized to i8
/// over the slice's own [min, max] range.
#[cfg(target_arch = "x86_64")]
fn quantize_dot_model(act: &[f32], q: &[u8; 32]) -> (i32, f32) {
    let max = act.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let min = act.iter().copied().fold(f32::INFINITY, f32::min);
    let scale = if max > min { 127.0 / (max - min) } else { 1.0 };
    let sum = act
        .iter()
        .zip(q)
        .map(|(&a, &n)| i32::from(((a - min) * scale).round() as i8) * i32::from(n))
        .sum();
    (sum, scale)
}

#[cfg(target_arch = "x86_64")]
fn avx512_quantize_dot_checked(act: &[f32], q: &[u8; 32]) -> Option<(i32, f32)> {
    if !(is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512vnni"))
    {
        return None;
    }
    // SAFETY: AVX-512F/BW/VNNI verified above; `q` is 32 readable bytes.
    Some(unsafe {
        avx512_quantize_dot(
            act,
            std::arch::x86_64::_mm256_loadu_si256(q.as_ptr().cast()),
        )
    })
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512_quantize_dot_ramp_matches_scalar_model() {
    let act: Vec<f32> = (0..32).map(|i| i as f32 - 10.0).collect();
    let q: [u8; 32] = std::array::from_fn(|i| (i % 16) as u8);
    let Some((sum, scale)) = avx512_quantize_dot_checked(&act, &q) else {
        return;
    };
    // range 31 -> scale 127/31; the minimum maps to 0 and the maximum to 127.
    assert!((scale - 127.0 / 31.0).abs() < 1e-6, "scale={scale}");
    assert_eq!((sum, scale), quantize_dot_model(&act, &q));
    assert!(sum > 0);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512_quantize_dot_constant_slice_uses_unit_scale_and_zero_sum() {
    let q = [15u8; 32];
    let Some((sum, scale)) = avx512_quantize_dot_checked(&[3.5f32; 32], &q) else {
        return;
    };
    // max == min takes the `else` branch: scale 1.0, every activation quantizes to 0.
    assert_eq!((sum, scale), (0, 1.0));
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_avx512_quantize_dot_zero_nibbles_give_zero_and_max_nibbles_dominate() {
    let act: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let Some((zero, _)) = avx512_quantize_dot_checked(&act, &[0u8; 32]) else {
        return;
    };
    assert_eq!(zero, 0);
    let mut hot = [0u8; 32];
    hot[31] = 15; // only the activation at the range maximum (-> 127) contributes
    let (sum, _) = avx512_quantize_dot_checked(&act, &hot).expect("features checked above");
    assert_eq!(sum, 15 * 127);
}
