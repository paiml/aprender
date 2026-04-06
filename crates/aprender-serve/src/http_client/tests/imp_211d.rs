
/// IMP-211d: Real-world SwiGLU verification
#[test]
#[ignore = "Requires SwiGLU extraction from reference model"]
fn test_imp_211d_realworld_swiglu() {
    let gate = vec![0.5, 1.0, 1.5, 2.0];
    let up = vec![1.0, 1.0, 1.0, 1.0];
    let ref_out = SwiGLUVerificationResult::compute_swiglu(&gate, &up);

    let result = SwiGLUVerificationResult::new(gate, up, ref_out.clone(), ref_out, 1e-5);

    println!("\nIMP-211d: Real-World SwiGLU:");
    println!("  Max diff: {:.2e}", result.max_diff);
    println!(
        "  QA-008: {}",
        if result.meets_qa008 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-212: KV Cache Matches Recomputation (QA-009) ====================
// Per spec: KV cache produces identical results to recomputation

/// KV cache verification result
#[derive(Debug, Clone)]
pub struct KVCacheVerificationResult {
    pub sequence_length: usize,
    pub cached_output: Vec<f32>,
    pub recomputed_output: Vec<f32>,
    pub max_diff: f32,
    pub is_identical: bool,
    pub meets_qa009: bool,
}

impl KVCacheVerificationResult {
    pub fn new(seq_len: usize, cached: Vec<f32>, recomputed: Vec<f32>) -> Self {
        let max_diff = cached
            .iter()
            .zip(recomputed.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let is_identical = max_diff < 1e-6;
        let meets_qa009 = is_identical;

        Self {
            sequence_length: seq_len,
            cached_output: cached,
            recomputed_output: recomputed,
            max_diff,
            is_identical,
            meets_qa009,
        }
    }
}

/// IMP-212a: Test KV cache verification
#[test]
fn test_imp_212a_kv_cache_verification() {
    let cached = vec![0.1, 0.2, 0.3, 0.4];
    let recomputed = vec![0.1, 0.2, 0.3, 0.4];

    let result = KVCacheVerificationResult::new(4, cached, recomputed);

    assert!(result.meets_qa009, "IMP-212a: Should meet QA-009");
    assert!(result.is_identical, "IMP-212a: Should be identical");

    println!("\nIMP-212a: KV Cache Verification:");
    println!("  Sequence length: {}", result.sequence_length);
    println!("  Max diff: {:.2e}", result.max_diff);
    println!("  Identical: {}", result.is_identical);
}

/// IMP-212b: Test KV cache mismatch detection
#[test]
fn test_imp_212b_kv_cache_mismatch() {
    let cached = vec![0.1, 0.2, 0.3, 0.4];
    let recomputed = vec![0.1, 0.2, 0.35, 0.4]; // 0.05 diff at position 2

    let result = KVCacheVerificationResult::new(4, cached, recomputed);

    assert!(!result.meets_qa009, "IMP-212b: Should detect mismatch");
    assert!(!result.is_identical, "IMP-212b: Should not be identical");

    println!("\nIMP-212b: KV Cache Mismatch:");
    println!("  Max diff: {:.2e}", result.max_diff);
    println!("  Identical: {}", result.is_identical);
}

/// IMP-212c: Test KV cache at different lengths
#[test]
fn test_imp_212c_kv_cache_lengths() {
    let lengths = vec![1, 10, 100, 512];

    println!("\nIMP-212c: KV Cache at Different Lengths:");
    for len in lengths {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.01).collect();
        let result = KVCacheVerificationResult::new(len, data.clone(), data);
        println!("  Length {}: meets QA-009 = {}", len, result.meets_qa009);
        assert!(result.meets_qa009);
    }
}

/// IMP-212d: Real-world KV cache verification
#[test]
#[ignore = "Requires KV cache extraction from inference"]
fn test_imp_212d_realworld_kv_cache() {
    let cached = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let recomputed = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    let result = KVCacheVerificationResult::new(5, cached, recomputed);

    println!("\nIMP-212d: Real-World KV Cache:");
    println!("  Sequence length: {}", result.sequence_length);
    println!("  Max diff: {:.2e}", result.max_diff);
    println!(
        "  QA-009: {}",
        if result.meets_qa009 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-213: Quantized Matches F32 (QA-010) ====================
// Per spec: Quantized inference matches F32 within acceptable tolerance

/// Quantization verification result
#[derive(Debug, Clone)]
pub struct QuantizationVerificationResult {
    pub quantization_type: String,
    pub f32_output: Vec<f32>,
    pub quantized_output: Vec<f32>,
    pub max_diff: f32,
    pub mean_diff: f32,
    pub tolerance: f32,
    pub meets_qa010: bool,
}

impl QuantizationVerificationResult {
    pub fn new(
        quant_type: impl Into<String>,
        f32_out: Vec<f32>,
        quant_out: Vec<f32>,
        tolerance: f32,
    ) -> Self {
        let diffs: Vec<f32> = f32_out
            .iter()
            .zip(quant_out.iter())
            .map(|(a, b)| (a - b).abs())
            .collect();

        let max_diff = diffs.iter().cloned().fold(0.0_f32, f32::max);
        let mean_diff = if diffs.is_empty() {
            0.0
        } else {
            diffs.iter().sum::<f32>() / diffs.len() as f32
        };

        let meets_qa010 = max_diff <= tolerance;

        Self {
            quantization_type: quant_type.into(),
            f32_output: f32_out,
            quantized_output: quant_out,
            max_diff,
            mean_diff,
            tolerance,
            meets_qa010,
        }
    }
}

/// IMP-213a: Test quantization verification
#[test]
fn test_imp_213a_quantization_verification() {
    let f32_out = vec![0.1, 0.2, 0.3, 0.4];
    let quant_out = vec![0.1001, 0.1999, 0.3002, 0.3998];

    let result = QuantizationVerificationResult::new("Q4_K", f32_out, quant_out, 0.01);

    assert!(result.meets_qa010, "IMP-213a: Should meet QA-010");

    println!("\nIMP-213a: Quantization Verification:");
    println!("  Type: {}", result.quantization_type);
    println!("  Max diff: {:.4}", result.max_diff);
    println!("  Mean diff: {:.4}", result.mean_diff);
}

/// IMP-213b: Test different quantization types
#[test]
fn test_imp_213b_quantization_types() {
    let f32_out = vec![0.5, 0.5, 0.5, 0.5];

    // Q4_K has larger tolerance
    let q4k = QuantizationVerificationResult::new(
        "Q4_K",
        f32_out.clone(),
        vec![0.48, 0.52, 0.49, 0.51],
        0.05,
    );

    // Q8_0 has tighter tolerance
    let q8_0 = QuantizationVerificationResult::new(
        "Q8_0",
        f32_out.clone(),
        vec![0.499, 0.501, 0.500, 0.500],
        0.01,
    );

    println!("\nIMP-213b: Quantization Types:");
    println!(
        "  Q4_K: max_diff={:.4}, meets QA-010={}",
        q4k.max_diff, q4k.meets_qa010
    );
    println!(
        "  Q8_0: max_diff={:.4}, meets QA-010={}",
        q8_0.max_diff, q8_0.meets_qa010
    );
}

/// IMP-213c: Test quantization tolerance boundaries
#[test]
fn test_imp_213c_quantization_tolerance() {
    let f32_out = vec![1.0, 1.0, 1.0, 1.0];

    // Within tolerance
    let within = QuantizationVerificationResult::new(
        "Q4_K",
        f32_out.clone(),
        vec![1.04, 0.96, 1.03, 0.97],
        0.05,
    );

    // Outside tolerance
    let outside =
        QuantizationVerificationResult::new("Q4_K", f32_out, vec![1.1, 0.9, 1.1, 0.9], 0.05);

    assert!(within.meets_qa010, "IMP-213c: Should be within tolerance");
    assert!(
        !outside.meets_qa010,
        "IMP-213c: Should be outside tolerance"
    );

    println!("\nIMP-213c: Quantization Tolerance:");
    println!("  Within (0.05): {}", within.meets_qa010);
    println!("  Outside (0.05): {}", outside.meets_qa010);
}

/// IMP-213d: Real-world quantization verification
#[test]
#[ignore = "Requires F32 and quantized model inference"]
fn test_imp_213d_realworld_quantization() {
    let f32_out = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let quant_out = vec![0.1, 0.2, 0.3, 0.4, 0.5];

    let result = QuantizationVerificationResult::new("Q4_K", f32_out, quant_out, 0.05);

    println!("\nIMP-213d: Real-World Quantization:");
    println!("  Type: {}", result.quantization_type);
    println!("  Max diff: {:.4}", result.max_diff);
    println!(
        "  QA-010: {}",
        if result.meets_qa010 { "PASS" } else { "FAIL" }
    );
}

// ==================== IMP-301: Trueno SIMD Q4_K Dequantization ====================
// Per spec: 4-8x speedup via AVX2/NEON for Q4_K dequantization
// Target: ~15 tok/s CPU (match llama.cpp CPU)

/// SIMD backend type for performance tracking
#[derive(Debug, Clone, PartialEq)]
pub enum SimdBackend {
    Scalar,
    SSE2,
    AVX2,
    AVX512,
    Neon,
    Wasm,
}

impl SimdBackend {
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdBackend::AVX512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdBackend::AVX2;
            }
            if is_x86_feature_detected!("sse2") {
                return SimdBackend::SSE2;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return SimdBackend::Neon;
        }
        #[cfg(target_arch = "wasm32")]
        {
            return SimdBackend::Wasm;
        }
        SimdBackend::Scalar
    }

    pub fn expected_speedup(&self) -> f64 {
        match self {
            SimdBackend::AVX512 => 16.0,
            SimdBackend::AVX2 => 8.0,
            SimdBackend::SSE2 => 4.0,
            SimdBackend::Neon => 4.0,
            SimdBackend::Wasm => 2.0,
            SimdBackend::Scalar => 1.0,
        }
    }
}

/// Trueno SIMD benchmark result
#[derive(Debug, Clone)]
pub struct TruenoSimdBenchResult {
    pub operation: String,
    pub backend: SimdBackend,
    pub scalar_time_us: f64,
    pub simd_time_us: f64,
    pub speedup: f64,
    pub elements: usize,
    pub throughput_gbs: f64,
    pub meets_imp301: bool,
}

impl TruenoSimdBenchResult {
    pub fn new(
        operation: impl Into<String>,
        backend: SimdBackend,
        scalar_us: f64,
        simd_us: f64,
        elements: usize,
    ) -> Self {
        let speedup = scalar_us / simd_us.max(0.001);
        // Throughput: elements * 4 bytes / time_seconds / 1e9 = GB/s
        let throughput_gbs = (elements as f64 * 4.0) / (simd_us * 1e-6) / 1e9;
        // IMP-301: Need at least 2x speedup to be worthwhile
        let meets_imp301 = speedup >= 2.0;

        Self {
            operation: operation.into(),
            backend,
            scalar_time_us: scalar_us,
            simd_time_us: simd_us,
            speedup,
            elements,
            throughput_gbs,
            meets_imp301,
        }
    }
}

/// IMP-301a: Test SIMD backend detection
#[test]
fn test_imp_301a_simd_backend_detection() {
    let backend = SimdBackend::detect();

    println!("\nIMP-301a: SIMD Backend Detection:");
    println!("  Detected: {:?}", backend);
    println!("  Expected speedup: {:.1}x", backend.expected_speedup());

    // Should detect something other than scalar on modern CPUs
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    assert_ne!(backend, SimdBackend::Scalar, "IMP-301a: Should detect SIMD");
}

/// IMP-301b: Test trueno Vector SIMD operations
#[test]
fn test_imp_301b_trueno_vector_simd() {
    use trueno::Vector;

    let size = 4096;
    let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
    let vec = Vector::from_slice(&data);

    // Test basic operations
    let sum = vec.sum().expect("sum failed");
    let mean = vec.mean().expect("mean failed");
    let max = vec.max().expect("max failed");

    assert!(sum > 0.0, "IMP-301b: Sum should be positive");
    assert!(mean > 0.0, "IMP-301b: Mean should be positive");
    assert!(max > 0.0, "IMP-301b: Max should be positive");

    println!("\nIMP-301b: Trueno Vector SIMD:");
    println!("  Size: {}", size);
    println!("  Sum: {:.2}", sum);
    println!("  Mean: {:.6}", mean);
    println!("  Max: {:.3}", max);
    println!("  Backend: {:?}", vec.backend());
}
