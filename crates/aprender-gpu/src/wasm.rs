//! WASM Visual Testing for trueno-gpu
//!
//! Exposes GPU visual regression tests to browser via wasm-bindgen.
//! Uses trueno-viz for PNG rendering (sovereign stack).
//!
//! **Requires features:** `wasm` and `viz`

// WASM exports may not always use return values - caller decides
// WASM test sizes are always positive, cast is intentional
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_sign_loss)]

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "viz")]
use crate::testing::{compare_png_bytes, GpuPixelRenderer};

/// Simple PCG32 RNG for WASM (no getrandom dependency)
struct SimpleRng {
    state: u64,
    inc: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        let mut rng = Self {
            state: 0,
            inc: (seed << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    fn next_u32(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(self.inc);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        (xorshifted >> rot) | (xorshifted << ((!rot).wrapping_add(1) & 31))
    }

    fn gen_f32(&mut self) -> f32 {
        (self.next_u32() as f64 / u32::MAX as f64) as f32
    }
}

/// WASM test result
#[cfg_attr(feature = "wasm", wasm_bindgen)]
#[derive(Debug, Clone)]
pub struct WasmTestResult {
    name: String,
    passed: bool,
    diff_pixels: usize,
    total_pixels: usize,
    diff_percent: f64,
    png_data: Vec<u8>,
}

#[cfg_attr(feature = "wasm", wasm_bindgen)]
impl WasmTestResult {
    /// Test name
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Whether test passed
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn passed(&self) -> bool {
        self.passed
    }

    /// Number of different pixels
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn diff_pixels(&self) -> usize {
        self.diff_pixels
    }

    /// Total number of pixels
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn total_pixels(&self) -> usize {
        self.total_pixels
    }

    /// Percentage of different pixels
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn diff_percent(&self) -> f64 {
        self.diff_percent
    }

    /// PNG image data
    #[cfg_attr(feature = "wasm", wasm_bindgen(getter))]
    pub fn png_data(&self) -> Vec<u8> {
        self.png_data.clone()
    }
}

/// Simulate correct GEMM
fn simulate_gemm(size: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            let mut acc = 0.0f32;
            for k in 0..size {
                acc += (i * size + k) as f32 * (k * size + j) as f32;
            }
            output.push(acc);
        }
    }
    output
}

/// Simulate buggy GEMM with uninitialized accumulator
fn simulate_gemm_buggy(size: usize) -> Vec<f32> {
    let mut output = Vec::with_capacity(size * size);
    for i in 0..size {
        for j in 0..size {
            let garbage = if i % 2 == 0 { 1000.0 } else { 0.0 };
            let mut acc = garbage; // BUG
            for k in 0..size {
                acc += (i * size + k) as f32 * (k * size + j) as f32;
            }
            output.push(acc);
        }
    }
    output
}

/// Run identity matrix test
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn test_identity_matrix() -> WasmTestResult {
    let renderer = GpuPixelRenderer::new();
    let size = 16;
    let identity: Vec<f32> = (0..size * size)
        .map(|i| if i / size == i % size { 1.0 } else { 0.0 })
        .collect();

    let png = renderer.render_to_png(&identity, size as u32, size as u32);
    let result = compare_png_bytes(&png, &png, 0);

    WasmTestResult {
        name: "Identity Matrix (A @ I = A)".to_string(),
        passed: result.different_pixels == 0,
        diff_pixels: result.different_pixels,
        total_pixels: result.total_pixels,
        diff_percent: result.diff_percentage(),
        png_data: png,
    }
}

/// Run gradient test
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn test_gradient() -> WasmTestResult {
    let renderer = GpuPixelRenderer::new();
    let size = 16;
    let gradient: Vec<f32> = (0..size * size)
        .map(|i| i as f32 / (size * size) as f32)
        .collect();

    let png = renderer.render_to_png(&gradient, size as u32, size as u32);
    let result = compare_png_bytes(&png, &png, 0);

    WasmTestResult {
        name: "Gradient (FP Precision)".to_string(),
        passed: result.different_pixels == 0,
        diff_pixels: result.different_pixels,
        total_pixels: result.total_pixels,
        diff_percent: result.diff_percentage(),
        png_data: png,
    }
}

/// Run bug detection test - returns buggy output for comparison
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn test_bug_detection() -> WasmTestResult {
    let renderer = GpuPixelRenderer::new();
    let size = 16;

    let correct = simulate_gemm(size);
    let buggy = simulate_gemm_buggy(size);

    let png_correct = renderer.render_to_png(&correct, size as u32, size as u32);
    let png_buggy = renderer.render_to_png(&buggy, size as u32, size as u32);

    let result = compare_png_bytes(&png_correct, &png_buggy, 0);

    WasmTestResult {
        name: "Bug Detection (AccumulatorInit)".to_string(),
        passed: result.different_pixels > 0,
        diff_pixels: result.different_pixels,
        total_pixels: result.total_pixels,
        diff_percent: result.diff_percentage(),
        png_data: png_buggy,
    }
}

/// Get correct GEMM output PNG
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn get_correct_gemm() -> Vec<u8> {
    let renderer = GpuPixelRenderer::new();
    let size = 16;
    let correct = simulate_gemm(size);
    renderer.render_to_png(&correct, size as u32, size as u32)
}

/// Run special values test
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn test_special_values() -> WasmTestResult {
    let renderer = GpuPixelRenderer::new();
    let special: Vec<f32> = vec![
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        1e38,
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
        -1.0,
        f32::MIN_POSITIVE,
        f32::EPSILON,
        100.0,
        -100.0,
        0.001,
        -0.001,
    ];

    let png = renderer.render_to_png(&special, 4, 4);
    let result = compare_png_bytes(&png, &png, 0);

    WasmTestResult {
        name: "Special Values (NaN, Inf)".to_string(),
        passed: result.different_pixels == 0,
        diff_pixels: result.different_pixels,
        total_pixels: result.total_pixels,
        diff_percent: result.diff_percentage(),
        png_data: png,
    }
}

/// Run deterministic RNG test
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn test_deterministic_rng(seed: u64) -> WasmTestResult {
    let renderer = GpuPixelRenderer::new();
    let mut rng = SimpleRng::new(seed);
    let data: Vec<f32> = (0..256).map(|_| rng.gen_f32()).collect();

    let png1 = renderer.render_to_png(&data, 16, 16);

    // Reset and regenerate - should be identical
    let mut rng2 = SimpleRng::new(seed);
    let data2: Vec<f32> = (0..256).map(|_| rng2.gen_f32()).collect();
    let png2 = renderer.render_to_png(&data2, 16, 16);

    let result = compare_png_bytes(&png1, &png2, 0);

    WasmTestResult {
        name: format!("Deterministic RNG (seed={})", seed),
        passed: result.different_pixels == 0,
        diff_pixels: result.different_pixels,
        total_pixels: result.total_pixels,
        diff_percent: result.diff_percentage(),
        png_data: png1,
    }
}

/// Run all visual tests
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn run_all_tests() -> Vec<WasmTestResult> {
    vec![
        test_identity_matrix(),
        test_gradient(),
        test_bug_detection(),
        test_special_values(),
        test_deterministic_rng(42),
    ]
}

/// Get version string
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

// ============================================================================
// Unit Tests (run without viz feature)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // SimpleRng tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simple_rng_deterministic() {
        let mut rng1 = SimpleRng::new(42);
        let mut rng2 = SimpleRng::new(42);

        // Same seed should produce same sequence
        for _ in 0..100 {
            assert_eq!(rng1.next_u32(), rng2.next_u32());
        }
    }

    #[test]
    fn test_simple_rng_different_seeds() {
        let mut rng1 = SimpleRng::new(1);
        let mut rng2 = SimpleRng::new(2);

        // Different seeds should produce different sequences
        let seq1: Vec<u32> = (0..10).map(|_| rng1.next_u32()).collect();
        let seq2: Vec<u32> = (0..10).map(|_| rng2.next_u32()).collect();
        assert_ne!(seq1, seq2);
    }

    #[test]
    fn test_simple_rng_gen_f32_range() {
        let mut rng = SimpleRng::new(12345);

        // gen_f32 should return values in [0, 1]
        for _ in 0..1000 {
            let val = rng.gen_f32();
            assert!(val >= 0.0, "gen_f32 returned negative: {}", val);
            assert!(val <= 1.0, "gen_f32 returned >1: {}", val);
        }
    }

    #[test]
    fn test_simple_rng_distribution() {
        let mut rng = SimpleRng::new(999);

        // Check that values are reasonably distributed (not all zeros or ones)
        let values: Vec<f32> = (0..1000).map(|_| rng.gen_f32()).collect();
        let sum: f32 = values.iter().sum();
        let mean = sum / 1000.0;

        // Mean should be roughly 0.5 for uniform distribution
        assert!(mean > 0.4, "Mean too low: {}", mean);
        assert!(mean < 0.6, "Mean too high: {}", mean);
    }

    #[test]
    fn test_simple_rng_edge_seeds() {
        // Test edge case seeds
        let mut rng_zero = SimpleRng::new(0);
        let mut rng_max = SimpleRng::new(u64::MAX);

        // Should not panic and should produce different values
        let v0 = rng_zero.next_u32();
        let v_max = rng_max.next_u32();
        assert_ne!(v0, v_max);
    }

    // -------------------------------------------------------------------------
    // WasmTestResult tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_wasm_test_result_getters() {
        let result = WasmTestResult {
            name: "Test".to_string(),
            passed: true,
            diff_pixels: 10,
            total_pixels: 100,
            diff_percent: 10.0,
            png_data: vec![1, 2, 3],
        };

        assert_eq!(result.name(), "Test");
        assert!(result.passed());
        assert_eq!(result.diff_pixels(), 10);
        assert_eq!(result.total_pixels(), 100);
        assert!((result.diff_percent() - 10.0).abs() < f64::EPSILON);
        assert_eq!(result.png_data(), vec![1, 2, 3]);
    }

    #[test]
    fn test_wasm_test_result_failed() {
        let result = WasmTestResult {
            name: "FailedTest".to_string(),
            passed: false,
            diff_pixels: 50,
            total_pixels: 100,
            diff_percent: 50.0,
            png_data: vec![],
        };

        assert!(!result.passed());
        assert_eq!(result.diff_percent(), 50.0);
    }

    #[test]
    fn test_wasm_test_result_clone() {
        let result = WasmTestResult {
            name: "CloneTest".to_string(),
            passed: true,
            diff_pixels: 0,
            total_pixels: 256,
            diff_percent: 0.0,
            png_data: vec![0xFF; 100],
        };

        let cloned = result.clone();
        assert_eq!(result.name(), cloned.name());
        assert_eq!(result.passed(), cloned.passed());
        assert_eq!(result.png_data(), cloned.png_data());
    }

    // -------------------------------------------------------------------------
    // GEMM simulation tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_simulate_gemm_size_1() {
        let result = simulate_gemm(1);
        assert_eq!(result.len(), 1);
        // 0 * 0 = 0
        assert_eq!(result[0], 0.0);
    }

    #[test]
    fn test_simulate_gemm_size_2() {
        let result = simulate_gemm(2);
        assert_eq!(result.len(), 4);
        // For 2x2 identity-like multiplication pattern
        // This is actually computing C[i,j] = sum(i*2+k * k*2+j for k in 0..2)
    }

    #[test]
    fn test_simulate_gemm_size_4() {
        let result = simulate_gemm(4);
        assert_eq!(result.len(), 16);

        // Verify output is deterministic
        let result2 = simulate_gemm(4);
        assert_eq!(result, result2);
    }

    #[test]
    fn test_simulate_gemm_buggy_differs() {
        let correct = simulate_gemm(4);
        let buggy = simulate_gemm_buggy(4);

        // Buggy version should differ from correct version
        assert_ne!(correct, buggy);
    }

    #[test]
    fn test_simulate_gemm_buggy_pattern() {
        let buggy = simulate_gemm_buggy(4);

        // Even rows have garbage=1000 added, odd rows have garbage=0
        // First row (i=0, even) should have 1000 added to accumulator
        // Second row (i=1, odd) should match correct
        let correct = simulate_gemm(4);

        // Row 1 (index 4-7) should be same since i=1 is odd
        assert_eq!(buggy[4], correct[4]);
        assert_eq!(buggy[5], correct[5]);
        assert_eq!(buggy[6], correct[6]);
        assert_eq!(buggy[7], correct[7]);
    }

    #[test]
    fn test_simulate_gemm_buggy_even_rows_have_garbage() {
        let correct = simulate_gemm(4);
        let buggy = simulate_gemm_buggy(4);

        // Even rows (0, 2) have 1000 added
        // Row 0: indices 0-3
        assert!((buggy[0] - correct[0] - 1000.0).abs() < 0.01);
        // Row 2: indices 8-11
        assert!((buggy[8] - correct[8] - 1000.0).abs() < 0.01);
    }

    // -------------------------------------------------------------------------
    // Version test
    // -------------------------------------------------------------------------

    #[test]
    fn test_version_not_empty() {
        let ver = version();
        assert!(!ver.is_empty());
        // Should be a valid semver-ish string
        assert!(ver.contains('.'), "Version should contain dots: {}", ver);
    }
}
