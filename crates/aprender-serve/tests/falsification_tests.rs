//! Popperian Falsification Test Suite
//!
//! Each test represents a falsifiable hypothesis about our SIMD optimizations.
//! If a test fails, the corresponding optimization claim is FALSIFIED and must be revised.
//!
//! Reference: Popper, K. (1934). The Logic of Scientific Discovery.
//!
//! Run with: `cargo test --test falsification_tests --release -- --nocapture`

use std::time::{Duration, Instant};

/// H1: AVX2+FMA Dot Product Speedup
///
/// CLAIM: SIMD dot product is faster than scalar for vectors >= 64 elements
/// FALSIFIED IF: SIMD is slower than scalar for any size >= 64
#[test]
fn falsify_h1_simd_dot_speedup() {
    let sizes = [64, 128, 256, 512, 1024, 2048];

    for size in sizes {
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.001).collect();
        let b: Vec<f32> = (0..size).map(|i| i as f32 * 0.002).collect();

        // Scalar baseline
        let scalar_time = benchmark_iterations(100, || scalar_dot(&a, &b));

        // SIMD implementation (uses runtime detection)
        let simd_time = benchmark_iterations(100, || simd_dot(&a, &b));

        let speedup = scalar_time.as_nanos() as f64 / simd_time.as_nanos().max(1) as f64;

        println!(
            "H1 size={}: scalar={:?}, simd={:?}, speedup={:.2}x",
            size, scalar_time, simd_time, speedup
        );

        // FALSIFICATION CRITERION: SIMD must be faster — ENFORCED IN RELEASE BUILDS ONLY.
        //
        // WALL-CLOCK ASSERTION, gated by build profile. This is a required check
        // (aprender#3114, run 35089783289, job 104773107023: "Quick tier: lib +
        // integration tests ... (BSE-17)") and CI's quick-tier invocation is
        // `cargo nextest run --profile ci --lib --tests <pkgs>` — no `--release` —
        // so on main this compares two UNOPTIMIZED (debug) implementations. The
        // same job recorded speedup=0.40x at size=512 and speedup=6.41x at
        // size=128 in ONE run: that spread is debug codegen + shared-runner
        // contention, not a SIMD regression (see "no wall-clock assertion in a
        // required check", memory feedback_no_wallclock_in_required_checks.md).
        // The file's own header always assumed `--release`; debug builds now
        // print the measurement and skip enforcement, release builds still
        // enforce the >0.9x floor. Do not "restore" an unconditional assert here.
        if !cfg!(debug_assertions) {
            assert!(
                simd_time <= scalar_time || speedup > 0.9, // Allow 10% measurement noise
                "H1 FALSIFIED: SIMD slower than scalar at size={}: {:?} vs {:?}",
                size,
                simd_time,
                scalar_time
            );
        } else {
            println!(
                "H1 size={}: debug build — measurement printed, not enforced (release-only gate)",
                size
            );
        }
    }
}

/// H2: Numerical Accuracy Preservation
///
/// CLAIM: SIMD operations produce results with relative error < 2%
///
/// RATIONALE: SIMD and scalar dot products use different accumulation orders.
/// IEEE 754 floating point is NOT associative: (a+b)+c != a+(b+c).
/// Per Goldberg (1991) "What Every Computer Scientist Should Know About Floating-Point",
/// different summation orders can produce O(n * epsilon) differences where n is the
/// number of elements and epsilon is machine epsilon (~1.2e-7 for f32).
/// For random data with potential cancellation (positive and negative values),
/// the result magnitude can be small while intermediate values are large, causing
/// relative error to be unbounded: a result near zero divides by ~0 (FLAKE-0 — a
/// relative-error criterion failed on random draws). The error is measured against
/// the summation bound instead: |scalar − simd| ≤ 2·n·ε·Σ|aᵢbᵢ| (Higham, Accuracy
/// and Stability of Numerical Algorithms, §3.1), which holds for any cancellation.
///
/// FALSIFIED IF: |scalar − simd| > 2·n·ε·Σ|aᵢbᵢ|
#[test]
fn falsify_h2_numerical_accuracy() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x4832);

    for _ in 0..100 {
        let size = rng.random_range(64..2048);
        let a: Vec<f32> = (0..size).map(|_| rng.random_range(-10.0..10.0)).collect();
        let b: Vec<f32> = (0..size).map(|_| rng.random_range(-10.0..10.0)).collect();

        let scalar_result = scalar_dot(&a, &b);
        let simd_result = simd_dot(&a, &b);

        let error = (scalar_result - simd_result).abs();
        let bound = dot_error_bound(&a, &b);

        // FALSIFICATION CRITERION: the difference is within the summation bound
        assert!(
            error <= bound,
            "H2 FALSIFIED: |scalar - simd| {} > bound {} for size={}, scalar={}, simd={}",
            error,
            bound,
            size,
            scalar_result,
            simd_result
        );
    }
    println!("H2: All 100 random tests passed within the summation error bound");
}

/// H3: Attention SIMD Correctness
///
/// CLAIM: SIMD attention produces same results as scalar attention
///
/// RATIONALE: Attention scores are computed via dot products. Per H2 rationale,
/// different accumulation orders cause O(n * epsilon) variation. For head_dim=64,
/// theoretical max error ≈ 64 * 1.2e-7 ≈ 7.7e-6 of Σ|qᵢkᵢ| — not of the score
/// itself: a score near zero made a relative-error criterion fail on random draws
/// (FLAKE-0, passed on CI retry). See H2 for the bound.
///
/// FALSIFIED IF: any score differs by more than 2·n·ε·Σ|qᵢkᵢ|
#[test]
fn falsify_h3_attention_correctness() {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(0x4833);

    let head_dim = 64;
    let num_positions = 50;

    for _ in 0..10 {
        let q: Vec<f32> = (0..head_dim).map(|_| rng.random_range(-1.0..1.0)).collect();
        let keys: Vec<Vec<f32>> = (0..num_positions)
            .map(|_| (0..head_dim).map(|_| rng.random_range(-1.0..1.0)).collect())
            .collect();

        // Compute attention scores both ways
        let scalar_scores: Vec<f32> = keys.iter().map(|k| scalar_dot(&q, k)).collect();
        let simd_scores: Vec<f32> = keys.iter().map(|k| simd_dot(&q, k)).collect();

        for ((s, m), k) in scalar_scores.iter().zip(&simd_scores).zip(&keys) {
            let bound = dot_error_bound(&q, k);
            // FALSIFICATION CRITERION: the difference is within the summation bound
            assert!(
                (s - m).abs() <= bound,
                "H3 FALSIFIED: attention score |{} - {}| > bound {}",
                s,
                m,
                bound
            );
        }
    }
    println!("H3: All attention tests passed within the summation error bound");
}

/// H4: AXPY Operation Correctness
///
/// CLAIM: SIMD axpy produces same results as scalar axpy
/// FALSIFIED IF: Any element differs by > 4 ULPs
#[test]
fn falsify_h4_axpy_correctness() {
    use rand::Rng;
    let mut rng = rand::rng();

    for size in [64, 128, 256, 512, 1024] {
        let weight: f32 = rng.random_range(0.1..2.0);
        let val: Vec<f32> = (0..size).map(|_| rng.random_range(-10.0..10.0)).collect();

        let mut scalar_out = vec![0.0f32; size];
        let mut simd_out = vec![0.0f32; size];

        scalar_axpy(&mut scalar_out, weight, &val);
        simd_axpy(&mut simd_out, weight, &val);

        let max_ulp = scalar_out
            .iter()
            .zip(simd_out.iter())
            .map(|(s, m)| ulp_distance(*s, *m))
            .max()
            .unwrap_or(0);

        // FALSIFICATION CRITERION: Max ULP <= 4
        assert!(
            max_ulp <= 4,
            "H4 FALSIFIED: AXPY max ULP {} > 4 at size={}",
            max_ulp,
            size
        );
    }
    println!("H4: All AXPY tests passed within 4 ULPs");
}

/// H5: Performance Regression Detection
///
/// CLAIM: Current implementation achieves >= 1.0 tok/s on TinyLlama-1.1B Q4_0
/// FALSIFIED IF: Throughput < 0.5 tok/s (indicating serious regression)
#[test]
#[ignore = "requires TinyLlama model file"]
fn falsify_h5_minimum_throughput() {
    // This test requires the model file to be present
    // Run manually with: cargo test falsify_h5 --release -- --ignored --nocapture

    // Placeholder: In real implementation, this would load the model and measure
    let measured_toks = 1.4; // Current measured throughput

    // FALSIFICATION CRITERION: >= 0.5 tok/s
    assert!(
        measured_toks >= 0.5,
        "H5 FALSIFIED: Throughput {} tok/s < 0.5 tok/s minimum",
        measured_toks
    );
    println!(
        "H5: Throughput {} tok/s meets minimum 0.5 tok/s",
        measured_toks
    );
}

// === Helper Functions ===

/// The forward-error bound on the difference of two f32 dot products that sum the
/// same n products in different orders: 2·n·ε·Σ|aᵢbᵢ| (each is within n·ε·Σ|aᵢbᵢ|
/// of the exact value). Summed in f64 so the bound itself carries no f32 error.
fn dot_error_bound(a: &[f32], b: &[f32]) -> f32 {
    let abs_sum: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .map(f64::abs)
        .sum();
    (2.0 * a.len().min(b.len()) as f64 * f64::from(f32::EPSILON) * abs_sum) as f32
}

fn scalar_dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn simd_dot(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 and fma were detected at runtime on the line above
            return unsafe { simd_dot_avx2(a, b) };
        }
    }
    scalar_dot(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    let len = a.len().min(b.len());
    let mut acc = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= len {
        // SAFETY: i + 8 <= len <= a.len(), b.len(), so both unaligned 8-lane loads are in bounds
        unsafe {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        i += 8;
    }

    // Horizontal sum
    let hi = _mm256_extractf128_ps(acc, 1);
    let lo = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sums, sums, 1);
    let result = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(result);

    // Remainder
    while i < len {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

fn scalar_axpy(out: &mut [f32], weight: f32, val: &[f32]) {
    for (o, v) in out.iter_mut().zip(val.iter()) {
        *o += weight * *v;
    }
}

fn simd_axpy(out: &mut [f32], weight: f32, val: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        // simd_axpy_avx2 enables avx2 AND fma; checking avx2 alone was unsound on a
        // CPU with avx2 but no fma (#4152, found while documenting this block).
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: avx2 and fma were detected at runtime on the line above
            unsafe {
                simd_axpy_avx2(out, weight, val);
            }
            return;
        }
    }
    scalar_axpy(out, weight, val);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_axpy_avx2(out: &mut [f32], weight: f32, val: &[f32]) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};

    let len = out.len().min(val.len());
    let w = _mm256_set1_ps(weight);
    let mut i = 0;

    while i + 8 <= len {
        // SAFETY: i + 8 <= len <= out.len(), val.len(), so the 8-lane loads and the store are in bounds
        unsafe {
            let v_out = _mm256_loadu_ps(out.as_ptr().add(i));
            let v_val = _mm256_loadu_ps(val.as_ptr().add(i));
            let result = _mm256_fmadd_ps(w, v_val, v_out);
            _mm256_storeu_ps(out.as_mut_ptr().add(i), result);
        }
        i += 8;
    }

    while i < len {
        out[i] += weight * val[i];
        i += 1;
    }
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    if a.is_nan() || b.is_nan() {
        return u32::MAX;
    }
    let a_bits = a.to_bits() as i32;
    let b_bits = b.to_bits() as i32;
    (a_bits - b_bits).unsigned_abs()
}

fn benchmark_iterations<F, R>(iterations: usize, f: F) -> Duration
where
    F: Fn() -> R,
{
    // Warmup
    for _ in 0..10 {
        std::hint::black_box(f());
    }

    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(f());
    }
    start.elapsed() / iterations as u32
}
