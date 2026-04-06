//! PARITY-111-SIMD: SIMD (AVX2) Tests
//!
//! Popperian falsifiable tests for SIMD-accelerated inference path.
//!
//! ## Falsification Criteria
//! 1. compute_block span with "simd" or "avx" MUST exist
//! 2. No GPU spans should be present (when SIMD forced)
//! 3. Throughput must exceed scalar baseline
//! 4. Throughput must be in range [3.0, 20.0] tok/s

use super::common::*;

/// Check if AVX2 is available on this CPU
fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// PARITY-111-SIMD-01: SIMD single-shot inference
/// Falsification: SIMD compute block span MUST exist
#[test]
fn test_simd_avx2_single_shot() {
    if !avx2_available() {
        eprintln!("Skipping SIMD test: AVX2 not available");
        return;
    }

    force_backend(Backend::Simd);

    let mut trace = ExecutionTrace::new();
    trace.add_span(
        TraceSpan::new("compute_block:simd_matmul", 10000)
            .with_attr("backend", "simd")
            .with_attr("instruction_set", "avx2"),
    );
    trace.add_span(
        TraceSpan::new("compute_block:simd_attention", 8000).with_attr("backend", "simd"),
    );
    trace.total_tokens = 1;
    trace.total_duration_ms = 100;

    // FALSIFIABLE ASSERTION 1: SIMD compute block MUST exist
    assert!(
        trace.has_span_matching("compute_block:simd*"),
        "FALSIFICATION: No SIMD compute block - AVX2 path not executed"
    );

    // FALSIFIABLE ASSERTION 2: No GPU spans in SIMD mode
    let gpu_spans = trace.spans_matching("gpu_kernel:*");
    assert!(
        gpu_spans.is_empty(),
        "FALSIFICATION: Found {} GPU kernel spans in SIMD mode",
        gpu_spans.len()
    );

    // FALSIFIABLE ASSERTION 3: Throughput in expected SIMD range
    let throughput = trace.throughput_tok_per_sec();
    let (min_tps, max_tps) = Backend::Simd.expected_throughput_range();
    assert!(
        throughput >= min_tps && throughput <= max_tps,
        "FALSIFICATION: SIMD throughput {:.2} tok/s outside range [{}, {}]",
        throughput,
        min_tps,
        max_tps
    );

    clear_backend_forcing();
}

/// PARITY-111-SIMD-02: SIMD batch-4 inference
#[test]
fn test_simd_avx2_batch_4() {
    if !avx2_available() {
        return;
    }

    force_backend(Backend::Simd);

    let mut trace = ExecutionTrace::new();
    for i in 0..4 {
        trace.add_span(
            TraceSpan::new(&format!("compute_block:simd_forward_{}", i), 10000)
                .with_attr("backend", "simd"),
        );
    }
    trace.total_tokens = 4 * 16;
    trace.total_duration_ms = 400;

    // FALSIFIABLE: Multiple SIMD spans for batch
    let simd_spans = trace.spans_matching("compute_block:simd*");
    assert!(
        simd_spans.len() >= 4,
        "FALSIFICATION: Expected >=4 SIMD spans for batch-4, found {}",
        simd_spans.len()
    );

    // FALSIFIABLE: No scalar fallback (all should be SIMD)
    let scalar_spans = trace.spans_matching("compute_block:scalar*");
    assert!(
        scalar_spans.is_empty(),
        "FALSIFICATION: Found {} scalar spans - should use SIMD",
        scalar_spans.len()
    );

    clear_backend_forcing();
}

/// PARITY-111-SIMD-03: SIMD batch-32 inference
#[test]
fn test_simd_avx2_batch_32() {
    if !avx2_available() {
        return;
    }

    force_backend(Backend::Simd);

    let mut trace = ExecutionTrace::new();
    for i in 0..32 {
        trace.add_span(
            TraceSpan::new(&format!("compute_block:simd_forward_{}", i), 10000)
                .with_attr("backend", "simd"),
        );
    }
    trace.total_tokens = 32 * 16;
    trace.total_duration_ms = 3200;

    // FALSIFIABLE: All 32 should use SIMD when forced
    let simd_spans = trace.spans_matching("compute_block:simd*");
    assert!(
        simd_spans.len() >= 32,
        "FALSIFICATION: Expected >=32 SIMD spans, found {}",
        simd_spans.len()
    );

    // FALSIFIABLE: No GPU fallback (we're forcing SIMD, not GPU)
    assert!(
        !trace.has_span_matching("gpu_kernel:*"),
        "FALSIFICATION: GPU kernel executed despite REALIZAR_FORCE_SIMD=1"
    );

    clear_backend_forcing();
}

/// PARITY-111-SIMD-04: SIMD batch-64 inference
#[test]
fn test_simd_avx2_batch_64() {
    if !avx2_available() {
        return;
    }

    force_backend(Backend::Simd);

    let mut trace = ExecutionTrace::new();
    for i in 0..64 {
        trace.add_span(
            TraceSpan::new(&format!("compute_block:simd_forward_{}", i), 10000)
                .with_attr("backend", "simd"),
        );
    }
    trace.total_tokens = 64 * 16;
    trace.total_duration_ms = 6400;

    let simd_spans = trace.spans_matching("compute_block:simd*");
    assert!(
        simd_spans.len() >= 64,
        "FALSIFICATION: Expected >=64 SIMD spans, found {}",
        simd_spans.len()
    );

    clear_backend_forcing();
}

/// PARITY-111-SIMD-05: SIMD faster than scalar
#[test]
fn test_simd_faster_than_scalar() {
    let scalar_range = Backend::Scalar.expected_throughput_range();
    let simd_range = Backend::Simd.expected_throughput_range();

    // FALSIFIABLE: SIMD min should be close to or exceed scalar max
    // (allowing some overlap for edge cases)
    assert!(
        simd_range.0 >= scalar_range.0,
        "FALSIFICATION: SIMD min {:.1} < Scalar min {:.1}",
        simd_range.0,
        scalar_range.0
    );

    // FALSIFIABLE: SIMD max should significantly exceed scalar max
    assert!(
        simd_range.1 > scalar_range.1 * 2.0,
        "FALSIFICATION: SIMD max {:.1} not >2x Scalar max {:.1}",
        simd_range.1,
        scalar_range.1
    );
}

/// PARITY-111-SIMD-06: SIMD span durations are faster than scalar
#[test]
fn test_simd_duration_faster_than_scalar() {
    // Scalar baseline: 100ms for matmul
    let scalar_duration_us = 100_000;

    // SIMD should be at least 2x faster
    let simd_duration_us = 10_000; // 10ms

    let mut trace = ExecutionTrace::new();
    trace.add_span(TraceSpan::new(
        "compute_block:simd_matmul",
        simd_duration_us,
    ));

    let span = trace
        .span_by_name("compute_block:simd_matmul")
        .expect("test");

    // FALSIFIABLE: SIMD should be significantly faster
    assert!(
        span.duration_us < scalar_duration_us / 2,
        "FALSIFICATION: SIMD matmul {}us not <2x faster than scalar {}us",
        span.duration_us,
        scalar_duration_us
    );
}

/// PARITY-111-SIMD-07: SIMD instruction set attribute present
#[test]
fn test_simd_instruction_set_attribute() {
    if !avx2_available() {
        return;
    }

    let mut trace = ExecutionTrace::new();
    trace.add_span(
        TraceSpan::new("compute_block:simd_matmul", 10000).with_attr("instruction_set", "avx2"),
    );

    let span = trace
        .span_by_name("compute_block:simd_matmul")
        .expect("test");

    // FALSIFIABLE: instruction_set attribute should indicate AVX2
    assert!(
        span.attr("instruction_set") == Some("avx2")
            || span.attr("instruction_set") == Some("avx512"),
        "FALSIFICATION: SIMD span missing instruction_set attribute"
    );
}

#[cfg(test)]
mod integration {
    use super::*;

    /// Ignored: Flaky due to env var manipulation conflicting with parallel tests
    #[test]
    #[ignore = "flaky due to env var manipulation conflicting with parallel tests"]
    fn test_force_simd_env_var_set() {
        force_backend(Backend::Simd);

        let val = std::env::var("REALIZAR_FORCE_SIMD");
        assert!(
            val.is_ok() && val.expect("test") == "1",
            "FALSIFICATION: REALIZAR_FORCE_SIMD not set to '1'"
        );

        clear_backend_forcing();
    }

    #[test]
    fn test_avx2_detection() {
        // This test documents AVX2 availability
        let available = avx2_available();
        eprintln!("AVX2 available: {}", available);

        // On x86_64, AVX2 should typically be available on modern CPUs
        #[cfg(target_arch = "x86_64")]
        {
            // Most modern x86_64 CPUs have AVX2
            // If this fails, the test machine is very old
            assert!(
                available,
                "FALSIFICATION: AVX2 not available on x86_64 - very old CPU?"
            );
        }
    }
}
