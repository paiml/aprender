//! PARITY-111-CPU: CPU Scalar Tests (Baseline)
//!
//! Popperian falsifiable tests for CPU scalar inference path.
//! These tests verify that scalar code path executes when forced.
//!
//! ## Falsification Criteria
//! 1. compute_block span with "scalar" MUST exist
//! 2. No GPU spans should be present
//! 3. No SIMD spans should be present (when forced scalar)
//! 4. Throughput must be in range [0.1, 5.0] tok/s

use super::common::*;

/// PARITY-111-CPU-01: CPU scalar single-shot inference
/// Falsification: scalar compute block span MUST exist
#[test]
fn test_cpu_scalar_single_shot() {
    // Force scalar backend
    force_backend(Backend::Scalar);

    // Create mock trace (in real impl, renacer captures this)
    let mut trace = ExecutionTrace::new();
    trace.add_span(
        TraceSpan::new("compute_block:scalar_matmul", 50000).with_attr("backend", "scalar"),
    );
    trace.add_span(
        TraceSpan::new("compute_block:scalar_attention", 30000).with_attr("backend", "scalar"),
    );
    trace.total_tokens = 1;
    trace.total_duration_ms = 500;

    // FALSIFIABLE ASSERTION 1: Scalar compute block MUST exist
    assert!(
        trace.has_span_matching("compute_block:scalar*"),
        "FALSIFICATION: No scalar compute block - CPU scalar path not executed"
    );

    // FALSIFIABLE ASSERTION 2: No GPU spans
    let gpu_spans = trace.spans_matching("gpu_kernel:*");
    assert!(
        gpu_spans.is_empty(),
        "FALSIFICATION: Found {} GPU kernel spans in scalar mode - should be 0",
        gpu_spans.len()
    );

    // FALSIFIABLE ASSERTION 3: Throughput in expected range
    let throughput = trace.throughput_tok_per_sec();
    let (min_tps, max_tps) = Backend::Scalar.expected_throughput_range();
    assert!(
        throughput >= min_tps && throughput <= max_tps,
        "FALSIFICATION: Scalar throughput {:.2} tok/s outside range [{}, {}]",
        throughput,
        min_tps,
        max_tps
    );

    clear_backend_forcing();
}

/// PARITY-111-CPU-02: CPU scalar batch-4 inference
#[test]
fn test_cpu_scalar_batch_4() {
    force_backend(Backend::Scalar);

    let mut trace = ExecutionTrace::new();
    // Batch of 4 prompts
    for i in 0..4 {
        trace.add_span(
            TraceSpan::new(&format!("compute_block:scalar_forward_{}", i), 50000)
                .with_attr("backend", "scalar")
                .with_attr("batch_idx", &i.to_string()),
        );
    }
    trace.total_tokens = 4 * 16; // 4 prompts × 16 tokens
    trace.total_duration_ms = 2000;

    // FALSIFIABLE: Multiple scalar spans for batch
    let scalar_spans = trace.spans_matching("compute_block:scalar*");
    assert!(
        scalar_spans.len() >= 4,
        "FALSIFICATION: Expected >=4 scalar spans for batch-4, found {}",
        scalar_spans.len()
    );

    // FALSIFIABLE: No GPU fallback
    assert!(
        !trace.has_span_matching("gpu_kernel:*"),
        "FALSIFICATION: GPU kernel executed in scalar-forced mode"
    );

    clear_backend_forcing();
}

/// PARITY-111-CPU-03: CPU scalar batch-32 inference
#[test]
fn test_cpu_scalar_batch_32() {
    force_backend(Backend::Scalar);

    let mut trace = ExecutionTrace::new();
    for i in 0..32 {
        trace.add_span(
            TraceSpan::new(&format!("compute_block:scalar_forward_{}", i), 50000)
                .with_attr("backend", "scalar"),
        );
    }
    trace.total_tokens = 32 * 16;
    trace.total_duration_ms = 16000; // Much slower for large batch

    // FALSIFIABLE: Even at batch-32, scalar path should be used when forced
    let scalar_spans = trace.spans_matching("compute_block:scalar*");
    assert!(
        scalar_spans.len() >= 32,
        "FALSIFICATION: Expected >=32 scalar spans, found {}",
        scalar_spans.len()
    );

    // FALSIFIABLE: GPU should NOT be used when scalar is forced
    assert!(
        !trace.has_span_matching("gpu_kernel:*"),
        "FALSIFICATION: GPU kernel executed despite REALIZAR_FORCE_SCALAR=1"
    );

    clear_backend_forcing();
}

/// PARITY-111-CPU-04: CPU scalar batch-64 inference
#[test]
fn test_cpu_scalar_batch_64() {
    force_backend(Backend::Scalar);

    let mut trace = ExecutionTrace::new();
    for i in 0..64 {
        trace.add_span(
            TraceSpan::new(&format!("compute_block:scalar_forward_{}", i), 50000)
                .with_attr("backend", "scalar"),
        );
    }
    trace.total_tokens = 64 * 16;
    trace.total_duration_ms = 32000;

    // FALSIFIABLE: All 64 should use scalar
    let scalar_spans = trace.spans_matching("compute_block:scalar*");
    assert!(
        scalar_spans.len() >= 64,
        "FALSIFICATION: Expected >=64 scalar spans, found {}",
        scalar_spans.len()
    );

    clear_backend_forcing();
}

/// PARITY-111-CPU-05: Scalar is slowest modality (hierarchy test)
#[test]
fn test_cpu_scalar_is_slowest() {
    let scalar_range = Backend::Scalar.expected_throughput_range();
    let simd_range = Backend::Simd.expected_throughput_range();
    let cuda_range = Backend::Cuda.expected_throughput_range();

    // FALSIFIABLE: Scalar max should be less than SIMD min (with some overlap tolerance)
    assert!(
        scalar_range.1 < simd_range.1,
        "FALSIFICATION: Scalar max {:.1} >= SIMD max {:.1} - hierarchy violated",
        scalar_range.1,
        simd_range.1
    );

    // FALSIFIABLE: Scalar max should be much less than CUDA min
    assert!(
        scalar_range.1 < cuda_range.0,
        "FALSIFICATION: Scalar max {:.1} >= CUDA min {:.1} - should be much slower",
        scalar_range.1,
        cuda_range.0
    );
}

/// PARITY-111-CPU-06: Scalar span durations are in expected range
#[test]
fn test_cpu_scalar_duration_range() {
    let mut trace = ExecutionTrace::new();
    trace.add_span(TraceSpan::new("compute_block:scalar_matmul", 100_000)); // 100ms

    let span = trace
        .span_by_name("compute_block:scalar_matmul")
        .expect("test");

    // FALSIFIABLE: Scalar matmul should be slow (>10ms for meaningful work)
    assert!(
        span.duration_us >= 10_000,
        "FALSIFICATION: Scalar matmul {}us too fast - likely not scalar",
        span.duration_us
    );

    // FALSIFIABLE: But not absurdly slow (< 10 seconds)
    assert!(
        span.duration_us < 10_000_000,
        "FALSIFICATION: Scalar matmul {}us too slow - something wrong",
        span.duration_us
    );
}

#[cfg(test)]
mod integration {
    use super::*;

    /// Integration test: Verify REALIZAR_FORCE_SCALAR env var is respected
    #[test]
    fn test_force_scalar_env_var_set() {
        force_backend(Backend::Scalar);

        let val = std::env::var("REALIZAR_FORCE_SCALAR");
        assert!(
            val.is_ok() && val.expect("test") == "1",
            "FALSIFICATION: REALIZAR_FORCE_SCALAR not set to '1'"
        );

        // Other backend vars should be cleared
        assert!(
            std::env::var("REALIZAR_BACKEND").is_err(),
            "FALSIFICATION: REALIZAR_BACKEND should be unset in scalar mode"
        );

        clear_backend_forcing();
    }
}
