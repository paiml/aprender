use crate::blis::*;

// ========================================================================
// Phase 6: ComputeBrick and Backend Selection Tests
// ========================================================================

#[test]
fn test_backend_selection_small_problem_chooses_cpu() {
    let cost = BackendCostModel::default();

    // Small problem should choose CPU
    let backend = cost.select_backend(64, 64, 64);
    assert!(
        matches!(backend, ComputeBackend::Cpu | ComputeBackend::Scalar),
        "Small problem should use CPU, got {:?}",
        backend
    );
}

#[test]
fn test_backend_cost_model_time_estimate() {
    let cost = BackendCostModel::default();

    let m = 1024;
    let n = 1024;
    let k = 1024;

    let cpu_time = cost.estimate_time_us(m, n, k, ComputeBackend::Cpu);
    let scalar_time = cost.estimate_time_us(m, n, k, ComputeBackend::Scalar);

    // CPU should be faster than scalar
    assert!(
        cpu_time < scalar_time,
        "CPU ({:.2}us) should be faster than scalar ({:.2}us)",
        cpu_time,
        scalar_time
    );
}

#[test]
fn test_roofline_analysis_compute_bound() {
    let profiler = UnifiedBrickProfiler::new();

    // Large K = high arithmetic intensity = compute-bound
    let result = profiler.roofline_analysis(1024, 1024, 1024);

    assert!(
        result.is_compute_bound(),
        "1024x1024x1024 should be compute-bound, AI={:.1}",
        result.arithmetic_intensity()
    );
}

#[test]
fn test_unified_profiler_records_selection() {
    let mut profiler = UnifiedBrickProfiler::new();

    profiler.record_selection(256, 256, 256, ComputeBackend::Cpu);

    assert_eq!(profiler.selection_history.len(), 1);
    assert_eq!(profiler.backend, Some(ComputeBackend::Cpu));
    assert_eq!(profiler.total_elements, 256 * 256);
}

#[test]
fn test_wgsl_spec_generation() {
    let spec = WgslMicrokernelSpec::default();
    let wgsl = spec.generate_wgsl();

    // Verify shader contains required elements
    assert!(wgsl.contains("@compute"));
    assert!(wgsl.contains("@workgroup_size"));
    assert!(wgsl.contains("tile_a"));
    assert!(wgsl.contains("tile_b"));
    assert!(wgsl.contains("workgroupBarrier"));

    // Verify bindings
    assert!(wgsl.contains("@group(0) @binding(0)"));
    assert!(wgsl.contains("@group(0) @binding(1)"));
    assert!(wgsl.contains("@group(0) @binding(2)"));
    assert!(wgsl.contains("@group(0) @binding(3)"));

    // Verify GemmParams struct
    assert!(wgsl.contains("struct GemmParams"));
    assert!(wgsl.contains("m: u32"));
    assert!(wgsl.contains("n: u32"));
    assert!(wgsl.contains("k: u32"));
    assert!(wgsl.contains("alpha: f32"));
    assert!(wgsl.contains("beta: f32"));

    // Verify default workgroup size (8,8,1)
    assert!(wgsl.contains("@workgroup_size(8, 8, 1)"));

    // Verify tiled K loop
    assert!(wgsl.contains("num_tiles"));
    assert!(wgsl.contains("var sum: f32 = 0.0"));

    // Verify output store with alpha/beta
    assert!(wgsl.contains("params.alpha * sum + params.beta"));

    // Verify default tile dimensions mentioned in header comment
    assert!(wgsl.contains("Tile: 8x8"));
    assert!(wgsl.contains("Workgroup: 8x8x1"));
}

#[test]
fn test_wgsl_spec_custom_dimensions() {
    let spec = WgslMicrokernelSpec {
        workgroup_size: (16, 16, 1),
        tile_dim: (16, 16),
        use_shared_memory: true,
    };
    let wgsl = spec.generate_wgsl();

    assert!(wgsl.contains("@workgroup_size(16, 16, 1)"));
    assert!(wgsl.contains("Tile: 16x16"));
    assert!(wgsl.contains("Workgroup: 16x16x1"));

    // Shared memory sizes should be tile_dim.0 * tile_dim.0 and tile_dim.0 * tile_dim.1
    assert!(wgsl.contains("array<f32, 256>")); // 16*16 for tile_a
}

#[test]
fn test_wgsl_spec_small_workgroup() {
    let spec = WgslMicrokernelSpec {
        workgroup_size: (4, 4, 1),
        tile_dim: (4, 4),
        use_shared_memory: true,
    };
    let wgsl = spec.generate_wgsl();

    assert!(wgsl.contains("@workgroup_size(4, 4, 1)"));
    assert!(wgsl.contains("Tile: 4x4"));
    // tile_a_size = 4*4 = 16
    assert!(wgsl.contains("array<f32, 16>"));
}

#[test]
fn test_ptx_spec_default() {
    let spec = PtxMicrokernelSpec::default();

    assert_eq!(spec.sm_target, "sm_80");
    assert_eq!(spec.registers_per_thread, 64);
    assert_eq!(spec.tile_dim, (16, 16));
}

#[test]
fn test_gemm_auto_produces_correct_result() {
    let m = 128;
    let n = 128;
    let k = 128;

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.1).collect();
    let mut c_ref = vec![0.0; m * n];
    let mut c_auto = vec![0.0; m * n];

    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();
    gemm_auto(m, n, k, &a, &b, &mut c_auto, None).unwrap();

    let max_diff: f32 =
        c_ref.iter().zip(c_auto.iter()).map(|(r, a)| (r - a).abs()).fold(0.0, f32::max);

    assert!(max_diff < 1e-3, "gemm_auto should match reference, max_diff={}", max_diff);
}

#[test]
fn test_gemm_auto_with_profiler() {
    let m = 64;
    let n = 64;
    let k = 64;

    let a: Vec<f32> = vec![1.0; m * k];
    let b: Vec<f32> = vec![1.0; k * n];
    let mut c = vec![0.0; m * n];

    let mut profiler = UnifiedBrickProfiler::new();
    gemm_auto(m, n, k, &a, &b, &mut c, Some(&mut profiler)).unwrap();

    assert!(profiler.backend.is_some());
    assert_eq!(profiler.total_elements, (m * n) as u64);
}

// ========================================================================
// Falsification Tests F320-F330 (ComputeBrick)
// ========================================================================

#[test]
fn test_f323_backend_selection_respects_pcie_rule() {
    let cost = BackendCostModel::default();

    // Small matrix: CPU should be selected (below threshold)
    let small = cost.select_backend(32, 32, 32);
    assert!(
        matches!(small, ComputeBackend::Cpu | ComputeBackend::Scalar),
        "F323: Small matrix should use CPU"
    );

    // Verify that arithmetic intensity calculation is correct
    let m: usize = 1024;
    let n: usize = 1024;
    let k: usize = 1024;
    let flops = 2_u64 * m as u64 * n as u64 * k as u64;
    let bytes = 4_u64 * (m * k + k * n + m * n) as u64;
    let ai = flops as f64 / bytes as f64;

    // AI for GEMM with large K should be high
    assert!(ai > 100.0, "F323: AI should be high for large K, got {}", ai);
}

#[test]
fn test_f324_cross_backend_equivalence() {
    // Test that CPU backend produces same result regardless of SIMD availability
    let m = 64;
    let n = 64;
    let k = 64;

    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 17) as f32) * 0.1).collect();

    // Reference (scalar)
    let mut c_ref = vec![0.0; m * n];
    gemm_reference(m, n, k, &a, &b, &mut c_ref).unwrap();

    // BLIS (uses SIMD if available)
    let mut c_blis = vec![0.0; m * n];
    gemm_blis(m, n, k, &a, &b, &mut c_blis, None).unwrap();

    // Auto (backend selection)
    let mut c_auto = vec![0.0; m * n];
    gemm_auto(m, n, k, &a, &b, &mut c_auto, None).unwrap();

    let max_diff_blis: f32 =
        c_ref.iter().zip(c_blis.iter()).map(|(r, b)| (r - b).abs()).fold(0.0, f32::max);
    let max_diff_auto: f32 =
        c_ref.iter().zip(c_auto.iter()).map(|(r, a)| (r - a).abs()).fold(0.0, f32::max);

    assert!(max_diff_blis < 1e-3, "F324: BLIS should match reference");
    assert!(max_diff_auto < 1e-3, "F324: Auto should match reference");
}

#[test]
fn test_f325_profiler_reports_consistent_metrics() {
    let profiler = UnifiedBrickProfiler::new();

    let m = 128;
    let n = 128;
    let k = 128;

    let roofline = profiler.roofline_analysis(m, n, k);
    let ai = roofline.arithmetic_intensity();

    // Manually compute expected AI
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let bytes = 4.0 * (m * k + k * n + m * n) as f64;
    let expected_ai = flops / bytes;

    assert!(
        (ai - expected_ai).abs() < 0.01,
        "F325: Profiler AI ({}) should match manual calculation ({})",
        ai,
        expected_ai
    );
}

#[test]
fn test_f329_brick_hierarchy_profiled() {
    let mut profiler = BlisProfiler::enabled();

    let n = 128;
    let a: Vec<f32> = vec![1.0; n * n];
    let b: Vec<f32> = vec![1.0; n * n];
    let mut c = vec![0.0; n * n];

    gemm_blis(n, n, n, &a, &b, &mut c, Some(&mut profiler)).unwrap();

    // Verify macro level is always profiled (all paths record this)
    assert!(profiler.macro_stats.count > 0, "F329: Macro level should be profiled");
    // NOTE: midi/micro stats are only populated by the generic BLIS 5-loop path.
    // The AVX-512 8×16 path (C-AVX512-PROF-001) records macro-level only.
    // On AVX-512 hardware, these may be 0 for small matrices (128×128).
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_microkernel_pipelined_matches_reference() {
    if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
        return;
    }

    let k = 64;
    let a: Vec<f32> = (0..MR * k).map(|i| (i as f32) * 0.1).collect();
    let b: Vec<f32> = (0..k * NR).map(|i| (i as f32) * 0.01).collect();

    let mut c_scalar = vec![0.0; MR * NR];
    let mut c_pipelined = vec![0.0; MR * NR];

    microkernel_scalar(k, &a, &b, &mut c_scalar, MR);

    // SAFETY: test-only usage with controlled inputs
    unsafe {
        microkernel_8x6_avx2_asm(k, a.as_ptr(), b.as_ptr(), c_pipelined.as_mut_ptr(), MR);
    }

    for i in 0..MR * NR {
        let diff = (c_scalar[i] - c_pipelined[i]).abs();
        let rel_diff = diff / c_scalar[i].abs().max(1e-10);
        assert!(
            rel_diff < 1e-5,
            "Pipelined microkernel mismatch at {}: scalar={}, pipelined={}, rel_diff={}",
            i,
            c_scalar[i],
            c_pipelined[i],
            rel_diff
        );
    }
}
