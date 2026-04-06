use super::*;

/// Returns the expected backend when AVX-512 is excluded (memory-bound/mixed ops).
#[cfg(target_arch = "x86_64")]
fn expected_non_avx512_backend() -> Backend {
    if is_x86_feature_detected!("avx2") {
        Backend::AVX2
    } else if is_x86_feature_detected!("avx") {
        Backend::AVX
    } else if is_x86_feature_detected!("sse2") {
        Backend::SSE2
    } else {
        Backend::Scalar
    }
}

/// Returns the expected backend for compute-bound ops (allows AVX-512).
#[cfg(target_arch = "x86_64")]
fn expected_compute_backend() -> Backend {
    if is_x86_feature_detected!("avx512f") {
        Backend::AVX512
    } else {
        expected_non_avx512_backend()
    }
}

#[test]
fn test_backend_enum() {
    assert_eq!(Backend::Scalar, Backend::Scalar);
    assert_ne!(Backend::Scalar, Backend::AVX2);
}

#[test]
fn test_op_complexity_ordering() {
    assert!(OpComplexity::Low < OpComplexity::Medium);
    assert!(OpComplexity::Medium < OpComplexity::High);
}

#[test]
fn test_select_best_available_backend() {
    let backend = select_best_available_backend();

    // On x86_64, we should get at least SSE2 (baseline for x86_64)
    // or a more advanced SIMD backend if available
    #[cfg(target_arch = "x86_64")]
    {
        // x86_64 baseline is SSE2, so we should never get Scalar on x86_64
        assert_ne!(backend, Backend::Scalar);
        // Verify it's one of the x86 SIMD backends
        assert!(matches!(backend, Backend::SSE2 | Backend::AVX | Backend::AVX2 | Backend::AVX512));
    }

    // On other platforms, we might get Scalar or platform-specific SIMD
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Just verify we got a valid backend
        assert!(matches!(
            backend,
            Backend::Scalar
                | Backend::SSE2
                | Backend::AVX
                | Backend::AVX2
                | Backend::AVX512
                | Backend::NEON
                | Backend::WasmSIMD
        ));
    }
}

#[test]
fn test_backend_selection_is_deterministic() {
    // Backend selection should be deterministic (same result on multiple calls)
    let backend1 = select_best_available_backend();
    let backend2 = select_best_available_backend();
    assert_eq!(backend1, backend2);
}

#[test]
fn test_backend_selection_is_cached() {
    // Verify backend selection is cached (OnceLock)
    // Multiple calls should return the same backend without re-detection
    let backend1 = select_best_available_backend();

    // Call 1000 times to ensure caching is working
    // If not cached, this would be significantly slower
    for _ in 0..1000 {
        let backend = select_best_available_backend();
        assert_eq!(backend, backend1, "Backend selection must be consistent");
    }
}

#[test]
fn test_backend_select_best() {
    // Test Backend::select_best() method
    let backend = Backend::select_best();
    assert_eq!(backend, select_best_available_backend());
}

#[test]
fn test_backend_variants() {
    // Test all backend variants
    let backends = vec![
        Backend::Scalar,
        Backend::SSE2,
        Backend::AVX,
        Backend::AVX2,
        Backend::AVX512,
        Backend::NEON,
        Backend::WasmSIMD,
        Backend::GPU,
        Backend::Auto,
    ];

    // Verify all variants are distinct
    for (i, backend1) in backends.iter().enumerate() {
        for (j, backend2) in backends.iter().enumerate() {
            if i == j {
                assert_eq!(backend1, backend2);
            } else {
                assert_ne!(backend1, backend2);
            }
        }
    }
}

#[test]
fn test_backend_debug() {
    // Verify Debug trait works
    let backend = Backend::AVX2;
    let debug_str = format!("{:?}", backend);
    assert!(debug_str.contains("AVX2"));

    let backend = Backend::Auto;
    let debug_str = format!("{:?}", backend);
    assert!(debug_str.contains("Auto"));
}

#[test]
fn test_backend_clone() {
    let backend = Backend::AVX2;
    #[allow(clippy::clone_on_copy)]
    let cloned = backend.clone();
    assert_eq!(backend, cloned);
}

#[test]
fn test_backend_copy() {
    let backend = Backend::SSE2;
    let copied = backend;
    assert_eq!(backend, copied);
}

#[test]
fn test_op_complexity_values() {
    assert_eq!(OpComplexity::Low as i32, 0);
    assert_eq!(OpComplexity::Medium as i32, 1);
    assert_eq!(OpComplexity::High as i32, 2);
}

#[test]
fn test_op_complexity_ord() {
    // Test PartialOrd
    assert!(OpComplexity::Low < OpComplexity::Medium);
    assert!(OpComplexity::Medium < OpComplexity::High);
    assert!(OpComplexity::Low < OpComplexity::High);

    // Test Ord
    use std::cmp::Ordering;
    assert_eq!(OpComplexity::Low.cmp(&OpComplexity::Medium), Ordering::Less);
    assert_eq!(OpComplexity::Medium.cmp(&OpComplexity::High), Ordering::Less);
    assert_eq!(OpComplexity::High.cmp(&OpComplexity::Medium), Ordering::Greater);
    assert_eq!(OpComplexity::Low.cmp(&OpComplexity::Low), Ordering::Equal);
}

#[test]
fn test_op_complexity_eq() {
    assert_eq!(OpComplexity::Low, OpComplexity::Low);
    assert_eq!(OpComplexity::Medium, OpComplexity::Medium);
    assert_eq!(OpComplexity::High, OpComplexity::High);
    assert_ne!(OpComplexity::Low, OpComplexity::High);
}

#[test]
fn test_op_complexity_debug() {
    let complexity = OpComplexity::Medium;
    let debug_str = format!("{:?}", complexity);
    assert!(debug_str.contains("Medium"));
}

#[test]
fn test_op_complexity_clone() {
    let complexity = OpComplexity::High;
    #[allow(clippy::clone_on_copy)]
    let cloned = complexity.clone();
    assert_eq!(complexity, cloned);
}

#[test]
fn test_op_complexity_copy() {
    let complexity = OpComplexity::Low;
    let copied = complexity;
    assert_eq!(complexity, copied);
}

#[test]
fn test_trueno_error_reexport() {
    // Verify error types are re-exported correctly
    let _: Result<()> = Ok(());
    let err: Result<()> = Err(TruenoError::EmptyVector);
    assert!(err.is_err());
}

#[test]
fn test_vector_reexport() {
    // Verify Vector is re-exported correctly
    let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
    assert_eq!(v.len(), 3);
}

#[test]
fn test_matrix_reexport() {
    // Verify Matrix is re-exported correctly
    let m = Matrix::zeros(2, 2);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 2);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_detect_x86_backend() {
    let backend = detect_x86_backend();
    // On x86_64, we should get at least SSE2
    assert!(matches!(backend, Backend::SSE2 | Backend::AVX | Backend::AVX2));
    // Should NOT return AVX-512 (intentionally avoided for safety)
    assert_ne!(backend, Backend::AVX512);
}

#[test]
fn test_operation_type_enum() {
    // Verify OperationType variants are distinct
    assert_ne!(OperationType::MemoryBound, OperationType::ComputeBound);
    assert_ne!(OperationType::MemoryBound, OperationType::Mixed);
    assert_ne!(OperationType::ComputeBound, OperationType::Mixed);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_select_backend_for_memory_bound_prefers_avx2() {
    let backend = select_backend_for_operation(OperationType::MemoryBound);

    // Should prefer AVX2 over AVX-512 for memory-bound operations
    // (Based on AVX-512 performance analysis showing 0.67-1.01x scalar)
    assert_eq!(backend, expected_non_avx512_backend());
    assert_ne!(backend, Backend::AVX512);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_select_backend_for_compute_bound_allows_avx512() {
    let backend = select_backend_for_operation(OperationType::ComputeBound);

    // Should prefer AVX-512 for compute-bound operations where it excels (7-14x scalar)
    assert_eq!(backend, expected_compute_backend());
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_select_backend_for_mixed_prefers_avx2() {
    let backend = select_backend_for_operation(OperationType::Mixed);

    // Mixed operations should default to AVX2 for safety (avoid AVX-512 thermal throttling)
    assert_eq!(backend, expected_non_avx512_backend());
    assert_ne!(backend, Backend::AVX512);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_default_backend_selection_avoids_avx512() {
    // The default backend selection (detect_x86_backend) should avoid AVX-512
    let default_backend = select_best_available_backend();

    // Even on CPUs with AVX-512, default selection should prefer AVX2
    assert_eq!(default_backend, expected_non_avx512_backend());

    // Verify AVX-512 is NOT returned by default
    assert_ne!(default_backend, Backend::AVX512);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_backend_selection_consistency() {
    // Memory-bound and Mixed should return same backend (AVX2-first)
    let memory_backend = select_backend_for_operation(OperationType::MemoryBound);
    let mixed_backend = select_backend_for_operation(OperationType::Mixed);

    assert_eq!(memory_backend, mixed_backend);

    // Compute-bound may differ (allows AVX-512)
    let compute_backend = select_backend_for_operation(OperationType::ComputeBound);

    // Compute uses expected_compute_backend, memory uses expected_non_avx512_backend
    assert_eq!(compute_backend, expected_compute_backend());
    assert_eq!(memory_backend, expected_non_avx512_backend());
}

#[cfg(not(target_arch = "x86_64"))]
#[test]
fn test_select_backend_for_operation_non_x86() {
    // On non-x86 platforms, all operation types should return platform-specific backend
    let memory = select_backend_for_operation(OperationType::MemoryBound);
    let compute = select_backend_for_operation(OperationType::ComputeBound);
    let mixed = select_backend_for_operation(OperationType::Mixed);

    // All should return the same backend (ARM NEON, WASM SIMD, or Scalar)
    assert_eq!(memory, compute);
    assert_eq!(memory, mixed);

    #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
    {
        #[cfg(target_feature = "neon")]
        assert_eq!(memory, Backend::NEON);
    }

    #[cfg(target_arch = "wasm32")]
    {
        #[cfg(target_feature = "simd128")]
        assert_eq!(memory, Backend::WasmSIMD);
    }
}

// ========================================================================
// Additional Coverage Tests
// ========================================================================

#[test]
fn test_operation_type_debug() {
    let mem = OperationType::MemoryBound;
    let debug_str = format!("{:?}", mem);
    assert!(debug_str.contains("MemoryBound"));

    let compute = OperationType::ComputeBound;
    let debug_str = format!("{:?}", compute);
    assert!(debug_str.contains("ComputeBound"));

    let mixed = OperationType::Mixed;
    let debug_str = format!("{:?}", mixed);
    assert!(debug_str.contains("Mixed"));
}

#[test]
fn test_operation_type_clone() {
    let op_type = OperationType::ComputeBound;
    #[allow(clippy::clone_on_copy)]
    let cloned = op_type.clone();
    assert_eq!(op_type, cloned);
}

#[test]
fn test_operation_type_copy() {
    let op_type = OperationType::Mixed;
    let copied = op_type;
    assert_eq!(op_type, copied);
}

#[test]
fn test_operation_type_equality() {
    assert_eq!(OperationType::MemoryBound, OperationType::MemoryBound);
    assert_eq!(OperationType::ComputeBound, OperationType::ComputeBound);
    assert_eq!(OperationType::Mixed, OperationType::Mixed);
}

#[test]
fn test_backend_all_variants_debug() {
    // Test Debug for all backend variants
    assert!(format!("{:?}", Backend::Scalar).contains("Scalar"));
    assert!(format!("{:?}", Backend::SSE2).contains("SSE2"));
    assert!(format!("{:?}", Backend::AVX).contains("AVX"));
    assert!(format!("{:?}", Backend::AVX512).contains("AVX512"));
    assert!(format!("{:?}", Backend::NEON).contains("NEON"));
    assert!(format!("{:?}", Backend::WasmSIMD).contains("WasmSIMD"));
    assert!(format!("{:?}", Backend::GPU).contains("GPU"));
}

#[test]
fn test_symmetric_eigen_reexport() {
    // Verify SymmetricEigen is re-exported correctly
    // Just verify it's accessible
    let _ = std::mem::size_of::<SymmetricEigen>();
}

#[test]
fn test_hash_reexport() {
    // Verify hash functions are re-exported correctly
    let result = hash_bytes(b"test");
    assert_ne!(result, 0);

    let key_hash = hash_key("test_key");
    assert_ne!(key_hash, 0);

    let keys = ["a", "b", "c", "d"];
    let batch_result = hash_keys_batch(&keys);
    assert_eq!(batch_result.len(), 4);
}
