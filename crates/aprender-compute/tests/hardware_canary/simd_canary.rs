// ============================================================================
// SIMD CANARY TESTS
// ============================================================================

/// Canary Test: Verify AVX-512 is detected when using native RUSTFLAGS
///
/// This test panics if AVX-512 is NOT detected, proving that RUSTFLAGS
/// are correctly enabling CPU features.
///
/// **If this test fails:**
/// - Your RUSTFLAGS may not include `-C target-cpu=native`
/// - Your CPU may not support AVX-512 (unlikely on Threadripper)
/// - Run: `RUSTFLAGS="-C target-cpu=native" cargo test`
#[test]
#[cfg(target_arch = "x86_64")]
#[ignore = "Requires AVX-512 hardware (not available on CI runners)"]
fn canary_avx512_detected() {
    use std::arch::is_x86_feature_detected;

    // Check AVX-512 Foundation (minimum for AVX-512)
    let avx512f = is_x86_feature_detected!("avx512f");

    if !avx512f {
        // Collect diagnostic information
        let avx2 = is_x86_feature_detected!("avx2");
        let avx = is_x86_feature_detected!("avx");
        let fma = is_x86_feature_detected!("fma");

        panic!(
            "\n\
            ╔══════════════════════════════════════════════════════════════════════════════╗\n\
            ║  SIMD CANARY FAILED: AVX-512 NOT DETECTED!                                   ║\n\
            ╠══════════════════════════════════════════════════════════════════════════════╣\n\
            ║  This Lambda Labs box has a Threadripper with AVX-512 support.               ║\n\
            ║  If this test fails, RUSTFLAGS are not set correctly.                        ║\n\
            ║                                                                              ║\n\
            ║  FIX: Use `make coverage` or set RUSTFLAGS='-C target-cpu=native'            ║\n\
            ║                                                                              ║\n\
            ║  Detected features:                                                          ║\n\
            ║    AVX-512F: {} (MISSING - this is the problem!)                            ║\n\
            ║    AVX2:     {}                                                             ║\n\
            ║    AVX:      {}                                                             ║\n\
            ║    FMA:      {}                                                             ║\n\
            ╚══════════════════════════════════════════════════════════════════════════════╝\n",
            avx512f, avx2, avx, fma
        );
    }

    // Also verify we're using the AVX-512 backend
    let backend = trueno::Backend::select_best();
    assert!(
        matches!(backend, trueno::Backend::AVX512 | trueno::Backend::AVX2),
        "Expected AVX512 or AVX2 backend, got {:?}",
        backend
    );

    println!("SIMD CANARY PASSED: AVX-512 detected and enabled");
}

/// Canary Test: Verify at least AVX2 is detected (fallback for non-AVX512 systems)
#[test]
#[cfg(target_arch = "x86_64")]
fn canary_avx2_minimum() {
    use std::arch::is_x86_feature_detected;

    let avx2 = is_x86_feature_detected!("avx2");
    let fma = is_x86_feature_detected!("fma");

    assert!(
        avx2 && fma,
        "SIMD CANARY FAILED: AVX2+FMA not detected! \
         This is the MINIMUM for modern SIMD. CPU: {:?}, FMA: {:?}",
        avx2,
        fma
    );

    println!("SIMD CANARY PASSED: AVX2+FMA detected");
}

/// Canary Test: Backend selection returns appropriate SIMD level
#[test]
fn canary_backend_selection_not_scalar() {
    let backend = trueno::Backend::select_best();

    // On a Threadripper, we should NEVER fall back to Scalar
    #[cfg(target_arch = "x86_64")]
    {
        assert_ne!(
            backend,
            trueno::Backend::Scalar,
            "SIMD CANARY FAILED: Backend fell back to Scalar on x86_64! \
             This indicates SIMD detection is broken or disabled."
        );
    }

    println!("BACKEND CANARY PASSED: Selected {:?}", backend);
}
