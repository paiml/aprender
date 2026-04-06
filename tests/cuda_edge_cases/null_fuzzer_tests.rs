// ============================================================================
// F1: Null Pointer Sentinel Fuzzer -- Device Memory Safety
// ============================================================================

use trueno_cuda_edge::null_fuzzer::{
    InjectionStrategy, NonNullDevicePtr, NullFuzzerConfig, NullSentinelFuzzer,
};

/// Verify NonNullDevicePtr enforces non-null constraint at construction.
/// Critical for trueno's GPU memory allocation wrappers.
#[test]
fn device_ptr_construction_safety() {
    // Null address must be rejected
    let null_result = NonNullDevicePtr::<f32>::new(0);
    assert!(null_result.is_err());

    // Valid GPU address should succeed
    let valid_result = NonNullDevicePtr::<f32>::new(0x7f00_0000_0000);
    assert!(valid_result.is_ok());

    // Verify address is preserved
    let ptr = valid_result.unwrap();
    assert_eq!(ptr.addr(), 0x7f00_0000_0000);
}

/// Test type-level safety for different element types.
#[test]
fn device_ptr_type_safety() {
    // f32 tensor buffer
    let f32_ptr = NonNullDevicePtr::<f32>::new(0x1000).unwrap();
    assert_eq!(f32_ptr.addr(), 0x1000);

    // f16 tensor buffer (different type parameter)
    let f16_ptr = NonNullDevicePtr::<u16>::new(0x2000).unwrap();
    assert_eq!(f16_ptr.addr(), 0x2000);

    // u8 quantized buffer
    let u8_ptr = NonNullDevicePtr::<u8>::new(0x3000).unwrap();
    assert_eq!(u8_ptr.addr(), 0x3000);
}

/// Test injection strategies for GPU kernel fuzzing.
#[test]
fn injection_strategies_for_kernel_calls() {
    // Periodic: inject every N kernel launches
    let periodic = InjectionStrategy::Periodic { interval: 100 };
    assert!(periodic.should_inject(0));
    assert!(periodic.should_inject(100));
    assert!(!periodic.should_inject(50));

    // Size threshold: inject for large allocations
    let size_based = InjectionStrategy::SizeThreshold {
        threshold_bytes: 1024 * 1024 * 1024, // 1 GB
    };
    // Requires context, returns false without it
    assert!(!size_based.should_inject(0));

    // Probabilistic: deterministic for reproducibility
    let prob = InjectionStrategy::Probabilistic { probability: 0.25 };
    // 25% = inject when (idx % 100) < 25
    assert!(prob.should_inject(0));
    assert!(prob.should_inject(24));
    assert!(!prob.should_inject(25));
}

/// Test fuzzer state machine for GPU call sequences.
#[test]
fn fuzzer_call_sequence() {
    let config = NullFuzzerConfig {
        strategy: InjectionStrategy::Periodic { interval: 5 },
        total_calls: 100,
        fail_fast: false,
    };

    let mut fuzzer = NullSentinelFuzzer::new(config);

    // Track injection pattern
    let mut injections = Vec::new();
    for _ in 0..20 {
        injections.push(fuzzer.next_call());
    }

    // Should inject at 0, 5, 10, 15
    assert!(injections[0]);
    assert!(!injections[1]);
    assert!(injections[5]);
    assert!(injections[10]);
    assert!(injections[15]);
}
