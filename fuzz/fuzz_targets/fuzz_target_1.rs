#![no_main]

use libfuzzer_sys::fuzz_target;
use aprender::chaos::ChaosConfig;
use std::time::Duration;

fuzz_target!(|data: &[u8]| {
    // Test ChaosConfig builder pattern with arbitrary inputs
    // Based on renacer's chaos testing approach

    if data.len() >= 17 {
        // Extract fuzzing parameters from data
        let memory_limit = usize::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]);

        let cpu_limit_raw = f64::from_le_bytes([
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]);

        let signal_injection = data[16] % 2 == 0;

        // Test that builder pattern handles all inputs gracefully
        let config = ChaosConfig::new()
            .with_memory_limit(memory_limit)
            .with_cpu_limit(cpu_limit_raw)
            .with_signal_injection(signal_injection)
            .build();

        // Invariants that must always hold
        assert!(config.cpu_limit >= 0.0 && config.cpu_limit <= 1.0);
        assert_eq!(config.memory_limit, memory_limit);
        assert_eq!(config.signal_injection, signal_injection);

        // Test cloning
        let cloned = config.clone();
        assert_eq!(config, cloned);
    }

    // Test preset configurations
    let gentle = ChaosConfig::gentle();
    assert!(gentle.memory_limit > 0);
    assert!(gentle.cpu_limit > 0.0 && gentle.cpu_limit <= 1.0);

    let aggressive = ChaosConfig::aggressive();
    assert!(aggressive.memory_limit > 0);
    assert!(aggressive.cpu_limit > 0.0 && aggressive.cpu_limit <= 1.0);
});
