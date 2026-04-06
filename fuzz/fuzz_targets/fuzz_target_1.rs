#![no_main]

//! Fuzz testing target for certeza
//!
//! Based on renacer's fuzz testing approach (Sprint 29).
//! Tests the TruenoVec API for crashes, panics, and memory safety issues.

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Test TruenoVec operations with fuzzed input
    if data.is_empty() {
        return;
    }

    let mut vec = certeza::TruenoVec::new();

    // Interpret fuzzed data as operations
    for chunk in data.chunks(2) {
        if chunk.is_empty() {
            continue;
        }

        let op = chunk[0] % 5; // 5 operations
        let value = if chunk.len() > 1 { chunk[1] } else { 0 };

        match op {
            0 => {
                // Push operation
                vec.push(value as i32);
            }
            1 => {
                // Pop operation
                let _ = vec.pop();
            }
            2 => {
                // Get operation
                let idx = value as usize;
                if idx < vec.len() {
                    let _ = vec.get(idx);
                }
            }
            3 => {
                // Clear operation
                vec.clear();
            }
            4 => {
                // Iterator operation
                let sum: i32 = vec.iter().sum();
                let _ = sum;
            }
            _ => unreachable!(),
        }
    }

    // Verify invariants hold after fuzzing
    assert_eq!(vec.len(), vec.iter().count());
});
