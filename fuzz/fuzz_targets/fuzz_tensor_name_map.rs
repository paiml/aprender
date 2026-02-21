#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz layout contract with arbitrary tensor names and shapes
    // Targets: enforce_import_contract, name normalization
    if let Ok(s) = std::str::from_utf8(data) {
        // Test with 2D shape
        let _ =
            aprender::format::layout_contract::enforce_import_contract(s, &[128, 256], 256, 128);

        // Test with 1D shape
        let _ = aprender::format::layout_contract::enforce_import_contract(s, &[512], 0, 512);
    }

    // Test with arbitrary dimensions from data bytes
    if data.len() >= 4 {
        let d1 = u16::from_le_bytes([data[0], data[1]]) as usize;
        let d2 = u16::from_le_bytes([data[2], data[3]]) as usize;
        if d1 > 0 && d2 > 0 {
            let _ = aprender::format::layout_contract::enforce_import_contract(
                "test.weight",
                &[d1, d2],
                d2,
                d1,
            );
        }
    }
});
