#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Fuzz APR format reading with arbitrary bytes
    // Targets: header parsing, metadata extraction, version detection
    if data.is_empty() {
        return;
    }

    let dir = std::env::temp_dir();
    let path = dir.join("fuzz_apr_metadata.apr");
    if let Ok(mut f) = std::fs::File::create(&path) {
        let _ = f.write_all(data);
        let _ = f.flush();

        // Try reading as APR file — should return error, never panic
        let _ = aprender::format::rosetta::FormatType::from_magic(&path);
    }
    let _ = std::fs::remove_file(&path);
});
