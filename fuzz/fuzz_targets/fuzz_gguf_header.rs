#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Fuzz GGUF header parsing with arbitrary bytes
    // Targets: magic detection, version parsing, tensor count parsing
    if data.is_empty() {
        return;
    }

    // Write to temp file and attempt format detection
    let dir = std::env::temp_dir();
    let path = dir.join("fuzz_gguf_header.bin");
    if let Ok(mut f) = std::fs::File::create(&path) {
        let _ = f.write_all(data);
        let _ = f.flush();
        let _ = aprender::format::rosetta::FormatType::from_magic(&path);
    }
    let _ = std::fs::remove_file(&path);
});
