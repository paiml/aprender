#![no_main]

use libfuzzer_sys::fuzz_target;
use std::io::Write;

fuzz_target!(|data: &[u8]| {
    // Fuzz format detection with arbitrary bytes
    // Targets: from_magic(), from_extension() with various extensions
    if data.is_empty() {
        return;
    }

    let dir = std::env::temp_dir();

    // Test with various extensions
    for ext in &["gguf", "safetensors", "apr", "bin", "pt", "onnx"] {
        let path = dir.join(format!("fuzz_format.{ext}"));
        if let Ok(mut f) = std::fs::File::create(&path) {
            let _ = f.write_all(data);
            let _ = f.flush();
            let _ = aprender::format::rosetta::FormatType::from_magic(&path);
        }
        let _ = std::fs::remove_file(&path);
    }

    // Also test from_extension with arbitrary path strings
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = aprender::format::rosetta::FormatType::from_extension(
            std::path::Path::new(s),
        );
    }
});
