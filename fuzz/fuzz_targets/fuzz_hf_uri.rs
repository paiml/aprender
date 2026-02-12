#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz Source::parse with arbitrary strings
    // Targets: hf:// URI parsing, URL detection, local path handling
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = aprender::format::converter_types::Source::parse(s);
    }

    // Also try with hf:// prefix to exercise HuggingFace parsing path
    if data.len() > 5 {
        if let Ok(s) = std::str::from_utf8(data) {
            let uri = format!("hf://{s}");
            let _ = aprender::format::converter_types::Source::parse(&uri);
        }
    }
});
