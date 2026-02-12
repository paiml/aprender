#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz ShardedIndex::parse with arbitrary strings
    // Targets: JSON parsing, weight_map extraction, shard file validation
    if let Ok(s) = std::str::from_utf8(data) {
        let _ = aprender::format::converter_types::ShardedIndex::parse(s);
    }
});
