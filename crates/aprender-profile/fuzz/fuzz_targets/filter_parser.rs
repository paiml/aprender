#![no_main]

use libfuzzer_sys::fuzz_target;
use renacer::filter::SyscallFilter;

fuzz_target!(|data: &[u8]| {
    // Convert arbitrary bytes to UTF-8 string (lossy conversion)
    if let Ok(input) = std::str::from_utf8(data) {
        // Attempt to parse the filter expression
        // This should not panic regardless of input
        let _ = SyscallFilter::from_expr(input);
    }
});
