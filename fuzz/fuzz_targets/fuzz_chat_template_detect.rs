#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz auto_detect_template with arbitrary model names
    // Targets: template matching, fallback behavior
    if let Ok(s) = std::str::from_utf8(data) {
        let template = aprender::text::chat_template::auto_detect_template(s);

        // Also exercise format_message with the detected template
        let _ = template.format_message("user", "test");

        // And format_conversation
        let messages = vec![aprender::text::chat_template::ChatMessage::user(
            "hello".to_string(),
        )];
        let _ = template.format_conversation(&messages);
    }
});
