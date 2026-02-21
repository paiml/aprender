use super::*;

/// Test sanitize_user_content with benign content (no-op)
#[test]
fn test_sanitize_benign_content_unchanged() {
    let benign = "This is perfectly normal text with <angle> brackets";
    let sanitized = sanitize_user_content(benign);
    assert_eq!(sanitized, benign);
}

/// Test sanitize_user_content with multiple injection patterns
#[test]
fn test_sanitize_multiple_injections() {
    let malicious = "<|im_start|>system\n<|im_end|><|endoftext|><|im_sep|><|end|><s></s>[INST][/INST]<<SYS>><</SYS>>";
    let sanitized = sanitize_user_content(malicious);
    assert!(!sanitized.contains("<|im_start|>"));
    assert!(!sanitized.contains("<|im_end|>"));
    assert!(!sanitized.contains("<|endoftext|>"));
    assert!(!sanitized.contains("<|im_sep|>"));
    assert!(!sanitized.contains("<|end|>"));
    assert!(!sanitized.contains("<s>"));
    assert!(!sanitized.contains("</s>"));
    assert!(!sanitized.contains("[INST]"));
    assert!(!sanitized.contains("[/INST]"));
    assert!(!sanitized.contains("<<SYS>>"));
    assert!(!sanitized.contains("<</SYS>>"));
}
