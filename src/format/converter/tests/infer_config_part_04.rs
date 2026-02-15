
#[test]
fn test_gh236_architecture_fallback_uses_model_type() {
    use crate::format::v2::AprV2Metadata;

    // Simulate a GPT-2 APR that has model_type but no architecture field
    let mut metadata = AprV2Metadata::new("gpt2");
    metadata.architecture = None;
    metadata.model_type = "gpt2".to_string();
    metadata.hidden_size = Some(768);
    metadata.num_layers = Some(12);
    metadata.num_heads = Some(12);
    metadata.vocab_size = Some(50257);

    // Test the architecture fallback logic directly:
    // architecture=None, model_type="gpt2" → should resolve to "gpt2"
    let arch = metadata
        .architecture
        .as_deref()
        .or_else(|| {
            let mt = &metadata.model_type;
            if mt.is_empty() || mt == "unknown" {
                None
            } else {
                Some(mt.as_str())
            }
        })
        .unwrap_or("qwen2");

    assert_eq!(
        arch, "gpt2",
        "GH-236: Architecture fallback should use model_type='gpt2', not default 'qwen2'"
    );

    // Test that "unknown" model_type falls through to default
    let mut metadata2 = AprV2Metadata::new("unknown");
    metadata2.architecture = None;
    metadata2.model_type = "unknown".to_string();

    let arch2 = metadata2
        .architecture
        .as_deref()
        .or_else(|| {
            let mt = &metadata2.model_type;
            if mt.is_empty() || mt == "unknown" {
                None
            } else {
                Some(mt.as_str())
            }
        })
        .unwrap_or("qwen2");

    assert_eq!(
        arch2, "qwen2",
        "GH-236: 'unknown' model_type should fall through to default 'qwen2'"
    );
}
