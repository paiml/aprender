
#[test]
fn test_chat_chunk_choice_streaming_content() {
    let choice = ChatChunkChoice {
        index: 0,
        delta: ChatDelta {
            role: None,
            content: Some("partial".to_string()), reasoning_content: None },
        finish_reason: None,
    };

    let json = serde_json::to_string(&choice).expect("serialize");
    assert!(json.contains("partial"));
    // finish_reason may be present as null when serialized
    assert!(json.contains("delta"));
}
