
#[test]
fn test_batch_tokenize_response_serialize_more_cov() {
    let resp = BatchTokenizeResponse {
        results: vec![
            TokenizeResponse {
                token_ids: vec![1, 2],
                num_tokens: 2,
            },
            TokenizeResponse {
                token_ids: vec![3, 4, 5],
                num_tokens: 3,
            },
        ],
    };
    let json = serde_json::to_string(&resp).expect("serialize");
    assert!(json.contains("[1,2]"));
    assert!(json.contains("num_tokens"));
}

#[test]
fn test_stream_token_event_serialize_more_cov() {
    let event = StreamTokenEvent {
        token_id: 42,
        text: "hello".to_string(),
    };
    let json = serde_json::to_string(&event).expect("serialize");
    assert!(json.contains("42"));
    assert!(json.contains("hello"));
}

#[test]
fn test_stream_done_event_serialize_more_cov() {
    let event = StreamDoneEvent { num_generated: 15 };
    let json = serde_json::to_string(&event).expect("serialize");
    assert!(json.contains("15"));
}
