
#[test]
fn test_sliding_window_attention_debug_clone() {
    let swa = SlidingWindowAttention::new(64, 1024).expect("test");
    let debug = format!("{:?}", swa);
    assert!(debug.contains("SlidingWindowAttention"));

    let cloned = swa.clone();
    assert_eq!(cloned.window_size(), swa.window_size());
}
