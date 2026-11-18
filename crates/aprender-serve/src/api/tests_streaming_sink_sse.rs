//! Per-token streaming sink behaviour that `qwen3-moe-streaming-sse-v1` rests on:
//! every generated token reaches the SSE channel before the next one is produced,
//! and a vanished client stops generation. The full HTTP/SSE path over a real
//! qwen3_moe GGUF is model-gated and is NOT exercised here.

use super::openai_handlers::streaming_token_sink;
use crate::metrics::MetricsCollector;
use std::sync::Arc;

/// Each token is delivered to the channel as soon as the sink is called, in order,
/// and the sink tells the generator to continue.
#[test]
fn sink_delivers_each_token_before_the_next_is_generated() {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(8);
    let metrics = Arc::new(MetricsCollector::new());
    let mut sink = streaming_token_sink(tx, metrics.clone());

    for token in [11u32, 22, 33] {
        assert!(
            sink(token),
            "sink must say continue while the client is connected"
        );
        assert_eq!(
            rx.try_recv(),
            Ok(Ok(token)),
            "token {token} must already be queued"
        );
    }
    assert_eq!(metrics.streams_abandoned(), 0);
}

/// A dropped receiver (client gone) makes the sink return false (stop generating)
/// and records exactly one abandoned stream.
#[test]
fn sink_stops_generation_when_the_client_is_gone() {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(8);
    let metrics = Arc::new(MetricsCollector::new());
    let mut sink = streaming_token_sink(tx, metrics.clone());
    drop(rx);

    assert!(!sink(7), "sink must say stop once the receiver is dropped");
    assert_eq!(metrics.streams_abandoned(), 1);
}
