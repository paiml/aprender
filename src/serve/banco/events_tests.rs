//! EventBus and WebSocket endpoint tests.

use super::events::{BancoEvent, EventBus};

// ============================================================================
// EventBus unit tests
// ============================================================================

#[test]
#[allow(non_snake_case)]
fn test_EVENT_001_emit_and_receive() {
    let bus = EventBus::new(16);
    let mut rx = bus.subscribe();

    bus.emit(&BancoEvent::ModelUnloaded);

    let msg = rx.try_recv().expect("should receive event");
    assert!(msg.contains("model_unloaded"), "got: {msg}");
}

#[test]
#[allow(non_snake_case)]
fn test_EVENT_002_multiple_subscribers() {
    let bus = EventBus::new(16);
    let mut rx1 = bus.subscribe();
    let mut rx2 = bus.subscribe();

    bus.emit(&BancoEvent::FileUploaded {
        file_id: "f-1".to_string(),
        name: "test.txt".to_string(),
    });

    let msg1 = rx1.try_recv().expect("rx1");
    let msg2 = rx2.try_recv().expect("rx2");
    assert_eq!(msg1, msg2);
    assert!(msg1.contains("file_uploaded"));
}

#[test]
#[allow(non_snake_case)]
fn test_EVENT_003_no_subscriber_ok() {
    let bus = EventBus::new(16);
    // Emit with no subscribers — should not panic
    bus.emit(&BancoEvent::SystemEvent { message: "test".to_string() });
    assert_eq!(bus.subscriber_count(), 0);
}

#[test]
#[allow(non_snake_case)]
fn test_EVENT_004_subscriber_count() {
    let bus = EventBus::new(16);
    assert_eq!(bus.subscriber_count(), 0);
    let _rx1 = bus.subscribe();
    assert_eq!(bus.subscriber_count(), 1);
    let _rx2 = bus.subscribe();
    assert_eq!(bus.subscriber_count(), 2);
}

#[test]
#[allow(non_snake_case)]
fn test_EVENT_005_event_serialization() {
    let event = BancoEvent::TrainingMetric { run_id: "run-1".to_string(), step: 42, loss: 0.5 };
    let json = serde_json::to_value(&event).expect("serialize");
    assert_eq!(json["type"], "training_metric");
    assert_eq!(json["data"]["step"], 42);
    assert!((json["data"]["loss"].as_f64().expect("loss") - 0.5).abs() < f64::EPSILON);
}

#[test]
#[allow(non_snake_case)]
fn test_EVENT_006_all_event_types_serialize() {
    let events = vec![
        BancoEvent::ModelLoaded { model_id: "m".to_string(), format: "gguf".to_string() },
        BancoEvent::ModelUnloaded,
        BancoEvent::TrainingStarted { run_id: "r".to_string(), method: "lora".to_string() },
        BancoEvent::TrainingMetric { run_id: "r".to_string(), step: 1, loss: 2.0 },
        BancoEvent::TrainingComplete { run_id: "r".to_string() },
        BancoEvent::FileUploaded { file_id: "f".to_string(), name: "a.txt".to_string() },
        BancoEvent::RagIndexed { doc_count: 1, chunk_count: 5 },
        BancoEvent::MergeComplete { merge_id: "m".to_string(), strategy: "slerp".to_string() },
        BancoEvent::SystemEvent { message: "test".to_string() },
    ];
    for event in &events {
        let json = serde_json::to_string(event).expect("serialize");
        assert!(json.contains("\"type\""), "missing type field: {json}");
    }
}

#[test]
#[allow(non_snake_case)]
fn test_EVENT_007_default_capacity() {
    let bus = EventBus::default();
    assert_eq!(bus.subscriber_count(), 0);
    // Default capacity is 256 — emit many without subscriber
    for i in 0..300 {
        bus.emit(&BancoEvent::SystemEvent { message: format!("event {i}") });
    }
}

// ============================================================================
// WebSocket endpoint test (upgrade request)
// ============================================================================

#[tokio::test]
#[allow(non_snake_case)]
async fn test_WS_HDL_001_ws_endpoint_exists() {
    use axum::{body::Body, http::Request};
    use tower::ServiceExt;

    let app = super::router::create_banco_router(super::state::BancoStateInner::with_defaults());
    // Regular GET without WebSocket upgrade headers should return 400 (bad request)
    let response = app
        .oneshot(Request::get("/api/v1/ws").body(Body::empty()).expect("req"))
        .await
        .expect("resp");
    // Without upgrade headers, axum returns the handler error
    // The route exists — that's what we're testing
    let status = response.status().as_u16();
    assert!(status != 404, "WebSocket route should exist, got 404");
}
