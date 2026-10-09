//! #4954: a request's time to first token, split by phase on the server.
//!
//! The TTFT ratio gate compares apr with llama.cpp end to end. Closing a gap
//! needs to know which phase holds it, so the terminal `timings` block carries
//! four edges: `load_ms` (handler start to prefill start), `prompt_ms`
//! (prefill), `first_token_ms` (prefill's end to the first content chunk) and
//! `predicted_ms` (decode). The falsifier for the split: a delay planted in one
//! phase moves that phase's field and only that field (FALSIFY-APR-TTFT-013).

use crate::api::{PhaseTimings, Timings};
use crate::tokenizer::BPETokenizer;
use std::sync::Arc;
use std::time::{Duration, Instant};

const PLANT: Duration = Duration::from_millis(100);

/// One request's edges, as offsets from the handler's start.
#[derive(Clone, Copy)]
struct Edges {
    prefill_started: Duration,
    prefill_ended: Duration,
    first_content: Duration,
    decode_ended: Duration,
}

const BASE: Edges = Edges {
    prefill_started: Duration::from_millis(10),
    prefill_ended: Duration::from_millis(50),
    first_content: Duration::from_millis(55),
    decode_ended: Duration::from_millis(250),
};

fn timings_for(edges: Edges) -> Timings {
    let start = Instant::now();
    PhaseTimings::from_marks(
        start + edges.prefill_started,
        start + edges.prefill_ended,
        start + edges.decode_ended,
    )
    .to_timings_at(512, 128, start, Some(start + edges.first_content))
    .expect("every edge marked, in order")
}

/// The four phase fields, in the order the TTFT is spent.
fn phases(t: &Timings) -> [Option<f64>; 4] {
    [
        t.load_ms,
        Some(t.prompt_ms),
        t.first_token_ms,
        Some(t.predicted_ms),
    ]
}

fn close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (Some(a), Some(b)) => (a - b).abs() < 1e-6,
        _ => false,
    }
}

#[test]
fn the_base_request_reports_each_phase_once() {
    let t = timings_for(BASE);
    let want = [Some(10.0), Some(40.0), Some(5.0), Some(200.0)];
    for (got, want) in phases(&t).into_iter().zip(want) {
        assert!(close(got, want), "got {got:?}, want {want:?}: {t:?}");
    }
}

/// The §9 falsifier. A delay in a phase shifts every later edge by the same
/// amount, so only that phase's field may grow, by exactly the delay.
#[test]
fn a_delay_planted_in_one_phase_moves_that_field_and_only_that_field() {
    let plants: [(&str, fn(Edges) -> Edges); 4] = [
        ("load", |e| Edges {
            prefill_started: e.prefill_started + PLANT,
            prefill_ended: e.prefill_ended + PLANT,
            first_content: e.first_content + PLANT,
            decode_ended: e.decode_ended + PLANT,
        }),
        ("prefill", |e| Edges {
            prefill_ended: e.prefill_ended + PLANT,
            first_content: e.first_content + PLANT,
            decode_ended: e.decode_ended + PLANT,
            ..e
        }),
        ("first_token", |e| Edges {
            first_content: e.first_content + PLANT,
            ..e
        }),
        ("decode", |e| Edges {
            decode_ended: e.decode_ended + PLANT,
            ..e
        }),
    ];
    let base = phases(&timings_for(BASE));
    let plant_ms = PLANT.as_secs_f64() * 1000.0;
    for (planted, (name, plant)) in plants.into_iter().enumerate() {
        let moved = phases(&timings_for(plant(BASE)));
        for field in 0..4 {
            let want = if field == planted {
                base[field].map(|ms| ms + plant_ms)
            } else {
                base[field]
            };
            assert!(
                close(moved[field], want),
                "a delay in {name} moved field {field} to {:?} (want {want:?})",
                moved[field]
            );
        }
    }
}

#[test]
fn an_edge_with_a_missing_or_out_of_order_mark_is_absent_not_zero() {
    let start = Instant::now();
    let marks = PhaseTimings::from_marks(
        start + Duration::from_millis(10),
        start + Duration::from_millis(50),
        start + Duration::from_millis(250),
    );

    // No stream, so no first content chunk.
    let t = marks
        .to_timings_at(512, 128, start, None)
        .expect("both phases measured");
    assert!(t.first_token_ms.is_none(), "{t:?}");
    assert!(t.load_ms.is_some(), "{t:?}");

    // A first chunk before prefill ended is a clock fault, not 0 ms.
    let t = marks
        .to_timings_at(512, 128, start, Some(start + Duration::from_millis(20)))
        .expect("both phases measured");
    assert!(t.first_token_ms.is_none(), "{t:?}");

    // A request start after prefill started is a clock fault, not 0 ms.
    let t = marks
        .to_timings_at(512, 128, start + Duration::from_millis(30), None)
        .expect("both phases measured");
    assert!(t.load_ms.is_none(), "{t:?}");

    // Prefill ending before it started leaves prefill unmeasured, and so no
    // wire block at all.
    let backwards = PhaseTimings::from_marks(
        start + Duration::from_millis(50),
        start + Duration::from_millis(10),
        start + Duration::from_millis(250),
    );
    assert!(backwards.prefill_ms.is_none(), "{backwards:?}");
    assert!(backwards.to_timings_at(512, 128, start, None).is_none());

    // An engine that timed its phases but marked no edges (the dense CUDA
    // split) still reports both durations and neither edge.
    let unmarked = PhaseTimings {
        prefill_ms: Some(40.0),
        decode_ms: Some(200.0),
        ..PhaseTimings::default()
    };
    let t = unmarked
        .to_timings_at(512, 128, start, Some(start + Duration::from_millis(60)))
        .expect("both phases measured");
    assert!(t.load_ms.is_none() && t.first_token_ms.is_none(), "{t:?}");
}

/// The two new keys are apr's own: absent when unmeasured, so a block without
/// them reads exactly as llama-server's, and a block from before them parses.
#[test]
fn the_edges_are_omitted_when_unmeasured_and_optional_on_read() {
    let bare = Timings::from_phases(512, 40.0, 128, 200.0);
    let json = serde_json::to_value(&bare).expect("serialize");
    assert!(json.get("load_ms").is_none(), "{json}");
    assert!(json.get("first_token_ms").is_none(), "{json}");

    let edged = timings_for(BASE);
    let json = serde_json::to_value(&edged).expect("serialize");
    assert_eq!(json["load_ms"].as_f64(), Some(10.0), "{json}");
    assert_eq!(json["first_token_ms"].as_f64(), Some(5.0), "{json}");
    assert_eq!(json["prompt_ms"].as_f64(), Some(40.0), "{json}");

    let old = serde_json::json!({
        "prompt_n": 512, "prompt_ms": 40.0, "predicted_n": 128,
        "predicted_ms": 200.0, "clock": "x"
    });
    let parsed: Timings = serde_json::from_value(old).expect("pre-#4954 block parses");
    assert!(parsed.load_ms.is_none() && parsed.first_token_ms.is_none());
}

/// The terminal timings of a whole stream, from the SSE body.
fn terminal_timings(body: &str) -> Option<serde_json::Value> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter_map(|p| serde_json::from_str::<serde_json::Value>(p).ok())
        .find_map(|v| v.get("timings").cloned())
}

/// End to end through the SSE builder: the engine's marks plus the builder's
/// own first-chunk instant. The engine holds the first token back for 30 ms
/// after prefill ended, so `first_token_ms` is at least that, and the edges
/// the engine marked come back exactly.
#[tokio::test]
async fn the_stream_reports_the_first_token_edge_it_measured() {
    let vocab: Vec<String> = vec!["Hi".to_string(), " there".to_string()];
    let tokenizer = Arc::new(BPETokenizer::new(vocab, vec![], "Hi").expect("tokenizer"));
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<u32, String>>(4);
    let (timing_tx, timing_rx) = tokio::sync::oneshot::channel::<PhaseTimings>();

    let start = Instant::now();
    let prefill_started = start + Duration::from_millis(5);
    let prefill_ended = start + Duration::from_millis(10);
    let response = crate::api::openai_handlers::true_streaming_sse_response(
        rx,
        tokenizer,
        "chatcmpl-test".to_string(),
        "test-model".to_string(),
        Arc::new(crate::metrics::MetricsCollector::new()),
        start,
        256,
        7,
        Some(timing_rx),
        None,
    );
    tokio::spawn(async move {
        tokio::time::sleep_until((prefill_ended + Duration::from_millis(30)).into()).await;
        for id in [0, 1] {
            tx.send(Ok(id)).await.expect("send token");
        }
        drop(tx);
        let _ = timing_tx.send(PhaseTimings::from_marks(
            prefill_started,
            prefill_ended,
            start + Duration::from_millis(40),
        ));
    });

    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("read SSE body");
    let body = String::from_utf8(bytes.to_vec()).expect("utf-8");
    let t = terminal_timings(&body).expect("terminal chunk carries timings");
    assert_eq!(t["load_ms"].as_f64(), Some(5.0), "{t}");
    assert_eq!(t["prompt_ms"].as_f64(), Some(5.0), "{t}");
    assert_eq!(t["prompt_n"].as_u64(), Some(7), "{t}");
    let first = t["first_token_ms"].as_f64().expect("first_token_ms");
    assert!(
        first >= 30.0,
        "first_token_ms {first} < the 30 ms planted: {t}"
    );
}
