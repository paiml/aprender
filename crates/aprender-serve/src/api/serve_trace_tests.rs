//! apr-trace-v1 falsifiers (APR-OBS-001 OBS-09).

use super::*;

fn traced(events: &[(TraceStep, usize, u64)]) -> InferenceTracer {
    let mut t = tracer_for(Some("step")).expect("step is traced");
    for &(step, iteration, us) in events {
        t.record_timed(step, iteration, None, us);
    }
    t
}

fn serve<'a>(
    level: &'a str,
    events: &'a [TraceEvent],
    layers: Option<&'a [LayerTiming]>,
    wall_us: u64,
) -> ServeTrace<'a> {
    ServeTrace {
        level: Some(level),
        events,
        layers,
        wall_us,
        prompt_tokens: 4,
        completion_tokens: 3,
        num_layers: 2,
    }
}

fn layer(total_us: u64, calls: u32) -> LayerTiming {
    LayerTiming {
        total_us,
        calls,
        kind: "deltanet",
    }
}

/// FALSIFY-TRACE-001: a trace whose tracer did not run is never `Measured`.
#[test]
fn falsify_trace_001_no_events_is_not_measured() {
    let (_, step, _) = traces_for(&serve("step", &[], None, 10_000));
    let step = step.expect("the header still gets its wall-clock row");
    assert_eq!(step.provenance, TraceProvenance::WallClockTotal);

    // A CPU turn asked for layers timed none.
    let (_, _, layers) = traces_for(&serve("layer", &[], None, 10_000));
    assert_eq!(
        layers.expect("wall-clock row").provenance,
        TraceProvenance::WallClockTotal
    );

    // A tally whose every layer ran zero forwards measured nothing either.
    let idle = [layer(0, 0), layer(0, 0)];
    let (_, _, layers) = traces_for(&serve("layer", &[], Some(&idle), 10_000));
    assert_eq!(
        layers.expect("wall-clock row").provenance,
        TraceProvenance::WallClockTotal
    );
}

/// FALSIFY-TRACE-001, the tracer half: a tracer that is not tracing records
/// nothing, so no reply built from it can claim a measurement.
#[test]
fn falsify_trace_001_disabled_tracer_records_nothing() {
    let mut t = InferenceTracer::disabled();
    t.record_timed(TraceStep::TransformerBlock, 0, None, 5);
    assert!(t.events().is_empty());
    assert!(tracer_for(Some("brick")).is_none());
    assert!(tracer_for(None).is_none());
}

/// FALSIFY-TRACE-002: per-layer time that sums past the request's wall clock
/// is not a measurement of that request.
#[test]
fn falsify_trace_002_layer_sum_over_wall_is_not_measured() {
    let over = [layer(600, 3), layer(500, 3)];
    let (_, _, l) = traces_for(&serve("layer", &[], Some(&over), 1_000));
    assert_eq!(
        l.expect("wall-clock row").provenance,
        TraceProvenance::WallClockTotal
    );

    let t = traced(&[
        (TraceStep::TransformerBlock, 0, 900),
        (TraceStep::Decode, 0, 200),
    ]);
    let (_, s, _) = traces_for(&serve("step", t.events(), None, 1_000));
    assert_eq!(
        s.expect("wall-clock row").provenance,
        TraceProvenance::WallClockTotal
    );
}

#[test]
fn measured_layers_are_reported_per_layer_within_wall() {
    let tally = [
        layer(300, 3),
        LayerTiming {
            total_us: 200,
            calls: 3,
            kind: "attention",
        },
    ];
    let (b, s, l) = traces_for(&serve("layer", &[], Some(&tally), 1_000));
    assert!(b.is_none() && s.is_none());
    let l = l.expect("layer trace");
    assert_eq!(l.provenance, TraceProvenance::Measured);
    assert_eq!(l.total_time_us, 1_000);
    let names: Vec<_> = l.breakdown.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["layer[0] deltanet", "layer[1] attention"]);
    let sum: u64 = l.breakdown.iter().map(|o| o.time_us).sum();
    assert!(sum <= l.total_time_us);
}

#[test]
fn measured_steps_split_prefill_decode_emit() {
    let t = traced(&[
        (TraceStep::TransformerBlock, 0, 400),
        (TraceStep::Decode, 0, 5),
        (TraceStep::TransformerBlock, 1, 100),
        (TraceStep::Decode, 1, 5),
        (TraceStep::TransformerBlock, 2, 110),
        (TraceStep::Decode, 2, 5),
    ]);
    let (_, s, _) = traces_for(&serve("step", t.events(), None, 1_000));
    let s = s.expect("step trace");
    assert_eq!(s.provenance, TraceProvenance::Measured);
    assert_eq!(s.operations, 6);
    let rows: Vec<_> = s
        .breakdown
        .iter()
        .map(|o| (o.name.as_str(), o.time_us))
        .collect();
    assert_eq!(rows, [("prefill", 400), ("decode", 210), ("emit", 15)]);
}

/// FALSIFY-TRACE-002, the boundary: a sum EXACTLY at the wall clock is still a
/// measurement. Kills `total > wall_us` -> `>=` in both `step_trace` and
/// `layer_trace`; one unit over is the refusal side of the same edge.
#[test]
fn falsify_trace_002_sum_at_wall_is_measured_one_over_is_not() {
    let case = |sum_extra: u64| {
        let at = [layer(600, 3), layer(400 + sum_extra, 3)];
        let (_, _, l) = traces_for(&serve("layer", &[], Some(&at), 1_000));
        let t = traced(&[
            (TraceStep::TransformerBlock, 0, 800),
            (TraceStep::TransformerBlock, 1, 150),
            (TraceStep::Decode, 1, 50 + sum_extra),
        ]);
        let (_, s, _) = traces_for(&serve("step", t.events(), None, 1_000));
        (
            l.expect("layer row").provenance,
            s.expect("step row").provenance,
        )
    };
    use TraceProvenance::{Measured, WallClockTotal};
    assert_eq!(case(0), (Measured, Measured), "sum == wall");
    assert_eq!(case(1), (WallClockTotal, WallClockTotal), "sum == wall + 1");
}

/// FALSIFY-TRACE-005 (TR-09, CRUX F-07): chrome is a format of the apr-trace-v1
/// document. Its rows are exactly the document's rows (names and durations, in
/// order), every duration is a non-negative integer, and the rows end inside the
/// `request` span — for a measured layer trace and an unmeasured one alike.
#[test]
fn falsify_trace_005_chrome_is_a_format_of_the_document() {
    let layers = [layer(300, 3), layer(500, 3)];
    let docs = [
        apr_trace(&serve("layer", &[], Some(&layers), 1_000)).expect("measured"),
        apr_trace(&serve("layer", &[], None, 1_000)).expect("wall clock"),
    ];
    for doc in &docs {
        let chrome = chrome_trace(doc);
        let events = chrome["traceEvents"].as_array().expect("traceEvents");
        let span = events[0]["dur"].as_u64().expect("request dur");
        assert_eq!(span, doc.total_time_us);
        let rows: Vec<(String, u64)> = events[1..]
            .iter()
            .map(|e| {
                let ts = e["ts"].as_u64().expect("ts is a non-negative integer");
                let dur = e["dur"].as_u64().expect("dur is a non-negative integer");
                assert!(ts + dur <= span, "row {e} ends past the request span");
                (e["name"].as_str().expect("name").to_string(), dur)
            })
            .collect();
        let want: Vec<(String, u64)> = doc
            .breakdown
            .iter()
            .map(|o| (o.name.clone(), o.time_us))
            .collect();
        assert_eq!(rows, want);
        assert_eq!(
            chrome["metadata"]["provenance"],
            serde_json::to_value(doc.provenance).expect("provenance")
        );
    }
}

/// `apr_trace` is the one level's document `traces_for` answers with.
#[test]
fn apr_trace_is_the_requested_levels_document() {
    assert!(apr_trace(&serve("none", &[], None, 10)).is_none());
    for level in ["brick", "step", "layer"] {
        assert_eq!(
            apr_trace(&serve(level, &[], None, 10))
                .expect("document")
                .level,
            level
        );
    }
}
