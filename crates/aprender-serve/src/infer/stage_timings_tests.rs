//! PMAT-3598 row 1 — the arithmetic and the planted delay, both directions.

use super::*;

/// The env var is process-global, so the cases that set it are serialized behind one lock rather
/// than racing each other through `cargo test`'s threads.
static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn with_delay<T>(spec: Option<&str>, f: impl FnOnce() -> T) -> T {
    let _g = ENV_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    match spec {
        Some(s) => std::env::set_var("APR_STAGE_DELAY_MS", s),
        None => std::env::remove_var("APR_STAGE_DELAY_MS"),
    }
    let out = f();
    std::env::remove_var("APR_STAGE_DELAY_MS");
    out
}

#[test]
fn an_unmeasured_stage_contributes_nothing_and_is_not_zero() {
    // The whole point: `None` and `Some(0.0)` must not be the same number downstream.
    let mut t = StageTimings {
        load_ms: Some(100.0),
        prefill_ms: Some(50.0),
        ..StageTimings::default()
    };
    assert_eq!(t.measured_sum_ms(), 150.0);
    assert_eq!(t.measured(), ["load_ms", "prefill_ms"]);
    t.close(200.0);
    assert_eq!(t.unattributed_ms, Some(50.0));

    let mut z = StageTimings {
        load_ms: Some(100.0),
        prefill_ms: Some(50.0),
        h2d_ms: Some(0.0),
        ..StageTimings::default()
    };
    z.close(200.0);
    assert_eq!(
        z.measured_sum_ms(),
        150.0,
        "a measured zero still sums to 0"
    );
    assert_eq!(
        z.measured(),
        ["load_ms", "h2d_ms", "prefill_ms"],
        "a measured zero is REPORTED as measured; an absent one is not"
    );
}

#[test]
fn the_books_close_exactly_so_a_tolerance_is_about_attribution_not_arithmetic() {
    for wall in [0.0_f64, 1.0, 14_060.0, 1e6] {
        let mut t = StageTimings {
            load_ms: Some(wall * 0.3),
            h2d_ms: Some(wall * 0.2),
            decode_ms: Some(wall * 0.1),
            ..StageTimings::default()
        };
        t.close(wall);
        let diff = (t.measured_sum_ms() + t.unattributed_ms.expect("closed")
            - t.wall_ms.expect("closed"))
        .abs();
        assert!(
            diff < 1e-9,
            "books did not close at wall={wall}: off by {diff}"
        );
    }
}

#[test]
fn a_run_that_measured_nothing_attributes_the_whole_wall_clock_to_nobody() {
    // The honest degenerate case: an uninstrumented backend says "14 s, none of it attributed",
    // which is a finding. It must not say "14 s, all of it decode".
    let mut t = StageTimings::default();
    t.close(14_060.0);
    assert_eq!(t.unattributed_ms, Some(14_060.0));
    assert!(t.measured().is_empty());
}

#[test]
fn the_guard_halves_are_inside_validate_and_never_double_counted() {
    // #3604 needs the split; the books must still close. If `validate_ref_ms`/`validate_probe_ms`
    // were summed alongside `validate_ms` the guard would be counted twice and `unattributed_ms`
    // would go negative — a residual that lies in the other direction.
    let mut t = StageTimings {
        load_ms: Some(100.0),
        validate_ms: Some(900.0),
        validate_ref_ms: Some(800.0),
        validate_probe_ms: Some(100.0),
        ..StageTimings::default()
    };
    assert_eq!(
        t.measured_sum_ms(),
        1000.0,
        "the halves must not be re-added"
    );
    t.close(1200.0);
    assert_eq!(t.unattributed_ms, Some(200.0));
    assert!(
        t.unattributed_ms.expect("closed") >= 0.0,
        "double counting drives the residual negative"
    );
    assert!(t.measured().contains(&"validate_ref_ms"));
}

#[test]
fn a_planted_delay_lands_in_its_own_stage_and_in_no_other() {
    // THE falsifier of this row. Plant 200 ms in h2d; h2d must move and the others must not.
    let plant = 200.0;
    let (_, h2d) = with_delay(Some("h2d:200"), || timed("h2d", || {}));
    let (_, load) = with_delay(Some("h2d:200"), || timed("load", || {}));
    let (_, prefill) = with_delay(Some("h2d:200"), || timed("prefill", || {}));

    assert!(
        h2d >= plant,
        "the planted stage did not move: h2d {h2d} ms < {plant} ms"
    );
    for (name, v) in [("load", load), ("prefill", prefill)] {
        assert!(
            v < plant / 2.0,
            "a delay planted in h2d moved {name} too ({v} ms) — the row cannot localise"
        );
    }
}

#[test]
fn no_plant_means_no_delay_the_control_for_the_control() {
    // Without this, a `timed` that always slept would pass the case above.
    let (_, h2d) = with_delay(None, || timed("h2d", || {}));
    assert!(h2d < 100.0, "unplanted stage slept anyway: {h2d} ms");
}

#[test]
fn a_malformed_or_foreign_plant_never_fails_a_run() {
    for spec in ["", "h2d", "h2d:", "h2d:abc", "nosuchstage:500", ":500"] {
        let d = with_delay(Some(spec), || planted_delay("h2d"));
        assert!(d.is_none(), "spec {spec:?} produced a delay");
    }
    // And a well-formed plant for a DIFFERENT stage is not this stage's delay.
    assert!(with_delay(Some("decode:500"), || planted_delay("h2d")).is_none());
    assert!(with_delay(Some("decode:500"), || planted_delay("decode")).is_some());
}

/// #3598 quorum round 1, lane 1: a report that was never `close`d must say so, not print zeros.
///
/// Uninstrumented generate paths return `InferenceResult::default()`, which carries a
/// `StageTimings::default()`. With `wall_ms: f64` that reported `wall_ms: 0.0` and
/// `unattributed_ms: 0.0` — a run that took no time and attributed all of it — which is the
/// absent-is-not-zero rule broken in the two fields that close the books.
#[test]
fn an_unclosed_report_has_no_wall_clock_and_no_residual() {
    let t = StageTimings::default();
    assert!(!t.is_closed(), "a default report was never closed");
    assert_eq!(
        t.wall_ms, None,
        "absent, not 0.0 — nothing measured the wall clock"
    );
    assert_eq!(
        t.unattributed_ms, None,
        "no wall clock means no residual to state"
    );

    // And the control: once closed, both are present, so `None` means UNCLOSED and not
    // "closed with nothing in it".
    let mut c = StageTimings::default();
    c.close(1234.0);
    assert!(c.is_closed());
    assert_eq!(c.wall_ms, Some(1234.0));
    assert_eq!(
        c.unattributed_ms,
        Some(1234.0),
        "no stages measured ⇒ all of it unattributed"
    );
}
