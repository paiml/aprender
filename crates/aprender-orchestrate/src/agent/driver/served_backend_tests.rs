//! #3719: what a run reports it ran on, from the server's per-completion `used_gpu`.

use super::*;

fn tally(answers: &[Option<bool>]) -> BackendTally {
    let t = BackendTally::default();
    for a in answers {
        t.record(*a);
    }
    t
}

#[test]
fn every_completion_on_the_gpu_ran_gpu_without_a_fallback() {
    let r = tally(&[Some(true), Some(true), Some(true)]).report(true);
    assert_eq!(r, BackendReport { requested: "gpu", ran: Some("gpu"), fell_back: Some(false) });
}

/// The #4958 shape: the GPU was asked for, a turn fell back, and serve still
/// answered 200. One CPU completion makes the whole run a fallback.
#[test]
fn one_cpu_completion_under_gpu_is_a_fallback() {
    for answers in
        [vec![Some(false)], vec![Some(true), Some(false), Some(true)], vec![None, Some(false)]]
    {
        let r = tally(&answers).report(true);
        assert_eq!(r.ran, Some("cpu"), "{answers:?}");
        assert_eq!(r.fell_back, Some(true), "{answers:?}");
    }
}

/// A completion the server did not report on leaves the run unmeasured:
/// `null`, never `"gpu"`.
#[test]
fn an_unreported_completion_is_not_measured() {
    for answers in [vec![None], vec![Some(true), None], vec![None, Some(true)]] {
        let r = tally(&answers).report(true);
        assert_eq!((r.ran, r.fell_back), (None, None), "{answers:?}");
    }
}

#[test]
fn a_run_with_no_completion_is_not_measured() {
    let r = tally(&[]).report(true);
    assert_eq!(r, BackendReport { requested: "gpu", ran: None, fell_back: None });
}

#[test]
fn the_cpu_asked_for_and_given_is_not_a_fallback() {
    let r = tally(&[Some(false), Some(false)]).report(false);
    assert_eq!(r, BackendReport { requested: "cpu", ran: Some("cpu"), fell_back: Some(false) });
    // The GPU given when the CPU was asked for is not a fallback either (the
    // `apr chat --json` rule); `ran` says what happened.
    let r = tally(&[Some(true)]).report(false);
    assert_eq!(r, BackendReport { requested: "cpu", ran: Some("gpu"), fell_back: Some(false) });
}

#[test]
fn used_gpu_is_read_only_as_a_boolean() {
    use serde_json::json;
    assert_eq!(served_used_gpu(&json!({"used_gpu": true})), Some(true));
    assert_eq!(served_used_gpu(&json!({"used_gpu": false})), Some(false));
    assert_eq!(served_used_gpu(&json!({"choices": []})), None, "absent");
    assert_eq!(served_used_gpu(&json!({"used_gpu": null})), None);
    assert_eq!(served_used_gpu(&json!({"used_gpu": "true"})), None, "a string is not a report");
    assert_eq!(served_used_gpu(&json!({"used_gpu": 1})), None);
}

#[test]
fn the_json_object_has_the_apr_chat_shape_and_null_for_unmeasured() {
    let measured = BackendReport { requested: "gpu", ran: Some("cpu"), fell_back: Some(true) };
    assert_eq!(
        measured.to_json(),
        serde_json::json!({"requested": "gpu", "ran": "cpu", "fell_back": true})
    );
    let unmeasured = BackendReport { requested: "gpu", ran: None, fell_back: None };
    let j = unmeasured.to_json();
    assert!(j["ran"].is_null() && j["fell_back"].is_null(), "{j}");
    assert_eq!(j.as_object().map(serde_json::Map::len), Some(3), "the keys are present: {j}");
}
