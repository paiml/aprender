//! Animation timing verification.
//!
//! Compares declared animation events against actual timing data
//! (from render reports, metadata, or frame analysis).

use super::types::{
    AnimationEvent, AnimationReport, AnimationTimeline, AnimationVerdict, EventResult,
};

/// Actual observed animation timing from render analysis.
#[derive(Clone, Debug)]
pub struct ObservedEvent {
    /// Event name (must match declaration)
    pub name: String,
    /// Observed time in seconds
    pub time_secs: f64,
}

/// Verify animation timeline against observed events.
///
/// For each declared event, finds the matching observed event by name
/// and computes the timing delta.
#[must_use]
pub fn verify_timeline(
    timeline: &AnimationTimeline,
    observed: &[ObservedEvent],
    tolerance_ms: f64,
) -> AnimationReport {
    if !timeline.has_events() {
        return AnimationReport {
            video_id: timeline.video_id.clone(),
            verdict: AnimationVerdict::NoEvents,
            events: Vec::new(),
            total_events: 0,
            verified_events: 0,
            max_delta_ms: 0.0,
            mean_delta_ms: 0.0,
        };
    }

    let mut results = Vec::new();
    let mut verified = 0;
    let mut max_delta: f64 = 0.0;
    let mut delta_sum: f64 = 0.0;
    let mut delta_count: usize = 0;

    for event in &timeline.events {
        let observed_event = find_observed_by_name(observed, &event.name);
        let result = match observed_event {
            Some(obs) => {
                let delta_ms = (obs.time_secs - event.expected_secs) * 1000.0;
                let abs_delta = delta_ms.abs();
                let passed = abs_delta <= tolerance_ms;
                if passed {
                    verified += 1;
                }
                if abs_delta > max_delta {
                    max_delta = abs_delta;
                }
                delta_sum += abs_delta;
                delta_count += 1;
                EventResult {
                    name: event.name.clone(),
                    event_type: event.event_type.clone(),
                    expected_secs: event.expected_secs,
                    actual_secs: Some(obs.time_secs),
                    delta_ms: Some(delta_ms),
                    passed,
                }
            }
            None => EventResult {
                name: event.name.clone(),
                event_type: event.event_type.clone(),
                expected_secs: event.expected_secs,
                actual_secs: None,
                delta_ms: None,
                passed: false,
            },
        };
        results.push(result);
    }

    let mean_delta = if delta_count > 0 {
        delta_sum / delta_count as f64
    } else {
        0.0
    };

    let verdict = if verified == timeline.events.len() {
        AnimationVerdict::Pass
    } else {
        AnimationVerdict::Fail
    };

    AnimationReport {
        video_id: timeline.video_id.clone(),
        verdict,
        events: results,
        total_events: timeline.events.len(),
        verified_events: verified,
        max_delta_ms: max_delta,
        mean_delta_ms: mean_delta,
    }
}

/// Verify animation events against expected timing from a flat list.
///
/// Simplified API when you have expected events without a full timeline.
#[must_use]
pub fn verify_events(
    expected: &[AnimationEvent],
    observed: &[ObservedEvent],
    tolerance_ms: f64,
    video_id: &str,
) -> AnimationReport {
    let timeline = AnimationTimeline {
        video_id: video_id.to_string(),
        events: expected.to_vec(),
    };
    verify_timeline(&timeline, observed, tolerance_ms)
}

fn find_observed_by_name<'a>(
    observed: &'a [ObservedEvent],
    name: &str,
) -> Option<&'a ObservedEvent> {
    observed.iter().find(|o| o.name == name)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::animation::types::AnimationEventType;

    fn make_event(name: &str, secs: f64) -> AnimationEvent {
        AnimationEvent {
            name: name.to_string(),
            event_type: AnimationEventType::PhysicsEvent,
            expected_secs: secs,
            duration_secs: None,
            easing: None,
        }
    }

    fn make_observed(name: &str, secs: f64) -> ObservedEvent {
        ObservedEvent {
            name: name.to_string(),
            time_secs: secs,
        }
    }

    #[test]
    fn test_perfect_match() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![make_event("land_0", 1.7), make_event("land_1", 2.4)],
        };
        let observed = vec![make_observed("land_0", 1.7), make_observed("land_1", 2.4)];
        let report = verify_timeline(&timeline, &observed, 20.0);
        assert_eq!(report.verdict, AnimationVerdict::Pass);
        assert_eq!(report.verified_events, 2);
    }

    #[test]
    fn test_within_tolerance() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![make_event("land_0", 1.7)],
        };
        let observed = vec![make_observed("land_0", 1.71)]; // 10ms delta
        let report = verify_timeline(&timeline, &observed, 20.0);
        assert_eq!(report.verdict, AnimationVerdict::Pass);
    }

    #[test]
    fn test_exceeds_tolerance() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![make_event("land_0", 1.7)],
        };
        let observed = vec![make_observed("land_0", 1.408)]; // -292ms
        let report = verify_timeline(&timeline, &observed, 20.0);
        assert_eq!(report.verdict, AnimationVerdict::Fail);
        assert!((report.max_delta_ms - 292.0).abs() < 0.1);
    }

    #[test]
    fn test_missing_observed_event() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![make_event("land_0", 1.7)],
        };
        let observed: Vec<ObservedEvent> = vec![];
        let report = verify_timeline(&timeline, &observed, 20.0);
        assert_eq!(report.verdict, AnimationVerdict::Fail);
        assert!(report.events[0].actual_secs.is_none());
    }

    #[test]
    fn test_no_events() {
        let timeline = AnimationTimeline {
            video_id: "empty".to_string(),
            events: vec![],
        };
        let observed: Vec<ObservedEvent> = vec![];
        let report = verify_timeline(&timeline, &observed, 20.0);
        assert_eq!(report.verdict, AnimationVerdict::NoEvents);
    }

    #[test]
    fn test_partial_match() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![make_event("land_0", 1.7), make_event("land_1", 2.4)],
        };
        let observed = vec![make_observed("land_0", 1.7)]; // land_1 missing
        let report = verify_timeline(&timeline, &observed, 20.0);
        assert_eq!(report.verdict, AnimationVerdict::Fail);
        assert_eq!(report.verified_events, 1);
        assert_eq!(report.total_events, 2);
    }

    #[test]
    fn test_mean_delta() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![make_event("a", 1.0), make_event("b", 2.0)],
        };
        // 10ms and 20ms deltas -> mean = 15ms
        let observed = vec![make_observed("a", 1.01), make_observed("b", 2.02)];
        let report = verify_timeline(&timeline, &observed, 25.0);
        assert_eq!(report.verdict, AnimationVerdict::Pass);
        assert!((report.mean_delta_ms - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_verify_events_api() {
        let events = vec![make_event("land", 1.5)];
        let observed = vec![make_observed("land", 1.5)];
        let report = verify_events(&events, &observed, 20.0, "test");
        assert_eq!(report.verdict, AnimationVerdict::Pass);
        assert_eq!(report.video_id, "test");
    }

    #[test]
    fn test_event_result_details() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![make_event("bounce", 3.0)],
        };
        let observed = vec![make_observed("bounce", 3.015)];
        let report = verify_timeline(&timeline, &observed, 20.0);
        let ev = &report.events[0];
        assert_eq!(ev.name, "bounce");
        assert!((ev.actual_secs.unwrap() - 3.015).abs() < f64::EPSILON);
        assert!((ev.delta_ms.unwrap() - 15.0).abs() < 0.1);
        assert!(ev.passed);
    }

    #[test]
    fn test_negative_delta() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![make_event("land", 1.7)],
        };
        let observed = vec![make_observed("land", 1.69)]; // 10ms early
        let report = verify_timeline(&timeline, &observed, 20.0);
        assert_eq!(report.verdict, AnimationVerdict::Pass);
        let delta = report.events[0].delta_ms.unwrap();
        assert!(delta < 0.0); // negative means early
    }
}
