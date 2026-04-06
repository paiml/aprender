//! Types for animation verification.
//!
//! Provides structures for declaring expected animation events and
//! verifying their timing in rendered output.

use serde::{Deserialize, Serialize};

/// Animation timeline declaration — the expected events in a render.
///
/// This is the contract between the renderer (rmedia) and the verifier (probar).
/// The renderer writes a timeline JSON alongside the video; the verifier checks it.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimationTimeline {
    /// Video identifier
    pub video_id: String,
    /// Expected animation events
    pub events: Vec<AnimationEvent>,
}

impl AnimationTimeline {
    /// Total number of events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Check if timeline has any events.
    #[must_use]
    pub fn has_events(&self) -> bool {
        !self.events.is_empty()
    }
}

/// A declared animation event with expected timing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnimationEvent {
    /// Event name/label (e.g., "bullet_0_land", "logo_bounce_start")
    pub name: String,
    /// Event type
    pub event_type: AnimationEventType,
    /// Expected time in seconds
    pub expected_secs: f64,
    /// Expected duration in seconds (for events with duration)
    pub duration_secs: Option<f64>,
    /// Easing function name (for transitions)
    pub easing: Option<String>,
}

/// Types of animation events.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnimationEventType {
    /// Element appears/enters
    Enter,
    /// Element exits/disappears
    Exit,
    /// Transition starts (fade, slide, etc.)
    TransitionStart,
    /// Transition ends
    TransitionEnd,
    /// Keyframe hit (specific animation point)
    Keyframe,
    /// Physics event (bounce, land, etc.)
    PhysicsEvent,
}

impl std::fmt::Display for AnimationEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Enter => write!(f, "enter"),
            Self::Exit => write!(f, "exit"),
            Self::TransitionStart => write!(f, "transition_start"),
            Self::TransitionEnd => write!(f, "transition_end"),
            Self::Keyframe => write!(f, "keyframe"),
            Self::PhysicsEvent => write!(f, "physics_event"),
        }
    }
}

/// Animation verification report.
#[derive(Clone, Debug, Serialize)]
pub struct AnimationReport {
    /// Video identifier
    pub video_id: String,
    /// Overall verdict
    pub verdict: AnimationVerdict,
    /// Per-event results
    pub events: Vec<EventResult>,
    /// Total events declared
    pub total_events: usize,
    /// Events verified within tolerance
    pub verified_events: usize,
    /// Maximum timing delta in milliseconds
    pub max_delta_ms: f64,
    /// Mean timing delta in milliseconds
    pub mean_delta_ms: f64,
}

/// Per-event verification result.
#[derive(Clone, Debug, Serialize)]
pub struct EventResult {
    /// Event name
    pub name: String,
    /// Event type
    pub event_type: AnimationEventType,
    /// Expected time in seconds
    pub expected_secs: f64,
    /// Actual time in seconds (None if not detected)
    pub actual_secs: Option<f64>,
    /// Delta in milliseconds
    pub delta_ms: Option<f64>,
    /// Whether the event passed timing check
    pub passed: bool,
}

/// Overall animation verification verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnimationVerdict {
    /// All events verified within tolerance
    Pass,
    /// One or more events failed
    Fail,
    /// No events to verify
    NoEvents,
}

impl std::fmt::Display for AnimationVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::NoEvents => write!(f, "NO EVENTS"),
        }
    }
}

/// Easing function definitions for animation curves.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EasingFunction {
    /// Linear interpolation
    Linear,
    /// Quadratic ease-in
    EaseIn,
    /// Quadratic ease-out
    EaseOut,
    /// Quadratic ease-in-out
    EaseInOut,
    /// Cubic ease-in
    CubicIn,
    /// Cubic ease-out
    CubicOut,
    /// Cubic ease-in-out
    CubicInOut,
    /// Bounce effect
    Bounce,
    /// Custom cubic bezier
    CubicBezier(f64, f64, f64, f64),
}

impl EasingFunction {
    /// Evaluate the easing function at time t (0.0-1.0).
    ///
    /// Returns the interpolated value (0.0-1.0).
    #[must_use]
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Linear => t,
            Self::EaseIn => t * t,
            Self::EaseOut => t * (2.0 - t),
            Self::EaseInOut => {
                if t < 0.5 {
                    2.0 * t * t
                } else {
                    -1.0 + (4.0 - 2.0 * t) * t
                }
            }
            Self::CubicIn => t * t * t,
            Self::CubicOut => {
                let t1 = t - 1.0;
                t1 * t1 * t1 + 1.0
            }
            Self::CubicInOut => {
                if t < 0.5 {
                    4.0 * t * t * t
                } else {
                    let t1 = 2.0 * t - 2.0;
                    0.5 * t1 * t1 * t1 + 1.0
                }
            }
            Self::Bounce => bounce_ease_out(t),
            Self::CubicBezier(x1, y1, x2, y2) => cubic_bezier_approx(t, *x1, *y1, *x2, *y2),
        }
    }
}

/// Bounce easing function.
fn bounce_ease_out(t: f64) -> f64 {
    if t < 1.0 / 2.75 {
        7.5625 * t * t
    } else if t < 2.0 / 2.75 {
        let t = t - 1.5 / 2.75;
        7.5625 * t * t + 0.75
    } else if t < 2.5 / 2.75 {
        let t = t - 2.25 / 2.75;
        7.5625 * t * t + 0.9375
    } else {
        let t = t - 2.625 / 2.75;
        7.5625 * t * t + 0.984_375
    }
}

/// Approximate cubic bezier evaluation (for CSS-style timing functions).
fn cubic_bezier_approx(t: f64, _x1: f64, y1: f64, _x2: f64, y2: f64) -> f64 {
    // Simple approximation using De Casteljau subdivision
    // Control points: (0,0), (x1,y1), (x2,y2), (1,1)
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let t2 = t * t;
    mt2 * mt * 0.0 + 3.0 * mt2 * t * y1 + 3.0 * mt * t2 * y2 + t2 * t * 1.0
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_animation_verdict_display() {
        assert_eq!(AnimationVerdict::Pass.to_string(), "PASS");
        assert_eq!(AnimationVerdict::Fail.to_string(), "FAIL");
        assert_eq!(AnimationVerdict::NoEvents.to_string(), "NO EVENTS");
    }

    #[test]
    fn test_animation_event_type_display() {
        assert_eq!(AnimationEventType::Enter.to_string(), "enter");
        assert_eq!(
            AnimationEventType::PhysicsEvent.to_string(),
            "physics_event"
        );
    }

    #[test]
    fn test_timeline_event_count() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![
                AnimationEvent {
                    name: "event1".to_string(),
                    event_type: AnimationEventType::Enter,
                    expected_secs: 1.0,
                    duration_secs: None,
                    easing: None,
                },
                AnimationEvent {
                    name: "event2".to_string(),
                    event_type: AnimationEventType::Exit,
                    expected_secs: 2.0,
                    duration_secs: None,
                    easing: None,
                },
            ],
        };
        assert_eq!(timeline.event_count(), 2);
        assert!(timeline.has_events());
    }

    #[test]
    fn test_timeline_empty() {
        let timeline = AnimationTimeline {
            video_id: "empty".to_string(),
            events: vec![],
        };
        assert_eq!(timeline.event_count(), 0);
        assert!(!timeline.has_events());
    }

    #[test]
    fn test_timeline_json_roundtrip() {
        let timeline = AnimationTimeline {
            video_id: "test".to_string(),
            events: vec![AnimationEvent {
                name: "bullet_land".to_string(),
                event_type: AnimationEventType::PhysicsEvent,
                expected_secs: 1.7,
                duration_secs: Some(0.05),
                easing: Some("bounce".to_string()),
            }],
        };
        let json = serde_json::to_string(&timeline).unwrap();
        let parsed: AnimationTimeline = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.video_id, "test");
        assert_eq!(parsed.events.len(), 1);
        assert_eq!(
            parsed.events[0].event_type,
            AnimationEventType::PhysicsEvent
        );
    }

    #[test]
    fn test_easing_linear() {
        let f = EasingFunction::Linear;
        assert!((f.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((f.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((f.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_easing_ease_in() {
        let f = EasingFunction::EaseIn;
        assert!((f.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((f.evaluate(0.5) - 0.25).abs() < f64::EPSILON); // t^2
        assert!((f.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_easing_ease_out() {
        let f = EasingFunction::EaseOut;
        assert!((f.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((f.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
        // ease-out should be faster at start
        assert!(f.evaluate(0.5) > 0.5);
    }

    #[test]
    fn test_easing_ease_in_out() {
        let f = EasingFunction::EaseInOut;
        assert!((f.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((f.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((f.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_easing_cubic_in() {
        let f = EasingFunction::CubicIn;
        assert!((f.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((f.evaluate(0.5) - 0.125).abs() < f64::EPSILON); // t^3
        assert!((f.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_easing_cubic_out() {
        let f = EasingFunction::CubicOut;
        assert!((f.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((f.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_easing_cubic_in_out() {
        let f = EasingFunction::CubicInOut;
        assert!((f.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((f.evaluate(0.5) - 0.5).abs() < f64::EPSILON);
        assert!((f.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_easing_bounce() {
        let f = EasingFunction::Bounce;
        assert!((f.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((f.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
        // Bounce should have oscillations
        assert!(f.evaluate(0.5) > 0.0);
    }

    #[test]
    fn test_easing_cubic_bezier() {
        let f = EasingFunction::CubicBezier(0.25, 0.1, 0.25, 1.0); // CSS "ease"
        assert!((f.evaluate(0.0)).abs() < f64::EPSILON);
        assert!((f.evaluate(1.0) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_easing_clamp() {
        let f = EasingFunction::Linear;
        assert!((f.evaluate(-0.5)).abs() < f64::EPSILON);
        assert!((f.evaluate(1.5) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_event_result() {
        let result = EventResult {
            name: "bullet_land".to_string(),
            event_type: AnimationEventType::PhysicsEvent,
            expected_secs: 1.7,
            actual_secs: Some(1.71),
            delta_ms: Some(10.0),
            passed: true,
        };
        assert!(result.passed);
    }

    #[test]
    fn test_animation_report_serialization() {
        let report = AnimationReport {
            video_id: "test".to_string(),
            verdict: AnimationVerdict::Pass,
            events: vec![],
            total_events: 0,
            verified_events: 0,
            max_delta_ms: 0.0,
            mean_delta_ms: 0.0,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"verdict\":\"Pass\""));
    }
}
