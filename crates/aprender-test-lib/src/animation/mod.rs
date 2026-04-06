//! Animation Verification: timing, easing curves, physics events.
//!
//! Verifies that rendered animations match their declared timelines
//! and easing functions.
//!
//! # Architecture
//!
//! ```text
//! AnimationTimeline (JSON) ──→ timing::verify_timeline
//!                                      │
//! ObservedEvents ─────────────────────►│
//!                                      │
//!                               AnimationReport
//!
//! Keyframes ──→ easing::verify_easing ──→ EasingVerification
//! ```
//!
//! # Integration with rmedia
//!
//! rmedia writes animation timelines as JSON alongside rendered videos.
//! probar reads these timelines and verifies actual timing matches intent.

pub mod easing;
pub mod timing;
pub mod types;

pub use easing::{sample_easing, verify_easing, EasingVerification, Keyframe};
pub use timing::{verify_events, verify_timeline, ObservedEvent};
pub use types::{
    AnimationEvent, AnimationEventType, AnimationReport, AnimationTimeline, AnimationVerdict,
    EasingFunction, EventResult,
};
