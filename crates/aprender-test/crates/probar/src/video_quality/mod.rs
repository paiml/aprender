//! Video Quality Verification: codec, resolution, FPS, duration validation.
//!
//! Probes rendered video files with ffprobe and validates metadata
//! against expected properties.
//!
//! # Usage
//!
//! ```text
//! Video ──→ probe::probe_video ──→ VideoProbe
//!                                       │
//!           VideoExpectations ──→ validation::validate_video
//!                                       │
//!                                VideoQualityReport
//! ```

pub mod probe;
pub mod types;
pub mod validation;

pub use probe::{build_ffprobe_args, parse_ffprobe_json, probe_video};
pub use types::{VideoCheck, VideoExpectations, VideoProbe, VideoQualityReport, VideoVerdict};
pub use validation::validate_video;
