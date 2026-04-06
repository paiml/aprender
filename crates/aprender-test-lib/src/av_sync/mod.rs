//! AV Sync Testing: Verify rendered audio-visual synchronization against EDL ground truth.
//!
//! This module extracts audio from rendered video, detects tick onset times,
//! and compares against EDL (Edit Decision List) declarations from rmedia.
//!
//! # Architecture
//!
//! ```text
//! EDL JSON ──→ types::EditDecisionList
//!                       │
//! Video ──→ extraction::extract_audio ──→ detection::detect_onsets
//!                                                    │
//!                                     comparison::compare_edl_to_onsets
//!                                                    │
//!                                              AvSyncReport
//! ```

pub mod comparison;
pub mod detection;
pub mod extraction;
pub mod types;

pub use comparison::compare_edl_to_onsets;
pub use detection::{detect_onsets, DetectionConfig};
pub use extraction::{build_ffmpeg_args, default_edl_path, extract_audio, DEFAULT_SAMPLE_RATE};
pub use types::{
    AudioOnset, AudioTickPlacement, AvSyncReport, EditDecision, EditDecisionList,
    SegmentSyncResult, SyncVerdict, TickDelta,
};
