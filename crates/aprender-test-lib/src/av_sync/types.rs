//! EDL (Edit Decision List) types and AV sync report structures.
//!
//! These types mirror rmedia's EDL output format with `Deserialize` added.
//! No rmedia dependency -- probar reads the JSON wire format independently.

use serde::{Deserialize, Serialize};

/// Edit decision list for a rendered video.
///
/// Contains all audio tick placements organized by segment.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditDecisionList {
    /// Unique identifier for the rendered video
    pub video_id: String,
    /// One entry per segment containing audio ticks
    pub decisions: Vec<EditDecision>,
}

impl EditDecisionList {
    /// Returns true if any decision has tick placements.
    #[must_use]
    pub fn has_ticks(&self) -> bool {
        self.decisions.iter().any(|d| !d.ticks.is_empty())
    }

    /// Total number of ticks across all segments.
    #[must_use]
    pub fn tick_count(&self) -> usize {
        self.decisions.iter().map(|d| d.ticks.len()).sum()
    }
}

/// A single segment's edit decisions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditDecision {
    /// Segment name (e.g., "P2-key_terms")
    pub segment: String,
    /// Frame rate of the rendered video
    pub fps: u32,
    /// Audio sample rate (typically 48000 Hz)
    pub sample_rate: u32,
    /// Audio tick placements within this segment
    pub ticks: Vec<AudioTickPlacement>,
}

/// Placement of an audio tick relative to a visual event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AudioTickPlacement {
    /// Index of the bullet in the drop sequence
    pub bullet_index: usize,
    /// Time (seconds) when the bullet visually lands on screen
    pub visual_land_secs: f64,
    /// Time (seconds) when the audio tick should start playing
    pub audio_place_secs: f64,
    /// Milliseconds of audio anticipation (currently always 0.0 for drop sounds)
    pub peak_anticipation_ms: f64,
    /// Milliseconds of perceptual lead time
    pub perceptual_lead_ms: f64,
}

/// Detected audio onset from waveform analysis.
#[derive(Clone, Debug)]
pub struct AudioOnset {
    /// Time in seconds where the onset was detected
    pub time_secs: f64,
    /// Energy level in dB at the onset
    pub energy_db: f64,
    /// Sample index in the PCM stream
    pub sample_index: usize,
}

/// Result of comparing EDL declarations against actual audio.
#[derive(Clone, Debug, Serialize)]
pub struct AvSyncReport {
    /// Video identifier from EDL
    pub video_id: String,
    /// Overall verdict
    pub verdict: SyncVerdict,
    /// Per-segment results
    pub segments: Vec<SegmentSyncResult>,
    /// Total ticks declared in EDL
    pub total_ticks: usize,
    /// Number of ticks matched within tolerance
    pub matched_ticks: usize,
    /// Percentage of ticks that passed (0.0-100.0)
    pub coverage_pct: f64,
    /// Maximum absolute delta in milliseconds
    pub max_delta_ms: f64,
    /// Mean absolute delta in milliseconds
    pub mean_delta_ms: f64,
}

/// Per-segment sync verification results.
#[derive(Clone, Debug, Serialize)]
pub struct SegmentSyncResult {
    /// Segment name
    pub segment: String,
    /// Per-tick deltas
    pub ticks: Vec<TickDelta>,
    /// Whether all ticks in this segment passed
    pub all_passed: bool,
}

/// Delta between declared and actual tick timing.
#[derive(Clone, Debug, Serialize)]
pub struct TickDelta {
    /// Segment name (for display)
    pub segment: String,
    /// Bullet index from EDL
    pub bullet_index: usize,
    /// Declared audio placement time (seconds)
    pub declared_secs: f64,
    /// Actual detected onset time (seconds), None if no match found
    pub actual_secs: Option<f64>,
    /// Delta in milliseconds (actual - declared), None if no match
    pub delta_ms: Option<f64>,
    /// Whether this tick passed the tolerance check
    pub passed: bool,
}

/// Overall sync verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum SyncVerdict {
    /// All ticks within tolerance
    Pass,
    /// One or more ticks exceed tolerance
    Fail,
    /// No ticks found in EDL
    NoTicks,
}

impl std::fmt::Display for SyncVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "PASS"),
            Self::Fail => write!(f, "FAIL"),
            Self::NoTicks => write!(f, "NO TICKS"),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn sample_edl() -> EditDecisionList {
        EditDecisionList {
            video_id: "demo-bench".to_string(),
            decisions: vec![
                EditDecision {
                    segment: "P2-key_terms".to_string(),
                    fps: 24,
                    sample_rate: 48000,
                    ticks: vec![
                        AudioTickPlacement {
                            bullet_index: 0,
                            visual_land_secs: 1.700,
                            audio_place_secs: 1.658,
                            peak_anticipation_ms: 0.0,
                            perceptual_lead_ms: 41.667,
                        },
                        AudioTickPlacement {
                            bullet_index: 1,
                            visual_land_secs: 2.400,
                            audio_place_secs: 2.358,
                            peak_anticipation_ms: 0.0,
                            perceptual_lead_ms: 41.667,
                        },
                    ],
                },
                EditDecision {
                    segment: "P4-reflection".to_string(),
                    fps: 24,
                    sample_rate: 48000,
                    ticks: vec![],
                },
            ],
        }
    }

    #[test]
    fn test_edl_has_ticks() {
        let edl = sample_edl();
        assert!(edl.has_ticks());
    }

    #[test]
    fn test_edl_no_ticks() {
        let edl = EditDecisionList {
            video_id: "empty".to_string(),
            decisions: vec![EditDecision {
                segment: "seg".to_string(),
                fps: 24,
                sample_rate: 48000,
                ticks: vec![],
            }],
        };
        assert!(!edl.has_ticks());
    }

    #[test]
    fn test_edl_tick_count() {
        let edl = sample_edl();
        assert_eq!(edl.tick_count(), 2);
    }

    #[test]
    fn test_edl_tick_count_empty() {
        let edl = EditDecisionList {
            video_id: "empty".to_string(),
            decisions: vec![],
        };
        assert_eq!(edl.tick_count(), 0);
    }

    #[test]
    fn test_edl_json_roundtrip() {
        let edl = sample_edl();
        let json = serde_json::to_string_pretty(&edl).unwrap();
        let parsed: EditDecisionList = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.video_id, "demo-bench");
        assert_eq!(parsed.decisions.len(), 2);
        assert_eq!(parsed.decisions[0].ticks.len(), 2);
        assert!((parsed.decisions[0].ticks[0].visual_land_secs - 1.700).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edl_deserialize_from_rmedia_format() {
        let json = r#"{
            "video_id": "test-video",
            "decisions": [{
                "segment": "P2-key_terms",
                "fps": 24,
                "sample_rate": 48000,
                "ticks": [{
                    "bullet_index": 0,
                    "visual_land_secs": 1.7,
                    "audio_place_secs": 1.658,
                    "peak_anticipation_ms": 0.0,
                    "perceptual_lead_ms": 41.667
                }]
            }]
        }"#;
        let edl: EditDecisionList = serde_json::from_str(json).unwrap();
        assert_eq!(edl.video_id, "test-video");
        assert_eq!(edl.decisions[0].ticks[0].bullet_index, 0);
    }

    #[test]
    fn test_sync_verdict_display() {
        assert_eq!(SyncVerdict::Pass.to_string(), "PASS");
        assert_eq!(SyncVerdict::Fail.to_string(), "FAIL");
        assert_eq!(SyncVerdict::NoTicks.to_string(), "NO TICKS");
    }

    #[test]
    fn test_sync_verdict_equality() {
        assert_eq!(SyncVerdict::Pass, SyncVerdict::Pass);
        assert_ne!(SyncVerdict::Pass, SyncVerdict::Fail);
    }

    #[test]
    fn test_audio_onset_creation() {
        let onset = AudioOnset {
            time_secs: 1.5,
            energy_db: -20.0,
            sample_index: 72000,
        };
        assert!((onset.time_secs - 1.5).abs() < f64::EPSILON);
        assert_eq!(onset.sample_index, 72000);
    }

    #[test]
    fn test_tick_delta_passed() {
        let delta = TickDelta {
            segment: "seg".to_string(),
            bullet_index: 0,
            declared_secs: 1.7,
            actual_secs: Some(1.71),
            delta_ms: Some(10.0),
            passed: true,
        };
        assert!(delta.passed);
        assert!((delta.delta_ms.unwrap() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tick_delta_no_match() {
        let delta = TickDelta {
            segment: "seg".to_string(),
            bullet_index: 0,
            declared_secs: 1.7,
            actual_secs: None,
            delta_ms: None,
            passed: false,
        };
        assert!(!delta.passed);
        assert!(delta.actual_secs.is_none());
    }

    #[test]
    fn test_av_sync_report_serialization() {
        let report = AvSyncReport {
            video_id: "test".to_string(),
            verdict: SyncVerdict::Pass,
            segments: vec![],
            total_ticks: 3,
            matched_ticks: 3,
            coverage_pct: 100.0,
            max_delta_ms: 5.0,
            mean_delta_ms: 3.0,
        };
        let json = serde_json::to_string(&report).unwrap();
        assert!(json.contains("\"verdict\":\"Pass\""));
        assert!(json.contains("\"total_ticks\":3"));
    }

    #[test]
    fn test_segment_sync_result() {
        let result = SegmentSyncResult {
            segment: "P2-key_terms".to_string(),
            ticks: vec![],
            all_passed: true,
        };
        assert!(result.all_passed);
        assert_eq!(result.segment, "P2-key_terms");
    }

    #[test]
    fn test_edit_decision_clone() {
        let decision = EditDecision {
            segment: "test".to_string(),
            fps: 24,
            sample_rate: 48000,
            ticks: vec![],
        };
        let cloned = decision;
        assert_eq!(cloned.segment, "test");
        assert_eq!(cloned.fps, 24);
    }

    #[test]
    fn test_audio_tick_placement_clone() {
        let tick = AudioTickPlacement {
            bullet_index: 0,
            visual_land_secs: 1.7,
            audio_place_secs: 1.658,
            peak_anticipation_ms: 0.0,
            perceptual_lead_ms: 41.667,
        };
        let cloned = tick;
        assert_eq!(cloned.bullet_index, 0);
        assert!((cloned.perceptual_lead_ms - 41.667).abs() < f64::EPSILON);
    }
}
