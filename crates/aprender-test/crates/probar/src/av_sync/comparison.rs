//! Comparison of EDL declarations against detected audio onsets.
//!
//! For each declared tick in the EDL, finds the nearest detected onset
//! within a search window and computes the delta. Aggregates results
//! into an `AvSyncReport`.

use super::types::{
    AudioOnset, AvSyncReport, EditDecisionList, SegmentSyncResult, SyncVerdict, TickDelta,
};

/// Maximum search window in seconds for matching an onset to a declared tick.
const MATCH_WINDOW_SECS: f64 = 0.5;

/// Compare EDL declarations against detected onsets.
///
/// For each EDL tick, finds the nearest detected onset within 500ms.
/// Returns an `AvSyncReport` with per-tick deltas and overall verdict.
pub fn compare_edl_to_onsets(
    edl: &EditDecisionList,
    onsets: &[AudioOnset],
    tolerance_ms: f64,
) -> AvSyncReport {
    if !edl.has_ticks() {
        return AvSyncReport {
            video_id: edl.video_id.clone(),
            verdict: SyncVerdict::NoTicks,
            segments: Vec::new(),
            total_ticks: 0,
            matched_ticks: 0,
            coverage_pct: 0.0,
            max_delta_ms: 0.0,
            mean_delta_ms: 0.0,
        };
    }

    let mut acc = ComparisonAccumulator::default();

    let segments: Vec<SegmentSyncResult> = edl
        .decisions
        .iter()
        .filter(|d| !d.ticks.is_empty())
        .map(|decision| compare_segment(decision, onsets, tolerance_ms, &mut acc))
        .collect();

    acc.into_report(edl.video_id.clone(), segments)
}

/// Accumulated statistics across all segments.
#[derive(Default)]
struct ComparisonAccumulator {
    total_ticks: usize,
    matched_ticks: usize,
    all_deltas: Vec<f64>,
    max_delta_ms: f64,
}

impl ComparisonAccumulator {
    fn record_match(&mut self, abs_delta: f64, passed: bool) {
        self.total_ticks += 1;
        self.all_deltas.push(abs_delta);
        if abs_delta > self.max_delta_ms {
            self.max_delta_ms = abs_delta;
        }
        if passed {
            self.matched_ticks += 1;
        }
    }

    fn record_miss(&mut self) {
        self.total_ticks += 1;
    }

    #[allow(clippy::cast_precision_loss)]
    fn into_report(self, video_id: String, segments: Vec<SegmentSyncResult>) -> AvSyncReport {
        let mean_delta_ms = if self.all_deltas.is_empty() {
            0.0
        } else {
            self.all_deltas.iter().sum::<f64>() / self.all_deltas.len() as f64
        };

        let coverage_pct = if self.total_ticks > 0 {
            (self.matched_ticks as f64 / self.total_ticks as f64) * 100.0
        } else {
            0.0
        };

        let verdict = if self.matched_ticks == self.total_ticks {
            SyncVerdict::Pass
        } else {
            SyncVerdict::Fail
        };

        AvSyncReport {
            video_id,
            verdict,
            segments,
            total_ticks: self.total_ticks,
            matched_ticks: self.matched_ticks,
            coverage_pct,
            max_delta_ms: self.max_delta_ms,
            mean_delta_ms,
        }
    }
}

/// Compare a single segment's ticks against detected onsets.
fn compare_segment(
    decision: &super::types::EditDecision,
    onsets: &[AudioOnset],
    tolerance_ms: f64,
    acc: &mut ComparisonAccumulator,
) -> SegmentSyncResult {
    let mut segment_all_passed = true;

    let tick_deltas: Vec<TickDelta> = decision
        .ticks
        .iter()
        .map(|tick| {
            let declared = tick.audio_place_secs;
            let nearest = find_nearest_onset(onsets, declared, MATCH_WINDOW_SECS);
            let (actual_secs, delta_ms, passed) = if let Some(onset) = nearest {
                let delta = (onset.time_secs - declared) * 1000.0;
                let abs_delta = delta.abs();
                let tick_passed = abs_delta <= tolerance_ms;
                acc.record_match(abs_delta, tick_passed);
                (Some(onset.time_secs), Some(delta), tick_passed)
            } else {
                acc.record_miss();
                (None, None, false)
            };
            if !passed {
                segment_all_passed = false;
            }
            TickDelta {
                segment: decision.segment.clone(),
                bullet_index: tick.bullet_index,
                declared_secs: declared,
                actual_secs,
                delta_ms,
                passed,
            }
        })
        .collect();

    SegmentSyncResult {
        segment: decision.segment.clone(),
        ticks: tick_deltas,
        all_passed: segment_all_passed,
    }
}

/// Find the nearest onset to a declared time within a search window.
fn find_nearest_onset(
    onsets: &[AudioOnset],
    declared_secs: f64,
    window_secs: f64,
) -> Option<&AudioOnset> {
    onsets
        .iter()
        .filter(|o| (o.time_secs - declared_secs).abs() <= window_secs)
        .min_by(|a, b| {
            let da = (a.time_secs - declared_secs).abs();
            let db = (b.time_secs - declared_secs).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::av_sync::types::{AudioTickPlacement, EditDecision};

    fn make_edl(ticks: Vec<(usize, f64, f64)>) -> EditDecisionList {
        EditDecisionList {
            video_id: "test-video".to_string(),
            decisions: vec![EditDecision {
                segment: "P2-key_terms".to_string(),
                fps: 24,
                sample_rate: 48000,
                ticks: ticks
                    .into_iter()
                    .map(|(idx, visual, audio)| AudioTickPlacement {
                        bullet_index: idx,
                        visual_land_secs: visual,
                        audio_place_secs: audio,
                        peak_anticipation_ms: 0.0,
                        perceptual_lead_ms: 0.0,
                    })
                    .collect(),
            }],
        }
    }

    fn make_onsets(times: &[f64]) -> Vec<AudioOnset> {
        times
            .iter()
            .enumerate()
            .map(|(_i, &t)| AudioOnset {
                time_secs: t,
                energy_db: -20.0,
                sample_index: (t * 48000.0) as usize,
            })
            .collect()
    }

    #[test]
    fn test_perfect_sync() {
        let edl = make_edl(vec![(0, 1.7, 1.7), (1, 2.4, 2.4)]);
        let onsets = make_onsets(&[1.7, 2.4]);
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::Pass);
        assert_eq!(report.total_ticks, 2);
        assert_eq!(report.matched_ticks, 2);
        assert!((report.coverage_pct - 100.0).abs() < f64::EPSILON);
        assert!(report.max_delta_ms < f64::EPSILON);
    }

    #[test]
    fn test_within_tolerance() {
        let edl = make_edl(vec![(0, 1.7, 1.7)]);
        let onsets = make_onsets(&[1.71]); // 10ms delta
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::Pass);
        assert_eq!(report.matched_ticks, 1);
        assert!((report.max_delta_ms - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_exceeds_tolerance() {
        let edl = make_edl(vec![(0, 1.7, 1.7)]);
        let onsets = make_onsets(&[1.408]); // -292ms delta
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::Fail);
        assert_eq!(report.matched_ticks, 0);
        assert!((report.max_delta_ms - 292.0).abs() < 0.1);
    }

    #[test]
    fn test_consistent_drift() {
        let edl = make_edl(vec![(0, 1.7, 1.7), (1, 2.4, 2.4), (2, 3.1, 3.1)]);
        let onsets = make_onsets(&[1.408, 2.108, 2.808]); // all -292ms
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::Fail);
        assert_eq!(report.total_ticks, 3);
        assert_eq!(report.matched_ticks, 0);
        assert!((report.max_delta_ms - 292.0).abs() < 0.1);
        assert!((report.mean_delta_ms - 292.0).abs() < 0.1);
    }

    #[test]
    fn test_no_ticks() {
        let edl = EditDecisionList {
            video_id: "empty".to_string(),
            decisions: vec![EditDecision {
                segment: "seg".to_string(),
                fps: 24,
                sample_rate: 48000,
                ticks: vec![],
            }],
        };
        let onsets = make_onsets(&[1.0, 2.0]);
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::NoTicks);
        assert_eq!(report.total_ticks, 0);
    }

    #[test]
    fn test_no_onsets_detected() {
        let edl = make_edl(vec![(0, 1.7, 1.7)]);
        let onsets: Vec<AudioOnset> = Vec::new();
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::Fail);
        assert_eq!(report.total_ticks, 1);
        assert_eq!(report.matched_ticks, 0);
    }

    #[test]
    fn test_onset_outside_match_window() {
        let edl = make_edl(vec![(0, 1.7, 1.7)]);
        let onsets = make_onsets(&[0.5]); // 1.2s away, outside 500ms window
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::Fail);
        assert!(report.segments[0].ticks[0].actual_secs.is_none());
    }

    #[test]
    fn test_multiple_segments() {
        let edl = EditDecisionList {
            video_id: "multi-seg".to_string(),
            decisions: vec![
                EditDecision {
                    segment: "P2-key_terms".to_string(),
                    fps: 24,
                    sample_rate: 48000,
                    ticks: vec![AudioTickPlacement {
                        bullet_index: 0,
                        visual_land_secs: 1.7,
                        audio_place_secs: 1.7,
                        peak_anticipation_ms: 0.0,
                        perceptual_lead_ms: 0.0,
                    }],
                },
                EditDecision {
                    segment: "P4-reflection".to_string(),
                    fps: 24,
                    sample_rate: 48000,
                    ticks: vec![AudioTickPlacement {
                        bullet_index: 0,
                        visual_land_secs: 5.0,
                        audio_place_secs: 5.0,
                        peak_anticipation_ms: 0.0,
                        perceptual_lead_ms: 0.0,
                    }],
                },
            ],
        };

        let onsets = make_onsets(&[1.7, 5.0]);
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::Pass);
        assert_eq!(report.segments.len(), 2);
        assert!(report.segments[0].all_passed);
        assert!(report.segments[1].all_passed);
    }

    #[test]
    fn test_partial_match() {
        let edl = make_edl(vec![(0, 1.7, 1.7), (1, 2.4, 2.4)]);
        let onsets = make_onsets(&[1.7]); // Only first tick matches
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::Fail);
        assert_eq!(report.matched_ticks, 1);
        assert_eq!(report.total_ticks, 2);
        assert!((report.coverage_pct - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nearest_onset_selection() {
        // Two onsets near the declared time -- should pick the closest
        let edl = make_edl(vec![(0, 1.7, 1.7)]);
        let onsets = make_onsets(&[1.69, 1.75]); // 10ms early vs 50ms late
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);

        assert_eq!(report.verdict, SyncVerdict::Pass);
        let delta = report.segments[0].ticks[0].delta_ms.unwrap();
        assert!(delta.abs() < 11.0, "should pick nearest onset (10ms delta)");
    }

    #[test]
    fn test_zero_tolerance() {
        let edl = make_edl(vec![(0, 1.7, 1.7)]);
        let onsets = make_onsets(&[1.701]); // 1ms delta
        let report = compare_edl_to_onsets(&edl, &onsets, 0.0);

        assert_eq!(report.verdict, SyncVerdict::Fail);
    }

    #[test]
    fn test_report_video_id() {
        let edl = make_edl(vec![(0, 1.7, 1.7)]);
        let onsets = make_onsets(&[1.7]);
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);
        assert_eq!(report.video_id, "test-video");
    }

    #[test]
    fn test_tick_delta_segment_propagation() {
        let edl = make_edl(vec![(0, 1.7, 1.7)]);
        let onsets = make_onsets(&[1.7]);
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);
        assert_eq!(report.segments[0].ticks[0].segment, "P2-key_terms");
    }

    #[test]
    fn test_empty_decisions() {
        let edl = EditDecisionList {
            video_id: "empty".to_string(),
            decisions: vec![],
        };
        let onsets = make_onsets(&[1.0]);
        let report = compare_edl_to_onsets(&edl, &onsets, 20.0);
        assert_eq!(report.verdict, SyncVerdict::NoTicks);
    }
}
