//! AV sync command handler.
//!
//! Orchestrates: parse EDL -> extract audio -> detect onsets -> compare -> render report.

use crate::commands::{AvSyncCheckArgs, AvSyncOutputFormat, AvSyncReportArgs};
use crate::config::CliConfig;
use crate::error::{CliError, CliResult};
use jugar_probar::av_sync::{
    compare_edl_to_onsets, default_edl_path, detect_onsets, extract_audio, DetectionConfig,
    EditDecisionList, SyncVerdict, DEFAULT_SAMPLE_RATE,
};
use std::path::Path;

/// Execute the av-sync check command for a single video.
pub fn execute_check(config: &CliConfig, args: &AvSyncCheckArgs) -> CliResult<()> {
    let video_path = &args.video;
    if !video_path.exists() {
        return Err(CliError::invalid_argument(format!(
            "Video file not found: {}",
            video_path.display()
        )));
    }

    let edl_path = args
        .edl
        .clone()
        .unwrap_or_else(|| default_edl_path(video_path));

    if !edl_path.exists() {
        return Err(CliError::invalid_argument(format!(
            "EDL file not found: {} (use --edl to specify)",
            edl_path.display()
        )));
    }

    let edl = load_edl(&edl_path)?;
    let report = run_av_sync_check(video_path, &edl, args.tolerance_ms, config)?;

    match args.format {
        AvSyncOutputFormat::Text => {
            print!(
                "{}",
                render_text_report(&report, args.tolerance_ms, args.detailed)
            );
        }
        AvSyncOutputFormat::Json => {
            let json = serde_json::to_string_pretty(&report).map_err(|e| {
                CliError::report_generation(format!("JSON serialization error: {e}"))
            })?;
            println!("{json}");
        }
    }

    if report.verdict == SyncVerdict::Fail {
        Err(CliError::test_execution(format!(
            "AV sync check failed: {}/{} ticks passed (max drift: {:.1}ms)",
            report.matched_ticks, report.total_ticks, report.max_delta_ms
        )))
    } else {
        Ok(())
    }
}

/// Execute the av-sync report command for a directory.
pub fn execute_report(config: &CliConfig, args: &AvSyncReportArgs) -> CliResult<()> {
    let dir = &args.dir;
    if !dir.exists() || !dir.is_dir() {
        return Err(CliError::invalid_argument(format!(
            "Directory not found: {}",
            dir.display()
        )));
    }

    let edl_files = find_edl_files(dir);
    if edl_files.is_empty() {
        return Err(CliError::invalid_argument(format!(
            "No .edl.json files found in {}",
            dir.display()
        )));
    }

    let (reports, all_passed) = check_all_edls(&edl_files, args.tolerance_ms, config)?;
    let output = format_report_output(&reports, &args.format, args.tolerance_ms)?;
    write_output(&output, args.output.as_deref())?;

    if all_passed {
        Ok(())
    } else {
        Err(CliError::test_execution(
            "One or more videos failed AV sync check".to_string(),
        ))
    }
}

/// Process all EDL files and collect reports.
fn check_all_edls(
    edl_files: &[std::path::PathBuf],
    tolerance_ms: f64,
    config: &CliConfig,
) -> CliResult<(Vec<jugar_probar::av_sync::AvSyncReport>, bool)> {
    let mut all_passed = true;
    let mut reports = Vec::new();

    for edl_path in edl_files {
        let edl = load_edl(edl_path)?;
        if let Some(vp) = find_video_for_edl(edl_path) {
            let report = run_av_sync_check(&vp, &edl, tolerance_ms, config)?;
            if report.verdict == SyncVerdict::Fail {
                all_passed = false;
            }
            reports.push(report);
        } else {
            eprintln!("Warning: No video found for EDL {}", edl_path.display());
        }
    }

    Ok((reports, all_passed))
}

/// Format reports into a string based on output format.
fn format_report_output(
    reports: &[jugar_probar::av_sync::AvSyncReport],
    format: &AvSyncOutputFormat,
    tolerance_ms: f64,
) -> CliResult<String> {
    match format {
        AvSyncOutputFormat::Text => {
            let mut output = String::new();
            for report in reports {
                output.push_str(&render_text_report(report, tolerance_ms, true));
                output.push('\n');
            }
            Ok(output)
        }
        AvSyncOutputFormat::Json => serde_json::to_string_pretty(reports)
            .map_err(|e| CliError::report_generation(format!("JSON serialization error: {e}"))),
    }
}

/// Write output to file or stdout.
fn write_output(content: &str, out_path: Option<&Path>) -> CliResult<()> {
    out_path.map_or_else(
        || {
            print!("{content}");
            Ok(())
        },
        |path| {
            std::fs::write(path, content)
                .map_err(|e| CliError::report_generation(format!("Failed to write report: {e}")))
        },
    )
}

/// Run the AV sync check pipeline for a single video.
fn run_av_sync_check(
    video_path: &Path,
    edl: &EditDecisionList,
    tolerance_ms: f64,
    config: &CliConfig,
) -> CliResult<jugar_probar::av_sync::AvSyncReport> {
    if config.verbosity.is_verbose() {
        eprintln!("Extracting audio from {}...", video_path.display());
    }

    let sample_rate = edl
        .decisions
        .first()
        .map_or(DEFAULT_SAMPLE_RATE, |d| d.sample_rate);

    let samples = extract_audio(video_path, sample_rate)
        .map_err(|e| CliError::test_execution(format!("Audio extraction failed: {e}")))?;

    if config.verbosity.is_verbose() {
        #[allow(clippy::cast_precision_loss)]
        let duration_secs = (samples.len() as f64) / f64::from(sample_rate);
        eprintln!(
            "Extracted {} samples ({duration_secs:.1}s at {sample_rate} Hz)",
            samples.len(),
        );
    }

    let detection_config = DetectionConfig::default().with_sample_rate(sample_rate);
    let onsets = detect_onsets(&samples, &detection_config);

    if config.verbosity.is_verbose() {
        eprintln!("Detected {} audio onsets", onsets.len());
    }

    Ok(compare_edl_to_onsets(edl, &onsets, tolerance_ms))
}

/// Load and parse an EDL JSON file.
fn load_edl(path: &Path) -> CliResult<EditDecisionList> {
    let content = std::fs::read_to_string(path).map_err(|e| {
        CliError::invalid_argument(format!("Failed to read EDL file {}: {e}", path.display()))
    })?;
    serde_json::from_str(&content).map_err(|e| {
        CliError::invalid_argument(format!("Failed to parse EDL file {}: {e}", path.display()))
    })
}

/// Find all .edl.json files in a directory.
fn find_edl_files(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().is_some_and(|e| e == "json") {
                if let Some(stem) = path.file_stem() {
                    if stem.to_string_lossy().ends_with(".edl") {
                        files.push(path);
                    }
                }
            }
        }
    }
    files.sort();
    files
}

/// Find the video file corresponding to an EDL file.
///
/// Convention: `video.edl.json` -> `video.mp4` or `video.mov`
fn find_video_for_edl(edl_path: &Path) -> Option<std::path::PathBuf> {
    let parent = edl_path.parent()?;
    let stem = edl_path.file_stem()?; // "video.edl"
    let video_stem = stem.to_string_lossy().strip_suffix(".edl")?.to_string();

    for ext in &["mp4", "mov", "mkv", "webm"] {
        let candidate = parent.join(format!("{video_stem}.{ext}"));
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Render a text report for an AV sync check.
#[must_use]
pub fn render_text_report(
    report: &jugar_probar::av_sync::AvSyncReport,
    tolerance_ms: f64,
    detailed: bool,
) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "AV Sync: {} (tolerance: {tolerance_ms}ms)\n",
        report.video_id
    ));

    if detailed {
        for segment in &report.segments {
            out.push_str(&format!("  {}:\n", segment.segment));
            for tick in &segment.ticks {
                let status = if tick.passed { "PASS" } else { "FAIL" };
                match (tick.actual_secs, tick.delta_ms) {
                    (Some(actual), Some(delta)) => {
                        out.push_str(&format!(
                            "    bullet[{}]: declared={:.3}s actual={:.3}s delta={:.1}ms {status}\n",
                            tick.bullet_index, tick.declared_secs, actual, delta
                        ));
                    }
                    _ => {
                        out.push_str(&format!(
                            "    bullet[{}]: declared={:.3}s actual=NONE {status}\n",
                            tick.bullet_index, tick.declared_secs
                        ));
                    }
                }
            }
        }
    }

    out.push_str(&format!(
        "Verdict: {} ({}/{} ticks passed, max drift: {:.1}ms)\n",
        report.verdict, report.matched_ticks, report.total_ticks, report.max_delta_ms
    ));

    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use jugar_probar::av_sync::{AvSyncReport, SegmentSyncResult, TickDelta};

    fn sample_pass_report() -> AvSyncReport {
        AvSyncReport {
            video_id: "demo-bench".to_string(),
            verdict: SyncVerdict::Pass,
            segments: vec![SegmentSyncResult {
                segment: "P2-key_terms".to_string(),
                ticks: vec![TickDelta {
                    segment: "P2-key_terms".to_string(),
                    bullet_index: 0,
                    declared_secs: 1.7,
                    actual_secs: Some(1.71),
                    delta_ms: Some(10.0),
                    passed: true,
                }],
                all_passed: true,
            }],
            total_ticks: 1,
            matched_ticks: 1,
            coverage_pct: 100.0,
            max_delta_ms: 10.0,
            mean_delta_ms: 10.0,
        }
    }

    fn sample_fail_report() -> AvSyncReport {
        AvSyncReport {
            video_id: "demo-bench".to_string(),
            verdict: SyncVerdict::Fail,
            segments: vec![SegmentSyncResult {
                segment: "P2-key_terms".to_string(),
                ticks: vec![TickDelta {
                    segment: "P2-key_terms".to_string(),
                    bullet_index: 0,
                    declared_secs: 1.7,
                    actual_secs: Some(1.408),
                    delta_ms: Some(-292.0),
                    passed: false,
                }],
                all_passed: false,
            }],
            total_ticks: 1,
            matched_ticks: 0,
            coverage_pct: 0.0,
            max_delta_ms: 292.0,
            mean_delta_ms: 292.0,
        }
    }

    #[test]
    fn test_render_text_report_pass() {
        let report = sample_pass_report();
        let text = render_text_report(&report, 20.0, false);
        assert!(text.contains("AV Sync: demo-bench"));
        assert!(text.contains("PASS"));
        assert!(text.contains("1/1 ticks passed"));
    }

    #[test]
    fn test_render_text_report_fail_detailed() {
        let report = sample_fail_report();
        let text = render_text_report(&report, 20.0, true);
        assert!(text.contains("FAIL"));
        assert!(text.contains("bullet[0]"));
        assert!(text.contains("-292.0ms"));
        assert!(text.contains("0/1 ticks passed"));
    }

    #[test]
    fn test_render_text_report_no_details() {
        let report = sample_pass_report();
        let text = render_text_report(&report, 20.0, false);
        assert!(!text.contains("bullet[0]")); // No per-tick details
        assert!(text.contains("Verdict:"));
    }

    #[test]
    fn test_render_text_report_no_match() {
        let report = AvSyncReport {
            video_id: "test".to_string(),
            verdict: SyncVerdict::Fail,
            segments: vec![SegmentSyncResult {
                segment: "seg".to_string(),
                ticks: vec![TickDelta {
                    segment: "seg".to_string(),
                    bullet_index: 0,
                    declared_secs: 1.7,
                    actual_secs: None,
                    delta_ms: None,
                    passed: false,
                }],
                all_passed: false,
            }],
            total_ticks: 1,
            matched_ticks: 0,
            coverage_pct: 0.0,
            max_delta_ms: 0.0,
            mean_delta_ms: 0.0,
        };
        let text = render_text_report(&report, 20.0, true);
        assert!(text.contains("actual=NONE"));
    }

    #[test]
    fn test_find_edl_files_empty() {
        let temp = tempfile::TempDir::new().unwrap();
        let files = find_edl_files(temp.path());
        assert!(files.is_empty());
    }

    #[test]
    fn test_find_edl_files_with_edl() {
        let temp = tempfile::TempDir::new().unwrap();
        std::fs::write(temp.path().join("video.edl.json"), "{}").unwrap();
        std::fs::write(temp.path().join("other.json"), "{}").unwrap();
        let files = find_edl_files(temp.path());
        assert_eq!(files.len(), 1);
        assert!(files[0].to_string_lossy().contains("video.edl.json"));
    }

    #[test]
    fn test_find_video_for_edl_mp4() {
        let temp = tempfile::TempDir::new().unwrap();
        let edl = temp.path().join("demo.edl.json");
        let video = temp.path().join("demo.mp4");
        std::fs::write(&edl, "{}").unwrap();
        std::fs::write(&video, "").unwrap();
        let found = find_video_for_edl(&edl);
        assert_eq!(found, Some(video));
    }

    #[test]
    fn test_find_video_for_edl_mov() {
        let temp = tempfile::TempDir::new().unwrap();
        let edl = temp.path().join("demo.edl.json");
        let video = temp.path().join("demo.mov");
        std::fs::write(&edl, "{}").unwrap();
        std::fs::write(&video, "").unwrap();
        let found = find_video_for_edl(&edl);
        assert_eq!(found, Some(video));
    }

    #[test]
    fn test_find_video_for_edl_none() {
        let temp = tempfile::TempDir::new().unwrap();
        let edl = temp.path().join("demo.edl.json");
        std::fs::write(&edl, "{}").unwrap();
        let found = find_video_for_edl(&edl);
        assert!(found.is_none());
    }

    #[test]
    fn test_load_edl_valid() {
        let temp = tempfile::TempDir::new().unwrap();
        let edl_path = temp.path().join("test.edl.json");
        std::fs::write(&edl_path, r#"{"video_id":"test","decisions":[]}"#).unwrap();
        let edl = load_edl(&edl_path).unwrap();
        assert_eq!(edl.video_id, "test");
    }

    #[test]
    fn test_load_edl_invalid_json() {
        let temp = tempfile::TempDir::new().unwrap();
        let edl_path = temp.path().join("bad.edl.json");
        std::fs::write(&edl_path, "not json").unwrap();
        let result = load_edl(&edl_path);
        assert!(result.is_err());
    }

    #[test]
    fn test_load_edl_missing_file() {
        let result = load_edl(Path::new("/nonexistent/file.edl.json"));
        assert!(result.is_err());
    }
}
