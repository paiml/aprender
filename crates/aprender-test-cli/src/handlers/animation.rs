//! Animation verification command handler.
//!
//! Orchestrates: parse timeline -> parse observed -> verify -> render report.

use crate::commands::{AnimationCheckArgs, OutputFormat};
use crate::config::CliConfig;
use crate::error::{CliError, CliResult};
use jugar_probar::animation::{
    verify_timeline, AnimationTimeline, AnimationVerdict, ObservedEvent,
};

/// Execute the animation check command.
pub fn execute_check(config: &CliConfig, args: &AnimationCheckArgs) -> CliResult<()> {
    ensure_exists(&args.timeline, "Timeline file not found")?;
    ensure_exists(&args.observed, "Observed events file not found")?;

    if config.verbosity.is_verbose() {
        println!("Verifying animation timeline: {}", args.timeline.display());
    }

    let timeline = load_timeline(&args.timeline)?;
    let observed_events = load_observed_events(&args.observed)?;
    let report = verify_timeline(&timeline, &observed_events, args.tolerance_ms);

    emit_report(args, &report)?;

    if report.verdict == AnimationVerdict::Pass {
        Ok(())
    } else {
        Err(CliError::test_execution(format!(
            "Animation verification failed: {} ({}/{} events passed)",
            report.video_id, report.verified_events, report.total_events
        )))
    }
}

fn ensure_exists(path: &std::path::Path, what: &str) -> CliResult<()> {
    if path.exists() {
        Ok(())
    } else {
        Err(CliError::invalid_argument(format!(
            "{what}: {}",
            path.display()
        )))
    }
}

fn load_timeline(path: &std::path::Path) -> CliResult<AnimationTimeline> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| CliError::test_execution(format!("Failed to read timeline: {e}")))?;
    serde_json::from_str(&body)
        .map_err(|e| CliError::test_execution(format!("Failed to parse timeline JSON: {e}")))
}

fn load_observed_events(path: &std::path::Path) -> CliResult<Vec<ObservedEvent>> {
    let body = std::fs::read_to_string(path)
        .map_err(|e| CliError::test_execution(format!("Failed to read observed events: {e}")))?;
    let raw: Vec<ObservedEventJson> = serde_json::from_str(&body).map_err(|e| {
        CliError::test_execution(format!("Failed to parse observed events JSON: {e}"))
    })?;
    Ok(raw
        .into_iter()
        .map(|o| ObservedEvent {
            name: o.name,
            time_secs: o.time_secs,
        })
        .collect())
}

fn emit_report(
    args: &AnimationCheckArgs,
    report: &jugar_probar::animation::AnimationReport,
) -> CliResult<()> {
    match args.format {
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&report)
                .map_err(|e| CliError::test_execution(format!("JSON serialization failed: {e}")))?;
            println!("{json}");
        }
        OutputFormat::Text => render_text_report(report, args.tolerance_ms),
    }
    Ok(())
}

/// JSON-deserializable observed event (matches the file format).
#[derive(serde::Deserialize)]
struct ObservedEventJson {
    name: String,
    time_secs: f64,
}

fn render_text_report(report: &jugar_probar::animation::AnimationReport, tolerance_ms: f64) {
    println!(
        "Animation: {} (tolerance: {:.0}ms)",
        report.video_id, tolerance_ms
    );
    for event in &report.events {
        let status = if event.passed { "PASS" } else { "FAIL" };
        match (event.actual_secs, event.delta_ms) {
            (Some(actual), Some(delta)) => {
                println!(
                    "  {}: expected={:.3}s actual={:.3}s delta={:.1}ms {}",
                    event.name, event.expected_secs, actual, delta, status
                );
            }
            _ => {
                println!(
                    "  {}: expected={:.3}s actual=MISSING {}",
                    event.name, event.expected_secs, status
                );
            }
        }
    }
    println!(
        "Verdict: {} ({}/{} events, max delta: {:.1}ms, mean delta: {:.1}ms)",
        report.verdict,
        report.verified_events,
        report.total_events,
        report.max_delta_ms,
        report.mean_delta_ms
    );
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use jugar_probar::animation::AnimationEventType;
    use jugar_probar::animation::{AnimationReport, AnimationVerdict, EventResult};

    fn sample_report() -> AnimationReport {
        AnimationReport {
            video_id: "test".to_string(),
            verdict: AnimationVerdict::Pass,
            events: vec![EventResult {
                name: "land_0".to_string(),
                event_type: AnimationEventType::PhysicsEvent,
                expected_secs: 1.7,
                actual_secs: Some(1.71),
                delta_ms: Some(10.0),
                passed: true,
            }],
            total_events: 1,
            verified_events: 1,
            max_delta_ms: 10.0,
            mean_delta_ms: 10.0,
        }
    }

    #[test]
    fn test_render_text_report_pass() {
        let report = sample_report();
        render_text_report(&report, 20.0);
    }

    #[test]
    fn test_render_text_report_missing_event() {
        let mut report = sample_report();
        report.verdict = AnimationVerdict::Fail;
        report.events.push(EventResult {
            name: "land_1".to_string(),
            event_type: AnimationEventType::PhysicsEvent,
            expected_secs: 2.4,
            actual_secs: None,
            delta_ms: None,
            passed: false,
        });
        report.total_events = 2;
        render_text_report(&report, 20.0);
    }

    #[test]
    fn test_execute_check_missing_timeline() {
        let config = CliConfig::new();
        let args = AnimationCheckArgs {
            timeline: std::path::PathBuf::from("/nonexistent/timeline.json"),
            observed: std::path::PathBuf::from("/nonexistent/observed.json"),
            tolerance_ms: 20.0,
            format: OutputFormat::Text,
        };
        let result = execute_check(&config, &args);
        assert!(result.is_err());
    }

    #[test]
    fn test_observed_event_json_deserialization() {
        let json = r#"[{"name": "land_0", "time_secs": 1.71}]"#;
        let events: Vec<ObservedEventJson> = serde_json::from_str(json).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "land_0");
    }
}
