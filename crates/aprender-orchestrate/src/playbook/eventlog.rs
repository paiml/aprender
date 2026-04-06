//! Append-only JSONL event log for playbook execution (PB-005)
//!
//! Each playbook run appends events to a `.events.jsonl` file alongside the
//! playbook YAML. Events are timestamped and tagged with a run ID.

use super::types::{PipelineEvent, TimestampedEvent};
use anyhow::{Context, Result};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Derive the event log path from a playbook path: `.yaml` → `.events.jsonl`
pub fn event_log_path(playbook_path: &Path) -> PathBuf {
    let stem = playbook_path.file_stem().unwrap_or_default().to_string_lossy();
    playbook_path.with_file_name(format!("{}.events.jsonl", stem))
}

/// Generate a unique run ID: `"r-{short_hex}"`
pub fn generate_run_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();

    // Combine timestamp nanos with process ID for uniqueness
    let seed = now.as_nanos() ^ (std::process::id() as u128);
    format!("r-{:012x}", seed & 0xFFFF_FFFF_FFFF)
}

/// Get the current UTC timestamp in ISO 8601 format
pub fn now_iso8601() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Append a pipeline event to the event log
pub fn append_event(playbook_path: &Path, event: PipelineEvent) -> Result<()> {
    let path = event_log_path(playbook_path);
    let timestamped = TimestampedEvent { ts: now_iso8601(), event };

    let json = serde_json::to_string(&timestamped).context("failed to serialize event")?;

    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("failed to open event log: {}", path.display()))?;

    writeln!(file, "{}", json).context("failed to write event")?;

    Ok(())
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;

    #[test]
    fn test_PB005_event_log_path() {
        let path = event_log_path(Path::new("/tmp/pipeline.yaml"));
        assert_eq!(path, PathBuf::from("/tmp/pipeline.events.jsonl"));
    }

    #[test]
    fn test_PB005_generate_run_id_format() {
        let id = generate_run_id();
        assert!(id.starts_with("r-"));
        assert!(id.len() > 2);
    }

    #[test]
    fn test_PB005_generate_run_id_unique() {
        let id1 = generate_run_id();
        // Small sleep to ensure different nanos
        std::thread::sleep(std::time::Duration::from_millis(1));
        let id2 = generate_run_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_PB005_append_event() {
        let dir = tempfile::tempdir().expect("tempdir creation failed");
        let playbook_path = dir.path().join("test.yaml");

        append_event(
            &playbook_path,
            PipelineEvent::RunStarted {
                playbook: "test".to_string(),
                run_id: "r-abc123".to_string(),
                batuta_version: "0.6.5".to_string(),
            },
        )
        .expect("unexpected failure");

        append_event(
            &playbook_path,
            PipelineEvent::StageCached {
                stage: "hello".to_string(),
                cache_key: "blake3:key".to_string(),
                reason: "cache_key matches".to_string(),
            },
        )
        .expect("unexpected failure");

        let log_path = event_log_path(&playbook_path);
        let content = std::fs::read_to_string(&log_path).expect("fs read failed");
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 2);

        // Parse each line as valid JSON
        let event1: TimestampedEvent =
            serde_json::from_str(lines[0]).expect("json deserialize failed");
        assert!(matches!(event1.event, PipelineEvent::RunStarted { .. }));

        let event2: TimestampedEvent =
            serde_json::from_str(lines[1]).expect("json deserialize failed");
        assert!(matches!(event2.event, PipelineEvent::StageCached { .. }));
    }

    #[test]
    fn test_PB005_now_iso8601_format() {
        let ts = now_iso8601();
        // Should match ISO 8601 pattern
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
    }
}
