//! JSON export for profile data. Spec section 10.

use super::catalog::FullProfile;
use anyhow::Result;
use std::path::Path;

/// Export a profile to JSON file.
pub fn export_json(profile: &FullProfile, path: &Path) -> Result<()> {
    let json = serde_json::to_string_pretty(profile)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a profile from JSON file.
pub fn load_json(path: &Path) -> Result<FullProfile> {
    let content = std::fs::read_to_string(path)?;
    let profile: FullProfile = serde_json::from_str(&content)?;
    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::catalog::{ThroughputMetrics, TimingMetrics};

    #[test]
    fn test_export_load_roundtrip() {
        let profile = FullProfile {
            version: "2.0".to_string(),
            timestamp: "2026-04-04T12:00:00Z".to_string(),
            timing: TimingMetrics { wall_clock_time_us: 23.2, samples: 50, ..Default::default() },
            throughput: ThroughputMetrics { tflops: 11.6, ..Default::default() },
            ..Default::default()
        };

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_profile.json");
        export_json(&profile, &path).unwrap();
        let loaded = load_json(&path).unwrap();
        assert_eq!(loaded.version, "2.0");
        assert!((loaded.timing.wall_clock_time_us - 23.2).abs() < 0.01);
    }

    /// FALSIFY-CGP-062: diff must not require re-profiling.
    /// Loading two saved JSONs should be < 100ms.
    #[test]
    fn test_json_load_speed() {
        let profile = FullProfile { version: "2.0".to_string(), ..Default::default() };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("speed_test.json");
        export_json(&profile, &path).unwrap();

        let start = std::time::Instant::now();
        let _ = load_json(&path).unwrap();
        let _ = load_json(&path).unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed.as_millis() < 100,
            "Loading two JSONs took {}ms (should be <100ms)",
            elapsed.as_millis()
        );
    }
}
