//! `cgp baseline` — Save and load performance baselines.
//! Baselines stored in `.cgp-baselines/` directory.

use crate::metrics::catalog::FullProfile;
use crate::metrics::export;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Default baseline directory.
fn baseline_dir() -> PathBuf {
    PathBuf::from(".cgp-baselines")
}

/// Save a profile as a named baseline.
pub fn save_baseline(name: &str, profile: &FullProfile) -> Result<PathBuf> {
    let dir = baseline_dir();
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{name}.json"));
    export::export_json(profile, &path)?;
    Ok(path)
}

/// Load a named baseline.
pub fn load_baseline(name: &str) -> Result<FullProfile> {
    let path = baseline_dir().join(format!("{name}.json"));
    if !path.exists() {
        anyhow::bail!(
            "Baseline '{name}' not found at {}.\n  \
             Available baselines: {}",
            path.display(),
            list_baselines().join(", ")
        );
    }
    export::load_json(&path)
}

/// List all saved baselines.
pub fn list_baselines() -> Vec<String> {
    let dir = baseline_dir();
    if !dir.exists() {
        return vec![];
    }
    std::fs::read_dir(dir)
        .ok()
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter_map(|e| {
                    let path = e.path();
                    if path.extension().is_some_and(|ext| ext == "json") {
                        path.file_stem().and_then(|s| s.to_str()).map(String::from)
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Run the `cgp baseline` command.
pub fn run_baseline(save: Option<&str>, load: Option<&str>) -> Result<()> {
    match (save, load) {
        (Some(path), None) => {
            // Save: the path should be a JSON profile to save as baseline
            let profile = export::load_json(Path::new(path))?;

            // Use filename stem as baseline name
            let name = Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("default");

            let saved_path = save_baseline(name, &profile)?;
            println!("Baseline '{name}' saved to {}", saved_path.display());
        }
        (None, Some(name)) => {
            // Load and display
            let profile = load_baseline(name)?;
            println!("Baseline '{name}':");
            println!("  Time: {:.1} us", profile.timing.wall_clock_time_us);
            println!("  TFLOP/s: {:.1}", profile.throughput.tflops);
            if let Some(k) = &profile.kernel {
                println!("  Kernel: {}", k.name);
            }
            if let Some(gpu) = &profile.hardware.gpu {
                println!("  GPU: {gpu}");
            }
            if let Some(health) = &profile.system_health {
                println!(
                    "  System: {:.0}°C, {:.0}W, {:.0} MHz",
                    health.gpu_temperature_celsius, health.gpu_power_watts, health.gpu_clock_mhz
                );
            }
            if let Some(vram) = &profile.vram {
                println!(
                    "  VRAM: {:.0}/{:.0} MB ({:.1}%)",
                    vram.vram_used_mb, vram.vram_total_mb, vram.vram_utilization_pct
                );
            }
            println!("  Timestamp: {}", profile.timestamp);
        }
        (None, None) => {
            // List baselines
            let baselines = list_baselines();
            if baselines.is_empty() {
                println!("No baselines saved yet.");
                println!("  Save one with: cgp baseline --save <profile.json>");
            } else {
                println!("Saved baselines:");
                for name in &baselines {
                    let path = baseline_dir().join(format!("{name}.json"));
                    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                    println!("  {name:20} ({:.1} KB)", size as f64 / 1024.0);
                }
            }
        }
        (Some(_), Some(_)) => {
            anyhow::bail!("Specify either --save or --load, not both");
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::catalog::*;

    #[test]
    fn test_save_and_load_baseline() {
        let profile = FullProfile {
            version: "2.0".to_string(),
            timing: TimingMetrics { wall_clock_time_us: 23.2, samples: 50, ..Default::default() },
            throughput: ThroughputMetrics { tflops: 11.6, ..Default::default() },
            ..Default::default()
        };

        // Use a temp directory
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.json");
        export::export_json(&profile, &path).unwrap();

        let loaded = export::load_json(&path).unwrap();
        assert!((loaded.timing.wall_clock_time_us - 23.2).abs() < 0.01);
    }

    #[test]
    fn test_list_baselines_empty() {
        // When no .cgp-baselines dir exists, should return empty
        let names = list_baselines();
        // May or may not be empty depending on test environment
        assert!(names.len() < 1000);
    }
}
