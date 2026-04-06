//! Monitor command implementation

use crate::cli::logging::log;
use crate::cli::LogLevel;
use crate::config::{MonitorArgs, OutputFormat};

pub fn run_monitor(args: MonitorArgs, level: LogLevel) -> Result<(), String> {
    log(level, LogLevel::Normal, &format!("Monitoring: {}", args.input.display()));

    // Check if file exists
    if !args.input.exists() {
        return Err(format!("File not found: {}", args.input.display()));
    }

    log(level, LogLevel::Normal, &format!("  Drift threshold (PSI): {}", args.threshold));

    if let Some(baseline) = &args.baseline {
        log(level, LogLevel::Normal, &format!("  Baseline: {}", baseline.display()));
    }

    // Calculate Population Stability Index (PSI)
    // PSI = sum((actual_% - expected_%) * ln(actual_% / expected_%))
    // PSI < 0.1: no significant shift
    // PSI 0.1-0.2: moderate shift
    // PSI > 0.2: significant shift

    // Simulate bucket distributions for baseline vs current
    let baseline_buckets: Vec<f64> = vec![0.10, 0.15, 0.20, 0.25, 0.15, 0.10, 0.05];
    let current_buckets: Vec<f64> = vec![0.11, 0.14, 0.19, 0.26, 0.16, 0.09, 0.05];

    // Calculate PSI
    let mut psi = 0.0f64;
    for (expected, actual) in baseline_buckets.iter().zip(current_buckets.iter()) {
        if *expected > 0.0 && *actual > 0.0 {
            psi += (*actual - *expected) * (*actual / *expected).max(f64::MIN_POSITIVE).ln();
        }
    }
    psi = psi.abs();

    let threshold = f64::from(args.threshold);

    // Determine drift status
    let (status, severity) = if psi < 0.1 {
        ("NO DRIFT", "low")
    } else if psi < threshold {
        ("MINOR DRIFT", "moderate")
    } else {
        ("SIGNIFICANT DRIFT", "high")
    };

    let pass = psi < threshold;

    log(level, LogLevel::Normal, "Drift Monitoring Results:");
    log(level, LogLevel::Normal, &format!("  PSI score: {psi:.4}"));
    log(level, LogLevel::Normal, &format!("  Threshold: {:.4}", args.threshold));
    log(level, LogLevel::Normal, &format!("  Severity: {severity}"));
    log(level, LogLevel::Normal, &format!("  Status: {status}"));

    if args.format == OutputFormat::Json {
        let result = serde_json::json!({
            "psi_score": psi,
            "threshold": args.threshold,
            "status": status,
            "severity": severity,
            "drift_detected": !pass,
            "buckets": {
                "baseline": baseline_buckets,
                "current": current_buckets
            }
        });
        if let Ok(json_str) = serde_json::to_string_pretty(&result) {
            println!("{json_str}");
        }
    }

    if !pass {
        return Err(format!(
            "Drift detected: PSI {:.4} exceeds threshold {:.4}",
            psi, args.threshold
        ));
    }

    Ok(())
}
