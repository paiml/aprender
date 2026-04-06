//! Self-drift detection and enforcement for the batuta CLI.
//!
//! Checks whether batuta's own published dependencies are up to date.
//! Only warns about batuta itself — not the entire ecosystem.
//! Use `batuta stack drift` for full ecosystem analysis.

use tracing::warn;

use crate::ansi_colors::Colorize;
use crate::stack;

/// Check if the environment requests strict mode (BATUTA_STRICT=1).
fn is_strict_env() -> bool {
    std::env::var("BATUTA_STRICT")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Format self-drift as a concise warning (non-blocking).
fn format_self_drift_warning(batuta_version: &str, drifts: &[stack::DriftReport]) -> String {
    let mut output = String::new();
    output.push_str(&format!(
        "{}\n\n",
        format!("⚠️  batuta {} has outdated dependencies", batuta_version).bright_yellow().bold()
    ));

    for drift in drifts {
        output.push_str(&format!(
            "   {} {} → {}\n",
            drift.dependency, drift.uses_version, drift.latest_version
        ));
    }

    output.push_str(&format!("\n{}", "Update: cargo install batuta\n".dimmed()));

    output
}

/// Format self-drift as a blocking error.
fn format_self_drift_error(batuta_version: &str, drifts: &[stack::DriftReport]) -> String {
    let mut output = String::new();
    output.push_str(&format!("🔴 batuta {} has outdated dependencies\n\n", batuta_version));

    for drift in drifts {
        output.push_str(&format!(
            "   {} {} → {}\n",
            drift.dependency, drift.uses_version, drift.latest_version
        ));
    }

    output.push_str("\nUpdate: cargo install batuta\n");
    output.push_str("Or use --allow-drift to bypass.\n");

    output
}

/// Check batuta's own published dependencies for drift.
///
/// Returns `None` if check cannot be performed (offline, etc.)
/// Returns `Some(empty)` if no drift detected
/// Returns `Some(drifts)` if drift detected
fn check_self_drift() -> anyhow::Result<Option<(String, Vec<stack::DriftReport>)>> {
    let rt = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return Ok(None),
    };

    rt.block_on(async {
        let mut client = stack::CratesIoClient::new().with_persistent_cache();
        let mut checker = stack::DriftChecker::new();

        match checker.detect_self_drift(&mut client).await {
            Ok(drifts) => {
                let version = checker
                    .latest_versions()
                    .get("batuta")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| env!("CARGO_PKG_VERSION").to_string());
                Ok(Some((version, drifts)))
            }
            Err(_) => Ok(None),
        }
    })
}

/// Get the drift warning marker file path (workspace-scoped).
/// Uses workspace root hash to scope warnings per project.
fn drift_marker_path() -> std::path::PathBuf {
    let workspace_id = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.bytes().map(|b| b as u64).sum::<u64>() % 100000))
        .unwrap_or(0);
    std::env::temp_dir().join(format!("batuta-drift-shown-{}", workspace_id))
}

/// Check if drift warning was already shown this session.
fn drift_already_shown() -> bool {
    let marker = drift_marker_path();
    if marker.exists() {
        if let Ok(meta) = std::fs::metadata(&marker) {
            if let Ok(modified) = meta.modified() {
                if let Ok(elapsed) = modified.elapsed() {
                    return elapsed.as_secs() < 3600; // 1 hour
                }
            }
        }
    }
    false
}

/// Mark drift warning as shown for this session.
fn mark_drift_shown() {
    let _ = std::fs::write(drift_marker_path(), "shown");
}

/// Enforce self-drift checking with smart tolerance.
///
/// Only checks batuta's own dependencies — not the full ecosystem.
/// Default: shows warning ONCE per hour, never blocks.
/// With --strict or BATUTA_STRICT=1: always blocks on drift.
pub(crate) fn enforce_drift_check(strict: bool) -> anyhow::Result<()> {
    let strict_mode = strict || is_strict_env();

    // Only check once per hour unless strict mode (Muda elimination)
    if !strict_mode && drift_already_shown() {
        return Ok(());
    }

    let Some((version, drifts)) = check_self_drift()? else {
        return Ok(());
    };
    if drifts.is_empty() {
        return Ok(());
    }

    if strict_mode {
        eprintln!("{}", format_self_drift_error(&version, &drifts));
        std::process::exit(1);
    } else if !drift_already_shown() {
        warn!("batuta has outdated dependencies (non-blocking)");
        eprintln!("{}", format_self_drift_warning(&version, &drifts));
        mark_drift_shown();
    }

    Ok(())
}
