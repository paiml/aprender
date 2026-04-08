//! `apr diagnose` — Automated Five Whys for training checkpoints.
//!
//! Reads checkpoint metadata, optionally runs evaluation, and performs
//! automated root cause analysis for training failures.

use std::path::Path;

use colored::Colorize;

use crate::{error::CliError, output};

type Result<T> = std::result::Result<T, CliError>;

/// Run automated diagnosis on a training checkpoint.
#[provable_contracts_macros::contract("apr-cli-operations-v1", equation = "side_effect_classification")]
pub(crate) fn run(
    checkpoint_dir: &Path,
    data_path: Option<&Path>,
    model_size: Option<&str>,
    num_classes: usize,
    json_output: bool,
) -> Result<()> {
    if !checkpoint_dir.is_dir() {
        return Err(CliError::ValidationFailed(format!(
            "Checkpoint directory not found: {}",
            checkpoint_dir.display()
        )));
    }

    let mut findings: Vec<Finding> = Vec::new();
    let mut recommendations: Vec<Recommendation> = Vec::new();

    check_checkpoint_integrity(checkpoint_dir, &mut findings, &mut recommendations)?;
    let epoch_metrics = check_loss_curve(
        checkpoint_dir,
        num_classes,
        &mut findings,
        &mut recommendations,
    );
    #[cfg(feature = "training")]
    let eval_report = run_evaluation(
        checkpoint_dir,
        data_path,
        model_size,
        num_classes,
        &mut findings,
    )?;
    #[cfg(not(feature = "training"))]
    let eval_report: Option<serde_json::Value> = None;
    check_data_quality(data_path, &mut findings, &mut recommendations);
    generate_recommendations(&findings, &mut recommendations);

    if json_output {
        output_json(
            checkpoint_dir,
            &findings,
            &recommendations,
            &epoch_metrics,
            eval_report.as_ref(),
        );
        return Ok(());
    }

    output_text(&findings, &epoch_metrics, recommendations);

    Ok(())
}

// ── WHY 1: Checkpoint integrity ──────────────────────────────────────────────

fn check_checkpoint_integrity(
    checkpoint_dir: &Path,
    findings: &mut Vec<Finding>,
    recommendations: &mut Vec<Recommendation>,
) -> Result<()> {
    let expected_files = [
        "metadata.json",
        "model.safetensors",
        "config.json",
        "adapter_config.json",
    ];
    let mut missing_files: Vec<&str> = Vec::new();
    for f in &expected_files {
        if !checkpoint_dir.join(f).exists() {
            missing_files.push(f);
        }
    }
    if !missing_files.is_empty() {
        findings.push(Finding {
            category: "Checkpoint Integrity",
            severity: Severity::Error,
            message: format!("Missing files: {}", missing_files.join(", ")),
        });
    }

    // Load metadata.json
    let meta_path = checkpoint_dir.join("metadata.json");
    let metadata: Option<serde_json::Value> = if meta_path.exists() {
        let meta_str = std::fs::read_to_string(&meta_path).map_err(|e| {
            CliError::ValidationFailed(format!("Failed to read metadata.json: {e}"))
        })?;
        Some(
            serde_json::from_str(&meta_str)
                .map_err(|e| CliError::ValidationFailed(format!("Invalid metadata.json: {e}")))?,
        )
    } else {
        findings.push(Finding {
            category: "Checkpoint Integrity",
            severity: Severity::Error,
            message: "metadata.json not found — cannot analyze training metrics".to_string(),
        });
        None
    };

    // Check class_weights saved
    let has_class_weights = metadata
        .as_ref()
        .and_then(|m| m.get("class_weights"))
        .is_some_and(|v| !v.is_null());

    if !has_class_weights {
        findings.push(Finding {
            category: "Checkpoint Integrity",
            severity: Severity::Warning,
            message: "class_weights NOT saved in metadata.json — eval may use different weights than training".to_string(),
        });
        recommendations.push(Recommendation {
            priority: "P0",
            action: "Fix: Save class_weights in checkpoint metadata (entrenar bug fix)".to_string(),
        });
    }

    Ok(())
}

include!("diagnose_analysis.rs");

// ── WHY 4: Data quality (via alimentar if data available) ────────────────────

fn check_data_quality(
    data_path: Option<&Path>,
    findings: &mut Vec<Finding>,
    recommendations: &mut Vec<Recommendation>,
) {
    let Some(data) = data_path else {
        return;
    };

    if !data.exists() {
        return;
    }

    if let Ok(dataset) = alimentar::ArrowDataset::from_json(data) {
        let imbalance = alimentar::imbalance::ImbalanceDetector::new("label").analyze(&dataset);
        if let Ok(report) = imbalance {
            if report.metrics.imbalance_ratio > 5.0 {
                findings.push(Finding {
                    category: "Data Quality",
                    severity: Severity::Warning,
                    message: format!(
                        "Class imbalance {:.1}:1 in test data",
                        report.metrics.imbalance_ratio
                    ),
                });
                recommendations.push(Recommendation {
                    priority: "P1",
                    action: "Use stratified train/val/test split (apr data split)".to_string(),
                });
            }
        }
    }
}

// ── Generate recommendations ─────────────────────────────────────────────────

fn generate_recommendations(findings: &[Finding], recommendations: &mut Vec<Recommendation>) {
    let has_collapse = findings.iter().any(|f| f.category == "Prediction Collapse");
    if has_collapse {
        recommendations.push(Recommendation {
            priority: "P0",
            action: "Retrain with stratified split and verified class_weights".to_string(),
        });
    }

    if findings
        .iter()
        .any(|f| f.category == "Loss Curve" && f.severity == Severity::Error)
    {
        recommendations.push(Recommendation {
            priority: "P1",
            action: "Use LR finder to validate learning rate".to_string(),
        });
    }
}

// ── Output: JSON ─────────────────────────────────────────────────────────────

fn output_json(
    checkpoint_dir: &Path,
    findings: &[Finding],
    recommendations: &[Recommendation],
    epoch_metrics: &[EpochInfo],
    eval_report: Option<&serde_json::Value>,
) {
    #[allow(clippy::disallowed_methods)] // serde_json::json!() macro uses infallible unwrap
    let report = serde_json::json!({
        "checkpoint": checkpoint_dir.display().to_string(),
        "findings": findings.iter().map(|f| serde_json::json!({
            "category": f.category,
            "severity": format!("{:?}", f.severity),
            "message": f.message,
        })).collect::<Vec<_>>(),
        "recommendations": recommendations.iter().map(|r| serde_json::json!({
            "priority": r.priority,
            "action": r.action,
        })).collect::<Vec<_>>(),
        "epoch_metrics": epoch_metrics.iter().map(|e| serde_json::json!({
            "epoch": e.epoch + 1,
            "train_loss": e.train_loss,
            "val_loss": e.val_loss,
            "val_accuracy": e.val_accuracy,
        })).collect::<Vec<_>>(),
        "eval_report": eval_report,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_default()
    );
}

// ── Output: text ─────────────────────────────────────────────────────────────

fn output_text(
    findings: &[Finding],
    epoch_metrics: &[EpochInfo],
    recommendations: Vec<Recommendation>,
) {
    output::header("SSC Training Diagnosis (Five Whys)");
    println!();

    let mut why_num = 1;
    let categories_in_order = [
        "Accuracy",
        "Prediction Collapse",
        "Loss Curve",
        "Checkpoint Integrity",
        "Data Quality",
        "Calibration",
        "Evaluation",
        "Data",
    ];

    for cat in &categories_in_order {
        let cat_findings: Vec<_> = findings.iter().filter(|f| f.category == *cat).collect();
        if cat_findings.is_empty() {
            continue;
        }

        let severity_icon = match cat_findings
            .iter()
            .map(|f| f.severity)
            .max()
            .unwrap_or(Severity::Info)
        {
            Severity::Error => "!!".red().bold(),
            Severity::Warning => "! ".yellow().bold(),
            Severity::Info => "i ".blue(),
        };

        println!("{}  WHY {why_num}: {}", severity_icon, cat.white().bold());
        for f in cat_findings {
            println!("     {}", f.message);
        }
        println!();
        why_num += 1;
    }

    // Loss curve table
    if !epoch_metrics.is_empty() {
        println!("{}", "Epoch History:".white().bold());
        for e in epoch_metrics {
            let min_val_loss = epoch_metrics
                .iter()
                .map(|x| x.val_loss)
                .fold(f64::MAX, f64::min);
            let marker = if (e.val_loss - min_val_loss).abs() < f64::EPSILON {
                " <- BEST".green().to_string()
            } else {
                String::new()
            };
            println!(
                "  Epoch {:>2}: train_loss={:.4}  val_loss={:.4}  val_acc={:.1}%{marker}",
                e.epoch + 1,
                e.train_loss,
                e.val_loss,
                e.val_accuracy * 100.0,
            );
        }
        println!();
    }

    // Recommendations
    if !recommendations.is_empty() {
        println!("{}", "RECOMMENDATIONS:".cyan().bold());
        // Sort by priority
        let mut recs = recommendations;
        recs.sort_by(|a, b| a.priority.cmp(b.priority));
        for (i, r) in recs.iter().enumerate() {
            println!("  {}. [{}] {}", i + 1, r.priority.yellow(), r.action);
        }
    }
}

// ── Internal types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Debug)]
struct Finding {
    category: &'static str,
    severity: Severity,
    message: String,
}

struct Recommendation {
    priority: &'static str,
    action: String,
}

struct EpochInfo {
    epoch: usize,
    train_loss: f64,
    val_loss: f64,
    val_accuracy: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── Helper: create a Finding ────────────────────────────────────────────

    fn finding(category: &'static str, severity: Severity, msg: &str) -> Finding {
        Finding {
            category,
            severity,
            message: msg.to_string(),
        }
    }

    fn epoch(epoch: usize, train_loss: f64, val_loss: f64, val_accuracy: f64) -> EpochInfo {
        EpochInfo {
            epoch,
            train_loss,
            val_loss,
            val_accuracy,
        }
    }

    // ── Severity ordering ───────────────────────────────────────────────────

    #[test]
    fn severity_ordering_info_lt_warning_lt_error() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
        assert!(Severity::Info < Severity::Error);
    }

    #[test]
    fn severity_equality() {
        assert_eq!(Severity::Info, Severity::Info);
        assert_eq!(Severity::Warning, Severity::Warning);
        assert_eq!(Severity::Error, Severity::Error);
        assert_ne!(Severity::Info, Severity::Error);
    }

    #[test]
    fn severity_max_picks_highest() {
        let severities = [Severity::Info, Severity::Error, Severity::Warning];
        assert_eq!(severities.iter().copied().max(), Some(Severity::Error));
    }

    #[test]
    fn severity_max_single_info() {
        let severities = [Severity::Info];
        assert_eq!(severities.iter().copied().max(), Some(Severity::Info));
    }

    #[test]
    fn severity_debug_format() {
        assert_eq!(format!("{:?}", Severity::Info), "Info");
        assert_eq!(format!("{:?}", Severity::Warning), "Warning");
        assert_eq!(format!("{:?}", Severity::Error), "Error");
    }

    // ── generate_recommendations ────────────────────────────────────────────

    #[test]
    fn generate_recommendations_adds_retrain_on_prediction_collapse() {
        let findings = vec![finding(
            "Prediction Collapse",
            Severity::Error,
            "80% of predictions go to class 0",
        )];
        let mut recs = Vec::new();
        generate_recommendations(&findings, &mut recs);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].priority, "P0");
        assert!(recs[0].action.contains("Retrain"));
    }

    #[test]
    fn generate_recommendations_adds_lr_finder_on_loss_curve_error() {
        let findings = vec![finding(
            "Loss Curve",
            Severity::Error,
            "Loss DIVERGED",
        )];
        let mut recs = Vec::new();
        generate_recommendations(&findings, &mut recs);
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].priority, "P1");
        assert!(recs[0].action.contains("LR finder"));
    }

    #[test]
    fn generate_recommendations_no_action_on_loss_curve_warning() {
        let findings = vec![finding(
            "Loss Curve",
            Severity::Warning,
            "Initial loss high",
        )];
        let mut recs = Vec::new();
        generate_recommendations(&findings, &mut recs);
        assert!(recs.is_empty(), "Warning-level loss curve should not trigger LR finder rec");
    }

    #[test]
    fn generate_recommendations_both_collapse_and_loss() {
        let findings = vec![
            finding("Prediction Collapse", Severity::Error, "collapsed"),
            finding("Loss Curve", Severity::Error, "diverged"),
        ];
        let mut recs = Vec::new();
        generate_recommendations(&findings, &mut recs);
        assert_eq!(recs.len(), 2);
        let priorities: Vec<&str> = recs.iter().map(|r| r.priority).collect();
        assert!(priorities.contains(&"P0"));
        assert!(priorities.contains(&"P1"));
    }

    #[test]
    fn generate_recommendations_empty_findings() {
        let findings: Vec<Finding> = Vec::new();
        let mut recs = Vec::new();
        generate_recommendations(&findings, &mut recs);
        assert!(recs.is_empty());
    }

    #[test]
    fn generate_recommendations_irrelevant_categories_ignored() {
        let findings = vec![
            finding("Checkpoint Integrity", Severity::Error, "Missing files"),
            finding("Data Quality", Severity::Warning, "Imbalanced"),
        ];
        let mut recs = Vec::new();
        generate_recommendations(&findings, &mut recs);
        assert!(recs.is_empty());
    }

    // ── analyze_loss_curve ──────────────────────────────────────────────────

    #[test]
    fn analyze_loss_curve_detects_divergence() {
        let metrics = vec![
            epoch(0, 1.0, 1.1, 0.5),
            epoch(1, 2.0, 2.2, 0.4),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let divergence = findings
            .iter()
            .find(|f| f.message.contains("DIVERGED"));
        assert!(divergence.is_some(), "Should detect divergence when loss doubles");
        assert_eq!(divergence.expect("checked above").severity, Severity::Error);

        let rec = recs.iter().find(|r| r.action.contains("early stopping"));
        assert!(rec.is_some(), "Should recommend early stopping on divergence");
    }

    #[test]
    fn analyze_loss_curve_no_divergence_within_threshold() {
        // 1.4x increase — under the 1.5x threshold
        let metrics = vec![
            epoch(0, 1.0, 1.0, 0.6),
            epoch(1, 1.4, 1.3, 0.55),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let divergence = findings.iter().any(|f| f.message.contains("DIVERGED"));
        assert!(!divergence, "1.4x increase should not trigger divergence");
    }

    #[test]
    fn analyze_loss_curve_detects_high_initial_loss() {
        // For 5 classes, random baseline = ln(5) ~= 1.609
        // 5x random baseline = ~8.05, so initial loss of 10.0 should trigger
        let metrics = vec![
            epoch(0, 10.0, 10.0, 0.2),
            epoch(1, 9.0, 9.5, 0.22),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let high_loss = findings.iter().any(|f| f.message.contains("random baseline"));
        assert!(high_loss, "Initial loss 10.0 >> 5*ln(5) should be flagged");
    }

    #[test]
    fn analyze_loss_curve_normal_initial_loss_no_warning() {
        // For 5 classes, random baseline = ln(5) ~= 1.609
        // 5x = ~8.05. Initial loss of 2.0 is well below.
        let metrics = vec![
            epoch(0, 2.0, 2.1, 0.4),
            epoch(1, 1.5, 1.6, 0.55),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let high_loss = findings.iter().any(|f| f.message.contains("random baseline"));
        assert!(!high_loss, "Normal initial loss should not trigger warning");
    }

    #[test]
    fn analyze_loss_curve_identifies_best_epoch_when_not_last() {
        let metrics = vec![
            epoch(0, 1.0, 0.8, 0.7),
            epoch(1, 0.8, 0.6, 0.8),   // best val_loss
            epoch(2, 0.9, 0.9, 0.65),  // got worse
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let best_epoch = findings.iter().find(|f| f.message.contains("Best checkpoint"));
        assert!(best_epoch.is_some(), "Should identify best epoch when it's not the last");
        let msg = &best_epoch.expect("checked").message;
        assert!(msg.contains("epoch 2"), "Best epoch is epoch index 1 = display epoch 2");
        assert!(msg.contains("WORSE"), "Should note training made model worse after best");
    }

    #[test]
    fn analyze_loss_curve_no_best_epoch_message_when_last_is_best() {
        let metrics = vec![
            epoch(0, 1.0, 1.0, 0.5),
            epoch(1, 0.8, 0.8, 0.6),
            epoch(2, 0.6, 0.6, 0.7),  // last and best
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let best = findings.iter().any(|f| f.message.contains("Best checkpoint"));
        assert!(!best, "No best-epoch finding when last epoch is the best");
    }

    #[test]
    fn analyze_loss_curve_binary_classification_baseline() {
        // For 2 classes, random baseline = ln(2) ~= 0.693
        // 5x = ~3.465, so initial of 4.0 should trigger
        let metrics = vec![
            epoch(0, 4.0, 4.0, 0.5),
            epoch(1, 3.5, 3.5, 0.52),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 2, &mut findings, &mut recs);

        let high_loss = findings.iter().any(|f| f.message.contains("random baseline"));
        assert!(high_loss, "4.0 > 5 * ln(2) ~= 3.47 should flag initial loss");
    }

    // ── check_checkpoint_integrity ──────────────────────────────────────────

    #[test]
    fn check_checkpoint_integrity_all_files_present() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        // Create all expected files
        for name in &["metadata.json", "model.safetensors", "config.json", "adapter_config.json"] {
            std::fs::File::create(base.join(name)).expect("create file");
        }

        // Create valid metadata.json with class_weights
        let meta = serde_json::json!({
            "class_weights": [1.0, 2.0, 1.5]
        });
        std::fs::write(
            base.join("metadata.json"),
            serde_json::to_string(&meta).expect("serialize"),
        )
        .expect("write metadata");

        let mut findings = Vec::new();
        let mut recs = Vec::new();
        let result = check_checkpoint_integrity(base, &mut findings, &mut recs);
        assert!(result.is_ok());
        assert!(
            findings.is_empty(),
            "No findings expected when all files present with class_weights: got {findings:?}",
        );
    }

    #[test]
    fn check_checkpoint_integrity_missing_files_detected() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        // Only create metadata.json (missing the other 3)
        let meta = serde_json::json!({ "class_weights": [1.0] });
        std::fs::write(
            base.join("metadata.json"),
            serde_json::to_string(&meta).expect("serialize"),
        )
        .expect("write");

        let mut findings = Vec::new();
        let mut recs = Vec::new();
        let result = check_checkpoint_integrity(base, &mut findings, &mut recs);
        assert!(result.is_ok());

        let missing = findings.iter().find(|f| f.message.contains("Missing files"));
        assert!(missing.is_some(), "Should detect missing files");
        let msg = &missing.expect("checked").message;
        assert!(msg.contains("model.safetensors"));
        assert!(msg.contains("config.json"));
        assert!(msg.contains("adapter_config.json"));
    }

    #[test]
    fn check_checkpoint_integrity_no_metadata_json() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        let mut findings = Vec::new();
        let mut recs = Vec::new();
        let result = check_checkpoint_integrity(base, &mut findings, &mut recs);
        assert!(result.is_ok());

        let no_meta = findings
            .iter()
            .any(|f| f.message.contains("metadata.json not found"));
        assert!(no_meta, "Should report missing metadata.json");
    }

    #[test]
    fn check_checkpoint_integrity_no_class_weights_warns() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        // metadata.json exists but without class_weights
        let meta = serde_json::json!({ "train_loss": 0.5 });
        std::fs::write(
            base.join("metadata.json"),
            serde_json::to_string(&meta).expect("serialize"),
        )
        .expect("write");

        let mut findings = Vec::new();
        let mut recs = Vec::new();
        let result = check_checkpoint_integrity(base, &mut findings, &mut recs);
        assert!(result.is_ok());

        let no_cw = findings
            .iter()
            .any(|f| f.message.contains("class_weights NOT saved"));
        assert!(no_cw, "Should warn about missing class_weights");

        let p0_rec = recs.iter().any(|r| r.priority == "P0" && r.action.contains("class_weights"));
        assert!(p0_rec, "Should have P0 recommendation for class_weights");
    }

    #[test]
    fn check_checkpoint_integrity_null_class_weights_warns() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        let meta = serde_json::json!({ "class_weights": null });
        std::fs::write(
            base.join("metadata.json"),
            serde_json::to_string(&meta).expect("serialize"),
        )
        .expect("write");

        let mut findings = Vec::new();
        let mut recs = Vec::new();
        check_checkpoint_integrity(base, &mut findings, &mut recs).expect("ok");

        let no_cw = findings.iter().any(|f| f.message.contains("class_weights NOT saved"));
        assert!(no_cw, "Null class_weights should be treated as missing");
    }

    #[test]
    fn check_checkpoint_integrity_invalid_json_returns_error() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        std::fs::write(base.join("metadata.json"), "not valid json{{{").expect("write");

        let mut findings = Vec::new();
        let mut recs = Vec::new();
        let result = check_checkpoint_integrity(base, &mut findings, &mut recs);
        assert!(result.is_err(), "Invalid JSON should produce an error");
    }

    // ── collect_epoch_metrics ───────────────────────────────────────────────

    #[test]
    fn collect_epoch_metrics_reads_epoch_dirs() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        for i in 0..3 {
            let epoch_dir = base.join(format!("epoch_{i}"));
            std::fs::create_dir(&epoch_dir).expect("mkdir");
            let meta = serde_json::json!({
                "epoch": i,
                "train_loss": 1.0 - (i as f64 * 0.2),
                "val_loss": 1.1 - (i as f64 * 0.15),
                "val_accuracy": 0.5 + (i as f64 * 0.1),
            });
            std::fs::write(
                epoch_dir.join("metadata.json"),
                serde_json::to_string(&meta).expect("ser"),
            )
            .expect("write");
        }

        let mut metrics = Vec::new();
        collect_epoch_metrics(base, &mut metrics);
        assert_eq!(metrics.len(), 3);
    }

    #[test]
    fn collect_epoch_metrics_ignores_non_epoch_dirs() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        // Create non-epoch directories
        std::fs::create_dir(base.join("checkpoint_1")).expect("mkdir");
        std::fs::create_dir(base.join("best_model")).expect("mkdir");

        let mut metrics = Vec::new();
        collect_epoch_metrics(base, &mut metrics);
        assert!(metrics.is_empty());
    }

    #[test]
    fn collect_epoch_metrics_skips_epoch_without_metadata() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        // epoch_0 has metadata, epoch_1 does not
        let e0 = base.join("epoch_0");
        std::fs::create_dir(&e0).expect("mkdir");
        let meta = serde_json::json!({
            "epoch": 0, "train_loss": 1.0, "val_loss": 1.0, "val_accuracy": 0.5
        });
        std::fs::write(
            e0.join("metadata.json"),
            serde_json::to_string(&meta).expect("ser"),
        )
        .expect("write");

        std::fs::create_dir(base.join("epoch_1")).expect("mkdir");
        // no metadata.json in epoch_1

        let mut metrics = Vec::new();
        collect_epoch_metrics(base, &mut metrics);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].epoch, 0);
    }

    #[test]
    fn collect_epoch_metrics_handles_missing_fields_with_defaults() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        let e0 = base.join("epoch_0");
        std::fs::create_dir(&e0).expect("mkdir");
        // metadata.json with only partial fields
        let meta = serde_json::json!({ "epoch": 2 });
        std::fs::write(
            e0.join("metadata.json"),
            serde_json::to_string(&meta).expect("ser"),
        )
        .expect("write");

        let mut metrics = Vec::new();
        collect_epoch_metrics(base, &mut metrics);
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].epoch, 2);
        assert!((metrics[0].train_loss - 0.0).abs() < f64::EPSILON);
        assert!((metrics[0].val_loss - 0.0).abs() < f64::EPSILON);
        assert!((metrics[0].val_accuracy - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn collect_epoch_metrics_empty_dir() {
        let dir = TempDir::new().expect("tempdir");
        let mut metrics = Vec::new();
        collect_epoch_metrics(dir.path(), &mut metrics);
        assert!(metrics.is_empty());
    }

    // ── check_loss_curve (integration of collect + analyze) ─────────────────

    #[test]
    fn check_loss_curve_returns_sorted_metrics() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        // Create epoch dirs in reverse order
        for i in [2, 0, 1] {
            let epoch_dir = base.join(format!("epoch_{i}"));
            std::fs::create_dir(&epoch_dir).expect("mkdir");
            let meta = serde_json::json!({
                "epoch": i, "train_loss": 1.0, "val_loss": 1.0, "val_accuracy": 0.5
            });
            std::fs::write(
                epoch_dir.join("metadata.json"),
                serde_json::to_string(&meta).expect("ser"),
            )
            .expect("write");
        }

        // check_loss_curve expects checkpoint_dir to be a child of parent
        let checkpoint = base.join("epoch_2");
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        let metrics = check_loss_curve(&checkpoint, 5, &mut findings, &mut recs);

        assert_eq!(metrics.len(), 3);
        assert_eq!(metrics[0].epoch, 0);
        assert_eq!(metrics[1].epoch, 1);
        assert_eq!(metrics[2].epoch, 2);
    }

    #[test]
    fn check_loss_curve_no_analysis_with_single_epoch() {
        let dir = TempDir::new().expect("tempdir");
        let base = dir.path();

        let e0 = base.join("epoch_0");
        std::fs::create_dir(&e0).expect("mkdir");
        let meta = serde_json::json!({
            "epoch": 0, "train_loss": 1.0, "val_loss": 1.0, "val_accuracy": 0.5
        });
        std::fs::write(
            e0.join("metadata.json"),
            serde_json::to_string(&meta).expect("ser"),
        )
        .expect("write");

        let mut findings = Vec::new();
        let mut recs = Vec::new();
        let metrics = check_loss_curve(&e0, 5, &mut findings, &mut recs);
        assert_eq!(metrics.len(), 1);
        assert!(findings.is_empty(), "Single epoch should not trigger analysis");
    }

    // ── check_data_quality ──────────────────────────────────────────────────

    #[test]
    fn check_data_quality_none_path_returns_immediately() {
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        check_data_quality(None, &mut findings, &mut recs);
        assert!(findings.is_empty());
        assert!(recs.is_empty());
    }

    #[test]
    fn check_data_quality_nonexistent_path_returns_immediately() {
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        check_data_quality(
            Some(Path::new("/nonexistent/path/data.json")),
            &mut findings,
            &mut recs,
        );
        assert!(findings.is_empty());
        assert!(recs.is_empty());
    }

    // ── output_json ─────────────────────────────────────────────────────────

    #[test]
    fn output_json_produces_valid_json_structure() {
        // We can't easily capture println! output, but we can verify the
        // serde_json::json! construction doesn't panic with various inputs.
        let findings = vec![
            finding("Accuracy", Severity::Info, "90%"),
            finding("Loss Curve", Severity::Error, "Diverged"),
        ];
        let recs = vec![Recommendation {
            priority: "P0",
            action: "Retrain".to_string(),
        }];
        let metrics = vec![epoch(0, 1.0, 1.1, 0.5)];

        // Build the JSON the same way output_json does
        #[allow(clippy::disallowed_methods)]
        let report = serde_json::json!({
            "checkpoint": "/test/path",
            "findings": findings.iter().map(|f| serde_json::json!({
                "category": f.category,
                "severity": format!("{:?}", f.severity),
                "message": f.message,
            })).collect::<Vec<_>>(),
            "recommendations": recs.iter().map(|r| serde_json::json!({
                "priority": r.priority,
                "action": r.action,
            })).collect::<Vec<_>>(),
            "epoch_metrics": metrics.iter().map(|e| serde_json::json!({
                "epoch": e.epoch + 1,
                "train_loss": e.train_loss,
                "val_loss": e.val_loss,
                "val_accuracy": e.val_accuracy,
            })).collect::<Vec<_>>(),
            "eval_report": serde_json::Value::Null,
        });

        let serialized = serde_json::to_string_pretty(&report).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&serialized).expect("parse back");

        assert_eq!(parsed["findings"].as_array().expect("array").len(), 2);
        assert_eq!(parsed["recommendations"].as_array().expect("array").len(), 1);
        assert_eq!(parsed["epoch_metrics"].as_array().expect("array").len(), 1);
        assert_eq!(parsed["findings"][0]["severity"], "Info");
        assert_eq!(parsed["findings"][1]["severity"], "Error");
        assert_eq!(parsed["epoch_metrics"][0]["epoch"], 1); // epoch 0 + 1
    }

    #[test]
    fn output_json_empty_collections() {
        let findings: Vec<Finding> = Vec::new();
        let recs: Vec<Recommendation> = Vec::new();
        let metrics: Vec<EpochInfo> = Vec::new();

        #[allow(clippy::disallowed_methods)]
        let report = serde_json::json!({
            "checkpoint": "/empty",
            "findings": findings.iter().map(|f| serde_json::json!({
                "category": f.category,
                "severity": format!("{:?}", f.severity),
                "message": f.message,
            })).collect::<Vec<_>>(),
            "recommendations": recs.iter().map(|r| serde_json::json!({
                "priority": r.priority,
                "action": r.action,
            })).collect::<Vec<_>>(),
            "epoch_metrics": metrics.iter().map(|e| serde_json::json!({
                "epoch": e.epoch + 1,
                "train_loss": e.train_loss,
                "val_loss": e.val_loss,
                "val_accuracy": e.val_accuracy,
            })).collect::<Vec<_>>(),
            "eval_report": serde_json::Value::Null,
        });

        let serialized = serde_json::to_string_pretty(&report).expect("serialize");
        let parsed: serde_json::Value = serde_json::from_str(&serialized).expect("parse back");
        assert!(parsed["findings"].as_array().expect("array").is_empty());
        assert!(parsed["recommendations"].as_array().expect("array").is_empty());
        assert!(parsed["epoch_metrics"].as_array().expect("array").is_empty());
    }

    // ── run() error paths ───────────────────────────────────────────────────

    #[test]
    fn run_rejects_nonexistent_directory() {
        let result = run(
            Path::new("/nonexistent/checkpoint/dir"),
            None,
            None,
            5,
            false,
        );
        assert!(result.is_err());
        let err_msg = format!("{}", result.expect_err("should fail"));
        assert!(err_msg.contains("not found"), "Error should mention not found: {err_msg}");
    }

    #[test]
    fn run_rejects_file_as_checkpoint_dir() {
        let dir = TempDir::new().expect("tempdir");
        let file_path = dir.path().join("somefile.txt");
        std::fs::write(&file_path, "hello").expect("write");

        let result = run(&file_path, None, None, 5, false);
        assert!(result.is_err());
    }

    // ── analyze_loss_curve edge cases ───────────────────────────────────────

    #[test]
    fn analyze_loss_curve_exact_threshold_no_divergence() {
        // Exactly 1.5x should NOT trigger (> 1.5, not >=)
        let metrics = vec![
            epoch(0, 1.0, 1.0, 0.5),
            epoch(1, 1.5, 1.5, 0.4),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let divergence = findings.iter().any(|f| f.message.contains("DIVERGED"));
        assert!(!divergence, "Exactly 1.5x should not trigger divergence (> not >=)");
    }

    #[test]
    fn analyze_loss_curve_just_over_threshold() {
        let metrics = vec![
            epoch(0, 1.0, 1.0, 0.5),
            epoch(1, 1.51, 1.51, 0.4),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let divergence = findings.iter().any(|f| f.message.contains("DIVERGED"));
        assert!(divergence, "1.51x should trigger divergence");
    }

    #[test]
    fn analyze_loss_curve_single_class_baseline() {
        // ln(1) = 0, so any positive loss is > 0 * 5
        // But 0 * 5 = 0, and any loss > 0 should trigger
        let metrics = vec![
            epoch(0, 0.1, 0.1, 0.9),
            epoch(1, 0.05, 0.05, 0.95),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 1, &mut findings, &mut recs);

        // ln(1) = 0.0, 5*0 = 0. initial_loss 0.1 > 0. Should trigger.
        let high_loss = findings.iter().any(|f| f.message.contains("random baseline"));
        assert!(high_loss, "Any positive loss > 5 * ln(1) = 0 for single class");
    }

    #[test]
    fn analyze_loss_curve_multiple_findings_combined() {
        // Both divergence AND high initial loss
        let metrics = vec![
            epoch(0, 50.0, 50.0, 0.1),
            epoch(1, 100.0, 100.0, 0.05),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let has_divergence = findings.iter().any(|f| f.message.contains("DIVERGED"));
        let has_high_loss = findings.iter().any(|f| f.message.contains("random baseline"));
        assert!(has_divergence, "Should detect divergence");
        assert!(has_high_loss, "Should detect high initial loss");
    }

    #[test]
    fn analyze_loss_curve_decreasing_loss_no_warnings() {
        let metrics = vec![
            epoch(0, 1.5, 1.6, 0.4),
            epoch(1, 1.0, 1.1, 0.6),
            epoch(2, 0.5, 0.6, 0.8),
        ];
        let mut findings = Vec::new();
        let mut recs = Vec::new();
        analyze_loss_curve(&metrics, 5, &mut findings, &mut recs);

        let errors = findings.iter().any(|f| f.severity == Severity::Error);
        let warnings = findings.iter().any(|f| f.severity == Severity::Warning);
        assert!(!errors, "Healthy training should have no errors");
        assert!(!warnings, "Healthy training should have no warnings");
    }

    // ── Finding / Recommendation struct tests ───────────────────────────────

    #[test]
    fn finding_struct_fields_accessible() {
        let f = Finding {
            category: "Test",
            severity: Severity::Warning,
            message: "test message".to_string(),
        };
        assert_eq!(f.category, "Test");
        assert_eq!(f.severity, Severity::Warning);
        assert_eq!(f.message, "test message");
    }

    #[test]
    fn recommendation_struct_fields_accessible() {
        let r = Recommendation {
            priority: "P2",
            action: "Do something".to_string(),
        };
        assert_eq!(r.priority, "P2");
        assert_eq!(r.action, "Do something");
    }

    #[test]
    fn epoch_info_struct_fields() {
        let e = EpochInfo {
            epoch: 5,
            train_loss: 0.42,
            val_loss: 0.45,
            val_accuracy: 0.88,
        };
        assert_eq!(e.epoch, 5);
        assert!((e.train_loss - 0.42).abs() < f64::EPSILON);
        assert!((e.val_loss - 0.45).abs() < f64::EPSILON);
        assert!((e.val_accuracy - 0.88).abs() < f64::EPSILON);
    }

    // ── Severity clone/copy ─────────────────────────────────────────────────

    #[test]
    fn severity_is_copy() {
        let s = Severity::Error;
        let s2 = s; // Copy
        assert_eq!(s, s2);
    }

    #[test]
    fn severity_clone() {
        let s = Severity::Warning;
        let s2 = s.clone();
        assert_eq!(s, s2);
    }
}
