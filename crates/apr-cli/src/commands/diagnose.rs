//! `apr diagnose` — Automated Five Whys for training checkpoints.
//!
//! Reads checkpoint metadata, optionally runs evaluation, and performs
//! automated root cause analysis for training failures.

use std::path::Path;

use colored::Colorize;

use crate::{error::CliError, output};

type Result<T> = std::result::Result<T, CliError>;

/// Run automated diagnosis on a training checkpoint.
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
    let epoch_metrics = check_loss_curve(checkpoint_dir, num_classes, &mut findings, &mut recommendations);
    let eval_report = run_evaluation(
        checkpoint_dir,
        data_path,
        model_size,
        num_classes,
        &mut findings,
    )?;
    check_data_quality(data_path, &mut findings, &mut recommendations);
    generate_recommendations(&findings, &mut recommendations);

    if json_output {
        output_json(
            checkpoint_dir,
            &findings,
            &recommendations,
            &epoch_metrics,
            &eval_report,
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
        Some(serde_json::from_str(&meta_str).map_err(|e| {
            CliError::ValidationFailed(format!("Invalid metadata.json: {e}"))
        })?)
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
        .map_or(false, |v| !v.is_null());

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
        let imbalance = alimentar::imbalance::ImbalanceDetector::new("label")
            .analyze(&dataset);
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
    let has_collapse = findings
        .iter()
        .any(|f| f.category == "Prediction Collapse");
    if has_collapse {
        recommendations.push(Recommendation {
            priority: "P0",
            action: "Retrain with stratified split and verified class_weights".to_string(),
        });
    }

    if findings.iter().any(|f| {
        f.category == "Loss Curve"
            && f.severity == Severity::Error
    }) {
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
    eval_report: &Option<serde_json::Value>,
) {
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
    println!("{}", serde_json::to_string_pretty(&report).unwrap_or_default());
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

        println!(
            "{}  WHY {why_num}: {}",
            severity_icon,
            cat.white().bold()
        );
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
            let marker = if e.val_loss
                == epoch_metrics
                    .iter()
                    .map(|x| x.val_loss)
                    .fold(f64::MAX, f64::min)
            {
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
        recs.sort_by(|a, b| a.priority.cmp(&b.priority));
        for (i, r) in recs.iter().enumerate() {
            println!(
                "  {}. [{}] {}",
                i + 1,
                r.priority.yellow(),
                r.action
            );
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
