// ── WHY 2: Loss curve analysis ───────────────────────────────────────────────

fn check_loss_curve(
    checkpoint_dir: &Path,
    num_classes: usize,
    findings: &mut Vec<Finding>,
    recommendations: &mut Vec<Recommendation>,
) -> Vec<EpochInfo> {
    // Look for training_state.json or parse metadata from multiple epoch dirs
    let parent = checkpoint_dir.parent();
    let mut epoch_metrics: Vec<EpochInfo> = Vec::new();

    if let Some(parent_dir) = parent {
        collect_epoch_metrics(parent_dir, &mut epoch_metrics);
    }

    epoch_metrics.sort_by_key(|e| e.epoch);

    if epoch_metrics.len() >= 2 {
        analyze_loss_curve(&epoch_metrics, num_classes, findings, recommendations);
    }

    epoch_metrics
}

fn collect_epoch_metrics(parent_dir: &Path, epoch_metrics: &mut Vec<EpochInfo>) {
    if let Ok(entries) = std::fs::read_dir(parent_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("epoch_") {
                let epoch_meta = entry.path().join("metadata.json");
                if let Ok(content) = std::fs::read_to_string(&epoch_meta) {
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&content) {
                        epoch_metrics.push(EpochInfo {
                            epoch: val
                                .get("epoch")
                                .and_then(serde_json::Value::as_u64)
                                .unwrap_or(0) as usize,
                            train_loss: val
                                .get("train_loss")
                                .and_then(serde_json::Value::as_f64)
                                .unwrap_or(0.0),
                            val_loss: val
                                .get("val_loss")
                                .and_then(serde_json::Value::as_f64)
                                .unwrap_or(0.0),
                            val_accuracy: val
                                .get("val_accuracy")
                                .and_then(serde_json::Value::as_f64)
                                .unwrap_or(0.0),
                        });
                    }
                }
            }
        }
    }
}

fn analyze_loss_curve(
    epoch_metrics: &[EpochInfo],
    num_classes: usize,
    findings: &mut Vec<Finding>,
    recommendations: &mut Vec<Recommendation>,
) {
    // Check for divergence
    let first_loss = epoch_metrics[0].train_loss;
    let last_loss = epoch_metrics.last().map_or(0.0, |e| e.train_loss);
    if last_loss > first_loss * 1.5 {
        findings.push(Finding {
            category: "Loss Curve",
            severity: Severity::Error,
            message: format!(
                "Loss DIVERGED: epoch 1 = {first_loss:.2} -> final = {last_loss:.2} ({:.0}% increase)",
                (last_loss / first_loss - 1.0) * 100.0
            ),
        });
        recommendations.push(Recommendation {
            priority: "P1",
            action: "Reduce to 1 epoch or add early stopping".to_string(),
        });
    }

    // Check loss scale anomaly (random baseline for K classes is -ln(1/K))
    let random_baseline = (num_classes as f64).ln();
    if first_loss > random_baseline * 5.0 {
        findings.push(Finding {
            category: "Loss Curve",
            severity: Severity::Warning,
            message: format!(
                "Initial loss {first_loss:.2} is {:.1}x the random baseline ({random_baseline:.2} for {num_classes} classes) — possible loss accumulation bug",
                first_loss / random_baseline
            ),
        });
    }

    // Best epoch
    if let Some(best) = epoch_metrics.iter().min_by(|a, b| {
        a.val_loss.partial_cmp(&b.val_loss).unwrap_or(std::cmp::Ordering::Equal)
    }) {
        if best.epoch < epoch_metrics.len() - 1 {
            findings.push(Finding {
                category: "Loss Curve",
                severity: Severity::Info,
                message: format!(
                    "Best checkpoint: epoch {} (val_loss={:.4}, val_acc={:.1}%) — training made model WORSE after this",
                    best.epoch + 1,
                    best.val_loss,
                    best.val_accuracy * 100.0
                ),
            });
        }
    }
}

// ── WHY 3: Run evaluation if data provided ───────────────────────────────────

#[cfg(feature = "training")]
fn run_evaluation(
    checkpoint_dir: &Path,
    data_path: Option<&Path>,
    model_size: Option<&str>,
    num_classes: usize,
    findings: &mut Vec<Finding>,
) -> Result<Option<serde_json::Value>> {
    use entrenar::finetune::classify_pipeline::ClassifyConfig;
    use entrenar::finetune::{evaluate_checkpoint, SSC_LABELS};
    let mut eval_report: Option<serde_json::Value> = None;

    let Some(data) = data_path else {
        return Ok(eval_report);
    };

    if !data.exists() {
        findings.push(Finding {
            category: "Data",
            severity: Severity::Warning,
            message: format!("Test data not found: {}", data.display()),
        });
        return Ok(eval_report);
    }

    // GH-377: Resolve model config — error on unknown instead of silent tiny()
    let model_config = match super::model_config::resolve_transformer_config_by_size(model_size) {
        Ok(config) => config,
        Err(e) => {
            findings.push(Finding {
                category: "Model",
                severity: Severity::Error,
                message: format!("Cannot resolve model architecture: {e}"),
            });
            return Ok(eval_report);
        }
    };

    let classify_config = ClassifyConfig {
        num_classes,
        ..ClassifyConfig::default()
    };

    let label_names: Vec<String> = if num_classes == 5 {
        SSC_LABELS.iter().map(|s| (*s).to_string()).collect()
    } else {
        (0..num_classes).map(|i| format!("class_{i}")).collect()
    };

    eprintln!("{}", "Running evaluation on test set...".yellow());

    match evaluate_checkpoint(
        checkpoint_dir,
        data,
        &model_config,
        classify_config,
        &label_names,
    ) {
        Ok(report) => {
            analyze_eval_report(&report, num_classes, &label_names, findings);
            // Use report's built-in JSON serialization
            eval_report = serde_json::from_str(&report.to_json()).ok();
        }
        Err(e) => {
            findings.push(Finding {
                category: "Evaluation",
                severity: Severity::Error,
                message: format!("Evaluation failed: {e}"),
            });
        }
    }

    Ok(eval_report)
}

#[cfg(feature = "training")]
fn analyze_eval_report(
    report: &entrenar::finetune::ClassifyEvalReport,
    num_classes: usize,
    label_names: &[String],
    findings: &mut Vec<Finding>,
) {
    let accuracy = report.accuracy;
    let total_samples = report.total_samples;

    // Prediction collapse detection from confusion matrix
    let mut class_predictions = vec![0usize; num_classes];
    for row in &report.confusion_matrix {
        for (pred_class, &count) in row.iter().enumerate() {
            if pred_class < num_classes {
                class_predictions[pred_class] += count;
            }
        }
    }
    let max_pred_class = class_predictions
        .iter()
        .enumerate()
        .max_by_key(|(_, c)| **c)
        .map_or(0, |(i, _)| i);
    let max_pred_pct = if total_samples > 0 {
        class_predictions[max_pred_class] as f64 / total_samples as f64
    } else {
        0.0
    };

    if max_pred_pct > 0.5 {
        findings.push(Finding {
            category: "Prediction Collapse",
            severity: Severity::Error,
            message: format!(
                "{:.1}% of predictions go to class {} ({}) — model collapsed",
                max_pred_pct * 100.0,
                max_pred_class,
                label_names
                    .get(max_pred_class)
                    .map_or("?", String::as_str)
            ),
        });
    }

    // Accuracy vs majority baseline
    let majority_baseline = report.baseline_majority;

    if accuracy < majority_baseline {
        findings.push(Finding {
            category: "Accuracy",
            severity: Severity::Error,
            message: format!(
                "Accuracy {:.1}% is BELOW majority baseline ({:.1}%) — model is worse than always predicting majority class",
                accuracy * 100.0,
                majority_baseline * 100.0
            ),
        });
    } else {
        findings.push(Finding {
            category: "Accuracy",
            severity: Severity::Info,
            message: format!(
                "Accuracy {:.1}% (majority baseline: {:.1}%)",
                accuracy * 100.0,
                majority_baseline * 100.0
            ),
        });
    }

    // Calibration (ECE)
    if report.ece > 0.15 {
        findings.push(Finding {
            category: "Calibration",
            severity: Severity::Warning,
            message: format!(
                "ECE = {:.3} — model is poorly calibrated (>0.15)",
                report.ece
            ),
        });
    }
}
